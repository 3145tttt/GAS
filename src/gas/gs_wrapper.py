import lpips

import torch
from torch import nn
from typing import Tuple, Optional, Any, List
from torch.nn.functional import interpolate
from torch_ema import ExponentialMovingAverage
from ml_collections import ConfigDict

from src.gas.base_model import BaseModel
from src.gas.generalized_solver import GeneralizedSolver
from src.gas.adversarial_module.dist_adv_loss import DistAdversarialTraining
from src.gas.synt_data import SyntDataType

class GSWrapper(nn.Module):
    """Generalised Solver wrapper. 
    
    This class integrates all the logic needed to train or evaluate 
    the given generative model using Generalised Solver. 
    
    Method `student_sampler_fn` is used to call the sampler.
    Method `forward` is called in the training loop to calculate the losses.
        The model can be trained in both default or adversarial modes.
    
    Attributes:
        model (BaseModel): Underlying model instance wrapped in BaseModel interface.
        solver_config (ConfigDict): Solver configuration dictionary.
        solver (GeneralizedSolver): Generalised Solver instance that is trained/evaluated.
        
        loss_fn_vgg (nn.Module): VGG model instance to calculate LPIPS loss.
        adv_loss (DistAdversarialTraining): Adversarial training class instance. 
    """
    
    def __init__(self, model: BaseModel, solver_config: ConfigDict):
        """Initialize the Generalised Solver wrapper.
    
        Args:
            model (BaseModel): Instance of a BaseModel class. 
                Its `decode`, `set_condition` methods and 
                `model_fn`, `ns` and `t_eps` attributes are used.
            solver_config (ConfigDict): Solver configuration dictionary.
                Must include steps, order, loss_config, 
                t_parametrization and use_theory_coef.
        """
        super().__init__()
        self.model = model
        self.solver_config = solver_config
        self.t_eps = self.model.t_eps
        
        # create lpips
        self.loss_fn_vgg = lpips.LPIPS(net='vgg').requires_grad_(False)
        self.loss_fn_vgg.eval()

        # construct loss
        self.loss_config = self.solver_config.loss_config
        assert self.loss_config.loss_type in ["GS", "GAS"]
        if self.loss_config.loss_type == "GAS":
            self.adv_loss = DistAdversarialTraining(self.loss_config)

        # setup solver
        solver = self.get_base_solver()
        self.steps = self.solver_config.steps
        self.order = self.solver_config.order

        # init t steps
        assert self.solver_config.t_parametrization == "mu_logit"
        self.eps_mu_offset = 1e-5
        self.mu_logit = nn.Parameter(torch.ones(self.steps - 1), requires_grad=True)
        t_unif = torch.linspace(1., self.t_eps, self.steps + 1).flip(0)
        self.mu_logit.data = self.get_inv_t_steps(t_unif)

        solver.get_time_steps = lambda **kwargs: self.get_t_steps(**kwargs)

        # init t_couple
        self.t_couple = nn.Parameter(torch.zeros(self.steps), requires_grad=True)
        solver.t_couple = self.t_couple

        # init coef
        for i in range(1, self.order + 1):
            cname, aname = f'c{i}_diff', f'a{i}_diff'

            self.register_parameter(
                param=nn.Parameter(torch.zeros(self.steps), requires_grad=True),
                name=cname
            )
            self.register_parameter(
                param=nn.Parameter(torch.zeros(self.steps), requires_grad=True),
                name=aname
            )

            solver.__setattr__(cname, self.__getattr__(cname))
            solver.__setattr__(aname, self.__getattr__(aname))

        # theory coef
        solver.use_theory_coef = self.solver_config.use_theory_coef
        if not solver.use_theory_coef:
            solver.init_coefs(
                steps=self.steps,
                order=self.order,
                timesteps=self.get_t_steps()
            )
        # end init solver
        self.solver = solver

    # timesteps logic
    def get_t_steps(self, **kwargs) -> torch.Tensor:
        """Get generation timesteps."""
        logits = self.mu_logit
        t = self.get_mu_t_steps(logits)
        
        return t.flip(0)
    
    def get_mu_t_steps(self, mu_logit: torch.Tensor) -> torch.Tensor:
        """Use stick-breaking transform for getting timesteps from logits.
        Timesteps are calculated following Eq. 14 from the GAS paper.
        """
        t_offset = self.t_eps

        mu = mu_logit.sigmoid()
        mu = mu * (1 - 2 * self.eps_mu_offset) + self.eps_mu_offset

        t_steps = 1 - torch.cumprod(mu, 0)
        t_steps = t_steps * (1 - t_offset) + t_offset
        t_steps = torch.cat(
            [
                torch.zeros_like(t_steps[:1]) + t_offset,
                t_steps,
                torch.ones_like(t_steps[:1])
            ]
        )
        return t_steps
    
    def get_inv_t_steps(self, t_steps) -> torch.Tensor:
        """Function to inverse initialized timesteps."""
        t_steps = t_steps[1:-1]
        t_steps = 1 - (t_steps - self.t_eps) / (1 - self.t_eps)
        t_steps = t_steps / torch.concat([torch.ones_like(t_steps[:1]), t_steps[:-1]])
        t_steps = (t_steps - self.eps_mu_offset) / (1 - 2 * self.eps_mu_offset)

        return t_steps.logit()
    
    # utilities
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Loads EMA parameters checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        ema = ExponentialMovingAverage(self.parameters(), 0.1)
        ema.load_state_dict(checkpoint['ema'])
        ema.copy_to(self.parameters())

    def parameters(self) -> List[nn.parameter.Parameter]:
        """Returns list of specified solver and wrapper parameters."""
        return list(p for p in super().parameters() if p.requires_grad)

    def interpolate_lpips(self, x: torch.Tensor) -> torch.Tensor:
        """Utility function to resize images for LPIPS calculation."""
        return interpolate(x, size=224, mode='bilinear').clip(-1., 1.)

    # solvers
    def get_base_solver(self) -> GeneralizedSolver:
        """Initialises Generalized Solver from model_fn 
        and noise scheduler of the BaseModel instance.
        """
        solver = GeneralizedSolver(
            model_fn=self.model.model_fn,
            noise_schedule=self.model.ns,
        )
        return solver

    def student_sampler_fn(self, noise: torch.Tensor, **kwargs) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """Calls `sample` method of the Generalised Solver. 
        
        Args:
            noise (torch.Tensor): An initial noise tensor to start sampling process from.
        
        Returns:
            None: A placeholder for consistency with latent models.
            torch.tensor: Sampled images
        """
        images = self.solver.sample(
            x=noise,
            steps=self.steps,
            order=self.order,
        )
        return None, images
    
    # training function
    def forward(self, batch: SyntDataType, return_timesteps: bool = False, is_train: bool = True) -> dict:
        """Forward function used in training loop. Evaluates solver and calculates losses.
        
        Args:
            batch (SyntDataType): Dataset tuple of size 4. 
                First two arguments are treated like torch.Tensor noise and images samples.
                Second two arguments are optional and can be used in GSWrapperLatent for latent diffusion models.
                They are treated as latents tensors and conditions.
            return_timesteps (bool): Flag whether to return timestep of the current step.
            is_train (bool): Flag whether forward is called in the train loop. 
                Used in `discriminator_step` method of an DistAdversarialTraining instance.
            
        Returns:
            dict: Dictionary of all losses and model outputs. 
                Has `loss_total` key as a weighted sum of adversarial and distillation losses.
        """
        assert len(batch) == 4, f"len(batch) is expected to be 4, yours is {len(batch)}"
        noise, images, _, _ = batch

        d = {}
        if return_timesteps:
            d['timesteps'] = self.solver.get_time_steps()
        _, student_images = self.student_sampler_fn(noise)

        d['loss_l1'] = torch.abs(student_images - images).mean((1, 2, 3))
        d['loss_l2'] = torch.square(student_images - images).mean((1, 2, 3))

        d['x0_s'] = self.interpolate_lpips(student_images)
        d['x0_t'] = self.interpolate_lpips(images)

        d['loss_lpips'] = self.loss_fn_vgg(d['x0_s'], d['x0_t']).flatten(0)

        if self.loss_config.loss_type == 'GAS':
            # disctiminator step optim
            with torch.no_grad():
                _, student_images_disc = self.student_sampler_fn(
                    torch.randn_like(noise)
                )
            res = self.adv_loss.discriminator_step(
                FakeSamples=student_images_disc,
                RealSamples=images,
                is_train=is_train
            )
            d['dis_loss_adv'] = res[0]
            d['dis_scores_fake'] = res[1]
            d['dis_signs_fake'] = res[1].sign()
            d['dis_r1'] = res[2]
            d['dis_r2'] = res[3]

            # generator step optim
            loss_adv, res = self.adv_loss.AccumulateGeneratorGradients(
                FakeSamples=student_images,
                RealSamples=images
            )
            d['gen_loss_adv'] = loss_adv
            d['gen_fake_gen'] = res[1]
            d['gen_signs_fake'] = res[1].sign()

            assert d['gen_loss_adv'].shape == d[self.loss_config.loss_key].shape, f"""
                Shape of generator loss is not equal to distillation loss shape. 
                ({d['gen_loss_adv'].shape} vs {d[self.loss_config.loss_key].shape}).
            """

        d['loss_total'] = self.loss_config.disc_weight * d.get('gen_loss_adv', 0.) + d[self.loss_config.loss_key]

        return d
    
    
class GSWrapperLatent(GSWrapper):
    """Generalised Solver wrapper adapted for latent models."""
    def __init__(self, model: nn.Module, solver_config: ConfigDict):
        super().__init__(model=model, solver_config=solver_config)

    def student_sampler_fn(
        self,
        noise: torch.Tensor,
        decode: bool = False,
        condition: Any = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Calls `sample` method of the Generalised Solver. 
        
        Args:
            noise (torch.Tensor): An initial noise tensor to start sampling process from.
        
        Returns:
            torch.Tensor: Predicted latents that are the direct output of the model.
            Optional[torch.Tensor]: Predicted images (decoded latents).
                Not None if decode flag is set True.
        """
        images = None
        if condition is not None:
            self.model.set_condition(condition)

        latents = self.solver.sample(
            x=noise,
            steps=self.steps,
            order=self.order,
        )

        if decode:
            images = self.model.decode(latents)

        return latents, images
    
    def forward(self, batch: SyntDataType, return_timesteps: bool = False, is_train: bool = True) -> dict:
        assert len(batch) == 4, f"len(batch) is expected to be 4, yours is {len(batch)}"
        noise, images, latents, condition = batch

        d = {}
        if return_timesteps:
            d['timesteps'] = self.solver.get_time_steps()
        student_latents, _ = self.student_sampler_fn(
            noise,
            condition=condition
        )

        d['loss_l1_latents'] = torch.abs(latents - student_latents).mean((1, 2, 3))
        d['loss_l2_latents'] = torch.square(latents - student_latents).mean((1, 2, 3))
        d['x0_t'] = self.interpolate_lpips(images)
        d['latents_s'] = student_latents

        if self.loss_config.loss_type == "GAS":
            with torch.no_grad():
                student_latents_disc, _ = self.student_sampler_fn(
                    torch.randn_like(noise)
                )
            res = self.adv_loss.discriminator_step(
                FakeSamples=student_latents_disc,
                RealSamples=latents,
                is_train=is_train
            )

            d['dis_loss_adv'] = res[0]
            d['dis_scores_fake'] = res[1]
            d['dis_signs_fake'] = res[1].sign()
            d['dis_r1'] = res[2]
            d['dis_r2'] = res[3]

            # generator step
            loss_adv, res = self.adv_loss.AccumulateGeneratorGradients(
                FakeSamples=student_latents,
                RealSamples=latents
            )
            d['gen_loss_adv'] = loss_adv
            d['gen_fake_gen'] = res[1]
            d['gen_signs_fake'] = res[1].sign()

            assert d['gen_loss_adv'].shape == d[self.loss_config.loss_key].shape, f"SHAPE = {d['gen_loss_adv'].shape}, {d[self.loss_config.loss_key].shape}"

        d['loss_total'] = self.loss_config.disc_weight * d.get('gen_loss_adv', 0.) + d[self.loss_config.loss_key]

        return d