import dnnlib
import pickle
import torch
import numpy as np
import torch.utils.checkpoint as cp
from torch import nn

from omegaconf import OmegaConf
from ml_collections import ConfigDict

from typing import List, Tuple, Any, Optional
from src.ldm.util import instantiate_from_config
from src.gas.solver_utils import NoiseScheduleVP, model_wrapper

class BaseModel:
    """Abstract base class for models with a unified interface.

    This class defines the standard interface expected from all models
    in the framework. Subclasses should implement the abstract methods
    to provide model-specific functionality.

    Attributes:
        model (torch.nn.Module): Underlying model instance.
        t_eps (float): Last timestep in the model inference.
        ns (NoiseScheduleVP): VP-SDE noise schedule (set in `setup_ns_model_fn`).
        model_fn (model_wrapper.model_fn): Model function wrapper 
            accepting continous time as input (set in `setup_ns_model_fn`).
        image_channels (int): Number of input channels, defaults to 3.
    """
        
    def __init__(self, config: ConfigDict, device=torch.device('cuda')):
        """Initialize the base model.

        Args:
            config (ConfigDict): Configuration dictionary containing model parameters.
            device (torch.device, optional): Device to run the model on.
        """
        self.device = device
        
        self.config = config
        self.model = self.load_model(self.config.path)
        
        assert self.config.t_eps is not None, "t epsilon is None"
        self.t_eps = self.config.t_eps
        self.setup_net_params()
        
        self.ns, self.model_fn = self.setup_ns_model_fn()
        self.image_channels = 3
    
    def load_model(self, path: str) -> nn.Module:
        """Load the model from the given checkpoint or config path.

        Args:
            path (str): Path to the model checkpoint or config.

        Returns:
            torch.nn.Module: Loaded model instance.

        Raises:
            NotImplementedError: Must be implemented in subclass.
        """
        raise NotImplementedError
    
    def setup_net_params(self) -> None:
        """Set up network parameters.
        Helper function for noise level schedules and other model-specific hyperparameters.

        Raises:
            NotImplementedError: Must be implemented in subclass.
        """
        raise NotImplementedError
    
    def setup_ns_model_fn(self) -> Tuple[NoiseScheduleVP, Any]:
        """Set up VP-SDE noise scheduler and model function
        class attributes to be used in solver evaluation.

        Returns:
            tuple:
                NoiseScheduleVP: Noise scheduler.
                model_wrapper.model_fn: A noise prediction model function 
                    which accepts the continuous-time input.

        Raises:
            NotImplementedError: Must be implemented in subclass.
        """
        raise NotImplementedError
    
    def setup_unconditional_conditioning(self) -> None:
        """Set up an unconditional conditioning class attribute for conditional models.

        Raises:
            NotImplementedError: Must be implemented in subclass.
        """
        raise NotImplementedError
    
    def set_condition(self) -> None:
        """Evaluates conditioning for given class label or generation prompt in conditional models.
        Sets corresponding `condition` and `unconditional_condition` attributes in model_fn.
        
        Args:
            condition (Any): Evaluation condition. 

        Raises:
            AssertionError: If `condition` is None.
            NotImplementedError: Must be implemented in subclass.
        """
        raise NotImplementedError
    
    def iterate_condition(self, idxs: List[int]) -> Any:
        """Iterates through `condition_loader` attribute.
        Initializes it if was not set before.
        
        Args:
            idxs (List[int]): Indexes to be used in iterating specified conditions.
            
        Returns:
            Any: Indexed class labels or strings of prompts.
        
        Raises:
            NotImplementedError: Must be implemented in subclass.
        """
        raise NotImplementedError
    
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decodes latent representations into images. 
        Used only in latent models.
        
        Args:
            latents (torch.Tensor): Latents to be decoded into images 
                using decode method of underlying model.
            
        Returns:
            torch.Tensor: Output of the decode method.

        Raises:
            NotImplementedError: Must be implemented in subclass.
        """
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: Optional[torch.Tensor]) -> torch.Tensor:
        """Evaluates the underlying generative model. 
        
        Args:
            x (torch.Tensor): Model input.
            t (torch.Tensor): Timestep the model is evaluated at.
            cond (Optional[torch.Tensor]): Encoded conditioning for conditional models.
            
        Returns:
            torch.Tensor: Output of the model evaluation.

        Raises:
            NotImplementedError: Must be implemented in subclass.
        """
        raise NotImplementedError
    

class EDMModel(BaseModel):
    def __init__(self, config: ConfigDict, device=torch.device('cuda')):
        super().__init__(config, device)
        self.image_size = self.model.img_resolution
        
    def load_model(self, path: str) -> nn.Module:
        with dnnlib.util.open_url(path, verbose=1) as f:
            net = pickle.load(f)['ema'].to(torch.float32)

        for param in net.parameters():
            param.requires_grad = False

        net.eval()
        net.to(self.device)
        
        return net
    
    def setup_net_params(self) -> None:
        # Helper functions for VP & VE noise level schedules.
        vp_sigma = lambda beta_d, beta_min: lambda t: (np.e ** (0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5

        # Select default noise level range based on the specified time step discretization.
        sigma_min = vp_sigma(beta_d=19.9, beta_min=0.1)(t=self.t_eps)
        sigma_max = vp_sigma(beta_d=19.9, beta_min=0.1)(t=1.)

        # Adjust noise levels based on what's supported by the network.
        sigma_min = max(sigma_min, self.model.sigma_min)
        sigma_max = min(sigma_max, self.model.sigma_max)

        # Compute corresponding betas for VP.
        vp_beta_d = 2 * (np.log(sigma_min ** 2 + 1) / self.t_eps - np.log(sigma_max ** 2 + 1)) / (self.t_eps - 1)
        vp_beta_min = np.log(sigma_max ** 2 + 1) - 0.5 * vp_beta_d

        # Define noise level schedule.
        sigma = vp_sigma(vp_beta_d, vp_beta_min)
        # Define scaling schedule.
        s = lambda t: 1 / (1 + sigma(t) ** 2).sqrt()

        self.s = s
        self.sigma = sigma
        self.vp_beta_d = vp_beta_d
        self.vp_beta_min = vp_beta_min
        
    def setup_ns_model_fn(self) -> Tuple[NoiseScheduleVP, Any]:
        ns = NoiseScheduleVP(
            'linear',
            continuous_beta_0=self.vp_beta_min,
            continuous_beta_1=self.vp_beta_min + self.vp_beta_d
        )
 
        model_fn = model_wrapper(
            model=self.forward,
            noise_schedule=ns,
            model_type="x_start"
        )
        
        return ns, model_fn

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.model(x / self.s(t), self.sigma(t))
    
    
class LDMModel(BaseModel):
    def __init__(self, config: ConfigDict, device=torch.device('cuda')):
        super().__init__(config, device)
        self.condition_loader = None
        if self.config.conditional:
            self.setup_unconditional_conditioning()
        
        self.image_size = self.model.model.diffusion_model.image_size
    
    def load_model(self, path: str) -> nn.Module:
        config = ConfigDict(OmegaConf.load(path))
        state_dict = torch.load(config.ckpt_path)["state_dict"]

        net = instantiate_from_config(config.model)
        net.load_state_dict(state_dict, strict=False)

        if 'use_ema' in config.model.params:
            if config.model.params.use_ema:
                net.model_ema.copy_to(net.model)

        for param in net.parameters():
            param.requires_grad = False

        net.eval()
        net.to(self.device)
        
        return net
    
    def setup_net_params(self) -> None:
        self.model.alphas_cumprod = self.model.alphas_cumprod.float()
     
    def setup_ns_model_fn(self) -> Tuple[NoiseScheduleVP, Any]:   
        ns = NoiseScheduleVP(
            'discrete',
            alphas_cumprod=self.model.alphas_cumprod
        )

        model_fn = model_wrapper(
            model=self.forward,
            noise_schedule=ns,
            model_type="noise",
            guidance_type="classifier-free",
            guidance_scale=self.config.guidance_scale
        )
        
        return ns, model_fn
    
    def setup_unconditional_conditioning(self) -> None:
        self.unconditional_condition = self.model.get_learned_conditioning(
            {self.model.cond_stage_key: torch.tensor([1000]).to(self.model.device)}
        )
            
    def set_condition(self, condition: torch.Tensor) -> None:
        assert condition is not None, 'The conditioned passed is None'
        
        bs = len(condition)
        condition = self.model.get_learned_conditioning(
            {self.model.cond_stage_key: condition}
        )
        self.model_fn.condition = condition
        self.model_fn.unconditional_condition = self.unconditional_condition.repeat(bs, 1, 1)
    
    def iterate_condition(self, idxs: List[int]) -> torch.Tensor:
        if self.condition_loader is None:
            self.condition_loader = torch.tensor([*range(1000)] * 50, device=self.device)
        idxs = np.remainder(idxs, self.condition_loader.shape[0])
        return self.condition_loader[idxs]
    
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.model.differentiable_decode_first_stage(latents)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.model.apply_model(x, t.expand((x.shape[0])), cond)
    
    
class SDModel(LDMModel):
    def __init__(self, config: ConfigDict, device=torch.device('cuda')):
        super().__init__(config, device)
        self.image_channels = 4
        
    def setup_unconditional_conditioning(self) -> None:
        self.unconditional_condition = self.model.get_learned_conditioning([""])

    def set_condition(self, condition: List[str])  -> None:
        assert condition is not None, 'The conditioned passed is None'
        
        bs = len(condition)
        condition = self.model.get_learned_conditioning(condition)
        self.model_fn.condition = condition
        self.model_fn.unconditional_condition = self.unconditional_condition.repeat(bs, 1, 1)
        
    def iterate_condition(self, idxs: List[int]) -> List[str]:
        if self.condition_loader is None:
            self.condition_loader = np.array([prompt.strip() for prompt in open(self.config.prompts_path)])
        
        idxs = np.remainder(idxs, len(self.condition_loader))
        return list(self.condition_loader[idxs])
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        output = cp.checkpoint(
            self.model.apply_model, 
            x, t.expand((x.shape[0])), cond, 
            use_reentrant=False
        )
        return output