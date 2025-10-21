import torch
from ml_collections import ConfigDict
from typing import Optional, Tuple, List

from src.gas.base_model import BaseModel
from src.gas.uni_pc import UniPC
from src.ld3.ipndm import iPNDM


def uni_pc_sampler(
    model: BaseModel,
    noise: torch.Tensor,
    solver_config: ConfigDict,
    condition: Optional[torch.Tensor] = None
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """Samples images using UniPC solver [Zhao et al., 2023]
  
    Args:
        model (BaseModel): Underlying instance with `model_fn` and `ns` (noise scheduler) attributes.
        noise (torch.Tensor): Initial noise sample from which the
            denoising process begins.
        solver_config (ConfigDict): Solver config dictionary with 
            `steps`, `order`, `t_eps`, `time_type` and `variant` fields.
        condition (Optional[torch.Tensor]): Class labels or prompts to condition the generation on. 
            Defaults to None.

    Returns:
        Tuple[Optional[torch.Tensor], torch.Tensor]: Sampled latents and decoded images. 
            If generative model is EDM (pixel-space), first return argument is set to None.
    """

    if condition is not None:
        model.set_condition(condition)

    solver = UniPC(
        model.model_fn,
        model.ns,
        variant=solver_config.variant
    )

    # returns images for EDM and latents for LDM
    samples = solver.sample(
        noise,
        steps=solver_config.steps,
        t_start=1.,
        t_end=solver_config.t_eps,
        order=solver_config.order,
        skip_type=solver_config.time_type,
        method='multistep',
        denoise_to_zero=False,
        lower_order_final=True
    )

    if model.config.type != 'EDM':
        return samples, model.decode(samples)

    return None, samples


# Timesteps from LD3 https://github.com/vinhsuhi/LD3/blob/ec1bf603fb19696966ca30198ed209ae6488a3e5/utils.py#L57
gits_prior_timesteps = {
    5: [14.6146, 4.39, 1.5286, 0.6526, 0.2667, 0.0292],
    6: [14.6146, 4.7242, 1.9132, 0.9324, 0.4557, 0.1801, 0.0292],
    7: [14.6146, 6.4477, 2.2797, 1.1629, 0.6114, 0.3058, 0.1258, 0.0292],
    8: [14.6146, 6.4477, 2.7391, 1.4467, 0.8319, 0.4936, 0.2667, 0.1258, 0.0292]
}


def ipndm_sampler(
    model: BaseModel,
    noise: torch.Tensor,
    solver_config: ConfigDict,
    condition: List[str]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Samples images using iPNDM (Improved Pseudo Linear Multistep) solver.
    Uses GITS timesteps precalculated in LD3 codebase.
  
    Args:
        model (BaseModel): Underlying instance with 
            `model_fn` and `ns` (noise scheduler) attributes.
        noise (torch.Tensor): Initial noise sample from which the denoising process begins.
        solver_config (ConfigDict): Solver config dictionary with `steps`, `order` fields.
        condition (Optional[torch.Tensor]): Prompts to condition the generation on.

    Returns:
        Tuple[Optional[torch.Tensor], torch.Tensor]: Sampled latents and decoded images. 
            If generative model is EDM (pixel-space), first return argument is set to None.
    """

    model.set_condition(condition)
    timesteps = model.ns.inverse_lambda(
        -torch.log(torch.tensor(gits_prior_timesteps[solver_config.steps]))
    ).to(model.device).float()

    solver = iPNDM(model.ns)
    latents = solver.sample_simple(
        model_fn=model.model_fn,
        x=noise,
        timesteps=timesteps,
        timesteps2=timesteps,
        order=solver_config.order
    )
    images = model.decode(latents)

    return latents, images


SAMPLING_ALGS = {
    'EDM': uni_pc_sampler,
    'LDM': uni_pc_sampler,
    'SD': ipndm_sampler,
}
