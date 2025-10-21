from collections import defaultdict

import torch

import wandb
from src.gas.gs_wrapper import GSWrapper
from src.gas.synt_data import SyntDataLoaders
from src.gas.utils.loggers import log_end_img, log_t_steps_plot

NOT_LOG_KEYS = ["timesteps", "x0_s", "x0_t", "latents_s"]


@torch.no_grad()
def evaluate_wrapper(
    gs_wrapper: GSWrapper, 
    data: SyntDataLoaders, 
    device: torch.device, 
    suff: str, 
    global_step: int
) -> None:
    """Evaluating GS on test dataset and visualization batch for logging."""
    batch = [v.to(device) if isinstance(v, torch.Tensor) else v for v in data.vis_batch]

    d_res = {}

    out_d = gs_wrapper.forward(batch=batch, return_timesteps=True, is_train=False)
    log_t_steps_plot(
        t_steps=out_d["timesteps"],
        global_step=global_step,
        key=f"eval_image{suff}/t_steps",
    )
    for k, v in out_d.items():
        if k not in NOT_LOG_KEYS:
            d_res[f"vis_stat/{k}{suff}"] = v.mean().item()

    if "x0_s" not in out_d:
        out_d["x0_s"] = gs_wrapper.model.decode(out_d["latents_s"])
    log_end_img(
        out_d["x0_s"],
        out_d["x0_t"],
        global_step=global_step,
        key=f"vis_stat{suff}/backward_end_inter",
    )

    log_d = defaultdict(float)
    num_elements = 0
    for batch in data.test_loader:
        batch = [v.to(device) if isinstance(v, torch.Tensor) else v for v in batch]
        out_d = gs_wrapper.forward(batch=batch, return_timesteps=False, is_train=False)
        bs = batch[0].shape[0]
        num_elements += bs
        for k, v in out_d.items():
            if k not in NOT_LOG_KEYS:
                log_d[k] += v.mean().item() * bs

    for k, v in log_d.items():
        if k not in NOT_LOG_KEYS:
            d_res[f"val_stat/{k}{suff}"] = v / num_elements

    wandb.log(d_res, step=global_step)
    if "x0_s" not in out_d:
        out_d["x0_s"] = gs_wrapper.model.decode(out_d["latents_s"])
    log_end_img(
        out_d["x0_s"],
        out_d["x0_t"],
        global_step=global_step,
        key=f"val_stat{suff}/backward_end_inter",
    )
