import comet_ml
import wandb
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid
import os

from src.gas.gs_wrapper import GSWrapper


EXP = None
LOGGER = "None"

def setup_comet(config, date_str):
    global EXP
    EXP = comet_ml.start(
        project_name=config.logging.project_name
    )
    EXP.log_parameters(config.to_dict())
    EXP.set_name(f"{config.logging.run_name}_{date_str}")
    # code
    for path in os.listdir("./"):
        if path.endswith(".py"):
            EXP.log_code(file_name=path)
    EXP.log_code(folder="./src/gas")

def setup_wandb(config, date_str):
    wandb.login(force=True)
    wandb.init(
        project=config.logging.project_name,
        name=f"{config.logging.run_name}_{date_str}",
        config=config,
        save_code=True,
    )
    # code
    wandb.run.log_code("./", include_fn=lambda path: path.endswith(".py"))

def setup_logger(config, date_str):
    global LOGGER
    LOGGER = config.logging.get("logger_type", "wandb")
    assert LOGGER in ["wandb", "comet_ml"], f"logger_type = {LOGGER}"
    if LOGGER == "wandb":
        setup_wandb(config, date_str)
    else:
        setup_comet(config, date_str)

def finish_logger():
    global LOGGER, EXP
    if LOGGER == "wandb":
        wandb.finish()
    elif LOGGER == "comet_ml":
        EXP.end()
    else:
        raise TypeError(f"Logger = {LOGGER}")

# Logging functions
def log_plt_fig(fig, key: str, global_step: int) -> None:
    global LOGGER, EXP
    fig.tight_layout()
    if LOGGER == "wandb":
        wandb.log({key: wandb.Image(fig)}, step=global_step)
    elif LOGGER == "comet_ml":
        EXP.log_figure(figure=fig, figure_name=key, step=global_step)
    else:
        raise TypeError(f"Logger = {LOGGER}")
    plt.close("all")

def log_metrics(d, step):
    global LOGGER, EXP
    if LOGGER == "wandb":
        wandb.log(d, step=step)
    elif LOGGER == "comet_ml":
        EXP.log_metrics(d, step=step)
    else:
        raise TypeError(f"Logger = {LOGGER}")


@torch.no_grad()
def log_t_steps_plot(
    t_steps: torch.Tensor, global_step: int = None, key: str = None
) -> None:

    t_steps = t_steps.detach().cpu().numpy()

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.plot(t_steps)

    ax.set_xlabel("Step")
    ax.set_ylabel("Time")
    ax.grid()

    if global_step is None:
        return

    log_plt_fig(fig=fig, key=key, global_step=global_step)


@torch.no_grad()
def vis_grid(a: torch.Tensor, ax=None) -> None:
    a = a.detach().cpu()

    nrow = int(np.around(np.sqrt(a.shape[0])))
    a = make_grid(a, nrow=nrow).permute(1, 2, 0).numpy()
    a = a / 2 + 0.5
    a = np.clip(a, 0, 1)
    if ax is None:
        plt.imshow(a)
    else:
        ax.imshow(a)


@torch.no_grad()
def log_end_img(
    x_s: torch.Tensor, x_t: torch.Tensor, global_step: int = None, key: str = None
) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    vis_grid(x_s, ax=ax[0])
    ax[0].axis("off")
    ax[0].set_title("Student")

    vis_grid(x_t, ax=ax[1])
    ax[1].axis("off")
    ax[1].set_title("Teacher")

    if global_step is None:
        return

    log_plt_fig(fig=fig, key=key, global_step=global_step)


@torch.no_grad()
def log_weights(model: GSWrapper, global_step: int, suff: str = "") -> None:
    d = {}
    key = f"weights_stats{suff}"

    for t, p in model.named_parameters():
        if p.requires_grad:
            data = p.data.detach().clone().cpu().numpy()
            if np.prod(data.shape) > 12:
                d[f"{key}/{t}_norm"] = np.linalg.norm(data)
                continue
            for i, v in enumerate(data):
                d[f"{key}_{t}/{i:02d}"] = v

    log_metrics(d, step=global_step)


@torch.no_grad()
def log_grads(model: GSWrapper, global_step: int) -> None:
    d = {}
    key = "grads_stats"
    for t, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            data = p.grad.detach().clone().cpu().numpy()
            if np.prod(data.shape) > 12:
                d[f"{key}/{t}_norm"] = np.linalg.norm(data)
                continue
            for i, v in enumerate(data):
                d[f"{key}_{t}/{i:02d}"] = v

    log_metrics(d, step=global_step)


@torch.no_grad()
def log_t_steps(t_steps: torch.Tensor, global_step: int, key: str = "t_stats") -> None:
    t_steps = t_steps.detach().clone().cpu().numpy()

    d = {}
    for i, t in enumerate(t_steps):
        d[f"{key}/t_{i:02d}"] = t

    log_metrics(d, step=global_step)
