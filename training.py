import datetime
import os
import time

import torch
from ml_collections import ConfigDict
from torch_ema import ExponentialMovingAverage
from tqdm import tqdm

import wandb
from evaluate import NOT_LOG_KEYS, evaluate_wrapper
from src.gas.gs_wrapper import GSWrapper
from src.gas.synt_data import SyntDataset
from src.gas.utils.loggers import log_end_img, log_grads, log_t_steps, log_weights


def train(
    config: ConfigDict,
    gs_wrapper: GSWrapper,
    ema: ExponentialMovingAverage,
    data: SyntDataset,
    optim: torch.optim.Adam,
    device: torch.device,
):
    ct = datetime.datetime.now()
    date_str = ct.strftime("%m_%d_%H_%M_%S")

    dir = os.path.join("./checkpoints", date_str)
    os.makedirs(dir, exist_ok=False)
    config.training.checkpoints_dir = dir

    print(f"\n🚀 START TRAINING: {date_str}")
    print("=" * 40 + " Config Info " + "=" * 40)
    print(config)
    print("=" * 90 + "\n")

    wandb.login(force=True)
    wandb.init(
        project=config.logging.project_name,
        name=f"{config.logging.run_name}_{date_str}",
        config=config,
        save_code=True,
    )
    wandb.run.log_code("./", include_fn=lambda path: path.endswith(".py"))

    global_step = 0
    pbar = tqdm(range(config.training.n_iters), dynamic_ncols=True)

    for _ in range(config.training.epoch_num):
        for batch in data.train_loader:
            if global_step == config.training.n_iters:
                break
            global_step += 1

            t_start = time.time()

            batch = [v.to(device) if isinstance(v, torch.Tensor) else v for v in batch]

            res_d = gs_wrapper.forward(batch=batch, return_timesteps=True)
            loss = res_d["loss_total"].mean() / config.training.iters_to_accumulate
            loss.backward()
            log_d = {"optim/time": time.time() - t_start}

            if global_step % config.training.iters_to_accumulate == 0:
                if global_step % config.logging.log_weights_freq == 0:
                    log_grads(model=gs_wrapper, global_step=global_step)

                grad_norm = torch.nn.utils.clip_grad_norm_(gs_wrapper.parameters(), 1.0)

                optim.step()
                optim.zero_grad()
                ema.update(gs_wrapper.parameters())

                if global_step % config.logging.log_weights_freq == 0:
                    log_t_steps(res_d["timesteps"], global_step=global_step)
                    log_weights(model=gs_wrapper, global_step=global_step)

                log_d["optim/grad_norm"] = grad_norm
                log_d["optim/lr"] = optim.param_groups[0]["lr"]

            for k, v in res_d.items():
                if k not in NOT_LOG_KEYS:
                    log_d[f"train/{k}"] = v.mean().item()

            wandb.log(log_d, step=global_step)

            if global_step % config.logging.eval_freq == 0 or global_step == 1:
                if "x0_s" not in res_d:
                    with torch.no_grad():
                        res_d["x0_s"] = gs_wrapper.model.decode(res_d["latents_s"])
                log_end_img(
                    res_d["x0_s"],
                    res_d["x0_t"],
                    global_step=global_step,
                    key="train/backward_end_inter",
                )

                evaluate_wrapper(
                    gs_wrapper=gs_wrapper,
                    data=data,
                    device=device,
                    suff="",
                    global_step=global_step,
                )

                with ema.average_parameters():
                    evaluate_wrapper(
                        gs_wrapper=gs_wrapper,
                        data=data,
                        device=device,
                        suff="_ema",
                        global_step=global_step,
                    )
                    log_weights(model=gs_wrapper, global_step=global_step, suff="_ema")

            if global_step % config.logging.checkpoint_freq == 0 or global_step == 1:
                torch.save(
                    {
                        "ema": ema.state_dict(),
                        "model": gs_wrapper.parameters(),
                        "optim": optim.state_dict(),
                        "step": global_step,
                    },
                    os.path.join(dir, f"{global_step}.pt"),
                )

            pbar.update(1)

    wandb.finish()
