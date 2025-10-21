import os
import pickle
from collections import defaultdict

import click
import torch
from tqdm import tqdm


@click.command()
@click.option(
    "--synt_dir",
    help="Path to the teacher dir",
    metavar="PATH",
    type=str,
    required=True,
)
@click.option(
    "--out_pkl", help="Path to pkl dataset", metavar="PATH", type=str, required=True
)
@click.option(
    "--num_samples",
    help="Number of samples to add to the final dataset",
    type=int,
    required=True,
    default=50000,
)
def main(synt_dir, out_pkl, num_samples):
    """Collate teachers subdirs

    Example:

    \b
    python collate.py --synt_dir=dir_synt --out_pkl=out_name.pkl
    """
    assert os.path.splitext(out_pkl)[1] == ".pkl"

    paths = sorted(os.listdir(synt_dir))
    data = defaultdict(list)

    pbar = tqdm(total=num_samples)
    total_samples = 0

    for p in paths:
        checkpoint = torch.load(os.path.join(synt_dir, p), weights_only=False)
        for k, v in checkpoint.items():
            if isinstance(v, torch.Tensor):
                data[k] += [v]
            elif isinstance(v, list):
                data[k] += v
            else:
                raise NotImplementedError(f"Unknown {type(v)} type to collate.")
            diff = len(v)

        total_samples += diff
        pbar.update(diff)

        if total_samples >= num_samples:
            break

    for k, v in data.items():
        data[k] = torch.cat(v, dim=0) if isinstance(v[0], torch.Tensor) else v
        data[k] = data[k][:num_samples]
        assert (
            len(data[k]) == num_samples
        ), f"Data shape is {len(data[k])}, but at least {num_samples} was expected"

    with open(out_pkl, "wb") as f:
        pickle.dump(data, f)


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------
