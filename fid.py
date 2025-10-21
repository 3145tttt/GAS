# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Script for calculating Frechet Inception Distance (FID)."""

import os
import pickle

import click
import numpy as np
import PIL.Image
import scipy.linalg
import torch
import tqdm

import dnnlib
from torch_utils import distributed as dist

# ----------------------------------------------------------------------------


class ImageFolderDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path,  # Path to directory.
    ):
        self._path = path
        assert os.path.isdir(self._path)
        self._all_fnames = {
            os.path.relpath(os.path.join(root, fname), start=self._path)
            for root, _, files in os.walk(self._path)
            for fname in files
        }

        PIL.Image.init()
        self._image_fnames = sorted(
            fname
            for fname in self._all_fnames
            if self._file_ext(fname) in PIL.Image.EXTENSION
        )
        if len(self._image_fnames) == 0:
            raise IOError("No image files found in the specified path")

        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)

        self._raw_shape = list(raw_shape)
        self.image_shape = list(self._raw_shape[1:])

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _open_file(self, fname):
        return open(os.path.join(self._path, fname), "rb")

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis]  # HW => HWC
        image = image.transpose(2, 0, 1)  # HWC => CHW
        return image

    def __len__(self):
        return self._raw_shape[0]

    def __getitem__(self, idx):
        image = self._load_raw_image(idx)
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        return image.copy(), 0.0


# ----------------------------------------------------------------------------


def calculate_inception_stats(
    image_path,
    num_expected=None,
    seed=0,
    max_batch_size=64,
    num_workers=3,
    prefetch_factor=2,
    device=torch.device("cuda"),
):
    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # Load Inception-v3 model.
    # This is a direct PyTorch translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    dist.print0("Loading Inception-v3 model...")
    detector_url = "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl"
    detector_kwargs = dict(return_features=True)
    feature_dim = 2048
    with dnnlib.util.open_url(detector_url, verbose=(dist.get_rank() == 0)) as f:
        detector_net = pickle.load(f).to(device)

    # List images.
    dist.print0(f'Loading images from "{image_path}"...')
    dataset_obj = ImageFolderDataset(path=image_path)
    if num_expected is not None and len(dataset_obj) < num_expected:
        raise click.ClickException(
            f"Found {len(dataset_obj)} images, but expected at least {num_expected}"
        )
    if len(dataset_obj) < 2:
        raise click.ClickException(
            f"Found {len(dataset_obj)} images, but need at least 2 to compute statistics"
        )

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Divide images into batches.
    num_batches = (
        (len(dataset_obj) - 1) // (max_batch_size * dist.get_world_size()) + 1
    ) * dist.get_world_size()
    all_batches = torch.arange(len(dataset_obj)).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]
    data_loader = torch.utils.data.DataLoader(
        dataset_obj,
        batch_sampler=rank_batches,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )

    # Accumulate statistics.
    dist.print0(f"Calculating statistics for {len(dataset_obj)} images...")
    mu = torch.zeros([feature_dim], dtype=torch.float64, device=device)
    sigma = torch.zeros([feature_dim, feature_dim], dtype=torch.float64, device=device)
    for images, _labels in tqdm.tqdm(
        data_loader, unit="batch", disable=(dist.get_rank() != 0)
    ):
        torch.distributed.barrier()
        if images.shape[0] == 0:
            continue
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])
        features = detector_net(images.to(device), **detector_kwargs).to(torch.float64)
        mu += features.sum(0)
        sigma += features.T @ features

    # Calculate grand totals.
    torch.distributed.all_reduce(mu)
    torch.distributed.all_reduce(sigma)
    mu /= len(dataset_obj)
    sigma -= mu.ger(mu) * len(dataset_obj)
    sigma /= len(dataset_obj) - 1
    return mu.cpu().numpy(), sigma.cpu().numpy()


# ----------------------------------------------------------------------------


def calculate_fid_from_inception_stats(mu, sigma, mu_ref, sigma_ref):
    m = np.square(mu - mu_ref).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma, sigma_ref), disp=False)
    fid = m + np.trace(sigma + sigma_ref - s * 2)
    return float(np.real(fid))


# ----------------------------------------------------------------------------


@click.group()
def main():
    """Calculate Frechet Inception Distance (FID).

    Examples:

    \b
    # Calculate FID
    torchrun --standalone --nproc_per_node=1 fid.py calc --images=fid-tmp \\
        --ref=./fid-refs/cifar10-32x32.npz
    """


# ----------------------------------------------------------------------------


@main.command()
@click.option(
    "--images",
    "image_path",
    help="Path to the images",
    metavar="PATH|ZIP",
    type=str,
    required=True,
)
@click.option(
    "--ref",
    "ref_path",
    help="Dataset reference statistics ",
    metavar="NPZ|URL",
    type=str,
    required=True,
)
@click.option(
    "--num",
    "num_expected",
    help="Number of images to use",
    metavar="INT",
    type=click.IntRange(min=2),
    default=50000,
    show_default=True,
)
@click.option(
    "--seed",
    help="Random seed for selecting the images",
    metavar="INT",
    type=int,
    default=0,
    show_default=True,
)
@click.option(
    "--batch",
    help="Maximum batch size",
    metavar="INT",
    type=click.IntRange(min=1),
    default=64,
    show_default=True,
)
def calc(image_path, ref_path, num_expected, seed, batch):
    """Calculate FID for a given set of images."""
    torch.multiprocessing.set_start_method("spawn")
    dist.init()

    dist.print0(f'Loading dataset reference statistics from "{ref_path}"...')
    ref = None
    if dist.get_rank() == 0:
        with dnnlib.util.open_url(ref_path) as f:
            ref = dict(np.load(f))

    mu, sigma = calculate_inception_stats(
        image_path=image_path,
        num_expected=num_expected,
        seed=seed,
        max_batch_size=batch,
    )
    dist.print0("Calculating FID...")
    if dist.get_rank() == 0:
        fid = calculate_fid_from_inception_stats(mu, sigma, ref["mu"], ref["sigma"])
        print(f"{fid:g}")
    torch.distributed.barrier()


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------
