from torch import nn
import dnnlib
import src.gas.adversarial_module.legacy as legacy
from src.R3GAN.Networks import Discriminator, Convolution

url_dict = {
    "CIFAR10": "https://huggingface.co/brownvc/R3GAN-CIFAR10/resolve/main/network-snapshot-final.pkl",
    "FFHQ-64x64": "https://huggingface.co/brownvc/R3GAN-FFHQ-64x64/resolve/main/network-snapshot-final.pkl"
}

class Constant(nn.Module):
    """Helper module to adapt discriminator to CIFAR dataset."""
    def __init__(self):
        super().__init__()
        
    def forward(self, y):
        return 1

def load_r3gan_disc(disc_type: str) -> Discriminator:
    """Load specified pretrained or randomly initialized discriminator."""
    assert disc_type in ['latent_3', 'latent_4', 'CIFAR10', 'FFHQ-64x64']
    
    if disc_type.startswith('latent'):
        # latent models use randomly initialized models
        in_channels = int(disc_type.split('_')[-1])
        D = load_defualt_disc(in_channels)
        return D

    # CIFAR and FFHQ have their own pretrained checkpoints
    url = url_dict[disc_type]
    with dnnlib.util.open_url(url) as f:
        d = legacy.load_network_pkl(f)
    D = d['D']

    # We don't use conditioning
    if hasattr(D.Model, 'EmbeddingLayer'):
        D.Model.EmbeddingLayer = Constant()
    return D


def load_defualt_disc(in_channels: int) -> Discriminator:
    """Load discriminator with randomly initialized weights."""
    WidthPerStage = [3 * x // 4 for x in [1024, 1024, 1024, 1024, 512]]
    BlocksPerStage = [2 * x for x in [1, 1, 1, 1, 1]]
    CardinalityPerStage = [3 * x for x in [32, 32, 32, 32, 16]]

    D = Discriminator(
        WidthPerStage=[*reversed(WidthPerStage)],
        CardinalityPerStage=[*reversed(CardinalityPerStage)],
        BlocksPerStage=[*reversed(BlocksPerStage)],
        ExpansionFactor=2,
    )
    if in_channels == 4:
        D.ExtractionLayer = Convolution(4, WidthPerStage[-1], KernelSize=1)

    return D