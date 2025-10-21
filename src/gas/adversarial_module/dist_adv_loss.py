"""
Adapted from the official R3GAN implementation:

    https://github.com/brownvc/R3GAN

See:
    Nick Huang, Aaron Gokaslan, Volodymyr Kuleshov, James Tompkin (2024).
    "The GAN is dead; long live the GAN! A Modern GAN Baseline."
    NeurIPS 2024. https://arxiv.org/abs/2501.05441
"""

import torch
import torch.nn as nn

from src.gas.adversarial_module.load_discriminator import load_r3gan_disc

class DistAdversarialTraining:
    def __init__(self, config_loss):

        self.Discriminator = load_r3gan_disc(config_loss.disc_type)

        self.Discriminator.eval()
        self.Discriminator.cuda()
        self.Discriminator.requires_grad_(False)

        self.Opt = torch.optim.Adam(self.Discriminator.parameters(), lr=config_loss.disc_lr)

    def ZeroCenteredGradientPenalty(self, Samples, Critics):
        Gradient, = torch.autograd.grad(outputs=Critics.sum(), inputs=Samples, create_graph=True)
        return Gradient.square().sum([1, 2, 3])

    def AccumulateGeneratorGradients(self, FakeSamples, RealSamples, Conditions=None, Scale=1):
        RealSamples = RealSamples.detach()

        FakeLogits = self.Discriminator(FakeSamples, Conditions)
        RealLogits = self.Discriminator(RealSamples, Conditions)

        RelativisticLogits = FakeLogits - RealLogits
        AdversarialLoss = nn.functional.softplus(-RelativisticLogits)

        return (Scale * AdversarialLoss), [x.detach() for x in [AdversarialLoss, RelativisticLogits]]

    def AccumulateDiscriminatorGradients(self, FakeSamples, RealSamples, Conditions=None, Gamma=0.2, Scale=1, is_train=True):
        RealSamples = RealSamples.detach().requires_grad_(True)
        FakeSamples = FakeSamples.detach().requires_grad_(True)

        RealLogits = self.Discriminator(RealSamples, Conditions)
        FakeLogits = self.Discriminator(FakeSamples, Conditions)

        if is_train:
            R1Penalty = self.ZeroCenteredGradientPenalty(RealSamples, RealLogits)
            R2Penalty = self.ZeroCenteredGradientPenalty(FakeSamples, FakeLogits)
        else:
            R1Penalty = torch.zeros_like(RealLogits)
            R2Penalty = torch.zeros_like(RealLogits)
        RelativisticLogits = RealLogits - FakeLogits
        AdversarialLoss = nn.functional.softplus(-RelativisticLogits)

        DiscriminatorLoss = AdversarialLoss + (Gamma / 2) * (R1Penalty + R2Penalty)

        return Scale * DiscriminatorLoss, [x.detach() for x in [AdversarialLoss, RelativisticLogits, R1Penalty, R2Penalty]]

    def discriminator_step(self, FakeSamples, RealSamples, is_train):

        self.Opt.zero_grad()
        if is_train:
            self.Discriminator.requires_grad_(True)

        loss_dis, res = self.AccumulateDiscriminatorGradients(
            FakeSamples=FakeSamples,
            RealSamples=RealSamples,
            is_train=is_train
        )

        if is_train:
            loss_dis.mean().backward()
            self.Opt.step()
            self.Discriminator.requires_grad_(False)

        return res
