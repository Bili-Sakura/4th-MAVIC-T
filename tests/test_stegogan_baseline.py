# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Tests for StegoGAN baseline components."""

import pytest
import torch

from src.models.stegogan_model import (
    StegoGANGeneratorA,
    StegoGANGeneratorB,
    StegoGANDiscriminator,
    StegoGANLoss,
    create_generator_a,
    create_generator_b,
    create_discriminator,
)
from src.pipelines.stegogan import StegoGANPipeline, StegoGANPipelineOutput
from src.schedulers import StegoGANScheduler, StegoGANSchedulerOutput


class TestStegoGANGeneratorA:
    def test_create_default(self):
        gen = create_generator_a(input_nc=3, output_nc=3)
        assert isinstance(gen, StegoGANGeneratorA)

    def test_forward_shape(self):
        gen = create_generator_a(input_nc=3, output_nc=3, ngf=16, n_blocks=2)
        x = torch.randn(2, 3, 64, 64)
        out = gen(x)
        assert out.shape == (2, 3, 64, 64)

    def test_forward_with_extra_feature(self):
        gen = create_generator_a(
            input_nc=3, output_nc=3, ngf=16, n_blocks=2,
            resnet_layer=8, use_fusion_block=True,
        )
        x = torch.randn(1, 3, 64, 64)
        # Extra feature should match the bottleneck size (ngf * 4 = 64, spatial=16x16)
        extra = torch.randn(1, 64, 16, 16)
        out = gen(x, extra_feature=extra)
        assert out.shape == (1, 3, 64, 64)

    def test_forward_without_extra_feature(self):
        gen = create_generator_a(input_nc=1, output_nc=1, ngf=16, n_blocks=2)
        x = torch.randn(1, 1, 32, 32)
        out = gen(x)
        assert out.shape == (1, 1, 32, 32)

    def test_single_channel(self):
        gen = create_generator_a(input_nc=1, output_nc=1, ngf=16, n_blocks=2)
        x = torch.randn(2, 1, 32, 32)
        out = gen(x)
        assert out.shape == (2, 1, 32, 32)


class TestStegoGANGeneratorB:
    def test_create_default(self):
        gen = create_generator_b(input_nc=3, output_nc=3)
        assert isinstance(gen, StegoGANGeneratorB)

    def test_forward_returns_tuple(self):
        gen = create_generator_b(input_nc=3, output_nc=3, ngf=16, n_blocks=2)
        x = torch.randn(2, 3, 64, 64)
        result = gen(x)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_forward_shapes(self):
        gen = create_generator_b(input_nc=3, output_nc=3, ngf=16, n_blocks=2)
        x = torch.randn(2, 3, 64, 64)
        output, features_discarded, mask = gen(x)
        assert output.shape == (2, 3, 64, 64)
        assert mask.shape[0] == 2
        assert mask.shape[1] == 1  # single-channel mask

    def test_single_channel(self):
        gen = create_generator_b(input_nc=1, output_nc=1, ngf=16, n_blocks=2)
        x = torch.randn(1, 1, 32, 32)
        output, features_discarded, mask = gen(x)
        assert output.shape == (1, 1, 32, 32)


class TestStegoGANDiscriminator:
    def test_create_default(self):
        disc = create_discriminator(input_nc=3)
        assert isinstance(disc, StegoGANDiscriminator)

    def test_forward_shape(self):
        disc = create_discriminator(input_nc=3, ndf=16, n_layers=2)
        x = torch.randn(2, 3, 64, 64)
        out = disc(x)
        assert out.shape[0] == 2
        assert out.shape[1] == 1  # single output channel

    def test_single_channel_input(self):
        disc = create_discriminator(input_nc=1, ndf=16, n_layers=2)
        x = torch.randn(1, 1, 32, 32)
        out = disc(x)
        assert out.shape[0] == 1


class TestStegoGANLoss:
    def test_lsgan_real(self):
        loss_fn = StegoGANLoss(gan_mode="lsgan")
        pred = torch.randn(2, 1, 8, 8)
        loss = loss_fn(pred, True)
        assert loss.ndim == 0  # scalar

    def test_lsgan_fake(self):
        loss_fn = StegoGANLoss(gan_mode="lsgan")
        pred = torch.randn(2, 1, 8, 8)
        loss = loss_fn(pred, False)
        assert loss.ndim == 0

    def test_vanilla(self):
        loss_fn = StegoGANLoss(gan_mode="vanilla")
        pred = torch.randn(2, 1, 8, 8)
        loss = loss_fn(pred, True)
        assert loss.ndim == 0

    def test_wgangp(self):
        loss_fn = StegoGANLoss(gan_mode="wgangp")
        pred = torch.randn(2, 1, 8, 8)
        loss_real = loss_fn(pred, True)
        loss_fake = loss_fn(pred, False)
        assert loss_real.ndim == 0
        assert loss_fake.ndim == 0


class TestStegoGANScheduler:
    def test_linear_policy(self):
        params = [torch.nn.Parameter(torch.randn(1))]
        opt = torch.optim.Adam(params, lr=0.001)
        sched = StegoGANScheduler(opt, lr_policy="linear", n_epochs=10, n_epochs_decay=10)
        out = sched.step()
        assert isinstance(out, StegoGANSchedulerOutput)
        assert out.lr > 0

    def test_step_policy(self):
        params = [torch.nn.Parameter(torch.randn(1))]
        opt = torch.optim.Adam(params, lr=0.001)
        sched = StegoGANScheduler(opt, lr_policy="step", step_size=5, gamma=0.5)
        out = sched.step()
        assert isinstance(out, StegoGANSchedulerOutput)

    def test_cosine_policy(self):
        params = [torch.nn.Parameter(torch.randn(1))]
        opt = torch.optim.Adam(params, lr=0.001)
        sched = StegoGANScheduler(opt, lr_policy="cosine", n_epochs=10, n_epochs_decay=10)
        out = sched.step()
        assert isinstance(out, StegoGANSchedulerOutput)

    def test_invalid_policy(self):
        params = [torch.nn.Parameter(torch.randn(1))]
        opt = torch.optim.Adam(params, lr=0.001)
        with pytest.raises(NotImplementedError):
            StegoGANScheduler(opt, lr_policy="invalid")


class TestStegoGANPipeline:
    def _make_pipeline(self) -> StegoGANPipeline:
        gen = create_generator_a(input_nc=1, output_nc=1, ngf=16, n_blocks=2)
        return StegoGANPipeline(generator=gen)

    def test_pt_output(self):
        pipe = self._make_pipeline()
        source = torch.randn(1, 1, 32, 32)
        out = pipe(source_image=source, output_type="pt")
        assert isinstance(out, StegoGANPipelineOutput)
        assert out.images.shape == (1, 1, 32, 32)

    def test_np_output(self):
        pipe = self._make_pipeline()
        source = torch.randn(1, 1, 32, 32)
        out = pipe(source_image=source, output_type="np")
        assert isinstance(out, StegoGANPipelineOutput)
        assert out.images.shape == (1, 32, 32, 1)

    def test_pil_output(self):
        pipe = self._make_pipeline()
        source = torch.randn(1, 1, 32, 32)
        out = pipe(source_image=source, output_type="pil")
        assert isinstance(out, StegoGANPipelineOutput)
        assert len(out.images) == 1

    def test_3channel(self):
        gen = create_generator_a(input_nc=3, output_nc=3, ngf=16, n_blocks=2)
        pipe = StegoGANPipeline(generator=gen)
        source = torch.randn(2, 3, 64, 64)
        out = pipe(source_image=source, output_type="pt")
        assert out.images.shape == (2, 3, 64, 64)
