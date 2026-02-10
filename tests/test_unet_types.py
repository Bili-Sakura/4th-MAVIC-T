"""Tests for all supported UNet types (adm, edm, edm2, vdm, sid).

Validates model creation, forward pass, and calling conventions
across DDBM, BiBBDM, and I2SB baselines with synthetic tensors.
"""

import pytest
import torch

from src.models.unet_ddbm import (
    DDBMUNet,
    EDMUNet,
    EDM2UNet,
    VDMUNet,
    SiDUNet,
    create_model as create_ddbm_model,
    get_unet_type_config,
    SUPPORTED_UNET_TYPES,
    UNET_TYPE_ADM,
    UNET_TYPE_EDM,
    UNET_TYPE_EDM2,
    UNET_TYPE_VDM,
    UNET_TYPE_SID,
    _parse_create_model_args,
)
from src.models.unet_bibbdm import (
    BiBBDMUNet,
    EDMBiBBDMUNet,
    EDM2BiBBDMUNet,
    VDMBiBBDMUNet,
    SiDBiBBDMUNet,
    create_model as create_bibbdm_model,
)
from src.models.unet_i2sb import (
    I2SBUNet,
    EDMI2SBUNet,
    EDM2I2SBUNet,
    VDMI2SBUNet,
    SiDI2SBUNet,
    create_model as create_i2sb_model,
)


# ---------------------------------------------------------------------------
# Type registry tests
# ---------------------------------------------------------------------------


class TestUNetTypeRegistry:
    def test_supported_types(self):
        assert SUPPORTED_UNET_TYPES == ("adm", "edm", "edm2", "vdm", "sid")

    @pytest.mark.parametrize("unet_type", SUPPORTED_UNET_TYPES)
    def test_get_config(self, unet_type):
        cfg = get_unet_type_config(unet_type)
        assert cfg["implemented"] is True
        assert isinstance(cfg["source"], str)
        assert isinstance(cfg["description"], str)

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="Unknown unet_type"):
            get_unet_type_config("invalid")


# ---------------------------------------------------------------------------
# Parse helper tests
# ---------------------------------------------------------------------------


class TestParseCreateModelArgs:
    def test_empty_attention(self):
        attn, cm = _parse_create_model_args(32, "", "")
        assert attn == ()
        assert cm is None

    def test_string_channel_mult(self):
        attn, cm = _parse_create_model_args(256, "32,16,8", "1,1,2,2,4,4")
        assert cm == (1, 1, 2, 2, 4, 4)
        assert isinstance(attn, tuple)

    def test_tuple_channel_mult(self):
        attn, cm = _parse_create_model_args(64, "32,16", (1, 2, 3, 4))
        assert cm == (1, 2, 3, 4)


# ---------------------------------------------------------------------------
# DDBM UNet type tests
# ---------------------------------------------------------------------------


_DDBM_EXPECTED_CLASSES = {
    UNET_TYPE_ADM: DDBMUNet,
    UNET_TYPE_EDM: EDMUNet,
    UNET_TYPE_EDM2: EDM2UNet,
    UNET_TYPE_VDM: VDMUNet,
    UNET_TYPE_SID: SiDUNet,
}


class TestDDBMUNetTypes:
    @pytest.mark.parametrize("unet_type", SUPPORTED_UNET_TYPES)
    def test_create_model(self, unet_type):
        kwargs = {}
        if unet_type == "edm2":
            kwargs["sigma_data"] = 0.5
        elif unet_type == "vdm":
            kwargs["gamma_min"] = -13.3
            kwargs["gamma_max"] = 5.0
        model = create_ddbm_model(
            image_size=32, in_channels=1, num_channels=32,
            num_res_blocks=1, attention_resolutions="",
            condition_mode="concat", unet_type=unet_type, **kwargs,
        )
        assert isinstance(model, _DDBM_EXPECTED_CLASSES[unet_type])

    @pytest.mark.parametrize("unet_type", SUPPORTED_UNET_TYPES)
    def test_forward_conditional(self, unet_type):
        kwargs = {}
        if unet_type == "edm2":
            kwargs["sigma_data"] = 0.5
        elif unet_type == "vdm":
            kwargs["gamma_min"] = -13.3
            kwargs["gamma_max"] = 5.0
        model = create_ddbm_model(
            image_size=32, in_channels=1, num_channels=32,
            num_res_blocks=1, attention_resolutions="",
            condition_mode="concat", unet_type=unet_type, **kwargs,
        )
        x = torch.randn(2, 1, 32, 32)
        xT = torch.randn(2, 1, 32, 32)
        t = torch.tensor([0.5, 0.8])
        out = model(x, t, xT=xT)
        assert out.shape == x.shape

    @pytest.mark.parametrize("unet_type", SUPPORTED_UNET_TYPES)
    def test_forward_unconditional(self, unet_type):
        kwargs = {}
        if unet_type == "edm2":
            kwargs["sigma_data"] = 0.5
        elif unet_type == "vdm":
            kwargs["gamma_min"] = -13.3
            kwargs["gamma_max"] = 5.0
        model = create_ddbm_model(
            image_size=32, in_channels=1, num_channels=32,
            num_res_blocks=1, attention_resolutions="",
            condition_mode=None, unet_type=unet_type, **kwargs,
        )
        x = torch.randn(2, 1, 32, 32)
        t = torch.tensor([0.5, 0.8])
        out = model(x, t)
        assert out.shape == x.shape

    def test_invalid_unet_type(self):
        with pytest.raises(ValueError, match="not supported"):
            create_ddbm_model(unet_type="invalid")


# ---------------------------------------------------------------------------
# BiBBDM UNet type tests
# ---------------------------------------------------------------------------


_BIBBDM_EXPECTED_CLASSES = {
    UNET_TYPE_ADM: BiBBDMUNet,
    UNET_TYPE_EDM: EDMBiBBDMUNet,
    UNET_TYPE_EDM2: EDM2BiBBDMUNet,
    UNET_TYPE_VDM: VDMBiBBDMUNet,
    UNET_TYPE_SID: SiDBiBBDMUNet,
}


class TestBiBBDMUNetTypes:
    @pytest.mark.parametrize("unet_type", SUPPORTED_UNET_TYPES)
    def test_create_model(self, unet_type):
        kwargs = {}
        if unet_type == "edm2":
            kwargs["sigma_data"] = 0.5
        elif unet_type == "vdm":
            kwargs["gamma_min"] = -13.3
            kwargs["gamma_max"] = 5.0
        model = create_bibbdm_model(
            image_size=32, in_channels=1, num_channels=32,
            num_res_blocks=1, attention_resolutions="",
            condition_mode="concat", objective="dlns",
            unet_type=unet_type, **kwargs,
        )
        assert isinstance(model, _BIBBDM_EXPECTED_CLASSES[unet_type])

    @pytest.mark.parametrize("unet_type", SUPPORTED_UNET_TYPES)
    def test_forward_conditional(self, unet_type):
        kwargs = {}
        if unet_type == "edm2":
            kwargs["sigma_data"] = 0.5
        elif unet_type == "vdm":
            kwargs["gamma_min"] = -13.3
            kwargs["gamma_max"] = 5.0
        model = create_bibbdm_model(
            image_size=32, in_channels=1, num_channels=32,
            num_res_blocks=1, attention_resolutions="",
            condition_mode="concat", objective="noise",
            unet_type=unet_type, **kwargs,
        )
        x_t = torch.randn(2, 1, 32, 32)
        context = torch.randn(2, 1, 32, 32)
        t = torch.tensor([0.5, 0.8])
        out = model(x_t, t, context=context)
        assert out.shape == x_t.shape

    @pytest.mark.parametrize("unet_type", SUPPORTED_UNET_TYPES)
    def test_dual_output_channels(self, unet_type):
        """Dual-learning objectives produce 2x output channels."""
        kwargs = {}
        if unet_type == "edm2":
            kwargs["sigma_data"] = 0.5
        elif unet_type == "vdm":
            kwargs["gamma_min"] = -13.3
            kwargs["gamma_max"] = 5.0
        model = create_bibbdm_model(
            image_size=32, in_channels=1, num_channels=32,
            num_res_blocks=1, attention_resolutions="",
            condition_mode="concat", objective="dlns",
            unet_type=unet_type, **kwargs,
        )
        x_t = torch.randn(2, 1, 32, 32)
        context = torch.randn(2, 1, 32, 32)
        t = torch.tensor([0.5, 0.8])
        out = model(x_t, t, context=context)
        assert out.shape == (2, 2, 32, 32)  # 2x channels for dlns

    def test_invalid_unet_type(self):
        with pytest.raises(ValueError, match="not supported"):
            create_bibbdm_model(unet_type="invalid")


# ---------------------------------------------------------------------------
# I2SB UNet type tests
# ---------------------------------------------------------------------------


_I2SB_EXPECTED_CLASSES = {
    UNET_TYPE_ADM: I2SBUNet,
    UNET_TYPE_EDM: EDMI2SBUNet,
    UNET_TYPE_EDM2: EDM2I2SBUNet,
    UNET_TYPE_VDM: VDMI2SBUNet,
    UNET_TYPE_SID: SiDI2SBUNet,
}


class TestI2SBUNetTypes:
    @pytest.mark.parametrize("unet_type", SUPPORTED_UNET_TYPES)
    def test_create_model(self, unet_type):
        kwargs = {}
        if unet_type == "edm2":
            kwargs["sigma_data"] = 0.5
        elif unet_type == "vdm":
            kwargs["gamma_min"] = -13.3
            kwargs["gamma_max"] = 5.0
        model = create_i2sb_model(
            image_size=32, in_channels=1, num_channels=32,
            num_res_blocks=1, attention_resolutions="",
            condition_mode="concat", unet_type=unet_type, **kwargs,
        )
        assert isinstance(model, _I2SB_EXPECTED_CLASSES[unet_type])

    @pytest.mark.parametrize("unet_type", SUPPORTED_UNET_TYPES)
    def test_forward_conditional(self, unet_type):
        kwargs = {}
        if unet_type == "edm2":
            kwargs["sigma_data"] = 0.5
        elif unet_type == "vdm":
            kwargs["gamma_min"] = -13.3
            kwargs["gamma_max"] = 5.0
        model = create_i2sb_model(
            image_size=32, in_channels=1, num_channels=32,
            num_res_blocks=1, attention_resolutions="",
            condition_mode="concat", unet_type=unet_type, **kwargs,
        )
        x = torch.randn(2, 1, 32, 32)
        cond = torch.randn(2, 1, 32, 32)
        t = torch.tensor([0.5, 0.8])
        out = model(x, t, cond=cond)
        assert out.shape == x.shape

    @pytest.mark.parametrize("unet_type", SUPPORTED_UNET_TYPES)
    def test_forward_unconditional(self, unet_type):
        kwargs = {}
        if unet_type == "edm2":
            kwargs["sigma_data"] = 0.5
        elif unet_type == "vdm":
            kwargs["gamma_min"] = -13.3
            kwargs["gamma_max"] = 5.0
        model = create_i2sb_model(
            image_size=32, in_channels=1, num_channels=32,
            num_res_blocks=1, attention_resolutions="",
            condition_mode=None, unet_type=unet_type, **kwargs,
        )
        x = torch.randn(2, 1, 32, 32)
        t = torch.tensor([0.5, 0.8])
        out = model(x, t)
        assert out.shape == x.shape

    def test_invalid_unet_type(self):
        with pytest.raises(ValueError, match="not supported"):
            create_i2sb_model(unet_type="invalid")


# ---------------------------------------------------------------------------
# EDM2 preconditioning-specific tests
# ---------------------------------------------------------------------------


class TestEDM2Preconditioning:
    def test_sigma_data_default(self):
        model = create_ddbm_model(
            image_size=32, in_channels=1, num_channels=32,
            num_res_blocks=1, attention_resolutions="",
            condition_mode="concat", unet_type="edm2",
        )
        assert model.sigma_data == 0.5

    def test_sigma_data_custom(self):
        model = create_ddbm_model(
            image_size=32, in_channels=1, num_channels=32,
            num_res_blocks=1, attention_resolutions="",
            condition_mode="concat", unet_type="edm2",
            sigma_data=1.0,
        )
        assert model.sigma_data == 1.0

    def test_output_differentiable(self):
        model = create_ddbm_model(
            image_size=32, in_channels=1, num_channels=32,
            num_res_blocks=1, attention_resolutions="",
            condition_mode="concat", unet_type="edm2",
        )
        x = torch.randn(1, 1, 32, 32, requires_grad=True)
        xT = torch.randn(1, 1, 32, 32)
        t = torch.tensor([0.5])
        out = model(x, t, xT=xT)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# VDM logSNR normalization tests
# ---------------------------------------------------------------------------


class TestVDMNormalization:
    def test_gamma_defaults(self):
        model = create_ddbm_model(
            image_size=32, in_channels=1, num_channels=32,
            num_res_blocks=1, attention_resolutions="",
            condition_mode="concat", unet_type="vdm",
        )
        assert model.gamma_min == -13.3
        assert model.gamma_max == 5.0

    def test_gamma_custom(self):
        model = create_ddbm_model(
            image_size=32, in_channels=1, num_channels=32,
            num_res_blocks=1, attention_resolutions="",
            condition_mode="concat", unet_type="vdm",
            gamma_min=-20.0, gamma_max=10.0,
        )
        assert model.gamma_min == -20.0
        assert model.gamma_max == 10.0

    def test_accepts_negative_timesteps(self):
        """VDM timesteps are logSNR (gamma), which can be negative."""
        model = create_ddbm_model(
            image_size=32, in_channels=1, num_channels=32,
            num_res_blocks=1, attention_resolutions="",
            condition_mode="concat", unet_type="vdm",
        )
        x = torch.randn(2, 1, 32, 32)
        xT = torch.randn(2, 1, 32, 32)
        t = torch.tensor([-5.0, 2.0])
        out = model(x, t, xT=xT)
        assert out.shape == x.shape


# ---------------------------------------------------------------------------
# ModelMixin serialization tests
# ---------------------------------------------------------------------------


class TestModelSerialization:
    @pytest.mark.parametrize("unet_type,cls", [
        ("adm", DDBMUNet),
        ("edm", EDMUNet),
        ("edm2", EDM2UNet),
        ("vdm", VDMUNet),
        ("sid", SiDUNet),
    ])
    def test_config_roundtrip(self, unet_type, cls, tmp_path):
        """save_pretrained + from_pretrained should roundtrip."""
        kwargs = {}
        if unet_type == "edm2":
            kwargs["sigma_data"] = 0.5
        elif unet_type == "vdm":
            kwargs["gamma_min"] = -13.3
            kwargs["gamma_max"] = 5.0
        model = create_ddbm_model(
            image_size=32, in_channels=1, num_channels=32,
            num_res_blocks=1, attention_resolutions="",
            condition_mode="concat", unet_type=unet_type, **kwargs,
        )
        model.save_pretrained(tmp_path / unet_type)
        loaded = cls.from_pretrained(tmp_path / unet_type)
        assert type(loaded).__name__ == type(model).__name__
