"""UniDB model factory."""

from src.models.unet_unidb import UniDBConditionalUNet


def create_model(
    in_channels: int = 3,
    out_channels: int = 3,
    nf: int = 64,
    depth: int = 4,
) -> UniDBConditionalUNet:
    """Create UniDB ConditionalUNet (noise predictor)."""
    return UniDBConditionalUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        nf=nf,
        depth=depth,
    )
