# SID2 Implementation Alignment Checklist

This note documents how the repository's standalone SID2 path aligns with the
core method choices from:

- Simpler Diffusion (SiD2): https://arxiv.org/html/2410.19324v1

## Paper-grounded items implemented

- Sigmoid weighting for the denoising objective:
  - Uses SiD2-style sigmoid weighting over x-space reconstruction loss:
    - `w(lambda_t) = sigmoid(lambda_t - b)`.
  - Supports the `-d lambda_t / dt` factor from the paper objective.
  - Implemented in `examples/sid2/trainer.py`.

- Resolution-aware sigmoid bias:
  - Defaults follow appendix guidance:
    - `b=0` (128²), `b=-1` (256²), `b=-3` (512²), and `b=-4` for 1024².
  - Configured in `examples/sid2/config.py`.

- Cosine-interpolated training/sampling schedule:
  - Adds dedicated `SiD2Scheduler` with schedule types:
    - `cosine`, `shifted_cosine`, `cosine_interpolated`.
  - Default is `cosine_interpolated` with low/high noise dimensions
    generalized from the paper's `low_32_high_512` setting.
  - Implemented in `src/schedulers/scheduling_sid2.py`.

- Mean parameterization:
  - Defaults to `v` prediction, as used in SiD2 appendix settings.

## Important caveat versus full paper setup

- The strongest SiD2 results rely on architectural updates (Residual U-ViTs,
  flop-heavy scaling via patching, and guidance-interval tuning).
- This repository keeps its existing conditional UNet family for baseline
  comparability across MAVIC-T methods.
- Therefore this integration reproduces the key schedule + objective changes of
  SiD2 as a standalone baseline, but not the full U-ViT architecture stack.
