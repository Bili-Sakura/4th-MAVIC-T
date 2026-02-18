# SID Implementation Alignment Checklist

This note checks whether the repository's standalone SID path follows the key mechanics from the Simple Diffusion paper.

## Paper-grounded items

- Shifted cosine schedule:
  - Uses `logSNR_shifted(t) = logSNR_cosine(t) + 2*log(noise_d / image_d)`.
  - Implemented in `src/schedulers/scheduling_sid.py`.
- Forward process:
  - Uses `z_t = alpha_t * x + sigma_t * eps`, with `alpha_t^2 + sigma_t^2 = 1`.
  - Used in both scheduler `add_noise` and SID trainer objective.
- Prediction parameterization:
  - Supports both `eps` and `v` prediction.
  - SID config defaults to `v` for better high-resolution stability (as recommended in the paper).
- Optional multiscale loss:
  - Uses weighted pooled MSE over powers-of-two scales with `1/s` weighting.
  - Enabled by default in `examples/sid/config.py`.

## Important caveat versus the full paper setup

- The paper's strongest results rely on additional architectural changes (for example U-ViT scaling/downsampling design), while this repo's SID baseline keeps the repo's existing conditional UNet family and training stack for fair baseline comparability across MAVIC-T methods.
- Therefore, this implementation matches the core schedule/objective formulation, but is not a full reproduction of the paper's largest architecture ablations.

## Related references

- Simple Diffusion paper: https://arxiv.org/html/2301.11093
- Chinese blog note (user-provided): https://spaces.ac.cn/archives/10047
