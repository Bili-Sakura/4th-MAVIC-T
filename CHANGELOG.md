# Changelog

All notable changes to **pytorch-image-translation-model** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-03-04

### Added

- Initial PyPI release of `pytorch-image-translation-model`.
- **Models**: UNet backbones (ADM, EDM, EDM2, VDM), DiT backbones (PixNerd, PixelDiT, SiT), and GAN models (CUT, StegoGAN, Pix2Pix-Turbo, CycleGAN-Turbo).
- **Schedulers**: DDBM, DDIB, BiBBDM, BDBM, DBIM, UniDB, I2SB, SiD, SiD2, CDTSDE, EDM2, VDM, DAB, StegoGAN, CUT, and Turbo schedulers.
- **Pipelines**: Inference pipelines for all supported methods (pixel-space and latent-space variants).
- **Utilities**: Training helpers, metrics (LPIPS, L1, FID), VAE utilities, dataset tools, and efficient attention support.
- Packaging via `pyproject.toml` with optional dependency groups (`training`, `metrics`, `dev`, `all`).
- GitHub Actions workflow for automated PyPI publishing on tagged releases.

[Unreleased]: https://github.com/Bili-Sakura/4th-MAVIC-T/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Bili-Sakura/4th-MAVIC-T/releases/tag/v0.1.0
