# 4th-MAVIC-T

[![PyPI version](https://img.shields.io/pypi/v/pytorch-image-translation-model.svg)](https://pypi.org/project/pytorch-image-translation-model) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![SAR2EO](https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg)](https://swanlab.cn/@EarthBridge/sar2eo/overview) [![SAR2IR](https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg)](https://swanlab.cn/@EarthBridge/sar2ir/overview) [![RGB2IR](https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg)](https://swanlab.cn/@EarthBridge/rgb2ir/overview) [![SAR2RGB](https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg)](https://swanlab.cn/@EarthBridge/sar2rgb/overview)

## Installation

### Install from PyPI

```bash
# Core library only
pip install pytorch-image-translation-model

# With training extras (accelerate, peft, datasets, tensorboard, swanlab)
pip install pytorch-image-translation-model[training]

# With metrics extras (torchmetrics, lpips, torch-fidelity, scipy)
pip install pytorch-image-translation-model[metrics]

# Everything
pip install pytorch-image-translation-model[all]
```

> **Note:** PyTorch is listed as a dependency but you may want to install a specific CUDA build first.
> See [PyTorch — Get Started](https://pytorch.org/get-started/previous-versions/) for details.

### Install from source (development)

1. Install from `requirements.txt` (recommended)

```bash
conda create -n rsgen python=3.12
conda activate rsgen
# we are using PyTorch 2.8.0 torchaudio 2.8.0 torchvision 0.23.0 from https://download.pytorch.org/whl/cu126
# other version mostly would work as long installed follow https://pytorch.org/get-started/previous-versions/
pip install torch==2.8.0+cu126 torchaudio==2.8.0+cu126 torchvision==0.23.0+cu126 --index-url https://download.pytorch.org/whl/cu126
# install other packages
pip install -r requirements.txt
pip install swanlab
# optional
# pip install muon-optimizer
```

2. Install from `environment.yaml`

```bash
conda env create -f environment.yaml
conda activate rsgen
```

3. Editable install (for contributors)

```bash
pip install -e ".[dev]"
```

### Path configuration (optional)

If you clone the repo to a custom location, set `PROJECT_ROOT` to your project directory. Scripts will then resolve paths relative to it.

```bash
# Option 1: Source paths.env (auto-detects project root from file location)
source paths.env

# Option 2: Set manually before running scripts
export PROJECT_ROOT=/path/to/4th-MAVIC-T
```

Without this, paths are inferred from the script location (works when run from the project root).

### Project structure

| Directory | Purpose |
| :--- | :--- |
| **`datasets/`** | `BiliSakura/MACIV-T-2025-Structure-Refined`: `manifests/`, `{task}/train/{input,target}/`, `val/{task}/input/`, `test/{task}/`. See `docs/dataset.md`. |
| **`models/`** | Pre-trained model weights. |
| **`src/models/`** | Model implementations: `unet_ddbm`, `unet_dbim`, `unet_bibbdm`, `unet_bdbm`, `unet_i2sb`, `unet_cdtsde`, `unet_ddib`, `unet_unidb`, `cut_model`, `stegogan_model`, `pix2pix_turbo`, `cyclegan_turbo`. |
| **`examples/`** | Trainer and sample scripts per method (ddib, ddbm, dbim, bibbdm, bdbm, i2sb, sid, sid2, cdtsde, cut, stegogan, img2img_turbo, domain_classifier). |
| **`scripts/`** | Training launchers, dataset preparation, manifest rewriting, and utilities. |
| **`ckpt/`** | Checkpoints and SwanLab logs from training runs. |

### Pre-trained models (MaRS-Base)

Some scripts use pre-trained MaRS encoders for representation alignment or validation-set creation. Please pre-download them from [HuggingFace/BiliSakura](https://huggingface.co/BiliSakura) to your local `models/` folder:

| Model | HuggingFace ID | Local path |
| :--- | :--- | :--- |
| MaRS-Base-RGB | `BiliSakura/MaRS-Base-RGB` | `models/BiliSakura/MaRS-Base-RGB` |
| MaRS-Base-SAR | `BiliSakura/MaRS-Base-SAR` | `models/BiliSakura/MaRS-Base-SAR` |

```bash
# From project root
mkdir -p models/BiliSakura
huggingface-cli download BiliSakura/MaRS-Base-RGB --local-dir models/BiliSakura/MaRS-Base-RGB
huggingface-cli download BiliSakura/MaRS-Base-SAR --local-dir models/BiliSakura/MaRS-Base-SAR
```

If you use a custom project location, ensure the paths resolve correctly (e.g. via `PROJECT_ROOT` or by placing the models under your project’s `models/BiliSakura/` directory).

### Rewriting manifest paths

Manifest files (`paired_val_*.txt`, `bad_samples.txt`) may contain machine-specific absolute paths. Run the following to rewrite them for your setup:

```bash
# Dry run first
python scripts/rewrite_manifest_paths.py --dry_run

# Rewrite to your dataset root
python scripts/rewrite_manifest_paths.py --dataset_root /path/to/MACIV-T-2025-Structure-Refined

# Or use PROJECT_ROOT from env (see Path configuration above)
python scripts/rewrite_manifest_paths.py
```

### Experiment tracking with SwanLab

Training scripts support [SwanLab](https://swanlab.cn) for experiment tracking. Install with `pip install swanlab` (see Installation above).

**Enable SwanLab** — The DDBM scripts in `scripts/DDBM_Pixel_Medium-0213/` already use `--log_with swanlab`. For other trainers, add:

```bash
--log_with swanlab
```

**Log location** — SwanLab logs are stored under `./ckpt/swanlog` (full path: `ckpt/swanlog/run-<experiment_id>`).

**Optional metadata** — Customize run name, tags, and description:

```bash
--log_with swanlab \
--swanlab_experiment_name my-run-name \
--swanlab_tags baseline,rgb2ir \
--swanlab_description "DDBM Pixel Medium RGB→IR"
```

**Storage modes** — By default, data syncs to SwanLab cloud. For offline-only logging:

```bash
--swanlab_init_kwargs_json '{"mode":"offline"}'
```

To sync offline logs later: `swanlab sync ./ckpt/swanlog/run-xxx`

### Standalone SID / SID2 baselines

SID and SID2 are available as dedicated baseline entrypoints (instead of only a UNet type toggle under other methods):

```bash
# Example: standalone SID on rgb2ir
python -m examples.sid.train --task rgb2ir

# Example: standalone SID2 on rgb2ir
python -m examples.sid2.train --task rgb2ir
```

Implementation notes:

- SID alignment checklist: `docs/sid_alignment_checklist.md`
- SID2 alignment checklist: `docs/sid2_alignment_checklist.md`

## Credits

### Library credits

<a href="https://github.com/huggingface/diffusers">diffusers</a>.

### Reference papers

<a href="https://openreview.net/forum?id=FKksTayvGo">Denoising Diffusion Bridge Models (DDBM, ICLR 2024)</a>

<a href="https://openreview.net/forum?id=5HLoTvVGDe">Dual Diffusion Implicit Bridges (DDIB, ICLR 2023)</a>

<a href="http://openaccess.thecvf.com/content/CVPR2023/papers/Li_BBDM_Image-to-Image_Translation_With_Brownian_Bridge_Diffusion_Models_CVPR_2023_paper.pdf">BBDM (CVPR 2023)</a>

<a href="https://doi.org/10.1109/TPAMI.2025.3597667">BiBBDM (TPAMI 2025)</a>

<a href="https://arxiv.org/abs/2502.09655">Bidirectional Diffusion Bridge Models (DBIM/UniDB)</a>

<a href="https://openreview.net/forum?id=WH2Cy3eQd0">I2SB (ICML 2023)</a>


<a href="https://openreview.net/forum?id=it0GTdiW9t">CDTSDE (ICLR 2026)</a>

<a href="https://proceedings.mlr.press/v202/hoogeboom23a.html">Simple Diffusion (SiD, ICML 2023)</a>

<a href="https://openaccess.thecvf.com/content/CVPR2025/html/Hoogeboom_Simpler_Diffusion_1.5_FID_on_ImageNet512_with_Pixel-space_Diffusion_CVPR_2025_paper.html">Simpler Diffusion / SiD2 (CVPR 2025)</a>

<a href="http://openaccess.thecvf.com//content/CVPR2024/papers/Karras_Analyzing_and_Improving_the_Training_Dynamics_of_Diffusion_Models_CVPR_2024_paper.pdf">Analyzing and Improving the Training Dynamics of Diffusion Models (EDM2, CVPR 2024)</a>

<a href="https://proceedings.neurips.cc/paper/2021/hash/b578f2a52a0229873fefc2a4b06377fa-Abstract.html">Variational Diffusion Models (VDM, NeurIPS 2021)</a>

<a href="https://link.springer.com/chapter/10.1007/978-3-030-58545-7_19">Contrastive Unpaired Translation (CUT, ECCV 2020)</a>

<a href="https://openaccess.thecvf.com/content_iccv_2017/html/Zhu_Unpaired_Image-To-Image_Translation_ICCV_2017_paper.html">CycleGAN (ICCV 2017)</a>

<a href="https://doi.org/10.48550/arXiv.2403.12036">One-Step Image Translation with Text-to-Image Models (img2img-turbo, 2024)</a>

<a href="https://arxiv.org/abs/2410.06940">Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think (REPA, ICLR 2025)</a>

<a href="https://openaccess.thecvf.com/content/CVPR2025/html/Xiao_Deterministic_Image-to-Image_Translation_via_Denoising_Brownian_Bridge_Models_with_Dual_CVPR_2025_paper.html">Deterministic Image-to-Image Translation via Denoising Brownian Bridge Models with Dual Approximators (DAB, CVPR 2025)</a>

<a href="https://openaccess.thecvf.com/content/CVPR2024/papers/Wu_StegoGAN_Leveraging_Steganography_for_Non-Bijective_Image-to-Image_Translation_CVPR_2024_paper.pdf">StegoGAN: Leveraging Steganography for Non-Bijective Image-to-Image Translation (CVPR 2024)</a>
