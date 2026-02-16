# 4th-MAVIC-T

## Installation

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
| **`src/models/`** | Model implementations: `unet_ddbm`, `unet_dbim`, `unet_bibbdm`, `unet_bdbm`, `unet_i2sb`, `unet_ddib`, `cut_model`, `pix2pix_turbo`, `cyclegan_turbo`. |
| **`examples/`** | Trainer and sample scripts per method (ddib, ddbm, dbim, bibbdm, bdbm, i2sb, cut, img2img_turbo, domain_classifier). |
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

## Citations

```bibtex
@inproceedings{su2023ddib,
  title = {Dual Diffusion Implicit Bridges for Image-to-Image Translation},
  author = {Su, Xuan and Song, Jiaming and Meng, Chenlin and Ermon, Stefano},
  booktitle = {International Conference on Learning Representations},
  year = {2023},
  url = {https://openreview.net/forum?id=5HLoTvVGDe}
}

@inproceedings{zhou2024ddbm,
  title = {Denoising Diffusion Bridge Models},
  author = {Zhou, Linqi and Lou, Aaron and Khanna, Samar and Ermon, Stefano},
  booktitle = {International Conference on Learning Representations},
  year = {2024},
  url = {https://openreview.net/forum?id=FKksTayvGo}
}

@inproceedings{li2023bbdm,
  title = {BBDM: Image-to-Image Translation with Brownian Bridge Diffusion Models},
  author = {Li, Bo and Xue, Kaitao and Liu, Bin and Lai, Yu-Kun},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year = {2023},
  pages = {1952--1961},
  url = {http://openaccess.thecvf.com/content/CVPR2023/papers/Li_BBDM_Image-to-Image_Translation_With_Brownian_Bridge_Diffusion_Models_CVPR_2023_paper.pdf}
}

@article{xue2025bibbdm,
  title = {BiBBDM: Bidirectional Image Translation With Brownian Bridge Diffusion Models},
  author = {Xue, Kaitao and Li, Bo and Liu, Ziyi and He, Zhifen and Liu, Bin and Zhang, Congxuan and Lai, Yu-Kun},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year = {2025},
  volume = {47},
  number = {11},
  pages = {10546--10559},
  doi = {10.1109/TPAMI.2025.3597667}
}

@article{kieu2025bidirectional,
  title = {Bidirectional Diffusion Bridge Models},
  author = {Kieu, Duc and Do, Kien and Nguyen, Toan and Nguyen, Dang and Nguyen, Thin},
  journal = {arXiv preprint arXiv:2502.09655},
  year = {2025},
  url = {https://arxiv.org/abs/2502.09655}
}

@inproceedings{liu2023i2sb,
  title = {I2SB: Image-to-Image Schr{\"o}dinger Bridge},
  author = {Liu, Guan-Horng and Vahdat, Arash and Huang, De-An and Theodorou, Evangelos and Nie, Weili and Anandkumar, Anima},
  booktitle = {International Conference on Machine Learning},
  year = {2023},
  url = {https://openreview.net/forum?id=WH2Cy3eQd0}
}

@inproceedings{park2020cut,
  title = {Contrastive Learning for Unpaired Image-to-Image Translation},
  author = {Park, Taesung and Efros, Alexei A. and Zhang, Richard and Zhu, Jun-Yan},
  booktitle = {European Conference on Computer Vision},
  year = {2020},
  pages = {319--345},
  doi = {10.1007/978-3-030-58545-7_19}
}

@inproceedings{zhu2017cyclegan,
  title = {Unpaired Image-to-Image Translation Using Cycle-Consistent Adversarial Networks},
  author = {Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A.},
  booktitle = {Proceedings of the IEEE International Conference on Computer Vision},
  year = {2017},
  pages = {2223--2232},
  url = {https://openaccess.thecvf.com/content_iccv_2017/html/Zhu_Unpaired_Image-To-Image_Translation_ICCV_2017_paper.html}
}

@article{parmar2024onestep,
  title = {One-Step Image Translation with Text-to-Image Models},
  author = {Parmar, Gaurav and Park, Taesung and Narasimhan, Srinivasa and Zhu, Jun-Yan},
  year = {2024},
  doi = {10.48550/arXiv.2403.12036}
}
```
