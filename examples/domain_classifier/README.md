# Domain Classifier Pre-Training (ResNet-18)

This example pre-trains a **binary real/fake classifier** on MAVIC-T target domains.
It is intended for **no-reference evaluation** of generated images on val/test splits:
the classifier outputs a soft probability `p_real` for each generated image.

## Model

- Backbone: `transformers.ResNetForImageClassification` (ResNet-18)
- Initialization: `Francesco/resnet18-224-1k`
- Head replacement: ImageNet 1000-way head -> 2-way head (`fake`, `real`)
- Input channel adaptation:
  - `EO`, `IR`, `SAR` -> 1 channel
  - `RGB` -> 3 channels
  - first convolution weights are adapted when channels differ

## Data protocol

Training uses **only refined `train` split** (never val/test):

- Positive (`real`) samples: `target_path` from tasks of the target domain.
- Negative (`fake`) samples: `input_path` from the same task records by default.
- `*_crop_aug` tasks are included automatically when available.

Native resolutions are kept by default:

- `EO`: 256
- `IR`: 1024
- `RGB`: 1024

## Domain-specific normalization

For domain datasets, we pre-compute **mean and std** over the positive (real) images
instead of using ImageNet statistics. This improves training when the domain distribution
differs from natural images.

```bash
# Pre-compute stats for IR domain
python -m examples.domain_classifier.compute_dataset_stats --target_domain ir --output_path ./stats/ir_domain_stats.json

# Train with dataset stats
python -m examples.domain_classifier.train_domain_classifier --target_domain ir --dataset_stats_path ./stats/ir_domain_stats.json
```

## Train

### Ready-to-use scripts (recommended)

```bash
# Train IR-domain classifier (auto-computes stats if missing)
bash scripts/train_domain_classifier_ir.sh

# Train EO-domain classifier
bash scripts/train_domain_classifier_eo.sh

# Train RGB-domain classifier
bash scripts/train_domain_classifier_rgb.sh

# Pre-compute stats for all domains
bash scripts/compute_domain_stats.sh ir eo rgb
```

### Manual training

```bash
# Default: IR-domain classifier (uses rgb2ir + sar2ir + crop_aug)
python -m examples.domain_classifier.train_domain_classifier

# EO-domain classifier
python -m examples.domain_classifier.train_domain_classifier --target_domain eo

# RGB-domain classifier with domain stats
python -m examples.domain_classifier.train_domain_classifier --target_domain rgb --dataset_stats_path ./stats/rgb_domain_stats.json
```

Useful overrides:

- `--train_batch_size`
- `--num_epochs`
- `--learning_rate`
- `--output_dir`
- `--dataset_stats_path` (path to pre-computed mean/std JSON)
- `--positive_tasks_csv` (explicit task list)
- `--include_crop_aug true/false`

## Score generated outputs (no reference)

```bash
python -m examples.domain_classifier.score_generated \
  --model_path ./ckpt/domain_classifier/ir-resnet18-real-fake/best \
  --input_dir ./outputs/sar2ir_test
```

Scoring automatically uses the normalization stats from `training_setup.json` when
the model was trained with `--dataset_stats_path`. Override with `--dataset_stats_path`
if needed.

Output files:

- `realness_scores.csv`: per-image probabilities
- `realness_summary.json`: aggregate statistics (`mean_p_real`, quantiles, etc.)

