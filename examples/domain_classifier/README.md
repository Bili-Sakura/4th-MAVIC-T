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

## Train

```bash
# Default: IR-domain classifier (uses rgb2ir + sar2ir + crop_aug)
python -m examples.domain_classifier.train_domain_classifier

# EO-domain classifier
python -m examples.domain_classifier.train_domain_classifier --target_domain eo

# RGB-domain classifier
python -m examples.domain_classifier.train_domain_classifier --target_domain rgb
```

Useful overrides:

- `--train_batch_size`
- `--num_epochs`
- `--learning_rate`
- `--output_dir`
- `--positive_tasks_csv` (explicit task list)
- `--include_crop_aug true/false`

## Score generated outputs (no reference)

```bash
python -m examples.domain_classifier.score_generated \
  --model_path ./ckpt/domain_classifier/ir-resnet18-real-fake/best \
  --input_dir ./outputs/sar2ir_test
```

Output files:

- `realness_scores.csv`: per-image probabilities
- `realness_summary.json`: aggregate statistics (`mean_p_real`, quantiles, etc.)

