# EXP_0222_SAR2RGB_MULTIRES

Follow-up experiment for `sar2rgb` with multi-resolution training.

## Stage A: 512 crop training (runtime crop from 1024)

- `train_ddbm_sar2rgb_512_8gpu.sh`
- `train_dbim_sar2rgb_512_8gpu.sh`

## Stage B: direct 1024 fine-tune

- `train_ddbm_sar2rgb_1024_8gpu.sh`
- `train_dbim_sar2rgb_1024_8gpu.sh`

Stage B scripts auto-try to pick the latest checkpoint from Stage A.
You can override explicitly:

```bash
RESUME_FROM_CHECKPOINT=/path/to/checkpoint-XXXX bash scripts/EXP_0222_SAR2RGB_MULTIRES/train_ddbm_sar2rgb_1024_8gpu.sh
```
