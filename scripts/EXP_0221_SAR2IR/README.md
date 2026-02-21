# EXP_0221_SAR2IR

Focused recovery experiment for the failing `sar2ir` task.

- Uses **8-GPU training by default** (`NGPU=8`).
- Uses runtime random crop (`1024 -> 512`) during train.
- Uses SAR-specific lighter architecture overrides:
  - `num_channels=96`
  - `channel_mult="1,1,2,2,4,4"`
- Includes both DDBM and DBIM scripts:
  - `train_ddbm_sar2ir_8gpu.sh`
  - `train_dbim_sar2ir_8gpu.sh`

Run:

```bash
bash scripts/EXP_0221_SAR2IR/train_ddbm_sar2ir_8gpu.sh
bash scripts/EXP_0221_SAR2IR/train_dbim_sar2ir_8gpu.sh
```
