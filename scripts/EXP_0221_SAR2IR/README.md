# EXP_0221_SAR2IR

Focused recovery experiment for the failing `sar2ir` task.

- Uses **8-GPU training by default** (`NGPU=8`).
- Uses runtime random crop (`1024 -> 512`) during train.
- Includes both DDBM and DBIM scripts:
  - `train_ddbm_sar2ir_8gpu.sh`
  - `train_dbim_sar2ir_8gpu.sh`

Run:

```bash
bash scripts/EXP_0221_SAR2IR/train_ddbm_sar2ir_8gpu.sh
bash scripts/EXP_0221_SAR2IR/train_dbim_sar2ir_8gpu.sh
```
