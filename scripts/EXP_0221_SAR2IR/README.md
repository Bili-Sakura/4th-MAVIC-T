# EXP_0221_SAR2IR

Focused recovery experiment for the failing `sar2ir` task.

- Uses **8-GPU training by default** (`NGPU=8`).
- Uses runtime random crop (`1024 -> 512`) during train.
- Uses SAR-specific lighter architecture overrides:
  - `num_channels=96`
  - `channel_mult="1,1,2,2,4,4"`
- DBIM-only script:
  - `train_dbim_sar2ir_8gpu.sh`

The 512px checkpoint from this experiment is used by
`EXP_0222_SAR2RGB_MULTIRES/train_dbim_sar2ir_1024_from_0221_8gpu.sh`
for direct 1024px tuning.

Run:

```bash
bash scripts/EXP_0221_SAR2IR/train_dbim_sar2ir_8gpu.sh
```
