# Roadmap and Experiments

> Take our previous results as a quick start point, see [staled-experiments](./staled-roadmaps-experiments/experiment_observations.md) and [staled-roadmap](./staled-roadmaps-experiments/roadmap.md).

## DDBM-Pixel-Medium (2026/02/13)


**Pipeline**: `DDBMPipeline` (DDBM in pixel space, no VAE).

| Task      | Pixel shape    | DDBM tier | ~Params |
|-----------|----------------|-----------|---------|
| RGB → IR  | `(512, 512)`   | medium    | ~120 M  |
| SAR → IR  | `(512, 512)`   | medium    | ~120 M  |
| SAR → RGB | `(512, 512)`   | medium    | ~120 M  |
| SAR → EO  | `(256, 256)`   | small     | ~20 M   |

```bash
bash scripts/DDBM_Pixel_Medium-0213/train_rgb2ir.sh
bash scripts/DDBM_Pixel_Medium-0213/train_sar2ir.sh
bash scripts/DDBM_Pixel_Medium-0213/train_sar2rgb.sh
bash scripts/DDBM_Pixel_Medium-0213/train_sar2eo.sh
```