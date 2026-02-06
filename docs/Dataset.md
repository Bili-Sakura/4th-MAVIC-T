# 4th-MAVIC-T

## Dataset Structure (BiliSakura/MAVIC-T-2025)

- EO/SAR paired (Gotcha-style): `Train_Data_and_Validation_EO_SAR/EO|SAR/{train,validation}` — 68,151 train + 21,260 val PNGs, all 256×256, 1-band `uint8`, filenames 1:1.
- City SAR/IR/RGB tiles (per coordinate folder):
  - Bingham: RGB 119 (3-band `uint8`, ~3157×3240), IR 119 (1-band `uint8`), SAR 1,821 (1-band `uint8`, 653–1343 px).
  - Centerfield: RGB 843 / IR 844 (396–661 px, `uint8`), SAR 6,126 (`uint8`, 275–2583 px).
  - Manhattan: RGB 382 (mix `uint8`/`uint16`), IR 660 (mix `uint8`/`uint16`), SAR 904 (`uint8`), sizes 656–2627 px.
  - UC Davis: RGB 929 / IR 941 (`uint8`), SAR 1,977 (`uint8`), sizes 405–1350 px.
- Validation tasks (`val/`): 60 each for `rgb2ir`, `sar2ir`, `sar2rgb`.
- Test tasks (`test/`): `rgb2ir` (60), `sar2eo` (3,586), `sar2ir` (60), `sar2rgb` (60).

## Test Dataset Statistics (per task)

| Subfolder | Images | Format | Bands | Image Size Range |
|-----------|--------|--------|-------|------------------|
| `rgb2ir` | 60 | TIFF | 3 | ~1332×1291 px (4 unique sizes) |
| `sar2eo` | 3,586 | PNG | 1 | 256×256 px (uniform) |
| `sar2ir` | 60 | TIFF | 1 | 319×302 to 2423×2518 px (39 unique sizes) |
| `sar2rgb` | 60 | TIFF | 1 | 396×374 to 2423×2518 px (39 unique sizes) |

### Detailed Statistics (test)

#### rgb2ir (RGB to Infrared)
- Total: 60
- Sizes: Width 1332-1333 (mean 1332.80, std 0.40); Height 1291-1292 (mean 1291.40, std 0.49); 4 unique sizes
- Pixel ranges: Band 0 [0,254] mean 76.06; Band 1 [0,254] mean 80.89; Band 2 [0,254] mean 73.56

#### sar2eo (SAR to EO)
- Total: 3,586
- Sizes: 256×256 uniform
- Pixel ranges: Band 0 [0,255] mean 98.70

#### sar2ir (SAR to Infrared)
- Total: 60
- Sizes: Width 319-2423 (mean 1373.92, std 457.14); Height 302-2518 (mean 1422.03, std 486.65); 39 unique sizes
- Pixel ranges: Band 0 [0,255] mean 47.26

#### sar2rgb (SAR to RGB)
- Total: 60
- Sizes: Width 396-2423 (mean 1357.12, std 461.99); Height 374-2518 (mean 1403.48, std 493.07); 39 unique sizes
- Pixel ranges: Band 0 [0,255] mean 46.95

### Key Observations

1. `sar2eo` test is fully uniform (256×256, 1-band `uint8`); EO/SAR train+val are also uniform and paired by filename.
2. `rgb2ir` is near-uniform; `sar2ir` and `sar2rgb` are highly variable in size.
3. City SAR/IR/RGB sources differ in CRS/resolution; SAR often EPSG:32612 while RGB/IR often EPSG:26912 → reprojection required before pairing.
4. Manhattan RGB/IR includes `uint16`; others are `uint8`. Normalize per dtype before training.

## Proposed ML-Ready Restructure (4 image-to-image tasks)

- Layout: `data/{task}/{split}/{input,target}/...` plus manifests `manifests/{task}_{split}.csv` with `input_path,target_path,tile,source_city`.
- SAR→EO: use existing 256×256 pairs; symlink/copy into `data/sar2eo/{train,val}`.
- RGB→IR: per tile, pair `mosiac_rgb`/`*_rgb.tiff` with matching `mosiac_ir`/`*_ir.tiff`; split by tile to avoid leakage (e.g., 80/10/10 train/val/test).
- SAR→IR and SAR→RGB: for each SAR scene, reproject to the IR/RGB grid (`rasterio.warp.reproject` to target CRS/resolution) and pair with the tile’s IR/RGB mosaic; keep splits by tile.
- Optional: chip to fixed 256/512 patches with overlap for batching; store patch coords in manifests.
- Normalize consistently: scale `uint16` to [0,1] (or 0–65535) and handle `uint8` consistently (0–1 or 0–255).