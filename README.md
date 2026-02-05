# 4th-MAVIC-T

## Test Dataset Statistics

The test dataset (`datasets/mavic_t_2025_test`) contains 4 subfolders with different image translation tasks.

### Dataset Overview

| Subfolder | Images | Format | Bands | Image Size Range |
|-----------|--------|--------|-------|------------------|
| `rgb2ir` | 60 | TIFF | 3 | ~1332×1291 px (4 unique sizes) |
| `sar2eo` | 3,586 | PNG | 1 | 256×256 px (uniform) |
| `sar2ir` | 60 | TIFF | 1 | 319×302 to 2423×2518 px (39 unique sizes) |
| `sar2rgb` | 60 | TIFF | 1 | 396×374 to 2423×2518 px (39 unique sizes) |

### Detailed Statistics

#### rgb2ir (RGB to Infrared)
- **Total Images**: 60
- **Image Sizes**: 
  - Width: 1332-1333 px (mean: 1332.80, std: 0.40)
  - Height: 1291-1292 px (mean: 1291.40, std: 0.49)
  - Unique size combinations: 4
- **Pixel Value Ranges**:
  - Band 0: [0, 254], Mean: 76.06, Std: 28.92
  - Band 1: [0, 254], Mean: 80.89, Std: 26.79
  - Band 2: [0, 254], Mean: 73.56, Std: 23.75

#### sar2eo (SAR to EO)
- **Total Images**: 3,586
- **Image Sizes**: 
  - All images: 256×256 px (completely uniform)
- **Pixel Value Ranges**:
  - Band 0: [0, 255], Mean: 98.70, Std: 28.96

#### sar2ir (SAR to Infrared)
- **Total Images**: 60
- **Image Sizes**: 
  - Width: 319-2423 px (mean: 1373.92, std: 457.14)
  - Height: 302-2518 px (mean: 1422.03, std: 486.65)
  - Unique size combinations: 39
- **Pixel Value Ranges**:
  - Band 0: [0, 255], Mean: 47.26, Std: 25.14

#### sar2rgb (SAR to RGB)
- **Total Images**: 60
- **Image Sizes**: 
  - Width: 396-2423 px (mean: 1357.12, std: 461.99)
  - Height: 374-2518 px (mean: 1403.48, std: 493.07)
  - Unique size combinations: 39
- **Pixel Value Ranges**:
  - Band 0: [0, 255], Mean: 46.95, Std: 24.67

### Key Observations

1. **Uniform Dataset**: `sar2eo` contains 3,586 uniformly-sized 256×256 PNG images
2. **Near-Uniform Dataset**: `rgb2ir` has consistent sizes (~1332×1291 px) with only 4 size variations
3. **Variable Size Datasets**: Both `sar2ir` and `sar2rgb` show high variability in image dimensions (39 unique sizes each), ranging from ~300×300 to ~2400×2500 pixels
4. **Pixel Value Ranges**: All datasets use 8-bit pixel values (0-255 range), with different mean intensities across tasks

### Analysis Script

Run `analyze_test_dataset.py` to regenerate these statistics:
```bash
python analyze_test_dataset.py
```

Results are saved to `test_dataset_stats.txt`.