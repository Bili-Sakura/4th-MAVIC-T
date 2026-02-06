"""
Analyze the MAVIC-T 2025 test dataset statistics.
Focus on image sizes per subfolder and pixel value ranges.
"""

import os
import numpy as np
from pathlib import Path
from collections import defaultdict
import rasterio
from PIL import Image
from tqdm import tqdm


def analyze_image(image_path, use_rasterio=False):
    """Analyze a single image and return statistics."""
    try:
        if use_rasterio:
            with rasterio.open(image_path) as src:
                # Read all bands
                data = src.read()  # Shape: (bands, height, width)
                height, width = src.height, src.width
                num_bands = src.count
                dtype = src.dtypes[0]
                
                # Calculate statistics per band
                stats = []
                for band_idx in range(num_bands):
                    band_data = data[band_idx]
                    stats.append({
                        'min': float(np.min(band_data)),
                        'max': float(np.max(band_data)),
                        'mean': float(np.mean(band_data)),
                        'std': float(np.std(band_data)),
                        'dtype': str(dtype)
                    })
        else:
            # Use PIL for PNG files
            img = Image.open(image_path)
            width, height = img.size
            num_bands = len(img.getbands())
            
            # Convert to numpy array
            img_array = np.array(img)
            
            # Handle different array shapes
            if len(img_array.shape) == 2:
                # Grayscale
                stats = [{
                    'min': float(np.min(img_array)),
                    'max': float(np.max(img_array)),
                    'mean': float(np.mean(img_array)),
                    'std': float(np.std(img_array)),
                    'dtype': str(img_array.dtype)
                }]
            else:
                # Multi-channel (RGB, etc.)
                stats = []
                for band_idx in range(img_array.shape[2]):
                    band_data = img_array[:, :, band_idx]
                    stats.append({
                        'min': float(np.min(band_data)),
                        'max': float(np.max(band_data)),
                        'mean': float(np.mean(band_data)),
                        'std': float(np.std(band_data)),
                        'dtype': str(img_array.dtype)
                    })
        
        return {
            'width': width,
            'height': height,
            'num_bands': num_bands,
            'band_stats': stats,
            'success': True
        }
    except Exception as e:
        return {
            'error': str(e),
            'success': False
        }


def analyze_subfolder(subfolder_path, subfolder_name):
    """Analyze all images in a subfolder."""
    print(f"\n{'='*60}")
    print(f"Analyzing subfolder: {subfolder_name}")
    print(f"{'='*60}")
    
    # Get all image files
    image_files = []
    for ext in ['*.tiff', '*.tif', '*.png', '*.jpg', '*.jpeg']:
        image_files.extend(list(Path(subfolder_path).glob(ext)))
    
    if not image_files:
        print(f"No image files found in {subfolder_name}")
        return None
    
    print(f"Found {len(image_files)} image files")
    
    # Determine if we should use rasterio (for TIFF) or PIL (for PNG)
    use_rasterio = any(f.suffix.lower() in ['.tiff', '.tif'] for f in image_files)
    
    # Analyze each image
    results = []
    sizes = []
    pixel_ranges = defaultdict(list)
    
    for img_path in tqdm(image_files, desc=f"Processing {subfolder_name}"):
        result = analyze_image(str(img_path), use_rasterio=use_rasterio)
        if result['success']:
            results.append(result)
            sizes.append((result['width'], result['height']))
            
            # Collect pixel ranges for each band
            for band_idx, band_stat in enumerate(result['band_stats']):
                pixel_ranges[band_idx].append({
                    'min': band_stat['min'],
                    'max': band_stat['max'],
                    'mean': band_stat['mean'],
                    'std': band_stat['std']
                })
        else:
            print(f"\nError processing {img_path.name}: {result.get('error', 'Unknown error')}")
    
    if not results:
        print(f"No successful analyses in {subfolder_name}")
        return None
    
    # Calculate statistics
    sizes_array = np.array(sizes)
    unique_sizes = set(sizes)
    
    # Overall statistics
    stats = {
        'subfolder': subfolder_name,
        'total_images': len(results),
        'image_sizes': {
            'unique_count': len(unique_sizes),
            'unique_sizes': sorted(list(unique_sizes)),
            'width': {
                'min': int(np.min(sizes_array[:, 0])),
                'max': int(np.max(sizes_array[:, 0])),
                'mean': float(np.mean(sizes_array[:, 0])),
                'std': float(np.std(sizes_array[:, 0]))
            },
            'height': {
                'min': int(np.min(sizes_array[:, 1])),
                'max': int(np.max(sizes_array[:, 1])),
                'mean': float(np.mean(sizes_array[:, 1])),
                'std': float(np.std(sizes_array[:, 1]))
            }
        },
        'num_bands': results[0]['num_bands'],
        'pixel_value_ranges': {}
    }
    
    # Calculate pixel value statistics per band
    for band_idx in range(stats['num_bands']):
        band_data = pixel_ranges[band_idx]
        stats['pixel_value_ranges'][f'band_{band_idx}'] = {
            'min': float(np.min([b['min'] for b in band_data])),
            'max': float(np.max([b['max'] for b in band_data])),
            'mean_min': float(np.mean([b['min'] for b in band_data])),
            'mean_max': float(np.mean([b['max'] for b in band_data])),
            'overall_mean': float(np.mean([b['mean'] for b in band_data])),
            'overall_std': float(np.mean([b['std'] for b in band_data])),
            'dtype': band_data[0].get('dtype', 'unknown') if band_data else 'unknown'
        }
    
    return stats


def print_statistics(stats):
    """Print formatted statistics."""
    if stats is None:
        return
    
    print(f"\n{'='*60}")
    print(f"Statistics for: {stats['subfolder']}")
    print(f"{'='*60}")
    
    print(f"\nTotal Images: {stats['total_images']}")
    print(f"Number of Bands: {stats['num_bands']}")
    
    print(f"\n--- Image Sizes ---")
    print(f"Unique size combinations: {stats['image_sizes']['unique_count']}")
    if stats['image_sizes']['unique_count'] <= 10:
        print(f"Unique sizes: {stats['image_sizes']['unique_sizes']}")
    
    print(f"\nWidth:")
    print(f"  Min: {stats['image_sizes']['width']['min']} px")
    print(f"  Max: {stats['image_sizes']['width']['max']} px")
    print(f"  Mean: {stats['image_sizes']['width']['mean']:.2f} px")
    print(f"  Std: {stats['image_sizes']['width']['std']:.2f} px")
    
    print(f"\nHeight:")
    print(f"  Min: {stats['image_sizes']['height']['min']} px")
    print(f"  Max: {stats['image_sizes']['height']['max']} px")
    print(f"  Mean: {stats['image_sizes']['height']['mean']:.2f} px")
    print(f"  Std: {stats['image_sizes']['height']['std']:.2f} px")
    
    print(f"\n--- Pixel Value Ranges ---")
    for band_name, band_stats in stats['pixel_value_ranges'].items():
        print(f"\n{band_name.upper()}:")
        print(f"  Global Min: {band_stats['min']:.2f}")
        print(f"  Global Max: {band_stats['max']:.2f}")
        print(f"  Mean Min (across images): {band_stats['mean_min']:.2f}")
        print(f"  Mean Max (across images): {band_stats['mean_max']:.2f}")
        print(f"  Overall Mean: {band_stats['overall_mean']:.2f}")
        print(f"  Overall Std: {band_stats['overall_std']:.2f}")
        print(f"  Data Type: {band_stats['dtype']}")


def main():
    """Main function to analyze the test dataset."""
    dataset_path = Path("datasets/mavic_t_2025_test")
    
    if not dataset_path.exists():
        print(f"Error: Dataset path {dataset_path} does not exist!")
        return
    
    print(f"Analyzing test dataset at: {dataset_path.absolute()}")
    
    # Get all subfolders
    subfolders = [d for d in dataset_path.iterdir() if d.is_dir()]
    
    if not subfolders:
        print("No subfolders found!")
        return
    
    print(f"\nFound {len(subfolders)} subfolders: {[s.name for s in subfolders]}")
    
    # Analyze each subfolder
    all_stats = []
    for subfolder in subfolders:
        stats = analyze_subfolder(subfolder, subfolder.name)
        if stats:
            print_statistics(stats)
            all_stats.append(stats)
    
    # Print summary
    print(f"\n\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"\nTotal subfolders analyzed: {len(all_stats)}")
    for stats in all_stats:
        print(f"\n{stats['subfolder']}:")
        print(f"  Images: {stats['total_images']}")
        print(f"  Size range: {stats['image_sizes']['width']['min']}x{stats['image_sizes']['height']['min']} to "
              f"{stats['image_sizes']['width']['max']}x{stats['image_sizes']['height']['max']}")
        print(f"  Unique sizes: {stats['image_sizes']['unique_count']}")
        for band_name, band_stats in stats['pixel_value_ranges'].items():
            print(f"  {band_name} pixel range: [{band_stats['min']:.2f}, {band_stats['max']:.2f}]")


if __name__ == "__main__":
    main()
