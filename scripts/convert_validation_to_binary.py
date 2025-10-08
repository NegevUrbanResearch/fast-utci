"""
Convert Grasshopper validation CSV to binary format for web viewer.

This script converts the 15th_Aug_MRT.csv validation data into the same
binary format used for UTCI analysis results, enabling easy comparison
in the web viewer.
"""

import struct
import numpy as np
import pandas as pd
from pathlib import Path


def parse_validation_csv(csv_path: str) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Parse Grasshopper validation CSV.
    
    The CSV has columns: Hour, pixel10*10, mrt 0, mrt 1, utci, color
    Each row represents one position at one hour.
    
    Args:
        csv_path: Path to validation CSV file
        
    Returns:
        Tuple of (positions, utci_by_hour, pixel_ids)
    """
    print(f"[LOAD] Reading validation CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"[INFO] Total records: {len(df):,}")
    print(f"[INFO] Columns: {list(df.columns)}")
    
    # The CSV uses 10x10 grid, and 'pixel10*10' is the position identifier
    # However, it doesn't contain actual x,y,z coordinates
    # We'll need to infer or use placeholder positions
    
    # Get unique pixels (positions)
    unique_pixels = df['pixel10*10'].dropna().unique()
    num_positions = len(unique_pixels)
    
    print(f"[INFO] Unique positions: {num_positions}")
    
    # Parse hours (format is "0-1", "1-2", etc.)
    df['hour_start'] = df['Hour'].str.split('-').str[0].astype(int)
    unique_hours = sorted(df['hour_start'].unique())
    num_hours = len(unique_hours)
    
    print(f"[INFO] Hours: {num_hours} ({min(unique_hours)} to {max(unique_hours)})")
    
    # Create placeholder positions (we don't have actual coordinates)
    # We'll use a grid layout based on pixel ID
    positions = []
    pixel_to_idx = {}
    
    for idx, pixel_id in enumerate(sorted(unique_pixels)):
        # Create a simple grid layout for visualization purposes
        # This won't match real positions but maintains relative relationships
        grid_x = idx % 100
        grid_y = idx // 100
        
        # Scale to match approximate real-world coordinates
        # (adjust based on actual model bounds if needed)
        x = -2470.0 + grid_x * 10.0
        y = -619.0 + grid_y * 10.0
        z = 1.5  # Human height
        
        positions.append([x, y, z])
        pixel_to_idx[pixel_id] = idx
    
    positions = np.array(positions, dtype=np.float32)
    
    # Extract UTCI values organized by hour
    utci_by_hour = np.full((num_hours, num_positions), np.nan, dtype=np.float32)
    
    for _, row in df.iterrows():
        pixel_id = row['pixel10*10']
        if pd.isna(pixel_id):
            continue
            
        hour = row['hour_start']
        utci = row['utci']
        
        if pixel_id in pixel_to_idx and hour in unique_hours:
            pos_idx = pixel_to_idx[pixel_id]
            hour_idx = unique_hours.index(hour)
            utci_by_hour[hour_idx, pos_idx] = float(utci)
    
    # Report statistics
    valid_count = np.sum(~np.isnan(utci_by_hour))
    total_count = utci_by_hour.size
    print(f"[INFO] Valid UTCI values: {valid_count:,} / {total_count:,} ({valid_count/total_count*100:.1f}%)")
    
    return positions, utci_by_hour, unique_hours


def export_validation_binary(
    positions: np.ndarray,
    utci_by_hour: np.ndarray,
    hours: list,
    output_path: Path
) -> None:
    """
    Export validation data to binary format.
    
    Uses the same format as export_for_viewer.py for consistency.
    
    Args:
        positions: Array of positions (N, 3) as float32
        utci_by_hour: Array of UTCI values (num_hours, num_positions) as float32
        hours: List of hour integers
        output_path: Path to output .bin file
    """
    num_positions = len(positions)
    num_hours = len(hours)
    
    print(f"[SAVE] Exporting {num_positions} positions × {num_hours} hours to binary...")
    
    with open(output_path, 'wb') as f:
        # Header: num_positions, num_hours (uint32, uint32)
        f.write(struct.pack('II', num_positions, num_hours))
        
        # Positions: write once (float32)
        positions_flat = positions.flatten()
        f.write(positions_flat.tobytes())
        
        # UTCI values: write hour by hour (float32)
        for hour_idx in range(num_hours):
            utci_hour = utci_by_hour[hour_idx, :]
            f.write(utci_hour.tobytes())
    
    file_size_kb = output_path.stat().st_size / 1024
    file_size_mb = file_size_kb / 1024
    print(f"[OK] Binary data: {output_path.name} ({file_size_mb:.2f} MB)")


def main():
    """Convert Grasshopper validation CSV to binary format."""
    print("=" * 60)
    print("CONVERT GRASSHOPPER VALIDATION TO BINARY")
    print("=" * 60)
    
    # Input/output paths
    csv_path = "data/validation/15th_Aug_MRT.csv"
    output_dir = Path("data/validation")
    output_path = output_dir / "grasshopper_aug15_fullday.bin"
    
    # Check input file exists
    if not Path(csv_path).exists():
        print(f"[ERROR] Input file not found: {csv_path}")
        return 1
    
    # Parse CSV
    positions, utci_by_hour, hours = parse_validation_csv(csv_path)
    
    # Export binary
    output_dir.mkdir(parents=True, exist_ok=True)
    export_validation_binary(positions, utci_by_hour, hours, output_path)
    
    # Calculate statistics
    utci_flat = utci_by_hour.flatten()
    utci_valid = utci_flat[~np.isnan(utci_flat)]
    
    print(f"\n[STATS] Validation data statistics:")
    print(f"  Positions: {len(positions):,}")
    print(f"  Hours: {len(hours)}")
    print(f"  UTCI range: {utci_valid.min():.1f}°C to {utci_valid.max():.1f}°C")
    print(f"  UTCI mean: {utci_valid.mean():.1f}°C")
    print(f"\n[OK] Conversion complete!")
    
    return 0


if __name__ == "__main__":
    exit(main())
