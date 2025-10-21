"""
ABOUTME: Convert new detailed Grasshopper validation CSV to binary format for web viewer
This handles the 15_Aug_10_11.csv format with actual X,Y,Z coordinates and human_to_sky values
"""

import struct
import numpy as np
import pandas as pd
from pathlib import Path

def parse_new_validation_csv(csv_path: str) -> tuple:
    """
    Parse new Grasshopper validation CSV with actual coordinates.
    
    Format: x, y, z, human_to_sky, mrt0, mrt1, utci 0, utci 1, utci average
    
    Args:
        csv_path: Path to validation CSV file
        
    Returns:
        Tuple of (positions, utci_by_hour, human_to_sky_values, num_hours)
    """
    print(f"[LOAD] Reading validation CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"[INFO] Total records: {len(df):,}")
    print(f"[INFO] Columns: {list(df.columns)}")
    
    # Extract unique positions
    num_positions = len(df)
    print(f"[INFO] Positions: {num_positions:,}")
    
    # Create positions array (X, Y, Z)
    positions = np.column_stack([
        df['x'].values,
        df['y'].values,
        df['z'].values
    ]).astype(np.float32)
    
    # Extract human_to_sky values
    human_to_sky = df['human_to_sky'].values.astype(np.float32)
    
    print(f"\nHuman-to-sky distribution:")
    for val in sorted(np.unique(human_to_sky)):
        count = np.sum(human_to_sky == val)
        print(f"  {val}: {count} points ({count/len(human_to_sky)*100:.1f}%)")
    
    # We have UTCI for two hours (10 and 11), but we need full day (24 hours)
    # Create 24-hour array, filling hours 0-9 and 12-23 with NaN
    num_hours = 24
    utci_by_hour = np.full((num_hours, num_positions), np.nan, dtype=np.float32)
    
    # Fill hour 10 and 11
    utci_by_hour[10, :] = df['utci 0'].values.astype(np.float32)
    utci_by_hour[11, :] = df['utci 1'].values.astype(np.float32)
    
    # Report statistics
    valid_count = np.sum(~np.isnan(utci_by_hour))
    total_count = utci_by_hour.size
    print(f"\n[INFO] UTCI values filled:")
    print(f"  Hour 10: {num_positions:,} values")
    print(f"  Hour 11: {num_positions:,} values")
    print(f"  Other hours: filled with NaN (not in validation data)")
    print(f"  Total: {valid_count:,} / {total_count:,} ({valid_count/total_count*100:.1f}%)")
    
    return positions, utci_by_hour, human_to_sky, num_hours

def export_validation_binary(
    positions: np.ndarray,
    utci_by_hour: np.ndarray,
    human_to_sky: np.ndarray,
    output_path: Path
) -> None:
    """
    Export validation data to binary format with exposure data.
    
    Extended format that includes human_to_sky values:
    - Header: num_positions (uint32), num_hours (uint32)
    - Positions: num_positions × 3 floats (x, y, z)
    - Human-to-sky: num_positions floats
    - UTCI by hour: num_hours × num_positions floats
    
    Args:
        positions: Array of positions (N, 3) as float32
        utci_by_hour: Array of UTCI values (num_hours, num_positions) as float32
        human_to_sky: Array of human-to-sky exposure values (N,) as float32
        output_path: Path to output .bin file
    """
    num_positions = len(positions)
    num_hours = utci_by_hour.shape[0]
    
    print(f"\n[SAVE] Exporting {num_positions:,} positions × {num_hours} hours to binary...")
    
    with open(output_path, 'wb') as f:
        # Header: num_positions, num_hours (uint32, uint32)
        f.write(struct.pack('II', num_positions, num_hours))
        
        # Positions: write once (float32)
        positions_flat = positions.flatten()
        f.write(positions_flat.tobytes())
        
        # Human-to-sky exposure values (float32)
        f.write(human_to_sky.tobytes())
        
        # UTCI values: write hour by hour (float32)
        for hour_idx in range(num_hours):
            utci_hour = utci_by_hour[hour_idx, :]
            f.write(utci_hour.tobytes())
    
    file_size_kb = output_path.stat().st_size / 1024
    file_size_mb = file_size_kb / 1024
    print(f"[OK] Binary data: {output_path.name} ({file_size_mb:.2f} MB)")

def main():
    """Convert new Grasshopper validation CSV to binary format."""
    print("=" * 80)
    print("CONVERT NEW GRASSHOPPER VALIDATION TO BINARY")
    print("=" * 80)
    
    # Input/output paths
    csv_path = "data/validation/15_Aug_10_11.csv"
    output_dir = Path("data/validation")
    output_path = output_dir / "grasshopper_aug15_detailed.bin"
    
    # Check input file exists
    if not Path(csv_path).exists():
        print(f"\n[ERROR] Input file not found: {csv_path}")
        print("This script requires the new detailed validation file.")
        return 1
    
    # Parse CSV
    positions, utci_by_hour, human_to_sky, num_hours = parse_new_validation_csv(csv_path)
    
    # Export binary
    output_dir.mkdir(parents=True, exist_ok=True)
    export_validation_binary(positions, utci_by_hour, human_to_sky, output_path)
    
    # Calculate statistics (only hours 10 and 11)
    utci_h10 = utci_by_hour[10, :]
    utci_h11 = utci_by_hour[11, :]
    utci_both = np.concatenate([utci_h10, utci_h11])
    
    print(f"\n[STATS] Validation data statistics:")
    print(f"  Positions: {len(positions):,}")
    print(f"  Hours with data: 10-11 (out of 24 total)")
    print(f"  Position range:")
    print(f"    X: [{positions[:,0].min():.2f}, {positions[:,0].max():.2f}]")
    print(f"    Y: [{positions[:,1].min():.2f}, {positions[:,1].max():.2f}]")
    print(f"    Z: [{positions[:,2].min():.2f}, {positions[:,2].max():.2f}]")
    print(f"  UTCI range (hours 10-11): {utci_both.min():.1f}°C to {utci_both.max():.1f}°C")
    print(f"  UTCI mean: {utci_both.mean():.1f}°C")
    print(f"  Human-to-sky values: {sorted(np.unique(human_to_sky))}")
    
    print(f"\n[OK] Conversion complete!")
    print(f"\n💡 NOTE: This binary includes human-to-sky exposure data.")
    print(f"   The viewer will need to be updated to read and use this data.")
    
    return 0

if __name__ == "__main__":
    exit(main())

