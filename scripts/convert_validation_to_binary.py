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
    Parse Grasshopper validation CSV - FULL dataset (all 4158 points per hour).
    
    The CSV has columns: Hour, pixel10*10, mrt 0, mrt 1, utci, color
    Most rows have NaN in pixel10*10 - we'll use ALL rows regardless.
    
    Args:
        csv_path: Path to validation CSV file
        
    Returns:
        Tuple of (positions, utci_by_hour, hours)
    """
    print(f"[LOAD] Reading validation CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"[INFO] Total records: {len(df):,}")
    print(f"[INFO] Columns: {list(df.columns)}")
    
    # Parse hours (format is "0-1", "1-2", etc.)
    df['hour_start'] = df['Hour'].str.split('-').str[0].astype(int)
    unique_hours = sorted(df['hour_start'].unique())
    num_hours = len(unique_hours)
    
    print(f"[INFO] Hours: {num_hours} ({min(unique_hours)} to {max(unique_hours)})")
    
    # Get rows for first hour to determine number of positions
    first_hour_data = df[df['hour_start'] == unique_hours[0]]
    num_positions = len(first_hour_data)
    
    print(f"[INFO] Positions per hour: {num_positions}")
    print(f"[INFO] Pixel IDs (non-null): {df['pixel10*10'].notna().sum()} (these are debug markers)")
    
    # Create synthetic grid positions since CSV doesn't have coordinates
    # Use the SAME coordinate range as our analysis data for proper spatial matching
    # Our 5m grid bounds: X: [-3636.6, -2651.6], Y: [-613.3, -193.3], Z: 1.5
    
    # Validation has ~4158 points, which suggests a ~65x64 grid
    # We'll create a grid that fits within our analysis bounds
    grid_rows = 65
    grid_cols = 64
    
    print(f"[INFO] Creating {grid_rows}x{grid_cols} synthetic grid matching our analysis bounds...")
    
    # Use our actual grid coordinate range
    x_min, x_max = -3636.6, -2651.6
    y_min, y_max = -613.3, -193.3
    z_height = 1.5
    
    # Create evenly spaced grid points
    x_coords = np.linspace(x_min, x_max, grid_cols)
    y_coords = np.linspace(y_min, y_max, grid_rows)
    
    positions = []
    for i in range(num_positions):
        # Map linear index to grid coordinates
        row = i // grid_cols
        col = i % grid_cols
        
        if row < grid_rows and col < grid_cols:
            x = x_coords[col]
            y = y_coords[row]
            z = z_height
            positions.append([x, y, z])
    
    positions = np.array(positions, dtype=np.float32)
    
    print(f"[INFO] Position range:")
    print(f"  X: [{positions[:, 0].min():.1f}, {positions[:, 0].max():.1f}]")
    print(f"  Y: [{positions[:, 1].min():.1f}, {positions[:, 1].max():.1f}]")
    print(f"  Z: {positions[:, 2].mean():.1f}")
    
    # Extract UTCI values organized by hour
    utci_by_hour = np.full((num_hours, num_positions), np.nan, dtype=np.float32)
    
    for hour_idx, hour in enumerate(unique_hours):
        hour_data = df[df['hour_start'] == hour].sort_index()
        
        # Take first num_positions rows for this hour
        hour_utci = hour_data['utci'].values[:num_positions]
        utci_by_hour[hour_idx, :len(hour_utci)] = hour_utci.astype(np.float32)
    
    # Report statistics
    valid_count = np.sum(~np.isnan(utci_by_hour))
    total_count = utci_by_hour.size
    print(f"[INFO] Valid UTCI values: {valid_count:,} / {total_count:,} ({valid_count/total_count*100:.1f}%)")
    
    # Show statistics per hour
    print(f"\n[STATS] UTCI by hour:")
    for hour_idx, hour in enumerate(unique_hours):
        hour_data = utci_by_hour[hour_idx, :]
        hour_valid = hour_data[~np.isnan(hour_data)]
        if len(hour_valid) > 0:
            print(f"  Hour {hour:2d}: {len(hour_valid):4d} points, "
                  f"range [{hour_valid.min():.1f}, {hour_valid.max():.1f}], "
                  f"mean {hour_valid.mean():.1f}°C")
    
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
