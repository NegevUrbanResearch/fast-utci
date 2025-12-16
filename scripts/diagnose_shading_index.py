"""
Diagnostic script to analyze Shading Index calculation issues.

This script compares UTCI and Shading Index patterns to identify discrepancies
and verify the calculation logic.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, Tuple
import struct


def load_binary_data(binary_path: str) -> Dict[str, Any]:
    """Load binary analysis data."""
    with open(binary_path, 'rb') as f:
        # Read header
        num_positions = struct.unpack('I', f.read(4))[0]
        num_hours = struct.unpack('I', f.read(4))[0]
        
        # Read positions
        positions = np.frombuffer(f.read(num_positions * 3 * 4), dtype=np.float32)
        positions = positions.reshape(num_positions, 3)
        
        # Check if Shading Index exists
        old_format_size = 8 + (num_positions * 3 * 4) + (num_positions * num_hours * 4)
        current_pos = f.tell()
        f.seek(0, 2)  # Seek to end
        file_size = f.tell()
        f.seek(current_pos)  # Back to position
        
        has_shading_index = False
        shading_indices = None
        
        if file_size > old_format_size:
            # New format - read has_shading_index flag
            has_shading_index = struct.unpack('I', f.read(4))[0] != 0
            if has_shading_index:
                shading_indices = np.frombuffer(f.read(num_positions * 4), dtype=np.float32)
        
        # Read UTCI data
        utci_data = np.frombuffer(f.read(num_positions * num_hours * 4), dtype=np.float32)
        utci_data = utci_data.reshape(num_positions, num_hours)
        
        return {
            'positions': positions,
            'utci_data': utci_data,
            'shading_indices': shading_indices,
            'has_shading_index': has_shading_index,
            'num_positions': num_positions,
            'num_hours': num_hours
        }


def load_metadata(metadata_path: str) -> Dict[str, Any]:
    """Load metadata JSON."""
    with open(metadata_path, 'r') as f:
        return json.load(f)


def analyze_patterns(data: Dict[str, Any], metadata: Dict[str, Any]) -> None:
    """Analyze UTCI and Shading Index patterns."""
    positions = data['positions']
    utci_data = data['utci_data']
    shading_indices = data['shading_indices']
    num_positions = data['num_positions']
    num_hours = data['num_hours']
    
    print("=" * 80)
    print("SHADING INDEX DIAGNOSTIC ANALYSIS")
    print("=" * 80)
    
    # Calculate UTCI statistics
    utci_min = np.nanmin(utci_data)
    utci_max = np.nanmax(utci_data)
    utci_mean = np.nanmean(utci_data)
    
    # Calculate mean UTCI per position (across all hours)
    utci_per_position = np.nanmean(utci_data, axis=1)
    
    print(f"\nUTCI Statistics:")
    print(f"  Min: {utci_min:.2f}°C")
    print(f"  Max: {utci_max:.2f}°C")
    print(f"  Mean: {utci_mean:.2f}°C")
    print(f"  Per-position mean range: {np.nanmin(utci_per_position):.2f} to {np.nanmax(utci_per_position):.2f}°C")
    
    if shading_indices is not None:
        shading_min = np.min(shading_indices)
        shading_max = np.max(shading_indices)
        shading_mean = np.mean(shading_indices)
        
        print(f"\nShading Index Statistics:")
        print(f"  Min: {shading_min:.3f}")
        print(f"  Max: {shading_max:.3f}")
        print(f"  Mean: {shading_mean:.3f}")
        
        # Find positions with lowest UTCI (should be most shaded)
        lowest_utci_indices = np.argsort(utci_per_position)[:10]
        highest_utci_indices = np.argsort(utci_per_position)[-10:]
        
        print(f"\n" + "=" * 80)
        print("ANALYSIS: Positions with LOWEST UTCI (should be most shaded)")
        print("=" * 80)
        print(f"{'Index':<8} {'Position (x,y,z)':<30} {'Mean UTCI':<12} {'Shading Index':<15} {'Expected':<15}")
        print("-" * 80)
        
        for idx in lowest_utci_indices:
            pos = positions[idx]
            mean_utci = utci_per_position[idx]
            shading_idx = shading_indices[idx]
            expected = "HIGH (shaded)" if shading_idx > 0.7 else "LOW (exposed) - ISSUE!"
            status = "OK" if shading_idx > 0.7 else "ISSUE"
            print(f"{idx:<8} ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})  {mean_utci:>8.2f}°C    {shading_idx:>8.3f}        {expected:<15} {status}")
        
        print(f"\n" + "=" * 80)
        print("ANALYSIS: Positions with HIGHEST UTCI (should be least shaded)")
        print("=" * 80)
        print(f"{'Index':<8} {'Position (x,y,z)':<30} {'Mean UTCI':<12} {'Shading Index':<15} {'Expected':<15}")
        print("-" * 80)
        
        for idx in highest_utci_indices:
            pos = positions[idx]
            mean_utci = utci_per_position[idx]
            shading_idx = shading_indices[idx]
            expected = "LOW (exposed)" if shading_idx < 0.3 else "HIGH (shaded) - ISSUE!"
            status = "OK" if shading_idx < 0.3 else "ISSUE"
            print(f"{idx:<8} ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})  {mean_utci:>8.2f}°C    {shading_idx:>8.3f}        {expected:<15} {status}")
        
        # Correlation analysis
        print(f"\n" + "=" * 80)
        print("CORRELATION ANALYSIS")
        print("=" * 80)
        
        # Calculate correlation (should be negative: low UTCI = high Shading Index)
        valid_mask = ~np.isnan(utci_per_position)
        correlation = np.corrcoef(utci_per_position[valid_mask], shading_indices[valid_mask])[0, 1]
        
        print(f"UTCI vs Shading Index correlation: {correlation:.4f}")
        print(f"  Expected: Negative correlation (low UTCI -> high Shading Index)")
        if correlation > -0.5:
            print(f"  WARNING: Correlation is not strongly negative! This suggests a calculation issue.")
        else:
            print(f"  OK: Correlation is negative as expected")
        
        # Count mismatches
        low_utci_mask = utci_per_position < np.percentile(utci_per_position, 10)  # Bottom 10%
        high_utci_mask = utci_per_position > np.percentile(utci_per_position, 90)  # Top 10%
        
        low_utci_low_shading = np.sum((low_utci_mask) & (shading_indices < 0.5))
        high_utci_high_shading = np.sum((high_utci_mask) & (shading_indices > 0.5))
        
        print(f"\nMismatch Counts:")
        print(f"  Low UTCI but Low Shading Index (<0.5): {low_utci_low_shading} positions")
        print(f"  High UTCI but High Shading Index (>0.5): {high_utci_high_shading} positions")
        
        if low_utci_low_shading > num_positions * 0.05:  # More than 5% mismatch
            print(f"  WARNING: Significant number of mismatches detected!")
    else:
        print("\nWARNING: No Shading Index data found in binary file!")


def main():
    """Main diagnostic function."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python diagnose_shading_index.py <metadata_json_path>")
        print("\nExample:")
        print("  python diagnose_shading_index.py data/analysis/20250815_grid_2m_fullday/metadata.json")
        sys.exit(1)
    
    metadata_path = Path(sys.argv[1])
    if not metadata_path.exists():
        print(f"Error: Metadata file not found: {metadata_path}")
        sys.exit(1)
    
    metadata = load_metadata(metadata_path)
    
    # Find binary file - use analysis_id from metadata
    metadata_dir = metadata_path.parent
    analysis_id = metadata.get('analysis_id', metadata_path.stem)
    binary_filename = f"{analysis_id}.bin"
    binary_path = metadata_dir / binary_filename
    
    if not binary_path.exists():
        print(f"Error: Binary file not found: {binary_path}")
        sys.exit(1)
    
    print(f"Loading data from:")
    print(f"  Metadata: {metadata_path}")
    print(f"  Binary: {binary_path}")
    
    data = load_binary_data(str(binary_path))
    analyze_patterns(data, metadata)


if __name__ == "__main__":
    main()

