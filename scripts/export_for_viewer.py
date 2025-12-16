"""
Export UTCI analysis results to optimized binary format for web viewer.

This script converts UTCI results from demo_workflow.py into:
1. Binary data files (.bin) for fast loading
2. JSON metadata files with analysis information

Binary format:
- Single hour: [num_positions][positions: x,y,z][utci_values]
- Full day: [num_positions, num_hours][positions: x,y,z][utci_h0...h23]
"""

import struct
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime


def generate_analysis_id(
    date: str,
    grid_size: float,
    analysis_type: str,
    hour: Optional[int] = None
) -> str:
    """
    Generate descriptive analysis ID for filename.
    
    Args:
        date: Date string in format YYYYMMDD (e.g., "20250815")
        grid_size: Grid spacing in meters (e.g., 2.0)
        analysis_type: "single_hour" or "full_day"
        hour: Hour of day (0-23) for single hour analysis
        
    Returns:
        Analysis ID string (e.g., "20250815_grid_2m_single_h13")
    """
    grid_str = f"grid_{grid_size:.0f}m" if grid_size == int(grid_size) else f"grid_{grid_size}m"
    
    if analysis_type == "single_hour" and hour is not None:
        return f"{date}_{grid_str}_single_h{hour:02d}"
    else:
        return f"{date}_{grid_str}_fullday"


def extract_date_from_period(analysis_period: Any) -> str:
    """
    Extract date string from analysis period.
    
    Args:
        analysis_period: Ladybug AnalysisPeriod object
        
    Returns:
        Date string in YYYYMMDD format
    """
    if hasattr(analysis_period, 'start_month') and hasattr(analysis_period, 'start_day'):
        # Assume current year or 2025 as default
        year = 2025
        month = analysis_period.start_month
        day = analysis_period.start_day
        return f"{year}{month:02d}{day:02d}"
    return "20250815"  # Default fallback


def export_binary_single_hour(
    utci_results: Dict[str, Any],
    output_path: Path
) -> None:
    """
    Export single hour UTCI data to binary format.
    
    Binary structure:
        [4 bytes: num_positions]
        [num_positions × 12 bytes: positions as float32 x,y,z]
        [num_positions × 4 bytes: utci values as float32]
    
    Args:
        utci_results: UTCI results dictionary
        output_path: Path to output .bin file
    """
    # Extract positions and UTCI values
    positions = []
    utci_values = []
    
    for pos_key in sorted(utci_results.keys()):
        data = utci_results[pos_key]
        pos = data['position']
        utci = data['utci']
        
        # For single hour, utci should be a single value or list with one element
        if isinstance(utci, (list, np.ndarray)):
            utci_val = float(utci[0])
        else:
            utci_val = float(utci)
        
        positions.append([float(pos[0]), float(pos[1]), float(pos[2])])
        utci_values.append(utci_val)
    
    num_positions = len(positions)
    
    # Write binary file
    with open(output_path, 'wb') as f:
        # Header: num_positions (uint32)
        f.write(struct.pack('I', num_positions))
        
        # Positions: flatten and write as float32
        positions_flat = np.array(positions, dtype=np.float32).flatten()
        f.write(positions_flat.tobytes())
        
        # UTCI values: write as float32
        utci_array = np.array(utci_values, dtype=np.float32)
        f.write(utci_array.tobytes())
    
    file_size_kb = output_path.stat().st_size / 1024
    print(f"[SAVE] Binary data: {output_path.name} ({file_size_kb:.1f} KB)")


def export_binary_full_day(
    utci_results: Dict[str, Any],
    output_path: Path,
    num_hours: int = 24,
    shading_indices: Optional[np.ndarray] = None
) -> None:
    """
    Export full day UTCI data to binary format.
    
    Binary structure (with optional Shading Index):
        [8 bytes: num_positions (uint32), num_hours (uint32)]
        [num_positions × 12 bytes: positions as float32 x,y,z]
        [4 bytes: has_shading_index (uint32, 0 or 1)]
        [IF has_shading_index == 1: num_positions × 4 bytes: shading_index as float32]
        [num_positions × 4 bytes: utci values hour 0 as float32]
        [num_positions × 4 bytes: utci values hour 1 as float32]
        ...
        [num_positions × 4 bytes: utci values hour 23 as float32]
    
    Args:
        utci_results: UTCI results dictionary
        output_path: Path to output .bin file
        num_hours: Number of hours (default 24)
        shading_indices: Optional Shading Index array (shape: n_positions,)
    """
    # Extract positions and UTCI values organized by hour
    sorted_keys = sorted(utci_results.keys())
    num_positions = len(sorted_keys)
    
    positions = []
    utci_by_hour = [[] for _ in range(num_hours)]
    
    for pos_key in sorted_keys:
        data = utci_results[pos_key]
        pos = data['position']
        utci_vals = data['utci']
        
        positions.append([float(pos[0]), float(pos[1]), float(pos[2])])
        
        # Organize UTCI values by hour
        if isinstance(utci_vals, (list, np.ndarray)):
            for hour_idx, utci_val in enumerate(utci_vals[:num_hours]):
                utci_by_hour[hour_idx].append(float(utci_val))
        else:
            # Single value - replicate for all hours
            for hour_idx in range(num_hours):
                utci_by_hour[hour_idx].append(float(utci_vals))
    
    # Validate shading_indices if provided
    has_shading_index = shading_indices is not None
    if has_shading_index:
        shading_indices = np.asarray(shading_indices, dtype=np.float32)
        if len(shading_indices) != num_positions:
            raise ValueError(
                f"shading_indices length ({len(shading_indices)}) doesn't match "
                f"num_positions ({num_positions})"
            )
    
    # Write binary file
    with open(output_path, 'wb') as f:
        # Header: num_positions, num_hours (uint32, uint32)
        f.write(struct.pack('II', num_positions, num_hours))
        
        # Positions: write once (float32)
        positions_flat = np.array(positions, dtype=np.float32).flatten()
        f.write(positions_flat.tobytes())
        
        # Shading Index flag and data (if present)
        f.write(struct.pack('I', 1 if has_shading_index else 0))
        if has_shading_index:
            f.write(shading_indices.tobytes())
        
        # UTCI values: write hour by hour (float32)
        for hour_idx in range(num_hours):
            utci_array = np.array(utci_by_hour[hour_idx], dtype=np.float32)
            f.write(utci_array.tobytes())
    
    file_size_kb = output_path.stat().st_size / 1024
    file_size_mb = file_size_kb / 1024
    print(f"[SAVE] Binary data: {output_path.name} ({file_size_mb:.2f} MB)")


# NOTE: _calculate_hour_statistics has been moved to fast_utci.utci.statistics
# to avoid code duplication. Import it instead of defining it here.


def _extract_location_from_epw(epw_file: str) -> Dict[str, Any]:
    """
    Extract location data from EPW weather file.
    
    Args:
        epw_file: Path to EPW file
        
    Returns:
        Dictionary with latitude, longitude, timezone, city
    """
    from ladybug.epw import EPW
    
    epw = EPW(epw_file)
    location = epw.location
    
    return {
        'latitude': float(location.latitude),
        'longitude': float(location.longitude),
        'timezone': float(location.time_zone),
        'city': str(location.city)
    }


def _calculate_sun_positions(
    epw_file: str,
    year: int,
    month: int,
    day: int
) -> List[Dict[str, Any]]:
    """
    Calculate sun positions for all 24 hours of a specific date.
    
    Args:
        epw_file: Path to EPW file
        year: Year
        month: Month (1-12)
        day: Day of month
        
    Returns:
        List of sun position dicts for each hour (0-23)
    """
    from ladybug.epw import EPW
    from ladybug.sunpath import Sunpath
    from ladybug.dt import DateTime
    import math
    
    # Load EPW and create sunpath
    epw = EPW(epw_file)
    location = epw.location
    sunpath = Sunpath.from_location(location)
    
    sun_positions = []
    
    for hour in range(24):
        dt = DateTime(month, day, hour)
        sun = sunpath.calculate_sun_from_date_time(dt)
        
        # Calculate vector pointing to sun (same as solar.py)
        if sun.is_during_day:
            alt_rad = math.radians(sun.altitude)
            azi_rad = math.radians(sun.azimuth)
            
            # Standard spherical to Cartesian conversion
            x = math.sin(azi_rad) * math.cos(alt_rad)  # East component
            y = math.cos(azi_rad) * math.cos(alt_rad)  # North component  
            z = math.sin(alt_rad)                      # Up component
            
            vector = [float(x), float(y), float(z)]
        else:
            vector = [0.0, 0.0, 0.0]
        
        sun_positions.append({
            'hour': hour,
            'altitude': float(sun.altitude),
            'azimuth': float(sun.azimuth),
            'is_up': bool(sun.is_during_day),
            'vector': vector
        })
    
    return sun_positions


def export_metadata_json(
    utci_results: Dict[str, Any],
    analysis_id: str,
    analysis_type: str,
    grid_size: float,
    date_str: str,
    hours: List[int],
    model_file: str,
    epw_file: str,
    runtime_seconds: float,
    output_path: Path,
    coordinate_system: str = "xy_ground",
    shading_indices: Optional[np.ndarray] = None
) -> None:
    """
    Export analysis metadata to JSON.
    
    Uses shared utilities from fast_utci.utci.statistics to avoid code duplication
    and ensure NaN-safe calculations for valid JSON output.
    
    Args:
        utci_results: UTCI results dictionary
        analysis_id: Analysis identifier
        analysis_type: "single_hour" or "full_day"
        grid_size: Grid spacing in meters
        date_str: Date string (YYYYMMDD)
        hours: List of hours analyzed
        model_file: Path to 3D model file
        epw_file: Path to EPW weather file
        runtime_seconds: Total computation time
        output_path: Path to output .json file
    """
    # Import shared utilities from fast_utci.utci.statistics
    from fast_utci.utci.statistics import (
        extract_all_utci_values,
        extract_positions,
        calculate_utci_statistics,
        calculate_hour_statistics
    )
    
    # Use shared utilities for data extraction (NaN-safe)
    all_utci = extract_all_utci_values(utci_results)
    all_positions = extract_positions(utci_results)
    
    # Create metadata dictionary
    metadata = {
        "analysis_id": analysis_id,
        "date": date_str,
        "grid_size": float(grid_size),
        "analysis_type": analysis_type,
        "hours": hours,
        "bounds": {
            "x_min": float(all_positions[:, 0].min()),
            "x_max": float(all_positions[:, 0].max()),
            "y_min": float(all_positions[:, 1].min()),
            "y_max": float(all_positions[:, 1].max()),
            "z": float(all_positions[0, 2])
        },
        # Use shared utility for NaN-safe statistics
        "utci_range": calculate_utci_statistics(utci_results),
        "num_positions": len(utci_results),
        "model_file": str(model_file),
        "epw_file": str(epw_file),
        "generation_date": datetime.now().isoformat(),
        "runtime_seconds": float(runtime_seconds),
        "coordinate_system": coordinate_system  # "xy_ground" or "xz_ground"
    }
    
    # Add per-hour statistics for full day analysis (NaN-safe)
    if analysis_type == "full_day":
        metadata["hour_statistics"] = calculate_hour_statistics(utci_results, hours)
    
    # Add Shading Index metadata if available
    if shading_indices is not None:
        shading_indices = np.asarray(shading_indices)
        # Filter out NaN values for statistics
        valid_indices = shading_indices[~np.isnan(shading_indices)]
        if len(valid_indices) > 0:
            metadata["has_shading_index"] = True
            metadata["shading_index_range"] = {
                "min": float(np.nanmin(shading_indices)),
                "max": float(np.nanmax(shading_indices))
            }
        else:
            metadata["has_shading_index"] = False
    else:
        metadata["has_shading_index"] = False
    
    # Extract and add location data from EPW file
    try:
        metadata["location"] = _extract_location_from_epw(epw_file)
    except Exception as e:
        print(f"[WARN] Could not extract location from EPW: {e}")
    
    # Calculate and add sun positions
    try:
        year = int(date_str[:4]) if len(date_str) >= 4 else 2025
        month = int(date_str[4:6]) if len(date_str) >= 6 else 8
        day = int(date_str[6:8]) if len(date_str) >= 8 else 15
        metadata["sun_positions"] = _calculate_sun_positions(epw_file, year, month, day)
    except Exception as e:
        print(f"[WARN] Could not calculate sun positions: {e}")
    
    # Write JSON
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"[SAVE] Metadata: {output_path.name}")


def export_utci_for_viewer(
    utci_results: Dict[str, Any],
    analysis_type: str,
    grid_size: float,
    model_file: str,
    epw_file: str,
    runtime_seconds: float,
    analysis_period: Optional[Any] = None,
    target_hours: Optional[List[int]] = None,
    output_dir: str = "data/analyses",
    coordinate_system: str = "xy_ground",
    category: Optional[str] = None,  # NEW: category for subdirectory
    shading_indices: Optional[np.ndarray] = None  # NEW: optional Shading Index array
) -> tuple[str, str]:
    """
    Export UTCI results to optimized binary format for web viewer.
    
    Args:
        utci_results: UTCI results from UTCICalculator.compute_utci()
        analysis_type: "single_hour" or "full_day"
        grid_size: Grid spacing in meters
        model_file: Path to 3D model file
        epw_file: Path to EPW weather file
        runtime_seconds: Total computation time in seconds
        analysis_period: Optional Ladybug AnalysisPeriod object
        target_hours: Optional list of target hours
        output_dir: Output directory for exported files
        
    Returns:
        Tuple of (binary_path, metadata_path)
    """
    # Determine output directory
    if category:
        output_path = Path(output_dir) / category
    else:
        output_path = Path(output_dir)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Extract date from analysis period
    date_str = extract_date_from_period(analysis_period) if analysis_period else "20250815"
    
    # Determine hours
    if target_hours is None:
        if analysis_type == "single_hour":
            # Try to extract hour from first result
            first_data = next(iter(utci_results.values()))
            if 'datetime' in first_data and first_data['datetime'] is not None:
                import pandas as pd
                hours = [pd.to_datetime(first_data['datetime'][0]).hour]
            else:
                hours = [13]  # Default
        else:
            hours = list(range(24))
    else:
        hours = target_hours
    
    # Generate analysis ID
    if category:
        # Use model name as analysis_id when in category mode
        model_name = Path(model_file).stem  # e.g., "existing_buildings_01"
        analysis_id = model_name
    else:
        # Original date-based naming
        hour_single = hours[0] if analysis_type == "single_hour" else None
        analysis_id = generate_analysis_id(date_str, grid_size, analysis_type, hour_single)
    
    # Export binary data
    binary_filename = f"{analysis_id}.bin"
    binary_path = output_path / binary_filename
    
    if analysis_type == "single_hour":
        export_binary_single_hour(utci_results, binary_path)
    else:
        export_binary_full_day(utci_results, binary_path, num_hours=len(hours), shading_indices=shading_indices)
    
    # Export metadata
    metadata_filename = f"{analysis_id}.json"
    metadata_path = output_path / metadata_filename
    
    export_metadata_json(
        utci_results=utci_results,
        analysis_id=analysis_id,
        analysis_type=analysis_type,
        grid_size=grid_size,
        date_str=date_str,
        hours=hours,
        model_file=model_file,
        epw_file=epw_file,
        runtime_seconds=runtime_seconds,
        output_path=metadata_path,
        coordinate_system=coordinate_system,
        shading_indices=shading_indices  # NEW: pass Shading Index for metadata
    )
    
    print(f"[OK] Exported analysis: {analysis_id}")
    print(f"  Binary: {binary_path}")
    print(f"  Metadata: {metadata_path}")
    
    
    return str(binary_path), str(metadata_path)


if __name__ == "__main__":
    print("Export for Viewer - Usage Example")
    print("=" * 50)
    print("Import this module in demo_workflow.py:")
    print()
    print("  from export_for_viewer import export_utci_for_viewer")
    print()
    print("  # After computing UTCI")
    print("  export_utci_for_viewer(")
    print("      utci_results=utci_results,")
    print("      analysis_type=analysis_mode,")
    print("      grid_size=GRID_SIZE,")
    print("      model_file=model_file,")
    print("      epw_file=epw_file,")
    print("      runtime_seconds=total_time,")
    print("      analysis_period=analysis_period,")
    print("      target_hours=target_hours")
    print("  )")
