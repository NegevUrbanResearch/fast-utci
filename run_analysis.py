"""
Interactive Full Day UTCI Analysis with Web Viewer Export

This script provides an interactive workflow to:
1. Load 3D model and weather data
2. Compute MRT using parallel processing
3. Compute UTCI thermal comfort for full day (24 hours)
4. Export optimized data for web viewer

Usage:
    python run_analysis.py
    
To programmatically run analysis with custom parameters, import run_analysis_core().
"""

from pathlib import Path
import time
import numpy as np
import os
import sys
from typing import Tuple, Dict, Any, Optional
import gc
from fast_utci.shared import load_config

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent / 'scripts'))

from export_for_viewer import export_utci_for_viewer

# Default grid spacing (can be modified)
DEFAULT_GRID_SIZE = 2.0  # meters

# Default analysis date
DEFAULT_MONTH = 8
DEFAULT_DAY = 15


def get_analysis_date() -> Tuple[int, int]:
    """Get analysis date input."""
    print("\n" + "=" * 60)
    print("ANALYSIS DATE SELECTION")
    print("=" * 60)
    print(f"Enter the date to analyze:")
    
    while True:
        try:
            month_input = input(f"Month (1-12, or press Enter for {DEFAULT_MONTH}): ").strip()
            month = int(month_input) if month_input else DEFAULT_MONTH
            if not 1 <= month <= 12:
                print("[ERROR] Month must be between 1 and 12.")
                continue
                
            day_input = input(f"Day (1-31, or press Enter for {DEFAULT_DAY}): ").strip()
            day = int(day_input) if day_input else DEFAULT_DAY
            if not 1 <= day <= 31:
                print("[ERROR] Day must be between 1 and 31.")
                continue
            
            return month, day
        except ValueError:
            print("[ERROR] Please enter valid numbers.")


def run_analysis_core(
    month: int = DEFAULT_MONTH,
    day: int = DEFAULT_DAY,
    grid_size: float = DEFAULT_GRID_SIZE,
    model_file: str = "data/3d_models/100_test.glb",
    epw_file: str = "data/weather/ISR_D_Beer.Sheva.401900_TMYx/ISR_D_Beer.Sheva.401900_TMYx.epw",
    embree_quality: str = "low",
    intersects_any: bool = True,
    export_csv: bool = False,
    verbose: bool = True,
    category: Optional[str] = None  # NEW: category for subdirectory organization
) -> Dict[str, Any]:
    """
    Run full day UTCI analysis with specified parameters.
    
    Args:
        month: Analysis month (1-12)
        day: Analysis day (1-31)
        grid_size: Grid spacing in meters
        model_file: Path to 3D model file
        epw_file: Path to EPW weather file
        embree_quality: Embree ray tracing quality ('low', 'medium', 'high')
        intersects_any: Use optimized intersection mode
        export_csv: Export results to CSV file (default: False)
        verbose: Print detailed progress messages
        
    Returns:
        Dictionary containing analysis results and metadata
    """
    if verbose:
        print("=" * 60)
        print("FAST-UTCI FULL DAY ANALYSIS")
        print("=" * 60)
        print(f"Date: Month {month}, Day {day}")
        print(f"Grid: {grid_size}m spacing")
        print(f"Embree: {embree_quality} quality, intersects_any={intersects_any}")
    
    
    # Check files exist
    for file_path, name in [(model_file, "3D model"), (epw_file, "EPW weather")]:
        if not Path(file_path).exists():
            raise FileNotFoundError(f"{name} file not found: {file_path}")
    
    if verbose:
        print(f"[OK] Files found: model, weather")
    
    try:
        # Load data
        if verbose:
            print("\n" + "=" * 60)
            print("STEP 1: LOADING PROJECT DATA")
            print("=" * 60)
        
        from fast_utci.shared.io import read_project_data, get_combined_mesh, get_ground_bounds
        cfg = load_config()
        t0 = time.perf_counter()
        scene, weather_df, epw_data = read_project_data(
            model_file, epw_file, verbose=False
        )
        t1 = time.perf_counter()
        
        model = get_combined_mesh(scene)
        if verbose:
            print(f"[OK] Model loaded: {len(model.vertices):,} vertices, {len(model.faces):,} faces")
            print(f"[OK] Weather loaded: {len(weather_df):,} hours")
            print(f"[TIME] Load time: {(t1-t0):.2f}s")
        
        # Compute MRT
        if verbose:
            print("\n" + "=" * 60)
            print("STEP 2: COMPUTING MRT")
            print("=" * 60)
        
        from fast_utci.mrt import (
            MRTCalculator, create_rectangular_grid, create_analysis_period
        )
        from fast_utci.mrt.grid import AnalysisGrid
        
        mrt_calc = MRTCalculator(context_meshes=[model], config=cfg.mrt)
        mrt_calc.set_location_from_epw(epw_file)
        
        # Create analysis grid with dynamic bounds
        # Strategy: Use ground bounds from scene graph or largest flat mesh
        model_bounds = get_ground_bounds(scene)
        
        # Assume standard orientation (XY is ground plane)
        axis_indices = [0, 1]  # X, Y
        vertical_axis = 2  # Z
        
        if verbose:
            print(f"[INFO] Using ground bounds for grid (XY plane)")
        
        # For backward compatibility, still check if we need to detect orientation
        # (in case get_ground_bounds returned full model bounds)
        ranges = model_bounds[1] - model_bounds[0]
        xy_area = ranges[0] * ranges[1]
        xz_area = ranges[0] * ranges[2]
        yz_area = ranges[1] * ranges[2]
        
        # Only reorient if the ground plane is clearly not XY
        if xz_area > xy_area * 1.5 or yz_area > xy_area * 1.5:
            # Find the largest plane (ground plane)
            if xy_area >= xz_area and xy_area >= yz_area:
                # Standard orientation: XY is ground, Z is vertical
                axis_indices = [0, 1]  # X, Y
                vertical_axis = 2  # Z
                if verbose:
                    print(f"[INFO] Detected XY as ground plane ({xy_area/1000:.1f}k m²)")
            elif xz_area >= xy_area and xz_area >= yz_area:
                # Z-up orientation: XZ is ground, Y is vertical
                axis_indices = [0, 2]  # X, Z
                vertical_axis = 1  # Y
                if verbose:
                    print(f"[INFO] Detected XZ as ground plane ({xz_area/1000:.1f}k m²) - Z-up model")
            else:
                # YZ is ground, X is vertical (rare)
                axis_indices = [1, 2]  # Y, Z
                vertical_axis = 0  # X
                if verbose:
                    print(f"[INFO] Detected YZ as ground plane ({yz_area/1000:.1f}k m²) - X-up model")
        
        # Apply insets to avoid mesh boundary issues
        # Grid positions too close to mesh boundaries can cause ray intersection failures
        inset_min = np.array([2.0, 1.0])  # [axis0_inset, axis1_inset] in meters
        inset_max = np.array([1.0, 1.0])  # [axis0_inset, axis1_inset] in meters
        
        # Extract bounds for the detected ground plane axes
        bounds_min = np.array([model_bounds[0][axis_indices[0]], model_bounds[0][axis_indices[1]]]) + inset_min
        bounds_max = np.array([model_bounds[1][axis_indices[0]], model_bounds[1][axis_indices[1]]]) - inset_max
        
        if verbose:
            axis_names = ['X', 'Y', 'Z']
            print(f"[INFO] Grid bounds: {axis_names[axis_indices[0]]}=[{bounds_min[0]:.2f}, {bounds_max[0]:.2f}], "
                  f"{axis_names[axis_indices[1]]}=[{bounds_min[1]:.2f}, {bounds_max[1]:.2f}]")
        
        # Create grid on the detected ground plane
        # The grid function creates an XY grid, so we create it and then transform if needed
        vertical_offset = 1.5
        if 'vertical_axis' in locals():
            z_base = model_bounds[0][vertical_axis]
        else:
            z_base = model_bounds[0][2]  # Default to Z axis for base layer
        
        # Create grid in XY space
        grid = create_rectangular_grid(
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            grid_size=grid_size,
            z_height=z_base + vertical_offset
        )
        
        # Transform grid points if not standard XY orientation
        if 'axis_indices' in locals() and axis_indices != [0, 1]:
            # Need to remap coordinates to match model orientation
            points = grid.points.copy()
            new_points = np.zeros_like(points)
            
            # Map the 2D grid coordinates to the correct 3D axes
            new_points[:, axis_indices[0]] = points[:, 0]  # First grid axis
            new_points[:, axis_indices[1]] = points[:, 1]  # Second grid axis  
            new_points[:, vertical_axis] = points[:, 2]     # Vertical axis
            
            # Update grid points
            grid = AnalysisGrid(
                points=new_points,
                normals=grid.normals,
                face_areas=grid.face_areas,
                mesh=grid.mesh,
                grid_size=grid.grid_size
            )
        
        if verbose:
            print(f"[OK] Grid created: {len(grid.points)} points at {grid_size}m spacing")
        
        # Create analysis period for full day (24 hours)
        analysis_period = create_analysis_period(
            start_month=month, start_day=day,
            end_month=month, end_day=day,
            start_hour=0, end_hour=23
        )
        target_hours = list(range(24))
        
        # Compute exposure and MRT
        if verbose:
            print(f"[INFO] Computing MRT for {len(grid.points)} positions...")
            print(f"[INFO] Boundary averaging enabled:")
            print(f"  Will calculate MRT at hour boundaries (N and N+1) and average UTCI")
            print(f"  Note: Hour 23 uses same value for both boundaries (no wrap to next day)")
        
        t2 = time.perf_counter()
        
        exposure_results = mrt_calc.compute_exposure(
            positions=grid.points,
            analysis_period=analysis_period,
            target_hours=target_hours
        )
        
        mrt_results = mrt_calc.compute_mrt(
            weather_data=epw_data,
            exposure_results=exposure_results,
            analysis_period=analysis_period,
            target_hours=target_hours
        )
        
        t3 = time.perf_counter()
        if verbose:
            print(f"[OK] MRT computed: {len(mrt_results)} positions")
            print(f"[TIME] MRT time: {(t3-t2):.2f}s")
        
        del exposure_results
        gc.collect()
        
        # Compute UTCI
        if verbose:
            print("\n" + "=" * 60)
            print("STEP 3: COMPUTING UTCI")
            print("=" * 60)
        
        from fast_utci.utci import UTCICalculator
        
        utci_calc = UTCICalculator(weather_data=weather_df, epw_object=epw_data, config=cfg.utci)
        
        t4 = time.perf_counter()
        utci_results = utci_calc.compute_utci(
            mrt_results=mrt_results,
            analysis_period=analysis_period,
            target_hours=target_hours,
            show_progress=cfg.utci.show_progress
        )
        t5 = time.perf_counter()
        
        if verbose:
            print(f"[OK] UTCI computed")
            print(f"[TIME] UTCI time: {(t5-t4):.2f}s")
        
        # Calculate statistics using shared utilities (NaN-safe)
        from fast_utci.utci.statistics import calculate_utci_statistics
        utci_stats = calculate_utci_statistics(utci_results)
        utci_min, utci_max, utci_mean = utci_stats['min'], utci_stats['max'], utci_stats['mean']
        
        # Export results
        if verbose:
            print("\n" + "=" * 60)
            print("STEP 4: EXPORTING RESULTS")
            print("=" * 60)
        
        # Export CSV (optional)
        csv_filename = None
        if export_csv:
            csv_filename = f"utci_results_grid_{grid_size:.0f}m_fullday.csv"
            utci_calc.to_csv(
                utci_results=utci_results,
                csv_path=csv_filename,
                include_weather=True,
                include_comfort_categories=True
            )
            if verbose:
                print(f"[OK] CSV exported: {csv_filename}")
        
        # Export for web viewer
        total_time = t5 - t0
        
        # Determine coordinate system for viewer
        # If using XZ plane, viewer needs to know to rotate the model
        if 'axis_indices' in locals() and axis_indices == [0, 2]:
            coordinate_system = "xz_ground"  # Y-up: XZ is ground, Y is vertical
        else:
            coordinate_system = "xy_ground"  # Z-up: XY is ground, Z is vertical (standard)
        
        binary_path, metadata_path = export_utci_for_viewer(
            utci_results=utci_results,
            analysis_type="full_day",
            grid_size=grid_size,
            model_file=model_file,
            epw_file=epw_file,
            runtime_seconds=total_time,
            analysis_period=analysis_period,
            target_hours=target_hours,
            coordinate_system=coordinate_system,
            category=category  # NEW: pass category for subdirectory organization
        )
        
        # Summary
        if verbose:
            print("\n" + "=" * 60)
            print("ANALYSIS COMPLETE")
            print("=" * 60)
            print(f"Positions analyzed: {len(utci_results)}")
            print(f"UTCI range: {utci_min:.1f} to {utci_max:.1f} C (mean: {utci_mean:.1f} C)")
            print(f"Total runtime: {total_time:.1f}s")
            print(f"\nOutput files:")
            if csv_filename:
                print(f"  - CSV: {csv_filename}")
            print(f"  - Web viewer data: {Path(binary_path).name}")
            print(f"  - Web viewer metadata: {Path(metadata_path).name}")
            print(f"\nTo visualize:")
            print(f"  1. Start HTTP server: python -m http.server 8000")
            print(f"  2. Open: http://localhost:8000/viewer/")
        
        return {
            "utci_results": utci_results,
            "csv_path": csv_filename,
            "binary_path": binary_path,
            "metadata_path": metadata_path,
            "utci_min": utci_min,
            "utci_max": utci_max,
            "utci_mean": utci_mean,
            "total_time": total_time,
            "num_positions": len(utci_results),
            "grid_size": grid_size,
            "month": month,
            "day": day
        }
        
    except Exception as e:
        if verbose:
            print(f"\n[ERROR] Error in workflow: {e}")
            import traceback
            traceback.print_exc()
        raise


def main() -> int:
    """Interactive CLI entry point for full day UTCI analysis."""
    
    print("=" * 60)
    print("FAST-UTCI FULL DAY ANALYSIS")
    print("=" * 60)
    print("Mode: Full day analysis (24 hours)")
    
    # Get analysis date
    analysis_month, analysis_day = get_analysis_date()
    print(f"\n[OK] Analysis date: Month {analysis_month}, Day {analysis_day}")
    
    try:
        # Run analysis with default settings
        results = run_analysis_core(
            month=analysis_month,
            day=analysis_day,
            verbose=True
        )
        return 0
        
    except Exception as e:
        print(f"\n[ERROR] Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
