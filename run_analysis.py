"""
Interactive UTCI Analysis with Web Viewer Export

This script provides an interactive workflow to:
1. Load 3D model and weather data
2. Compute MRT using parallel processing
3. Compute UTCI thermal comfort
4. Generate Plotly visualization (legacy)
5. Export optimized data for web viewer

Usage:
    python run_analysis.py
"""

from pathlib import Path
import time
import numpy as np
import os
import sys
from typing import Tuple, List, Optional
import pandas as pd
import psutil
import gc

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent / 'scripts'))

from fast_utci.colors import create_ladybug_utci_colorscale
from export_for_viewer import export_utci_for_viewer

# Default grid spacing (can be modified)
DEFAULT_GRID_SIZE = 2.0  # meters


def get_analysis_mode():
    """Get user choice for analysis type."""
    print("\n" + "="*60)
    print("ANALYSIS MODE SELECTION")
    print("="*60)
    print("1. Single Hour Analysis")
    print("2. Full Day Analysis (24 hours)")
    print("="*60)
    
    while True:
        choice = input("Enter your choice (1 or 2): ").strip()
        if choice == "1":
            return "single_hour"
        elif choice == "2":
            return "full_day"
        else:
            print("[ERROR] Invalid choice. Please enter 1 or 2.")


def get_single_hour():
    """Get hour input for single hour analysis."""
    print("\n" + "="*60)
    print("SINGLE HOUR SELECTION")
    print("="*60)
    print("Enter the hour to analyze (0-23):")
    print("  0 = Midnight (00:00-01:00)")
    print("  12 = Noon (12:00-13:00)")
    print("  13 = 1 PM (13:00-14:00) - Default")
    print("  23 = 11 PM (23:00-24:00)")
    
    while True:
        try:
            hour_input = input("Hour (0-23, or press Enter for 13): ").strip()
            if hour_input == "":
                return 13
            hour = int(hour_input)
            if 0 <= hour <= 23:
                return hour
            else:
                print("[ERROR] Hour must be between 0 and 23.")
        except ValueError:
            print("[ERROR] Please enter a valid number.")


def main():
    """Run the UTCI analysis workflow."""
    
    print("="*60)
    print("FAST-UTCI ANALYSIS WITH WEB VIEWER EXPORT")
    print("="*60)
    
    # Get analysis mode
    analysis_mode = get_analysis_mode()
    
    if analysis_mode == "single_hour":
        target_hour = get_single_hour()
        print(f"\n[OK] Selected: Single hour analysis for hour {target_hour:02d}:00")
    else:
        print("\n[OK] Selected: Full day analysis (24 hours)")
        target_hour = None
    
    # Configure performance optimizations
    if analysis_mode == "full_day":
        os.environ.setdefault("FAST_UTCI_VECTORIZED_SOLAR", "1")
        os.environ.setdefault("FAST_UTCI_VECTORIZED_UTCI", "1")
        os.environ.setdefault("FAST_UTCI_INTERSECTOR", "embree")
        os.environ.setdefault("FAST_UTCI_EMBREE_QUALITY", "low")
        os.environ.setdefault("FAST_UTCI_EMBREE_BUILD_BVH", "true")
        os.environ.setdefault("FAST_UTCI_INTERSECTS_ANY", "1")
        os.environ.setdefault("FAST_UTCI_BATCH_POSITIONS", "1")
        print("[PERF] Optimizations enabled for full day analysis")
    else:
        os.environ.setdefault("FAST_UTCI_INTERSECTOR", "embree")
        os.environ.setdefault("FAST_UTCI_EMBREE_QUALITY", "low")
        os.environ.setdefault("FAST_UTCI_EMBREE_BUILD_BVH", "true")
        os.environ.setdefault("FAST_UTCI_INTERSECTS_ANY", "1")
        os.environ.setdefault("FAST_UTCI_BATCH_POSITIONS", "1")
        print("[PERF] Optimizations enabled for single hour analysis")
    
    # File paths
    model_file = "data/3d_models/100.gltf"
    epw_file = "data/weather/ISR_Beer.Sheva.401900_MSI.epw"
    validation_csv = "data/validation/15th_Aug_MRT.csv"
    
    # Check files exist
    for file_path, name in [(model_file, "3D model"), (epw_file, "EPW weather")]:
        if not Path(file_path).exists():
            print(f"[ERROR] {name} file not found: {file_path}")
            return 1
    
    print(f"[OK] Files found: model, weather")
    
    try:
        # Load data
        print("\n" + "="*60)
        print("STEP 1: LOADING PROJECT DATA")
        print("="*60)
        
        from fast_utci.model_reader import read_project_data_enhanced
        t0 = time.perf_counter()
        enhanced_model, weather_df, epw_data = read_project_data_enhanced(
            model_file, epw_file, verbose=False
        )
        t1 = time.perf_counter()
        
        model = enhanced_model.get_combined_mesh()
        print(f"[OK] Model loaded: {len(model.vertices):,} vertices, {len(model.faces):,} faces")
        print(f"[OK] Weather loaded: {len(weather_df):,} hours")
        print(f"[TIME] Load time: {(t1-t0):.2f}s")
        
        # Compute MRT
        print("\n" + "="*60)
        print("STEP 2: COMPUTING MRT")
        print("="*60)
        
        from fast_utci.mrt import (
            MRTCalculator, create_rectangular_grid, create_analysis_period
        )
        
        mrt_calc = MRTCalculator(context_meshes=[model])
        mrt_calc.set_location_from_epw(epw_file)
        
        # Create analysis grid
        grid_size = DEFAULT_GRID_SIZE
        bounds_min = np.array([-2470.81, -619.8652])
        bounds_max = np.array([-1479.529, -196.4804])
        
        grid = create_rectangular_grid(
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            grid_size=grid_size,
            z_height=1.5
        )
        
        print(f"[OK] Grid created: {len(grid.points)} points at {grid_size}m spacing")
        
        # Create analysis period
        if analysis_mode == "single_hour":
            analysis_period = create_analysis_period(
                start_month=8, start_day=15,
                end_month=8, end_day=15,
                start_hour=target_hour, end_hour=target_hour
            )
            target_hours = [target_hour]
        else:
            analysis_period = create_analysis_period(
                start_month=8, start_day=15,
                end_month=8, end_day=15,
                start_hour=0, end_hour=23
            )
            target_hours = list(range(24))
        
        # Compute exposure and MRT
        print(f"[INFO] Computing MRT for {len(grid.points)} positions...")
        t2 = time.perf_counter()
        
        exposure_results = mrt_calc.compute_exposure(
            positions=grid.points,
            analysis_period=analysis_period,
            target_hours=target_hours
        )
        
        mrt_results = mrt_calc.compute_mrt(
            epw_data=epw_data,
            exposure_results=exposure_results,
            analysis_period=analysis_period,
            target_hours=target_hours
        )
        
        t3 = time.perf_counter()
        print(f"[OK] MRT computed: {len(mrt_results)} positions")
        print(f"[TIME] MRT time: {(t3-t2):.2f}s")
        
        del exposure_results
        gc.collect()
        
        # Compute UTCI
        print("\n" + "="*60)
        print("STEP 3: COMPUTING UTCI")
        print("="*60)
        
        from fast_utci.utci_calculator import UTCICalculator
        
        utci_calc = UTCICalculator(weather_data=weather_df, epw_object=epw_data)
        
        t4 = time.perf_counter()
        utci_results = utci_calc.compute_utci(
            mrt_results=mrt_results,
            analysis_period=analysis_period,
            target_hours=target_hours,
            show_progress=True
        )
        t5 = time.perf_counter()
        
        print(f"[OK] UTCI computed")
        print(f"[TIME] UTCI time: {(t5-t4):.2f}s")
        
        # Calculate statistics
        all_utci = []
        for pos_key, data in utci_results.items():
            if isinstance(data.get('utci'), (list, np.ndarray)):
                all_utci.extend(data['utci'])
            else:
                all_utci.append(data['utci'])
        
        all_utci = np.array(all_utci)
        utci_min, utci_max, utci_mean = np.min(all_utci), np.max(all_utci), np.mean(all_utci)
        
        # Export results
        print("\n" + "="*60)
        print("STEP 4: EXPORTING RESULTS")
        print("="*60)
        
        # Export CSV
        if analysis_mode == "single_hour":
            csv_filename = f"utci_results_grid_{grid_size:.0f}m_hour_{target_hour:02d}.csv"
        else:
            csv_filename = f"utci_results_grid_{grid_size:.0f}m_fullday.csv"
        
        utci_calc.to_csv(
            utci_results=utci_results,
            csv_path=csv_filename,
            include_weather=True,
            include_comfort_categories=True
        )
        print(f"[OK] CSV exported: {csv_filename}")
        
        # Export for web viewer
        total_time = t5 - t0
        binary_path, metadata_path = export_utci_for_viewer(
            utci_results=utci_results,
            analysis_type=analysis_mode,
            grid_size=grid_size,
            model_file=model_file,
            epw_file=epw_file,
            runtime_seconds=total_time,
            analysis_period=analysis_period,
            target_hours=target_hours
        )
        
        # Summary
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print(f"Positions analyzed: {len(utci_results)}")
        print(f"UTCI range: {utci_min:.1f} to {utci_max:.1f} C (mean: {utci_mean:.1f} C)")
        print(f"Total runtime: {total_time:.1f}s")
        print(f"\nOutput files:")
        print(f"  - CSV: {csv_filename}")
        print(f"  - Web viewer data: {Path(binary_path).name}")
        print(f"  - Web viewer metadata: {Path(metadata_path).name}")
        print(f"\nTo visualize:")
        print(f"  1. Start HTTP server: python -m http.server 8000")
        print(f"  2. Open: http://localhost:8000/viewer/")
        
        return 0
        
    except Exception as e:
        print(f"\n[ERROR] Error in workflow: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
