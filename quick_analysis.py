"""
Quick Automated UTCI Analysis

Runs a fast UTCI analysis with default settings:
- Single hour at 13:00 (1 PM)
- 10m grid spacing
- Original model (no simplification)
- Exports data for web viewer

Usage:
    python quick_analysis.py
"""

import sys
import os
import time
from pathlib import Path
import numpy as np
import gc

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

# Default settings
ANALYSIS_MODE = "single_hour"
TARGET_HOUR = 14
GRID_SIZE = 2.0


def main():
    """Run quick UTCI analysis with defaults."""

    print("=" * 60)
    print("QUICK UTCI ANALYSIS")
    print("=" * 60)
    print(f"Mode: Single hour at {TARGET_HOUR}:00")
    print(f"Grid: {GRID_SIZE}m spacing")
    print("=" * 60)

    # Performance optimizations
    os.environ.setdefault("FAST_UTCI_INTERSECTOR", "embree")
    os.environ.setdefault("FAST_UTCI_EMBREE_QUALITY", "medium")
    os.environ.setdefault("FAST_UTCI_EMBREE_BUILD_BVH", "true")
    os.environ.setdefault("FAST_UTCI_INTERSECTS_ANY", "0")
    os.environ.setdefault("FAST_UTCI_BATCH_POSITIONS", "1")

    # File paths
    model_file = "data/3d_models/100.gltf"
    epw_file = "data/weather/ISR_Beer.Sheva.401900_MSI.epw"

    # Check files
    for file_path in [model_file, epw_file]:
        if not Path(file_path).exists():
            print(f"[ERROR] File not found: {file_path}")
            return 1

    try:
        from fast_utci.model_reader import read_project_data_enhanced
        from fast_utci.mrt import MRTCalculator, create_rectangular_grid, create_analysis_period
        from fast_utci.utci_calculator import UTCICalculator
        from export_for_viewer import export_utci_for_viewer

        t_start = time.perf_counter()

        # Load data
        print("\n[1/4] Loading model and weather...")
        enhanced_model, weather_df, epw_data = read_project_data_enhanced(
            model_file, epw_file, verbose=False
        )
        model = enhanced_model.get_combined_mesh()
        print(f"      Loaded: {len(model.vertices):,} vertices, {len(weather_df):,} hours")

        # Setup MRT calculator
        print("[2/4] Computing MRT...")
        mrt_calc = MRTCalculator(context_meshes=[model])
        mrt_calc.set_location_from_epw(epw_file)

        # Create grid
        # Trimmed to building coverage (buildings end at x=-1487, trim to x=-1490 with 3m safety margin)
        bounds_min = np.array([-2470.81, -619.8652])
        bounds_max = np.array(
            [-1490.0, -196.4804]
        )  # Trimmed to avoid areas without building coverage
        grid = create_rectangular_grid(
            bounds_min=bounds_min, bounds_max=bounds_max, grid_size=GRID_SIZE, z_height=1.5
        )
        print(f"      Grid: {len(grid.points)} points")

        # Analysis period
        analysis_period = create_analysis_period(
            start_month=8,
            start_day=15,
            end_month=8,
            end_day=15,
            start_hour=TARGET_HOUR,
            end_hour=TARGET_HOUR,
        )
        target_hours = [TARGET_HOUR]

        # Compute MRT
        exposure_results = mrt_calc.compute_exposure(
            positions=grid.points, analysis_period=analysis_period, target_hours=target_hours
        )

        mrt_results = mrt_calc.compute_mrt(
            epw_data=epw_data,
            exposure_results=exposure_results,
            analysis_period=analysis_period,
            target_hours=target_hours,
        )

        del exposure_results
        gc.collect()

        print(f"      MRT computed: {len(mrt_results)} positions")

        # Compute UTCI
        print("[3/4] Computing UTCI...")
        utci_calc = UTCICalculator(weather_data=weather_df, epw_object=epw_data)

        utci_results = utci_calc.compute_utci(
            mrt_results=mrt_results,
            analysis_period=analysis_period,
            target_hours=target_hours,
            show_progress=False,
        )

        # Statistics
        all_utci = []
        for data in utci_results.values():
            if isinstance(data.get("utci"), (list, np.ndarray)):
                all_utci.extend(data["utci"])
            else:
                all_utci.append(data["utci"])

        all_utci = np.array(all_utci)
        utci_min, utci_max, utci_mean = np.min(all_utci), np.max(all_utci), np.mean(all_utci)

        print(f"      UTCI range: {utci_min:.1f} to {utci_max:.1f} C (mean: {utci_mean:.1f} C)")

        # Export
        print("[4/4] Exporting results...")

        # CSV
        csv_file = f"quick_analysis_hour_{TARGET_HOUR:02d}.csv"
        utci_calc.to_csv(
            utci_results=utci_results,
            csv_path=csv_file,
            include_weather=True,
            include_comfort_categories=True,
        )

        # Web viewer
        total_time = time.perf_counter() - t_start
        binary_path, metadata_path = export_utci_for_viewer(
            utci_results=utci_results,
            analysis_type=ANALYSIS_MODE,
            grid_size=GRID_SIZE,
            model_file=model_file,
            epw_file=epw_file,
            runtime_seconds=total_time,
            analysis_period=analysis_period,
            target_hours=target_hours,
        )

        # Summary
        print("\n" + "=" * 60)
        print("COMPLETE")
        print("=" * 60)
        print(f"Runtime: {total_time:.1f}s")
        print(f"Positions: {len(utci_results)}")
        print(f"UTCI: {utci_min:.1f} to {utci_max:.1f} C")
        print(f"\nFiles created:")
        print(f"  - {csv_file}")
        print(f"  - {Path(binary_path).name}")
        print(f"  - {Path(metadata_path).name}")
        print(f"\nView results:")
        print(f"  python -m http.server 8000")
        print(f"  http://localhost:8000/viewer/")
        print("=" * 60)

        return 0

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
