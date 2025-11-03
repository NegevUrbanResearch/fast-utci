"""
Generate baseline UTCI results from current implementation for validation.

This script captures the current behavior before refactoring to ensure
the refactored code produces identical results.
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fast_utci.utci import UTCICalculator
from fast_utci.mrt import MRTCalculator, create_rectangular_grid, create_analysis_period
from fast_utci.model_reader import read_project_data, get_combined_mesh
from fast_utci.shared import load_config

def generate_baseline():
    """Generate baseline UTCI results for testing."""
    print("Generating baseline UTCI results from current implementation...")
    
    # Use validation data
    model_file = "data/3d_models/100.gltf"
    epw_file = "data/weather/ISR_Beer.Sheva.401900_MSI.epw"
    
    # Load data
    print("Loading model and weather data...")
    scene, weather_df, epw_data = read_project_data(
        model_file, epw_file, verbose=False
    )
    model = get_combined_mesh(scene)
    
    # Create small test grid (for speed)
    print("Creating test grid...")
    bounds_min = np.array([-2468.81, -618.8652])
    bounds_max = np.array([-2400.0, -550.0])  # Small area for testing
    
    grid = create_rectangular_grid(
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        grid_size=10.0,  # Coarse grid for speed
        z_height=1.5
    )
    
    print(f"Grid size: {len(grid.points)} points")
    
    # Create analysis period (August 15, full day)
    # Note: Must use full day (0-23) for HourlyContinuousCollection compatibility
    analysis_period = create_analysis_period(
        start_month=8, start_day=15,
        end_month=8, end_day=15,
        start_hour=0, end_hour=23
    )
    target_hours = None  # Compute all hours
    
    # Load config from TOML
    cfg = load_config()
    
    # Compute MRT
    print("Computing MRT...")
    mrt_calc = MRTCalculator(context_meshes=[model], config=cfg.mrt)
    mrt_calc.set_location_from_epw(epw_file)
    
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
    
    # Compute UTCI using CURRENT implementation
    print("Computing UTCI...")
    utci_calc = UTCICalculator(weather_data=weather_df, epw_object=epw_data, config=cfg.utci)
    
    utci_results = utci_calc.compute_utci(
        mrt_results=mrt_results,
        analysis_period=analysis_period,
        target_hours=target_hours,
        show_progress=False
    )
    
    # Save baseline results
    print("Saving baseline results...")
    baseline_data = {
        'grid_points': grid.points,
        'num_positions': len(utci_results)
    }
    
    # Save UTCI values for each position
    for pos_key, data in utci_results.items():
        baseline_data[f'{pos_key}_utci'] = data['utci']
        baseline_data[f'{pos_key}_mrt0'] = data.get('mrt0', data.get('mrt'))
        baseline_data[f'{pos_key}_mrt1'] = data.get('mrt1', data.get('mrt0', data.get('mrt')))
        baseline_data[f'{pos_key}_position'] = data['position']
        if 'air_temp' in data:
            baseline_data[f'{pos_key}_air_temp'] = data['air_temp']
        if 'wind_speed' in data:
            baseline_data[f'{pos_key}_wind_speed'] = data['wind_speed']
        if 'relative_humidity' in data:
            baseline_data[f'{pos_key}_rh'] = data['relative_humidity']
    
    output_file = Path(__file__).parent / "utci_reference_results.npz"
    np.savez_compressed(output_file, **baseline_data)
    
    print(f"[OK] Baseline saved to: {output_file}")
    print(f"  Positions: {len(utci_results)}")
    
    # Determine number of hours from first result
    first_key = list(utci_results.keys())[0]
    num_hours = len(utci_results[first_key]['utci'])
    print(f"  Hours: {num_hours}")
    
    # Print sample UTCI values for verification
    first_key = list(utci_results.keys())[0]
    first_utci = utci_results[first_key]['utci']
    print(f"  Sample UTCI (first position): {first_utci}")
    print(f"  UTCI range: {np.min([d['utci'] for d in utci_results.values()]):.2f} to {np.max([d['utci'] for d in utci_results.values()]):.2f}")

if __name__ == "__main__":
    generate_baseline()

