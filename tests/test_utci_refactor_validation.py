"""
Integration test to validate UTCI refactoring produces identical results.

Compares new refactored implementation against baseline reference data
generated from the original implementation.
"""

import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fast_utci.utci import UTCICalculator
from fast_utci.mrt import MRTCalculator, create_rectangular_grid, create_analysis_period
from fast_utci.model_reader import read_project_data_enhanced


def load_baseline_reference():
    """Load baseline reference data."""
    baseline_file = Path(__file__).parent / "fixtures" / "utci_reference_results.npz"
    if not baseline_file.exists():
        raise FileNotFoundError(f"Baseline reference not found: {baseline_file}")
    
    data = np.load(baseline_file, allow_pickle=True)
    return data


def run_refactored_analysis():
    """Run the same analysis using refactored code."""
    # Use same parameters as baseline generation
    model_file = "data/3d_models/100.gltf"
    epw_file = "data/weather/ISR_Beer.Sheva.401900_MSI.epw"
    
    # Load data
    enhanced_model, weather_df, epw_data = read_project_data_enhanced(
        model_file, epw_file, verbose=False
    )
    model = enhanced_model.get_combined_mesh()
    
    # Create same test grid
    bounds_min = np.array([-2468.81, -618.8652])
    bounds_max = np.array([-2400.0, -550.0])
    
    grid = create_rectangular_grid(
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        grid_size=10.0,
        z_height=1.5
    )
    
    # Create analysis period (full day)
    analysis_period = create_analysis_period(
        start_month=8, start_day=15,
        end_month=8, end_day=15,
        start_hour=0, end_hour=23
    )
    
    # Compute MRT
    mrt_calc = MRTCalculator(context_meshes=[model])
    mrt_calc.set_location_from_epw(epw_file)
    
    exposure_results = mrt_calc.compute_exposure(
        positions=grid.points,
        analysis_period=analysis_period,
        target_hours=None
    )
    
    mrt_results = mrt_calc.compute_mrt(
        epw_data=epw_data,
        exposure_results=exposure_results,
        analysis_period=analysis_period,
        target_hours=None
    )
    
    # Compute UTCI using NEW refactored implementation
    utci_calc = UTCICalculator(weather_data=weather_df, epw_object=epw_data)
    
    utci_results = utci_calc.compute_utci(
        mrt_results=mrt_results,
        analysis_period=analysis_period,
        target_hours=None,
        show_progress=False
    )
    
    return utci_results, grid.points


def compare_results(baseline, refactored_results, tolerance=1e-10):
    """
    Compare refactored results against baseline.
    
    Args:
        baseline: Loaded baseline data
        refactored_results: Results from refactored code
        tolerance: Numerical tolerance for floating point comparison
        
    Returns:
        Tuple of (passed, errors)
    """
    errors = []
    num_positions = int(baseline['num_positions'])
    
    # Check number of positions matches
    if len(refactored_results) != num_positions:
        errors.append(f"Position count mismatch: expected {num_positions}, got {len(refactored_results)}")
        return False, errors
    
    # Compare each position
    max_utci_diff = 0.0
    max_mrt_diff = 0.0
    
    for pos_key in refactored_results.keys():
        # Get baseline data for this position
        baseline_utci = baseline[f'{pos_key}_utci']
        baseline_mrt0 = baseline[f'{pos_key}_mrt0']
        baseline_position = baseline[f'{pos_key}_position']
        
        # Get refactored data
        ref_data = refactored_results[pos_key]
        ref_utci = ref_data['utci']
        ref_mrt0 = ref_data['mrt0']
        ref_position = ref_data['position']
        
        # Compare positions
        pos_diff = np.abs(np.array(baseline_position) - np.array(ref_position))
        if np.max(pos_diff) > tolerance:
            errors.append(f"{pos_key}: Position mismatch (max diff: {np.max(pos_diff)})")
        
        # Compare UTCI values
        utci_diff = np.abs(baseline_utci - ref_utci)
        max_diff_this_pos = np.max(utci_diff)
        max_utci_diff = max(max_utci_diff, max_diff_this_pos)
        
        if max_diff_this_pos > tolerance:
            errors.append(f"{pos_key}: UTCI mismatch (max diff: {max_diff_this_pos})")
        
        # Compare MRT values
        mrt_diff = np.abs(baseline_mrt0 - ref_mrt0)
        max_mrt_diff_this_pos = np.max(mrt_diff)
        max_mrt_diff = max(max_mrt_diff, max_mrt_diff_this_pos)
        
        if max_mrt_diff_this_pos > tolerance:
            errors.append(f"{pos_key}: MRT mismatch (max diff: {max_mrt_diff_this_pos})")
    
    # Summary
    print(f"\nComparison Summary:")
    print(f"  Max UTCI difference: {max_utci_diff}")
    print(f"  Max MRT difference: {max_mrt_diff}")
    print(f"  Tolerance: {tolerance}")
    
    passed = len(errors) == 0
    return passed, errors


def main():
    """Run validation test."""
    print("="*60)
    print("UTCI REFACTORING VALIDATION TEST")
    print("="*60)
    
    print("\n[1/3] Loading baseline reference...")
    try:
        baseline = load_baseline_reference()
        print(f"  [OK] Loaded baseline with {int(baseline['num_positions'])} positions")
    except Exception as e:
        print(f"  [FAIL] Failed to load baseline: {e}")
        return 1
    
    print("\n[2/3] Running refactored analysis...")
    try:
        refactored_results, grid_points = run_refactored_analysis()
        print(f"  [OK] Computed {len(refactored_results)} positions")
    except Exception as e:
        print(f"  [FAIL] Failed to run refactored analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n[3/3] Comparing results...")
    try:
        passed, errors = compare_results(baseline, refactored_results)
        
        if passed:
            print("\n" + "="*60)
            print("[SUCCESS] Refactored implementation matches baseline!")
            print("="*60)
            return 0
        else:
            print("\n" + "="*60)
            print("[FAILURE] Refactored implementation differs from baseline")
            print("="*60)
            print(f"\nFound {len(errors)} error(s):")
            for error in errors[:10]:  # Show first 10 errors
                print(f"  - {error}")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more")
            return 1
    except Exception as e:
        print(f"  [FAIL] Comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

