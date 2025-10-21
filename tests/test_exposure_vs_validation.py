"""
ABOUTME: Test our exposure calculations against Grasshopper validation data
Uses our actual calculation pipeline for efficiency
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from ladybug.epw import EPW

from fast_utci.mrt.exposure import compute_exposure_batch
from fast_utci.mrt.mesh import load_context_meshes
from fast_utci.mrt.solar import get_sun_vectors


def test_exposure_against_grasshopper_validation():
    """
    Compare our sky exposure and solar exposure calculations with Grasshopper validation data.
    This tests the fundamental exposure calculations that affect MRT and UTCI.
    """
    
    # Load validation data
    val_file = Path("data/validation/15_Aug_10_11.csv")
    val_df = pd.read_csv(val_file)
    
    print(f"\n{'='*80}")
    print(f"EXPOSURE VALIDATION TEST")
    print(f"{'='*80}")
    print(f"Validation points: {len(val_df)}")
    
    # Extract unique positions (validation has same positions for both hours)
    positions = val_df[['x', 'y', 'z']].values
    unique_positions = np.unique(positions, axis=0)
    print(f"Unique positions: {len(unique_positions)}")
    
    # Load 3D model
    model_file = Path("data/3d_models/100_test.glb")
    mesh_context = load_context_meshes([str(model_file)])
    print(f"✓ Loaded 3D model ({len(mesh_context.mesh.faces)} faces)")
    
    # Load EPW for solar calculations
    epw_file = Path("data/weather/ISR_D_Beer.Sheva.401900_TMYx/ISR_D_Beer.Sheva.401900_TMYx.epw")
    epw = EPW(str(epw_file))
    
    # Get sun data for August 15, hours 10-11
    print(f"\nCalculating sun vectors...")
    sun_data = get_sun_vectors(
        epw.location,
        analysis_period=(8, 15, 8, 15)  # August 15 only
    )
    
    # Filter to hours 10-11
    from fast_utci.mrt.solar import filter_hours_by_local_time
    sun_data = filter_hours_by_local_time(sun_data, [10, 11])
    
    print(f"✓ Sun data: {len(sun_data.solar_times)} hours")
    
    # Calculate exposure for all validation positions
    print(f"\nCalculating exposure for {len(unique_positions)} positions...")
    exposure_results = compute_exposure_batch(
        unique_positions,
        sun_data,
        mesh_context,
        show_progress=True
    )
    
    print(f"✓ Calculated exposure")
    
    # Extract results
    sky_exposure = np.array([er.sky_exposure for er in exposure_results])
    solar_exposure_by_hour = np.array([er.fract_body_exp for er in exposure_results])
    
    print(f"\nSky exposure:")
    print(f"  Range: {sky_exposure.min():.3f} to {sky_exposure.max():.3f}")
    print(f"  Mean: {sky_exposure.mean():.3f}")
    
    print(f"\nSolar exposure by hour:")
    for hour_idx, hour in enumerate([10, 11]):
        solar_exp = solar_exposure_by_hour[:, hour_idx]
        print(f"  Hour {hour}:")
        print(f"    Range: {solar_exp.min():.3f} to {solar_exp.max():.3f}")
        print(f"    Mean: {solar_exp.mean():.3f}")
        print(f"    Fully exposed (>0.9): {np.sum(solar_exp > 0.9)} points")
        print(f"    Fully shaded (<0.1): {np.sum(solar_exp < 0.1)} points")
    
    # Now compare with validation data
    print(f"\n{'='*80}")
    print(f"COMPARISON WITH GRASSHOPPER")
    print(f"{'='*80}")
    
    # Create lookup for our results by position
    position_to_idx = {}
    for i, pos in enumerate(unique_positions):
        key = (round(pos[0], 2), round(pos[1], 2), round(pos[2], 2))
        position_to_idx[key] = i
    
    # Match validation data with our calculations
    matches = []
    for _, row in val_df.iterrows():
        key = (round(row['x'], 2), round(row['y'], 2), round(row['z'], 2))
        
        if key in position_to_idx:
            idx = position_to_idx[key]
            
            matches.append({
                'val_human_to_sky': row['human_to_sky'],
                'our_sky_exposure': sky_exposure[idx],
                'our_solar_exp_h10': solar_exposure_by_hour[idx, 0],
                'our_solar_exp_h11': solar_exposure_by_hour[idx, 1],
                'val_mrt0': row['mrt0'],
                'val_mrt1': row['mrt1'],
                'val_utci0': row['utci 0'],
                'val_utci1': row['utci 1'],
            })
    
    matches_df = pd.DataFrame(matches)
    print(f"Matched {len(matches_df)} points")
    
    # Analyze by Grasshopper shading category
    print(f"\n{'='*80}")
    print(f"ANALYSIS BY GRASSHOPPER SHADING CATEGORY")
    print(f"{'='*80}")
    
    for val_category in sorted(matches_df['val_human_to_sky'].unique()):
        subset = matches_df[matches_df['val_human_to_sky'] == val_category]
        
        category_name = {
            0.0: "FULLY SHADED (from sun)",
            0.5: "PARTIALLY SHADED (from sun)", 
            1.0: "FULLY EXPOSED (to sun)"
        }.get(val_category, f"UNKNOWN ({val_category})")
        
        print(f"\n{category_name}")
        print(f"  Points: {len(subset)}")
        print(f"\n  OUR SKY EXPOSURE (sky view factor):")
        print(f"    Mean: {subset['our_sky_exposure'].mean():.3f} ± {subset['our_sky_exposure'].std():.3f}")
        print(f"    Range: [{subset['our_sky_exposure'].min():.3f}, {subset['our_sky_exposure'].max():.3f}]")
        
        print(f"\n  OUR SOLAR EXPOSURE Hour 10:")
        print(f"    Mean: {subset['our_solar_exp_h10'].mean():.3f} ± {subset['our_solar_exp_h10'].std():.3f}")
        print(f"    Range: [{subset['our_solar_exp_h10'].min():.3f}, {subset['our_solar_exp_h10'].max():.3f}]")
        
        print(f"\n  OUR SOLAR EXPOSURE Hour 11:")
        print(f"    Mean: {subset['our_solar_exp_h11'].mean():.3f} ± {subset['our_solar_exp_h11'].std():.3f}")
        print(f"    Range: [{subset['our_solar_exp_h11'].min():.3f}, {subset['our_solar_exp_h11'].max():.3f}]")
        
        print(f"\n  VALIDATION MRT:")
        print(f"    Hour 10 mean: {subset['val_mrt0'].mean():.1f}°C")
        print(f"    Hour 11 mean: {subset['val_mrt1'].mean():.1f}°C")
        
        print(f"\n  VALIDATION UTCI:")
        print(f"    Hour 10 mean: {subset['val_utci0'].mean():.1f}°C")
        print(f"    Hour 11 mean: {subset['val_utci1'].mean():.1f}°C")
    
    # Overall correlation analysis
    print(f"\n{'='*80}")
    print(f"CORRELATION ANALYSIS")
    print(f"{'='*80}")
    
    # Sky exposure vs human_to_sky (should correlate with shading)
    sky_corr = np.corrcoef(matches_df['our_sky_exposure'], matches_df['val_human_to_sky'])[0, 1]
    print(f"\nSky exposure vs Grasshopper human_to_sky: {sky_corr:.3f}")
    print(f"  (Note: These measure different things - sky view vs solar exposure)")
    
    # Solar exposure should correlate perfectly with human_to_sky
    solar_h10_corr = np.corrcoef(matches_df['our_solar_exp_h10'], matches_df['val_human_to_sky'])[0, 1]
    solar_h11_corr = np.corrcoef(matches_df['our_solar_exp_h11'], matches_df['val_human_to_sky'])[0, 1]
    
    print(f"\nOur solar exposure (H10) vs Grasshopper human_to_sky: {solar_h10_corr:.3f}")
    print(f"Our solar exposure (H11) vs Grasshopper human_to_sky: {solar_h11_corr:.3f}")
    
    # Save detailed results
    output_file = Path("data/validation/exposure_comparison.csv")
    matches_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved detailed comparison to: {output_file}")
    
    # Key assertion: solar exposure should strongly correlate with human_to_sky
    print(f"\n{'='*80}")
    print(f"TEST RESULTS")
    print(f"{'='*80}")
    
    # Check if our solar exposure categories align with Grasshopper's
    our_categorized = np.where(matches_df['our_solar_exp_h10'] < 0.1, 0.0,
                               np.where(matches_df['our_solar_exp_h10'] > 0.9, 1.0, 0.5))
    agreement = np.mean(our_categorized == matches_df['val_human_to_sky'])
    
    print(f"\nCategory agreement (H10): {agreement*100:.1f}%")
    print(f"  (Shaded/Exposed classification match)")
    
    if agreement > 0.95:
        print(f"\n✅ EXCELLENT: Solar exposure calculation matches Grasshopper!")
    elif agreement > 0.85:
        print(f"\n⚠️  GOOD: Solar exposure mostly matches, some differences")
    else:
        print(f"\n❌ PROBLEM: Solar exposure differs significantly from Grasshopper")
    
    assert agreement > 0.85, f"Solar exposure agreement too low: {agreement*100:.1f}%"


if __name__ == "__main__":
    test_exposure_against_grasshopper_validation()

