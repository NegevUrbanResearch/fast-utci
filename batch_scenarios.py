"""
Batch UTCI Analysis for All Scenarios

Runs UTCI analysis sequentially for all 50 scenario variations.
Uses consistent settings across all scenarios for comparison.

Usage:
    python batch_scenarios.py [--grid-size 10.0] [--month 8] [--day 15]
"""

import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from run_analysis import run_analysis_core

# Scenario configuration
SCENARIO_CATEGORIES = [
    "existing_buildings",
    "existing_trees",
    "new_high_buildings",
    "new_low_buildings",
    "new_trees"
]

# Default analysis settings
DEFAULT_GRID_SIZE = 10.0
DEFAULT_MONTH = 8
DEFAULT_DAY = 15
EPW_FILE = "data/weather/ISR_D_Beer.Sheva.401900_TMYx/ISR_D_Beer.Sheva.401900_TMYx.epw"

def run_batch_scenarios(grid_size=DEFAULT_GRID_SIZE, month=DEFAULT_MONTH, day=DEFAULT_DAY):
    """Run analysis for all 50 scenarios sequentially."""
    
    print("="*70)
    print("BATCH UTCI ANALYSIS - 50 SCENARIOS")
    print("="*70)
    print(f"Settings: {grid_size}m grid, {month}/{day}, sequential processing")
    print(f"Total scenarios: {len(SCENARIO_CATEGORIES)} categories × 10 variants = 50")
    print("="*70)
    
    results = []
    start_time = time.perf_counter()
    
    scenario_num = 0
    for category in SCENARIO_CATEGORIES:
        for variant in range(1, 11):
            scenario_num += 1
            model_name = f"{category}_{variant:02d}"
            model_file = f"data/3d_models/scenarios/{category}/{model_name}.glb"
            
            print(f"\n{'='*70}")
            print(f"SCENARIO {scenario_num}/50: {category} - Variant {variant}")
            print(f"{'='*70}")
            print(f"Model: {model_file}")
            
            try:
                result = run_analysis_core(
                    month=month,
                    day=day,
                    grid_size=grid_size,
                    model_file=model_file,
                    epw_file=EPW_FILE,
                    embree_quality="low",
                    intersects_any=True,
                    export_csv=False,
                    verbose=False,  # Minimal logging for batch
                    category=category  # Pass category for subdirectory organization
                )
                
                # Log summary
                print(f"[OK] Completed in {result['total_time']:.1f}s")
                print(f"     UTCI range: {result['utci_min']:.1f} to {result['utci_max']:.1f}°C")
                print(f"     Positions: {result['num_positions']}")
                
                results.append({
                    "category": category,
                    "variant": variant,
                    "model_name": model_name,
                    "success": True,
                    "result": result
                })
                
            except Exception as e:
                print(f"[ERROR] Scenario failed: {e}")
                results.append({
                    "category": category,
                    "variant": variant,
                    "model_name": model_name,
                    "success": False,
                    "error": str(e)
                })
            
            # Progress indicator
            elapsed = time.perf_counter() - start_time
            avg_time = elapsed / scenario_num
            remaining = (50 - scenario_num) * avg_time
            print(f"Progress: {scenario_num}/50 ({scenario_num*2}%) | "
                  f"Elapsed: {elapsed/60:.1f}min | ETA: {remaining/60:.1f}min")
    
    # Final summary
    total_time = time.perf_counter() - start_time
    successful = sum(1 for r in results if r['success'])
    failed = sum(1 for r in results if not r['success'])
    
    print("\n" + "="*70)
    print("BATCH ANALYSIS COMPLETE")
    print("="*70)
    print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"Successful: {successful}/50")
    print(f"Failed: {failed}/50")
    
    if failed > 0:
        print("\nFailed scenarios:")
        for r in results:
            if not r['success']:
                print(f"  - {r['category']} variant {r['variant']}: {r['error']}")
    
    return results

def main():
    """CLI entry point with argument parsing."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch UTCI analysis for all scenarios')
    parser.add_argument('--grid-size', type=float, default=DEFAULT_GRID_SIZE,
                       help=f'Grid spacing in meters (default: {DEFAULT_GRID_SIZE})')
    parser.add_argument('--month', type=int, default=DEFAULT_MONTH,
                       help=f'Analysis month (default: {DEFAULT_MONTH})')
    parser.add_argument('--day', type=int, default=DEFAULT_DAY,
                       help=f'Analysis day (default: {DEFAULT_DAY})')
    
    args = parser.parse_args()
    
    results = run_batch_scenarios(
        grid_size=args.grid_size,
        month=args.month,
        day=args.day
    )
    
    return 0 if all(r['success'] for r in results) else 1

if __name__ == "__main__":
    exit(main())
