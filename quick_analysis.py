"""
Quick UTCI Analysis Runner

A minimal wrapper to run full day UTCI analysis with predefined parameters.
Modify the ANALYSIS_CONFIGS list below to run one or multiple analyses
with different settings in a single execution.

Usage:
    python quick_analysis.py
"""

import sys
from pathlib import Path

# Import the core analysis function
sys.path.insert(0, str(Path(__file__).parent))
from run_analysis import run_analysis_core

# ============================================================================
# ANALYSIS CONFIGURATIONS
# ============================================================================
# Each configuration is a dictionary with parameters for run_analysis_core().
# Add/modify configurations below as needed.

ANALYSIS_CONFIGS = [
    {
        "month": 8,
        "day": 15,
        "grid_size": 10.0,
        "model_file": "data/3d_models/original_with_layers.glb",
        "epw_file": "data/weather/ISR_D_Beer.Sheva.401900_TMYx/ISR_D_Beer.Sheva.401900_TMYx.epw",
        "embree_quality": "low",
        "intersects_any": True,
        "export_csv": False,
        "verbose": True
    },
    # Uncomment below to run multiple analyses:
    # {
    #     "month": 8,
    #     "day": 15,
    #     "grid_size": 5.0,
    #     "embree_quality": "medium",
    #     "intersects_any": True,
    #     "export_csv": False,
    #     "verbose": True
    # },
]


def main():
    """Run analysis with predefined configurations."""
    
    print("="*60)
    print(f"QUICK UTCI ANALYSIS - {len(ANALYSIS_CONFIGS)} Configuration(s)")
    print("="*60)
    
    results = []
    
    for i, config in enumerate(ANALYSIS_CONFIGS, 1):
        print(f"\n{'='*60}")
        print(f"RUNNING CONFIGURATION {i}/{len(ANALYSIS_CONFIGS)}")
        print(f"{'='*60}")
        print(f"Parameters: {config}")
        
        try:
            result = run_analysis_core(**config)
            results.append({"config": config, "result": result, "success": True})
            
            if config.get("verbose", True):
                print(f"\n[OK] Configuration {i} completed successfully")
            
        except Exception as e:
            print(f"\n[ERROR] Configuration {i} failed: {e}")
            import traceback
            traceback.print_exc()
            results.append({"config": config, "error": str(e), "success": False})
    
    # Summary
    print("\n" + "="*60)
    print("QUICK ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total configurations: {len(ANALYSIS_CONFIGS)}")
    print(f"Successful: {sum(1 for r in results if r['success'])}")
    print(f"Failed: {sum(1 for r in results if not r['success'])}")
    
    for i, result in enumerate(results, 1):
        if result["success"]:
            res = result["result"]
            print(f"\nConfig {i}: SUCCESS")
            print(f"  - Grid: {res['grid_size']}m")
            print(f"  - Date: {res['month']}/{res['day']}")
            print(f"  - UTCI: {res['utci_min']:.1f} to {res['utci_max']:.1f}C")
            print(f"  - Runtime: {res['total_time']:.1f}s")
            print(f"  - CSV: {res['csv_path']}")
        else:
            print(f"\nConfig {i}: FAILED")
            print(f"  - Error: {result['error']}")
    
    print("="*60)
    
    return 0 if all(r["success"] for r in results) else 1


if __name__ == "__main__":
    exit(main())
