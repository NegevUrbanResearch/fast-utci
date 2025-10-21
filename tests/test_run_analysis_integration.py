"""Quick integration test for run_analysis.py with refactored code."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from run_analysis import run_analysis_core

if __name__ == "__main__":
    print("Testing run_analysis.py with refactored UTCI calculator...")
    result = run_analysis_core(
        month=8, 
        day=15, 
        grid_size=10.0, 
        export_csv=False, 
        verbose=False
    )
    
    print(f"\n[SUCCESS] run_analysis_core completed")
    print(f"  Positions: {result['num_positions']}")
    print(f"  UTCI range: {result['utci_min']:.1f} to {result['utci_max']:.1f}")
    print(f"  Runtime: {result['total_time']:.1f}s")

