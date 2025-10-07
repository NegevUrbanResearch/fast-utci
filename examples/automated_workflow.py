#!/usr/bin/env python3
"""
Automated UTCI Workflow Script

This script runs the UTCI workflow with default settings:
- Single hour analysis at 13:00 (hour 13)
- No model simplification
- 50m grid spacing for faster testing

Usage:
    python run_automated_workflow.py
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def run_automated_workflow():
    """Run the UTCI workflow with automated default settings."""
    
    print("Starting Automated UTCI Workflow")
    print("=" * 50)
    print("Settings:")
    print("  - Analysis: Single hour at 13:00")
    print("  - Model: Original (no simplification)")
    print("  - Grid: 10m spacing")
    print("  - Validation: Against Grasshopper data")
    print("=" * 50)
    
    try:
        # Optional performance toggles via environment variables
        # Enable vectorized solar exposure by setting FAST_UTCI_VECTORIZED_SOLAR=1
        if os.getenv("FAST_UTCI_VECTORIZED_SOLAR"):
            print(f"[CONFIG] Vectorized solar exposure enabled (FAST_UTCI_VECTORIZED_SOLAR={os.getenv('FAST_UTCI_VECTORIZED_SOLAR')})")
        # Embree tuning can be applied via:
        # FAST_UTCI_INTERSECTOR=embree|trimesh
        # FAST_UTCI_EMBREE_QUALITY=low|medium|high
        # FAST_UTCI_EMBREE_BUILD_BVH=true|false
        # FAST_UTCI_EMBREE_PACKET_SIZE=0|4|8|16
        if os.getenv("FAST_UTCI_INTERSECTOR") or os.getenv("FAST_UTCI_EMBREE_QUALITY"):
            print("[CONFIG] Embree tuning via environment is active.")

        # Import the main workflow function
        import demo_workflow
        
        # Override the interactive prompts with automated choices
        # (We'll reference the module for mocking)
        
        # Mock the user input functions to return default values
        def mock_get_user_analysis_choice():
            print("[AUTO] Analysis Mode: Single hour (automated)")
            return "single_hour"
        
        def mock_get_user_simplification_choice():
            print("[AUTO] Model Simplification: No (automated)")
            return False
        
        def mock_get_user_grid_choice():
            print("[AUTO] Grid Size: 10m (automated)")
            return 10.0
        
        def mock_get_single_hour_input():
            print("[AUTO] Analysis Hour: 13:00 (automated)")
            return 13
        
        # Replace the interactive functions with our mocks
        demo_workflow.get_user_analysis_choice = mock_get_user_analysis_choice
        demo_workflow.get_user_simplification_choice = mock_get_user_simplification_choice
        demo_workflow.get_user_grid_choice = mock_get_user_grid_choice
        demo_workflow.get_single_hour_input = mock_get_single_hour_input
        
        # Run the main workflow
        result = demo_workflow.main()
        
        print("\n[SUCCESS] Automated workflow completed successfully!")
        print(f"[OUTPUT] Files generated in current directory")
        
        return result
        
    except Exception as e:
        print(f"\n[ERROR] Error in automated workflow: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    run_automated_workflow()
