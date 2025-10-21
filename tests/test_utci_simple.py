"""Simple test of refactored UTCI calculator (serial mode, no parallelism)."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

if __name__ == "__main__":
    from fast_utci.utci import UTCICalculator
    import pandas as pd
    
    print("Testing refactored UTCI calculator (serial mode)...")
    
    # Create simple test weather data
    weather_df = pd.DataFrame({
        'datetime': pd.date_range('2024-08-15', periods=24, freq='H'),
        'air_temp': np.linspace(20, 35, 24),
        'wind_speed': np.ones(24) * 2.0,
        'relative_humidity': np.ones(24) * 50.0
    })
    
    # Create test MRT results (small dataset - will use serial processing)
    mrt_results = {}
    for i in range(5):  # Only 5 positions - triggers serial mode
        mrt_results[f'position_{i}'] = {
            'position': (float(i), 0.0, 1.5),
            'mrt0': np.linspace(25, 40, 24),
            'mrt1': np.linspace(26, 41, 24)
        }
    
    # Test calculator
    calc = UTCICalculator(weather_data=weather_df)
    
    utci_results = calc.compute_utci(
        mrt_results=mrt_results,
        show_progress=False
    )
    
    print(f"[OK] Computed UTCI for {len(utci_results)} positions")
    
    # Check results
    for pos_key, data in utci_results.items():
        assert 'utci' in data
        assert 'position' in data
        assert len(data['utci']) == 24
        print(f"  {pos_key}: UTCI range {data['utci'].min():.1f} to {data['utci'].max():.1f}")
    
    print("\n[SUCCESS] Refactored UTCI calculator works correctly!")

