"""
Baseline tests for weather utilities consolidation.

This test suite captures the current behavior of weather loading and filtering
across different modules before consolidation. After refactoring, these tests
ensure the new shared.weather module produces identical results.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fast_utci.model_reader import read_weather_data
from fast_utci.mrt.period import filter_weather_data, create_analysis_period
from fast_utci.shared.weather import create_weather_adapter, EPWAdapter, DataFrameAdapter
from ladybug.epw import EPW


# Test data paths
EPW_FILE = "data/weather/ISR_Beer.Sheva.401900_MSI.epw"


def test_model_reader_weather_loading():
    """Test current model_reader.py weather loading behavior."""
    print("\n[TEST] model_reader.read_weather_data()...")
    
    df = read_weather_data(EPW_FILE)
    
    # Verify expected columns
    expected_columns = {
        'datetime', 'air_temp', 'wind_speed', 'relative_humidity',
        'global_horizontal_radiation', 'direct_normal_radiation',
        'diffuse_horizontal_radiation', 'horizontal_infrared_radiation_intensity',
        'surface_temp'
    }
    assert set(df.columns) == expected_columns, f"Unexpected columns: {df.columns}"
    
    # Verify data shape (8760 hours in a year)
    assert len(df) == 8760, f"Expected 8760 rows, got {len(df)}"
    
    # Verify data types
    assert df['air_temp'].dtype in [np.float64, float]
    assert df['wind_speed'].dtype in [np.float64, float]
    
    # Save baseline statistics
    baseline_stats = {
        'mean_air_temp': df['air_temp'].mean(),
        'mean_wind_speed': df['wind_speed'].mean(),
        'mean_rh': df['relative_humidity'].mean(),
        'mean_direct_rad': df['direct_normal_radiation'].mean(),
        'first_datetime': df['datetime'].iloc[0],
        'last_datetime': df['datetime'].iloc[-1]
    }
    
    print(f"  [OK] Loaded {len(df)} hours of weather data")
    print(f"  [OK] Mean air temp: {baseline_stats['mean_air_temp']:.2f}C")
    print(f"  [OK] Mean wind speed: {baseline_stats['mean_wind_speed']:.2f} m/s")
    
    return df, baseline_stats


def test_mrt_adapters_epw():
    """Test current mrt/adapters.py EPWAdapter behavior."""
    print("\n[TEST] mrt/adapters.py EPWAdapter...")
    
    epw = EPW(EPW_FILE)
    adapter = EPWAdapter(epw)
    
    # Test adapter methods
    temp = adapter.get_temperature()
    direct_rad = adapter.get_direct_radiation()
    diffuse_rad = adapter.get_diffuse_radiation()
    ir_rad = adapter.get_infrared_radiation()
    datetimes = adapter.get_datetimes()
    
    # Verify shapes
    assert len(temp) == 8760
    assert len(direct_rad) == 8760
    assert len(diffuse_rad) == 8760
    assert len(ir_rad) == 8760
    assert len(datetimes) == 8760
    
    # Verify data types
    assert isinstance(temp, np.ndarray)
    assert isinstance(direct_rad, np.ndarray)
    
    adapter_stats = {
        'mean_temp': temp.mean(),
        'mean_direct_rad': direct_rad.mean(),
        'mean_diffuse_rad': diffuse_rad.mean(),
        'mean_ir_rad': ir_rad.mean()
    }
    
    print(f"  [OK] EPWAdapter working correctly")
    print(f"  [OK] Mean temp: {adapter_stats['mean_temp']:.2f}C")
    
    return adapter, adapter_stats


def test_mrt_adapters_dataframe():
    """Test current mrt/adapters.py DataFrameAdapter behavior."""
    print("\n[TEST] mrt/adapters.py DataFrameAdapter...")
    
    # Load weather data first
    df = read_weather_data(EPW_FILE)
    adapter = DataFrameAdapter(df)
    
    # Test adapter methods
    temp = adapter.get_temperature()
    direct_rad = adapter.get_direct_radiation()
    diffuse_rad = adapter.get_diffuse_radiation()
    ir_rad = adapter.get_infrared_radiation()
    datetimes = adapter.get_datetimes()
    
    # Verify shapes
    assert len(temp) == 8760
    assert len(direct_rad) == 8760
    
    print(f"  [OK] DataFrameAdapter working correctly")
    
    return adapter


def test_create_weather_adapter():
    """Test adapter factory function."""
    print("\n[TEST] create_weather_adapter() factory...")
    
    # Test with EPW object
    epw = EPW(EPW_FILE)
    adapter1 = create_weather_adapter(epw)
    assert isinstance(adapter1, EPWAdapter)
    print("  [OK] EPW object -> EPWAdapter")
    
    # Test with DataFrame
    df = read_weather_data(EPW_FILE)
    adapter2 = create_weather_adapter(df)
    assert isinstance(adapter2, DataFrameAdapter)
    print("  [OK] DataFrame -> DataFrameAdapter")
    
    return adapter1, adapter2


def test_period_filtering():
    """Test current mrt/period.py filtering behavior."""
    print("\n[TEST] mrt/period.py filter_weather_data()...")
    
    # Load full year data
    df_full = read_weather_data(EPW_FILE)
    
    # Test 1: Filter to August 15 (single day)
    period_aug15 = create_analysis_period(
        start_month=8, start_day=15,
        end_month=8, end_day=15,
        start_hour=0, end_hour=23
    )
    df_filtered = filter_weather_data(df_full, analysis_period=period_aug15)
    
    assert len(df_filtered) == 24, f"Expected 24 hours, got {len(df_filtered)}"
    assert all(df_filtered['datetime'].dt.month == 8)
    assert all(df_filtered['datetime'].dt.day == 15)
    print(f"  [OK] Filtered to Aug 15: {len(df_filtered)} hours")
    
    # Test 2: Filter to specific hours (12-14)
    df_hours = filter_weather_data(df_full, target_hours=[12, 13, 14])
    expected_hours = 365 * 3  # 3 hours per day for full year
    assert len(df_hours) == expected_hours, f"Expected {expected_hours}, got {len(df_hours)}"
    print(f"  [OK] Filtered to hours 12-14: {len(df_hours)} hours")
    
    # Test 3: Combined period + hours
    df_combined = filter_weather_data(
        df_full,
        analysis_period=period_aug15,
        target_hours=[12, 13, 14]
    )
    assert len(df_combined) == 3, f"Expected 3 hours, got {len(df_combined)}"
    assert all(df_combined['datetime'].dt.hour.isin([12, 13, 14]))
    print(f"  [OK] Combined filter (Aug 15, hours 12-14): {len(df_combined)} hours")
    
    filtering_stats = {
        'full_year_rows': len(df_full),
        'aug15_rows': len(df_filtered),
        'hours_12_14_rows': len(df_hours),
        'combined_rows': len(df_combined)
    }
    
    return filtering_stats


def test_epw_vs_dataframe_consistency():
    """Test that EPW and DataFrame adapters produce identical results."""
    print("\n[TEST] EPW vs DataFrame adapter consistency...")
    
    epw = EPW(EPW_FILE)
    df = read_weather_data(EPW_FILE)
    
    epw_adapter = EPWAdapter(epw)
    df_adapter = DataFrameAdapter(df)
    
    # Compare outputs
    epw_temp = epw_adapter.get_temperature()
    df_temp = df_adapter.get_temperature()
    
    epw_direct = epw_adapter.get_direct_radiation()
    df_direct = df_adapter.get_direct_radiation()
    
    # Should be identical
    np.testing.assert_array_almost_equal(epw_temp, df_temp, decimal=5)
    np.testing.assert_array_almost_equal(epw_direct, df_direct, decimal=5)
    
    print("  [OK] EPW and DataFrame adapters produce identical results")


def save_baseline_reference():
    """Save baseline reference data for post-consolidation validation."""
    print("\n[BASELINE] Saving reference data...")
    
    df = read_weather_data(EPW_FILE)
    
    # Create analysis period for Aug 15
    period = create_analysis_period(
        start_month=8, start_day=15,
        end_month=8, end_day=15
    )
    df_filtered = filter_weather_data(df, analysis_period=period, target_hours=[12, 13, 14])
    
    # Save baseline
    baseline_data = {
        'full_year_shape': df.shape,
        'full_year_columns': list(df.columns),
        'full_year_air_temp_mean': df['air_temp'].mean(),
        'full_year_wind_speed_mean': df['wind_speed'].mean(),
        'filtered_aug15_12_14': df_filtered.to_dict('records'),
        'filtered_air_temps': df_filtered['air_temp'].values,
        'filtered_datetimes': [str(dt) for dt in df_filtered['datetime']]
    }
    
    # Save as numpy archive
    output_file = Path(__file__).parent / "fixtures" / "weather_baseline.npz"
    output_file.parent.mkdir(exist_ok=True)
    
    np.savez(
        output_file,
        full_year_air_temp=df['air_temp'].values,
        full_year_wind_speed=df['wind_speed'].values,
        full_year_rh=df['relative_humidity'].values,
        filtered_air_temp=df_filtered['air_temp'].values,
        filtered_wind_speed=df_filtered['wind_speed'].values,
        filtered_rh=df_filtered['relative_humidity'].values,
        allow_pickle=True
    )
    
    print(f"  [OK] Baseline saved to: {output_file}")
    print(f"  [OK] Full year mean air temp: {baseline_data['full_year_air_temp_mean']:.2f}C")
    print(f"  [OK] Filtered (Aug 15, 12-14h): {len(baseline_data['filtered_air_temps'])} values")
    
    return baseline_data


if __name__ == "__main__":
    print("=" * 70)
    print("WEATHER UTILITIES BASELINE TESTS")
    print("=" * 70)
    
    # Run all baseline tests
    df, stats1 = test_model_reader_weather_loading()
    adapter_epw, stats2 = test_mrt_adapters_epw()
    adapter_df = test_mrt_adapters_dataframe()
    adapters = test_create_weather_adapter()
    stats3 = test_period_filtering()
    test_epw_vs_dataframe_consistency()
    baseline = save_baseline_reference()
    
    print("\n" + "=" * 70)
    print("ALL BASELINE TESTS PASSED")
    print("=" * 70)
    print(f"\nBaseline captured:")
    print(f"  - Full year: {stats3['full_year_rows']} hours")
    print(f"  - Aug 15: {stats3['aug15_rows']} hours")
    print(f"  - Hours 12-14 only: {stats3['hours_12_14_rows']} hours")
    print(f"  - Aug 15 + hours 12-14: {stats3['combined_rows']} hours")
    print(f"\nReady for consolidation refactoring!")

