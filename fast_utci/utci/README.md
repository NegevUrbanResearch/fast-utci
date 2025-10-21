# UTCI Calculator Module

## What This Does

Computes Universal Thermal Climate Index (UTCI) from Mean Radiant Temperature (MRT) results and weather data using pythermalcomfort. This module provides a clean, modular architecture for UTCI calculations.

## Architecture

The UTCI module follows a clean separation of concerns:

```
fast_utci/utci/
├── calculator.py      # Main orchestrator
├── calculation.py     # Core UTCI computation (boundary averaging)
├── weather.py         # Weather data management
├── statistics.py      # Thermal comfort classification & stats
├── export.py          # CSV/JSON export functionality
└── config.py          # UTCI-specific configuration
```

### Design Principles

1. **Single Responsibility**: Each module has one clear purpose
2. **No Code Duplication**: Boundary averaging logic in ONE place
3. **Shared Utilities**: Reuses `fast_utci.shared` for parallel processing and config
4. **Type Safety**: Uses dataclasses for structured results
5. **Testable**: Clean interfaces make unit testing easy

## Quick Start

```python
from fast_utci.utci import UTCICalculator
from fast_utci.mrt import MRTCalculator, create_rectangular_grid

# Compute MRT (see fast_utci/mrt/README.md)
mrt_calc = MRTCalculator(context_meshes=['buildings.obj'])
mrt_calc.set_location_from_epw('weather.epw')
grid = create_rectangular_grid(bounds_min=[0,0], bounds_max=[100,100], grid_size=10.0)
mrt_results = mrt_calc.compute_mrt(epw_data, exposure_results)

# Compute UTCI
utci_calc = UTCICalculator(weather_data='weather.epw')
utci_results = utci_calc.compute_utci(
    mrt_results=mrt_results,
    show_progress=True
)

# Export results
utci_calc.to_csv(utci_results, 'output.csv')
summary = utci_calc.summary_statistics(utci_results)
```

## Key Features

### Boundary Averaging

UTCI calculations use boundary averaging for accurate temporal interpolation:

- For each hour N:
  - Calculate UTCI₀ using (MRT₀[N], weather[N])
  - Calculate UTCI₁ using (MRT₁[N], weather[N+1])
  - Average: UTCI_avg = (UTCI₀ + UTCI₁) / 2

This matches the Grasshopper OutdoorSolarMRT workflow.

### Parallel Processing

- **Automatic**: Switches to parallel mode for > 50 positions
- **Configurable**: Set `config.parallel.n_workers` or pass `n_workers=N`
- **Progress Tracking**: Real-time progress bars with ETA

### Weather Data Management

Flexible weather data loading:
- EPW files (via ladybug-core)
- pandas DataFrames
- EPW objects

Automatic filtering:
- By analysis period (month/day ranges)
- By target hours (specific hours of day)

### Thermal Comfort Classification

Automatic classification into UTCI categories:
- Extreme Cold (< -40°C)
- Very Cold (-40 to -27°C)
- Cold (-27 to -13°C)
- Cool (-13 to 9°C)
- **Comfortable (9 to 26°C)**
- Warm (26 to 32°C)
- Hot (32 to 38°C)
- Very Hot (38 to 46°C)
- Extreme Hot (> 46°C)

## Configuration

```python
from fast_utci.utci import UTCIConfig

config = UTCIConfig(
    enable_vectorized=True,  # Use numpy vectorization
    include_weather_in_results=True,  # Include air_temp, wind, RH in results
    include_datetime_in_results=True,  # Include datetime info
    parallel=ParallelConfig(
        n_workers=None,  # Auto-detect CPU count
        show_progress=True,
        parallel_threshold=50  # Min positions for parallel mode
    )
)

calc = UTCICalculator(weather_data='weather.epw', config=config)
```

## API Reference

### UTCICalculator

Main calculator class.

**Methods:**
- `__init__(weather_data, epw_object=None, config=None)`
- `compute_utci(mrt_results, analysis_period=None, target_hours=None, show_progress=None, n_workers=None)`
- `to_csv(utci_results, csv_path, include_weather=True, include_comfort_categories=True)`
- `summary_statistics(utci_results)`

### WeatherDataManager

Handles weather data loading and filtering.

**Methods:**
- `__init__(weather_source, epw_object=None)`
- `filter_by_period(analysis_period)`
- `filter_by_hours(target_hours)`
- `to_dataframe()`
- `to_numpy_arrays()`

### BoundaryAveragingCalculator

Core UTCI calculation with boundary averaging.

**Methods:**
- `calculate(mrt_data, weather_data, include_weather=True, include_datetime=True)`

Returns `UTCICalculationResult` dataclass.

## Performance

- **Vectorized calculations**: Uses numpy for 10-100x speedup
- **Parallel processing**: Automatic for large datasets
- **Memory efficient**: Zero-copy numpy views when possible
- **Progress tracking**: Real-time feedback with tqdm

## Testing

See `tests/test_utci_refactor_validation.py` for integration tests that validate identical results to the original implementation.

## Migration from Old Code

If you're using the old `fast_utci.utci_calculator` module:

```python
# Old way
from fast_utci.utci_calculator import UTCICalculator

# New way (same API!)
from fast_utci.utci import UTCICalculator
```

The API is identical - just update the import path.

