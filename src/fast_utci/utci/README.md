# UTCI Calculator Module

This module is part of the legacy Python/Ladybug CPU pipeline. It provides UTCI tools for reference calculations, validation, legacy exports, and parity checks.

The Python/Ladybug pathway was the scriptable Embree/parallel-CPU improvement over Grasshopper/Ladybug. The current WebGPU/Three.js route is the main and faster path for new analysis.

## What This Does

Computes Universal Thermal Climate Index (UTCI) from Mean Radiant Temperature (MRT) results and weather data using `pythermalcomfort`. In the legacy pathway, those MRT and weather inputs usually come from the Ladybug-backed MRT/EPW modules.

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

## Legacy Reference Example

The example below documents the Python/Ladybug reference path. For new production analysis, prefer the WebGPU/Three.js viewer path unless you specifically need parity data or a legacy export.

```python
from fast_utci.utci import UTCICalculator
from fast_utci.mrt import MRTCalculator, create_rectangular_grid, create_analysis_period
from fast_utci.shared import load_config
from fast_utci.shared.io import read_project_data, get_combined_mesh, get_ground_bounds
from ladybug.epw import EPW

cfg = load_config()

# Load model and get combined mesh
scene, _, _ = read_project_data('buildings.glb', 'weather.epw')
model = get_combined_mesh(scene)

# Compute MRT (see fast_utci/mrt/README.md)
mrt_calc = MRTCalculator(context_meshes=[model], config=cfg.mrt)
mrt_calc.set_location_from_epw('weather.epw')

# Create analysis grid
model_bounds = get_ground_bounds(scene)
grid = create_rectangular_grid(
    bounds_min=model_bounds[0][:2],
    bounds_max=model_bounds[1][:2],
    grid_size=10.0,
    z_height=1.5
)

# Compute MRT
period = create_analysis_period(start_month=8, start_day=15, start_hour=0, end_month=8, end_day=15, end_hour=23)
epw = EPW('weather.epw')
exposure_results = mrt_calc.compute_exposure(grid.points, period)
mrt_results = mrt_calc.compute_mrt(epw, exposure_results, period)

# Compute UTCI
utci_calc = UTCICalculator(weather_data='weather.epw', config=cfg.utci)
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
from fast_utci.shared import ParallelConfig

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

These notes describe performance within the legacy CPU reference path. They are useful for maintaining parity scripts and exports, but they should not be read as a recommendation to use Python for new large-scale analysis over the WebGPU/Three.js path.

- **Vectorized calculations**: Uses numpy for 10-100x speedup
- **Parallel processing**: Automatic for large datasets
- **Memory efficient**: Zero-copy numpy views when possible
- **Progress tracking**: Real-time feedback with tqdm

## Testing

See `tests/test_utci_refactor_validation.py` for integration tests that validate identical results to the original implementation.

## Migration from Old Code

If you need the compatibility import path:

```python
# Old way
from fast_utci.utci_calculator import UTCICalculator

# New way (same API!)
from fast_utci.utci import UTCICalculator
```

The API is identical - just update the import path.

