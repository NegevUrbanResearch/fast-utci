# Shared Utilities

This package contains modules shared across both MRT and UTCI calculators.

## Purpose

To avoid code duplication and ensure consistency, common functionality is centralized here:

- **Configuration**: Parallel processing, performance settings
- **Parallel Processing**: Chunking strategies, worker pool management
- **Weather Data**: Loading, filtering, and format conversion (EPW, DataFrame)
- **Environment Variables**: Centralized env var parsing

## Modules

### config.py

Shared configuration classes and environment variable helpers.

**Classes:**
- `ParallelConfig`: Parallel processing settings (n_workers, show_progress, parallel_threshold)
- `PerformanceConfig`: Performance optimization settings (batch_size, ray_max_distance)

**Functions:**
- `get_bool_env(key, default)`: Parse boolean environment variables
- `get_int_env(key, default)`: Parse integer environment variables
- `get_float_env(key, default)`: Parse float environment variables
- `get_str_env(key, default)`: Parse string environment variables

### weather.py

Unified weather data utilities consolidating functionality from:
- `model_reader.py::read_weather_data()` - DataFrame loading from EPW
- `mrt/adapters.py` - Weather adapters (moved here)
- `mrt/period.py::filter_weather_data()` - Period/hour filtering
- `utci/weather.py::WeatherDataManager` - Weather management

**Classes:**
- `WeatherDataSource`: Protocol defining weather adapter interface
- `EPWAdapter`: Adapter for Ladybug EPW weather data
- `DataFrameAdapter`: Adapter for pandas DataFrame weather data
- `WeatherDataManager`: High-level weather loading and filtering

**Functions:**
- `create_weather_adapter(data)`: Factory to create appropriate adapter
- `load_weather_data(file_path)`: Load EPW file, return (DataFrame, EPW)
- `filter_weather_data(df, period, hours)`: Filter DataFrame by period/hours

**Usage:**
```python
from fast_utci.shared.weather import load_weather_data, WeatherDataManager

# Load weather data
weather_df, epw = load_weather_data('weather.epw')

# Or use manager for filtering
mgr = WeatherDataManager('weather.epw')
mgr.filter_by_period(analysis_period)
mgr.filter_by_hours([12, 13, 14])
arrays = mgr.to_numpy_arrays()
```

### parallel_utils.py

Parallel processing utilities for efficient multi-core computation.

**Classes:**
- `ChunkStrategy`: Base class for chunking strategies
- `BalancedChunkStrategy`: Evenly distribute data across workers
- `SpatialChunkStrategy`: Sort spatially for better cache locality
- `ParallelProcessor`: Generic parallel processor with progress tracking

**Functions:**
- `create_balanced_chunks(data, n_workers)`: Create balanced chunks
- `create_spatial_chunks(positions, n_workers)`: Create spatial chunks

## Usage

### In MRT Module

```python
from fast_utci.shared import ParallelConfig, PerformanceConfig

@dataclass
class MRTConfig:
    # MRT-specific params
    human_height: float = 1.8
    
    # Shared configs via composition
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
```

### In UTCI Module

```python
from fast_utci.shared import ParallelConfig

@dataclass
class UTCIConfig:
    # UTCI-specific params
    enable_vectorized: bool = True
    
    # Shared config via composition
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
```

### Using Parallel Processor

```python
from fast_utci.shared import ParallelProcessor, BalancedChunkStrategy

processor = ParallelProcessor(n_workers=4, show_progress=True)
results = processor.process_chunks(
    data=my_data,
    worker_fn=my_worker_function,
    chunk_strategy=BalancedChunkStrategy(),
    description="Processing data"
)
```

## Design Pattern: Composition over Inheritance

Both MRT and UTCI configs **compose** the shared configs rather than inheriting:

```python
# Good: Composition
class MRTConfig:
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    
# Bad: Inheritance (avoided)
class MRTConfig(ParallelConfig):  # Don't do this
    pass
```

This allows:
- Independent evolution of each config
- Clear ownership of parameters
- Easier testing
- Better type hints

## Environment Variables

All environment variables are read through the shared helper functions:

```python
from fast_utci.shared import get_bool_env, get_int_env

enable_feature = get_bool_env("FAST_UTCI_ENABLE_FEATURE", default=False)
n_workers = get_int_env("FAST_UTCI_N_WORKERS", default=4)
```

Supported formats:
- **Boolean**: `1`, `true`, `yes`, `on` → True; `0`, `false`, `no`, `off`, `""` → False
- **Integer**: Any valid integer string
- **Float**: Any valid float string
- **String**: Raw string value

