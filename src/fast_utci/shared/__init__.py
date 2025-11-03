"""
Shared utilities for fast-utci.

This package contains modules shared across MRT and UTCI calculators,
including configuration, parallel processing utilities, weather data handling,
and common helpers.
"""

from .config import (
    ParallelConfig,
    PerformanceConfig,
    DEFAULT_PARALLEL_CONFIG,
    DEFAULT_PERFORMANCE_CONFIG,
    get_bool_env,
    get_int_env,
    get_float_env,
    get_str_env
)

from .parallel_utils import (
    ChunkStrategy,
    BalancedChunkStrategy,
    SpatialChunkStrategy,
    ParallelProcessor,
    create_balanced_chunks,
    create_spatial_chunks
)

from .weather import (
    WeatherDataSource,
    EPWAdapter,
    DataFrameAdapter,
    create_weather_adapter,
    WeatherDataManager,
    load_weather_data,
    filter_weather_data
)

__all__ = [
    # Config
    "ParallelConfig",
    "PerformanceConfig",
    "DEFAULT_PARALLEL_CONFIG",
    "DEFAULT_PERFORMANCE_CONFIG",
    "get_bool_env",
    "get_int_env",
    "get_float_env",
    "get_str_env",
    # Parallel utilities
    "ChunkStrategy",
    "BalancedChunkStrategy",
    "SpatialChunkStrategy",
    "ParallelProcessor",
    "create_balanced_chunks",
    "create_spatial_chunks",
    # Weather utilities
    "WeatherDataSource",
    "EPWAdapter",
    "DataFrameAdapter",
    "create_weather_adapter",
    "WeatherDataManager",
    "load_weather_data",
    "filter_weather_data"
]

