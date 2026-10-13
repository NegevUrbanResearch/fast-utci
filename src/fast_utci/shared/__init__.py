"""
Shared utilities for fast-utci.

This package contains modules shared across MRT and UTCI calculators,
including configuration, parallel processing utilities, weather data handling,
and common helpers.
"""

from .config import (
    ParallelConfig,
    PerformanceConfig,
    MRTConfig,
    UTCIConfig,
    AppConfig,
    load_config,
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

def __getattr__(name):
    """Defer weather dependencies until a weather export is requested."""
    if name in {
        "WeatherDataSource", "EPWAdapter", "DataFrameAdapter",
        "create_weather_adapter", "WeatherDataManager", "load_weather_data",
        "filter_weather_data",
    }:
        from importlib import import_module

        value = getattr(import_module("fast_utci.shared.weather"), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Config
    "ParallelConfig",
    "PerformanceConfig",
    "MRTConfig",
    "UTCIConfig",
    "AppConfig",
    "load_config",
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

