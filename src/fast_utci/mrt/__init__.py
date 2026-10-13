from fast_utci.shared import MRTConfig
from .grid import create_rectangular_grid, AnalysisGrid
from .period import AnalysisPeriod, create_analysis_period, create_validation_period_filter
from .cache import CacheManager
from .performance import PerformanceOptimizer
from .exceptions import (
    MRTCalculationError, 
    IntersectorError, 
    WeatherDataError, 
    ConfigurationError
)

# Import shared utilities (backward compatibility - now in fast_utci.shared)
from fast_utci.shared import (
    ParallelConfig,
    PerformanceConfig,
    ParallelProcessor,
    BalancedChunkStrategy,
    SpatialChunkStrategy,
)

__version__ = "0.1.0"
__all__ = [
    "MRTCalculator", 
    "MRTConfig", 
    "create_rectangular_grid", 
    "AnalysisGrid", 
    "AnalysisPeriod", 
    "create_analysis_period", 
    "create_validation_period_filter",
    "CacheManager",
    "PerformanceOptimizer",
    "WeatherDataSource",
    "EPWAdapter",
    "DataFrameAdapter",
    "create_weather_adapter",
    "MRTCalculationError",
    "IntersectorError",
    "WeatherDataError",
    "ConfigurationError",
    # Shared utilities
    "ParallelConfig",
    "PerformanceConfig",
    "ParallelProcessor",
    "BalancedChunkStrategy",
    "SpatialChunkStrategy"
]


def __getattr__(name):
    """Avoid loading the calculator while shared weather imports period helpers."""
    if name == "MRTCalculator":
        from .mrt_calculator import MRTCalculator

        value = MRTCalculator
    elif name in {"WeatherDataSource", "EPWAdapter", "DataFrameAdapter", "create_weather_adapter"}:
        from fast_utci.shared import weather

        value = getattr(weather, name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value
