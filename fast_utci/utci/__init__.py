"""
UTCI calculation module for fast-utci.

This package provides Universal Thermal Climate Index (UTCI) calculation
functionality that integrates with MRT results and weather data.
"""

from .config import (
    UTCIConfig,
    DEFAULT_CONFIG,
    CSV_ENCODING,
    CSV_INDEX,
    DEFAULT_N_WORKERS,
    DEFAULT_SHOW_PROGRESS,
    DEFAULT_PARALLEL_THRESHOLD
)

from .calculator import UTCICalculator
from .calculation import BoundaryAveragingCalculator, UTCICalculationResult
from .weather import WeatherDataManager
from .statistics import (
    classify_thermal_comfort,
    compute_summary_statistics,
    UTCI_COMFORT_THRESHOLDS,
    print_summary,
    extract_all_utci_values,
    extract_positions,
    calculate_utci_statistics,
    calculate_hour_statistics
)
from .export import to_csv, to_json

__version__ = "0.1.0"
__all__ = [
    # Config
    "UTCIConfig",
    "DEFAULT_CONFIG",
    "CSV_ENCODING",
    "CSV_INDEX",
    "DEFAULT_N_WORKERS",
    "DEFAULT_SHOW_PROGRESS",
    "DEFAULT_PARALLEL_THRESHOLD",
    # Main calculator
    "UTCICalculator",
    # Calculation components
    "BoundaryAveragingCalculator",
    "UTCICalculationResult",
    # Weather management
    "WeatherDataManager",
    # Statistics
    "classify_thermal_comfort",
    "compute_summary_statistics",
    "UTCI_COMFORT_THRESHOLDS",
    "print_summary",
    "extract_all_utci_values",
    "extract_positions",
    "calculate_utci_statistics",
    "calculate_hour_statistics",
    # Export
    "to_csv",
    "to_json"
]

