"""
UTCI calculation module for fast-utci.

This package provides Universal Thermal Climate Index (UTCI) calculation
functionality that integrates with MRT results and weather data.
"""

from fast_utci.shared import UTCIConfig

from .calculator import UTCICalculator
from .calculation import BoundaryAveragingCalculator, UTCICalculationResult
from fast_utci.shared.weather import WeatherDataManager
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
from fast_utci.shared.io.export import export_utci_results, export_utci_results_json

__version__ = "0.1.0"
__all__ = [
    # Config
    "UTCIConfig",
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
    "export_utci_results",
    "export_utci_results_json"
]

