"""
Custom exceptions for MRT calculations.

Provides a clear exception hierarchy for different types of errors that can occur
during MRT computation, making error handling more explicit and debugging easier.
"""


class MRTCalculationError(Exception):
    """Base exception for all MRT calculation errors."""
    pass


class IntersectorError(MRTCalculationError):
    """Exception raised when ray intersection operations fail."""
    pass


class WeatherDataError(MRTCalculationError):
    """Exception raised when weather data is invalid or missing."""
    pass


class ConfigurationError(MRTCalculationError):
    """Exception raised when configuration parameters are invalid."""
    pass

