"""
fast-utci: Rapidly compute 2D UTCI maps from 3D models.

This package provides tools for computing Universal Thermal Climate Index (UTCI)
values from 3D building models using Mean Radiant Temperature (MRT) calculations.
"""

from importlib import import_module

__version__ = "0.1.0"

__all__ = [
    "MRTCalculator",
    "MRTConfig",
    "UTCICalculator",
    "read_project_data",
    "get_combined_mesh",
    "get_ground_bounds",
]


def __getattr__(name):
    """Load public exports only when requested, keeping GIS imports independent."""
    modules = {
        "MRTCalculator": "fast_utci.mrt",
        "MRTConfig": "fast_utci.shared.config",
        "UTCICalculator": "fast_utci.utci",
        "read_project_data": "fast_utci.shared.io.model",
        "get_combined_mesh": "fast_utci.shared.io.scene",
        "get_ground_bounds": "fast_utci.shared.io.scene",
    }
    if name not in modules:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(modules[name]), name)
    globals()[name] = value
    return value

