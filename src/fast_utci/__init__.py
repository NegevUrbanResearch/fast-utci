"""
fast-utci: Rapidly compute 2D UTCI maps from 3D models.

This package provides tools for computing Universal Thermal Climate Index (UTCI)
values from 3D building models using Mean Radiant Temperature (MRT) calculations.
"""

from fast_utci.mrt import MRTCalculator, MRTConfig, DEFAULT_CONFIG
from fast_utci.utci import UTCICalculator
from fast_utci.model_reader import read_project_data, get_combined_mesh, get_ground_bounds

__version__ = "0.1.0"

__all__ = [
    "MRTCalculator",
    "MRTConfig", 
    "DEFAULT_CONFIG",
    "UTCICalculator",
    "read_project_data",
    "get_combined_mesh",
    "get_ground_bounds",
]

