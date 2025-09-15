"""
MRT Calculator for fast-utci

A modular, high-performance Mean Radiant Temperature calculator that matches
Grasshopper/Ladybug OutdoorSolarMRT results while being optimized for speed.

This package provides:
- Fast ray-based occlusion testing for sun and sky exposure
- SolarCal-based MRT calculations using ladybug-comfort
- Grid generation for analysis surfaces
- Parallel processing for performance
- Progress tracking for long-running calculations
"""

from .mrt_calculator import MRTCalculator

__version__ = "0.1.0"
__all__ = ["MRTCalculator"]
