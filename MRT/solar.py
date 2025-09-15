"""
Solar calculations for MRT computation.

Provides sunpath calculations and solar vector generation using Ladybug
to ensure parity with Grasshopper OutdoorSolarMRT component.
"""

import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass
from ladybug.location import Location
from ladybug.sunpath import Sunpath
from ladybug.dt import DateTime
import math


@dataclass
class SunData:
    """Container for sun position and visibility data."""
    sun_vectors: np.ndarray  # Shape: (n_hours, 3) - unit vectors pointing TO sun
    is_sun_up: np.ndarray    # Shape: (n_hours,) - boolean mask for sun-up hours
    solar_times: List[DateTime]  # Corresponding DateTime objects
    hoys: np.ndarray         # Hour of year indices
    

def get_sun_vectors(location: Location, 
                   analysis_period: Optional[Tuple[int, int, int, int]] = None,
                   north_degrees: float = 0.0) -> SunData:
    """
    Calculate sun vectors for all hours in analysis period using Ladybug sunpath.
    
    Args:
        location: Ladybug Location object with lat/lon/timezone
        analysis_period: Optional (start_month, start_day, end_month, end_day) 
                        defaults to full year
        north_degrees: North angle in degrees (0 = Y+ is north)
        
    Returns:
        SunData object with sun vectors, visibility mask, and timing info
    """
    sunpath = Sunpath.from_location(location)
    
    # Generate all hours for analysis period
    if analysis_period is None:
        # Full year: January 1 to December 31
        start_dt = DateTime(1, 1, 0)
        end_dt = DateTime(12, 31, 23)
    else:
        start_month, start_day, end_month, end_day = analysis_period
        start_dt = DateTime(start_month, start_day, 0)
        end_dt = DateTime(end_month, end_day, 23)
    
    # Generate hourly datetimes
    solar_times = []
    current_dt = start_dt
    while current_dt <= end_dt:
        solar_times.append(current_dt)
        current_dt = current_dt.add_hour(1)
    
    # Calculate sun positions and vectors
    sun_vectors = []
    is_sun_up = []
    hoys = []
    
    for dt in solar_times:
        sun = sunpath.calculate_sun_from_date_time(dt)
        hoys.append(dt.hoy)
        
        if sun.is_during_day:
            # Convert altitude/azimuth to Cartesian vector pointing TO sun
            alt_rad = math.radians(sun.altitude)
            azi_rad = math.radians(sun.azimuth)
            
            # Standard spherical to Cartesian conversion
            # Azimuth: 0° = North, 90° = East, 180° = South, 270° = West
            x = math.sin(azi_rad) * math.cos(alt_rad)  # East component
            y = math.cos(azi_rad) * math.cos(alt_rad)  # North component  
            z = math.sin(alt_rad)                      # Up component
            
            # Apply north rotation if specified
            if north_degrees != 0.0:
                north_rad = math.radians(north_degrees)
                cos_n = math.cos(north_rad)
                sin_n = math.sin(north_rad)
                
                # Rotate in XY plane
                x_rot = x * cos_n - y * sin_n
                y_rot = x * sin_n + y * cos_n
                x, y = x_rot, y_rot
            
            sun_vectors.append([x, y, z])
            is_sun_up.append(True)
        else:
            # Sun is down - use zero vector
            sun_vectors.append([0.0, 0.0, 0.0])
            is_sun_up.append(False)
    
    return SunData(
        sun_vectors=np.array(sun_vectors),
        is_sun_up=np.array(is_sun_up),
        solar_times=solar_times,
        hoys=np.array(hoys)
    )


def get_tregenza_dome_vectors() -> Tuple[np.ndarray, np.ndarray]:
    """
    Get Tregenza sky dome patch vectors and weights for sky exposure calculation.
    
    Returns:
        Tuple of (vectors, weights) where:
        - vectors: shape (145, 3) unit vectors pointing to sky patches
        - weights: shape (145,) solid angle weights for each patch
    """
    from ladybug.viewsphere import view_sphere
    
    # Get Tregenza dome vectors and weights
    tregenza_vecs = view_sphere.tregenza_dome_vectors
    tregenza_weights = view_sphere.tregenza_solid_angles
    
    # Convert to numpy arrays
    vectors = np.array([[v.x, v.y, v.z] for v in tregenza_vecs])
    weights = np.array(tregenza_weights)
    
    return vectors, weights


def filter_hours_by_local_time(sun_data: SunData, 
                              target_hours: List[int]) -> SunData:
    """
    Filter sun data to only include specific local hours.
    
    Args:
        sun_data: SunData object to filter
        target_hours: List of hours (0-23) to keep
        
    Returns:
        Filtered SunData object
    """
    # Extract local hours from DateTime objects
    local_hours = np.array([dt.hour for dt in sun_data.solar_times])
    
    # Create mask for target hours
    hour_mask = np.isin(local_hours, target_hours)
    
    # Apply filter
    return SunData(
        sun_vectors=sun_data.sun_vectors[hour_mask],
        is_sun_up=sun_data.is_sun_up[hour_mask],
        solar_times=[dt for i, dt in enumerate(sun_data.solar_times) if hour_mask[i]],
        hoys=sun_data.hoys[hour_mask]
    )
