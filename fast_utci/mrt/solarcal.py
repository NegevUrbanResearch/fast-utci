"""
SolarCal MRT calculations using ladybug-comfort for parity with Grasshopper.

Implements the OutdoorSolarCal algorithms to compute Mean Radiant Temperature
from weather data and exposure fractions.
"""

import numpy as np
from typing import Optional, Dict, Any
from dataclasses import dataclass
import warnings

from ladybug_comfort.collection.solarcal import OutdoorSolarCal
from ladybug_comfort.parameter.solarcal import SolarCalParameter
from ladybug.datacollection import HourlyContinuousCollection, HourlyDiscontinuousCollection
from ladybug.datatype.temperature import Temperature
from ladybug.datatype.energyflux import EnergyFlux
from ladybug.datatype.fraction import Fraction
from ladybug.header import Header
from ladybug.location import Location
from ladybug.analysisperiod import AnalysisPeriod


@dataclass 
class SolarCalResult:
    """Container for SolarCal MRT calculation results."""
    mrt: np.ndarray           # Mean Radiant Temperature (°C)
    short_erf: np.ndarray     # Shortwave Effective Radiant Field (W/m²)
    long_erf: np.ndarray      # Longwave Effective Radiant Field (W/m²)
    short_dmrt: np.ndarray    # Shortwave Delta MRT (°C)
    long_dmrt: np.ndarray     # Longwave Delta MRT (°C)


def create_solar_body_parameters(absorptivity: float = 0.7,
                                emissivity: float = 0.95) -> SolarCalParameter:
    """
    Create SolarCalParameter object for human body characteristics.
    
    Args:
        absorptivity: Solar absorptivity of skin/clothing (0-1)
        emissivity: Longwave emissivity of skin/clothing (0-1)
        
    Returns:
        SolarCalParameter object
    """
    return SolarCalParameter(
        body_absorptivity=absorptivity,
        body_emissivity=emissivity
    )


def compute_mrt_solarcal(air_temperature: np.ndarray,
                        direct_normal_rad: np.ndarray, 
                        diffuse_horizontal_rad: np.ndarray,
                        horizontal_infrared_rad: np.ndarray,
                        fract_body_exp: np.ndarray,
                        sky_exposure: float,
                        location: Location,
                        datetimes: Any,
                        ground_reflectance: float = 0.25,
                        solar_body_par: Optional[SolarCalParameter] = None) -> SolarCalResult:
    """
    Compute MRT using ladybug-comfort OutdoorSolarCal for parity with Grasshopper.
    
    Args:
        air_temperature: Array of air temperatures (°C) - used as surface_temp proxy
        direct_normal_rad: Array of direct normal radiation (W/m²)
        diffuse_horizontal_rad: Array of diffuse horizontal radiation (W/m²)
        horizontal_infrared_rad: Array of horizontal infrared radiation (W/m²)
        fract_body_exp: Array of solar exposure fractions (0-1) per timestep
        sky_exposure: Scalar sky exposure fraction (0-1)
        location: Ladybug Location object
        datetimes: Datetime objects corresponding to data arrays
        ground_reflectance: Ground reflectance factor (0-1)
        solar_body_par: Optional SolarBodyParameter object
        
    Returns:
        SolarCalResult with MRT and component values
    """
    # Create SolarCal wrapper and compute
    wrapper = SolarCalWrapper(location, ground_reflectance, solar_body_par)
    return wrapper.compute_mrt(
        air_temperature, direct_normal_rad, diffuse_horizontal_rad,
        horizontal_infrared_rad, fract_body_exp, sky_exposure, datetimes
    )


class SolarCalWrapper:
    """Simplified wrapper for SolarCal calculations."""
    
    def __init__(self, location: Location, ground_reflectance: float = 0.25, 
                 solar_body_par: Optional[SolarCalParameter] = None):
        self.location = location
        self.ground_reflectance = ground_reflectance
        self.solar_body_par = solar_body_par or create_solar_body_parameters()
    
    def compute_mrt(self, air_temperature: np.ndarray, direct_normal_rad: np.ndarray,
                   diffuse_horizontal_rad: np.ndarray, horizontal_infrared_rad: np.ndarray,
                   fract_body_exp: np.ndarray, sky_exposure: float, datetimes: Any) -> SolarCalResult:
        """Compute MRT with simplified collection handling."""
        
        # Create analysis period header
        dt = datetimes[0] if hasattr(datetimes, '__len__') else datetimes
        analysis_period_header = AnalysisPeriod(dt.month, dt.day, 0, dt.month, dt.day, 23)
        
        # Create headers
        temp_header = Header(Temperature(), 'C', analysis_period_header, {'location': self.location})
        flux_header = Header(EnergyFlux(), 'W/m2', analysis_period_header, {'location': self.location})
        
        # Create collections - use discontinuous for single values
        is_single_value = len(air_temperature) == 1 if hasattr(air_temperature, '__len__') else True
        
        if is_single_value:
            air_temp_coll = HourlyDiscontinuousCollection(temp_header, air_temperature, [dt])
            dir_norm_coll = HourlyDiscontinuousCollection(flux_header, direct_normal_rad, [dt])
            diff_horiz_coll = HourlyDiscontinuousCollection(flux_header, diffuse_horizontal_rad, [dt])
            horiz_ir_coll = HourlyDiscontinuousCollection(flux_header, horizontal_infrared_rad, [dt])
        else:
            air_temp_coll = HourlyContinuousCollection(temp_header, air_temperature)
            dir_norm_coll = HourlyContinuousCollection(flux_header, direct_normal_rad)
            diff_horiz_coll = HourlyContinuousCollection(flux_header, diffuse_horizontal_rad)
            horiz_ir_coll = HourlyContinuousCollection(flux_header, horizontal_infrared_rad)
        
        # Handle fract_body_exp
        if np.isscalar(fract_body_exp) or (hasattr(fract_body_exp, '__len__') and len(np.unique(fract_body_exp)) == 1):
            fract_exp_input = float(fract_body_exp[0] if hasattr(fract_body_exp, '__len__') else fract_body_exp)
        else:
            fract_header = Header(Fraction(), 'fraction', self.location)
            fract_exp_input = HourlyContinuousCollection(fract_header, fract_body_exp)
        
        # Create SolarCal and compute
        solar_cal = OutdoorSolarCal(
            self.location, dir_norm_coll, diff_horiz_coll, horiz_ir_coll,
            air_temp_coll, fract_exp_input, sky_exposure, 
            self.ground_reflectance, self.solar_body_par
        )
        
        return SolarCalResult(
            mrt=np.array(solar_cal.mean_radiant_temperature.values),
            short_erf=np.array(solar_cal.shortwave_effective_radiant_field.values),
            long_erf=np.array(solar_cal.longwave_effective_radiant_field.values),
            short_dmrt=np.array(solar_cal.shortwave_mrt_delta.values),
            long_dmrt=np.array(solar_cal.longwave_mrt_delta.values)
        )


