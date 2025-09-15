"""
SolarCal MRT calculations using ladybug-comfort for parity with Grasshopper.

Implements the OutdoorSolarCal algorithms to compute Mean Radiant Temperature
from weather data and exposure fractions.
"""

import numpy as np
from typing import Optional, Dict, Any
from dataclasses import dataclass
import warnings

try:
    from ladybug_comfort.collection.solarcal import OutdoorSolarCal
    from ladybug_comfort.parameter.solarcal import SolarCalParameter
    from ladybug.datacollection import HourlyContinuousCollection
    from ladybug.datatype.temperature import Temperature
    from ladybug.datatype.energyflux import EnergyFlux
    from ladybug.header import Header
    from ladybug.location import Location
    LADYBUG_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"ladybug-comfort not available: {e}. SolarCal will be limited.")
    LADYBUG_AVAILABLE = False
    # Create dummy classes for type hints when ladybug not available
    class Location: pass
    class SolarCalParameter: pass


@dataclass 
class SolarCalResult:
    """Container for SolarCal MRT calculation results."""
    mrt: np.ndarray           # Mean Radiant Temperature (°C)
    short_erf: np.ndarray     # Shortwave Effective Radiant Field (W/m²)
    long_erf: np.ndarray      # Longwave Effective Radiant Field (W/m²)
    short_dmrt: np.ndarray    # Shortwave Delta MRT (°C)
    long_dmrt: np.ndarray     # Longwave Delta MRT (°C)


def create_solar_body_parameters(absorptivity: float = 0.7,
                                emissivity: float = 0.95) -> Optional[Any]:
    """
    Create SolarCalParameter object for human body characteristics.
    
    Args:
        absorptivity: Solar absorptivity of skin/clothing (0-1)
        emissivity: Longwave emissivity of skin/clothing (0-1)
        
    Returns:
        SolarCalParameter object or None if ladybug-comfort not available
    """
    if not LADYBUG_AVAILABLE:
        return None
        
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
                        solar_body_par: Optional[Any] = None,
                        epw_data: Optional[Any] = None) -> SolarCalResult:
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
    if not LADYBUG_AVAILABLE:
        # Fallback: return air temperature as MRT
        warnings.warn("ladybug-comfort not available. Using air temperature as MRT.")
        n_hours = len(air_temperature)
        return SolarCalResult(
            mrt=air_temperature.copy(),
            short_erf=np.zeros(n_hours),
            long_erf=np.zeros(n_hours),
            short_dmrt=np.zeros(n_hours),
            long_dmrt=np.zeros(n_hours)
        )
    
    try:
        # Create analysis period for the headers
        from ladybug.analysisperiod import AnalysisPeriod
        
        # For proper SolarCal operation, we need to provide full daily data
        # and let SolarCal handle the filtering, just like Grasshopper does
        dt = datetimes[0]  # Use first datetime to determine the day
        analysis_period_header = AnalysisPeriod(dt.month, dt.day, 0, dt.month, dt.day, 23)
        
        # Create ladybug data collections
        temp_header = Header(Temperature(), 'C', analysis_period_header, {'location': location})
        flux_header = Header(EnergyFlux(), 'W/m2', analysis_period_header, {'location': location})
        
        # For single hour data, use HourlyDiscontinuousCollection instead
        from ladybug.datacollection import HourlyDiscontinuousCollection
        if len(datetimes) == 1:
            air_temp_coll = HourlyDiscontinuousCollection(temp_header, air_temperature, datetimes)
            dir_norm_coll = HourlyDiscontinuousCollection(flux_header, direct_normal_rad, datetimes)
            diff_horiz_coll = HourlyDiscontinuousCollection(flux_header, diffuse_horizontal_rad, datetimes)
            horiz_ir_coll = HourlyDiscontinuousCollection(flux_header, horizontal_infrared_rad, datetimes)
        else:
            air_temp_coll = HourlyContinuousCollection(temp_header, air_temperature)
            dir_norm_coll = HourlyContinuousCollection(flux_header, direct_normal_rad)
            diff_horiz_coll = HourlyContinuousCollection(flux_header, diffuse_horizontal_rad)
            horiz_ir_coll = HourlyContinuousCollection(flux_header, horizontal_infrared_rad)
        
        # Handle fract_body_exp - can be scalar or time series
        if np.isscalar(fract_body_exp) or len(np.unique(fract_body_exp)) == 1:
            # Use scalar value
            fract_exp_input = float(fract_body_exp[0] if hasattr(fract_body_exp, '__len__') else fract_body_exp)
        else:
            # Use time series - create collection
            from ladybug.datatype.fraction import Fraction
            fract_header = Header(Fraction(), 'fraction', location)
            fract_exp_input = HourlyContinuousCollection(fract_header, fract_body_exp, datetimes)
        
        # Create default body parameters if not provided
        if solar_body_par is None:
            solar_body_par = create_solar_body_parameters()
        
        # Create OutdoorSolarCal object
        solar_cal = OutdoorSolarCal(
            location,
            dir_norm_coll,
            diff_horiz_coll,
            horiz_ir_coll,
            air_temp_coll,        # surface_temperatures
            fract_exp_input,      # fraction_body_exposed
            sky_exposure,         # sky_exposure
            ground_reflectance,   # floor_reflectance
            solar_body_par        # solarcal_body_parameter
        )
        
        # Extract results
        mrt_values = np.array(solar_cal.mean_radiant_temperature.values)
        short_erf_values = np.array(solar_cal.shortwave_effective_radiant_field.values)
        long_erf_values = np.array(solar_cal.longwave_effective_radiant_field.values) 
        short_dmrt_values = np.array(solar_cal.shortwave_mrt_delta.values)
        long_dmrt_values = np.array(solar_cal.longwave_mrt_delta.values)
        
        return SolarCalResult(
            mrt=mrt_values,
            short_erf=short_erf_values,
            long_erf=long_erf_values,
            short_dmrt=short_dmrt_values,
            long_dmrt=long_dmrt_values
        )
        
    except Exception as e:
        warnings.warn(f"SolarCal calculation failed: {e}. Using air temperature as MRT.")
        n_hours = len(air_temperature)
        return SolarCalResult(
            mrt=air_temperature.copy(),
            short_erf=np.zeros(n_hours),
            long_erf=np.zeros(n_hours), 
            short_dmrt=np.zeros(n_hours),
            long_dmrt=np.zeros(n_hours)
        )


def simple_mrt_approximation(air_temperature: np.ndarray,
                           direct_normal_rad: np.ndarray,
                           diffuse_horizontal_rad: np.ndarray,
                           fract_body_exp: np.ndarray,
                           absorptivity: float = 0.7) -> np.ndarray:
    """
    Simple MRT approximation for cases where ladybug-comfort is not available.
    
    This is a basic approximation and should not be used for final results.
    The proper SolarCal implementation is needed for accuracy.
    
    Args:
        air_temperature: Air temperature array (°C)
        direct_normal_rad: Direct normal radiation (W/m²)
        diffuse_horizontal_rad: Diffuse horizontal radiation (W/m²)
        fract_body_exp: Solar exposure fraction (0-1)
        absorptivity: Solar absorptivity (0-1)
        
    Returns:
        Approximate MRT values (°C)
    """
    # Very simple approximation: add solar heating to air temperature
    # This is NOT accurate and only for fallback when ladybug is unavailable
    
    # Estimate solar heating effect
    total_solar = direct_normal_rad * fract_body_exp + diffuse_horizontal_rad * 0.5
    solar_heating = absorptivity * total_solar / 20.0  # Rough conversion to °C
    
    return air_temperature + solar_heating
