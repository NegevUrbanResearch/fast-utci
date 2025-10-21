"""
Unified weather data utilities for fast-utci.

This module consolidates weather data loading, filtering, and management
from multiple locations into a single source of truth shared by both
MRT and UTCI calculators.

Consolidates:
- model_reader.py::read_weather_data() - DataFrame loading
- mrt/adapters.py - Weather adapters (EPWAdapter, DataFrameAdapter)
- mrt/period.py::filter_weather_data() - Period/hour filtering
- utci/weather.py::WeatherDataManager - Weather management
"""

import numpy as np
import pandas as pd
from typing import Protocol, List, Union, Optional, Any, Tuple, Dict
from pathlib import Path
from datetime import datetime
import warnings

# Import period utilities for filtering
from fast_utci.mrt.period import AnalysisPeriod, create_hourly_mask


# Try to import ladybug for EPW handling
try:
    from ladybug.epw import EPW
    LADYBUG_AVAILABLE = True
except ImportError:
    warnings.warn("ladybug-core not available. EPW handling will be limited.")
    LADYBUG_AVAILABLE = False
    EPW = None


# ============================================================================
# Weather Data Adapters
# ============================================================================

class WeatherDataSource(Protocol):
    """
    Protocol for weather data sources.
    
    Defines the interface that all weather adapters must implement,
    enabling polymorphic weather data handling.
    """
    
    def get_temperature(self) -> np.ndarray:
        """Get air temperature array (C)."""
        ...
    
    def get_wind_speed(self) -> np.ndarray:
        """Get wind speed array (m/s)."""
        ...
    
    def get_relative_humidity(self) -> np.ndarray:
        """Get relative humidity array (%)."""
        ...
    
    def get_direct_radiation(self) -> np.ndarray:
        """Get direct normal radiation array (W/m^2)."""
        ...
    
    def get_diffuse_radiation(self) -> np.ndarray:
        """Get diffuse horizontal radiation array (W/m^2)."""
        ...
    
    def get_infrared_radiation(self) -> np.ndarray:
        """Get horizontal infrared radiation array (W/m^2)."""
        ...
    
    def get_datetimes(self) -> List:
        """Get datetime objects corresponding to data arrays."""
        ...


class EPWAdapter:
    """
    Adapter for Ladybug EPW weather data.
    
    Provides a clean interface to EPW data with filtering capabilities.
    Moved from mrt/adapters.py and enhanced with filtering methods.
    """
    
    def __init__(self, epw_data: Any):
        """
        Initialize EPW adapter.
        
        Args:
            epw_data: Ladybug EPW object
            
        Raises:
            ValueError: If EPW object is invalid
        """
        if not LADYBUG_AVAILABLE:
            raise ImportError("ladybug-core is required for EPW handling")
        
        if not hasattr(epw_data, 'dry_bulb_temperature'):
            raise ValueError("Invalid EPW object: missing dry_bulb_temperature")
        
        self.epw_data = epw_data
        self._filtered_indices = None  # For filtering support
    
    def get_temperature(self) -> np.ndarray:
        """Get air temperature array (C)."""
        temps = np.array(self.epw_data.dry_bulb_temperature.values)
        return self._apply_filter(temps)
    
    def get_wind_speed(self) -> np.ndarray:
        """Get wind speed array (m/s)."""
        wind = np.array(self.epw_data.wind_speed.values)
        return self._apply_filter(wind)
    
    def get_relative_humidity(self) -> np.ndarray:
        """Get relative humidity array (%)."""
        rh = np.array(self.epw_data.relative_humidity.values)
        return self._apply_filter(rh)
    
    def get_direct_radiation(self) -> np.ndarray:
        """Get direct normal radiation array (W/m^2)."""
        direct = np.array(self.epw_data.direct_normal_radiation.values)
        return self._apply_filter(direct)
    
    def get_diffuse_radiation(self) -> np.ndarray:
        """Get diffuse horizontal radiation array (W/m^2)."""
        diffuse = np.array(self.epw_data.diffuse_horizontal_radiation.values)
        return self._apply_filter(diffuse)
    
    def get_infrared_radiation(self) -> np.ndarray:
        """Get horizontal infrared radiation array (W/m^2)."""
        ir = np.array(self.epw_data.horizontal_infrared_radiation_intensity.values)
        return self._apply_filter(ir)
    
    def get_global_horizontal_radiation(self) -> np.ndarray:
        """Get global horizontal radiation array (W/m^2)."""
        ghr = np.array(self.epw_data.global_horizontal_radiation.values)
        return self._apply_filter(ghr)
    
    def get_datetimes(self) -> List:
        """Get datetime objects."""
        datetimes = self.epw_data.dry_bulb_temperature.datetimes
        if self._filtered_indices is not None:
            return [datetimes[i] for i in range(len(datetimes)) if self._filtered_indices[i]]
        return datetimes
    
    def filter_by_period(self, 
                        analysis_period: Optional[AnalysisPeriod] = None,
                        target_hours: Optional[List[int]] = None) -> 'EPWAdapter':
        """
        Filter weather data by analysis period and/or target hours.
        
        Args:
            analysis_period: Optional period filter (month/day ranges)
            target_hours: Optional hour filter (0-23)
            
        Returns:
            Self for method chaining
        """
        datetimes = self.epw_data.dry_bulb_temperature.datetimes
        self._filtered_indices = create_hourly_mask(datetimes, analysis_period, target_hours)
        return self
    
    def _apply_filter(self, array: np.ndarray) -> np.ndarray:
        """Apply filtering indices to an array."""
        if self._filtered_indices is not None:
            return array[self._filtered_indices]
        return array
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert EPW data to pandas DataFrame.
        
        Returns:
            DataFrame with weather variables
        """
        data = {
            'datetime': self.get_datetimes(),
            'air_temp': self.get_temperature(),
            'wind_speed': self.get_wind_speed(),
            'relative_humidity': self.get_relative_humidity(),
            'global_horizontal_radiation': self.get_global_horizontal_radiation(),
            'direct_normal_radiation': self.get_direct_radiation(),
            'diffuse_horizontal_radiation': self.get_diffuse_radiation(),
            'horizontal_infrared_radiation_intensity': self.get_infrared_radiation(),
            'surface_temp': self.get_temperature()  # Assume surface temp = air temp
        }
        return pd.DataFrame(data)
    
    def to_numpy_arrays(self) -> Dict[str, np.ndarray]:
        """
        Convert EPW data to numpy arrays dictionary.
        
        Returns:
            Dictionary of weather variable arrays
        """
        return {
            'air_temp': self.get_temperature(),
            'wind_speed': self.get_wind_speed(),
            'relative_humidity': self.get_relative_humidity(),
            'direct_normal_radiation': self.get_direct_radiation(),
            'diffuse_horizontal_radiation': self.get_diffuse_radiation(),
            'horizontal_infrared_radiation_intensity': self.get_infrared_radiation()
        }


class DataFrameAdapter:
    """
    Adapter for pandas DataFrame weather data.
    
    Provides a clean interface to DataFrame-based weather data with filtering.
    Moved from mrt/adapters.py and enhanced with filtering methods.
    """
    
    # Required columns for UTCI calculations
    REQUIRED_COLUMNS = {
        'air_temp', 'wind_speed', 'relative_humidity', 'datetime'
    }
    
    # Optional MRT-related columns
    OPTIONAL_COLUMNS = {
        'direct_normal_radiation',
        'diffuse_horizontal_radiation',
        'horizontal_infrared_radiation_intensity',
        'global_horizontal_radiation'
    }
    
    def __init__(self, df_data: pd.DataFrame):
        """
        Initialize DataFrame adapter.
        
        Args:
            df_data: pandas DataFrame with weather data columns
            
        Raises:
            ValueError: If required columns are missing
        """
        missing = self.REQUIRED_COLUMNS - set(df_data.columns)
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")
        
        self.df_data = df_data.copy()  # Avoid modifying original
        self._original_df = df_data  # Keep reference to original
    
    def get_temperature(self) -> np.ndarray:
        """Get air temperature array."""
        return self.df_data['air_temp'].values
    
    def get_wind_speed(self) -> np.ndarray:
        """Get wind speed array."""
        return self.df_data['wind_speed'].values
    
    def get_relative_humidity(self) -> np.ndarray:
        """Get relative humidity array."""
        return self.df_data['relative_humidity'].values
    
    def get_direct_radiation(self) -> np.ndarray:
        """Get direct normal radiation array."""
        if 'direct_normal_radiation' not in self.df_data.columns:
            warnings.warn("direct_normal_radiation not in DataFrame, returning zeros")
            return np.zeros(len(self.df_data))
        return self.df_data['direct_normal_radiation'].values
    
    def get_diffuse_radiation(self) -> np.ndarray:
        """Get diffuse horizontal radiation array."""
        if 'diffuse_horizontal_radiation' not in self.df_data.columns:
            warnings.warn("diffuse_horizontal_radiation not in DataFrame, returning zeros")
            return np.zeros(len(self.df_data))
        return self.df_data['diffuse_horizontal_radiation'].values
    
    def get_infrared_radiation(self) -> np.ndarray:
        """Get horizontal infrared radiation array."""
        if 'horizontal_infrared_radiation_intensity' not in self.df_data.columns:
            warnings.warn("horizontal_infrared_radiation_intensity not in DataFrame, returning zeros")
            return np.zeros(len(self.df_data))
        return self.df_data['horizontal_infrared_radiation_intensity'].values
    
    def get_datetimes(self) -> List:
        """Get datetime objects."""
        return self.df_data['datetime'].tolist()
    
    def filter_by_period(self,
                        analysis_period: Optional[AnalysisPeriod] = None,
                        target_hours: Optional[List[int]] = None) -> 'DataFrameAdapter':
        """
        Filter weather data by analysis period and/or target hours.
        
        Args:
            analysis_period: Optional period filter (month/day ranges)
            target_hours: Optional hour filter (0-23)
            
        Returns:
            Self for method chaining
        """
        datetimes = self._original_df['datetime'].tolist()
        mask = create_hourly_mask(datetimes, analysis_period, target_hours)
        self.df_data = self._original_df[mask].reset_index(drop=True)
        return self
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Get the underlying DataFrame.
        
        Returns:
            DataFrame with weather variables
        """
        return self.df_data.copy()
    
    def to_numpy_arrays(self) -> Dict[str, np.ndarray]:
        """
        Convert DataFrame data to numpy arrays dictionary.
        
        Returns:
            Dictionary of weather variable arrays
        """
        arrays = {
            'air_temp': self.get_temperature(),
            'wind_speed': self.get_wind_speed(),
            'relative_humidity': self.get_relative_humidity()
        }
        
        # Add optional columns if available
        if 'direct_normal_radiation' in self.df_data.columns:
            arrays['direct_normal_radiation'] = self.get_direct_radiation()
        if 'diffuse_horizontal_radiation' in self.df_data.columns:
            arrays['diffuse_horizontal_radiation'] = self.get_diffuse_radiation()
        if 'horizontal_infrared_radiation_intensity' in self.df_data.columns:
            arrays['horizontal_infrared_radiation_intensity'] = self.get_infrared_radiation()
        
        return arrays


def create_weather_adapter(weather_data: Any) -> Union[EPWAdapter, DataFrameAdapter]:
    """
    Factory function to create appropriate weather data adapter.
    
    Args:
        weather_data: EPW object or pandas DataFrame
        
    Returns:
        Appropriate adapter instance
        
    Raises:
        ValueError: If data type is not recognized
    """
    if hasattr(weather_data, 'dry_bulb_temperature'):
        return EPWAdapter(weather_data)
    elif hasattr(weather_data, 'columns'):
        return DataFrameAdapter(weather_data)
    else:
        raise ValueError(
            f"Unsupported weather data type: {type(weather_data)}. "
            "Expected EPW object or pandas DataFrame."
        )


# ============================================================================
# High-Level Weather Management
# ============================================================================

class WeatherDataManager:
    """
    High-level weather data manager for loading and filtering.
    
    Consolidates weather data loading logic from model_reader.py and
    utci/weather.py into a single unified interface.
    """
    
    def __init__(self, 
                 weather_source: Union[str, Path, pd.DataFrame, 'EPW'],
                 epw_object: Optional['EPW'] = None):
        """
        Initialize weather data manager.
        
        Args:
            weather_source: EPW file path, DataFrame, or EPW object
            epw_object: Optional pre-loaded EPW object (if weather_source is path)
        """
        self.weather_df = None
        self.epw_data = None
        self.adapter = None
        
        self._load_weather_data(weather_source, epw_object)
    
    def _load_weather_data(self,
                          weather_source: Union[str, Path, pd.DataFrame, 'EPW'],
                          epw_object: Optional['EPW'] = None):
        """Load weather data from various sources."""
        
        if isinstance(weather_source, (str, Path)):
            # Load from file
            self.weather_df, self.epw_data = load_weather_data(weather_source)
            if epw_object is not None:
                self.epw_data = epw_object  # Use provided EPW object
            self.adapter = EPWAdapter(self.epw_data)
            
        elif isinstance(weather_source, pd.DataFrame):
            # Use DataFrame directly
            self.weather_df = weather_source.copy()
            self.adapter = DataFrameAdapter(self.weather_df)
            
        elif hasattr(weather_source, 'dry_bulb_temperature'):
            # EPW object
            self.epw_data = weather_source
            self.adapter = EPWAdapter(self.epw_data)
            self.weather_df = self.adapter.to_dataframe()
            
        else:
            raise ValueError(f"Unsupported weather source type: {type(weather_source)}")
    
    def filter_by_period(self, analysis_period: Optional[AnalysisPeriod]) -> 'WeatherDataManager':
        """
        Filter weather data by analysis period.
        
        Args:
            analysis_period: Period filter (month/day ranges)
            
        Returns:
            Self for method chaining
        """
        if analysis_period is not None:
            # Store for combining with target_hours if called later
            self._pending_period = analysis_period
            self.adapter.filter_by_period(analysis_period=analysis_period, target_hours=None)
            self.weather_df = self.adapter.to_dataframe()
        return self
    
    def filter_by_hours(self, target_hours: Optional[List[int]]) -> 'WeatherDataManager':
        """
        Filter weather data by specific hours of day.
        
        Args:
            target_hours: List of hours (0-23) to include
            
        Returns:
            Self for method chaining
        """
        if target_hours is not None:
            # If period was already set, need to re-apply BOTH filters together
            # to avoid losing the period filter
            period = getattr(self, '_pending_period', None)
            self.adapter.filter_by_period(analysis_period=period, target_hours=target_hours)
            self.weather_df = self.adapter.to_dataframe()
        return self
    
    def to_dataframe(self) -> pd.DataFrame:
        """Get weather data as DataFrame."""
        return self.adapter.to_dataframe()
    
    def to_numpy_arrays(self) -> Dict[str, np.ndarray]:
        """Get weather data as numpy arrays dictionary."""
        return self.adapter.to_numpy_arrays()
    
    def get_adapter(self) -> Union[EPWAdapter, DataFrameAdapter]:
        """Get the underlying weather adapter."""
        return self.adapter


# ============================================================================
# Convenience Functions
# ============================================================================

def load_weather_data(file_path: Union[str, Path]) -> Tuple[pd.DataFrame, 'EPW']:
    """
    Load weather data from EPW file.
    
    Consolidates model_reader.py::read_weather_data() functionality.
    
    Args:
        file_path: Path to EPW weather file
        
    Returns:
        Tuple of (weather_df, epw_object)
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ImportError: If ladybug-core is not available
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Weather file not found: {file_path}")
    
    if not LADYBUG_AVAILABLE:
        raise ImportError("ladybug-core is required to load EPW files")
    
    # Load EPW
    epw = EPW(str(file_path))
    
    # Create adapter and convert to DataFrame
    adapter = EPWAdapter(epw)
    weather_df = adapter.to_dataframe()
    
    return weather_df, epw


def filter_weather_data(weather_df: pd.DataFrame,
                       analysis_period: Optional[AnalysisPeriod] = None,
                       target_hours: Optional[List[int]] = None) -> pd.DataFrame:
    """
    Filter weather DataFrame by analysis period and/or specific hours.
    
    Consolidates mrt/period.py::filter_weather_data() functionality.
    This function is kept for backward compatibility.
    
    Args:
        weather_df: Weather data DataFrame with 'datetime' column
        analysis_period: Optional period filter
        target_hours: Optional list of hours (0-23) to include
        
    Returns:
        Filtered weather DataFrame
    """
    if 'datetime' not in weather_df.columns:
        raise ValueError("Weather DataFrame must have 'datetime' column")
    
    # Use adapter for filtering
    adapter = DataFrameAdapter(weather_df)
    adapter.filter_by_period(analysis_period, target_hours)
    
    return adapter.to_dataframe()

