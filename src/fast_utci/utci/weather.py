"""
Weather data management for UTCI calculations.

Handles loading, filtering, and format conversion of weather data from
various sources (EPW files, DataFrames, etc.).

This module is now a thin wrapper around fast_utci.shared.weather.WeatherDataManager,
adding UTCI-specific functionality like time column management.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, List, Dict, Any
from pathlib import Path
import warnings

# Delegate to shared weather utilities
from fast_utci.shared.weather import WeatherDataManager as SharedWeatherManager

try:
    from ladybug.epw import EPW
    LADYBUG_AVAILABLE = True
except ImportError:
    warnings.warn("ladybug-core not available. EPW handling will be limited.")
    LADYBUG_AVAILABLE = False
    EPW = None


class WeatherDataManager:
    """
    UTCI-specific weather data manager.
    
    Thin wrapper around shared.weather.WeatherDataManager that adds
    UTCI-specific functionality like time column management and
    location information access.
    """
    
    def __init__(self, 
                 weather_source: Union[str, Path, pd.DataFrame, 'EPW'],
                 epw_object: Optional['EPW'] = None):
        """
        Initialize weather data manager.
        
        Args:
            weather_source: Weather data as file path, DataFrame, or EPW object
            epw_object: Optional EPW object for location info if weather_source is DataFrame
        """
        # Delegate to shared weather manager
        self._shared_manager = SharedWeatherManager(weather_source, epw_object)
        self.epw_data = self._shared_manager.epw_data
        self.weather_df = self._shared_manager.weather_df
        
        # Add UTCI-specific time columns
        self._add_time_columns()
    
    def _add_time_columns(self) -> None:
        """Add time columns to weather DataFrame if not present."""
        if 'hour' not in self.weather_df.columns:
            self.weather_df['hour'] = [dt.hour for dt in self.weather_df['datetime']]
        if 'month' not in self.weather_df.columns:
            self.weather_df['month'] = [dt.month for dt in self.weather_df['datetime']]
        if 'day' not in self.weather_df.columns:
            self.weather_df['day'] = [dt.day for dt in self.weather_df['datetime']]
    
    def filter_by_period(self, analysis_period: Optional[Any]) -> 'WeatherDataManager':
        """
        Filter weather data by analysis period.
        
        Args:
            analysis_period: Analysis period object
            
        Returns:
            New WeatherDataManager with filtered data
        """
        if analysis_period is None:
            return self
        
        # Delegate to shared manager
        self._shared_manager.filter_by_period(analysis_period)
        
        # Update local references
        self.weather_df = self._shared_manager.to_dataframe()
        self._add_time_columns()  # Re-add time columns after filtering
        
        return self
    
    def filter_by_hours(self, target_hours: Optional[List[int]]) -> 'WeatherDataManager':
        """
        Filter weather data by specific hours.
        
        Args:
            target_hours: List of hours (0-23) to keep, or None for all hours
            
        Returns:
            New WeatherDataManager with filtered data
        """
        if target_hours is None:
            return self
        
        # Delegate to shared manager
        self._shared_manager.filter_by_hours(target_hours)
        
        # Update local references
        self.weather_df = self._shared_manager.to_dataframe()
        self._add_time_columns()  # Re-add time columns after filtering
        
        return self
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Get weather data as pandas DataFrame.
        
        Returns:
            DataFrame with weather data
        """
        return self.weather_df.copy()
    
    def to_numpy_arrays(self) -> Dict[str, np.ndarray]:
        """
        Get weather data as dictionary of numpy arrays.
        
        Useful for vectorized calculations. Creates zero-copy views when possible.
        
        Returns:
            Dictionary with 'air_temp', 'wind_speed', 'relative_humidity', 'datetime' arrays
        """
        # Delegate to shared manager for core arrays
        arrays = self._shared_manager.to_numpy_arrays()
        
        # Add datetime array if present
        if 'datetime' in self.weather_df.columns:
            arrays['datetime'] = self.weather_df['datetime'].to_numpy(copy=False)
        
        return arrays
    
    def get_location_info(self) -> Optional[str]:
        """
        Get location information if available.
        
        Returns:
            Location string or None
        """
        if self.epw_data:
            return str(self.epw_data.location)
        return None
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for weather data.
        
        Returns:
            Dictionary with summary information
        """
        return {
            'n_hours': len(self.weather_df),
            'temp_range': (self.weather_df['air_temp'].min(), self.weather_df['air_temp'].max()),
            'wind_range': (self.weather_df['wind_speed'].min(), self.weather_df['wind_speed'].max()),
            'humidity_range': (self.weather_df['relative_humidity'].min(), self.weather_df['relative_humidity'].max()),
            'location': self.get_location_info()
        }
    
    def __len__(self) -> int:
        """Get number of hours in weather data."""
        return len(self.weather_df)
    
    def __repr__(self) -> str:
        """String representation."""
        location = self.get_location_info() or "Unknown"
        return f"WeatherDataManager({len(self)} hours, location={location})"

