"""
Analysis period filtering utilities for MRT calculations.

Provides functionality to filter weather data and results to specific
time periods, matching Grasshopper Analysis Period components.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class AnalysisPeriod:
    """Container for analysis period definition."""
    start_month: int
    start_day: int  
    start_hour: int
    end_month: int
    end_day: int
    end_hour: int
    
    def __post_init__(self):
        """Validate analysis period parameters."""
        if not (1 <= self.start_month <= 12):
            raise ValueError(f"Invalid start_month: {self.start_month}")
        if not (1 <= self.start_day <= 31):
            raise ValueError(f"Invalid start_day: {self.start_day}")
        if not (0 <= self.start_hour <= 23):
            raise ValueError(f"Invalid start_hour: {self.start_hour}")
        if not (1 <= self.end_month <= 12):
            raise ValueError(f"Invalid end_month: {self.end_month}")
        if not (1 <= self.end_day <= 31):
            raise ValueError(f"Invalid end_day: {self.end_day}")
        if not (0 <= self.end_hour <= 23):
            raise ValueError(f"Invalid end_hour: {self.end_hour}")


def create_analysis_period(start_month: int, start_day: int, 
                          end_month: int, end_day: int,
                          start_hour: int = 0, end_hour: int = 23) -> AnalysisPeriod:
    """
    Create analysis period specification.
    
    Args:
        start_month: Start month (1-12)
        start_day: Start day (1-31)
        end_month: End month (1-12)  
        end_day: End day (1-31)
        start_hour: Start hour (0-23), default 0
        end_hour: End hour (0-23), default 23
        
    Returns:
        AnalysisPeriod object
    """
    return AnalysisPeriod(
        start_month=start_month,
        start_day=start_day,
        start_hour=start_hour,
        end_month=end_month,
        end_day=end_day,
        end_hour=end_hour
    )


def create_hourly_mask(datetimes: List,
                      analysis_period: Optional[AnalysisPeriod] = None,
                      target_hours: Optional[List[int]] = None) -> np.ndarray:
    """
    Create boolean mask for filtering data to analysis period and/or specific hours.
    
    Args:
        datetimes: List of datetime objects (from EPW or solar calculations)
        analysis_period: Optional period filter
        target_hours: Optional list of hours (0-23) to include
        
    Returns:
        Boolean mask array for filtering
    """
    mask = np.ones(len(datetimes), dtype=bool)
    
    # Apply analysis period filter
    if analysis_period is not None:
        period_mask = np.zeros(len(datetimes), dtype=bool)
        
        for i, dt in enumerate(datetimes):
            # Check if datetime falls within analysis period
            dt_month = dt.month
            dt_day = dt.day  
            dt_hour = dt.hour
            
            # Handle year-spanning periods
            if analysis_period.start_month <= analysis_period.end_month:
                # Within same year
                in_month_range = (analysis_period.start_month <= dt_month <= analysis_period.end_month)
            else:
                # Spans year boundary
                in_month_range = (dt_month >= analysis_period.start_month) or (dt_month <= analysis_period.end_month)
            
            # Check day boundaries
            in_day_range = True
            if dt_month == analysis_period.start_month:
                in_day_range = in_day_range and (dt_day >= analysis_period.start_day)
            if dt_month == analysis_period.end_month:
                in_day_range = in_day_range and (dt_day <= analysis_period.end_day)
                
            # Check hour boundaries
            in_hour_range = True
            if (dt_month == analysis_period.start_month and 
                dt_day == analysis_period.start_day):
                in_hour_range = in_hour_range and (dt_hour >= analysis_period.start_hour)
            if (dt_month == analysis_period.end_month and 
                dt_day == analysis_period.end_day):
                in_hour_range = in_hour_range and (dt_hour <= analysis_period.end_hour)
                
            period_mask[i] = in_month_range and in_day_range and in_hour_range
            
        mask = mask & period_mask
    
    # Apply hourly filter
    if target_hours is not None:
        hour_mask = np.array([dt.hour in target_hours for dt in datetimes])
        mask = mask & hour_mask
    
    return mask


def filter_weather_data(weather_df: pd.DataFrame,
                       analysis_period: Optional[AnalysisPeriod] = None,
                       target_hours: Optional[List[int]] = None) -> pd.DataFrame:
    """
    Filter weather dataframe to analysis period and/or specific hours.
    
    Args:
        weather_df: Weather data DataFrame with 'datetime' column
        analysis_period: Optional period filter
        target_hours: Optional list of hours (0-23) to include
        
    Returns:
        Filtered weather DataFrame
    """
    if 'datetime' not in weather_df.columns:
        raise ValueError("Weather DataFrame must have 'datetime' column")
    
    # Convert datetime column to list for mask creation
    datetimes = weather_df['datetime'].tolist()
    
    # Create filter mask
    mask = create_hourly_mask(datetimes, analysis_period, target_hours)
    
    # Apply filter
    return weather_df[mask].reset_index(drop=True)


def filter_arrays_by_period(arrays_dict: dict,
                           datetimes: List,
                           analysis_period: Optional[AnalysisPeriod] = None,
                           target_hours: Optional[List[int]] = None) -> Tuple[dict, List]:
    """
    Filter multiple arrays and datetime list by analysis period.
    
    Args:
        arrays_dict: Dictionary of {name: array} to filter
        datetimes: List of datetime objects corresponding to arrays
        analysis_period: Optional period filter
        target_hours: Optional list of hours (0-23) to include
        
    Returns:
        Tuple of (filtered_arrays_dict, filtered_datetimes)
    """
    # Create filter mask
    mask = create_hourly_mask(datetimes, analysis_period, target_hours)
    
    # Filter arrays
    filtered_arrays = {}
    for name, array in arrays_dict.items():
        if hasattr(array, '__len__') and len(array) == len(datetimes):
            filtered_arrays[name] = np.asarray(array)[mask]
        else:
            # Keep scalar values unchanged
            filtered_arrays[name] = array
    
    # Filter datetimes
    filtered_datetimes = [dt for i, dt in enumerate(datetimes) if mask[i]]
    
    return filtered_arrays, filtered_datetimes


def get_august_15_analysis_period() -> AnalysisPeriod:
    """
    Get analysis period for August 15th (for validation testing).
    
    Returns:
        AnalysisPeriod for August 15th, all hours
    """
    return create_analysis_period(
        start_month=8, start_day=15,
        end_month=8, end_day=15,
        start_hour=0, end_hour=23
    )


def get_hour_13_14_filter() -> List[int]:
    """
    Get hour filter for hour 13 (1-2 PM) for validation testing.
    
    The file name "13_14" refers to the time period 13:00-14:00 (1-2 PM),
    which is represented by hour 13 in the EPW data.
    
    Returns:
        List of hours [13]
    """
    return [13]  # Hour 13 represents the 1-2 PM period (13:00-14:00)


def create_validation_period_filter() -> Tuple[AnalysisPeriod, List[int]]:
    """
    Create analysis period and hour filter for validation against GH data.
    
    Based on the validation file data/15th_aug_13_14_MRT.csv, this should
    filter to August 15th, hour 13 (1-2 PM local time).
    
    Returns:
        Tuple of (analysis_period, target_hours)
    """
    period = get_august_15_analysis_period()
    hours = get_hour_13_14_filter()
    return period, hours
