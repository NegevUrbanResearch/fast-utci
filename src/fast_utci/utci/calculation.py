"""
Core UTCI calculation logic with boundary averaging.

This module provides the single source of truth for UTCI calculations,
eliminating code duplication across the codebase.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass
import warnings

try:
    from pythermalcomfort.models import utci
    PYTHERMALCOMFORT_AVAILABLE = True
except ImportError:
    warnings.warn("pythermalcomfort not available. UTCI calculations will be limited.")
    PYTHERMALCOMFORT_AVAILABLE = False


@dataclass
class UTCICalculationResult:
    """
    Typed container for UTCI calculation results.
    
    Provides structured access to UTCI values and associated metadata,
    replacing plain dictionary returns for better type safety.
    """
    position: Tuple[float, float, float]
    utci: np.ndarray
    mrt0: np.ndarray
    mrt1: np.ndarray
    air_temp: Optional[np.ndarray] = None
    wind_speed: Optional[np.ndarray] = None
    relative_humidity: Optional[np.ndarray] = None
    datetime: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for backward compatibility.
        
        Returns:
            Dictionary with all non-None fields
        """
        result = {
            'position': self.position,
            'utci': self.utci,
            'mrt0': self.mrt0,
            'mrt1': self.mrt1,
            'mrt': self.mrt0,  # Backward compatibility
        }
        
        if self.air_temp is not None:
            result['air_temp'] = self.air_temp
        if self.wind_speed is not None:
            result['wind_speed'] = self.wind_speed
        if self.relative_humidity is not None:
            result['relative_humidity'] = self.relative_humidity
        if self.datetime is not None:
            result['datetime'] = self.datetime
            
        return result


class BoundaryAveragingCalculator:
    """
    Single source of truth for boundary averaging UTCI calculations.
    
    Implements the boundary averaging algorithm:
    For each hour N:
    - Calculate UTCI0 using (mrt0[N], weather[N])
    - Calculate UTCI1 using (mrt1[N], weather[N+1])
    - Average: utci_avg = (utci0 + utci1) / 2
    
    Supports both vectorized (numpy) and iterative (DataFrame) calculation modes.
    """
    
    def __init__(self, enable_vectorized: bool = True):
        """
        Initialize calculator.
        
        Args:
            enable_vectorized: Whether to use vectorized calculations when possible
        """
        if not PYTHERMALCOMFORT_AVAILABLE:
            raise RuntimeError("pythermalcomfort required for UTCI calculations")
        
        self.enable_vectorized = enable_vectorized
    
    def calculate(self, 
                  mrt_data: Dict[str, Any],
                  weather_data: Union[pd.DataFrame, Dict[str, np.ndarray]],
                  include_weather: bool = True,
                  include_datetime: bool = True) -> UTCICalculationResult:
        """
        Calculate UTCI for a single position using boundary averaging.
        
        Args:
            mrt_data: MRT data dictionary with 'position', 'mrt0', 'mrt1'
            weather_data: Weather data as DataFrame or dict of numpy arrays
            include_weather: Whether to include weather variables in result
            include_datetime: Whether to include datetime in result
            
        Returns:
            UTCICalculationResult with UTCI values and metadata
        """
        # Choose calculation path
        if self.enable_vectorized and isinstance(weather_data, dict):
            return self._calculate_vectorized(mrt_data, weather_data, include_weather, include_datetime)
        else:
            return self._calculate_iterative(mrt_data, weather_data, include_weather, include_datetime)
    
    def _calculate_vectorized(self,
                             mrt_data: Dict[str, Any],
                             weather_arrays: Dict[str, np.ndarray],
                             include_weather: bool,
                             include_datetime: bool) -> UTCICalculationResult:
        """
        Vectorized UTCI calculation using numpy arrays.
        
        Args:
            mrt_data: MRT data with position, mrt0, mrt1
            weather_arrays: Dict with air_temp, wind_speed, relative_humidity arrays
            include_weather: Include weather in result
            include_datetime: Include datetime in result
            
        Returns:
            UTCICalculationResult
        """
        # Extract MRT values
        mrt0_values = np.asarray(mrt_data.get('mrt0', mrt_data.get('mrt')))
        mrt1_values = np.asarray(mrt_data.get('mrt1', mrt0_values))
        position = mrt_data['position']
        n_hours = len(mrt0_values)
        
        # Extract weather arrays
        air = weather_arrays['air_temp']
        wind = weather_arrays['wind_speed']
        rh = weather_arrays['relative_humidity']
        dts = weather_arrays.get('datetime')
        
        # Prepare weather slices for boundary averaging
        air_n, wind_n, rh_n, air_n1, wind_n1, rh_n1 = self._prepare_weather_slices(
            air, wind, rh, n_hours
        )
        
        try:
            # UTCI0: using mrt0 and weather[N]
            utci0_vals = utci(tdb=air_n, tr=mrt0_values, v=wind_n, rh=rh_n, round_output=False)
            utci0_array = self._extract_utci_value(utci0_vals)
            
            # UTCI1: using mrt1 and weather[N+1]
            utci1_vals = utci(tdb=air_n1, tr=mrt1_values, v=wind_n1, rh=rh_n1, round_output=False)
            utci1_array = self._extract_utci_value(utci1_vals)
            
            # Average the two UTCI values for boundary averaging
            utci_array = (utci0_array + utci1_array) / 2.0
            
        except Exception as e:
            warnings.warn(f"Vectorized UTCI calculation failed; falling back to loop: {e}")
            # Fallback to iterative
            df = pd.DataFrame({
                'air_temp': air,
                'wind_speed': wind,
                'relative_humidity': rh
            })
            if dts is not None:
                df['datetime'] = dts
            return self._calculate_iterative(mrt_data, df, include_weather, include_datetime)
        
        # Build result
        return UTCICalculationResult(
            position=position,
            utci=utci_array,
            mrt0=mrt0_values,
            mrt1=mrt1_values,
            air_temp=air_n if include_weather else None,
            wind_speed=wind_n if include_weather else None,
            relative_humidity=rh_n if include_weather else None,
            datetime=dts[:n_hours] if include_datetime and dts is not None and len(dts) >= n_hours else None
        )
    
    def _calculate_iterative(self,
                            mrt_data: Dict[str, Any],
                            weather_df: pd.DataFrame,
                            include_weather: bool,
                            include_datetime: bool) -> UTCICalculationResult:
        """
        Iterative UTCI calculation using pandas DataFrame.
        
        Args:
            mrt_data: MRT data with position, mrt0, mrt1
            weather_df: DataFrame with air_temp, wind_speed, relative_humidity columns
            include_weather: Include weather in result
            include_datetime: Include datetime in result
            
        Returns:
            UTCICalculationResult
        """
        # Extract MRT values
        mrt0_values = mrt_data.get('mrt0', mrt_data.get('mrt'))
        mrt1_values = mrt_data.get('mrt1', mrt0_values)
        position = mrt_data['position']
        n_hours = len(mrt0_values)
        
        # Need N+1 hours of weather data for boundary averaging
        weather_subset = weather_df.iloc[:min(n_hours + 1, len(weather_df))].copy()
        
        # Pad if needed
        if len(weather_subset) < n_hours + 1:
            last_weather = weather_subset.iloc[-1:] if len(weather_subset) > 0 else weather_df.iloc[:1]
            while len(weather_subset) < n_hours + 1:
                weather_subset = pd.concat([weather_subset, last_weather], ignore_index=True)
        
        # Calculate UTCI for each hour with boundary averaging
        utci_values = []
        for i in range(n_hours):
            try:
                # Calculate UTCI0 using mrt0 and weather[i]
                utci0_result = utci(
                    tdb=weather_subset.iloc[i]['air_temp'],
                    tr=mrt0_values[i],
                    v=weather_subset.iloc[i]['wind_speed'],
                    rh=weather_subset.iloc[i]['relative_humidity'],
                    round_output=False
                )
                utci0 = self._extract_utci_value(utci0_result)
                
                # Calculate UTCI1 using mrt1 and weather[i+1]
                utci1_result = utci(
                    tdb=weather_subset.iloc[i+1]['air_temp'],
                    tr=mrt1_values[i],
                    v=weather_subset.iloc[i+1]['wind_speed'],
                    rh=weather_subset.iloc[i+1]['relative_humidity'],
                    round_output=False
                )
                utci1 = self._extract_utci_value(utci1_result)
                
                # Average the two UTCI values for boundary averaging
                utci_avg = (utci0 + utci1) / 2.0
                utci_values.append(utci_avg)
            except Exception as e:
                warnings.warn(f"UTCI calculation failed for hour {i}: {e}")
                utci_values.append(np.nan)
        
        # Build result
        return UTCICalculationResult(
            position=position,
            utci=np.array(utci_values),
            mrt0=mrt0_values,
            mrt1=mrt1_values,
            air_temp=weather_subset['air_temp'].values[:n_hours] if include_weather else None,
            wind_speed=weather_subset['wind_speed'].values[:n_hours] if include_weather else None,
            relative_humidity=weather_subset['relative_humidity'].values[:n_hours] if include_weather else None,
            datetime=weather_subset['datetime'].values[:n_hours] if include_datetime and 'datetime' in weather_subset.columns else None
        )
    
    @staticmethod
    def _prepare_weather_slices(air: np.ndarray, 
                                wind: np.ndarray, 
                                rh: np.ndarray,
                                n_hours: int) -> Tuple[np.ndarray, ...]:
        """
        Prepare weather array slices for boundary averaging.
        
        Returns slices for hour N and hour N+1, with padding if necessary.
        
        Args:
            air: Air temperature array
            wind: Wind speed array
            rh: Relative humidity array
            n_hours: Number of hours needed
            
        Returns:
            Tuple of (air_n, wind_n, rh_n, air_n1, wind_n1, rh_n1)
        """
        # Slice for hour N (utci0)
        air_n = air[:n_hours] if n_hours <= air.shape[0] else air[:air.shape[0]]
        wind_n = wind[:n_hours] if n_hours <= wind.shape[0] else wind[:wind.shape[0]]
        rh_n = rh[:n_hours] if n_hours <= rh.shape[0] else rh[:rh.shape[0]]
        
        # Slice for hour N+1 (utci1)
        air_n1 = air[1:n_hours+1] if n_hours+1 <= air.shape[0] else air[1:air.shape[0]]
        wind_n1 = wind[1:n_hours+1] if n_hours+1 <= wind.shape[0] else wind[1:wind.shape[0]]
        rh_n1 = rh[1:n_hours+1] if n_hours+1 <= rh.shape[0] else rh[1:rh.shape[0]]
        
        # Pad if needed
        if len(air_n) < n_hours and len(air_n) > 0:
            pad_n = n_hours - len(air_n)
            air_n = np.concatenate([air_n, np.repeat(air_n[-1], pad_n)])
            wind_n = np.concatenate([wind_n, np.repeat(wind_n[-1], pad_n)])
            rh_n = np.concatenate([rh_n, np.repeat(rh_n[-1], pad_n)])
        elif len(air_n) == 0:
            air_n = np.zeros(n_hours)
            wind_n = np.zeros(n_hours)
            rh_n = np.zeros(n_hours)
        
        if len(air_n1) < n_hours and len(air_n1) > 0:
            pad_n1 = n_hours - len(air_n1)
            air_n1 = np.concatenate([air_n1, np.repeat(air_n1[-1], pad_n1)])
            wind_n1 = np.concatenate([wind_n1, np.repeat(wind_n1[-1], pad_n1)])
            rh_n1 = np.concatenate([rh_n1, np.repeat(rh_n1[-1], pad_n1)])
        elif len(air_n1) == 0:
            air_n1 = np.zeros(n_hours)
            wind_n1 = np.zeros(n_hours)
            rh_n1 = np.zeros(n_hours)
        
        return air_n, wind_n, rh_n, air_n1, wind_n1, rh_n1
    
    @staticmethod
    def _extract_utci_value(utci_result) -> Union[float, np.ndarray]:
        """
        Extract numeric UTCI value(s) from pythermalcomfort result.
        
        Args:
            utci_result: Result from pythermalcomfort.models.utci()
            
        Returns:
            Float or numpy array of UTCI values
        """
        if hasattr(utci_result, 'utci'):
            val = utci_result.utci
        elif isinstance(utci_result, dict) and 'utci' in utci_result:
            val = utci_result['utci']
        else:
            val = utci_result
        
        # Return as array if input is array-like, otherwise as scalar
        return np.asarray(val, dtype=float) if isinstance(val, (list, np.ndarray)) else float(val)

