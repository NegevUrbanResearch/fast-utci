"""
UTCI Calculator module for fast-utci.

Computes Universal Thermal Climate Index (UTCI) from weather data and MRT results
using pythermalcomfort, following the architecture described in README.md.
"""

import numpy as np
import pandas as pd
from typing import List, Union, Optional, Dict, Any, Tuple
from pathlib import Path
import warnings

# Import our reader module for consistent data handling
from reader import read_weather_data, read_project_data

# Import pythermalcomfort for UTCI calculations
try:
    from pythermalcomfort.models import utci
    PYTHERMALCOMFORT_AVAILABLE = True
except ImportError:
    warnings.warn("pythermalcomfort not available. UTCI calculations will be limited.")
    PYTHERMALCOMFORT_AVAILABLE = False

# Import ladybug for EPW handling if available
try:
    from ladybug.epw import EPW
    from ladybug.location import Location
    LADYBUG_AVAILABLE = True
except ImportError:
    warnings.warn("ladybug-core not available. EPW handling will be limited.")
    LADYBUG_AVAILABLE = False


class UTCICalculator:
    """
    Universal Thermal Climate Index (UTCI) calculator.
    
    Combines MRT results with weather data (air temperature, humidity, wind speed)
    to compute UTCI thermal comfort indices for each analysis position.
    
    Features:
    - Direct integration with MRT calculation results
    - Efficient batch processing for large datasets
    - Support for time series and single-hour analysis
    - CSV export for validation and visualization
    """
    
    def __init__(self, 
                 weather_data: Optional[Union[str, Path, pd.DataFrame, EPW]] = None,
                 epw_object: Optional[EPW] = None):
        """
        Initialize UTCI calculator with weather data.
        
        Args:
            weather_data: Weather data as file path, DataFrame, or EPW object
            epw_object: Optional EPW object (for location info if weather_data is DataFrame)
        """
        self.epw_data = None
        self.weather_df = None
        
        if weather_data is not None:
            self.load_weather_data(weather_data, epw_object)
    
    def load_weather_data(self, 
                         weather_data: Union[str, Path, pd.DataFrame, EPW],
                         epw_object: Optional[EPW] = None) -> None:
        """
        Load weather data for UTCI calculations using reader module.
        
        Args:
            weather_data: Weather data as file path, DataFrame, or EPW object
            epw_object: Optional EPW object for location info
        """
        if isinstance(weather_data, (str, Path)):
            # Load from file using reader module
            self.weather_df = read_weather_data(weather_data)
            self.epw_data = EPW(str(weather_data))
        elif isinstance(weather_data, pd.DataFrame):
            # Use provided DataFrame
            self.weather_df = weather_data.copy()
            self.epw_data = epw_object
        elif isinstance(weather_data, EPW):
            # Convert EPW to DataFrame
            self.epw_data = weather_data
            self.weather_df = self._epw_to_dataframe(weather_data)
        else:
            raise ValueError(f"Unsupported weather_data type: {type(weather_data)}")
        
        # Add time columns if not present
        if 'hour' not in self.weather_df.columns:
            self.weather_df['hour'] = [dt.hour for dt in self.weather_df['datetime']]
        if 'month' not in self.weather_df.columns:
            self.weather_df['month'] = [dt.month for dt in self.weather_df['datetime']]
        if 'day' not in self.weather_df.columns:
            self.weather_df['day'] = [dt.day for dt in self.weather_df['datetime']]
        
        print(f"Loaded weather data:")
        if self.epw_data:
            print(f"  Location: {self.epw_data.location}")
        print(f"  Data points: {len(self.weather_df)} hours")
        print(f"  Temperature range: {self.weather_df['air_temp'].min():.1f} to {self.weather_df['air_temp'].max():.1f} °C")
        print(f"  Wind speed range: {self.weather_df['wind_speed'].min():.1f} to {self.weather_df['wind_speed'].max():.1f} m/s")
        print(f"  Humidity range: {self.weather_df['relative_humidity'].min():.1f} to {self.weather_df['relative_humidity'].max():.1f} %")
    
    def _epw_to_dataframe(self, epw_data: EPW) -> pd.DataFrame:
        """Convert EPW data to pandas DataFrame (fallback for direct EPW input)."""
        data = {
            'datetime': epw_data.dry_bulb_temperature.datetimes,
            'air_temp': epw_data.dry_bulb_temperature.values,
            'relative_humidity': epw_data.relative_humidity.values,
            'wind_speed': epw_data.wind_speed.values,
            'wind_direction': epw_data.wind_direction.values,
            'direct_normal_radiation': epw_data.direct_normal_radiation.values,
            'diffuse_horizontal_radiation': epw_data.diffuse_horizontal_radiation.values,
            'horizontal_infrared_radiation_intensity': epw_data.horizontal_infrared_radiation_intensity.values
        }
        
        return pd.DataFrame(data)
    
    def compute_utci(self, 
                     mrt_results: Dict[str, Any],
                     analysis_period: Optional[Any] = None,
                     target_hours: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Compute UTCI from MRT results and weather data.
        
        Args:
            mrt_results: Dictionary from MRTCalculator.compute_mrt()
            analysis_period: Optional time period filter
            target_hours: Optional hour filter (0-23)
            
        Returns:
            Dictionary with UTCI results per position
        """
        if not PYTHERMALCOMFORT_AVAILABLE:
            raise RuntimeError("pythermalcomfort required for UTCI calculations")
        
        if self.weather_df is None:
            raise ValueError("Weather data must be loaded before computing UTCI")
        
        # Filter weather data
        weather_filtered = self._filter_weather_data(analysis_period, target_hours)
        
        utci_results = {}
        
        for pos_key, mrt_data in mrt_results.items():
            mrt_values = mrt_data['mrt']
            position = mrt_data['position']
            
            # Get corresponding weather data
            n_hours = len(mrt_values)
            if len(weather_filtered) < n_hours:
                warnings.warn(f"Weather data length ({len(weather_filtered)}) < MRT data length ({n_hours})")
                # Pad weather data by repeating last values
                while len(weather_filtered) < n_hours:
                    weather_filtered = pd.concat([weather_filtered, weather_filtered.iloc[-1:]], ignore_index=True)
            
            # Truncate to match MRT data length
            weather_subset = weather_filtered.iloc[:n_hours].copy()
            
            # Compute UTCI for each hour
            utci_values = []
            
            for i in range(n_hours):
                try:
                    utci_result = utci(
                        tdb=weather_subset.iloc[i]['air_temp'],           # Air temperature [°C]
                        tr=mrt_values[i],                                 # Mean radiant temperature [°C]
                        v=weather_subset.iloc[i]['wind_speed'],           # Wind speed [m/s]
                        rh=weather_subset.iloc[i]['relative_humidity']    # Relative humidity [%]
                    )
                    # Extract numeric UTCI value from result
                    if hasattr(utci_result, 'utci'):
                        utci_val = float(utci_result.utci)
                    elif isinstance(utci_result, dict) and 'utci' in utci_result:
                        utci_val = float(utci_result['utci'])
                    else:
                        utci_val = float(utci_result)
                    
                    utci_values.append(utci_val)
                    
                except Exception as e:
                    warnings.warn(f"UTCI calculation failed for hour {i}: {e}")
                    utci_values.append(np.nan)
            
            # Store results
            utci_results[pos_key] = {
                'position': position,
                'utci': np.array(utci_values),
                'mrt': mrt_values,
                'air_temp': weather_subset['air_temp'].values[:n_hours],
                'wind_speed': weather_subset['wind_speed'].values[:n_hours],
                'relative_humidity': weather_subset['relative_humidity'].values[:n_hours],
                'datetime': weather_subset['datetime'].values[:n_hours] if 'datetime' in weather_subset.columns else None
            }
        
        return utci_results
    
    def _filter_weather_data(self, 
                           analysis_period: Optional[Any], 
                           target_hours: Optional[List[int]]) -> pd.DataFrame:
        """Filter weather data by analysis period and target hours."""
        df = self.weather_df.copy()
        
        # Filter by analysis period (e.g., August 15th)
        if analysis_period:
            if hasattr(analysis_period, 'start_month'):
                # Ladybug AnalysisPeriod object
                df = df[
                    (df['month'] >= analysis_period.start_month) & 
                    (df['month'] <= analysis_period.end_month) &
                    (df['day'] >= analysis_period.start_day) & 
                    (df['day'] <= analysis_period.end_day)
                ]
            else:
                # Assume tuple format (start_month, start_day, end_month, end_day)
                if len(analysis_period) >= 4:
                    start_month, start_day, end_month, end_day = analysis_period[:4]
                    df = df[
                        (df['month'] >= start_month) & 
                        (df['month'] <= end_month) &
                        (df['day'] >= start_day) & 
                        (df['day'] <= end_day)
                    ]
        
        # Filter by target hours
        if target_hours:
            df = df[df['hour'].isin(target_hours)]
        
        return df.reset_index(drop=True)
    
    def compute_utci_batch(self,
                          mrt_results: Dict[str, Any],
                          analysis_period: Optional[Any] = None,
                          target_hours: Optional[List[int]] = None,
                          show_progress: bool = True) -> Dict[str, Any]:
        """
        Compute UTCI for batch of positions with progress tracking.
        
        Args:
            mrt_results: Dictionary from MRTCalculator.compute_mrt()
            analysis_period: Optional time period filter
            target_hours: Optional hour filter
            show_progress: Whether to show progress bar
            
        Returns:
            Dictionary with UTCI results per position
        """
        if show_progress:
            from tqdm import tqdm
            pos_iter = tqdm(mrt_results.items(), desc="Computing UTCI", unit="pos")
        else:
            pos_iter = mrt_results.items()
        
        utci_results = {}
        
        # Get filtered weather data once
        weather_filtered = self._filter_weather_data(analysis_period, target_hours)
        
        for pos_key, mrt_data in pos_iter:
            mrt_values = mrt_data['mrt']
            position = mrt_data['position']
            
            # Get corresponding weather data
            n_hours = len(mrt_values)
            weather_subset = weather_filtered.iloc[:min(n_hours, len(weather_filtered))].copy()
            
            # Handle length mismatch
            if len(weather_subset) < n_hours:
                # Repeat last weather entry
                last_weather = weather_subset.iloc[-1:] if len(weather_subset) > 0 else weather_filtered.iloc[:1]
                while len(weather_subset) < n_hours:
                    weather_subset = pd.concat([weather_subset, last_weather], ignore_index=True)
            
            # Vectorized UTCI calculation for better performance
            utci_values = []
            
            for i in range(n_hours):
                try:
                    utci_result = utci(
                        tdb=weather_subset.iloc[i]['air_temp'],
                        tr=mrt_values[i],
                        v=weather_subset.iloc[i]['wind_speed'],
                        rh=weather_subset.iloc[i]['relative_humidity']
                    )
                    # Extract numeric UTCI value from result
                    if hasattr(utci_result, 'utci'):
                        utci_val = float(utci_result.utci)
                    elif isinstance(utci_result, dict) and 'utci' in utci_result:
                        utci_val = float(utci_result['utci'])
                    else:
                        utci_val = float(utci_result)
                    
                    utci_values.append(utci_val)
                except:
                    utci_values.append(np.nan)
            
            utci_results[pos_key] = {
                'position': position,
                'utci': np.array(utci_values),
                'mrt': mrt_values,
                'air_temp': weather_subset['air_temp'].values[:n_hours],
                'wind_speed': weather_subset['wind_speed'].values[:n_hours],
                'relative_humidity': weather_subset['relative_humidity'].values[:n_hours],
                'datetime': weather_subset['datetime'].values[:n_hours] if 'datetime' in weather_subset.columns else None
            }
        
        return utci_results
    
    def classify_thermal_comfort(self, utci_values: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Classify UTCI values into thermal comfort categories.
        
        Args:
            utci_values: Array of UTCI values in °C
            
        Returns:
            Tuple of (comfort_categories, category_counts)
        """
        # UTCI thermal comfort categories
        categories = np.full(utci_values.shape, 'unknown', dtype=object)
        
        # Apply UTCI classification
        categories[utci_values < -40] = 'extreme_cold'
        categories[(utci_values >= -40) & (utci_values < -27)] = 'very_cold'
        categories[(utci_values >= -27) & (utci_values < -13)] = 'cold'
        categories[(utci_values >= -13) & (utci_values < 9)] = 'cool'
        categories[(utci_values >= 9) & (utci_values < 26)] = 'comfortable'
        categories[(utci_values >= 26) & (utci_values < 32)] = 'warm'
        categories[(utci_values >= 32) & (utci_values < 38)] = 'hot'
        categories[(utci_values >= 38) & (utci_values < 46)] = 'very_hot'
        categories[utci_values >= 46] = 'extreme_hot'
        
        # Count categories
        unique, counts = np.unique(categories, return_counts=True)
        category_counts = dict(zip(unique, counts))
        
        return categories, category_counts
    
    def to_csv(self,
              utci_results: Dict[str, Any],
              csv_path: str,
              include_weather: bool = True,
              include_comfort_categories: bool = True) -> None:
        """
        Export UTCI results to CSV file.
        
        Args:
            utci_results: Dictionary from compute_utci()
            csv_path: Output CSV file path
            include_weather: Whether to include weather variables
            include_comfort_categories: Whether to include thermal comfort categories
        """
        rows = []
        
        for pos_key, data in utci_results.items():
            position = data['position']
            utci_vals = data['utci']
            mrt_vals = data['mrt']
            
            # Get comfort categories if requested
            if include_comfort_categories:
                comfort_categories, _ = self.classify_thermal_comfort(utci_vals)
            
            for i, (utci_val, mrt_val) in enumerate(zip(utci_vals, mrt_vals)):
                row = {
                    'position_id': pos_key,
                    'x': position[0],
                    'y': position[1],
                    'z': position[2],
                    'hour': i,
                    'utci': utci_val,
                    'mrt': mrt_val
                }
                
                # Add weather data if available and requested
                if include_weather and 'air_temp' in data:
                    row.update({
                        'air_temp': data['air_temp'][i],
                        'wind_speed': data['wind_speed'][i],
                        'relative_humidity': data['relative_humidity'][i]
                    })
                
                # Add comfort category if requested
                if include_comfort_categories:
                    row['comfort_category'] = comfort_categories[i]
                
                # Add datetime if available
                if data['datetime'] is not None and i < len(data['datetime']):
                    row['datetime'] = data['datetime'][i]
                
                rows.append(row)
        
        # Create DataFrame and export
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        
        print(f"Exported UTCI results to: {csv_path}")
        print(f"  Records: {len(df)}")
        print(f"  Positions: {len(utci_results)}")
        
        if include_comfort_categories and len(df) > 0:
            comfort_summary = df['comfort_category'].value_counts()
            print(f"  Comfort distribution: {dict(comfort_summary)}")
    
    def summary_statistics(self, utci_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute summary statistics for UTCI results.
        
        Args:
            utci_results: Dictionary from compute_utci()
            
        Returns:
            Dictionary with summary statistics
        """
        all_utci = []
        all_mrt = []
        positions = []
        
        for pos_key, data in utci_results.items():
            all_utci.extend(data['utci'])
            all_mrt.extend(data['mrt'])
            positions.append(data['position'])
        
        all_utci = np.array(all_utci)
        all_mrt = np.array(all_mrt)
        
        # Remove NaN values for statistics
        valid_utci = all_utci[~np.isnan(all_utci)]
        valid_mrt = all_mrt[~np.isnan(all_mrt)]
        
        # Compute comfort categories
        if len(valid_utci) > 0:
            _, comfort_counts = self.classify_thermal_comfort(valid_utci)
        else:
            comfort_counts = {}
        
        summary = {
            'total_positions': len(utci_results),
            'total_hours': len(all_utci),
            'valid_utci_values': len(valid_utci),
            'utci_stats': {
                'mean': float(np.mean(valid_utci)) if len(valid_utci) > 0 else np.nan,
                'min': float(np.min(valid_utci)) if len(valid_utci) > 0 else np.nan,
                'max': float(np.max(valid_utci)) if len(valid_utci) > 0 else np.nan,
                'std': float(np.std(valid_utci)) if len(valid_utci) > 0 else np.nan
            },
            'mrt_stats': {
                'mean': float(np.mean(valid_mrt)) if len(valid_mrt) > 0 else np.nan,
                'min': float(np.min(valid_mrt)) if len(valid_mrt) > 0 else np.nan,
                'max': float(np.max(valid_mrt)) if len(valid_mrt) > 0 else np.nan,
                'std': float(np.std(valid_mrt)) if len(valid_mrt) > 0 else np.nan
            },
            'comfort_distribution': comfort_counts,
            'position_bounds': {
                'x_min': float(np.min([p[0] for p in positions])) if positions else np.nan,
                'x_max': float(np.max([p[0] for p in positions])) if positions else np.nan,
                'y_min': float(np.min([p[1] for p in positions])) if positions else np.nan,
                'y_max': float(np.max([p[1] for p in positions])) if positions else np.nan,
                'z_min': float(np.min([p[2] for p in positions])) if positions else np.nan,
                'z_max': float(np.max([p[2] for p in positions])) if positions else np.nan
            }
        }
        
        return summary


def quick_utci_test(epw_file: str, mrt_results: Dict[str, Any]) -> Tuple[UTCICalculator, Dict[str, Any]]:
    """
    Quick test function for UTCI calculation validation.
    
    Args:
        epw_file: Path to EPW weather file
        mrt_results: MRT results from MRTCalculator
        
    Returns:
        Tuple of (UTCICalculator instance, UTCI results)
    """
    # Create UTCI calculator using reader module
    utci_calc = UTCICalculator(epw_file)
    
    # Compute UTCI
    utci_results = utci_calc.compute_utci_batch(mrt_results)
    
    # Print summary
    summary = utci_calc.summary_statistics(utci_results)
    
    print("\n=== UTCI Calculation Summary ===")
    print(f"Positions: {summary['total_positions']}")
    print(f"Hours: {summary['total_hours']}")
    print(f"UTCI range: {summary['utci_stats']['min']:.1f} to {summary['utci_stats']['max']:.1f} °C")
    print(f"MRT range: {summary['mrt_stats']['min']:.1f} to {summary['mrt_stats']['max']:.1f} °C")
    print(f"Comfort distribution: {summary['comfort_distribution']}")
    
    return utci_calc, utci_results


def integrated_mrt_utci_workflow(model_file: str, 
                                epw_file: str,
                                grid_size: float = 25.0,
                                target_hours: List[int] = [13]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Complete workflow: MRT calculation + UTCI calculation using reader module.
    
    Args:
        model_file: Path to 3D model file (.glb/.gltf)
        epw_file: Path to EPW weather file
        grid_size: Grid spacing for analysis points
        target_hours: Hours to analyze (default [13] for 1-2 PM)
        
    Returns:
        Tuple of (MRT results, UTCI results)
    """
    from MRT.mrt_calculator import MRTCalculator
    from MRT.period import create_validation_period_filter
    
    # Load data using reader module
    model, weather_df, epw_data = read_project_data(model_file, epw_file)
    
    print(f"Loaded project data:")
    print(f"  Model: {len(model.vertices)} vertices, {len(model.faces)} faces")
    print(f"  Weather: {len(weather_df)} hours")
    
    # Create MRT calculator
    mrt_calc = MRTCalculator(context_meshes=[model])
    mrt_calc.set_location_from_epw(epw_file)
    
    # Get validation period (August 15th)
    analysis_period, _ = create_validation_period_filter()
    
    # Generate test grid (for demonstration)
    from MRT.grid import create_rectangular_grid
    grid = create_rectangular_grid(
        x_min=-50, x_max=50, y_min=-50, y_max=50,
        grid_size=grid_size, z_height=0.1
    )
    
    print(f"Generated grid: {len(grid.points)} points")
    
    # Compute exposure
    exposure_results = mrt_calc.compute_exposure(
        positions=grid.points,
        analysis_period=analysis_period,
        target_hours=target_hours
    )
    
    # Compute MRT
    mrt_results = mrt_calc.compute_mrt(
        epw_data=epw_data,
        exposure_results=exposure_results,
        analysis_period=analysis_period,
        target_hours=target_hours
    )
    
    # Create UTCI calculator with the weather data
    utci_calc = UTCICalculator(weather_data=weather_df, epw_object=epw_data)
    
    # Compute UTCI
    utci_results = utci_calc.compute_utci_batch(
        mrt_results=mrt_results,
        analysis_period=analysis_period,
        target_hours=target_hours
    )
    
    # Print summary
    summary = utci_calc.summary_statistics(utci_results)
    print("\n=== Integrated MRT + UTCI Results ===")
    print(f"Grid size: {grid_size}m, Points: {len(grid.points)}")
    print(f"Target hours: {target_hours}")
    print(f"UTCI range: {summary['utci_stats']['min']:.1f} to {summary['utci_stats']['max']:.1f} °C")
    print(f"MRT range: {summary['mrt_stats']['min']:.1f} to {summary['mrt_stats']['max']:.1f} °C")
    print(f"Comfort distribution: {summary['comfort_distribution']}")
    
    return mrt_results, utci_results
