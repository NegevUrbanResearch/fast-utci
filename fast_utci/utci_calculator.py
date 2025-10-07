"""
UTCI Calculator module for fast-utci.

Computes Universal Thermal Climate Index (UTCI) from weather data and MRT results
using pythermalcomfort, following the architecture described in README.md.
"""

import numpy as np
import pandas as pd
from typing import List, Union, Optional, Dict, Any, Tuple
import os
from pathlib import Path
import warnings

# Import our reader module for consistent data handling
from .model_reader import read_weather_data, read_project_data

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

# UTCI thermal comfort thresholds
UTCI_COMFORT_THRESHOLDS = {
    'extreme_cold': (-float('inf'), -40),
    'very_cold': (-40, -27),
    'cold': (-27, -13),
    'cool': (-13, 9),
    'comfortable': (9, 26),
    'warm': (26, 32),
    'hot': (32, 38),
    'very_hot': (38, 46),
    'extreme_hot': (46, float('inf'))
}


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
            # Convert EPW to DataFrame using reader module
            self.epw_data = weather_data
            self.weather_df = read_weather_data(weather_data)
        else:
            raise ValueError(f"Unsupported weather_data type: {type(weather_data)}")
        
        # Add time columns if not present
        self._add_time_columns()
        
        print(f"Loaded weather data:")
        if self.epw_data:
            print(f"  Location: {self.epw_data.location}")
        print(f"  Data points: {len(self.weather_df)} hours")
        print(f"  Temperature range: {self.weather_df['air_temp'].min():.1f} to {self.weather_df['air_temp'].max():.1f} °C")
        print(f"  Wind speed range: {self.weather_df['wind_speed'].min():.1f} to {self.weather_df['wind_speed'].max():.1f} m/s")
        print(f"  Humidity range: {self.weather_df['relative_humidity'].min():.1f} to {self.weather_df['relative_humidity'].max():.1f} %")
    
    def _add_time_columns(self) -> None:
        """Add time columns to weather DataFrame if not present."""
        if 'hour' not in self.weather_df.columns:
            self.weather_df['hour'] = [dt.hour for dt in self.weather_df['datetime']]
        if 'month' not in self.weather_df.columns:
            self.weather_df['month'] = [dt.month for dt in self.weather_df['datetime']]
        if 'day' not in self.weather_df.columns:
            self.weather_df['day'] = [dt.day for dt in self.weather_df['datetime']]
    
    def compute_utci(self, 
                     mrt_results: Dict[str, Any],
                     analysis_period: Optional[Any] = None,
                     target_hours: Optional[List[int]] = None,
                     show_progress: bool = False,
                     n_workers: Optional[int] = None) -> Dict[str, Any]:
        """
        Compute UTCI from MRT results and weather data.
        
        Args:
            mrt_results: Dictionary from MRTCalculator.compute_mrt()
            analysis_period: Optional time period filter
            target_hours: Optional hour filter (0-23)
            show_progress: Whether to show progress bar
            n_workers: Number of parallel workers (default: CPU count - 1)
            
        Returns:
            Dictionary with UTCI results per position
        """
        if not PYTHERMALCOMFORT_AVAILABLE:
            raise RuntimeError("pythermalcomfort required for UTCI calculations")
        
        if self.weather_df is None:
            raise ValueError("Weather data must be loaded before computing UTCI")
        
        # Prepare weather data once
        weather_filtered = self._prepare_weather_data(analysis_period, target_hours)

        # Optional optimized numpy views for hot path
        use_vectorized = os.getenv("FAST_UTCI_VECTORIZED_UTCI", "1").lower() in ("1", "true", "yes", "on")
        if use_vectorized:
            self._weather_np = {
                'air_temp': weather_filtered['air_temp'].to_numpy(copy=False),
                'wind_speed': weather_filtered['wind_speed'].to_numpy(copy=False),
                'relative_humidity': weather_filtered['relative_humidity'].to_numpy(copy=False),
                'datetime': weather_filtered['datetime'].to_numpy(copy=False) if 'datetime' in weather_filtered.columns else None
            }
        else:
            self._weather_np = None
        
        # Use parallel processing for UTCI calculations
        n_positions = len(mrt_results)
        
        # Use serial processing for small datasets
        if n_positions < 50:
            return self._compute_utci_serial(mrt_results, weather_filtered, show_progress)
        
        # Use parallel processing for larger datasets
        return self._compute_utci_parallel(mrt_results, weather_filtered, show_progress, n_workers)
    
    def _prepare_weather_data(self, 
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
    
    def _compute_utci_serial(self, mrt_results, weather_filtered, show_progress):
        """Serial UTCI computation for small datasets."""
        utci_results = {}
        
        # Set up iteration with optional progress bar
        if show_progress:
            try:
                from tqdm import tqdm
                pos_iter = tqdm(mrt_results.items(), desc="Computing UTCI (serial)", unit="pos")
            except ImportError:
                pos_iter = mrt_results.items()
        else:
            pos_iter = mrt_results.items()
        
        for pos_key, mrt_data in pos_iter:
            utci_results[pos_key] = self._calculate_utci_for_position(mrt_data, weather_filtered)
        
        return utci_results
    
    def _compute_utci_parallel(self, mrt_results, weather_filtered, show_progress, n_workers):
        """Parallel UTCI computation for larger datasets."""
        import multiprocessing as mp
        from multiprocessing import Pool
        import time
        
        n_positions = len(mrt_results)
        
        if n_workers is None:
            n_workers = max(1, mp.cpu_count() - 1)
        
        # Convert mrt_results to list of (key, data) tuples for chunking
        mrt_items = list(mrt_results.items())
        
        # Create chunks with improved load balancing
        positions_per_worker = n_positions // n_workers
        extra_positions = n_positions % n_workers
        
        chunks = []
        start_idx = 0
        for worker_id in range(n_workers):
            chunk_size = positions_per_worker + (1 if worker_id < extra_positions else 0)
            end_idx = start_idx + chunk_size
            
            if start_idx < n_positions:
                chunks.append(mrt_items[start_idx:end_idx])
                start_idx = end_idx
        
        print(f"Processing {n_positions} UTCI calculations with {n_workers} workers in {len(chunks)} chunks")
        
        # Process chunks in parallel. Prefer numpy dict if available to reduce pickling cost.
        shared_weather = self._weather_np if getattr(self, '_weather_np', None) is not None else weather_filtered
        chunk_args = [(chunk, shared_weather) for chunk in chunks]
        
        with Pool(processes=n_workers) as pool:
            start_time = time.time()
            
            if show_progress:
                from tqdm import tqdm
                with tqdm(total=n_positions, desc="Computing UTCI (parallel)", unit="pos", 
                         mininterval=1.0, maxinterval=5.0, smoothing=0.1, leave=True) as pbar:
                    
                    utci_results = {}
                    for chunk_results in pool.imap(_compute_utci_chunk, chunk_args):
                        utci_results.update(chunk_results)
                        pbar.update(len(chunk_results))
                        
                        # Update description with time estimate
                        elapsed = time.time() - start_time
                        if elapsed > 0:
                            rate = pbar.n / elapsed
                            eta = (n_positions - pbar.n) / rate if rate > 0 else 0
                            pbar.set_description(f"Computing UTCI (parallel) ({rate:.1f} pos/s, ETA: {eta:.0f}s)")
            else:
                # No progress bar - use simple processing
                chunk_results_list = pool.map(_compute_utci_chunk, chunk_args)
                utci_results = {}
                for chunk_results in chunk_results_list:
                    utci_results.update(chunk_results)
        
        return utci_results
    
    def _calculate_utci_for_position(self, mrt_data: Dict[str, Any], weather_filtered: pd.DataFrame) -> Dict[str, Any]:
        """Calculate UTCI for a single position."""
        mrt_values = mrt_data['mrt']
        position = mrt_data['position']
        n_hours = len(mrt_values)
        
        # Optimized path: use numpy arrays and vectorized pythermalcomfort if enabled
        if getattr(self, '_weather_np', None) is not None:
            air = self._weather_np['air_temp']
            wind = self._weather_np['wind_speed']
            rh = self._weather_np['relative_humidity']
            dts = self._weather_np.get('datetime')

            # Slice needed hours, pad by repeating last if short
            length = min(n_hours, air.shape[0])
            air_slice = air[:length]
            wind_slice = wind[:length]
            rh_slice = rh[:length]
            if length < n_hours:
                pad_n = n_hours - length
                air_slice = np.concatenate([air_slice, np.repeat(air_slice[-1], pad_n)])
                wind_slice = np.concatenate([wind_slice, np.repeat(wind_slice[-1], pad_n)])
                rh_slice = np.concatenate([rh_slice, np.repeat(rh_slice[-1], pad_n)])

            # Call vectorized utci; pythermalcomfort.utci accepts numpy arrays
            try:
                utci_vals = utci(tdb=air_slice, tr=np.asarray(mrt_values), v=wind_slice, rh=rh_slice)
                # Normalize return to numpy array of floats
                if hasattr(utci_vals, 'utci'):
                    utci_array = np.asarray(utci_vals.utci, dtype=float)
                elif isinstance(utci_vals, dict) and 'utci' in utci_vals:
                    utci_array = np.asarray(utci_vals['utci'], dtype=float)
                else:
                    utci_array = np.asarray(utci_vals, dtype=float)
            except Exception as e:
                warnings.warn(f"Vectorized UTCI calculation failed; falling back to loop: {e}")
                utci_array = None

            if utci_array is not None:
                return {
                    'position': position,
                    'utci': utci_array,
                    'mrt': mrt_values,
                    'air_temp': air_slice,
                    'wind_speed': wind_slice,
                    'relative_humidity': rh_slice,
                    'datetime': dts[:n_hours] if dts is not None and dts.shape[0] >= n_hours else (dts if dts is not None else None)
                }

        # Fallback: original per-hour loop using DataFrame
        weather_subset = weather_filtered.iloc[:min(n_hours, len(weather_filtered))].copy()
        if len(weather_subset) < n_hours:
            last_weather = weather_subset.iloc[-1:] if len(weather_subset) > 0 else weather_filtered.iloc[:1]
            while len(weather_subset) < n_hours:
                weather_subset = pd.concat([weather_subset, last_weather], ignore_index=True)

        utci_values = []
        for i in range(n_hours):
            try:
                utci_result = utci(
                    tdb=weather_subset.iloc[i]['air_temp'],
                    tr=mrt_values[i],
                    v=weather_subset.iloc[i]['wind_speed'],
                    rh=weather_subset.iloc[i]['relative_humidity']
                )
                utci_values.append(self._extract_utci_value(utci_result))
            except Exception as e:
                warnings.warn(f"UTCI calculation failed for hour {i}: {e}")
                utci_values.append(np.nan)
        
        return {
            'position': position,
            'utci': np.array(utci_values),
            'mrt': mrt_values,
            'air_temp': weather_subset['air_temp'].values[:n_hours],
            'wind_speed': weather_subset['wind_speed'].values[:n_hours],
            'relative_humidity': weather_subset['relative_humidity'].values[:n_hours],
            'datetime': weather_subset['datetime'].values[:n_hours] if 'datetime' in weather_subset.columns else None
        }
    
    def _extract_utci_value(self, utci_result) -> float:
        """Extract numeric UTCI value from pythermalcomfort result."""
        if hasattr(utci_result, 'utci'):
            return float(utci_result.utci)
        elif isinstance(utci_result, dict) and 'utci' in utci_result:
            return float(utci_result['utci'])
        else:
            return float(utci_result)
    
    def classify_thermal_comfort(self, utci_values: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Classify UTCI values into thermal comfort categories.
        
        Args:
            utci_values: Array of UTCI values in °C
            
        Returns:
            Tuple of (comfort_categories, category_counts)
        """
        categories = np.full(utci_values.shape, 'unknown', dtype=object)
        
        # Apply UTCI classification using constants
        for category, (min_val, max_val) in UTCI_COMFORT_THRESHOLDS.items():
            if min_val == -float('inf'):
                categories[utci_values < max_val] = category
            elif max_val == float('inf'):
                categories[utci_values >= min_val] = category
            else:
                categories[(utci_values >= min_val) & (utci_values < max_val)] = category
        
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
                    'hour': pd.to_datetime(data['datetime'][i]).hour if data['datetime'] is not None and i < len(data['datetime']) else i,
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
    
    # Compute UTCI with progress bar
    utci_results = utci_calc.compute_utci(mrt_results, show_progress=True)
    
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
    from .mrt.mrt_calculator import MRTCalculator
    from .mrt.period import create_validation_period_filter
    
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
    from .mrt.grid import create_rectangular_grid
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
    utci_results = utci_calc.compute_utci(
        mrt_results=mrt_results,
        analysis_period=analysis_period,
        target_hours=target_hours,
        show_progress=True
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


def _compute_utci_chunk(args):
    """Worker function for parallel processing of UTCI calculation chunks.

    Accepts either a pandas DataFrame or a dict of numpy arrays for weather.
    Uses vectorized UTCI computation when arrays are provided.
    """
    chunk_data, weather_input = args
    
    # Determine input type
    is_np = isinstance(weather_input, dict)
    
    results = {}
    
    for pos_key, mrt_data in chunk_data:
        mrt_values = np.asarray(mrt_data['mrt'])
        position = mrt_data['position']
        n_hours = len(mrt_values)
        
        if is_np:
            air = weather_input['air_temp']
            wind = weather_input['wind_speed']
            rh = weather_input['relative_humidity']
            dts = weather_input.get('datetime')
            length = min(n_hours, air.shape[0])
            air_slice = air[:length]
            wind_slice = wind[:length]
            rh_slice = rh[:length]
            if length < n_hours and length > 0:
                pad_n = n_hours - length
                air_slice = np.concatenate([air_slice, np.repeat(air_slice[-1], pad_n)])
                wind_slice = np.concatenate([wind_slice, np.repeat(wind_slice[-1], pad_n)])
                rh_slice = np.concatenate([rh_slice, np.repeat(rh_slice[-1], pad_n)])
            elif length == 0:
                air_slice = np.zeros(n_hours)
                wind_slice = np.zeros(n_hours)
                rh_slice = np.zeros(n_hours)
            # Vectorized UTCI call
            try:
                from pythermalcomfort.models import utci
                utci_vals = utci(tdb=air_slice, tr=mrt_values, v=wind_slice, rh=rh_slice)
                if hasattr(utci_vals, 'utci'):
                    utci_array = np.asarray(utci_vals.utci, dtype=float)
                elif isinstance(utci_vals, dict) and 'utci' in utci_vals:
                    utci_array = np.asarray(utci_vals['utci'], dtype=float)
                else:
                    utci_array = np.asarray(utci_vals, dtype=float)
            except Exception as e:
                import warnings
                warnings.warn(f"Vectorized UTCI failed in worker; falling back to loop: {e}")
                utci_array = None
        else:
            # DataFrame path
            weather_filtered = weather_input
            weather_subset = weather_filtered.iloc[:min(n_hours, len(weather_filtered))].copy()
            if len(weather_subset) < n_hours:
                last_weather = weather_subset.iloc[-1:] if len(weather_subset) > 0 else weather_filtered.iloc[:1]
                while len(weather_subset) < n_hours:
                    weather_subset = pd.concat([weather_subset, last_weather], ignore_index=True)
            utci_vals = []
            for i in range(n_hours):
                try:
                    from pythermalcomfort.models import utci
                    res = utci(
                        tdb=weather_subset.iloc[i]['air_temp'],
                        tr=mrt_values[i],
                        v=weather_subset.iloc[i]['wind_speed'],
                        rh=weather_subset.iloc[i]['relative_humidity']
                    )
                    if hasattr(res, 'utci'):
                        utci_vals.append(float(res.utci))
                    elif isinstance(res, dict) and 'utci' in res:
                        utci_vals.append(float(res['utci']))
                    else:
                        utci_vals.append(float(res))
                except Exception:
                    utci_vals.append(np.nan)
            utci_array = np.asarray(utci_vals, dtype=float)
            air_slice = weather_subset['air_temp'].to_numpy()
            wind_slice = weather_subset['wind_speed'].to_numpy()
            rh_slice = weather_subset['relative_humidity'].to_numpy()
            dts = weather_subset['datetime'].to_numpy() if 'datetime' in weather_subset.columns else None
        
        # Control payload via env flags while keeping keys for compatibility
        include_weather = os.getenv("FAST_UTCI_INCLUDE_WEATHER_IN_RESULTS", "1").lower() in ("1", "true", "yes", "on")
        include_datetime = os.getenv("FAST_UTCI_INCLUDE_DATETIME_IN_RESULTS", "1").lower() in ("1", "true", "yes", "on")
        
        result_entry = {
            'position': position,
            'utci': utci_array if utci_array is not None else np.full(n_hours, np.nan),
            'mrt': mrt_values
        }
        if include_weather:
            result_entry['air_temp'] = air_slice
            result_entry['wind_speed'] = wind_slice
            result_entry['relative_humidity'] = rh_slice
        if include_datetime:
            result_entry['datetime'] = dts[:n_hours] if dts is not None and getattr(dts, 'shape', [0])[0] >= n_hours else (dts if dts is not None else None)
        else:
            result_entry['datetime'] = None
        
        results[pos_key] = result_entry
    
    return results
