"""
Main UTCI Calculator class for fast-utci.

Orchestrates weather data management, UTCI calculation, and result processing
using a clean, modular architecture.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import time

from .config import UTCIConfig, DEFAULT_CONFIG
from .weather import WeatherDataManager
from .calculation import BoundaryAveragingCalculator
from .statistics import compute_summary_statistics, classify_thermal_comfort
from .export import to_csv as export_to_csv
from fast_utci.shared import ParallelProcessor


class UTCICalculator:
    """
    Universal Thermal Climate Index (UTCI) calculator.
    
    Clean orchestrator that combines MRT results with weather data to compute
    UTCI thermal comfort indices. Uses modular architecture with separate
    components for weather management, calculation, and export.
    
    Features:
    - Direct integration with MRT calculation results
    - Efficient batch processing for large datasets
    - Support for time series and single-hour analysis
    - Flexible export options (CSV, statistics)
    
    Example:
        >>> from fast_utci.utci import UTCICalculator
        >>> calc = UTCICalculator(weather_data='weather.epw')
        >>> utci_results = calc.compute_utci(mrt_results)
        >>> calc.to_csv(utci_results, 'output.csv')
    """
    
    def __init__(self,
                 weather_data: Optional[Union[str, Path, Any]] = None,
                 epw_object: Optional[Any] = None,
                 config: Optional[UTCIConfig] = None):
        """
        Initialize UTCI calculator.
        
        Args:
            weather_data: Weather data as file path, DataFrame, or EPW object
            epw_object: Optional EPW object for location info if weather_data is DataFrame
            config: Optional UTCIConfig object (uses DEFAULT_CONFIG if None)
        """
        self.config = config or DEFAULT_CONFIG
        self.weather = None
        self.calculator = BoundaryAveragingCalculator(
            enable_vectorized=self.config.enable_vectorized
        )
        
        if weather_data is not None:
            self.load_weather_data(weather_data, epw_object)
    
    def load_weather_data(self,
                         weather_data: Union[str, Path, Any],
                         epw_object: Optional[Any] = None) -> None:
        """
        Load weather data for UTCI calculations.
        
        Args:
            weather_data: Weather data as file path, DataFrame, or EPW object
            epw_object: Optional EPW object for location info
        """
        self.weather = WeatherDataManager(weather_data, epw_object)
        
        summary = self.weather.get_summary()
        print(f"Loaded weather data:")
        if summary['location']:
            print(f"  Location: {summary['location']}")
        print(f"  Data points: {summary['n_hours']} hours")
        print(f"  Temperature range: {summary['temp_range'][0]:.1f} to {summary['temp_range'][1]:.1f} °C")
        print(f"  Wind speed range: {summary['wind_range'][0]:.1f} to {summary['wind_range'][1]:.1f} m/s")
        print(f"  Humidity range: {summary['humidity_range'][0]:.1f} to {summary['humidity_range'][1]:.1f} %")
    
    def compute_utci(self,
                     mrt_results: Dict[str, Any],
                     analysis_period: Optional[Any] = None,
                     target_hours: Optional[List[int]] = None,
                     show_progress: Optional[bool] = None,
                     n_workers: Optional[int] = None) -> Dict[str, Any]:
        """
        Compute UTCI from MRT results and weather data.
        
        Args:
            mrt_results: Dictionary from MRTCalculator.compute_mrt()
            analysis_period: Optional time period filter
            target_hours: Optional hour filter (0-23)
            show_progress: Whether to show progress bar (None = use config default)
            n_workers: Number of parallel workers (None = use config default)
            
        Returns:
            Dictionary with UTCI results per position
        """
        if self.weather is None:
            raise ValueError("Weather data must be loaded before computing UTCI")
        
        # Filter weather data
        weather_filtered = self.weather
        if analysis_period is not None:
            weather_filtered = weather_filtered.filter_by_period(analysis_period)
        if target_hours is not None:
            weather_filtered = weather_filtered.filter_by_hours(target_hours)
        
        # Determine parallel vs serial
        n_positions = len(mrt_results)
        use_progress = show_progress if show_progress is not None else self.config.show_progress
        
        if n_positions < self.config.parallel_threshold:
            return self._compute_serial(mrt_results, weather_filtered, use_progress)
        else:
            return self._compute_parallel(mrt_results, weather_filtered, use_progress, n_workers)
    
    def _compute_serial(self,
                       mrt_results: Dict[str, Any],
                       weather_filtered: WeatherDataManager,
                       show_progress: bool) -> Dict[str, Any]:
        """Serial UTCI computation for small datasets."""
        weather_df = weather_filtered.to_dataframe()
        utci_results = {}
        
        # Setup iteration with optional progress bar
        if show_progress:
            try:
                from tqdm import tqdm
                pos_iter = tqdm(mrt_results.items(), desc="Computing UTCI (serial)", unit="pos")
            except ImportError:
                pos_iter = mrt_results.items()
        else:
            pos_iter = mrt_results.items()
        
        for pos_key, mrt_data in pos_iter:
            result = self.calculator.calculate(
                mrt_data,
                weather_df,
                include_weather=self.config.include_weather_in_results,
                include_datetime=self.config.include_datetime_in_results
            )
            utci_results[pos_key] = result.to_dict()
        
        return utci_results
    
    def _compute_parallel(self,
                         mrt_results: Dict[str, Any],
                         weather_filtered: WeatherDataManager,
                         show_progress: bool,
                         n_workers: Optional[int]) -> Dict[str, Any]:
        """Parallel UTCI computation for larger datasets."""
        import multiprocessing as mp
        from multiprocessing import Pool
        import time
        
        # Prepare weather data for workers
        if self.config.enable_vectorized:
            weather_data = weather_filtered.to_numpy_arrays()
        else:
            weather_data = weather_filtered.to_dataframe()
        
        # Determine number of workers
        if n_workers is None:
            n_workers = max(1, mp.cpu_count() - 1)
        
        # Convert to list for chunking
        mrt_items = list(mrt_results.items())
        n_positions = len(mrt_items)
        
        # Create balanced chunks
        from fast_utci.shared import BalancedChunkStrategy
        chunk_strategy = BalancedChunkStrategy()
        chunks = chunk_strategy.create_chunks(np.array(mrt_items, dtype=object), n_workers)
        
        print(f"Processing {n_positions} UTCI calculations with {n_workers} workers in {len(chunks)} chunks")
        
        # Prepare chunk arguments with all parameters (avoid partial for Windows compatibility)
        chunk_args = [
            ((i, chunk), weather_data, self.config.enable_vectorized, 
             self.config.include_weather_in_results, self.config.include_datetime_in_results)
            for i, chunk in enumerate(chunks)
        ]
        
        # Process chunks in parallel
        with Pool(processes=n_workers) as pool:
            start_time = time.time()
            
            if show_progress:
                try:
                    from tqdm import tqdm
                    with tqdm(total=n_positions, desc="Computing UTCI (parallel)", unit="pos",
                             mininterval=1.0, maxinterval=5.0, smoothing=0.1, leave=True) as pbar:
                        
                        utci_results = {}
                        for chunk_result in pool.starmap(_compute_utci_worker_batch, chunk_args):
                            utci_results.update(chunk_result)
                            pbar.update(len(chunk_result))
                            
                            # Update description with time estimate
                            elapsed = time.time() - start_time
                            if elapsed > 0:
                                rate = pbar.n / elapsed
                                eta = (n_positions - pbar.n) / rate if rate > 0 else 0
                                pbar.set_description(f"Computing UTCI (parallel) ({rate:.1f} pos/s, ETA: {eta:.0f}s)")
                except ImportError:
                    # No tqdm - use simple processing
                    chunk_results_list = pool.starmap(_compute_utci_worker_batch, chunk_args)
                    utci_results = {}
                    for chunk_result in chunk_results_list:
                        utci_results.update(chunk_result)
            else:
                # No progress bar - simple processing
                chunk_results_list = pool.starmap(_compute_utci_worker_batch, chunk_args)
                utci_results = {}
                for chunk_result in chunk_results_list:
                    utci_results.update(chunk_result)
        
        return utci_results
    
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
        export_to_csv(
            utci_results,
            csv_path,
            include_weather=include_weather,
            include_comfort_categories=include_comfort_categories,
            csv_encoding=self.config.csv_encoding,
            csv_index=self.config.csv_index
        )
    
    def summary_statistics(self, utci_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute summary statistics for UTCI results.
        
        Args:
            utci_results: Dictionary from compute_utci()
            
        Returns:
            Dictionary with summary statistics
        """
        return compute_summary_statistics(utci_results)


def _compute_utci_worker_batch(args, weather_data, enable_vectorized, include_weather, include_datetime) -> Dict[str, Any]:
    """
    Worker function for parallel UTCI computation.
    
    Args:
        args: Tuple of (chunk_id, chunk_data)
        weather_data: Weather data (DataFrame or numpy dict)
        enable_vectorized: Whether to use vectorized calculations
        include_weather: Whether to include weather in results
        include_datetime: Whether to include datetime in results
    """
    chunk_id, chunk_data = args
    
    calculator = BoundaryAveragingCalculator(enable_vectorized=enable_vectorized)
    results = {}
    
    for pos_key, mrt_data in chunk_data:
        result = calculator.calculate(
            mrt_data,
            weather_data,
            include_weather=include_weather,
            include_datetime=include_datetime
        )
        results[pos_key] = result.to_dict()
    
    return results

