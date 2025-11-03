"""
Main MRT Calculator class for fast-utci.

Orchestrates weather data processing, exposure calculations, and SolarCal MRT
computation to match Grasshopper OutdoorSolarMRT results with optimized performance.

Basic usage:
    from fast_utci.mrt import MRTCalculator
    from ladybug.epw import EPW
    
    # Initialize with context geometry
    calculator = MRTCalculator(context_meshes=['building.obj'])
    
    # Load weather data and set location
    epw = EPW('weather.epw')
    calculator.set_location_from_epw('weather.epw')
    
    # Compute exposure and MRT for grid points
    exposure = calculator.compute_exposure(grid_points)
    results = calculator.compute_mrt(epw, exposure)
    
    # Export results
    calculator.to_csv(results, 'output.csv')
"""

import numpy as np
import pandas as pd
from typing import List, Union, Optional, Dict, Any, Tuple
from pathlib import Path
from functools import partial
import warnings
from tqdm import tqdm
import logging

from .solar import get_sun_vectors, filter_hours_by_local_time, SunData
from .mesh import load_context_meshes, MeshContext
from .exposure import compute_exposure, compute_exposure_batch, ExposureResult
from .solarcal import compute_mrt_solarcal, create_solar_body_parameters, SolarCalResult
from .grid import AnalysisGrid, create_grid_from_surface, create_rectangular_grid, load_surface_and_create_grid
from .period import AnalysisPeriod
from .boundary import create_boundary_arrays
from fast_utci.shared import MRTConfig
from fast_utci.shared.weather import WeatherDataManager, EPWAdapter, DataFrameAdapter

# Import parallel utilities from shared
from fast_utci.shared import ParallelProcessor

# Import ladybug for location and EPW handling
from ladybug.location import Location
from ladybug.epw import EPW

logger = logging.getLogger(__name__)


def _create_solarcal_from_epw(epw_data: Any, 
                               exposure: ExposureResult,
                               analysis_period: Optional[AnalysisPeriod],
                               target_hours: Optional[List[int]],
                               ground_reflectance: float,
                               body_params: Any) -> SolarCalResult:
    """
    Shared EPW to SolarCal computation logic.
    
    Args:
        epw_data: EPW object with weather data collections
        exposure: ExposureResult with solar and sky exposure
        analysis_period: Optional analysis period filter
        target_hours: Optional hour filter
        ground_reflectance: Ground reflectance factor
        body_params: SolarCalParameter object
        
    Returns:
        SolarCalResult with MRT and component values
    """
    from ladybug_comfort.collection.solarcal import OutdoorSolarCal
    from ladybug.analysisperiod import AnalysisPeriod as LBAnalysisPeriod
    from ladybug.datacollection import HourlyContinuousCollection
    from ladybug.header import Header
    from ladybug.datatype.fraction import Fraction
    
    # Use provided analysis_period or fallback to August 15
    if analysis_period:
        day_period = LBAnalysisPeriod(
            analysis_period.start_month,
            analysis_period.start_day,
            analysis_period.start_hour,
            analysis_period.end_month,
            analysis_period.end_day,
            analysis_period.end_hour
        )
    else:
        # Fallback to August 15 if no period provided
        day_period = LBAnalysisPeriod(8, 15, 0, 8, 15, 23)
    
    # Filter EPW collections to the day
    air_temp_coll = epw_data.dry_bulb_temperature.filter_by_analysis_period(day_period)
    dir_norm_coll = epw_data.direct_normal_radiation.filter_by_analysis_period(day_period)
    diff_horiz_coll = epw_data.diffuse_horizontal_radiation.filter_by_analysis_period(day_period)
    horiz_ir_coll = epw_data.horizontal_infrared_radiation_intensity.filter_by_analysis_period(day_period)
    
    # Validate exposure array length matches filtered weather data
    expected_hours = len(air_temp_coll.values)
    if len(exposure.fract_body_exp) != expected_hours:
        raise ValueError(
            f"Exposure array length ({len(exposure.fract_body_exp)}) "
            f"doesn't match weather data length ({expected_hours} hours). "
            f"Ensure analysis_period matches in both compute_exposure() and compute_mrt() calls. "
            f"Period: {day_period}"
        )
    
    # Verify the collection is created correctly
    assert len(exposure.fract_body_exp) == len(air_temp_coll.values), \
        f"Length mismatch: exposure={len(exposure.fract_body_exp)}, weather={len(air_temp_coll.values)}"
    
    # Create HourlyContinuousCollection for time-varying solar exposure
    fract_header = Header(Fraction(), 'fraction', day_period, {'location': epw_data.location})
    fract_body_exp_coll = HourlyContinuousCollection(fract_header, exposure.fract_body_exp)
    
    # Create OutdoorSolarCal with proper time-varying exposure
    solar_cal = OutdoorSolarCal(
        epw_data.location,
        dir_norm_coll,
        diff_horiz_coll,
        horiz_ir_coll,
        air_temp_coll,
        fract_body_exp_coll,  # Time-varying exposure (was: scalar bug)
        exposure.sky_exposure,
        ground_reflectance,
        body_params
    )
    
    # Extract results
    mrt_values = np.array(solar_cal.mean_radiant_temperature.values)
    short_erf_values = np.array(solar_cal.shortwave_effective_radiant_field.values)
    long_erf_values = np.array(solar_cal.longwave_effective_radiant_field.values)
    short_dmrt_values = np.array(solar_cal.shortwave_mrt_delta.values)
    long_dmrt_values = np.array(solar_cal.longwave_mrt_delta.values)
    
    # Filter to target hours if specified
    if target_hours:
        hour_indices = target_hours
        mrt_filtered = mrt_values[hour_indices]
        short_erf_filtered = short_erf_values[hour_indices]
        long_erf_filtered = long_erf_values[hour_indices]
        short_dmrt_filtered = short_dmrt_values[hour_indices]
        long_dmrt_filtered = long_dmrt_values[hour_indices]
    else:
        mrt_filtered = mrt_values
        short_erf_filtered = short_erf_values
        long_erf_filtered = long_erf_values
        short_dmrt_filtered = short_dmrt_values
        long_dmrt_filtered = long_dmrt_values
    
    return SolarCalResult(
        mrt=mrt_filtered,
        short_erf=short_erf_filtered,
        long_erf=long_erf_filtered,
        short_dmrt=short_dmrt_filtered,
        long_dmrt=long_dmrt_filtered
    )


class MRTCalculator:
    """
    High-performance MRT calculator with parity to Grasshopper OutdoorSolarMRT.
    
    Features:
    - Fast ray-based occlusion testing
    - SolarCal MRT calculations using ladybug-comfort
    - Grid-based analysis with parallel processing
    - Progress tracking for long-running calculations
    - CSV export for validation against Grasshopper results
    """
    
    def __init__(self, 
                 context_meshes: List[Union[str, Any]] = None,
                 location: Optional[Location] = None,
                 config: Optional[MRTConfig] = None):
        """
        Initialize MRT calculator with context and parameters.
        
        Args:
            context_meshes: List of mesh file paths or trimesh objects for occlusion
            location: Ladybug Location object (lat, lon, timezone)
            config: MRTConfig object (required - must be loaded from TOML via load_config())
        """
        if config is None:
            raise ValueError(
                "MRTConfig is required. Load from TOML using: "
                "from fast_utci.shared import load_config; cfg = load_config(); "
                "MRTCalculator(..., config=cfg.mrt)"
            )
        self.config = config
        
        # Load context geometry
        self.mesh_context = None
        if context_meshes:
            self.mesh_context = load_context_meshes(context_meshes, config=self.config)
            logger.info(f"Loaded context geometry: {len(self.mesh_context.mesh.faces)} faces, "
                        f"BVH acceleration: {self.mesh_context.has_bvh}")
        
        # Set location
        self.location = location
        
        # Create body parameters
        self.body_params = create_solar_body_parameters(
            self.config.absorptivity, self.config.emissivity
        )
        
        # Cache for solar data
        self._sun_data_cache = {}
    
    def set_location_from_epw(self, epw_file: Union[str, Path]):
        """Set location from EPW file."""
        epw = EPW(str(epw_file))
        self.location = epw.location
        logger.info(f"Location set from EPW: {self.location}")
    
    def get_sun_data(self, 
                    analysis_period: Optional[AnalysisPeriod] = None,
                    target_hours: Optional[List[int]] = None) -> SunData:
        """
        Get sun vector data with caching.
        
        Args:
            analysis_period: Optional time period filter
            target_hours: Optional hour filter (0-23)
            
        Returns:
            SunData object with sun vectors and timing
        """
        assert self.location is not None, "Location must be set before computing sun data"
        
        # Create cache key
        cache_key = (
            str(analysis_period) if analysis_period else "full_year",
            str(target_hours) if target_hours else "all_hours",
            self.config.north_degrees
        )
        
        if cache_key in self._sun_data_cache:
            return self._sun_data_cache[cache_key]
        
        # Compute sun data
        if analysis_period:
            period_tuple = (analysis_period.start_month, analysis_period.start_day,
                          analysis_period.end_month, analysis_period.end_day)
        else:
            period_tuple = None
            
        sun_data = get_sun_vectors(
            self.location, 
            analysis_period=period_tuple,
            north_degrees=self.config.north_degrees
        )
        
        # Apply hour filter if specified
        if target_hours:
            sun_data = filter_hours_by_local_time(sun_data, target_hours)
        
        # Cache result
        self._sun_data_cache[cache_key] = sun_data
        
        logger.info(f"Computed sun data: {len(sun_data.solar_times)} hours, "
                    f"{np.sum(sun_data.is_sun_up)} sun-up hours")
        
        return sun_data
    
    def compute_exposure(self, 
                        positions: np.ndarray,
                        analysis_period: Optional[AnalysisPeriod] = None,
                        target_hours: Optional[List[int]] = None,
                        n_workers: Optional[int] = None) -> List[ExposureResult]:
        """
        Compute solar and sky exposure for analysis positions.
        
        Args:
            positions: Shape (n_positions, 3) analysis positions
            analysis_period: Optional time period filter
            target_hours: Optional hour filter
            n_workers: Optional number of workers for parallel processing
            
        Returns:
            List of ExposureResult objects
        """
        positions = np.asarray(positions)
        if positions.ndim == 1:
            positions = positions.reshape(1, -1)
        
        # Get sun data
        sun_data = self.get_sun_data(analysis_period, target_hours)
        
        # Compute exposure for all positions
        results = compute_exposure_batch(
            positions=positions,
            sun_data=sun_data,
            mesh_context=self.mesh_context,
            pt_count=self.config.pt_count,
            height=self.config.human_height,
            show_progress=self.config.show_progress,
            n_workers=n_workers or self.config.n_workers,
            config=self.config
        )
        
        return results
    
    def compute_mrt(self, 
                   weather_data: Any,
                   exposure_results: List[ExposureResult],
                   analysis_period: Optional[AnalysisPeriod] = None,
                   target_hours: Optional[List[int]] = None,
                   n_workers: Optional[int] = None) -> Dict[str, Any]:
        """
        Compute MRT using SolarCal for given exposure results.
        
        Uses boundary averaging: for each hour N, calculates:
        - mrt0 using hour N's weather and exposure
        - mrt1 using hour N+1's weather and exposure
        This matches Grasshopper's OutdoorSolarMRT behavior.
        
        Args:
            weather_data: EPW object, weather DataFrame, or file path
            exposure_results: List of ExposureResult objects from compute_exposure
                            Should include N+1 hours for boundary averaging
            analysis_period: Optional time period filter
            target_hours: Optional hour filter
            n_workers: Number of parallel workers (default: CPU count - 1)
            
        Returns:
            Dictionary with MRT results per position, containing:
            - 'mrt0': MRT array using hour N data
            - 'mrt1': MRT array using hour N+1 data
            - Other analysis results
        """
        # Create and filter weather data using WeatherDataManager
        weather_manager = WeatherDataManager(weather_data)
        if analysis_period is not None:
            weather_manager.filter_by_period(analysis_period)
        if target_hours is not None:
            weather_manager.filter_by_hours(target_hours)
        
        adapter = weather_manager.get_adapter()
        
        # Get underlying data for SolarCal (EPW object or arrays)
        if isinstance(adapter, EPWAdapter):
            epw_data = adapter.epw_data
            weather_arrays = None
            filtered_datetimes = None
        else:  # DataFrameAdapter
            epw_data = None
            weather_arrays = adapter.to_numpy_arrays()
            filtered_datetimes = adapter.get_datetimes()
        
        # Use parallel processing for MRT calculations
        n_positions = len(exposure_results)
        
        # Use serial processing for small datasets
        threshold = self.config.parallel.parallel_threshold
        if n_positions < threshold:
            return self._compute_mrt_serial(
                exposure_results, epw_data, weather_arrays, filtered_datetimes, 
                analysis_period, target_hours
            )
        
        # Use parallel processing for larger datasets
        return self._compute_mrt_parallel(
            exposure_results, epw_data, weather_arrays, filtered_datetimes,
            analysis_period, target_hours, n_workers, show_progress=True
        )
    
    
    def _compute_mrt_serial(self, exposure_results, epw_data, weather_arrays, 
                           filtered_datetimes, analysis_period, target_hours):
        """Serial MRT computation for small datasets."""
        results = {}
        
        for i, exposure in enumerate(tqdm(exposure_results, desc="Computing MRT (serial)", unit="pos")):
            result = self._compute_single_mrt(
                i, exposure, epw_data, weather_arrays, filtered_datetimes,
                analysis_period, target_hours
            )
            results[f'position_{i}'] = result
        
        return results
    
    def _compute_mrt_parallel(self, exposure_results, epw_data, weather_arrays, 
                             filtered_datetimes, analysis_period, target_hours, n_workers, show_progress=True):
        """Parallel MRT computation for larger datasets using ParallelProcessor."""
        n_positions = len(exposure_results)
        
        # Create config with overrides
        parallel_config = self.config.parallel.with_overrides(
            n_workers=n_workers,
            show_progress=show_progress
        )
        
        # Prepare indexed data for chunking
        indexed_data = [(i, exposure_results[i]) for i in range(n_positions)]
        
        # Use functools.partial to create a picklable worker function
        worker_fn = partial(
            _worker_mrt_chunk,
            epw_data=epw_data,
            weather_arrays=weather_arrays,
            filtered_datetimes=filtered_datetimes,
            analysis_period=analysis_period,
            target_hours=target_hours,
            location=self.location,
            config=self.config,
            body_params=self.body_params
        )
        
        # Process and merge results automatically
        from fast_utci.shared import ParallelProcessor, BalancedChunkStrategy
        processor = ParallelProcessor(parallel_config=parallel_config)
        logger.info(f"Processing {n_positions} MRT calculations with {processor.n_workers} workers")
        
        return processor.process_and_merge(
            data=np.array(indexed_data, dtype=object),
            worker_fn=worker_fn,
            chunk_strategy=BalancedChunkStrategy(),
            description="Computing MRT (parallel)",
            merge_dict=True
        )
    
    def _compute_single_mrt(self, i, exposure, epw_data, weather_arrays, 
                           filtered_datetimes, analysis_period, target_hours):
        """
        Compute MRT for a single position using boundary averaging.
        
        Calculates MRT for N+1 hours, then creates:
        - mrt0: MRT values using hours [0:N]
        - mrt1: MRT values using hours [1:N+1]
        """
        # Compute SolarCal MRT
        if epw_data is not None:
            # Use EPW object directly for proper data collection handling
            mrt_result = _create_solarcal_from_epw(
                epw_data, exposure, analysis_period, target_hours,
                self.config.ground_reflectance, self.body_params
            )
        else:
            # Use DataFrame arrays
            # Ensure exposure arrays match weather length
            if filtered_datetimes is not None and len(exposure.fract_body_exp) != len(filtered_datetimes):
                min_len = min(len(exposure.fract_body_exp), len(filtered_datetimes))
                fract_exp = exposure.fract_body_exp[:min_len]
                # Truncate weather arrays to match
                for key in weather_arrays:
                    if isinstance(weather_arrays[key], np.ndarray):
                        weather_arrays[key] = weather_arrays[key][:min_len]
                filtered_datetimes = filtered_datetimes[:min_len]
            else:
                fract_exp = exposure.fract_body_exp
            
            mrt_result = compute_mrt_solarcal(
                air_temperature=weather_arrays['air_temp'],
                direct_normal_rad=weather_arrays['direct_normal_radiation'],
                diffuse_horizontal_rad=weather_arrays['diffuse_horizontal_radiation'],
                horizontal_infrared_rad=weather_arrays['horizontal_infrared_radiation_intensity'],
                fract_body_exp=fract_exp,
                sky_exposure=exposure.sky_exposure,
                location=self.location,
                datetimes=filtered_datetimes,
                ground_reflectance=self.config.ground_reflectance,
                solar_body_par=self.body_params
            )
        
        # Create boundary arrays for averaging
        mrt0, mrt1 = create_boundary_arrays(mrt_result.mrt)
        n_hours = len(mrt0)
        
        # Get fract_body_exp for result (may have been truncated)
        if epw_data is not None:
            fract_exp = exposure.fract_body_exp
        else:
            # fract_exp was already set above
            pass
        
        return {
            'position': exposure.position,
            'mrt0': mrt0,
            'mrt1': mrt1,
            'mrt': mrt0,  # Backward compatibility
            'short_erf': mrt_result.short_erf,
            'long_erf': mrt_result.long_erf,
            'short_dmrt': mrt_result.short_dmrt,
            'long_dmrt': mrt_result.long_dmrt,
            'fract_body_exp': fract_exp[:n_hours] if len(fract_exp) > n_hours else fract_exp,
            'sky_exposure': exposure.sky_exposure,
            'datetimes': filtered_datetimes if filtered_datetimes is not None else [None]
        }
    
    


def _worker_mrt_chunk(chunk: np.ndarray, epw_data: Any, weather_arrays: Optional[Dict[str, Any]],
                     filtered_datetimes: Optional[List], analysis_period: Any, target_hours: Optional[List[int]],
                     location: Any, config: Any, body_params: Any) -> Dict[str, Any]:
    """
    Module-level worker function for parallel processing of MRT calculation chunks.
    
    This function must be at module level to be picklable for multiprocessing.
    
    Args:
        chunk: Chunk of indexed exposure results to process
        epw_data: EPW data object (None if using DataFrame)
        weather_arrays: Filtered weather arrays dictionary (None if using EPW)
        filtered_datetimes: Filtered datetime list (None if using EPW)
        analysis_period: Analysis period object
        target_hours: Target hours list
        location: Location object
        config: MRT config
        body_params: SolarCal body parameters
        
    Returns:
        Dictionary of MRT results
    """
    results = {}
    
    for idx, exposure in chunk:
        # Compute SolarCal MRT
        if epw_data is not None:
            # Use EPW object directly for proper data collection handling
            mrt_result = _create_solarcal_from_epw(
                epw_data, exposure, analysis_period, target_hours,
                config.ground_reflectance, body_params
            )
            fract_exp = exposure.fract_body_exp
        else:
            # Use DataFrame arrays
            # Ensure exposure arrays match weather length
            if filtered_datetimes is not None and len(exposure.fract_body_exp) != len(filtered_datetimes):
                min_len = min(len(exposure.fract_body_exp), len(filtered_datetimes))
                fract_exp = exposure.fract_body_exp[:min_len]
                # Truncate weather arrays to match
                weather_truncated = {k: (v[:min_len] if isinstance(v, np.ndarray) else v) 
                                    for k, v in weather_arrays.items()}
                datetimes_truncated = filtered_datetimes[:min_len]
            else:
                fract_exp = exposure.fract_body_exp
                weather_truncated = weather_arrays
                datetimes_truncated = filtered_datetimes
            
            from .solarcal import compute_mrt_solarcal
            mrt_result = compute_mrt_solarcal(
                air_temperature=weather_truncated['air_temp'],
                direct_normal_rad=weather_truncated['direct_normal_radiation'],
                diffuse_horizontal_rad=weather_truncated['diffuse_horizontal_radiation'],
                horizontal_infrared_rad=weather_truncated['horizontal_infrared_radiation_intensity'],
                fract_body_exp=fract_exp,
                sky_exposure=exposure.sky_exposure,
                location=location,
                datetimes=datetimes_truncated,
                ground_reflectance=config.ground_reflectance,
                solar_body_par=body_params
            )
        
        # Create boundary arrays for averaging
        mrt0, mrt1 = create_boundary_arrays(mrt_result.mrt)
        n_hours = len(mrt0)
        
        results[f'position_{idx}'] = {
            'position': exposure.position,
            'mrt0': mrt0,
            'mrt1': mrt1,
            'mrt': mrt0,  # Backward compatibility
            'short_erf': mrt_result.short_erf,
            'long_erf': mrt_result.long_erf,
            'short_dmrt': mrt_result.short_dmrt,
            'long_dmrt': mrt_result.long_dmrt,
            'fract_body_exp': fract_exp[:n_hours] if len(fract_exp) > n_hours else fract_exp,
            'sky_exposure': exposure.sky_exposure,
            'datetimes': filtered_datetimes if filtered_datetimes is not None else [None]
        }
    
    return results




