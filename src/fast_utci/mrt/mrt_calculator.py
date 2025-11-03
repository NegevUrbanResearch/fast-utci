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
from .period import AnalysisPeriod, filter_weather_data, filter_arrays_by_period
from fast_utci.shared import MRTConfig

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
            f"doesn't match analysis period ({expected_hours} hours). "
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
                   epw_data: Any,
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
            epw_data: EPW object or weather DataFrame
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
        # Extract weather data
        if hasattr(epw_data, 'dry_bulb_temperature'):
            # EPW object
            air_temp = np.array(epw_data.dry_bulb_temperature.values)
            dir_norm_rad = np.array(epw_data.direct_normal_radiation.values)
            diff_horiz_rad = np.array(epw_data.diffuse_horizontal_radiation.values)
            horiz_ir_rad = np.array(epw_data.horizontal_infrared_radiation_intensity.values)
            datetimes = epw_data.dry_bulb_temperature.datetimes
        else:
            # DataFrame
            air_temp = epw_data['air_temp'].values
            dir_norm_rad = epw_data['direct_normal_radiation'].values
            diff_horiz_rad = epw_data['diffuse_horizontal_radiation'].values
            horiz_ir_rad = epw_data['horizontal_infrared_radiation_intensity'].values
            datetimes = epw_data['datetime'].tolist()
        
        # Apply period and hour filters to weather data
        weather_arrays = {
            'air_temp': air_temp,
            'dir_norm_rad': dir_norm_rad,
            'diff_horiz_rad': diff_horiz_rad,
            'horiz_ir_rad': horiz_ir_rad
        }
        
        filtered_weather, filtered_datetimes = filter_arrays_by_period(
            weather_arrays, datetimes, analysis_period, target_hours
        )
        
        # Use parallel processing for MRT calculations
        n_positions = len(exposure_results)
        
        # Use serial processing for small datasets
        threshold = self.config.parallel.parallel_threshold
        if n_positions < threshold:
            return self._compute_mrt_serial(
                exposure_results, epw_data, filtered_weather, filtered_datetimes, 
                analysis_period, target_hours
            )
        
        # Use parallel processing for larger datasets
        return self._compute_mrt_parallel(
            exposure_results, epw_data, filtered_weather, filtered_datetimes,
            analysis_period, target_hours, n_workers, show_progress=True
        )
    
    @staticmethod
    def _create_boundary_arrays(mrt_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create mrt0 and mrt1 arrays for boundary averaging.
        
        mrt0[i] uses hour i's data, mrt1[i] uses hour i+1's data.
        For the last hour, mrt1 duplicates the final value (no next-day wrap).
        
        Args:
            mrt_array: MRT values for N hours
            
        Returns:
            Tuple of (mrt0, mrt1) both with length N
        """
        if len(mrt_array) > 1:
            return mrt_array, np.concatenate([mrt_array[1:], [mrt_array[-1]]])
        return mrt_array, mrt_array
    
    def _compute_mrt_from_epw(self, epw_data, exposure, analysis_period, target_hours):
        """Compute MRT using EPW data collections for proper Grasshopper-like filtering."""
        return _create_solarcal_from_epw(
            epw_data, exposure, analysis_period, target_hours,
            self.config.ground_reflectance, self.body_params
        )
    
    def _compute_mrt_serial(self, exposure_results, epw_data, filtered_weather, 
                           filtered_datetimes, analysis_period, target_hours):
        """Serial MRT computation for small datasets."""
        results = {}
        
        for i, exposure in enumerate(tqdm(exposure_results, desc="Computing MRT (serial)", unit="pos")):
            result = self._compute_single_mrt(
                i, exposure, epw_data, filtered_weather, filtered_datetimes,
                analysis_period, target_hours
            )
            results[f'position_{i}'] = result
        
        return results
    
    def _compute_mrt_parallel(self, exposure_results, epw_data, filtered_weather, 
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
            filtered_weather=filtered_weather,
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
    
    def _compute_single_mrt(self, i, exposure, epw_data, filtered_weather, 
                           filtered_datetimes, analysis_period, target_hours):
        """
        Compute MRT for a single position using boundary averaging.
        
        Calculates MRT for N+1 hours, then creates:
        - mrt0: MRT values using hours [0:N]
        - mrt1: MRT values using hours [1:N+1]
        """
        # Ensure exposure arrays match filtered weather length
        if len(exposure.fract_body_exp) != len(filtered_datetimes):
            warnings.warn(f"Exposure array length ({len(exposure.fract_body_exp)}) "
                        f"doesn't match weather data length ({len(filtered_datetimes)})")
            # Truncate or pad as needed
            min_len = min(len(exposure.fract_body_exp), len(filtered_datetimes))
            fract_exp = exposure.fract_body_exp[:min_len]
            if len(filtered_datetimes) > min_len:
                # Pad weather data
                for key in filtered_weather:
                    filtered_weather[key] = filtered_weather[key][:min_len]
                filtered_datetimes = filtered_datetimes[:min_len]
        else:
            fract_exp = exposure.fract_body_exp
        
        # Compute SolarCal MRT using EPW data collections for proper filtering
        # This computes MRT for all N+1 hours
        if hasattr(epw_data, 'dry_bulb_temperature'):
            # Use EPW object directly for proper data collection handling
            mrt_result = self._compute_mrt_from_epw(
                epw_data, exposure, analysis_period, target_hours
            )
        else:
            # Fallback for DataFrame input
            mrt_result = compute_mrt_solarcal(
                air_temperature=filtered_weather['air_temp'],
                direct_normal_rad=filtered_weather['dir_norm_rad'],
                diffuse_horizontal_rad=filtered_weather['diff_horiz_rad'],
                horizontal_infrared_rad=filtered_weather['horiz_ir_rad'],
                fract_body_exp=fract_exp,
                sky_exposure=exposure.sky_exposure,
                location=self.location,
                datetimes=filtered_datetimes,
                ground_reflectance=self.config.ground_reflectance,
                solar_body_par=self.body_params
            )
        
        # Create boundary arrays for averaging
        mrt0, mrt1 = self._create_boundary_arrays(mrt_result.mrt)
        n_hours = len(mrt0)
        
        return {
            'position': exposure.position,
            'mrt0': mrt0,
            'mrt1': mrt1,
            'mrt': mrt0,  # Backward compatibility
            'short_erf': mrt_result.short_erf,
            'long_erf': mrt_result.long_erf,
            'short_dmrt': mrt_result.short_dmrt,
            'long_dmrt': mrt_result.long_dmrt,
            'fract_body_exp': fract_exp[:n_hours],
            'sky_exposure': exposure.sky_exposure,
            'datetimes': filtered_datetimes if not hasattr(epw_data, 'dry_bulb_temperature') else [None]
        }
    
    
    def to_csv(self, 
              results: Dict[str, Any],
              csv_path: str,
              grasshopper_format: bool = False) -> None:
        """
        Export MRT results to CSV file.
        
        Args:
            results: Results dictionary from compute_mrt
            csv_path: Path for CSV file
            grasshopper_format: If True, use Grasshopper-compatible format for validation
        """
        if grasshopper_format:
            self._export_grasshopper_csv(results, csv_path)
        else:
            self._export_standard_csv(results, csv_path)
    
    def _export_grasshopper_csv(self, results: Dict[str, Any], csv_path: str):
        """Export in Grasshopper validation format."""
        rows = []
        
        for pos_key, pos_data in results.items():
            # Extract position index from key (e.g., 'position_0' -> 0)
            try:
                pos_idx = int(pos_key.split('_')[1])
            except (IndexError, ValueError):
                pos_idx = 0
            
            mrt_values = pos_data['mrt']
            
            for hour_idx, mrt in enumerate(mrt_values):
                # Grid index format: pixel10*10
                pixel_id = pos_idx
                
                # Duplicate MRT in both columns for GH format
                mrt_0 = mrt
                mrt_1 = mrt
                
                # Placeholder values
                utci = 30.0  # Placeholder - compute separately
                color = "255,255,255"  # Placeholder
                
                rows.append([pixel_id, mrt_0, mrt_1, utci, color])
        
        # Create DataFrame and export
        df = pd.DataFrame(rows, columns=['pixel10*10', 'mrt 0', 'mrt 1', 'utci', 'color'])
        df.to_csv(csv_path, index=False)
        logger.info(f"Exported Grasshopper format CSV: {csv_path}")
    
    def _export_standard_csv(self, results: Dict[str, Any], csv_path: str):
        """Export in standard detailed format."""
        rows = []
        
        for pos_key, pos_data in results.items():
            position = pos_data['position']
            mrt_values = pos_data['mrt']
            fract_exp = pos_data['fract_body_exp']
            sky_exposure = pos_data['sky_exposure']
            
            # Create rows for each timestep
            for i, (mrt, fexp) in enumerate(zip(mrt_values, fract_exp)):
                rows.append({
                    'position_id': pos_key,
                    'x': position[0],
                    'y': position[1], 
                    'z': position[2],
                    'hour': i,  # Simplified - no datetime handling
                    'mrt': mrt,
                    'fract_body_exp': fexp,
                    'sky_exposure': sky_exposure,
                    'short_erf': pos_data['short_erf'][i],
                    'long_erf': pos_data['long_erf'][i],
                    'short_dmrt': pos_data['short_dmrt'][i],
                    'long_dmrt': pos_data['long_dmrt'][i]
                })
        
        # Export to CSV
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False, encoding=self.config.csv_encoding)
        logger.info(f"Exported CSV: {csv_path}")


def _worker_mrt_chunk(chunk: np.ndarray, epw_data: Any, filtered_weather: Dict[str, Any],
                     filtered_datetimes: List, analysis_period: Any, target_hours: Optional[List[int]],
                     location: Any, config: Any, body_params: Any) -> Dict[str, Any]:
    """
    Module-level worker function for parallel processing of MRT calculation chunks.
    
    This function must be at module level to be picklable for multiprocessing.
    
    Args:
        chunk: Chunk of indexed exposure results to process
        epw_data: EPW data object
        filtered_weather: Filtered weather data dictionary
        filtered_datetimes: Filtered datetime list
        analysis_period: Analysis period object
        target_hours: Target hours list
        location: Location object
        config: MRT config
        body_params: SolarCal body parameters
        
    Returns:
        Dictionary of MRT results
    """
    # Convert chunk array to list of tuples for compatibility with existing function
    chunk_list = [(idx, exp) for idx, exp in chunk]
    
    # Prepare args in the format expected by _compute_mrt_chunk
    chunk_args = (
        chunk_list,
        epw_data,
        filtered_weather,
        filtered_datetimes,
        analysis_period,
        target_hours,
        location,
        config,
        body_params
    )
    
    # Use existing standalone worker function
    return _compute_mrt_chunk(chunk_args)


def _compute_mrt_chunk(args):
    """Worker function for parallel processing of MRT calculation chunks."""
    chunk_data, epw_data, filtered_weather, filtered_datetimes, analysis_period, target_hours, location, config, body_params = args
    
    results = {}
    
    for i, exposure in chunk_data:
        # Ensure exposure arrays match filtered weather length
        if len(exposure.fract_body_exp) != len(filtered_datetimes):
            import warnings
            warnings.warn(f"Exposure array length ({len(exposure.fract_body_exp)}) "
                        f"doesn't match weather data length ({len(filtered_datetimes)})")
            # Truncate or pad as needed
            min_len = min(len(exposure.fract_body_exp), len(filtered_datetimes))
            fract_exp = exposure.fract_body_exp[:min_len]
            if len(filtered_datetimes) > min_len:
                # Pad weather data
                for key in filtered_weather:
                    filtered_weather[key] = filtered_weather[key][:min_len]
                filtered_datetimes = filtered_datetimes[:min_len]
        else:
            fract_exp = exposure.fract_body_exp
        
        # Compute SolarCal MRT using EPW data collections for proper filtering
        if hasattr(epw_data, 'dry_bulb_temperature'):
            # Use EPW object directly for proper data collection handling
            mrt_result = _create_solarcal_from_epw(
                epw_data, exposure, analysis_period, target_hours,
                config.ground_reflectance, body_params
            )
        else:
            # Fallback for DataFrame input
            from .solarcal import compute_mrt_solarcal
            mrt_result = compute_mrt_solarcal(
                air_temperature=filtered_weather['air_temp'],
                direct_normal_rad=filtered_weather['dir_norm_rad'],
                diffuse_horizontal_rad=filtered_weather['diff_horiz_rad'],
                horizontal_infrared_rad=filtered_weather['horiz_ir_rad'],
                fract_body_exp=fract_exp,
                sky_exposure=exposure.sky_exposure,
                location=location,
                datetimes=filtered_datetimes,
                ground_reflectance=config.ground_reflectance,
                solar_body_par=body_params
            )
        
        # Create boundary arrays for averaging
        mrt0, mrt1 = MRTCalculator._create_boundary_arrays(mrt_result.mrt)
        n_hours = len(mrt0)
        
        results[f'position_{i}'] = {
            'position': exposure.position,
            'mrt0': mrt0,
            'mrt1': mrt1,
            'mrt': mrt0,  # Backward compatibility
            'short_erf': mrt_result.short_erf,
            'long_erf': mrt_result.long_erf,
            'short_dmrt': mrt_result.short_dmrt,
            'long_dmrt': mrt_result.long_dmrt,
            'fract_body_exp': fract_exp[:n_hours],
            'sky_exposure': exposure.sky_exposure,
            'datetimes': filtered_datetimes if not hasattr(epw_data, 'dry_bulb_temperature') else [None]
        }
    
    return results


