"""
Human exposure calculations for MRT computation.

Computes solar exposure (fract_body_exp) and sky exposure via ray testing
against context geometry, matching Grasshopper Human-to-Sky Relation component.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from functools import partial
from tqdm import tqdm
import warnings
import logging

from .solar import SunData
from .mesh import MeshContext, batch_ray_intersections
from fast_utci.shared import MRTConfig
from .cache import get_cached_sky_vectors
from .performance import get_optimal_batch_size

logger = logging.getLogger(__name__)


@dataclass
class ExposureResult:
    """Container for exposure calculation results."""
    fract_body_exp: np.ndarray  # Shape: (n_hours,) fraction of body exposed to sun per hour
    sky_exposure: float         # Scalar fraction of visible sky (0-1)
    position: np.ndarray        # Shape: (3,) analysis position at ground level
    sample_points: np.ndarray   # Shape: (pt_count, 3) human body sample points used


def create_human_sample_points(position: np.ndarray,
                              pt_count: int,
                              height: float) -> np.ndarray:
    """
    Create vertical sample points representing human body for exposure testing.
    
    Args:
        position: Shape (3,) ground position (x, y, z)
        pt_count: Number of sample points along human height
        height: Human height in meters
        
    Returns:
        Sample points array of shape (pt_count, 3)
    """
    position = np.asarray(position)
    
    # Pre-allocate array for better performance
    sample_points = np.empty((pt_count, 3), dtype=np.float64)
    
    if pt_count == 1:
        # Single point at mid-height
        sample_points[0, 0] = position[0]
        sample_points[0, 1] = position[1]
        sample_points[0, 2] = position[2] + height / 2
    else:
        # Multiple points distributed along height
        z_offsets = np.linspace(height * 0.1, height * 0.9, pt_count)
        
        # Fill X and Y coordinates (same for all points)
        sample_points[:, 0] = position[0]
        sample_points[:, 1] = position[1]
        
        # Fill Z coordinates (distributed along height)
        sample_points[:, 2] = position[2] + z_offsets
        
    return sample_points


def compute_solar_exposure(sample_points: np.ndarray,
                          sun_data: SunData,
                          mesh_context: Optional[MeshContext] = None,
                          show_progress: bool = True,
                          config: Optional[MRTConfig] = None) -> np.ndarray:
    """
    Compute fraction of body exposed to direct sun for each hour.
    
    Args:
        sample_points: Shape (pt_count, 3) human body sample points
        sun_data: Solar vectors and timing data
        mesh_context: Optional context geometry for occlusion testing
        show_progress: Whether to show progress bar
        
    Returns:
        Array of shape (n_hours,) with fraction exposed per hour (0-1)
    """
    # Vectorized pathway via config
    if config and config.vectorized_solar:
        return _compute_solar_exposure_vectorized(sample_points, sun_data, mesh_context, show_progress, config)

    n_hours = len(sun_data.sun_vectors)
    n_points = len(sample_points)
    fract_body_exp = np.zeros(n_hours)
    
    # Process hours with progress bar
    hour_iter = range(n_hours)
    if show_progress:
        hour_iter = tqdm(hour_iter, desc="Computing solar exposure", unit="hours")
    
    for hour_idx in hour_iter:
        if not sun_data.is_sun_up[hour_idx]:
            # Sun is down, no exposure
            fract_body_exp[hour_idx] = 0.0
            continue
            
        sun_vector = sun_data.sun_vectors[hour_idx]
        
        if mesh_context is None:
            # No occlusion context - full exposure when sun is up
            fract_body_exp[hour_idx] = 1.0
        else:
            # Pre-allocate ray directions array (more efficient than np.tile)
            ray_directions = np.empty((n_points, 3), dtype=np.float64)
            ray_directions[:] = sun_vector  # Ray FROM position TO sun (matches Grasshopper sun_vector_reversed)
            
            # Test ray intersections with optimized batch sizing
            optimal_batch_size = get_optimal_batch_size(n_points, "solar", config)
            hits = batch_ray_intersections(
                origins=sample_points,
                directions=ray_directions,
                mesh_context=mesh_context,
                batch_size=optimal_batch_size
            )
            
            # Fraction of points NOT occluded (visible to sun)
            visible_points = np.sum(~hits)
            fract_body_exp[hour_idx] = visible_points / n_points
    
    return fract_body_exp
def _compute_solar_exposure_vectorized(sample_points: np.ndarray,
                                      sun_data: SunData,
                                      mesh_context: Optional[MeshContext],
                                      show_progress: bool,
                                      config: Optional[MRTConfig] = None) -> np.ndarray:
    """
    Vectorized solar exposure across multiple hours to reduce Python overhead
    and improve Embree throughput by using larger ray batches.
    """
    n_hours = len(sun_data.sun_vectors)
    n_points = len(sample_points)
    fract_body_exp = np.zeros(n_hours)

    # Quick exits
    if mesh_context is None:
        # Full exposure for all sun-up hours; zeros otherwise
        sun_up = np.asarray(sun_data.is_sun_up, dtype=bool)
        fract_body_exp[sun_up] = 1.0
        return fract_body_exp

    # Indices where sun is up
    sun_up_indices = np.flatnonzero(sun_data.is_sun_up)
    if sun_up_indices.size == 0:
        return fract_body_exp

    # Choose a reasonable hour batch size to control memory usage
    # Each hour batch allocates (batch_hours * n_points) rays
    # Start with ~8 hours per batch, adjust if very large n_points
    base_hours_per_batch = 8
    if n_points > 500:
        base_hours_per_batch = 4
    if n_points > 2000:
        base_hours_per_batch = 2

    hour_batches = np.array_split(sun_up_indices, max(1, int(np.ceil(len(sun_up_indices) / base_hours_per_batch))))

    batch_iter = hour_batches
    if show_progress:
        batch_iter = tqdm(hour_batches, desc="Computing solar exposure (vect)", unit="batches")

    for hour_batch in batch_iter:
        if len(hour_batch) == 0:
            continue

        # Directions: for each hour in batch, we need n_points rays towards sun
        batch_vectors = sun_data.sun_vectors[hour_batch]
        # Create all origins and directions for this batch
        # Origins: repeat sample_points for each hour
        all_origins = np.tile(sample_points, (len(hour_batch), 1))
        # Directions: for each hour vector, repeat it n_points times (FROM position TO sun)
        all_directions = np.repeat(batch_vectors, n_points, axis=0)

        optimal_batch_size = get_optimal_batch_size(len(all_origins), "solar", config)
        hits = batch_ray_intersections(
            origins=all_origins,
            directions=all_directions,
            mesh_context=mesh_context,
            batch_size=optimal_batch_size
        )

        # Reshape to [hours_in_batch, n_points]
        hits_reshaped = hits.reshape(len(hour_batch), n_points)
        # Fraction of points NOT occluded
        visible_frac = 1.0 - np.mean(hits_reshaped, axis=1)
        fract_body_exp[hour_batch] = visible_frac

    return fract_body_exp


def compute_sky_exposure(sample_points: np.ndarray,
                        mesh_context: Optional[MeshContext] = None,
                        show_progress: bool = True,
                        config: Optional[MRTConfig] = None) -> float:
    """
    Compute fraction of sky visible from sample points using Tregenza dome.
    
    Args:
        sample_points: Shape (pt_count, 3) human body sample points
        mesh_context: Optional context geometry for occlusion testing
        show_progress: Whether to show progress bar
        
    Returns:
        Sky exposure fraction (0-1)
    """
    n_points = len(sample_points)
    
    if mesh_context is None:
        # No occlusion - full sky exposure
        return 1.0
    
    # Get cached Tregenza dome vectors and weights (computed once globally)
    sky_vectors, sky_weights = get_cached_sky_vectors()
    n_sky_patches = len(sky_vectors)
    
    # Pre-allocate arrays for better performance
    total_visible_weight = 0.0
    total_weight = np.sum(sky_weights)
    
    # Pre-allocate ray origins array (reused for each point)
    ray_origins = np.empty((n_sky_patches, 3), dtype=np.float64)
    
    point_iter = range(n_points)
    if show_progress and n_points > 1:
        point_iter = tqdm(point_iter, desc="Computing sky exposure", unit="points")
    
    for point_idx in point_iter:
        point = sample_points[point_idx]
        
        # Fill pre-allocated ray origins array (more efficient than np.tile)
        ray_origins[:] = point  # Broadcast point to all rows
        
        # Test intersections (sky_vectors are already pre-computed)
        # Test intersections with sky patches using optimized batch sizing
        optimal_batch_size = get_optimal_batch_size(n_sky_patches, "sky", config)
        hits = batch_ray_intersections(
            origins=ray_origins,
            directions=sky_vectors,
            mesh_context=mesh_context,
            batch_size=optimal_batch_size
        )
        
        # Sum weights of visible (non-occluded) sky patches
        visible_weights = sky_weights[~hits]
        total_visible_weight += np.sum(visible_weights)
    
    # Average across sample points
    avg_visible_weight = total_visible_weight / n_points
    sky_exposure = avg_visible_weight / total_weight
    
    return float(sky_exposure)


def compute_exposure(position: np.ndarray,
                    sun_data: SunData,
                    mesh_context: Optional[MeshContext] = None,
                    pt_count: Optional[int] = None,
                    height: Optional[float] = None,
                    show_progress: bool = True,
                    config: Optional[MRTConfig] = None) -> ExposureResult:
    """
    Compute both solar and sky exposure for a single position.
    
    Args:
        position: Ground position (x, y, z)
        sun_data: Solar vectors and timing data
        mesh_context: Optional context geometry for occlusion testing
        pt_count: Number of sample points along human height (uses config if None)
        height: Human height in meters (uses config if None)
        show_progress: Whether to show progress bars
        config: Optional MRTConfig (required if pt_count/height not provided)
        
    Returns:
        ExposureResult with solar and sky exposure data
    """
    # Use config values if not explicitly provided
    if pt_count is None:
        if config is None:
            raise ValueError("pt_count must be provided or config must be supplied")
        pt_count = config.pt_count
    if height is None:
        if config is None:
            raise ValueError("height must be provided or config must be supplied")
        height = config.human_height
    
    # Create human sample points
    sample_points = create_human_sample_points(position, pt_count, height)
    
    # Compute solar exposure (time series)
    fract_body_exp = compute_solar_exposure(
        sample_points, sun_data, mesh_context, show_progress, config
    )
    
    # Compute sky exposure (scalar)
    sky_exposure = compute_sky_exposure(
        sample_points, mesh_context, show_progress, config
    )
    
    return ExposureResult(
        fract_body_exp=fract_body_exp,
        sky_exposure=sky_exposure,
        position=np.asarray(position),
        sample_points=sample_points
    )


def compute_exposure_batch(positions: np.ndarray,
                          sun_data: SunData,
                          mesh_context: Optional[MeshContext] = None,
                          pt_count: Optional[int] = None,
                          height: Optional[float] = None,
                          show_progress: bool = True,
                          n_workers: Optional[int] = None,
                          config: Optional[MRTConfig] = None) -> List[ExposureResult]:
    """
    Compute exposure for multiple positions with progress tracking and parallel processing.
    
    Args:
        positions: Shape (n_positions, 3) analysis positions
        sun_data: Solar vectors and timing data
        mesh_context: Optional context geometry for occlusion testing
        pt_count: Number of sample points along human height (uses config if None)
        height: Human height in meters (uses config if None)
        show_progress: Whether to show progress bars
        n_workers: Number of parallel workers (default: CPU count - 1)
        config: Optional MRTConfig (required if pt_count/height not provided)
        
    Returns:
        List of ExposureResult objects, one per position
    """
    # Use config values if not explicitly provided
    if pt_count is None:
        if config is None:
            raise ValueError("pt_count must be provided or config must be supplied")
        pt_count = config.pt_count
    if height is None:
        if config is None:
            raise ValueError("height must be provided or config must be supplied")
        height = config.human_height
    
    n_positions = len(positions)
    
    # Use serial processing for small datasets or when no context
    if n_positions < 100 or mesh_context is None:
        return _compute_exposure_serial(positions, sun_data, mesh_context, pt_count, height, show_progress, config)
    
    # Use parallel processing for larger datasets
    return _compute_exposure_parallel(positions, sun_data, mesh_context, pt_count, height, show_progress, n_workers, config)


def _compute_exposure_serial(positions: np.ndarray,
                           sun_data: SunData,
                           mesh_context: Optional[MeshContext],
                           pt_count: int,
                           height: float,
                           show_progress: bool,
                           config: Optional[MRTConfig]) -> List[ExposureResult]:
    """Serial exposure computation for small datasets."""
    n_positions = len(positions)
    results = []
    
    position_iter = range(n_positions)
    if show_progress:
        # Use minimal progress bar settings to reduce overhead
        position_iter = tqdm(position_iter, desc="Computing exposure (serial)", 
                           unit="pos", mininterval=1.0, maxinterval=5.0, 
                           smoothing=0.1, leave=False)
    
    for pos_idx in position_iter:
        position = positions[pos_idx]
        
        result = compute_exposure(
            position=position,
            sun_data=sun_data,
            mesh_context=mesh_context,
            pt_count=pt_count,
            height=height,
            show_progress=False,
            config=config
        )
        
        results.append(result)
    
    return results


def _compute_exposure_parallel(positions: np.ndarray,
                             sun_data: SunData,
                             mesh_context: MeshContext,
                             pt_count: int,
                             height: float,
                             show_progress: bool,
                             n_workers: Optional[int],
                             config: Optional[MRTConfig]) -> List[ExposureResult]:
    """Parallel exposure computation for larger datasets using ParallelProcessor with SpatialChunkStrategy."""
    n_positions = len(positions)
    
    # Get parallel config from MRT config or create default
    if config is not None:
        parallel_config = config.parallel.with_overrides(
            n_workers=n_workers,
            show_progress=show_progress
        )
    else:
        # Fallback for when config is None (shouldn't happen, but handle gracefully)
        import multiprocessing as mp
        from fast_utci.shared import ParallelConfig
        if n_workers is None:
            n_workers = max(1, mp.cpu_count() - 1)
        parallel_config = ParallelConfig(
            n_workers=n_workers,
            show_progress=show_progress,
            parallel_threshold=100
        )
    
    # Use functools.partial to create a picklable worker function
    worker_fn = partial(
        _worker_exposure_chunk,
        sun_data=sun_data,
        mesh_context=mesh_context,
        pt_count=pt_count,
        height=height,
        config=config
    )
    
    # Process and merge results automatically
    from fast_utci.shared import ParallelProcessor, SpatialChunkStrategy
    processor = ParallelProcessor(parallel_config=parallel_config)
    logger.info(f"Processing {n_positions} positions with {processor.n_workers} workers")
    
    return processor.process_and_merge(
        data=positions,
        worker_fn=worker_fn,
        chunk_strategy=SpatialChunkStrategy(),
        description="Computing exposure",
        merge_list=True
    )


def _worker_exposure_chunk(chunk: np.ndarray, sun_data: SunData, mesh_context: MeshContext,
                           pt_count: int, height: float, config: Optional[MRTConfig]) -> List[ExposureResult]:
    """
    Module-level worker function for parallel processing of position chunks.
    
    This function must be at module level to be picklable for multiprocessing.
    """
    # Prepare args in the format expected by _compute_exposure_chunk
    chunk_args = (
        chunk,
        sun_data,
        mesh_context,
        pt_count,
        height,
        None,  # progress_queue (not used)
        config
    )
    
    # Use existing standalone worker function
    return _compute_exposure_chunk(chunk_args)


def _compute_exposure_chunk(args):
    """Worker function for parallel processing of position chunks."""
    chunk_positions, sun_data, mesh_context, pt_count, height, _, config = args
    n_positions = len(chunk_positions)

    # Fast path: when pt_count==1 and env toggle is on, batch solar rays across positions per hour
    if config and config.batch_positions and pt_count == 1 and mesh_context is not None:
        # Prepare once
        positions = np.asarray(chunk_positions)
        # Create sample points: one per position at mid-height
        sample_points = positions.copy()
        sample_points[:, 2] = positions[:, 2] + height / 2.0

        # Compute solar exposure across positions by batching per hour
        n_hours = len(sun_data.sun_vectors)
        fract_body_exp_all = np.zeros((n_positions, n_hours))

        # For each hour, shoot one ray per position
        sun_up = np.asarray(sun_data.is_sun_up, dtype=bool)
        sun_up_indices = np.flatnonzero(sun_up)
        for hour_idx in sun_up_indices:
            sun_vec = sun_data.sun_vectors[hour_idx]
            ray_dirs = np.empty((n_positions, 3), dtype=np.float64)
            ray_dirs[:] = sun_vec  # Ray FROM position TO sun
            # Batch intersections in big chunks
            optimal_batch_size = get_optimal_batch_size(n_positions, "solar", config)
            hits = batch_ray_intersections(
                origins=sample_points,
                directions=ray_dirs,
                mesh_context=mesh_context,
                batch_size=optimal_batch_size
            )
            # Visible if no hit
            fract_body_exp_all[:, hour_idx] = (~hits).astype(float)

        # Sky exposure per position (still per position; could be batched further later)
        results = [None] * n_positions
        for i in range(n_positions):
            pos = positions[i]
            # One sample point array of shape (1,3)
            sp = np.array([[pos[0], pos[1], pos[2] + height / 2.0]], dtype=np.float64)
            sky = compute_sky_exposure(sp, mesh_context, show_progress=False, config=config)
            results[i] = ExposureResult(
                fract_body_exp=fract_body_exp_all[i],
                sky_exposure=sky,
                position=pos,
                sample_points=sp
            )
        return results

    # Default path: per-position processing
    results = [None] * n_positions
    for i, position in enumerate(chunk_positions):
        result = compute_exposure(
            position=position,
            sun_data=sun_data,
            mesh_context=mesh_context,
            pt_count=pt_count,
            height=height,
            show_progress=False,
            config=config
        )
        results[i] = result

    return results
