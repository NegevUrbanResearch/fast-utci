"""
Human exposure calculations for MRT computation.

Computes solar exposure (fract_body_exp) and sky exposure via ray testing
against context geometry, matching Grasshopper Human-to-Sky Relation component.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
import warnings

from .solar import SunData, get_tregenza_dome_vectors
from .mesh import MeshContext, batch_ray_intersections
from .config import DEFAULT_PT_COUNT, DEFAULT_HUMAN_HEIGHT, DEFAULT_BATCH_SIZE

# Global cache for sky vectors (computed once, reused across all positions)
_sky_vectors_cache = None
_sky_weights_cache = None

def _get_cached_sky_vectors():
    """Get cached sky vectors, computing them once if needed."""
    global _sky_vectors_cache, _sky_weights_cache
    if _sky_vectors_cache is None:
        _sky_vectors_cache, _sky_weights_cache = get_tregenza_dome_vectors()
        # Ensure optimal memory layout for better cache performance
        _sky_vectors_cache = np.ascontiguousarray(_sky_vectors_cache, dtype=np.float64)
        _sky_weights_cache = np.ascontiguousarray(_sky_weights_cache, dtype=np.float64)
    return _sky_vectors_cache, _sky_weights_cache

def _get_optimal_batch_size(n_rays: int, ray_type: str = "mixed") -> int:
    """
    Calculate optimal batch size based on ray count, type, and available memory.
    
    Args:
        n_rays: Number of rays to process
        ray_type: Type of rays ("sky", "solar", or "mixed")
        
    Returns:
        Optimal batch size that balances performance and memory safety
    """
    import psutil
    
    # Base batch sizes optimized for different ray types
    if ray_type == "sky":
        # Sky rays: typically 145 rays per position, can use larger batches
        base_batch_size = 20000
    elif ray_type == "solar":
        # Solar rays: typically 1 ray per position, use smaller batches
        base_batch_size = 5000
    else:
        # Mixed or unknown: use default
        base_batch_size = DEFAULT_BATCH_SIZE
    
    try:
        # Get available memory (in bytes)
        available_memory = psutil.virtual_memory().available
        
        # Estimate memory needed per ray (origins + directions + results)
        # 3 floats for origin + 3 floats for direction + 1 bool for result = 7 * 8 bytes
        memory_per_ray = 7 * 8  # 56 bytes per ray
        
        # Use 25% of available memory for ray processing (more generous than before)
        max_memory_for_rays = available_memory * 0.25
        
        # Calculate maximum safe batch size
        max_safe_batch = int(max_memory_for_rays / memory_per_ray)
        
        # Use the smaller of: base_batch_size, max_safe_batch, or n_rays
        optimal_batch_size = min(base_batch_size, max_safe_batch, n_rays)
        
        # Ensure minimum batch size for efficiency
        min_batch_size = 100 if ray_type == "solar" else 500
        optimal_batch_size = max(min_batch_size, optimal_batch_size)
        
        return optimal_batch_size
        
    except Exception:
        # Fallback to conservative batch size if memory detection fails
        return min(base_batch_size, n_rays, 5000)


@dataclass
class ExposureResult:
    """Container for exposure calculation results."""
    fract_body_exp: np.ndarray  # Shape: (n_hours,) fraction of body exposed to sun per hour
    sky_exposure: float         # Scalar fraction of visible sky (0-1)
    position: np.ndarray        # Shape: (3,) analysis position at ground level
    sample_points: np.ndarray   # Shape: (pt_count, 3) human body sample points used


def create_human_sample_points(position: np.ndarray,
                              pt_count: int = DEFAULT_PT_COUNT,
                              height: float = DEFAULT_HUMAN_HEIGHT) -> np.ndarray:
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
                          show_progress: bool = True) -> np.ndarray:
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
            ray_directions[:] = -sun_vector  # Broadcast -sun_vector to all rows
            
            # Test ray intersections with optimized batch sizing
            optimal_batch_size = _get_optimal_batch_size(n_points, "solar")
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


def compute_sky_exposure(sample_points: np.ndarray,
                        mesh_context: Optional[MeshContext] = None,
                        show_progress: bool = True) -> float:
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
    sky_vectors, sky_weights = _get_cached_sky_vectors()
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
        optimal_batch_size = _get_optimal_batch_size(n_sky_patches, "sky")
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
                    pt_count: int = DEFAULT_PT_COUNT,
                    height: float = DEFAULT_HUMAN_HEIGHT,
                    show_progress: bool = True) -> ExposureResult:
    """
    Compute both solar and sky exposure for a single position.
    
    Args:
        position: Ground position (x, y, z)
        sun_data: Solar vectors and timing data
        mesh_context: Optional context geometry for occlusion testing
        pt_count: Number of sample points along human height
        height: Human height in meters
        show_progress: Whether to show progress bars
        
    Returns:
        ExposureResult with solar and sky exposure data
    """
    # Create human sample points
    sample_points = create_human_sample_points(position, pt_count, height)
    
    # Compute solar exposure (time series)
    fract_body_exp = compute_solar_exposure(
        sample_points, sun_data, mesh_context, show_progress
    )
    
    # Compute sky exposure (scalar)
    sky_exposure = compute_sky_exposure(
        sample_points, mesh_context, show_progress
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
                          pt_count: int = DEFAULT_PT_COUNT,
                          height: float = DEFAULT_HUMAN_HEIGHT,
                          show_progress: bool = True,
                          n_workers: Optional[int] = None) -> List[ExposureResult]:
    """
    Compute exposure for multiple positions with progress tracking and parallel processing.
    
    Args:
        positions: Shape (n_positions, 3) analysis positions
        sun_data: Solar vectors and timing data
        mesh_context: Optional context geometry for occlusion testing
        pt_count: Number of sample points along human height
        height: Human height in meters
        show_progress: Whether to show progress bars
        n_workers: Number of parallel workers (default: CPU count - 1)
        
    Returns:
        List of ExposureResult objects, one per position
    """
    n_positions = len(positions)
    
    # Use serial processing for small datasets or when no context
    if n_positions < 100 or mesh_context is None:
        return _compute_exposure_serial(positions, sun_data, mesh_context, pt_count, height, show_progress)
    
    # Use parallel processing for larger datasets
    return _compute_exposure_parallel(positions, sun_data, mesh_context, pt_count, height, show_progress, n_workers)


def _compute_exposure_serial(positions: np.ndarray,
                           sun_data: SunData,
                           mesh_context: Optional[MeshContext],
                           pt_count: int,
                           height: float,
                           show_progress: bool) -> List[ExposureResult]:
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
            show_progress=False
        )
        
        results.append(result)
    
    return results


def _compute_exposure_parallel(positions: np.ndarray,
                             sun_data: SunData,
                             mesh_context: MeshContext,
                             pt_count: int,
                             height: float,
                             show_progress: bool,
                             n_workers: Optional[int]) -> List[ExposureResult]:
    """Parallel exposure computation for larger datasets."""
    import multiprocessing as mp
    from multiprocessing import Pool, Queue
    import time
    
    n_positions = len(positions)
    
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)
    
    # Sort positions spatially for better cache locality when accessing mesh data
    # Sort by X coordinate first, then Y, then Z for consistent ordering
    sorted_indices = np.lexsort((positions[:, 2], positions[:, 1], positions[:, 0]))
    sorted_positions = positions[sorted_indices]
    
    # Create chunks with improved load balancing
    # Use dynamic chunk sizing to ensure more even distribution
    base_chunk_size = max(100, n_positions // n_workers)
    chunks = []
    
    # Distribute positions more evenly across workers
    positions_per_worker = n_positions // n_workers
    extra_positions = n_positions % n_workers
    
    start_idx = 0
    for worker_id in range(n_workers):
        # Some workers get one extra position for better load balancing
        chunk_size = positions_per_worker + (1 if worker_id < extra_positions else 0)
        end_idx = start_idx + chunk_size
        
        if start_idx < n_positions:
            chunks.append(sorted_positions[start_idx:end_idx])
            start_idx = end_idx
    
    print(f"Processing {n_positions} positions with {n_workers} workers in {len(chunks)} chunks")
    
    # Process chunks in parallel
    if show_progress:
        # Use simpler progress tracking without Queue (avoid multiprocessing issues)
        chunk_args = [(chunk, sun_data, mesh_context, pt_count, height, None) for chunk in chunks]
        
        with Pool(processes=n_workers) as pool:
            # Use imap for progress tracking
            results = []
            start_time = time.time()
            
            with tqdm(total=n_positions, desc="Computing exposure", unit="pos", 
                     mininterval=1.0, maxinterval=5.0, smoothing=0.1, leave=True) as pbar:
                
                for chunk_results in pool.imap(_compute_exposure_chunk, chunk_args):
                    results.extend(chunk_results)
                    chunk_size = len(chunk_results)
                    pbar.update(chunk_size)
                    
                    # Update description with time estimate
                    elapsed = time.time() - start_time
                    if elapsed > 0:
                        rate = pbar.n / elapsed
                        eta = (n_positions - pbar.n) / rate if rate > 0 else 0
                        pbar.set_description(f"Computing exposure ({rate:.1f} pos/s, ETA: {eta:.0f}s)")
                
    else:
        # No progress bar - use simple processing
        chunk_args = [(chunk, sun_data, mesh_context, pt_count, height, None) for chunk in chunks]
        with Pool(processes=n_workers) as pool:
            chunk_results_list = pool.map(_compute_exposure_chunk, chunk_args)
            results = []
            for chunk_results in chunk_results_list:
                results.extend(chunk_results)
    
    return results


def _compute_exposure_chunk(args):
    """Worker function for parallel processing of position chunks."""
    chunk_positions, sun_data, mesh_context, pt_count, height, _ = args
    n_positions = len(chunk_positions)
    
    # Pre-allocate results list for better performance
    results = [None] * n_positions
    
    for i, position in enumerate(chunk_positions):
        result = compute_exposure(
            position=position,
            sun_data=sun_data,
            mesh_context=mesh_context,
            pt_count=pt_count,
            height=height,
            show_progress=False  # No progress bars in worker processes
        )
        results[i] = result
    
    return results
