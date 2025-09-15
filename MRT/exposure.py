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


@dataclass
class ExposureResult:
    """Container for exposure calculation results."""
    fract_body_exp: np.ndarray  # Shape: (n_hours,) fraction of body exposed to sun per hour
    sky_exposure: float         # Scalar fraction of visible sky (0-1)
    position: np.ndarray        # Shape: (3,) analysis position at ground level
    sample_points: np.ndarray   # Shape: (pt_count, 3) human body sample points used


def create_human_sample_points(position: np.ndarray,
                              pt_count: int = 1,
                              height: float = 1.8) -> np.ndarray:
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
    
    if pt_count == 1:
        # Single point at mid-height
        sample_z = position[2] + height / 2
        return np.array([[position[0], position[1], sample_z]])
    else:
        # Multiple points distributed along height
        z_offsets = np.linspace(height * 0.1, height * 0.9, pt_count)
        sample_points = []
        
        for z_offset in z_offsets:
            sample_z = position[2] + z_offset
            sample_points.append([position[0], position[1], sample_z])
            
        return np.array(sample_points)


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
            # Test occlusion for each sample point
            # Rays point FROM sample points TOWARD sun (reverse of sun vector)
            ray_directions = np.tile(-sun_vector, (n_points, 1))
            
            # Test ray intersections
            hits = batch_ray_intersections(
                origins=sample_points,
                directions=ray_directions,
                mesh_context=mesh_context,
                batch_size=min(1000, n_points)
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
    
    # Get Tregenza dome vectors and weights
    sky_vectors, sky_weights = get_tregenza_dome_vectors()
    n_sky_patches = len(sky_vectors)
    
    # Test sky visibility for each sample point
    total_visible_weight = 0.0
    total_weight = np.sum(sky_weights)
    
    point_iter = range(n_points)
    if show_progress and n_points > 1:
        point_iter = tqdm(point_iter, desc="Computing sky exposure", unit="points")
    
    for point_idx in point_iter:
        point = sample_points[point_idx:point_idx+1]  # Keep 2D shape
        
        # Create rays pointing to each sky patch
        ray_origins = np.tile(point, (n_sky_patches, 1))
        ray_directions = sky_vectors
        
        # Test intersections
        hits = batch_ray_intersections(
            origins=ray_origins,
            directions=ray_directions,
            mesh_context=mesh_context,
            batch_size=min(1000, n_sky_patches)
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
                    pt_count: int = 1,
                    height: float = 1.8,
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
                          pt_count: int = 1,
                          height: float = 1.8,
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
    
    # For small datasets, use serial processing to avoid overhead
    if n_positions < 50 or mesh_context is None:
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
        position_iter = tqdm(position_iter, desc="Computing exposure (serial)", unit="pos")
    
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
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    n_positions = len(positions)
    
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)
    
    # Split positions into chunks for parallel processing
    chunk_size = max(1, n_positions // (n_workers * 2))  # 2x workers for better load balancing
    chunks = []
    
    for i in range(0, n_positions, chunk_size):
        end_idx = min(i + chunk_size, n_positions)
        chunks.append((i, positions[i:end_idx]))
    
    print(f"Processing {n_positions} positions with {n_workers} workers in {len(chunks)} chunks")
    
    # Process chunks in parallel
    results = [None] * n_positions  # Pre-allocate results list
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all chunks
        future_to_chunk = {}
        for chunk_idx, (start_idx, chunk_positions) in enumerate(chunks):
            future = executor.submit(
                _process_exposure_chunk,
                chunk_positions,
                sun_data,
                mesh_context,
                pt_count,
                height,
                chunk_idx
            )
            future_to_chunk[future] = (chunk_idx, start_idx, len(chunk_positions))
        
        # Collect results with progress tracking
        completed_positions = 0
        
        if show_progress:
            pbar = tqdm(total=n_positions, desc="Computing exposure (parallel)", unit="pos")
        
        for future in as_completed(future_to_chunk):
            chunk_idx, start_idx, chunk_size = future_to_chunk[future]
            
            try:
                chunk_results = future.result()
                
                # Store results in correct positions
                for i, result in enumerate(chunk_results):
                    results[start_idx + i] = result
                
                completed_positions += len(chunk_results)
                
                if show_progress:
                    pbar.update(len(chunk_results))
                    
            except Exception as e:
                print(f"Chunk {chunk_idx} failed: {e}")
                # Fill with dummy results to maintain list structure
                for i in range(chunk_size):
                    dummy_result = ExposureResult(
                        fract_body_exp=np.array([0.0]),
                        sky_exposure=0.0,
                        position=positions[start_idx + i],
                        sample_points=np.array([[0, 0, 0]])
                    )
                    results[start_idx + i] = dummy_result
                
                if show_progress:
                    pbar.update(chunk_size)
        
        if show_progress:
            pbar.close()
    
    # Filter out any None results
    final_results = [r for r in results if r is not None]
    
    if len(final_results) != n_positions:
        print(f"Warning: Expected {n_positions} results, got {len(final_results)}")
    
    return final_results


def _process_exposure_chunk(positions: np.ndarray,
                          sun_data: SunData,
                          mesh_context: MeshContext,
                          pt_count: int,
                          height: float,
                          chunk_idx: int) -> List[ExposureResult]:
    """Process a chunk of positions in a separate process."""
    try:
        chunk_results = []
        
        for position in positions:
            result = compute_exposure(
                position=position,
                sun_data=sun_data,
                mesh_context=mesh_context,
                pt_count=pt_count,
                height=height,
                show_progress=False  # No progress bars in worker processes
            )
            chunk_results.append(result)
        
        return chunk_results
        
    except Exception as e:
        print(f"Error in chunk {chunk_idx}: {e}")
        # Return dummy results to maintain structure
        return [
            ExposureResult(
                fract_body_exp=np.array([0.0]),
                sky_exposure=0.0,
                position=pos,
                sample_points=np.array([[0, 0, 0]])
            )
            for pos in positions
        ]
