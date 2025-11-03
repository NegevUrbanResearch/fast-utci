"""
Context mesh handling with BVH acceleration for fast ray intersections.

Provides fast ray-mesh intersection testing for occlusion calculations
in MRT exposure computation.
"""

import numpy as np
import trimesh
from typing import List, Union, Optional, Tuple, Any
from dataclasses import dataclass
import warnings
import logging

from fast_utci.shared import MRTConfig
from .adapters import create_intersector_strategy

logger = logging.getLogger(__name__)


@dataclass
class MeshContext:
    """Container for context geometry with acceleration structures."""
    mesh: trimesh.Trimesh
    has_bvh: bool = False
    backend_name: str = "unknown"
    ray_intersector: Optional[Any] = None
    
    def __post_init__(self):
        """Initialize ray intersector using strategy pattern."""
        # Defer to default configuration when not provided explicitly
        mrt_config = getattr(self, "_mrt_config", None)
        
        try:
            strategy = create_intersector_strategy(
                self.mesh,
                intersector_choice=(mrt_config.intersector if mrt_config else "auto"),
                embree_quality=(mrt_config.embree_quality if mrt_config else "medium"),
                embree_build_bvh=(mrt_config.embree_build_bvh if mrt_config else True),
                embree_packet_size=(mrt_config.embree_packet_size if mrt_config else 0)
            )
            
            self.ray_intersector = strategy.get_intersector()
            self.backend_name = strategy.get_name()
            self.has_bvh = strategy.get_intersector() is not None
            
            logger.info(f"Ray intersector: {self.backend_name}")
            
        except Exception as e:
            warnings.warn(f"Intersector initialization failed: {e}")
            self.ray_intersector = None
            self.backend_name = "none"
            self.has_bvh = False


def load_context_meshes(mesh_sources: List[Union[str, trimesh.Trimesh]], config: Optional[MRTConfig] = None) -> MeshContext:
    """
    Load and combine context meshes into a single mesh with BVH acceleration.
    
    Args:
        mesh_sources: List of file paths or trimesh.Trimesh objects
        
    Returns:
        MeshContext with combined mesh and acceleration info
    """
    meshes = []
    
    for source in mesh_sources:
        if isinstance(source, str):
            # Load from file
            loaded = trimesh.load(source)
            if isinstance(loaded, trimesh.Trimesh):
                meshes.append(loaded)
            elif isinstance(loaded, trimesh.Scene):
                # Extract all meshes from scene
                for geom in loaded.geometry.values():
                    if isinstance(geom, trimesh.Trimesh):
                        meshes.append(geom)
            else:
                warnings.warn(f"Unsupported mesh type from {source}: {type(loaded)}")
        elif isinstance(source, trimesh.Trimesh):
            meshes.append(source)
        else:
            warnings.warn(f"Unsupported mesh source type: {type(source)}")
    
    assert meshes, "No valid meshes found in sources"
    
    # Combine all meshes
    if len(meshes) == 1:
        combined_mesh = meshes[0]
    else:
        combined_mesh = trimesh.util.concatenate(meshes)
    
    ctx = MeshContext(mesh=combined_mesh)
    # Inject config for intersector init
    if config is not None:
        setattr(ctx, "_mrt_config", config)
        ctx.__post_init__()
    return ctx


def ray_mesh_intersections(origins: np.ndarray,
                          directions: np.ndarray,
                          mesh_context: MeshContext,
                          max_distance: Optional[float] = None) -> np.ndarray:
    """
    Test ray-mesh intersections for occlusion detection.
    
    Args:
        origins: Shape (n_rays, 3) ray starting points
        directions: Shape (n_rays, 3) ray directions (should be unit vectors)
        mesh_context: MeshContext with geometry and acceleration
        max_distance: Maximum ray distance to test
        
    Returns:
        Boolean array of shape (n_rays,) indicating if ray hit mesh
    """
    origins = np.asarray(origins)
    directions = np.asarray(directions)
    
    assert origins.shape[0] == directions.shape[0], "Origins and directions must have same number of rays"
    assert origins.shape[1] == 3 and directions.shape[1] == 3, "Origins and directions must be 3D"
    
    try:
        # Use selected ray intersector (embree or trimesh)
        intersector = mesh_context.ray_intersector if mesh_context and mesh_context.ray_intersector is not None else mesh_context.mesh.ray
        mrt_config = getattr(mesh_context, "_mrt_config", None)
        # Resolve max distance from config if not provided
        if max_distance is None:
            max_distance = (mrt_config.performance.ray_max_distance if mrt_config else 1000.0)

        if (mrt_config.intersects_any if mrt_config else True) and hasattr(intersector, "intersects_any"):
            # Fast boolean occlusion check; ignores hit locations
            any_hits = intersector.intersects_any(
                ray_origins=origins,
                ray_directions=directions
            )
            # If max_distance is set, we conservatively return any_hits (no distance filter)
            return np.asarray(any_hits, dtype=bool)
        else:
            locations, ray_indices, face_indices = intersector.intersects_location(
                ray_origins=origins,
                ray_directions=directions,
                multiple_hits=False  # Only need first hit for occlusion
            )
            
            # Create boolean mask for hits
            hit_mask = np.zeros(len(origins), dtype=bool)
            if len(ray_indices) > 0:
                # Check distances for hits within max_distance
                hit_distances = np.linalg.norm(locations - origins[ray_indices], axis=1)
                valid_hits = hit_distances <= max_distance
                hit_mask[ray_indices[valid_hits]] = True
                
            return hit_mask
        
    except Exception as e:
        warnings.warn(f"Ray intersection failed: {e}. Returning no hits.")
        return np.zeros(len(origins), dtype=bool)


def batch_ray_intersections(origins: np.ndarray,
                           directions: np.ndarray,
                           mesh_context: MeshContext,
                           batch_size: int = 10000,
                           max_distance: Optional[float] = None) -> np.ndarray:
    """
    Perform ray intersections in batches to manage memory usage.
    
    Args:
        origins: Shape (n_rays, 3) ray starting points
        directions: Shape (n_rays, 3) ray directions
        mesh_context: MeshContext with geometry and acceleration
        batch_size: Number of rays to process at once
        max_distance: Maximum ray distance to test
        
    Returns:
        Boolean array of shape (n_rays,) indicating if ray hit mesh
    """
    n_rays = len(origins)
    
    # For small batches, process all at once
    if n_rays <= batch_size:
        return ray_mesh_intersections(origins, directions, mesh_context, max_distance)
    
    # Process in batches
    hit_results = np.zeros(n_rays, dtype=bool)
    
    for start_idx in range(0, n_rays, batch_size):
        end_idx = min(start_idx + batch_size, n_rays)
        
        batch_hits = ray_mesh_intersections(
            origins[start_idx:end_idx], 
            directions[start_idx:end_idx], 
            mesh_context, 
            max_distance
        )
        
        hit_results[start_idx:end_idx] = batch_hits
    
    return hit_results


def test_mesh_context():
    """Simple test function for mesh context functionality."""
    # Create a simple test mesh (cube)
    test_mesh = trimesh.creation.box(extents=[2, 2, 2])
    mesh_context = MeshContext(mesh=test_mesh)
    
    # Test ray pointing at cube center
    origins = np.array([[0, 0, 5]])  # Above cube
    directions = np.array([[0, 0, -1]])  # Pointing down
    
    hits = ray_mesh_intersections(origins, directions, mesh_context)
    
    logger.debug(f"Test mesh context: BVH={mesh_context.has_bvh}, Hit={hits[0]}")
    return hits[0]  # Should be True


if __name__ == "__main__":
    test_mesh_context()
