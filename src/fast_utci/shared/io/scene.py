"""
Scene graph utilities for fast-utci.

Provides functions for extracting information from trimesh.Scene objects,
including layer names, ground bounds, and combined meshes.
"""

import numpy as np
import trimesh
import logging

logger = logging.getLogger(__name__)


def get_layer_name_from_scene_graph(scene: trimesh.Scene, geometry_name: str) -> str:
    """
    Extract layer name from scene graph by traversing parent nodes.
    
    Returns actual layer names from the GLB file like 'ground', 'roads', 'treesurface', etc.
    
    Args:
        scene: trimesh.Scene object
        geometry_name: Name of the geometry to find layer for
        
    Returns:
        Layer name string, or 'unknown' if not found
    """
    graph = scene.graph
    
    # Find the node containing this geometry
    for node in graph.nodes:
        node_data = graph.transforms.node_data.get(node, {})
        
        if node_data.get('geometry') == geometry_name:
            # Found the node - traverse up to find layer parent
            current = node
            
            for _ in range(20):  # Max depth to prevent infinite loops
                parent = graph.transforms.parents.get(current)
                if not parent:
                    break
                
                # Check if parent is a layer node (meaningful name)
                if (not parent.isdigit() and 
                    not parent.startswith('GLTF') and 
                    not parent.startswith('Layer_') and 
                    parent not in ['world', 'base', 'Scene']):
                    return parent
                
                current = parent
            
            return 'unknown'
    
    return 'not_found'


def get_ground_bounds(scene: trimesh.Scene) -> np.ndarray:
    """
    Get bounds of ground plane for grid generation.
    
    Priority:
    1. Scene graph 'ground'/'base' layer name
    2. Largest flat mesh (simple heuristic)
    3. Full model bounds (fallback with warning)
    
    Args:
        scene: trimesh.Scene object
        
    Returns:
        Bounding box as numpy array [[x_min, y_min, z_min], [x_max, y_max, z_max]]
    """
    # Priority 1: Scene graph layer names
    for geom_name, geom in scene.geometry.items():
        if not isinstance(geom, trimesh.Trimesh):
            continue
            
        layer_name = get_layer_name_from_scene_graph(scene, geom_name)
        if layer_name and ('ground' in layer_name.lower() or 
                          'base' in layer_name.lower()):
            logger.info(f"Using '{layer_name}' layer for bounds")
            return geom.bounds
    
    # Priority 2: Largest flat mesh (simple heuristic)
    logger.info("No ground layer found in scene graph, using largest flat mesh")
    largest_area = 0
    base_mesh = None
    
    for geom in scene.geometry.values():
        if isinstance(geom, trimesh.Trimesh):
            bounds_size = geom.bounds[1] - geom.bounds[0]
            height = bounds_size[2]
            area_xy = bounds_size[0] * bounds_size[1]
            
            # Truly flat (< 1cm thick) and large
            if height < 0.01 and area_xy > largest_area:
                largest_area = area_xy
                base_mesh = geom
    
    if base_mesh is not None:
        logger.info(f"Found flat base mesh with area {largest_area:.1f}m²")
        return base_mesh.bounds
    
    # Priority 3: Full bounds (last resort)
    logger.warning("Could not find ground plane, using full model bounds")
    return scene.bounds


def get_combined_mesh(scene: trimesh.Scene) -> trimesh.Trimesh:
    """
    Get combined mesh from scene for MRT calculations.
    
    All geometries are concatenated into a single mesh for ray tracing.
    
    Args:
        scene: trimesh.Scene object
        
    Returns:
        Combined trimesh.Trimesh object
        
    Raises:
        ValueError: If no valid triangle meshes found in scene
    """
    meshes = [geom for geom in scene.geometry.values() 
              if isinstance(geom, trimesh.Trimesh)]
    
    if not meshes:
        raise ValueError("No valid triangle meshes found in scene")
    
    logger.info(f"Combining {len(meshes)} meshes for MRT calculations")
    combined = trimesh.util.concatenate(meshes)
    logger.info(f"Combined mesh: {len(combined.vertices):,} vertices, {len(combined.faces):,} faces")
    
    return combined

