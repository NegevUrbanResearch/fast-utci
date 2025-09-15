"""
Grid generation for analysis surfaces.

Creates sampling grids from surfaces/meshes for MRT analysis,
matching the LB Generate Point Grid component from Grasshopper.
"""

import numpy as np
import trimesh
from typing import Union, Tuple, List, Optional
from dataclasses import dataclass
import warnings


@dataclass
class AnalysisGrid:
    """Container for analysis grid data."""
    points: np.ndarray      # Shape: (n_points, 3) grid point positions
    normals: np.ndarray     # Shape: (n_points, 3) surface normals at points
    face_areas: np.ndarray  # Shape: (n_points,) face areas
    mesh: trimesh.Trimesh   # Original or generated mesh
    grid_size: float        # Grid spacing used
    

def create_grid_from_surface(surface_vertices: np.ndarray,
                            surface_faces: np.ndarray,
                            grid_size: float,
                            offset_distance: float = 0.0) -> AnalysisGrid:
    """
    Create analysis grid from surface mesh.
    
    Args:
        surface_vertices: Shape (n_vertices, 3) surface vertices
        surface_faces: Shape (n_faces, 3) face indices
        grid_size: Target grid spacing
        offset_distance: Distance to offset points from surface
        
    Returns:
        AnalysisGrid with generated points and metadata
    """
    # Create trimesh object
    surface_mesh = trimesh.Trimesh(vertices=surface_vertices, faces=surface_faces)
    
    # Generate grid points using face centers as starting point
    face_centers = surface_mesh.triangles_center
    face_normals = surface_mesh.face_normals
    face_areas = surface_mesh.area_faces
    
    # If grid_size is larger than typical face size, use face centers directly
    # Otherwise, subdivide faces to achieve target grid density
    median_face_area = np.median(face_areas)
    target_area = grid_size ** 2
    
    if median_face_area <= target_area * 2:
        # Use face centers directly
        grid_points = face_centers
        grid_normals = face_normals
        grid_face_areas = face_areas
    else:
        # Need to subdivide - use a simplified approach
        # Sample points more densely across larger faces
        grid_points = []
        grid_normals = []
        grid_face_areas = []
        
        for i, (center, normal, area) in enumerate(zip(face_centers, face_normals, face_areas)):
            if area <= target_area:
                # Face is small enough, use center
                grid_points.append(center)
                grid_normals.append(normal)
                grid_face_areas.append(area)
            else:
                # Face is large, add multiple sample points
                # Simple subdivision: add points in a local grid pattern
                n_subdivisions = max(1, int(np.sqrt(area / target_area)))
                
                # Get face vertices
                face_verts = surface_mesh.vertices[surface_mesh.faces[i]]
                
                # Create subdivision grid within triangle
                for u in np.linspace(0.1, 0.9, n_subdivisions):
                    for v in np.linspace(0.1, 0.9 - u, max(1, n_subdivisions - int(u * n_subdivisions))):
                        if u + v <= 0.9:  # Stay within triangle
                            # Barycentric coordinates
                            w = 1.0 - u - v
                            point = w * face_verts[0] + u * face_verts[1] + v * face_verts[2]
                            
                            grid_points.append(point)
                            grid_normals.append(normal)
                            grid_face_areas.append(target_area)
        
        grid_points = np.array(grid_points)
        grid_normals = np.array(grid_normals)
        grid_face_areas = np.array(grid_face_areas)
    
    # Apply offset if specified
    if offset_distance != 0.0:
        grid_points = grid_points + grid_normals * offset_distance
    
    return AnalysisGrid(
        points=grid_points,
        normals=grid_normals,
        face_areas=grid_face_areas,
        mesh=surface_mesh,
        grid_size=grid_size
    )


def create_rectangular_grid(bounds_min: np.ndarray,
                           bounds_max: np.ndarray, 
                           grid_size: float,
                           z_height: float = 0.0) -> AnalysisGrid:
    """
    Create rectangular grid for simple analysis areas.
    
    Args:
        bounds_min: Shape (2,) or (3,) minimum x,y coordinates (z optional)
        bounds_max: Shape (2,) or (3,) maximum x,y coordinates (z optional)
        grid_size: Grid spacing
        z_height: Z coordinate for all points if not specified in bounds
        
    Returns:
        AnalysisGrid with rectangular grid points
    """
    bounds_min = np.asarray(bounds_min)
    bounds_max = np.asarray(bounds_max)
    
    # Determine Z coordinate
    if len(bounds_min) == 3:
        z_coord = bounds_min[2]
    else:
        z_coord = z_height
    
    # Create grid coordinates
    x_coords = np.arange(bounds_min[0], bounds_max[0] + grid_size/2, grid_size)
    y_coords = np.arange(bounds_min[1], bounds_max[1] + grid_size/2, grid_size)
    
    # Create meshgrid
    X, Y = np.meshgrid(x_coords, y_coords)
    Z = np.full_like(X, z_coord)
    
    # Flatten to point list
    grid_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    
    # All normals point up
    grid_normals = np.tile([0, 0, 1], (len(grid_points), 1))
    
    # All areas are grid_size^2
    grid_face_areas = np.full(len(grid_points), grid_size ** 2)
    
    # Create a simple mesh for the rectangular grid (not used for intersections)
    dummy_mesh = trimesh.Trimesh(vertices=grid_points[:4], faces=[[0, 1, 2]])
    
    return AnalysisGrid(
        points=grid_points,
        normals=grid_normals,
        face_areas=grid_face_areas,
        mesh=dummy_mesh,
        grid_size=grid_size
    )


def load_surface_and_create_grid(surface_file: str,
                                grid_size: float,
                                offset_distance: float = 0.0) -> AnalysisGrid:
    """
    Load surface from file and create analysis grid.
    
    Args:
        surface_file: Path to surface mesh file (OBJ, PLY, STL, etc.)
        grid_size: Target grid spacing
        offset_distance: Distance to offset points from surface
        
    Returns:
        AnalysisGrid generated from loaded surface
    """
    try:
        loaded = trimesh.load(surface_file)
        
        if isinstance(loaded, trimesh.Trimesh):
            surface_mesh = loaded
        elif isinstance(loaded, trimesh.Scene):
            # Use first mesh in scene
            meshes = [geom for geom in loaded.geometry.values() 
                     if isinstance(geom, trimesh.Trimesh)]
            if not meshes:
                raise ValueError("No valid meshes found in scene")
            surface_mesh = meshes[0]
        else:
            raise ValueError(f"Unsupported file type: {type(loaded)}")
        
        return create_grid_from_surface(
            surface_mesh.vertices,
            surface_mesh.faces,
            grid_size,
            offset_distance
        )
        
    except Exception as e:
        raise ValueError(f"Failed to load surface from {surface_file}: {e}")


def filter_grid_by_bounds(grid: AnalysisGrid,
                         bounds_min: np.ndarray,
                         bounds_max: np.ndarray) -> AnalysisGrid:
    """
    Filter grid points to only include those within specified bounds.
    
    Args:
        grid: Input AnalysisGrid
        bounds_min: Shape (3,) minimum coordinates
        bounds_max: Shape (3,) maximum coordinates
        
    Returns:
        Filtered AnalysisGrid
    """
    bounds_min = np.asarray(bounds_min)
    bounds_max = np.asarray(bounds_max)
    
    # Create mask for points within bounds
    within_bounds = (
        (grid.points[:, 0] >= bounds_min[0]) & (grid.points[:, 0] <= bounds_max[0]) &
        (grid.points[:, 1] >= bounds_min[1]) & (grid.points[:, 1] <= bounds_max[1]) &
        (grid.points[:, 2] >= bounds_min[2]) & (grid.points[:, 2] <= bounds_max[2])
    )
    
    # Filter arrays
    filtered_points = grid.points[within_bounds]
    filtered_normals = grid.normals[within_bounds]
    filtered_areas = grid.face_areas[within_bounds]
    
    return AnalysisGrid(
        points=filtered_points,
        normals=filtered_normals,
        face_areas=filtered_areas,
        mesh=grid.mesh,  # Keep original mesh
        grid_size=grid.grid_size
    )
