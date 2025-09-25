"""
Enhanced Model Reader for fast-utci

This module provides enhanced 3D model reading capabilities that preserve
material information and layer structure for better visualization.
"""

from pathlib import Path
from typing import Tuple, Union, List, Dict, Any, Optional
import pandas as pd
import trimesh
import numpy as np
from collections import defaultdict

# Material type mapping for common building elements
MATERIAL_TYPE_MAPPING = {
    'building': {'color': '#2b2b2b', 'opacity': 0.9, 'name': 'Buildings'},
                'base': {'color': '#BDC3C7', 'opacity': 0.2, 'name': 'Base'},  # Model base/ground plane - moderate transparency
    'road': {'color': '#34495E', 'opacity': 0.7, 'name': 'Roads'},  # Actual roads on top of base
    'street': {'color': '#2C3E50', 'opacity': 0.7, 'name': 'Streets'},  # Streets on top of base
    'sidewalk': {'color': '#BDC3C7', 'opacity': 0.8, 'name': 'Sidewalks'},
    'vegetation': {'color': '#27AE60', 'opacity': 0.9, 'name': 'Trees'},  # Higher opacity for visibility
    'water': {'color': '#3498DB', 'opacity': 0.6, 'name': 'Water'},
    'default': {'color': '#27AE60', 'opacity': 0.9, 'name': 'Trees'}  # Default to trees
}


class ModelLayer:
    """Represents a single layer/material in the 3D model."""
    
    def __init__(self, name: str, mesh: trimesh.Trimesh, material_type: str = 'default'):
        self.name = name
        self.mesh = mesh
        self.material_type = material_type
        self.material_info = MATERIAL_TYPE_MAPPING.get(material_type, MATERIAL_TYPE_MAPPING['default'])
    
    def get_color(self) -> str:
        """Get the color for this layer."""
        return self.material_info['color']
    
    def get_opacity(self) -> float:
        """Get the opacity for this layer."""
        return self.material_info['opacity']
    
    def get_display_name(self) -> str:
        """Get the display name for this layer."""
        return self.material_info['name']


class EnhancedModel:
    """Enhanced model class that preserves layer information."""
    
    def __init__(self, layers: List[ModelLayer]):
        self.layers = layers
        self._combined_mesh = None
    
    def get_combined_mesh(self) -> trimesh.Trimesh:
        """Get a combined mesh for MRT calculations."""
        if self._combined_mesh is None:
            meshes = [layer.mesh for layer in self.layers]
            self._combined_mesh = trimesh.util.concatenate(meshes)
        return self._combined_mesh
    
    def get_layer_by_type(self, material_type: str) -> List[ModelLayer]:
        """Get all layers of a specific material type."""
        return [layer for layer in self.layers if layer.material_type == material_type]
    
    def get_layer_by_name(self, name: str) -> Optional[ModelLayer]:
        """Get a layer by its name."""
        for layer in self.layers:
            if layer.name == name:
                return layer
        return None
    
    def get_bounds(self) -> np.ndarray:
        """Get the bounds of the combined model."""
        return self.get_combined_mesh().bounds
    
    def get_bounds_for_layer_type(self, material_type: str) -> Optional[np.ndarray]:
        """Get the bounds for a specific layer type."""
        layers = self.get_layer_by_type(material_type)
        if not layers:
            return None
        
        # Combine all meshes of this type
        meshes = [layer.mesh for layer in layers]
        if not meshes:
            return None
        
        combined_mesh = trimesh.util.concatenate(meshes)
        return combined_mesh.bounds
    
    def get_vertices_count(self) -> int:
        """Get total number of vertices across all layers."""
        return sum(len(layer.mesh.vertices) for layer in self.layers)
    
    def get_faces_count(self) -> int:
        """Get total number of faces across all layers."""
        return sum(len(layer.mesh.faces) for layer in self.layers)
    
    def get_layer_info(self) -> List[Dict[str, Any]]:
        """Get information about all layers."""
        info = []
        for layer in self.layers:
            info.append({
                'name': layer.name,
                'material_type': layer.material_type,
                'display_name': layer.get_display_name(),
                'color': layer.get_color(),
                'opacity': layer.get_opacity(),
                'vertices': len(layer.mesh.vertices),
                'faces': len(layer.mesh.faces)
            })
        return info


def extract_road_lines_from_base_mesh(base_mesh: trimesh.Trimesh, mesh_name: str, verbose: bool = False) -> List[trimesh.Trimesh]:
    """
    Extract road lines from a base mesh by analyzing material groups and face clusters.
    
    Args:
        base_mesh: The base mesh to analyze
        mesh_name: Name of the mesh for debugging
        
    Returns:
        List of road line meshes extracted from the base mesh
    """
    road_lines = []
    
    try:
        # Since base meshes are truly flat, roads must be embedded as different materials
        # Check if mesh has material information
        if verbose and hasattr(base_mesh, 'visual') and hasattr(base_mesh.visual, 'material'):
            print(f"🔍 {mesh_name} has material info: {base_mesh.visual.material}")
        
        # Check for face groups (different materials)
        if hasattr(base_mesh, 'visual') and hasattr(base_mesh.visual, 'face_materials'):
            face_materials = base_mesh.visual.face_materials
            unique_materials = np.unique(face_materials)
            if verbose:
                print(f"🔍 {mesh_name} has {len(unique_materials)} different materials: {unique_materials}")
            
            # Extract faces with different materials as potential road lines
            for material_id in unique_materials:
                if material_id != 0:  # Assume material 0 is the base, others might be roads
                    road_faces = base_mesh.faces[face_materials == material_id]
                    if len(road_faces) > 0:
                        road_mesh = trimesh.Trimesh(vertices=base_mesh.vertices, faces=road_faces)
                        road_lines.append(road_mesh)
                        if verbose:
                            print(f"🛣️ Extracted {len(road_faces)} faces with material {material_id}")
        
        # Alternative: Analyze face connectivity to find linear patterns
        if not road_lines:
            faces = base_mesh.faces
            vertices = base_mesh.vertices
            
            # Find faces that form linear patterns (road-like shapes)
            # Group faces by connectivity and analyze their shape
            face_centers = np.mean(vertices[faces], axis=1)
            
            # Find faces that are arranged in lines (potential roads)
            from sklearn.cluster import DBSCAN
            
            # Cluster face centers to find linear arrangements
            clustering = DBSCAN(eps=10.0, min_samples=3).fit(face_centers)
            labels = clustering.labels_
            
            # Analyze each cluster for linear characteristics
            for cluster_id in np.unique(labels):
                if cluster_id == -1:  # Skip noise
                    continue
                    
                cluster_faces = faces[labels == cluster_id]
                if len(cluster_faces) > 5:  # Need enough faces for a road
                    # Create mesh for this cluster
                    cluster_mesh = trimesh.Trimesh(vertices=vertices, faces=cluster_faces)
                    
                    # Check if it's linear (high aspect ratio)
                    bounds_size = cluster_mesh.bounds[1] - cluster_mesh.bounds[0]
                    aspect_ratio = max(bounds_size[0], bounds_size[1]) / min(bounds_size[0], bounds_size[1])
                    
                    # If it's elongated and reasonably sized, consider it a road
                    if aspect_ratio > 5.0 and 10 < cluster_mesh.area < 1000:
                        road_lines.append(cluster_mesh)
                        # Note: This debug output is controlled by the calling function's verbose parameter
    
    except Exception as e:
        print(f"⚠️ Error extracting road lines from {mesh_name}: {e}")
    
    return road_lines


def detect_material_type(mesh_name: str, mesh_bounds: np.ndarray, mesh_volume: float, mesh_area: float = None, debug: bool = False, base_z_level: float = None, geom: trimesh.Trimesh = None) -> str:
    """
    Detect material type based on mesh properties.
    
    Args:
        mesh_name: Name of the mesh
        mesh_bounds: Bounding box of the mesh
        mesh_volume: Volume of the mesh
        mesh_area: Surface area of the mesh (optional)
        debug: Whether to print debug information
        base_z_level: Z coordinate of the base level (optional)
        geom: The trimesh object for detailed analysis (optional)
        
    Returns:
        Material type string
    """
    name_lower = mesh_name.lower()
    
    # Check name-based detection
    if any(keyword in name_lower for keyword in ['building', 'wall', 'roof', 'facade']):
        return 'building'
    elif any(keyword in name_lower for keyword in ['road', 'street', 'highway', 'pavement']):
        return 'road'
    elif any(keyword in name_lower for keyword in ['sidewalk', 'footpath', 'walkway']):
        return 'sidewalk'
    elif any(keyword in name_lower for keyword in ['tree', 'vegetation', 'plant', 'bush']):
        return 'vegetation'  # Combine trees and vegetation
    elif any(keyword in name_lower for keyword in ['water', 'river', 'lake', 'pond']):
        return 'water'
    
    # Check geometric properties
    bounds_size = mesh_bounds[1] - mesh_bounds[0]
    height = bounds_size[2]
    area_xy = bounds_size[0] * bounds_size[1]
    
    # Get Z coordinates for elevation-based classification
    min_z = mesh_bounds[0][2]
    max_z = mesh_bounds[1][2]
    
    # Heuristics for material detection based on actual mesh analysis
    # Base: truly flat, very large meshes (only the ground plane)
    if height <= 0.001 and area_xy > 10000:  # Truly flat and very large = base only
        if debug:
            print(f"BASE: {mesh_name} - height: {height:.3f}m, area: {area_xy:.1f}m², volume: {mesh_volume:.3f}m³, z: {min_z:.3f}-{max_z:.3f}m")
        return 'base'
    
    # Buildings: tall structures with significant height and reasonable area
    elif height > 2.0 and area_xy > 50:  # Tall and reasonably large = building
        return 'building'
    
    # Calculate shape characteristics for linear vs blob detection
    if area_xy > 0:
        aspect_ratio = len(geom.vertices) / area_xy if hasattr(geom, 'vertices') else 0
    else:
        aspect_ratio = 0
    
    # Roads: ANY linear elements (high aspect ratio) should be roads, regardless of height
    if aspect_ratio > 3.0:  # Linear elements = roads (prioritize shape over height)
        if debug:
            print(f"ROAD (linear): {mesh_name} - aspect_ratio: {aspect_ratio:.1f}, height: {height:.3f}m, z: {min_z:.3f}-{max_z:.3f}m")
        return 'road'
    
    # Trees: only truly elevated blob-like elements (low aspect ratio + elevated)
    if min_z > 3.0 and aspect_ratio < 2.0:  # Only truly elevated blob elements = trees
        if debug:
            print(f"TREE (blob): {mesh_name} - aspect_ratio: {aspect_ratio:.1f}, height: {height:.3f}m, z: {min_z:.3f}-{max_z:.3f}m")
        return 'vegetation'
    
    # Elements just slightly above base should be roads/layout, not trees
    elif min_z > 0.1 and min_z <= 3.0:  # Slightly above base = roads/layout elements
        if debug:
            print(f"ROAD (elevated): {mesh_name} - aspect_ratio: {aspect_ratio:.1f}, height: {height:.3f}m, z: {min_z:.3f}-{max_z:.3f}m")
        return 'road'
    
    # Elements at or below base level should never be trees
    elif min_z <= 0.1:  # At or below base level = roads/layout elements
        if debug:
            print(f"ROAD (base-level): {mesh_name} - aspect_ratio: {aspect_ratio:.1f}, height: {height:.3f}m, z: {min_z:.3f}-{max_z:.3f}m")
        return 'road'
    
    # Small volume elements = vegetation (only if truly elevated AND blob-like)
    elif mesh_volume < 3.0 and min_z > 3.0 and aspect_ratio < 2.0:  # Small volume AND truly elevated AND blob-like = vegetation
        if debug:
            print(f"TREE (small): {mesh_name} - aspect_ratio: {aspect_ratio:.1f}, height: {height:.3f}m, z: {min_z:.3f}-{max_z:.3f}m")
        return 'vegetation'
    # Default: if it has significant height and area, it's likely a building
    elif height > 1.0 and area_xy > 20:
        return 'building'
    
    return 'vegetation'  # Default fallback to vegetation


def read_enhanced_model(file_path: Union[str, Path], verbose: bool = False) -> EnhancedModel:
    """
    Read a 3D model file and return an enhanced model with layer information.
    
    Args:
        file_path: Path to the model file (.glb or .gltf)
        
    Returns:
        EnhancedModel with preserved layer information
    """
    file_path = Path(file_path)
    assert file_path.exists(), f"Model file not found: {file_path}"
    assert file_path.suffix.lower() in ['.glb', '.gltf'], f"Unsupported format: {file_path.suffix}"
    
    # Check if .gltf file is actually binary GLB format
    file_type = None
    if file_path.suffix.lower() == '.gltf':
        with open(file_path, 'rb') as f:
            magic = f.read(4)
            if magic == b'glTF':
                file_type = 'glb'
                print(f"🔧 Detected binary GLB file with .gltf extension: {file_path.name}")
    
    loaded = trimesh.load(str(file_path), file_type=file_type)
    
    layers = []
    
    if isinstance(loaded, trimesh.Trimesh):
        # Single mesh - create one layer
        material_type = detect_material_type(
            file_path.stem, 
            loaded.bounds, 
            loaded.volume if hasattr(loaded, 'volume') else 0,
            loaded.area if hasattr(loaded, 'area') else None,
            geom=loaded
        )
        layer = ModelLayer(file_path.stem, loaded, material_type)
        layers.append(layer)
        
    elif isinstance(loaded, trimesh.Scene):
        # Scene with multiple geometries
        if verbose:
            print(f"🔍 Analyzing {len(loaded.geometry)} geometries:")
        
        # First pass: analyze all meshes to understand the structure
        mesh_info = []
        for geom_name, geom in loaded.geometry.items():
            if isinstance(geom, trimesh.Trimesh):
                bounds = geom.bounds
                height = bounds[1][2] - bounds[0][2]
                area_xy = (bounds[1][0] - bounds[0][0]) * (bounds[1][1] - bounds[0][1])
                volume = geom.volume if hasattr(geom, 'volume') else 0
                
                mesh_info.append({
                    'name': geom_name,
                    'height': height,
                    'area_xy': area_xy,
                    'volume': volume,
                    'min_z': bounds[0][2],
                    'max_z': bounds[1][2],
                    'vertices': len(geom.vertices),
                    'faces': len(geom.faces)
                })
        
        # Sort by area to identify the largest meshes (likely base)
        mesh_info.sort(key=lambda x: x['area_xy'], reverse=True)
        
        if verbose:
            print("📊 Largest meshes (potential base elements):")
            for i, info in enumerate(mesh_info[:10]):  # Top 10 largest
                print(f"  {i+1}. {info['name']}: area={info['area_xy']:.1f}m², height={info['height']:.3f}m, z={info['min_z']:.3f}-{info['max_z']:.3f}m")
            
            print("📊 Smallest meshes (potential trees/roads):")
            for i, info in enumerate(mesh_info[-10:]):  # Bottom 10 smallest
                print(f"  {i+1}. {info['name']}: area={info['area_xy']:.1f}m², height={info['height']:.3f}m, z={info['min_z']:.3f}-{info['max_z']:.3f}m")
            
            # Analyze mesh shapes more thoroughly
            print("📏 Detailed mesh shape analysis:")
            
            # Group meshes by size ranges to understand the structure
            tiny_meshes = [info for info in mesh_info if info['area_xy'] < 1.0]
            small_meshes = [info for info in mesh_info if 1.0 <= info['area_xy'] < 10.0]
            medium_meshes = [info for info in mesh_info if 10.0 <= info['area_xy'] < 100.0]
            large_meshes = [info for info in mesh_info if info['area_xy'] >= 100.0]
            
            print(f"📊 Mesh size distribution:")
            print(f"  - Tiny (<1m²): {len(tiny_meshes)} meshes")
            print(f"  - Small (1-10m²): {len(small_meshes)} meshes") 
            print(f"  - Medium (10-100m²): {len(medium_meshes)} meshes")
            print(f"  - Large (≥100m²): {len(large_meshes)} meshes")
            
            # Analyze vertex density for different size ranges
            print(f"🔍 Vertex density analysis:")
            for size_range, meshes in [("Tiny", tiny_meshes[:5]), ("Small", small_meshes[:5]), ("Medium", medium_meshes[:5])]:
                if meshes:
                    print(f"  {size_range} meshes:")
                    for i, info in enumerate(meshes):
                        aspect_ratio = info['vertices'] / max(info['area_xy'], 1.0)
                        print(f"    {i+1}. {info['name']}: area={info['area_xy']:.1f}m², vertices={info['vertices']}, ratio={aspect_ratio:.1f}, z={info['min_z']:.3f}-{info['max_z']:.3f}m")
            
            # Look for potential road lines (linear elements)
            potential_roads = []
            for info in mesh_info:
                if info['area_xy'] > 0:
                    aspect_ratio = info['vertices'] / max(info['area_xy'], 1.0)
                    # Look for elements that might be road lines
                    if (info['area_xy'] < 50 and aspect_ratio > 3 and 
                        info['height'] > 0.01 and info['height'] < 1.0):  # Some height but not too tall
                        potential_roads.append(info)
            
            print(f"🛣️ Potential road lines: {len(potential_roads)}")
            for i, info in enumerate(potential_roads[:10]):  # Top 10
                aspect_ratio = info['vertices'] / max(info['area_xy'], 1.0)
                print(f"  {i+1}. {info['name']}: area={info['area_xy']:.1f}m², vertices={info['vertices']}, ratio={aspect_ratio:.1f}, height={info['height']:.3f}m, z={info['min_z']:.3f}-{info['max_z']:.3f}m")
        
        # Second pass: classify meshes and extract embedded road lines
        for geom_name, geom in loaded.geometry.items():
            if isinstance(geom, trimesh.Trimesh):
                # Detect material type based on geometry properties
                material_type = detect_material_type(
                    geom_name,
                    geom.bounds,
                    geom.volume if hasattr(geom, 'volume') else 0,
                    geom.area if hasattr(geom, 'area') else None,
                    geom=geom
                )
                
                # If this is a base mesh, try to extract embedded road lines
                if material_type == 'base':
                    road_lines = extract_road_lines_from_base_mesh(geom, geom_name, verbose=verbose)
                    if road_lines:
                        if verbose:
                            print(f"🛣️ Extracted {len(road_lines)} road line segments from {geom_name}")
                        # Add road line segments as separate layers
                        for i, road_line in enumerate(road_lines):
                            # Validate the road mesh before adding
                            if hasattr(road_line, 'vertices') and len(road_line.vertices) > 0:
                                road_layer = ModelLayer(f"{geom_name}_road_{i}", road_line, 'road')
                                layers.append(road_layer)
                            else:
                                if verbose:
                                    print(f"⚠️ Invalid road mesh created from {geom_name}_road_{i}")
                
                layer = ModelLayer(geom_name, geom, material_type)
                layers.append(layer)
        
        assert layers, "No valid meshes found in scene"
    
    else:
        raise ValueError(f"Unsupported object type: {type(loaded)}")
    
    print(f"📊 Loaded model with {len(layers)} layers")
    
    # Group layers by type for cleaner output
    layer_counts = {}
    total_vertices = 0
    total_faces = 0
    
    for layer in layers:
        layer_type = layer.get_display_name()
        if layer_type not in layer_counts:
            layer_counts[layer_type] = {'count': 0, 'vertices': 0, 'faces': 0}
        layer_counts[layer_type]['count'] += 1
        layer_counts[layer_type]['vertices'] += len(layer.mesh.vertices)
        layer_counts[layer_type]['faces'] += len(layer.mesh.faces)
        total_vertices += len(layer.mesh.vertices)
        total_faces += len(layer.mesh.faces)
    
    # Print summary instead of individual layers
    for layer_type, info in layer_counts.items():
        print(f"  - {layer_type}: {info['count']} objects, {info['vertices']:,} vertices, {info['faces']:,} faces")
    
    print(f"📊 Total: {total_vertices:,} vertices, {total_faces:,} faces")
    
    return EnhancedModel(layers)


def read_project_data_enhanced(model_path: Union[str, Path], 
                              weather_path: Union[str, Path], 
                              verbose: bool = False) -> Tuple[EnhancedModel, pd.DataFrame, Any]:
    """
    Read model and weather data with enhanced model support.
    
    Args:
        model_path: Path to the model file
        weather_path: Path to the EPW weather file
        
    Returns:
        Tuple of (enhanced_model, weather_df, epw_object)
    """
    enhanced_model = read_enhanced_model(model_path, verbose=verbose)
    weather_df = read_weather_data(weather_path)
    
    # Also return the original EPW object
    from ladybug.epw import EPW
    epw = EPW(str(weather_path))
    
    return enhanced_model, weather_df, epw


def read_weather_data(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Read EPW weather file and extract UTCI inputs as DataFrame.
    """
    file_path = Path(file_path)
    assert file_path.exists(), f"Weather file not found: {file_path}"

    from ladybug.epw import EPW
    epw = EPW(str(file_path))

    data = {
        'datetime': epw.dry_bulb_temperature.datetimes,
        'air_temp': epw.dry_bulb_temperature.values,
        'wind_speed': epw.wind_speed.values,
        'relative_humidity': epw.relative_humidity.values,
        'global_horizontal_radiation': epw.global_horizontal_radiation.values,
        'direct_normal_radiation': epw.direct_normal_radiation.values,
        'diffuse_horizontal_radiation': epw.diffuse_horizontal_radiation.values,
        'horizontal_infrared_radiation_intensity': epw.horizontal_infrared_radiation_intensity.values,
        'surface_temp': epw.dry_bulb_temperature.values
    }

    return pd.DataFrame(data)


def read_project_data(model_path: Union[str, Path], 
                      weather_path: Union[str, Path], 
                      verbose: bool = False) -> Tuple[trimesh.Trimesh, pd.DataFrame, Any]:
    """
    Compatibility wrapper matching reader.read_project_data signature.
    Returns combined trimesh mesh, weather DataFrame, and EPW.
    """
    enhanced_model, weather_df, epw = read_project_data_enhanced(model_path, weather_path, verbose=verbose)
    combined_mesh = enhanced_model.get_combined_mesh()
    return combined_mesh, weather_df, epw


# Example usage
if __name__ == "__main__":
    # Test the enhanced model reader
    model_path = "data/100_35164.gltf"
    
    try:
        enhanced_model = read_enhanced_model(model_path)
        
        # Get combined mesh for MRT calculations
        combined_mesh = enhanced_model.get_combined_mesh()
        print(f"📊 Combined mesh: {len(combined_mesh.vertices):,} vertices, {len(combined_mesh.faces):,} faces")
        
    except Exception as e:
        print(f"Error: {e}")
