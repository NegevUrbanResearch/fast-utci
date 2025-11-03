"""
Model Reader for fast-utci

This module provides 3D model loading capabilities with scene graph layer extraction.
"""

from pathlib import Path
from ladybug.epw import EPW
import pandas as pd
import trimesh
import numpy as np
import json
import struct


def _is_binary_gltf(file_path: Path) -> bool:
    """Check if a file is a binary GLTF (GLB) file."""
    with open(file_path, 'rb') as f:
        magic = f.read(4)
        return magic == b'glTF'


def load_glb_safe(file_path: Path) -> trimesh.Scene:
    """
    Safely load a GLB file, extracting only mesh primitives and skipping
    problematic polyline/curve data that may cause memory errors.
    
    Args:
        file_path: Path to the GLB file
        
    Returns:
        trimesh.Scene with only mesh geometries
    """
    print(f"[INFO] Using safe GLB loader to extract meshes only...")
    
    with open(file_path, 'rb') as f:
        # Read GLB header
        magic = struct.unpack('4s', f.read(4))[0]
        if magic != b'glTF':
            raise ValueError(f"Not a valid GLB file: {file_path}")
        
        version = struct.unpack('<I', f.read(4))[0]
        length = struct.unpack('<I', f.read(4))[0]
        
        # Read JSON chunk
        chunk_length = struct.unpack('<I', f.read(4))[0]
        chunk_type = struct.unpack('4s', f.read(4))[0]
        
        if chunk_type != b'JSON':
            raise ValueError(f"Expected JSON chunk, got {chunk_type}")
        
        json_data = json.loads(f.read(chunk_length).decode('utf-8'))
        
        # Read binary chunk
        chunk_length = struct.unpack('<I', f.read(4))[0]
        chunk_type = struct.unpack('4s', f.read(4))[0]
        
        if chunk_type != b'BIN\x00':
            raise ValueError(f"Expected BIN chunk, got {chunk_type}")
        
        binary_data = f.read(chunk_length)
    
    # Extract only mesh primitives from the GLTF structure
    meshes = []
    buffer_views = json_data.get('bufferViews', [])
    accessors = json_data.get('accessors', [])
    gltf_meshes = json_data.get('meshes', [])
    
    print(f"[INFO] Found {len(gltf_meshes)} mesh definitions in GLB")
    
    def get_accessor_data(accessor_index, expected_type=None):
        """Extract data from an accessor."""
        accessor = accessors[accessor_index]
        buffer_view_index = accessor.get('bufferView')
        if buffer_view_index is None:
            return None
            
        buffer_view = buffer_views[buffer_view_index]
        offset = buffer_view.get('byteOffset', 0) + accessor.get('byteOffset', 0)
        
        # Determine component type and size
        component_types = {
            5120: ('b', 1),   # BYTE
            5121: ('B', 1),   # UNSIGNED_BYTE
            5122: ('h', 2),   # SHORT
            5123: ('H', 2),   # UNSIGNED_SHORT
            5125: ('I', 4),   # UNSIGNED_INT
            5126: ('f', 4),   # FLOAT
        }
        
        comp_type, comp_size = component_types[accessor['componentType']]
        count = accessor['count']
        
        # Determine number of components per element
        type_sizes = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}
        num_components = type_sizes[accessor['type']]
        
        # Extract data
        data = []
        for i in range(count):
            element = []
            for j in range(num_components):
                byte_offset = offset + (i * num_components + j) * comp_size
                value = struct.unpack_from(comp_type, binary_data, byte_offset)[0]
                element.append(value)
            if num_components == 1:
                data.append(element[0])
            else:
                data.append(element)
        
        return np.array(data)
    
    # Extract mesh primitives
    mesh_index = 0
    skipped_lines = 0
    skipped_other = 0
    
    for gltf_mesh in gltf_meshes:
        for primitive in gltf_mesh.get('primitives', []):
            # Skip non-triangle primitives (LINES, POINTS, etc.)
            mode = primitive.get('mode', 4)  # Default is TRIANGLES (4)
            if mode == 1:  # LINES
                skipped_lines += 1
                continue
            elif mode != 4:  # Not TRIANGLES
                skipped_other += 1
                continue
            
            try:
                # Get vertex positions
                attributes = primitive.get('attributes', {})
                if 'POSITION' not in attributes:
                    continue
                
                vertices = get_accessor_data(attributes['POSITION'])
                if vertices is None or len(vertices) == 0:
                    continue
                
                # Get indices
                indices_accessor = primitive.get('indices')
                if indices_accessor is not None:
                    indices = get_accessor_data(indices_accessor)
                    if indices is None:
                        continue
                    faces = indices.reshape(-1, 3)
                else:
                    # No indices, assume sequential triangles
                    num_faces = len(vertices) // 3
                    faces = np.arange(num_faces * 3).reshape(-1, 3)
                
                # Create trimesh
                try:
                    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
                    # Check if mesh is valid (has vertices and faces)
                    if len(mesh.vertices) > 0 and len(mesh.faces) > 0:
                        meshes.append((f"mesh_{mesh_index}", mesh))
                        mesh_index += 1
                except Exception as e:
                    if mesh_index < 10:  # Only print first 10 errors
                        print(f"[WARN] Failed to create mesh {mesh_index}: {e}")
                    continue
                    
            except Exception as e:
                print(f"[WARN] Error processing primitive: {e}")
                continue
    
    print(f"[INFO] Successfully extracted {len(meshes)} valid triangle meshes")
    if skipped_lines > 0:
        print(f"[INFO] Skipped {skipped_lines} LINE primitives (polylines/curves)")
    if skipped_other > 0:
        print(f"[INFO] Skipped {skipped_other} other non-triangle primitives")
    
    if not meshes:
        raise ValueError("No valid triangle meshes found in GLB file (only found non-mesh geometry like curves/lines)")
    
    # Create a scene from the meshes
    geometry = {name: mesh for name, mesh in meshes}
    scene = trimesh.Scene(geometry=geometry)
    
    return scene


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
            print(f"[GRID] Using '{layer_name}' layer for bounds")
            return geom.bounds
    
    # Priority 2: Largest flat mesh (simple heuristic)
    print("[GRID] No ground layer found in scene graph, using largest flat mesh")
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
        print(f"[GRID] Found flat base mesh with area {largest_area:.1f}m²")
        return base_mesh.bounds
    
    # Priority 3: Full bounds (last resort)
    print("[WARN] Could not find ground plane, using full model bounds")
    return scene.bounds


def get_combined_mesh(scene: trimesh.Scene) -> trimesh.Trimesh:
    """
    Get combined mesh from scene for MRT calculations.
    
    All geometries are concatenated into a single mesh for ray tracing.
    
    Args:
        scene: trimesh.Scene object
        
    Returns:
        Combined trimesh.Trimesh object
    """
    meshes = [geom for geom in scene.geometry.values() 
              if isinstance(geom, trimesh.Trimesh)]
    
    if not meshes:
        raise ValueError("No valid triangle meshes found in scene")
    
    print(f"[INFO] Combining {len(meshes)} meshes for MRT calculations")
    combined = trimesh.util.concatenate(meshes)
    print(f"[INFO] Combined mesh: {len(combined.vertices):,} vertices, {len(combined.faces):,} faces")
    
    return combined


def read_project_data(model_path: str | Path, 
                              weather_path: str | Path, 
                      verbose: bool = False) -> tuple[trimesh.Scene, pd.DataFrame, EPW]:
    """
    Load 3D model and weather data.
    
    Args:
        model_path: Path to GLB/GLTF model file
        weather_path: Path to EPW weather file
        verbose: Print detailed loading information
        
    Returns:
        Tuple of (scene, weather_df, epw):
        - scene: trimesh.Scene with all geometries and scene graph
        - weather_df: Weather DataFrame
        - epw: EPW object
    """
    from fast_utci.shared.weather import load_weather_data
    
    file_path = Path(model_path)
    assert file_path.exists(), f"Model file not found: {file_path}"
    assert file_path.suffix.lower() in ['.glb', '.gltf'], f"Unsupported format: {file_path.suffix}"
    
    # Load model using safe GLB loader for binary files
    if file_path.suffix.lower() == '.glb' or _is_binary_gltf(file_path):
        scene = load_glb_safe(file_path)
    else:
        # Regular GLTF - use trimesh loader
        scene = trimesh.load(
            str(file_path),
            process=False,
            ignore_broken=True,
            skip_materials=True,
            skip_texture=True
        )
    
    if verbose:
        print(f"[INFO] Loaded model with {len(scene.geometry)} geometries")
        
        # Print layer info if verbose
        layer_names = set()
        for geom_name in scene.geometry.keys():
            layer_name = get_layer_name_from_scene_graph(scene, geom_name)
            if layer_name not in ['unknown', 'not_found']:
                layer_names.add(layer_name)
        
        if layer_names:
            print(f"[INFO] Scene graph layers: {', '.join(sorted(layer_names))}")
    
    # Load weather data
    weather_df, epw = load_weather_data(weather_path)
    
    return scene, weather_df, epw


def read_weather_data(file_path: str | Path) -> pd.DataFrame:
    """
    Read EPW weather file and extract UTCI inputs as DataFrame.
    
    NOTE: This function delegates to fast_utci.shared.weather.load_weather_data()
    for consolidation. The API remains backward compatible.
    
    Args:
        file_path: Path to EPW weather file
        
    Returns:
        Weather DataFrame
    """
    from fast_utci.shared.weather import load_weather_data
    
    weather_df, _ = load_weather_data(file_path)
    return weather_df


# Example usage
if __name__ == "__main__":
    # Test the model reader
    model_path = "data/3d_models/100_test.glb"
    
    try:
        scene, weather_df, epw = read_project_data(
            model_path,
            "data/weather/ISR_Beer.Sheva.401900_MSI.epw",
            verbose=True
        )
        
        # Get combined mesh for MRT calculations
        combined_mesh = get_combined_mesh(scene)
        
        # Get ground bounds for grid generation
        ground_bounds = get_ground_bounds(scene)
        print(f"[INFO] Ground bounds: {ground_bounds}")
        
    except Exception as e:
        print(f"Error: {e}")
