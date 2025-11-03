"""
GLB file loading utilities for fast-utci.

Provides safe GLB file loading that extracts only triangle meshes,
skipping problematic polyline/curve data.
"""

from pathlib import Path
import trimesh
import numpy as np
import json
import struct
import logging

logger = logging.getLogger(__name__)


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
        
    Raises:
        ValueError: If file is not a valid GLB file or contains no valid meshes
    """
    logger.info("Using safe GLB loader to extract meshes only...")
    
    with open(file_path, 'rb') as f:
        # Read GLB header
        magic = struct.unpack('4s', f.read(4))[0]
        if magic != b'glTF':
            raise ValueError(
                f"Not a valid GLB file: {file_path}. "
                f"Expected GLB format. Convert with: trimesh.exchange.export.export_mesh(...)"
            )
        
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
    
    logger.info(f"Found {len(gltf_meshes)} mesh definitions in GLB")
    
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
                    if mesh_index < 10:  # Only log first 10 errors
                        logger.warning(f"Failed to create mesh {mesh_index}: {e}")
                    continue
                    
            except Exception as e:
                logger.warning(f"Error processing primitive: {e}")
                continue
    
    logger.info(f"Successfully extracted {len(meshes)} valid triangle meshes")
    if skipped_lines > 0:
        logger.info(f"Skipped {skipped_lines} LINE primitives (polylines/curves)")
    if skipped_other > 0:
        logger.info(f"Skipped {skipped_other} other non-triangle primitives")
    
    if not meshes:
        raise ValueError(
            "No valid triangle meshes found in GLB file (only found non-mesh geometry like curves/lines). "
            "Ensure your GLB file contains triangle mesh primitives."
        )
    
    # Create a scene from the meshes
    geometry = {name: mesh for name, mesh in meshes}
    scene = trimesh.Scene(geometry=geometry)
    
    return scene

