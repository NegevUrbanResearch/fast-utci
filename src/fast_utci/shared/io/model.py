"""
High-level model loading for fast-utci.

Provides convenience functions for loading 3D models and weather data together.
"""

from pathlib import Path
from ladybug.epw import EPW
import pandas as pd
import trimesh
import logging

from .glb import load_glb_safe, _is_binary_gltf
from fast_utci.shared.weather import load_weather_data

logger = logging.getLogger(__name__)


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
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        ValueError: If model file format is not supported
    """
    file_path = Path(model_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Model file not found: {file_path}")
    
    if file_path.suffix.lower() not in ['.glb', '.gltf']:
        raise ValueError(
            f"Expected .glb or .gltf file. Found: {file_path.suffix}. "
            f"Convert with: trimesh.exchange.export.export_mesh(...)"
        )
    
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
        logger.info(f"Loaded model with {len(scene.geometry)} geometries")
        
        # Log layer info if verbose
        from .scene import get_layer_name_from_scene_graph
        layer_names = set()
        for geom_name in scene.geometry.keys():
            layer_name = get_layer_name_from_scene_graph(scene, geom_name)
            if layer_name not in ['unknown', 'not_found']:
                layer_names.add(layer_name)
        
        if layer_names:
            logger.info(f"Scene graph layers: {', '.join(sorted(layer_names))}")
    
    # Load weather data
    weather_df, epw = load_weather_data(weather_path)
    
    return scene, weather_df, epw

