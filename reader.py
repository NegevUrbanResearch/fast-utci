"""
Data reader module for fast-utci.

Reads 3D models and weather data for UTCI calculations.
Outputs are prepared for MRT calculation with Radiance.
"""

from pathlib import Path
from typing import Tuple, Union
import trimesh
import pandas as pd
from ladybug.epw import EPW


SUPPORTED_MODEL_FORMATS = ['.glb', '.gltf']

def read_model(file_path: Union[str, Path]) -> trimesh.Trimesh:
    """
    Read a 3D model file and return a trimesh object ready for Radiance.
    
    Handles both simple meshes and complex scenes by combining all geometry
    into a single mesh suitable for MRT calculations.
    """
    file_path = Path(file_path)
    assert file_path.exists(), f"Model file not found: {file_path}"
    assert file_path.suffix.lower() in SUPPORTED_MODEL_FORMATS, f"Unsupported format: {file_path.suffix}"
    
    loaded = trimesh.load(str(file_path))
    
    if isinstance(loaded, trimesh.Trimesh):
        return loaded
    elif isinstance(loaded, trimesh.Scene):
        # Extract all meshes from scene
        meshes = []
        for geom in loaded.geometry.values():
            if isinstance(geom, trimesh.Trimesh):
                meshes.append(geom)
        
        assert meshes, "No valid meshes found in scene"
        
        # Combine all meshes for Radiance processing
        return trimesh.util.concatenate(meshes)
    else:
        assert False, f"Unsupported object type: {type(loaded)}"
    


def read_weather_data(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Read EPW weather file and extract UTCI inputs as DataFrame.
    
    Returns weather data ready for MRT calculation with Radiance.
    """
    file_path = Path(file_path)
    assert file_path.exists(), f"Weather file not found: {file_path}"
    
    epw = EPW(str(file_path))
    
    # Extract data for UTCI calculations
    data = {
        'datetime': epw.dry_bulb_temperature.datetimes,
        'air_temp': epw.dry_bulb_temperature.values,
        'wind_speed': epw.wind_speed.values,
        'relative_humidity': epw.relative_humidity.values,
        'global_horizontal_radiation': epw.global_horizontal_radiation.values,
        'direct_normal_radiation': epw.direct_normal_radiation.values,
        'diffuse_horizontal_radiation': epw.diffuse_horizontal_radiation.values
    }
    
    return pd.DataFrame(data)

# for convenience
def read_project_data(model_path: Union[str, Path], 
                     weather_path: Union[str, Path]) -> Tuple[trimesh.Trimesh, pd.DataFrame]:
    """
    Read model and weather data for UTCI calculations.
    
    Returns model mesh ready for Radiance and weather DataFrame for MRT calculation.
    """
    model = read_model(model_path)
    weather_df = read_weather_data(weather_path)
    
    return model, weather_df




def main():
    """Example usage of the reader module."""
    model_path = "data/rec_model_no_curve.glb"
    weather_path = "data/ISR_D_Beer.Sheva.401900_TMYx/ISR_D_Beer.Sheva.401900_TMYx.epw"
    
    try:
        # Read project data
        model, weather_df = read_project_data(model_path, weather_path)
        
        print(f"Model loaded: {len(model.vertices)} vertices, {len(model.faces)} faces")
        print(f"Weather data: {len(weather_df)} hours")
        print(f"Date range: {weather_df['datetime'].iloc[0]} to {weather_df['datetime'].iloc[-1]}")
        print("\nSample weather data:")
        print(weather_df.head())
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
