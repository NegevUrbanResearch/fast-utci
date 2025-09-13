"""
Data reader module for fast-utci.

This module provides classes for reading and parsing 3D models and weather data files.
Uses trimesh for 3D model processing and ladybug-core for EPW weather file handling.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Union
import numpy as np
import trimesh
import pandas as pd
from ladybug.epw import EPW


# Constants for ModelReader
SUPPORTED_MODEL_FORMATS = ['.glb', '.gltf']
# Constants for EPW validation
VALIDATION_SAMPLE_SIZE = 10
TEMPERATURE_RANGE = (-50, 60)  # Celsius
HUMIDITY_RANGE = (0, 100)  # Percentage

class ModelReader:
    """
    Class for reading and parsing 3D model files using trimesh.
    
    Supports GLB and GLTF formats and provides validation and information extraction
    capabilities for 3D meshes.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.supported_formats = SUPPORTED_MODEL_FORMATS
    
    def read_model(self, file_path: Union[str, Path]) -> trimesh.Trimesh:
        """
        Read a 3D model file and return a trimesh object.
        
        Uses trimesh.load() to handle various 3D formats and automatically
        combines multiple meshes from scene objects into a single mesh.
        
        Args:
            file_path: Path to the 3D model file (GLB or GLTF)
            
        Returns:
            trimesh.Trimesh: Loaded and processed 3D model mesh
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is not supported or model is invalid
            Exception: If there's an error loading the model
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Model file not found: {file_path}")
        
        if file_path.suffix.lower() not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_path.suffix}. "
                           f"Supported formats: {self.supported_formats}")
        
        try:
            loaded = trimesh.load(str(file_path))
            
            # Handle different types of loaded objects
            if isinstance(loaded, trimesh.Trimesh):
                mesh = loaded
            elif isinstance(loaded, trimesh.Scene):
                # If it's a scene, combine all meshes
                meshes = []
                for name, geom in loaded.geometry.items():
                    if isinstance(geom, trimesh.Trimesh):
                        meshes.append(geom)
                    elif hasattr(geom, 'geometry'):  # Handle nested scenes
                        for sub_name, sub_geom in geom.geometry.items():
                            if isinstance(sub_geom, trimesh.Trimesh):
                                meshes.append(sub_geom)
                
                if not meshes:
                    raise ValueError("No valid meshes found in the loaded scene")
                
                # Combine all meshes into one
                mesh = trimesh.util.concatenate(meshes)
            else:
                raise ValueError(f"Unsupported loaded object type: {type(loaded)}")
            
            return mesh
            
        except (trimesh.exceptions.TrimeshException, ValueError) as e:
            self.logger.error(f"Error loading 3D model {file_path}: {str(e)}")
            raise ValueError(f"Invalid 3D model file: {str(e)}") from e
    
    def get_model_info(self, mesh: trimesh.Trimesh) -> Dict:
        """
        Extract comprehensive information about the 3D model.
        
        Args:
            mesh: trimesh object to analyze
            
        Returns:
            Dict containing model statistics and properties
        """
        info = {
            'vertices_count': len(mesh.vertices),
            'faces_count': len(mesh.faces),
            'bounds': mesh.bounds.tolist(),
            'center': mesh.center_mass.tolist(),
            'volume': float(mesh.volume) if mesh.is_volume else None,
            'surface_area': float(mesh.area) if hasattr(mesh, 'area') else None,
            'is_watertight': mesh.is_watertight,
            'is_winding_consistent': mesh.is_winding_consistent,
            'scale': float(mesh.scale)
        }
        
        return info
    
    def validate_model(self, mesh: trimesh.Trimesh) -> Tuple[bool, List[str]]:
        """
        Validate the 3D model for common issues that could affect UTCI calculations.
        
        Args:
            mesh: trimesh object to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        if len(mesh.vertices) == 0:
            issues.append("Model has no vertices")
        
        if len(mesh.faces) == 0:
            issues.append("Model has no faces")
        
        if not mesh.is_winding_consistent:
            issues.append("Model has inconsistent face winding")
        
        # Check for degenerate faces
        if hasattr(mesh, 'faces'):
            face_areas = mesh.area_faces
            if np.any(face_areas == 0):
                issues.append("Model contains degenerate faces (zero area)")
        
        is_valid = len(issues) == 0
        
        if not is_valid:
            self.logger.warning(f"Model validation issues: {issues}")
        
        return is_valid, issues


class WeatherReader:
    """
    Class for reading and parsing weather data files using ladybug-core.
    
    Uses the EPW class from ladybug-core to parse EnergyPlus Weather files
    and extract meteorological data for UTCI calculations.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def read_epw(self, file_path: Union[str, Path]) -> EPW:
        """
        Read an EPW weather file using ladybug-core.
        
        Uses the EPW class from ladybug-core to parse EnergyPlus Weather files
        and provide access to hourly meteorological data.
        
        Args:
            file_path: Path to the EPW file
            
        Returns:
            EPW: ladybug-core EPW object containing weather data
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            Exception: If ladybug-core fails to parse the EPW file
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"EPW file not found: {file_path}")
        
        try:
            epw = EPW(str(file_path))
            return epw
            
        except Exception as e:
            self.logger.warning(f"Error loading EPW file {file_path}: {str(e)}")
            raise
    
    def get_weather_info(self, epw: EPW) -> Dict:
        """
        Extract comprehensive information about the weather data.
        
        Args:
            epw: ladybug-core EPW object
            
        Returns:
            Dict containing weather data statistics and location information
        """
        # Get the length from one of the weather data collections
        data_points = len(epw.dry_bulb_temperature.values)
        
        # Get date range from the data collection
        start_date = epw.dry_bulb_temperature.datetimes[0].strftime('%Y-%m-%d %H:%M')
        end_date = epw.dry_bulb_temperature.datetimes[-1].strftime('%Y-%m-%d %H:%M')
        
        info = {
            'location': {
                'city': epw.location.city,
                'country': epw.location.country,
                'latitude': epw.location.latitude,
                'longitude': epw.location.longitude,
                'timezone': getattr(epw.location, 'time_zone', None),
                'elevation': epw.location.elevation
            },
            'data_points': data_points,
            'date_range': {
                'start': start_date,
                'end': end_date
            }
        }
        
        return info
    
    def extract_utci_inputs(self, epw: EPW) -> pd.DataFrame:
        """
        Extract the inputs needed for UTCI calculations from EPW data.
        
        Uses ladybug-core data collections to extract hourly meteorological data
        required for UTCI calculations: air temperature, wind speed, relative humidity,
        and solar radiation components.
        
        Args:
            epw: ladybug-core EPW object
            
        Returns:
            pd.DataFrame with columns: datetime, air_temp, wind_speed, relative_humidity,
            global_horizontal_radiation, direct_normal_radiation, diffuse_horizontal_radiation
        """
        # Extract data from the hourly collections
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
    
    def validate_weather_data(self, epw: EPW) -> Tuple[bool, List[str]]:
        """
        Validate the weather data for common issues that could affect UTCI calculations.
        
        Args:
            epw: ladybug-core EPW object
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check if data collections exist and have data
        if len(epw.dry_bulb_temperature.values) == 0:
            issues.append("No temperature data found")
            return False, issues
        
        data_length = len(epw.dry_bulb_temperature.values)
        
        # Check for missing critical data in first N hours (vectorized)
        sample_size = min(VALIDATION_SAMPLE_SIZE, data_length)
        sample_temps = epw.dry_bulb_temperature.values[:sample_size]
        sample_winds = epw.wind_speed.values[:sample_size]
        sample_humids = epw.relative_humidity.values[:sample_size]
        
        missing_data = []
        if any(t is None for t in sample_temps):
            missing_data.append("air temperature")
        if any(w is None for w in sample_winds):
            missing_data.append("wind speed")
        if any(h is None for h in sample_humids):
            missing_data.append("relative humidity")
        
        if missing_data:
            issues.append(f"Missing data for: {', '.join(set(missing_data))}")
        
        # Check for reasonable value ranges (vectorized)
        valid_temps = [t for t in epw.dry_bulb_temperature.values if t is not None]
        if valid_temps:
            if max(valid_temps) > TEMPERATURE_RANGE[1] or min(valid_temps) < TEMPERATURE_RANGE[0]:
                issues.append(f"Air temperature values outside reasonable range {TEMPERATURE_RANGE}°C")
        
        valid_humids = [h for h in epw.relative_humidity.values if h is not None]
        if valid_humids:
            if max(valid_humids) > HUMIDITY_RANGE[1] or min(valid_humids) < HUMIDITY_RANGE[0]:
                issues.append(f"Relative humidity values outside valid range {HUMIDITY_RANGE}%")
        
        is_valid = len(issues) == 0
        
        if not is_valid:
            self.logger.warning(f"Weather data validation issues: {issues}")
        
        return is_valid, issues


def read_project_data(model_path: Union[str, Path], 
                     weather_path: Union[str, Path]) -> Tuple[trimesh.Trimesh, EPW]:
    """
    Read both model and weather data for a project.
    
    Convenience function that uses ModelReader and WeatherReader to load
    both 3D model and EPW weather data in a single call.
    
    Args:
        model_path: Path to the 3D model file (GLB or GLTF)
        weather_path: Path to the EPW weather file
        
    Returns:
        Tuple of (model_mesh, weather_data) where weather_data is a ladybug-core EPW object
    """
    model_reader = ModelReader()
    weather_reader = WeatherReader()
    
    model = model_reader.read_model(model_path)
    weather = weather_reader.read_epw(weather_path)
    
    return model, weather


def validate_project_data(model: trimesh.Trimesh, 
                        weather: EPW) -> Tuple[bool, Dict[str, List[str]]]:
    """
    Validate both model and weather data for project readiness.
    
    Runs validation checks on both the 3D model and weather data to ensure
    they are suitable for UTCI calculations.
    
    Args:
        model: trimesh object containing 3D model data
        weather: ladybug-core EPW object containing weather data
        
    Returns:
        Tuple of (all_valid, validation_results) where validation_results
        contains lists of issues found for each data type
    """
    model_reader = ModelReader()
    weather_reader = WeatherReader()
    
    model_valid, model_issues = model_reader.validate_model(model)
    weather_valid, weather_issues = weather_reader.validate_weather_data(weather)
    
    validation_results = {
        'model': model_issues,
        'weather': weather_issues
    }
    
    all_valid = model_valid and weather_valid
    
    if not all_valid:
        logger = logging.getLogger(__name__)
        logger.warning(f"Project data validation issues: {validation_results}")
    
    return all_valid, validation_results


def main():
    """Example usage of the reader module."""
    # Set up minimal logging
    logging.basicConfig(level=logging.WARNING)
    
    # Example paths (adjust these to your actual file paths)
    model_path = "data/rec_model_no_curve.glb"
    weather_path = "data/ISR_D_Beer.Sheva.401900_TMYx/ISR_D_Beer.Sheva.401900_TMYx.epw"
    
    try:
        # Read project data
        model, weather = read_project_data(model_path, weather_path)
        
        # Get information about the data
        model_reader = ModelReader()
        weather_reader = WeatherReader()
        
        model_info = model_reader.get_model_info(model)
        weather_info = weather_reader.get_weather_info(weather)
        
        print("Model Information:")
        print(f"  Vertices: {model_info['vertices_count']}")
        print(f"  Faces: {model_info['faces_count']}")
        print(f"  Volume: {model_info['volume']}")
        print(f"  Surface Area: {model_info['surface_area']:.2f}")
        
        print("\nWeather Information:")
        print(f"  Location: {weather_info['location']['city']}, {weather_info['location']['country']}")
        print(f"  Data Points: {weather_info['data_points']}")
        print(f"  Date Range: {weather_info['date_range']['start']} to {weather_info['date_range']['end']}")
        
        # Validate data
        is_valid, validation_results = validate_project_data(model, weather)
        print(f"\nData Validation: {'PASSED' if is_valid else 'FAILED'}")
        if not is_valid:
            print("Issues found:")
            for data_type, issues in validation_results.items():
                if issues:
                    print(f"  {data_type}: {issues}")
        
        # Extract UTCI inputs
        utci_inputs = weather_reader.extract_utci_inputs(weather)
        print(f"\nUTCI Input Data Shape: {utci_inputs.shape}")
        print("Sample UTCI inputs:")
        print(utci_inputs.head())
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
