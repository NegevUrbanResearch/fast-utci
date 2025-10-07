"""
Adapters and strategy patterns for MRT calculations.

Provides abstraction layers for different data sources and intersection backends,
making the codebase more flexible, testable, and extensible.
"""

from __future__ import annotations
import numpy as np
from typing import Protocol, List, Union, Optional, Any
from datetime import datetime
import warnings

from .exceptions import WeatherDataError, IntersectorError, ConfigurationError


# ============================================================================
# Weather Data Adapters
# ============================================================================

class WeatherDataSource(Protocol):
    """Protocol for weather data sources."""
    
    def get_temperature(self) -> np.ndarray:
        """Get air temperature array (°C)."""
        ...
    
    def get_direct_radiation(self) -> np.ndarray:
        """Get direct normal radiation array (W/m²)."""
        ...
    
    def get_diffuse_radiation(self) -> np.ndarray:
        """Get diffuse horizontal radiation array (W/m²)."""
        ...
    
    def get_infrared_radiation(self) -> np.ndarray:
        """Get horizontal infrared radiation array (W/m²)."""
        ...
    
    def get_datetimes(self) -> List:
        """Get datetime objects corresponding to data arrays."""
        ...


class EPWAdapter:
    """Adapter for Ladybug EPW weather data."""
    
    def __init__(self, epw_data: Any):
        """
        Initialize EPW adapter.
        
        Args:
            epw_data: Ladybug EPW object
        """
        if not hasattr(epw_data, 'dry_bulb_temperature'):
            raise WeatherDataError("Invalid EPW object: missing dry_bulb_temperature")
        self.epw_data = epw_data
    
    def get_temperature(self) -> np.ndarray:
        """Get air temperature array."""
        return np.array(self.epw_data.dry_bulb_temperature.values)
    
    def get_direct_radiation(self) -> np.ndarray:
        """Get direct normal radiation array."""
        return np.array(self.epw_data.direct_normal_radiation.values)
    
    def get_diffuse_radiation(self) -> np.ndarray:
        """Get diffuse horizontal radiation array."""
        return np.array(self.epw_data.diffuse_horizontal_radiation.values)
    
    def get_infrared_radiation(self) -> np.ndarray:
        """Get horizontal infrared radiation array."""
        return np.array(self.epw_data.horizontal_infrared_radiation_intensity.values)
    
    def get_datetimes(self) -> List:
        """Get datetime objects."""
        return self.epw_data.dry_bulb_temperature.datetimes


class DataFrameAdapter:
    """Adapter for pandas DataFrame weather data."""
    
    def __init__(self, df_data: Any):
        """
        Initialize DataFrame adapter.
        
        Args:
            df_data: pandas DataFrame with weather data columns
        """
        required_columns = ['air_temp', 'direct_normal_radiation', 
                           'diffuse_horizontal_radiation', 
                           'horizontal_infrared_radiation_intensity']
        
        for col in required_columns:
            if col not in df_data.columns:
                raise WeatherDataError(f"DataFrame missing required column: {col}")
        
        self.df_data = df_data
    
    def get_temperature(self) -> np.ndarray:
        """Get air temperature array."""
        return self.df_data['air_temp'].values
    
    def get_direct_radiation(self) -> np.ndarray:
        """Get direct normal radiation array."""
        return self.df_data['direct_normal_radiation'].values
    
    def get_diffuse_radiation(self) -> np.ndarray:
        """Get diffuse horizontal radiation array."""
        return self.df_data['diffuse_horizontal_radiation'].values
    
    def get_infrared_radiation(self) -> np.ndarray:
        """Get horizontal infrared radiation array."""
        return self.df_data['horizontal_infrared_radiation_intensity'].values
    
    def get_datetimes(self) -> List:
        """Get datetime objects."""
        return self.df_data['datetime'].tolist()


def create_weather_adapter(weather_data: Any) -> Union[EPWAdapter, DataFrameAdapter]:
    """
    Factory function to create appropriate weather data adapter.
    
    Args:
        weather_data: EPW object or pandas DataFrame
        
    Returns:
        Appropriate adapter instance
        
    Raises:
        WeatherDataError: If data type is not recognized
    """
    if hasattr(weather_data, 'dry_bulb_temperature'):
        return EPWAdapter(weather_data)
    elif hasattr(weather_data, 'columns'):
        return DataFrameAdapter(weather_data)
    else:
        raise WeatherDataError(
            f"Unsupported weather data type: {type(weather_data)}. "
            "Expected EPW object or pandas DataFrame."
        )


# ============================================================================
# Ray Intersector Strategies
# ============================================================================

class RayIntersectorStrategy:
    """Base class for ray intersection strategies."""
    
    def __init__(self, mesh: Any):
        """
        Initialize intersector with mesh.
        
        Args:
            mesh: trimesh.Trimesh object
        """
        self.mesh = mesh
        self.intersector = None
        self.name = "unknown"
    
    def initialize(self) -> bool:
        """
        Initialize the intersection backend.
        
        Returns:
            True if initialization successful, False otherwise
        """
        raise NotImplementedError
    
    def get_intersector(self) -> Any:
        """Get the underlying ray intersector object."""
        return self.intersector
    
    def get_name(self) -> str:
        """Get the backend name."""
        return self.name


class EmbreeIntersectorStrategy(RayIntersectorStrategy):
    """Strategy for Embree-based ray intersection."""
    
    def __init__(self, mesh: Any, quality: str = "auto", 
                 build_bvh: bool = True, packet_size: int = 0):
        """
        Initialize Embree intersector strategy.
        
        Args:
            mesh: trimesh.Trimesh object
            quality: Quality setting ("auto", "low", "medium", "high")
            build_bvh: Whether to pre-build BVH structure
            packet_size: Ray packet size (0=auto, 4, 8, or 16)
        """
        super().__init__(mesh)
        self.quality = quality
        self.build_bvh = build_bvh
        self.packet_size = packet_size
        self.name = "embree"
    
    def initialize(self) -> bool:
        """Initialize Embree intersector."""
        try:
            from trimesh.ray.ray_pyembree import RayMeshIntersector as EmbreeIntersector
            self.intersector = EmbreeIntersector(self.mesh)
            
            # Apply quality settings if supported
            try:
                if hasattr(self.intersector, "set_quality") and self.quality != "auto":
                    self.intersector.set_quality(self.quality)
            except Exception:
                pass
            
            # Build BVH if supported
            try:
                if self.build_bvh and hasattr(self.intersector, "build_embree"):
                    self.intersector.build_embree()
            except Exception:
                pass
            
            # Set packet size if supported
            if self.packet_size in (4, 8, 16):
                try:
                    if hasattr(self.intersector, "set_packet_size"):
                        self.intersector.set_packet_size(self.packet_size)
                except Exception:
                    pass
            
            return True
            
        except ImportError:
            return False
        except Exception as e:
            warnings.warn(f"Embree initialization failed: {e}")
            return False


class TrimeshIntersectorStrategy(RayIntersectorStrategy):
    """Strategy for trimesh ray_triangle intersection."""
    
    def __init__(self, mesh: Any):
        """
        Initialize trimesh intersector strategy.
        
        Args:
            mesh: trimesh.Trimesh object
        """
        super().__init__(mesh)
        self.name = "trimesh"
    
    def initialize(self) -> bool:
        """Initialize trimesh intersector."""
        try:
            from trimesh.ray.ray_triangle import RayMeshIntersector as TriIntersector
            self.intersector = TriIntersector(self.mesh)
            return True
        except Exception as e:
            warnings.warn(f"Trimesh intersector initialization failed: {e}")
            return False


class FallbackIntersectorStrategy(RayIntersectorStrategy):
    """Fallback strategy when no intersector is available."""
    
    def __init__(self, mesh: Any):
        """
        Initialize fallback strategy.
        
        Args:
            mesh: trimesh.Trimesh object
        """
        super().__init__(mesh)
        self.name = "none"
    
    def initialize(self) -> bool:
        """Fallback has no initialization."""
        self.intersector = None
        return True


def create_intersector_strategy(mesh: Any, 
                                intersector_choice: str = "auto",
                                embree_quality: str = "auto",
                                embree_build_bvh: bool = True,
                                embree_packet_size: int = 0) -> RayIntersectorStrategy:
    """
    Factory function to create appropriate ray intersector strategy.
    
    Args:
        mesh: trimesh.Trimesh object
        intersector_choice: "auto", "embree", or "trimesh"
        embree_quality: Embree quality setting
        embree_build_bvh: Whether to build BVH for Embree
        embree_packet_size: Embree packet size
        
    Returns:
        Initialized intersector strategy
        
    Raises:
        ConfigurationError: If requested intersector is not available
    """
    # Try explicit choice first
    if intersector_choice == "embree":
        strategy = EmbreeIntersectorStrategy(
            mesh, embree_quality, embree_build_bvh, embree_packet_size
        )
        if strategy.initialize():
            return strategy
        else:
            raise ConfigurationError(
                "Embree intersector requested but pyembree is not available. "
                "Install pyembree or use 'auto' mode."
            )
    
    elif intersector_choice == "trimesh":
        strategy = TrimeshIntersectorStrategy(mesh)
        if strategy.initialize():
            return strategy
        else:
            raise ConfigurationError("Trimesh intersector initialization failed.")
    
    # Auto mode: try Embree, fallback to trimesh
    embree_strategy = EmbreeIntersectorStrategy(
        mesh, embree_quality, embree_build_bvh, embree_packet_size
    )
    if embree_strategy.initialize():
        return embree_strategy
    
    trimesh_strategy = TrimeshIntersectorStrategy(mesh)
    if trimesh_strategy.initialize():
        return trimesh_strategy
    
    # Last resort: fallback (no intersections)
    warnings.warn("No ray intersector available, using fallback (no-hit) mode")
    fallback_strategy = FallbackIntersectorStrategy(mesh)
    fallback_strategy.initialize()
    return fallback_strategy

