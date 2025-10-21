"""
Configuration parameters for MRT calculations.

Centralized configuration for MRT-specific parameters. Shared configuration
(parallel processing, performance) is imported from fast_utci.shared.
"""

from dataclasses import dataclass, field
from typing import Optional
import os

from fast_utci.shared.config import ParallelConfig, PerformanceConfig


@dataclass
class MRTConfig:
    """
    Configuration parameters for MRT calculations.
    
    This class contains MRT-specific calculation parameters. Shared settings
    like parallel processing and performance optimization are in separate
    config objects for reusability across MRT and UTCI modules.
    """
    
    # Human body parameters
    human_height: float = 1.8  # meters
    pt_count: int = 1  # Number of sample points along human height
    absorptivity: float = 0.7  # Solar absorptivity of skin/clothing (0-1)
    emissivity: float = 0.95   # Longwave emissivity of skin/clothing (0-1)
    
    # Solar and analysis parameters
    north_degrees: float = 0.0  # degrees - North angle (0 = Y+ is north)
    ground_reflectance: float = 0.25  # Ground reflectance factor (0-1)
    sky_exposure: float = 1.0  # Default sky exposure fraction (0-1)
    fract_body_exp: float = 1.0  # Default solar exposure fraction (0-1)
    
    # Shared configuration (composition pattern)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    
    # File I/O parameters
    csv_encoding: str = 'utf-8'
    csv_index: bool = False  # Whether to include row indices in CSV exports
    
    # Backward compatibility properties
    @property
    def n_workers(self) -> Optional[int]:
        """Backward compatibility for n_workers access."""
        return self.parallel.n_workers
    
    @property
    def show_progress(self) -> bool:
        """Backward compatibility for show_progress access."""
        return self.parallel.show_progress
    
    @property
    def batch_size(self) -> int:
        """Backward compatibility for batch_size access."""
        return self.performance.batch_size
    
    @property
    def ray_max_distance(self) -> float:
        """Backward compatibility for ray_max_distance access."""
        return self.performance.ray_max_distance


# Default configuration instance
DEFAULT_CONFIG = MRTConfig()

# Backward compatibility constants
DEFAULT_HUMAN_HEIGHT = DEFAULT_CONFIG.human_height
DEFAULT_PT_COUNT = DEFAULT_CONFIG.pt_count
DEFAULT_ABSORPTIVITY = DEFAULT_CONFIG.absorptivity
DEFAULT_EMISSIVITY = DEFAULT_CONFIG.emissivity
DEFAULT_NORTH_DEGREES = DEFAULT_CONFIG.north_degrees
DEFAULT_GROUND_REFLECTANCE = DEFAULT_CONFIG.ground_reflectance
DEFAULT_SKY_EXPOSURE = DEFAULT_CONFIG.sky_exposure
DEFAULT_FRACT_BODY_EXP = DEFAULT_CONFIG.fract_body_exp
CSV_ENCODING = DEFAULT_CONFIG.csv_encoding
CSV_INDEX = DEFAULT_CONFIG.csv_index

# Shared config backward compatibility
DEFAULT_N_WORKERS = DEFAULT_CONFIG.parallel.n_workers
DEFAULT_SHOW_PROGRESS = DEFAULT_CONFIG.parallel.show_progress
DEFAULT_BATCH_SIZE = DEFAULT_CONFIG.performance.batch_size
DEFAULT_RAY_MAX_DISTANCE = DEFAULT_CONFIG.performance.ray_max_distance


@dataclass
class EnvironmentConfig:
    """
    Environment variable configuration for MRT and UTCI calculations.
    
    Centralizes all environment variable reading with validation, type conversion,
    and documentation. Provides a single source of truth for all env-based settings
    shared across MRT and UTCI modules.
    """
    
    # Performance optimizations
    vectorized_solar: bool = False  # FAST_UTCI_VECTORIZED_SOLAR
    vectorized_utci: bool = True  # FAST_UTCI_VECTORIZED_UTCI
    batch_positions: bool = False  # FAST_UTCI_BATCH_POSITIONS
    
    # Ray intersector settings
    intersector: str = "auto"  # FAST_UTCI_INTERSECTOR: auto|embree|trimesh
    intersects_any: bool = False  # FAST_UTCI_INTERSECTS_ANY
    
    # Embree-specific settings
    embree_quality: str = "auto"  # FAST_UTCI_EMBREE_QUALITY: auto|low|medium|high
    embree_build_bvh: bool = True  # FAST_UTCI_EMBREE_BUILD_BVH
    embree_packet_size: int = 0  # FAST_UTCI_EMBREE_PACKET_SIZE: 0=auto, 4|8|16
    
    # UTCI-specific settings
    include_weather_in_results: bool = True  # FAST_UTCI_INCLUDE_WEATHER_IN_RESULTS
    include_datetime_in_results: bool = True  # FAST_UTCI_INCLUDE_DATETIME_IN_RESULTS
    
    @classmethod
    def from_environment(cls) -> 'EnvironmentConfig':
        """Create EnvironmentConfig from environment variables."""
        from fast_utci.shared.config import get_bool_env, get_int_env, get_str_env
        
        return cls(
            vectorized_solar=get_bool_env("FAST_UTCI_VECTORIZED_SOLAR", False),
            vectorized_utci=get_bool_env("FAST_UTCI_VECTORIZED_UTCI", True),
            batch_positions=get_bool_env("FAST_UTCI_BATCH_POSITIONS", False),
            intersector=get_str_env("FAST_UTCI_INTERSECTOR", "auto"),
            intersects_any=get_bool_env("FAST_UTCI_INTERSECTS_ANY", False),
            embree_quality=get_str_env("FAST_UTCI_EMBREE_QUALITY", "auto"),
            embree_build_bvh=get_bool_env("FAST_UTCI_EMBREE_BUILD_BVH", True),
            embree_packet_size=get_int_env("FAST_UTCI_EMBREE_PACKET_SIZE", 0),
            include_weather_in_results=get_bool_env("FAST_UTCI_INCLUDE_WEATHER_IN_RESULTS", True),
            include_datetime_in_results=get_bool_env("FAST_UTCI_INCLUDE_DATETIME_IN_RESULTS", True),
        )


# Global environment config instance
_env_config: Optional[EnvironmentConfig] = None


def get_env_config() -> EnvironmentConfig:
    """
    Get the global environment configuration.
    
    Lazily creates and caches the configuration on first access.
    
    Returns:
        EnvironmentConfig instance with current environment settings
    """
    global _env_config
    if _env_config is None:
        _env_config = EnvironmentConfig.from_environment()
    return _env_config
