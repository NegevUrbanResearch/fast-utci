"""
Configuration parameters for MRT calculations.

Centralized configuration for easy parameter experimentation and tuning.
All parameters have sensible defaults but can be easily modified.
"""

from dataclasses import dataclass
from typing import Optional
import os


@dataclass
class MRTConfig:
    """Configuration parameters for MRT calculations."""
    
    # Human body parameters
    human_height: float = 1.8  # meters
    pt_count: int = 1  # Number of sample points along human height
    absorptivity: float = 0.7  # Solar absorptivity of skin/clothing (0-1)
    emissivity: float = 0.95   # Longwave emissivity of skin/clothing (0-1)
    
    # Grid generation parameters
    grid_size: float = 10.0  # meters - grid spacing for analysis
    grid_offset: float = 0.0  # meters - offset distance from surface
    
    # Solar and analysis parameters
    north_degrees: float = 0.0  # degrees - North angle (0 = Y+ is north)
    ground_reflectance: float = 0.25  # Ground reflectance factor (0-1)
    
    # Performance parameters
    n_workers: Optional[int] = None  # None = auto-detect (CPU count - 1)
    batch_size: int = 10000  # Ray intersection batch size
    ray_max_distance: float = 1000.0  # Maximum ray distance for intersection testing
    show_progress: bool = True  # Show progress bars during calculations

    # Embree tuning (applied when backend supports)
    embree_quality: str = "auto"  # "auto" | "low" | "medium" | "high"
    embree_build_bvh: bool = True  # Pre-build acceleration structure when possible
    embree_ray_packet_size: int = 0  # 0 = auto; otherwise typical values: 4, 8, 16

    # Vectorization
    enable_vectorized_solar: bool = False  # Non-breaking default; can be overridden via env
    
    # SolarCal parameters
    sky_exposure: float = 1.0  # Default sky exposure fraction (0-1)
    fract_body_exp: float = 1.0  # Default solar exposure fraction (0-1)
    
    # File I/O parameters
    csv_encoding: str = 'utf-8'
    csv_index: bool = False  # Whether to include row indices in CSV exports


# Default configuration instance
DEFAULT_CONFIG = MRTConfig()

# Backward compatibility constants
DEFAULT_HUMAN_HEIGHT = DEFAULT_CONFIG.human_height
DEFAULT_PT_COUNT = DEFAULT_CONFIG.pt_count
DEFAULT_ABSORPTIVITY = DEFAULT_CONFIG.absorptivity
DEFAULT_EMISSIVITY = DEFAULT_CONFIG.emissivity
DEFAULT_GRID_SIZE = DEFAULT_CONFIG.grid_size
DEFAULT_GRID_OFFSET = DEFAULT_CONFIG.grid_offset
DEFAULT_NORTH_DEGREES = DEFAULT_CONFIG.north_degrees
DEFAULT_GROUND_REFLECTANCE = DEFAULT_CONFIG.ground_reflectance
DEFAULT_N_WORKERS = DEFAULT_CONFIG.n_workers
DEFAULT_BATCH_SIZE = DEFAULT_CONFIG.batch_size
DEFAULT_RAY_MAX_DISTANCE = DEFAULT_CONFIG.ray_max_distance
DEFAULT_SHOW_PROGRESS = DEFAULT_CONFIG.show_progress
DEFAULT_EMBREE_QUALITY = DEFAULT_CONFIG.embree_quality
DEFAULT_EMBREE_BUILD_BVH = DEFAULT_CONFIG.embree_build_bvh
DEFAULT_EMBREE_PACKET_SIZE = DEFAULT_CONFIG.embree_ray_packet_size
DEFAULT_ENABLE_VECTORIZED_SOLAR = DEFAULT_CONFIG.enable_vectorized_solar
DEFAULT_SKY_EXPOSURE = DEFAULT_CONFIG.sky_exposure
DEFAULT_FRACT_BODY_EXP = DEFAULT_CONFIG.fract_body_exp
CSV_ENCODING = DEFAULT_CONFIG.csv_encoding
CSV_INDEX = DEFAULT_CONFIG.csv_index


@dataclass
class EnvironmentConfig:
    """
    Environment variable configuration for MRT calculations.
    
    Centralizes all environment variable reading with validation, type conversion,
    and documentation. Provides a single source of truth for all env-based settings.
    """
    
    # Performance optimizations
    vectorized_solar: bool = False  # FAST_UTCI_VECTORIZED_SOLAR
    batch_positions: bool = False  # FAST_UTCI_BATCH_POSITIONS
    
    # Ray intersector settings
    intersector: str = "auto"  # FAST_UTCI_INTERSECTOR: auto|embree|trimesh
    intersects_any: bool = False  # FAST_UTCI_INTERSECTS_ANY
    
    # Embree-specific settings
    embree_quality: str = "auto"  # FAST_UTCI_EMBREE_QUALITY: auto|low|medium|high
    embree_build_bvh: bool = True  # FAST_UTCI_EMBREE_BUILD_BVH
    embree_packet_size: int = 0  # FAST_UTCI_EMBREE_PACKET_SIZE: 0=auto, 4|8|16
    
    @classmethod
    def from_environment(cls) -> 'EnvironmentConfig':
        """Create EnvironmentConfig from environment variables."""
        def get_bool(key: str, default: bool) -> bool:
            """Parse boolean environment variable."""
            value = os.getenv(key, "").lower()
            if value in ("1", "true", "yes", "on"):
                return True
            elif value in ("0", "false", "no", "off", ""):
                return default
            return default
        
        def get_int(key: str, default: int) -> int:
            """Parse integer environment variable."""
            try:
                return int(os.getenv(key, str(default)))
            except ValueError:
                return default
        
        def get_str(key: str, default: str) -> str:
            """Parse string environment variable."""
            return os.getenv(key, default)
        
        return cls(
            vectorized_solar=get_bool("FAST_UTCI_VECTORIZED_SOLAR", False),
            batch_positions=get_bool("FAST_UTCI_BATCH_POSITIONS", False),
            intersector=get_str("FAST_UTCI_INTERSECTOR", "auto"),
            intersects_any=get_bool("FAST_UTCI_INTERSECTS_ANY", False),
            embree_quality=get_str("FAST_UTCI_EMBREE_QUALITY", "auto"),
            embree_build_bvh=get_bool("FAST_UTCI_EMBREE_BUILD_BVH", True),
            embree_packet_size=get_int("FAST_UTCI_EMBREE_PACKET_SIZE", 0),
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


# Validation/testing parameters (keep for backward compatibility)
VALIDATION_ANALYSIS_PERIOD = {
    'start_month': 8,
    'start_day': 15,
    'start_hour': 0,
    'end_month': 8,
    'end_day': 15,
    'end_hour': 23
}
VALIDATION_TARGET_HOURS = [13]  # Hour 13 = 1-2 PM
