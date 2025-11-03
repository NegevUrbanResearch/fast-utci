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
    
    # Engine configuration
    intersector: str = "auto"  # auto|embree|trimesh
    embree_quality: str = "medium"  # low|medium|high
    embree_build_bvh: bool = True
    embree_packet_size: int = 0  # 0=auto, or 4|8|16
    intersects_any: bool = True

    # Feature toggles
    vectorized_solar: bool = True
    batch_positions: bool = False

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
