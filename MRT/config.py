"""
Configuration parameters for MRT calculations.

Centralized configuration for easy parameter experimentation and tuning.
All parameters have sensible defaults but can be easily modified.
"""

from dataclasses import dataclass
from typing import Optional


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
DEFAULT_SKY_EXPOSURE = DEFAULT_CONFIG.sky_exposure
DEFAULT_FRACT_BODY_EXP = DEFAULT_CONFIG.fract_body_exp
CSV_ENCODING = DEFAULT_CONFIG.csv_encoding
CSV_INDEX = DEFAULT_CONFIG.csv_index

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
