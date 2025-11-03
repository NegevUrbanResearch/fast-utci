"""
Shared configuration for fast-utci.

This module provides configuration classes and utilities shared across both
MRT and UTCI calculation modules, including parallel processing settings,
performance optimizations, and environment variable helpers.
"""

from dataclasses import dataclass, field
from typing import Optional
import os
import multiprocessing as mp


@dataclass
class ParallelConfig:
    """
    Shared parallel processing configuration.
    
    Used by both MRT and UTCI calculators to control parallel execution behavior.
    """
    n_workers: Optional[int] = None  # None = auto-detect (CPU count - 1)
    show_progress: bool = True  # Show progress bars during calculations
    parallel_threshold: int = 50  # Minimum items to trigger parallel processing
    
    def get_n_workers(self) -> int:
        """Get the actual number of workers to use."""
        if self.n_workers is None:
            return max(1, mp.cpu_count() - 1)
        return max(1, self.n_workers)


@dataclass
class PerformanceConfig:
    """
    Shared performance optimization settings.
    
    Controls batch sizes, ray tracing parameters, and other performance-critical settings.
    """
    batch_size: int = 10000  # Ray intersection batch size
    ray_max_distance: float = 1000.0  # Maximum ray distance for intersection testing (meters)


# Environment variable helper functions
def get_bool_env(key: str, default: bool) -> bool:
    """
    Parse boolean environment variable.
    
    Args:
        key: Environment variable name
        default: Default value if not set or invalid
        
    Returns:
        Boolean value
        
    Examples:
        >>> get_bool_env("FAST_UTCI_VERBOSE", False)
        False  # if not set
    """
    value = os.getenv(key, "").lower()
    if value in ("1", "true", "yes", "on"):
        return True
    elif value in ("0", "false", "no", "off", ""):
        return default
    return default


def get_int_env(key: str, default: int) -> int:
    """
    Parse integer environment variable.
    
    Args:
        key: Environment variable name
        default: Default value if not set or invalid
        
    Returns:
        Integer value
        
    Examples:
        >>> get_int_env("FAST_UTCI_N_WORKERS", 4)
        4  # if not set or invalid
    """
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


def get_float_env(key: str, default: float) -> float:
    """
    Parse float environment variable.
    
    Args:
        key: Environment variable name
        default: Default value if not set or invalid
        
    Returns:
        Float value
    """
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default


def get_str_env(key: str, default: str) -> str:
    """
    Parse string environment variable.
    
    Args:
        key: Environment variable name
        default: Default value if not set
        
    Returns:
        String value
    """
    return os.getenv(key, default)


# Default instances
DEFAULT_PARALLEL_CONFIG = ParallelConfig()
DEFAULT_PERFORMANCE_CONFIG = PerformanceConfig()

