"""
Unified configuration for fast-utci.

This module provides all configuration classes and utilities, including:
- Shared configs (ParallelConfig, PerformanceConfig)
- Module-specific configs (MRTConfig, UTCIConfig)
- Application-level config (AppConfig)
- TOML loading (load_config)
- Environment variable helpers
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import os
import multiprocessing as mp
import tomllib


@dataclass
class ParallelConfig:
    """
    Shared parallel processing configuration.
    
    Used by both MRT and UTCI calculators to control parallel execution behavior.
    
    All values must be provided from TOML configuration (no hardcoded defaults).
    """
    n_workers: Optional[int]  # None = auto-detect (CPU count - 1)
    show_progress: bool  # Show progress bars during calculations
    parallel_threshold: int  # Minimum items to trigger parallel processing
    
    def get_n_workers(self) -> int:
        """Get the actual number of workers to use."""
        if self.n_workers is None:
            return max(1, mp.cpu_count() - 1)
        return max(1, self.n_workers)
    
    def with_overrides(self, n_workers: Optional[int] = None, 
                      show_progress: Optional[bool] = None) -> 'ParallelConfig':
        """
        Create a new ParallelConfig with overridden values.
        
        Args:
            n_workers: Optional override for n_workers
            show_progress: Optional override for show_progress
            
        Returns:
            New ParallelConfig with overridden values
        """
        return ParallelConfig(
            n_workers=n_workers if n_workers is not None else self.n_workers,
            show_progress=show_progress if show_progress is not None else self.show_progress,
            parallel_threshold=self.parallel_threshold
        )


@dataclass
class PerformanceConfig:
    """
    Shared performance optimization settings.
    
    Controls batch sizes, ray tracing parameters, and other performance-critical settings.
    
    All values must be provided from TOML configuration (no hardcoded defaults).
    """
    batch_size: int  # Ray intersection batch size
    ray_max_distance: float  # Maximum ray distance for intersection testing (meters)


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


@dataclass
class MRTConfig:
    """
    Configuration parameters for MRT calculations.
    
    This class contains MRT-specific calculation parameters. Shared settings
    like parallel processing and performance optimization are in separate
    config objects for reusability across MRT and UTCI modules.
    
    All values must be provided from TOML configuration (no hardcoded defaults).
    """
    
    # Human body parameters
    human_height: float  # meters
    pt_count: int  # Number of sample points along human height
    absorptivity: float  # Solar absorptivity of skin/clothing (0-1)
    emissivity: float   # Longwave emissivity of skin/clothing (0-1)
    
    # Solar and analysis parameters
    north_degrees: float  # degrees - North angle (0 = Y+ is north)
    ground_reflectance: float  # Ground reflectance factor (0-1)
    
    # Engine configuration
    intersector: str  # auto|embree|trimesh
    embree_quality: str  # low|medium|high
    embree_build_bvh: bool
    embree_packet_size: int  # 0=auto, or 4|8|16
    intersects_any: bool

    # Feature toggles
    vectorized_solar: bool
    batch_positions: bool

    # Shared configuration (composition pattern)
    parallel: ParallelConfig
    performance: PerformanceConfig
    
    # File I/O parameters
    csv_encoding: str
    csv_index: bool  # Whether to include row indices in CSV exports
    
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


@dataclass
class UTCIConfig:
    """
    Configuration parameters for UTCI calculations.
    
    This class contains UTCI-specific calculation parameters. Shared settings
    like parallel processing are in separate config objects for reusability.
    
    All values must be provided from TOML configuration (no hardcoded defaults).
    """
    
    # Calculation settings
    enable_vectorized: bool  # Use vectorized UTCI calculations when possible
    
    # Result serialization (what to include in output)
    include_weather_in_results: bool  # Include air_temp, wind_speed, RH in results
    include_datetime_in_results: bool  # Include datetime information in results
    
    # Shared configuration (composition pattern)
    parallel: ParallelConfig
    
    # File I/O parameters
    csv_encoding: str
    csv_index: bool  # Whether to include row indices in CSV exports
    
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
    def parallel_threshold(self) -> int:
        """Backward compatibility for parallel_threshold access."""
        return self.parallel.parallel_threshold


@dataclass
class AppConfig:
    """Top-level application configuration loaded from TOML."""
    parallel: ParallelConfig
    performance: PerformanceConfig
    mrt: MRTConfig
    utci: UTCIConfig


def _resolve_n_workers(value: Optional[Any]) -> Optional[int]:
    """Resolve n_workers supporting "auto" and None."""
    if value is None:
        return None
    if isinstance(value, str) and value.lower() == "auto":
        return None
    try:
        return int(value)
    except Exception:
        return None


def _validate_config(data: dict) -> None:
    """Strictly validate sections and keys; error on missing/unknown."""
    required_sections = [
        "parallel", "performance", "engine", "features", "mrt", "utci"
    ]
    for sec in required_sections:
        if sec not in data or not isinstance(data[sec], dict):
            raise ValueError(f"Missing required section [{sec}] in TOML")

    required_keys = {
        "parallel": ["n_workers", "show_progress", "parallel_threshold"],
        "performance": ["batch_size", "ray_max_distance"],
        "engine": [
            "intersector", "embree_quality", "embree_build_bvh",
            "embree_packet_size", "intersects_any"
        ],
        "features": [
            "vectorized_solar", "batch_positions",
            "include_weather_in_results", "include_datetime_in_results"
        ],
        "mrt": [
            "human_height", "pt_count", "absorptivity", "emissivity",
            "north_degrees", "ground_reflectance", "csv_encoding", "csv_index"
        ],
        "utci": ["enable_vectorized", "csv_encoding", "csv_index"],
    }

    for sec, keys in required_keys.items():
        d = data[sec]
        unknown = set(d.keys()) - set(keys)
        if unknown:
            raise ValueError(f"Unknown keys in [{sec}]: {sorted(unknown)}")
        missing = [k for k in keys if k not in d]
        if missing:
            raise ValueError(f"Missing required keys in [{sec}]: {missing}")


def load_config(path: Optional[str] = None) -> AppConfig:
    """
    Load application configuration from TOML.
    
    Parses TOML configuration (single source of truth) and constructs
    domain configs used across MRT and UTCI modules. No environment
    variable fallback is used here.

    Args:
        path: Optional path to TOML file. Defaults to "fast_utci.toml" at repo root.

    Returns:
        AppConfig instance.

    Raises:
        FileNotFoundError if the TOML file is missing.
        ValueError for invalid structures.
    """
    cfg_path = Path(path or "fast_utci.toml")
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Missing config: {cfg_path}. Copy fast_utci.example.toml to fast_utci.toml and edit."
        )

    data = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
    _validate_config(data)

    par = data["parallel"]
    perf = data["performance"]
    engine = data["engine"]
    features = data["features"]
    mrt = data["mrt"]
    utci = data["utci"]

    parallel = ParallelConfig(
        n_workers=_resolve_n_workers(par["n_workers"]),
        show_progress=bool(par["show_progress"]),
        parallel_threshold=int(par["parallel_threshold"]),
    )

    performance = PerformanceConfig(
        batch_size=int(perf["batch_size"]),
        ray_max_distance=float(perf["ray_max_distance"]),
    )

    mrt_cfg = MRTConfig(
        human_height=float(mrt["human_height"]),
        pt_count=int(mrt["pt_count"]),
        absorptivity=float(mrt["absorptivity"]),
        emissivity=float(mrt["emissivity"]),
        north_degrees=float(mrt["north_degrees"]),
        ground_reflectance=float(mrt["ground_reflectance"]),
        # Engine
        intersector=str(engine["intersector"]),
        embree_quality=str(engine["embree_quality"]),
        embree_build_bvh=bool(engine["embree_build_bvh"]),
        embree_packet_size=int(engine["embree_packet_size"]),
        intersects_any=bool(engine["intersects_any"]),
        # Features
        vectorized_solar=bool(features["vectorized_solar"]),
        batch_positions=bool(features["batch_positions"]),
        # Shared
        parallel=parallel,
        performance=performance,
        # I/O
        csv_encoding=str(mrt["csv_encoding"]),
        csv_index=bool(mrt["csv_index"]),
    )

    utci_cfg = UTCIConfig(
        enable_vectorized=bool(utci["enable_vectorized"]),
        include_weather_in_results=bool(features["include_weather_in_results"]),
        include_datetime_in_results=bool(features["include_datetime_in_results"]),
        parallel=parallel,
        csv_encoding=str(utci["csv_encoding"]),
        csv_index=bool(utci["csv_index"]),
    )

    return AppConfig(
        parallel=parallel,
        performance=performance,
        mrt=mrt_cfg,
        utci=utci_cfg,
    )


# Note: No default instances - all config must come from TOML via load_config()

