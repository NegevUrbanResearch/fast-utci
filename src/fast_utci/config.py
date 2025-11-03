"""
Unified application configuration for fast-utci.

Parses TOML configuration (single source of truth) and constructs
domain configs used across MRT and UTCI modules. No environment
variable fallback is used here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import tomllib

from fast_utci.shared.config import ParallelConfig, PerformanceConfig
from fast_utci.mrt.config import MRTConfig
from fast_utci.utci.config import UTCIConfig


@dataclass
class AppConfig:
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


