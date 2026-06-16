from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyproj import Transformer

from .raw import ActiveCellArtifacts


_EPSG2039_TO_WGS84 = Transformer.from_crs("EPSG:2039", "EPSG:4326", always_xy=True)


@dataclass(frozen=True)
class DerivedTables:
    lon: np.ndarray
    lat: np.ndarray


def epsg2039_to_wgs84(x: float, y: float) -> tuple[float, float]:
    return _EPSG2039_TO_WGS84.transform(x, y)


def transform_positions_to_wgs84(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lon, lat = _EPSG2039_TO_WGS84.transform(positions[:, 0], positions[:, 1])
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    if not np.all(np.isfinite(lon)) or not np.all(np.isfinite(lat)):
        raise ValueError("EPSG:2039 to WGS84 transformation produced non-finite coordinates")
    return lon, lat


def build_derived_tables(artifacts: ActiveCellArtifacts) -> DerivedTables:
    lon, lat = transform_positions_to_wgs84(artifacts.positions)
    return DerivedTables(lon=lon, lat=lat)
