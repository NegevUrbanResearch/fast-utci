from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .contracts import GEOPARQUET_CONTRACT
from .raw import ActiveCellArtifacts
from .transforms import DerivedTables


def _wkb_point(lon: float, lat: float) -> bytes:
    return struct.pack("<BIdd", 1, 1, float(lon), float(lat))


def _nullable_float32(values: np.ndarray) -> pa.Array:
    values = np.asarray(values, dtype=np.float32)
    return pa.array(values, mask=~np.isfinite(values), type=pa.float32())


def _utci_columns(artifacts: ActiveCellArtifacts) -> dict[str, pa.Array]:
    hours = [int(hour) for hour in artifacts.metadata["hours"]]
    hour_to_index = {hour: index for index, hour in enumerate(hours)}
    columns: dict[str, pa.Array] = {}
    active_count = int(artifacts.metadata["activeCount"])
    for hour in range(24):
        source_index = hour_to_index.get(hour)
        if source_index is None:
            columns[f"utci_{hour:02d}"] = pa.nulls(active_count, type=pa.float32())
        else:
            columns[f"utci_{hour:02d}"] = _nullable_float32(artifacts.utci[:, source_index])
    return columns


def _geo_metadata_bytes() -> bytes:
    return json.dumps(
        GEOPARQUET_CONTRACT.geo_metadata(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def write_cells_geoparquet(
    path: Path,
    artifacts: ActiveCellArtifacts,
    tables: DerivedTables,
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    active_count = int(artifacts.metadata["activeCount"])
    geometry = [_wkb_point(lon, lat) for lon, lat in zip(tables.lon, tables.lat, strict=True)]
    table = pa.table(
        {
            "active_index": pa.array(range(active_count), type=pa.int64()),
            "canonical_index": pa.array(artifacts.canonical_indices, type=pa.uint32()),
            "geometry": pa.array(geometry, type=pa.binary()),
            "lon": pa.array(tables.lon, type=pa.float64()),
            "lat": pa.array(tables.lat, type=pa.float64()),
            "x": pa.array(artifacts.positions[:, 0], type=pa.float32()),
            "y": pa.array(artifacts.positions[:, 1], type=pa.float32()),
            "z": pa.array(artifacts.positions[:, 2], type=pa.float32()),
            "shading_index": _nullable_float32(artifacts.shading_index),
            **_utci_columns(artifacts),
        }
    )
    table = table.replace_schema_metadata({**(table.schema.metadata or {}), b"geo": _geo_metadata_bytes()})
    pq.write_table(table, path)
    return active_count


def read_geoparquet_schema(path: Path) -> list[dict[str, str]]:
    schema = pq.read_schema(path)
    return [{"name": field.name, "type": str(field.type)} for field in schema]
