from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .contracts import LEGACY_ALL_HOURS_GEOJSON_DEFAULT_MAX_ROWS
from .raw import ActiveCellArtifacts
from .summary import json_metric_value
from .transforms import DerivedTables


def _feature_for_active_row(
    artifacts: ActiveCellArtifacts,
    active_index: int,
    lon: float,
    lat: float,
) -> dict[str, Any]:
    hours = artifacts.metadata["hours"]
    x, y, z = artifacts.positions[active_index]
    utci_values = artifacts.utci[active_index]
    utci_by_hour = {
        str(hour): json_metric_value(utci_values[hour_index])
        for hour_index, hour in enumerate(hours)
    }
    return {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [float(lon), float(lat)],
        },
        "properties": {
            "active_index": int(active_index),
            "canonical_index": int(artifacts.canonical_indices[active_index]),
            "projected_x": float(x),
            "projected_y": float(y),
            "projected_z": float(z),
            "shading_index": json_metric_value(artifacts.shading_index[active_index]),
            "utci_by_hour": utci_by_hour,
            **{
                f"utci_{hour:02d}": json_metric_value(utci_values[hour_index])
                for hour_index, hour in enumerate(hours)
            },
        },
    }


def sample_active_row_indices(active_count: int, limit: int) -> list[int]:
    row_count = max(0, min(int(limit), active_count))
    if row_count <= 0:
        return []
    if row_count >= active_count:
        return list(range(active_count))
    if row_count == 1:
        return [0]
    return [int(index) for index in np.linspace(0, active_count - 1, num=row_count, dtype=np.int64)]


def write_geojson_stream(
    path: Path,
    name: str,
    artifacts: ActiveCellArtifacts,
    tables: DerivedTables,
    row_indices: Sequence[int] | None = None,
    properties: dict[str, Any] | None = None,
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    active_count = len(artifacts.canonical_indices)
    indices = list(range(active_count)) if row_indices is None else [int(index) for index in row_indices]
    with path.open("w", encoding="utf-8") as file:
        file.write('{"type":"FeatureCollection","name":')
        json.dump(name, file, allow_nan=False)
        file.write(',"crs":{"type":"name","properties":{"name":"EPSG:4326"}}')
        if properties is not None:
            file.write(',"properties":')
            json.dump(properties, file, allow_nan=False, separators=(",", ":"))
        file.write(',"features":[')
        for output_index, active_index in enumerate(indices):
            if output_index:
                file.write(",")
            json.dump(
                _feature_for_active_row(
                    artifacts,
                    active_index,
                    tables.lon[active_index],
                    tables.lat[active_index],
                ),
                file,
                allow_nan=False,
                separators=(",", ":"),
            )
        file.write("]}")
    return len(indices)


def write_legacy_all_hours_geojson(
    path: Path,
    name: str,
    artifacts: ActiveCellArtifacts,
    tables: DerivedTables,
    *,
    max_rows: int = LEGACY_ALL_HOURS_GEOJSON_DEFAULT_MAX_ROWS,
    force: bool = False,
) -> int:
    active_count = len(artifacts.canonical_indices)
    if active_count > max_rows and not force:
        raise ValueError(
            "Legacy all-hours GeoJSON would write "
            f"{active_count} rows, exceeding the max legacy row guard of {max_rows}. "
            "Pass force_legacy_geojson=True only for intentional legacy exports."
        )
    return write_geojson_stream(path, name, artifacts, tables)


def write_debug_sample_geojson(
    path: Path,
    name: str,
    artifacts: ActiveCellArtifacts,
    tables: DerivedTables,
    limit: int,
) -> int:
    active_count = len(artifacts.canonical_indices)
    indices = sample_active_row_indices(active_count, limit)
    return write_geojson_stream(
        path,
        name,
        artifacts,
        tables,
        row_indices=indices,
        properties={
            "sampleStrategy": "evenly-spaced-active-rows",
            "sourceRowCount": active_count,
            "featureCount": len(indices),
        },
    )
