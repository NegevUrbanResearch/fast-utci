from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import GEOPARQUET_CONTRACT
from .raw import ActiveCellArtifacts


_EXPECTED_ACTIVE_MASK_SOURCE = "base+road"
_MANIFEST_SHA_PLACEHOLDER = "0" * 64


@dataclass(frozen=True)
class PostprocessOutputs:
    cells_geoparquet_path: Path
    debug_geojson_path: Path
    manifest_path: Path
    legacy_geojson_path: Path | None = None


def metadata_base_name(metadata_path: Path) -> str:
    name = metadata_path.name
    suffix = ".metadata.json"
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return metadata_path.stem


def debug_base_name(raw_base_name: str) -> str:
    suffix = "_active-cells"
    if raw_base_name.endswith(suffix):
        return raw_base_name[: -len(suffix)]
    return raw_base_name


def bundle_name(metadata_path: Path) -> str:
    return debug_base_name(metadata_base_name(metadata_path))


def bundle_root(metadata_path: Path, out_dir: Path) -> Path:
    name = bundle_name(metadata_path)
    if out_dir.name == name:
        return out_dir
    return out_dir / name


def build_output_inventory(
    metadata_path: Path,
    out_dir: Path,
    *,
    include_legacy_geojson: bool = False,
) -> PostprocessOutputs:
    root = bundle_root(metadata_path, out_dir)
    raw_dir = root / "raw"
    qa_dir = root / "qa"
    optional_dir = root / "optional"
    for directory in (raw_dir, qa_dir, optional_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return PostprocessOutputs(
        cells_geoparquet_path=root / "cells.geoparquet",
        debug_geojson_path=qa_dir / "debug-sample.geojson",
        manifest_path=root / "manifest.json",
        legacy_geojson_path=(optional_dir / "all-hours.geojson") if include_legacy_geojson else None,
    )


def json_timing_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


def collector_timings(metadata: dict[str, Any]) -> dict[str, float | None]:
    raw_timings = metadata.get("timingsMs")
    if not isinstance(raw_timings, dict):
        return {}
    return {str(key): json_timing_value(value) for key, value in raw_timings.items()}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def _manifest_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, indent=2, allow_nan=False).encode("utf-8")


def write_manifest_json(path: Path, payload: dict[str, Any]) -> None:
    manifest = copy.deepcopy(payload)
    manifest_inventory = manifest["outputs"]["manifest"]
    manifest_inventory["path"] = str(path)
    manifest_inventory["sha256"] = _MANIFEST_SHA_PLACEHOLDER
    manifest_inventory["sha256Semantics"] = (
        "Hash of this manifest JSON with outputs.manifest.sha256 replaced by 64 zero characters."
    )

    size_bytes = 0
    while True:
        manifest_inventory["sizeBytes"] = size_bytes
        next_size = len(_manifest_bytes(manifest))
        if next_size == size_bytes:
            break
        size_bytes = next_size

    normalized_bytes = _manifest_bytes(manifest)
    manifest_inventory["sha256"] = hashlib.sha256(normalized_bytes).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_manifest_bytes(manifest))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_inventory(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sizeBytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _raw_file_path(metadata_path: Path, metadata: dict[str, Any], key: str) -> Path:
    files = metadata.get("files")
    if not isinstance(files, dict) or not isinstance(files.get(key), dict):
        raise ValueError(f"files.{key} is required")
    file_name = files[key].get("fileName")
    if not isinstance(file_name, str) or not file_name.strip():
        raise ValueError(f"files.{key}.fileName is required")
    return metadata_path.parent / file_name


def _raw_source_layout(metadata_path: Path) -> str:
    root = metadata_path.parent
    if root.name == "raw" and root.parent.name == bundle_name(metadata_path):
        return "bundle-raw"
    return "legacy-flat"


def _bundle_manifest(outputs: PostprocessOutputs, metadata_path: Path) -> dict[str, Any]:
    root = outputs.manifest_path.parent
    return {
        "name": bundle_name(metadata_path),
        "root": str(root),
        "layout": {
            "raw": str(root / "raw"),
            "cellsGeoparquet": str(outputs.cells_geoparquet_path),
            "manifest": str(outputs.manifest_path),
            "qa": str(root / "qa"),
            "optional": str(root / "optional"),
        },
    }


def _canonical_count(artifacts: ActiveCellArtifacts) -> int:
    value = artifacts.metadata.get("canonicalCount")
    if isinstance(value, int) and value >= 0:
        return value
    if artifacts.canonical_indices.size == 0:
        return int(artifacts.metadata["activeCount"])
    return max(int(np.max(artifacts.canonical_indices)) + 1, int(artifacts.metadata["activeCount"]))


def _active_ratio(active_count: int, canonical_count: int) -> float | None:
    if canonical_count <= 0:
        return None
    return active_count / canonical_count


def _raw_arrays_inventory(metadata_path: Path, artifacts: ActiveCellArtifacts) -> dict[str, Any]:
    raw_arrays: dict[str, Any] = {}
    for key in ("canonicalIndices", "positions", "utci", "shadingIndex"):
        raw_path = _raw_file_path(metadata_path, artifacts.metadata, key)
        descriptor = artifacts.metadata["arrays"][key]
        raw_arrays[key] = {
            **_file_inventory(raw_path),
            "dtype": descriptor["dtype"],
            "shape": descriptor["shape"],
            "layout": artifacts.metadata["layout"][key],
            "sourceChecksum": artifacts.metadata["files"][key]["checksum"],
        }
    return raw_arrays


def _optional_inventory(outputs: PostprocessOutputs, artifacts: ActiveCellArtifacts) -> dict[str, Any]:
    optional: dict[str, Any] = {}
    if outputs.legacy_geojson_path is not None and outputs.legacy_geojson_path.exists():
        optional["legacyAllHoursGeojson"] = {
            **_file_inventory(outputs.legacy_geojson_path),
            "rows": int(artifacts.metadata["activeCount"]),
            "note": "Opt-in legacy all-hours GeoJSON; not written by default.",
        }
    return optional


def build_manifest(
    *,
    artifacts: ActiveCellArtifacts,
    metadata_path: Path,
    georef_path: Path,
    outputs: PostprocessOutputs,
    debug_count: int,
    geoparquet_schema: list[dict[str, str]],
    timings: dict[str, float],
) -> dict[str, Any]:
    active_count = int(artifacts.metadata["activeCount"])
    canonical_count = _canonical_count(artifacts)
    copied_collector_timings = collector_timings(artifacts.metadata)
    collector_total_runtime_ms = copied_collector_timings.get("total")
    if collector_total_runtime_ms is None:
        collector_total_runtime_ms = copied_collector_timings.get("totalCollectorRuntime")

    return {
        "schemaVersion": "innovation-district-gis-postprocess-manifest/v2",
        "analysisId": artifacts.metadata["analysisId"],
        "bundle": _bundle_manifest(outputs, metadata_path),
        "crs": {
            "sourceProjected": "EPSG:2039",
            "geometry": "EPSG:4326",
            "lonLatColumns": "EPSG:4326",
        },
        "geoParquet": {
            "file": str(outputs.cells_geoparquet_path),
            **GEOPARQUET_CONTRACT.manifest_note(),
            "schema": copy.deepcopy(geoparquet_schema),
        },
        "raw": {
            "sourceLayout": _raw_source_layout(metadata_path),
            "metadata": _file_inventory(metadata_path),
            "georef": _file_inventory(georef_path),
            "hours": [int(hour) for hour in artifacts.metadata["hours"]],
            "arrays": _raw_arrays_inventory(metadata_path, artifacts),
        },
        "outputs": {
            "cellsGeoparquet": {
                **_file_inventory(outputs.cells_geoparquet_path),
                "rows": active_count,
                "schema": copy.deepcopy(geoparquet_schema),
            },
            "qaDebugSampleGeojson": {
                **_file_inventory(outputs.debug_geojson_path),
                "rows": debug_count,
                "sampleStrategy": "evenly-spaced-active-rows",
            },
            "manifest": {
                "path": str(outputs.manifest_path),
                "sizeBytes": None,
                "sha256": None,
            },
        },
        "qa": {
            "debugSampleGeojson": {
                **_file_inventory(outputs.debug_geojson_path),
                "rows": debug_count,
                "sampleStrategy": "evenly-spaced-active-rows",
            }
        },
        "optional": _optional_inventory(outputs, artifacts),
        "counts": {
            "activeRows": active_count,
            "canonicalRows": canonical_count,
            "rowCount": active_count,
            "rowCountEqualsActiveRows": True,
            "hourCount": int(artifacts.metadata["hourCount"]),
            "activeRatio": _active_ratio(active_count, canonical_count),
            "debugRows": debug_count,
        },
        "activeMask": {
            "source": artifacts.metadata["activeMask"]["source"],
            "checksum": artifacts.metadata["activeMask"]["checksum"],
            "signature": artifacts.metadata["activeMask"]["signature"],
            "expectedSource": _EXPECTED_ACTIVE_MASK_SOURCE,
            "validation": (
                "passed"
                if artifacts.metadata["activeMask"]["source"] == _EXPECTED_ACTIVE_MASK_SOURCE
                else "failed"
            ),
        },
        "collectorTimingsMs": copied_collector_timings,
        "postprocessorTimingsMs": timings,
        "totalExportRuntimeMs": (
            collector_total_runtime_ms + timings["totalPostprocessorRuntime"]
            if collector_total_runtime_ms is not None
            else None
        ),
        "timingSemantics": {
            "collectorTimingsMs": (
                "Raw collector timing payload copied from metadata.timingsMs before postprocessing."
            ),
            "postprocessorTimingsMs": (
                "Python postprocessing timings captured before the final manifest file write."
            ),
            "manifestWrite": (
                "Time to assemble the final manifest payload before the single manifest file write."
            ),
            "totalExportRuntimeMs": (
                "Collector total runtime plus the postprocessor runtime captured immediately before "
                "the final manifest file write; null when the collector total is absent."
            ),
            "totalPostprocessorRuntime": (
                "Elapsed runtime captured immediately before the final manifest file write."
            ),
        },
        "downstreamBoundary": {
            "note": (
                "Downstream consumers derive secondary map/chart artifacts from cells.geoparquet."
            )
        },
    }


def stale_flat_postprocess_paths(metadata_path: Path, out_dir: Path) -> list[Path]:
    raw_base_name = metadata_base_name(metadata_path)
    debug_name = debug_base_name(raw_base_name)
    candidates = [
        out_dir / f"{raw_base_name}.summary.json",
        out_dir / f"{debug_name}_debug-sample.geojson",
        out_dir / f"{debug_name}_manifest.json",
        out_dir / f"{raw_base_name}.geojson",
    ]
    legacy_pattern = re.compile(re.escape(debug_name) + r".*\.geojson$")
    for path in out_dir.glob("*.geojson") if out_dir.exists() else []:
        if legacy_pattern.fullmatch(path.name):
            candidates.append(path)
    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate not in seen:
            unique.append(candidate)
            seen.add(candidate)
    return unique
