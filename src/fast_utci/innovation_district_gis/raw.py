from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


_EXPECTED_ACTIVE_MASK_SOURCE = "base+road"
_EXPECTED_COORDINATE_SYSTEM = "projected-analysis"
_LAYOUTS = {
    "canonicalIndices": "point-major",
    "positions": "point-major-xyz",
    "utci": "point-major-hour",
    "shadingIndex": "point-major",
    "surfaceFlags": "point-major",
}
_DTYPES = {
    "u32": np.dtype("<u4"),
    "f32": np.dtype("<f4"),
    "u8": np.dtype("u1"),
}
SURFACE_FLAG_GROUND = 1
SURFACE_FLAG_STREET_SURFACE = 2
SURFACE_FLAG_BUILDING_FOOTPRINT = 4
_SURFACE_FLAG_ALLOWED_MASK = (
    SURFACE_FLAG_GROUND | SURFACE_FLAG_STREET_SURFACE | SURFACE_FLAG_BUILDING_FOOTPRINT
)
_SURFACE_FLAG_SAMPLED_SURFACE_MASK = SURFACE_FLAG_GROUND | SURFACE_FLAG_STREET_SURFACE


@dataclass(frozen=True)
class ActiveCellArtifacts:
    metadata_path: Path
    georef_path: Path
    metadata: dict[str, Any]
    georef: dict[str, Any]
    canonical_indices: np.ndarray
    positions: np.ndarray
    utci: np.ndarray
    shading_index: np.ndarray
    surface_flags: np.ndarray


@dataclass(frozen=True)
class SurfaceClassification:
    surface_class: np.ndarray
    is_street_surface: np.ndarray
    is_building_footprint: np.ndarray
    include_in_public_realm_stats: np.ndarray
    include_in_outdoor_surface_stats: np.ndarray


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} is required")
    return value


def _require_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _require_shape(value: Any, field: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty shape list")
    shape: list[int] = []
    for index, dimension in enumerate(value):
        if not isinstance(dimension, int) or dimension < 0:
            raise ValueError(f"{field}[{index}] must be a non-negative integer")
        shape.append(dimension)
    return tuple(shape)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_top_level_metadata(metadata: dict[str, Any]) -> tuple[int, int, list[int]]:
    _require_string(metadata.get("schemaVersion"), "schemaVersion")
    _require_string(metadata.get("analysisId"), "analysisId")
    coordinate_system = _require_string(metadata.get("coordinateSystem"), "coordinateSystem")
    if coordinate_system != _EXPECTED_COORDINATE_SYSTEM:
        raise ValueError(f'coordinateSystem must be "{_EXPECTED_COORDINATE_SYSTEM}"')

    active_count = _require_int(metadata.get("activeCount"), "activeCount")
    hour_count = _require_int(metadata.get("hourCount"), "hourCount")
    hours = metadata.get("hours")
    if not isinstance(hours, list) or len(hours) != hour_count:
        raise ValueError("hours length must match hourCount")
    parsed_hours = [_require_int(hour, f"hours[{index}]") for index, hour in enumerate(hours)]

    active_mask = metadata.get("activeMask")
    if not isinstance(active_mask, dict):
        raise ValueError("activeMask is required")
    source = _require_string(active_mask.get("source"), "activeMask.source")
    if source != _EXPECTED_ACTIVE_MASK_SOURCE:
        raise ValueError(f'activeMask.source must be "{_EXPECTED_ACTIVE_MASK_SOURCE}"')
    _require_string(active_mask.get("checksum"), "activeMask.checksum")
    _require_string(active_mask.get("signature"), "activeMask.signature")

    layouts = metadata.get("layout")
    if not isinstance(layouts, dict):
        raise ValueError("layout is required")
    for key, expected in _LAYOUTS.items():
        if layouts.get(key) != expected:
            raise ValueError(f'layout.{key} must be "{expected}"')

    return active_count, hour_count, parsed_hours


def _validate_georef(georef: dict[str, Any]) -> None:
    declared_crs = _require_string(georef.get("declared_crs"), "georef.declared_crs")
    if declared_crs != "EPSG:2039":
        raise ValueError('georef.declared_crs must be "EPSG:2039"')


def _load_array(
    metadata_path: Path,
    metadata: dict[str, Any],
    key: str,
    expected_dtype_name: str,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    arrays = metadata.get("arrays")
    files = metadata.get("files")
    if not isinstance(arrays, dict):
        raise ValueError("arrays is required")
    if not isinstance(files, dict):
        raise ValueError("files is required")

    descriptor = arrays.get(key)
    file_descriptor = files.get(key)
    if not isinstance(descriptor, dict):
        raise ValueError(f"arrays.{key} is required")
    if not isinstance(file_descriptor, dict):
        raise ValueError(f"files.{key} is required")

    dtype_name = _require_string(descriptor.get("dtype"), f"arrays.{key}.dtype")
    if dtype_name != expected_dtype_name:
        raise ValueError(f"arrays.{key}.dtype must be {expected_dtype_name}")
    if descriptor.get("endianness") != "little":
        raise ValueError(f"arrays.{key}.endianness must be little")

    dtype = _DTYPES.get(dtype_name)
    if dtype is None:
        raise ValueError(f"arrays.{key}.dtype is unsupported")

    shape = _require_shape(descriptor.get("shape"), f"arrays.{key}.shape")
    if shape != expected_shape:
        raise ValueError(f"arrays.{key}.shape must be {list(expected_shape)}")

    expected_byte_length = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    byte_length = _require_int(descriptor.get("byteLength"), f"arrays.{key}.byteLength")
    if byte_length != expected_byte_length:
        raise ValueError(
            f"arrays.{key}.byteLength {byte_length} does not match expected {expected_byte_length}"
        )

    file_name = _require_string(file_descriptor.get("fileName"), f"files.{key}.fileName")
    file_path = metadata_path.parent / file_name
    if not file_path.exists():
        raise ValueError(f"files.{key}.fileName does not exist: {file_path}")
    actual_byte_length = file_path.stat().st_size
    if actual_byte_length != byte_length:
        raise ValueError(
            f"{file_path.name} byteLength {actual_byte_length} does not match metadata {byte_length}"
        )

    expected_checksum = _require_string(file_descriptor.get("checksum"), f"files.{key}.checksum")
    actual_checksum = _sha256_file(file_path)
    if actual_checksum != expected_checksum:
        raise ValueError(f"files.{key}.checksum does not match {file_path.name}")

    array = np.fromfile(file_path, dtype=dtype)
    if array.dtype != dtype:
        raise ValueError(f"{key} dtype did not load as {dtype}")
    if array.nbytes != byte_length:
        raise ValueError(f"{key} loaded byteLength {array.nbytes} does not match metadata")
    if array.size != int(np.prod(shape, dtype=np.int64)):
        raise ValueError(f"{key} loaded element count does not match metadata shape")
    return array.reshape(shape)


def _validate_positions_are_projected_epsg2039(positions: np.ndarray) -> None:
    if not np.all(np.isfinite(positions)):
        raise ValueError("positions must contain only finite projected EPSG:2039 coordinates")
    xs = positions[:, 0]
    ys = positions[:, 1]
    if np.any((xs < 100_000) | (xs > 300_000) | (ys < 300_000) | (ys > 800_000)):
        raise ValueError("positions are outside the expected projected EPSG:2039 Israel range")


def classify_surface_flags(
    surface_flags: np.ndarray,
    *,
    validate: bool = False,
) -> SurfaceClassification:
    flags = np.asarray(surface_flags)
    if validate:
        if flags.dtype != np.dtype("u1"):
            raise ValueError("surface_flags must load as uint8")
        if flags.ndim != 1:
            raise ValueError("surface_flags must be a 1D point-major array")

        unsupported_mask = (flags & np.uint8(~_SURFACE_FLAG_ALLOWED_MASK & 0xFF)) != 0
        if np.any(unsupported_mask):
            unsupported_values = sorted({int(value) for value in flags[unsupported_mask].tolist()})
            raise ValueError(
                "surface_flags contains unsupported values outside "
                "ground|street_surface|building_footprint bitflags: "
                f"{unsupported_values}"
            )

        missing_sampled_surface = (flags & np.uint8(_SURFACE_FLAG_SAMPLED_SURFACE_MASK)) == 0
        if np.any(missing_sampled_surface):
            row_indices = np.flatnonzero(missing_sampled_surface)[:5].tolist()
            raise ValueError(
                "surface_flags rows must include sampled-surface provenance "
                "(ground or street_surface); unknown is not a legal active-row class. "
                f"Invalid active rows: {row_indices}"
            )

    is_street_surface = (flags & np.uint8(SURFACE_FLAG_STREET_SURFACE)) != 0
    is_building_footprint = (flags & np.uint8(SURFACE_FLAG_BUILDING_FOOTPRINT)) != 0
    surface_class = np.where(
        is_building_footprint,
        "building_footprint",
        np.where(is_street_surface, "street_surface", "ground"),
    )
    include_in_public_realm_stats = is_street_surface & ~is_building_footprint
    include_in_outdoor_surface_stats = ~is_building_footprint
    return SurfaceClassification(
        surface_class=np.asarray(surface_class),
        is_street_surface=np.asarray(is_street_surface),
        is_building_footprint=np.asarray(is_building_footprint),
        include_in_public_realm_stats=np.asarray(include_in_public_realm_stats),
        include_in_outdoor_surface_stats=np.asarray(include_in_outdoor_surface_stats),
    )


def load_active_cell_artifacts(metadata_path: str | Path, georef_path: str | Path) -> ActiveCellArtifacts:
    metadata_path = Path(metadata_path)
    georef_path = Path(georef_path)
    metadata = _read_json(metadata_path)
    georef = _read_json(georef_path)
    active_count, hour_count, _hours = _validate_top_level_metadata(metadata)
    _validate_georef(georef)

    canonical_indices = _load_array(
        metadata_path, metadata, "canonicalIndices", "u32", (active_count,)
    )
    positions = _load_array(metadata_path, metadata, "positions", "f32", (active_count, 3))
    utci = _load_array(metadata_path, metadata, "utci", "f32", (active_count, hour_count))
    shading_index = _load_array(metadata_path, metadata, "shadingIndex", "f32", (active_count,))
    surface_flags = _load_array(metadata_path, metadata, "surfaceFlags", "u8", (active_count,))

    _validate_positions_are_projected_epsg2039(positions)
    if not np.all(np.isfinite(canonical_indices)):
        raise ValueError("canonicalIndices must contain finite values")
    classify_surface_flags(surface_flags, validate=True)

    return ActiveCellArtifacts(
        metadata_path=metadata_path,
        georef_path=georef_path,
        metadata=metadata,
        georef=georef,
        canonical_indices=canonical_indices,
        positions=positions,
        utci=utci,
        shading_index=shading_index,
        surface_flags=surface_flags,
    )
