import hashlib
import json
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_tiny_raw_fixture(
    tmp_path: Path,
    *,
    positions: np.ndarray | None = None,
    canonical: np.ndarray | None = None,
    utci: np.ndarray | None = None,
    shading: np.ndarray | None = None,
    surface_flags: np.ndarray | None = None,
    hours: list[int] | None = None,
    canonical_count: int | None = None,
) -> tuple[Path, Path]:
    base = tmp_path / "2025-08-15_2m_active-cells"
    if positions is None:
        positions = np.array(
            [
                [180723.5, 575888.0, 1.5],
                [180725.5, 575890.0, 1.5],
            ],
            dtype="<f4",
        )
    if canonical is None:
        canonical = np.array([2, 5], dtype="<u4")
    if utci is None:
        utci = np.array(
            [
                [32.5, np.nan],
                [40.25, 28.0],
            ],
            dtype="<f4",
        )
    if shading is None:
        shading = np.array([0.25, 0.75], dtype="<f4")
    if surface_flags is None:
        surface_flags = np.array([1, 6], dtype=np.uint8)
    if hours is None:
        hours = [10, 11]

    positions = np.asarray(positions, dtype="<f4")
    canonical = np.asarray(canonical, dtype="<u4")
    utci = np.asarray(utci, dtype="<f4")
    shading = np.asarray(shading, dtype="<f4")
    surface_flags = np.asarray(surface_flags, dtype=np.uint8)

    files = {
        "positions": base.with_suffix(".positions.f32.bin"),
        "canonicalIndices": base.with_suffix(".canonical.u32.bin"),
        "utci": base.with_suffix(".utci.f32.bin"),
        "shadingIndex": base.with_suffix(".shading.f32.bin"),
        "surfaceFlags": base.with_suffix(".surface-flags.u8.bin"),
    }
    positions.tofile(files["positions"])
    canonical.tofile(files["canonicalIndices"])
    utci.tofile(files["utci"])
    shading.tofile(files["shadingIndex"])
    surface_flags.tofile(files["surfaceFlags"])

    active_count = int(positions.shape[0])
    hour_count = len(hours)
    if canonical_count is None:
        canonical_count = max(int(canonical.max(initial=0)) + 1, active_count)
    metadata = {
        "schemaVersion": "innovation-district-raw-export/v1",
        "analysisId": "Innovation-District/2025-08-15_2m_fullday",
        "coordinateSystem": "projected-analysis",
        "canonicalCount": canonical_count,
        "activeCount": active_count,
        "hourCount": hour_count,
        "hours": hours,
        "activeMask": {
            "source": "base+road",
            "checksum": "active-mask-checksum",
            "signature": "active-mask-signature",
        },
        "timingsMs": {
            "collectorSetup": 12.5,
            "scanActiveMask": 30.0,
            "total": 1250.0,
        },
        "layout": {
            "canonicalIndices": "point-major",
            "positions": "point-major-xyz",
            "utci": "point-major-hour",
            "shadingIndex": "point-major",
            "surfaceFlags": "point-major",
        },
        "arrays": {
            "canonicalIndices": {
                "dtype": "u32",
                "endianness": "little",
                "shape": [active_count],
                "byteLength": files["canonicalIndices"].stat().st_size,
            },
            "positions": {
                "dtype": "f32",
                "endianness": "little",
                "shape": [active_count, 3],
                "byteLength": files["positions"].stat().st_size,
            },
            "utci": {
                "dtype": "f32",
                "endianness": "little",
                "shape": [active_count, hour_count],
                "byteLength": files["utci"].stat().st_size,
            },
            "shadingIndex": {
                "dtype": "f32",
                "endianness": "little",
                "shape": [active_count],
                "byteLength": files["shadingIndex"].stat().st_size,
            },
            "surfaceFlags": {
                "dtype": "u8",
                "endianness": "little",
                "shape": [active_count],
                "byteLength": files["surfaceFlags"].stat().st_size,
            },
        },
        "files": {
            key: {"fileName": path.name, "checksum": _sha256(path)}
            for key, path in files.items()
        },
    }
    metadata_path = base.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    georef_path = tmp_path / "innovation_district.georef.json"
    georef_path.write_text(
        json.dumps(
            {
                "declared_crs": "EPSG:2039",
                "earth_anchor_point": {
                    "earth_basepoint_latitude": 0.0,
                    "earth_basepoint_longitude": 0.0,
                },
                "model_bounds": {
                    "min_x": 180700.0,
                    "max_x": 180750.0,
                    "min_y": 575850.0,
                    "max_y": 575910.0,
                },
            }
        ),
        encoding="utf-8",
    )
    return metadata_path, georef_path
