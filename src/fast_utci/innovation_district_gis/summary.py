from __future__ import annotations

import math
from typing import Any

import numpy as np

from .raw import ActiveCellArtifacts


_HOT_AREA_UTCI_THRESHOLD_C = 32.0


def json_metric_value(value: float) -> float | None:
    if math.isfinite(value):
        return float(value)
    return None


def _histogram(values: np.ndarray, bins: tuple[float, ...]) -> dict[str, int]:
    histogram: dict[str, int] = {}
    for start, end in zip(bins[:-1], bins[1:]):
        key = f"{start:g}..{end:g}"
        histogram[key] = int(np.count_nonzero((values >= start) & (values < end)))
    histogram[f">={bins[-1]:g}"] = int(np.count_nonzero(values >= bins[-1]))
    return histogram


def _metric_summary(values: np.ndarray, bins: tuple[float, ...] | None = None) -> dict[str, Any]:
    flat = values.reshape(-1)
    valid = flat[np.isfinite(flat)]
    summary: dict[str, Any] = {
        "valueCount": int(flat.size),
        "validCount": int(valid.size),
        "invalidCount": int(flat.size - valid.size),
    }
    if valid.size:
        summary.update(
            {
                "min": float(np.min(valid)),
                "max": float(np.max(valid)),
                "mean": float(np.mean(valid)),
            }
        )
    else:
        summary.update({"min": None, "max": None, "mean": None})
    if bins is not None:
        summary["histogram"] = _histogram(valid, bins)
    return summary


def _comfort_breakdown(valid_utci: np.ndarray) -> dict[str, int]:
    return {
        "no_heat_stress": int(np.count_nonzero(valid_utci < 26)),
        "moderate_heat_stress": int(np.count_nonzero((valid_utci >= 26) & (valid_utci < 32))),
        "strong_heat_stress": int(np.count_nonzero((valid_utci >= 32) & (valid_utci < 38))),
        "very_strong_heat_stress": int(np.count_nonzero((valid_utci >= 38) & (valid_utci < 46))),
        "extreme_heat_stress": int(np.count_nonzero(valid_utci >= 46)),
    }


def _utci_summary(values: np.ndarray) -> dict[str, Any]:
    valid = values.reshape(-1)
    valid = valid[np.isfinite(valid)]
    summary = _metric_summary(values, bins=(20, 26, 32, 38, 46))
    summary["comfortBreakdown"] = _comfort_breakdown(valid)
    summary["hotAreaShareUtciGe32"] = (
        float(np.mean(valid >= _HOT_AREA_UTCI_THRESHOLD_C)) if valid.size else None
    )
    return summary


def _per_hour_utci_summary(artifacts: ActiveCellArtifacts) -> list[dict[str, Any]]:
    per_hour: list[dict[str, Any]] = []
    for hour_index, hour in enumerate(artifacts.metadata["hours"]):
        hour_summary = _utci_summary(artifacts.utci[:, hour_index])
        hour_summary["hour"] = int(hour)
        per_hour.append(hour_summary)
    return per_hour


def build_summary(artifacts: ActiveCellArtifacts) -> dict[str, Any]:
    utci_summary = _utci_summary(artifacts.utci)
    utci_summary["perHour"] = _per_hour_utci_summary(artifacts)

    shading_summary = _metric_summary(artifacts.shading_index, bins=(0, 0.25, 0.5, 0.75, 1.0))
    shading_valid = artifacts.shading_index[np.isfinite(artifacts.shading_index)]
    shading_summary["percentShaded"] = (
        float(np.mean(shading_valid) * 100.0) if shading_valid.size else None
    )

    metadata = artifacts.metadata
    return {
        "schemaVersion": "innovation-district-gis-summary/v1",
        "analysisId": metadata["analysisId"],
        "coordinateSystem": {
            "input": "EPSG:2039",
            "output": "EPSG:4326",
        },
        "counts": {
            "activeRows": int(metadata["activeCount"]),
            "hourCount": int(metadata["hourCount"]),
        },
        "hours": metadata["hours"],
        "activeMask": metadata["activeMask"],
        "sourceFiles": metadata.get("files", {}),
        "utci": utci_summary,
        "shadingIndex": shading_summary,
    }
