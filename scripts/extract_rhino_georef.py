"""Extract georeferencing and shape inventory metadata from a Rhino .3dm file.

This script is intentionally a first-slice probe. It reads the Rhino file and
writes a JSON sidecar that makes coordinate assumptions explicit instead of
assuming a GLB export preserved GIS semantics.

Requires:
    pip install rhino3dm
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_SAMPLING_HINTS = {
    "ground",
    "terrain",
    "street",
    "road",
    "roads",
    "sidewalk",
    "sidewalks",
    "parking",
    "walkway",
    "train_tracks",
    "train_track",
}

DEFAULT_OCCLUDER_HINTS = {
    "existing_buildings",
    "existing_building",
    "buildings",
    "building",
    "new_buildings",
    "new_building",
    "trees_canopy",
    "tree_canopy",
    "trees",
    "tree",
    "vegetation",
    "new_trees",
    "new_tree",
}

DEFAULT_IGNORED_HINTS = {
    "district_outline",
    "outline",
    "trees_point",
    "tree_point",
}


def main() -> int:
    args = parse_args()

    try:
        import rhino3dm  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        print(
            "rhino3dm is required to read .3dm files. Install it with: "
            "python -m pip install rhino3dm",
            file=sys.stderr,
        )
        return 2

    model_path = args.model.resolve()
    if not model_path.exists():
        print(f"Model file not found: {model_path}", file=sys.stderr)
        return 2
    if model_path.suffix.lower() != ".3dm":
        print(f"Expected a .3dm file, received: {model_path}", file=sys.stderr)
        return 2

    model = rhino3dm.File3dm.Read(str(model_path))
    if model is None:
        print(f"Unable to read Rhino file: {model_path}", file=sys.stderr)
        return 1

    report = build_report(
        model=model,
        model_path=model_path,
        declared_crs=args.crs,
        project_id=args.project_id,
        sampling_hints=parse_name_set(args.sampling_layers) or DEFAULT_SAMPLING_HINTS,
        occluder_hints=parse_name_set(args.occluder_layers) or DEFAULT_OCCLUDER_HINTS,
        ignored_hints=parse_name_set(args.ignored_layers) or DEFAULT_IGNORED_HINTS,
    )

    output_path = args.out.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Wrote Rhino georef report: {output_path}")
    for warning in report["warnings"]:
        print(f"warning: {warning}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract georeferencing and layer/object inventory from a Rhino .3dm file."
    )
    parser.add_argument("--model", type=Path, required=True, help="Path to the source .3dm file.")
    parser.add_argument("--out", type=Path, required=True, help="Path to the output JSON sidecar.")
    parser.add_argument(
        "--project-id",
        default="Innovation-District",
        help="Project identifier to include in the report.",
    )
    parser.add_argument(
        "--crs",
        default=None,
        help="Declared source CRS, for example EPSG:2039. The script records but does not infer it.",
    )
    parser.add_argument(
        "--sampling-layers",
        default=None,
        help="Comma-separated layer-name hints for valid sampling surfaces.",
    )
    parser.add_argument(
        "--occluder-layers",
        default=None,
        help="Comma-separated layer-name hints for shade/occluder geometry.",
    )
    parser.add_argument(
        "--ignored-layers",
        default=None,
        help="Comma-separated layer-name hints for context/control layers that are not result surfaces.",
    )
    return parser.parse_args()


def build_report(
    *,
    model: Any,
    model_path: Path,
    declared_crs: str | None,
    project_id: str,
    sampling_hints: set[str],
    occluder_hints: set[str],
    ignored_hints: set[str],
) -> dict[str, Any]:
    warnings: list[str] = []
    layers = extract_layers(model)
    objects = extract_objects(model, layers)
    layer_summaries = summarize_layers(layers, objects, sampling_hints, occluder_hints, ignored_hints)
    model_bounds = merge_bounds([obj["bbox"] for obj in objects if obj.get("bbox")])
    earth_anchor = extract_earth_anchor(model)
    settings = extract_settings(model)

    if declared_crs is None:
        warnings.append("No source CRS was declared. Do not treat coordinates as GIS-ready yet.")
    if not earth_anchor.get("is_available"):
        warnings.append(
            "EarthAnchorPoint was not exposed by rhino3dm for this file/API version; run inside Rhino/RhinoCommon if georeferencing is expected."
        )
    elif not earth_anchor.get("has_usable_location"):
        warnings.append(
            "EarthAnchorPoint was present but did not expose a usable earth location/basepoint through rhino3dm."
        )
    elif earth_anchor.get("earth_location_is_set") is False:
        warnings.append("EarthAnchorPoint exists but Rhino reports earth location is not set.")
    elif earth_anchor_looks_like_zero_origin(earth_anchor):
        warnings.append(
            "EarthAnchorPoint reports latitude/longitude 0,0. Treat it as unset or placeholder until Rhino/GIS provenance confirms otherwise."
        )
    if not layers:
        warnings.append("No layer records were extracted. rhino3dm table access may be unavailable or the file has no layers.")
    if not objects:
        warnings.append("No object records were extracted. rhino3dm table access may be unavailable or the file has no objects.")
    elif not any(obj.get("bbox") for obj in objects):
        warnings.append("No object bounding boxes were extracted. Geometry bounding-box access may be unavailable through this API.")
    else:
        missing_bbox_count = sum(1 for obj in objects if not obj.get("bbox"))
        if missing_bbox_count:
            warnings.append(f"{missing_bbox_count} object(s) did not expose bounding boxes and were omitted from bounds.")
    unassigned_count = sum(1 for obj in objects if obj.get("layer_name") is None)
    if unassigned_count:
        warnings.append(f"{unassigned_count} object(s) could not be assigned to a Rhino layer.")
    if model_bounds and has_large_xy_coordinates(model_bounds):
        warnings.append(
            "Model XY bounds contain large coordinates. This may be projected CRS data; declare and validate the CRS."
        )
    if not any(layer["role_hint"] == "sampling" for layer in layer_summaries):
        warnings.append("No sampling layer hints matched layer names. Shape-aware export needs explicit layer mapping.")

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_id": project_id,
        "source_model": str(model_path),
        "declared_crs": declared_crs,
        "settings": settings,
        "earth_anchor_point": earth_anchor,
        "model_bounds": model_bounds,
        "layers": layer_summaries,
        "object_count": len(objects),
        "objects": objects,
        "shape_export_contract": {
            "status": "inventory_only",
            "canonical_grid_policy": "do_not_export_rectangular_cells_as_valid_results",
            "valid_result_policy": "only sampled/active surface points or cells may become GIS UTCI/shading results",
            "sampling_layer_hints": sorted(sampling_hints),
            "occluder_layer_hints": sorted(occluder_hints),
            "ignored_layer_hints": sorted(ignored_hints),
        },
        "warnings": warnings,
    }


def extract_settings(model: Any) -> dict[str, Any]:
    settings = getattr(model, "Settings", None)
    if settings is None:
        return {"is_available": False}

    return {
        "is_available": True,
        "model_unit_system": stringify(getattr(settings, "ModelUnitSystem", None)),
        "page_unit_system": stringify(getattr(settings, "PageUnitSystem", None)),
        "absolute_tolerance": finite_or_none(getattr(settings, "ModelAbsoluteTolerance", None)),
        "angle_tolerance_radians": finite_or_none(getattr(settings, "ModelAngleToleranceRadians", None)),
        "relative_tolerance": finite_or_none(getattr(settings, "ModelRelativeTolerance", None)),
    }


def extract_earth_anchor(model: Any) -> dict[str, Any]:
    settings = getattr(model, "Settings", None)
    anchor = getattr(settings, "EarthAnchorPoint", None) if settings is not None else None
    api_status = "settings"
    if anchor is None:
        anchor = getattr(model, "EarthAnchorPoint", None)
        api_status = "file3dm_fallback"
    if anchor is None:
        return {"is_available": False, "api_status": "not_exposed"}

    result: dict[str, Any] = {"is_available": True, "api_status": api_status}
    for output_name, candidates in {
        "earth_basepoint_latitude": ["EarthBasepointLatitude", "earthBasepointLatitude"],
        "earth_basepoint_longitude": ["EarthBasepointLongitude", "earthBasepointLongitude"],
        "earth_basepoint_elevation": ["EarthBasepointElevation", "earthBasepointElevation"],
        "model_base_point": ["ModelBasePoint", "modelBasePoint"],
        "model_north": ["ModelNorth", "modelNorth"],
        "model_east": ["ModelEast", "modelEast"],
        "name": ["Name", "name"],
        "description": ["Description", "description"],
    }.items():
        result[output_name] = get_first_attr(anchor, candidates)

    for key in ["model_base_point", "model_north", "model_east"]:
        result[key] = point_to_json(result[key])
    for key in [
        "earth_basepoint_latitude",
        "earth_basepoint_longitude",
        "earth_basepoint_elevation",
    ]:
        result[key] = finite_or_none(result[key])

    is_set = call_first(anchor, ["EarthLocationIsSet", "earthLocationIsSet"])
    if is_set is not None:
        result["earth_location_is_set"] = bool(is_set)

    result["has_usable_location"] = bool(
        result.get("earth_basepoint_latitude") is not None
        and result.get("earth_basepoint_longitude") is not None
        and result.get("model_base_point") is not None
    )

    return result


def extract_layers(model: Any) -> dict[str, dict[str, Any]]:
    layer_table = getattr(model, "Layers", None)
    layers: dict[str, dict[str, Any]] = {}
    if layer_table is None:
        return layers

    for index, layer in enumerate(iter_table(layer_table)):
        layer_index = int_or_none(getattr(layer, "Index", None))
        if layer_index is None:
            layer_index = index
        layer_id = stringify(getattr(layer, "Id", None)) or str(index)
        name = stringify(getattr(layer, "Name", None)) or f"layer_{index}"
        full_path = stringify(getattr(layer, "FullPath", None)) or name
        layers[layer_id] = {
            "id": layer_id,
            "index": layer_index,
            "name": name,
            "full_path": full_path,
            "normalized_name": normalize_name(full_path),
        }
    return layers


def extract_objects(model: Any, layers: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    object_table = getattr(model, "Objects", None)
    if object_table is None:
        return []

    layers_by_index = {layer["index"]: layer for layer in layers.values()}
    layers_by_id = {layer["id"]: layer for layer in layers.values()}
    objects: list[dict[str, Any]] = []
    for index, obj in enumerate(iter_table(object_table)):
        attributes = getattr(obj, "Attributes", None)
        geometry = getattr(obj, "Geometry", None)
        layer_index = int_or_none(getattr(attributes, "LayerIndex", None)) if attributes is not None else None
        layer_id = first_string_attr(attributes, ["LayerId", "Layer"]) if attributes is not None else None
        object_id = first_string_attr(attributes, ["Id", "ObjectId"]) if attributes is not None else None
        layer = layers_by_index.get(layer_index) if layer_index is not None else None
        if layer is None and layer_id is not None:
            layer = layers_by_id.get(layer_id)
        bbox = geometry_bbox(geometry)
        objects.append(
            {
                "index": index,
                "id": object_id,
                "name": stringify(getattr(attributes, "Name", None)) if attributes is not None else None,
                "layer_index": layer_index,
                "layer_id": layer_id,
                "layer_key": layer["id"] if layer else None,
                "layer_name": layer["full_path"] if layer else None,
                "geometry_type": type(geometry).__name__ if geometry is not None else None,
                "bbox": bbox,
            }
        )
    return objects


def summarize_layers(
    layers: dict[str, dict[str, Any]],
    objects: list[dict[str, Any]],
    sampling_hints: set[str],
    occluder_hints: set[str],
    ignored_hints: set[str],
) -> list[dict[str, Any]]:
    objects_by_layer: dict[str, list[dict[str, Any]]] = {}
    for obj in objects:
        layer_key = obj.get("layer_key")
        if layer_key:
            objects_by_layer.setdefault(layer_key, []).append(obj)

    summaries: list[dict[str, Any]] = []
    for layer_id, layer in layers.items():
        layer_objects = objects_by_layer.get(layer_id, [])
        normalized_values = layer_match_values(layer)
        role_hint = "unknown"
        if normalized_values & sampling_hints:
            role_hint = "sampling"
        elif normalized_values & occluder_hints:
            role_hint = "occluder"
        elif normalized_values & ignored_hints:
            role_hint = "ignored"

        geometry_counts: dict[str, int] = {}
        for obj in layer_objects:
            geometry_type = obj.get("geometry_type") or "unknown"
            geometry_counts[geometry_type] = geometry_counts.get(geometry_type, 0) + 1

        summaries.append(
            {
                **layer,
                "role_hint": role_hint,
                "object_count": len(layer_objects),
                "geometry_counts": geometry_counts,
                "bbox": merge_bounds([obj["bbox"] for obj in layer_objects if obj.get("bbox")]),
            }
        )

    return summaries


def layer_match_values(layer: dict[str, Any]) -> set[str]:
    names = {
        normalize_name(layer.get("name") or ""),
        normalize_name(layer.get("full_path") or ""),
        normalize_name((layer.get("full_path") or "").split("::")[-1]),
    }
    return {name for name in names if name}


def geometry_bbox(geometry: Any) -> dict[str, float] | None:
    if geometry is None:
        return None
    getter = getattr(geometry, "GetBoundingBox", None)
    if getter is None:
        return None

    try:
        bbox = getter()
    except TypeError:
        try:
            bbox = getter(True)
        except Exception:
            return None
    except Exception:
        return None

    min_point = getattr(bbox, "Min", None)
    max_point = getattr(bbox, "Max", None)
    if min_point is None or max_point is None:
        return None

    return bounds_from_points(point_to_json(min_point), point_to_json(max_point))


def bounds_from_points(
    min_point: dict[str, float] | None, max_point: dict[str, float] | None
) -> dict[str, float] | None:
    if not min_point or not max_point:
        return None
    values = {
        "min_x": min_point["x"],
        "min_y": min_point["y"],
        "min_z": min_point["z"],
        "max_x": max_point["x"],
        "max_y": max_point["y"],
        "max_z": max_point["z"],
    }
    if not all(math.isfinite(value) for value in values.values()):
        return None
    return values


def merge_bounds(bounds_list: list[dict[str, float]]) -> dict[str, float] | None:
    merged: dict[str, float] | None = None
    for bounds in bounds_list:
        if merged is None:
            merged = dict(bounds)
            continue
        merged = {
            "min_x": min(merged["min_x"], bounds["min_x"]),
            "min_y": min(merged["min_y"], bounds["min_y"]),
            "min_z": min(merged["min_z"], bounds["min_z"]),
            "max_x": max(merged["max_x"], bounds["max_x"]),
            "max_y": max(merged["max_y"], bounds["max_y"]),
            "max_z": max(merged["max_z"], bounds["max_z"]),
        }
    return merged


def point_to_json(point: Any) -> dict[str, float] | None:
    if point is None:
        return None
    try:
        x = finite_or_none(getattr(point, "X", None))
        y = finite_or_none(getattr(point, "Y", None))
        z = finite_or_none(getattr(point, "Z", None))
    except Exception:
        return None
    if x is None or y is None or z is None:
        return None
    return {"x": x, "y": y, "z": z}


def iter_table(table: Any) -> list[Any]:
    try:
        return list(table)
    except TypeError:
        count = getattr(table, "Count", None)
        if isinstance(count, int):
            return [table[index] for index in range(count)]
    return []


def get_first_attr(source: Any, names: list[str]) -> Any:
    for name in names:
        if hasattr(source, name):
            return getattr(source, name)
    return None


def call_first(source: Any, names: list[str]) -> Any:
    for name in names:
        method = getattr(source, name, None)
        if callable(method):
            try:
                return method()
            except Exception:
                return None
    return None


def stringify(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def first_string_attr(source: Any, names: list[str]) -> str | None:
    for name in names:
        value = stringify(getattr(source, name, None))
        if value is not None:
            return value
    return None


def finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def int_or_none(value: Any) -> int | None:
    number = finite_or_none(value)
    if number is None:
        return None
    return int(number)


def normalize_name(name: str) -> str:
    normalized = name.strip().lower()
    for old, new in [("::", "_"), (" ", "_"), ("-", "_"), ("(", ""), (")", "")]:
        normalized = normalized.replace(old, new)
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized


def parse_name_set(value: str | None) -> set[str]:
    if not value:
        return set()
    return {normalize_name(item) for item in value.split(",") if item.strip()}


def has_large_xy_coordinates(bounds: dict[str, float]) -> bool:
    return max(
        abs(bounds["min_x"]),
        abs(bounds["max_x"]),
        abs(bounds["min_y"]),
        abs(bounds["max_y"]),
    ) > 10000


def earth_anchor_looks_like_zero_origin(anchor: dict[str, Any]) -> bool:
    latitude = anchor.get("earth_basepoint_latitude")
    longitude = anchor.get("earth_basepoint_longitude")
    return latitude == 0.0 and longitude == 0.0


if __name__ == "__main__":
    raise SystemExit(main())
