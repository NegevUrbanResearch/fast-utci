from __future__ import annotations

import time
from pathlib import Path

from .contracts import LEGACY_ALL_HOURS_GEOJSON_DEFAULT_MAX_ROWS
from .manifest import PostprocessOutputs, build_manifest, build_output_inventory, write_manifest_json
from .geojson_outputs import write_debug_sample_geojson, write_legacy_all_hours_geojson
from .parquet_outputs import read_geoparquet_schema, write_cells_geoparquet
from .raw import load_active_cell_artifacts
from .transforms import build_derived_tables


def postprocess_active_cells(
    metadata_path: str | Path,
    georef_path: str | Path,
    out_dir: str | Path,
    debug_geojson_limit: int = 5000,
    include_legacy_geojson: bool = False,
    legacy_geojson_max_rows: int = LEGACY_ALL_HOURS_GEOJSON_DEFAULT_MAX_ROWS,
    force_legacy_geojson: bool = False,
) -> PostprocessOutputs:
    total_start = time.perf_counter()
    timings: dict[str, float] = {}

    metadata_path = Path(metadata_path)
    georef_path = Path(georef_path)
    out_dir = Path(out_dir)
    outputs = build_output_inventory(
        metadata_path,
        out_dir,
        include_legacy_geojson=include_legacy_geojson,
    )

    start = time.perf_counter()
    artifacts = load_active_cell_artifacts(metadata_path, georef_path)
    timings["binaryLoadValidation"] = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    tables = build_derived_tables(artifacts)
    timings["epsg2039ToWgs84"] = (time.perf_counter() - start) * 1000.0

    debug_name = outputs.debug_geojson_path.stem.removesuffix("_debug-sample")
    if outputs.legacy_geojson_path is not None:
        raw_base_name = metadata_path.name.removesuffix(".metadata.json")
        start = time.perf_counter()
        write_legacy_all_hours_geojson(
            outputs.legacy_geojson_path,
            raw_base_name,
            artifacts,
            tables,
            max_rows=legacy_geojson_max_rows,
            force=force_legacy_geojson,
        )
        timings["legacyGeojsonWrite"] = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    write_cells_geoparquet(outputs.cells_geoparquet_path, artifacts, tables)
    timings["cellsGeoparquetWrite"] = (time.perf_counter() - start) * 1000.0
    geoparquet_schema = read_geoparquet_schema(outputs.cells_geoparquet_path)

    start = time.perf_counter()
    debug_count = write_debug_sample_geojson(
        outputs.debug_geojson_path,
        f"{debug_name}_debug-sample",
        artifacts,
        tables,
        limit=int(debug_geojson_limit),
    )
    timings["debugSampleWrite"] = (time.perf_counter() - start) * 1000.0

    manifest_start = time.perf_counter()
    timings["manifestWrite"] = 0.0
    timings["totalPostprocessorRuntime"] = (manifest_start - total_start) * 1000.0
    manifest = build_manifest(
        artifacts=artifacts,
        metadata_path=metadata_path,
        georef_path=georef_path,
        outputs=outputs,
        debug_count=debug_count,
        geoparquet_schema=geoparquet_schema,
        timings=timings,
    )

    timings["manifestWrite"] = (time.perf_counter() - manifest_start) * 1000.0
    write_manifest_json(outputs.manifest_path, manifest)

    return outputs
