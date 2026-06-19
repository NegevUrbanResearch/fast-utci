import json
import struct
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyproj")
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from fast_utci.innovation_district_gis.orchestrator import postprocess_active_cells
from test_innovation_district_gis_fixtures import write_tiny_raw_fixture


def _geo_metadata(table) -> dict:
    metadata = table.schema.metadata or {}
    assert b"geo" in metadata
    return json.loads(metadata[b"geo"].decode("utf-8"))


def _wkb_point_lon_lat(value: bytes) -> tuple[float, float]:
    byte_order, geometry_type, lon, lat = struct.unpack("<BIdd", value)
    assert byte_order == 1
    assert geometry_type == 1
    return lon, lat


def test_postprocess_active_cells_writes_combined_geoparquet_manifest_and_debug_sample(
    tmp_path,
):
    hours = list(range(24))
    utci = np.arange(72, dtype="<f4").reshape(3, 24) + np.float32(20.0)
    utci[0, 23] = np.nan
    metadata_path, georef_path = write_tiny_raw_fixture(
        tmp_path,
        positions=np.array(
            [
                [180723.5, 575888.0, 1.5],
                [180725.5, 575890.0, 1.5],
                [180727.5, 575892.0, 1.5],
            ],
            dtype="<f4",
        ),
        canonical=np.array([2, 5, 8], dtype="<u4"),
        canonical_count=12,
        utci=utci,
        shading=np.array([0.25, 0.75, 0.5], dtype="<f4"),
        surface_flags=np.array([1, 6, 5], dtype=np.uint8),
        hours=hours,
    )
    out_dir = tmp_path / "out"

    outputs = postprocess_active_cells(
        metadata_path=metadata_path,
        georef_path=georef_path,
        out_dir=out_dir,
        debug_geojson_limit=2,
    )

    debug_geojson = json.loads(outputs.debug_geojson_path.read_text(encoding="utf-8"))
    manifest = json.loads(outputs.manifest_path.read_text(encoding="utf-8"))

    bundle_dir = out_dir / "2025-08-15_2m"
    assert outputs.cells_geoparquet_path == bundle_dir / "cells.geoparquet"
    assert outputs.debug_geojson_path == bundle_dir / "qa" / "debug-sample.geojson"
    assert outputs.manifest_path == bundle_dir / "manifest.json"
    assert (bundle_dir / "raw").is_dir()
    assert not (bundle_dir / "derived").exists()
    assert not list(bundle_dir.glob("*.geojson"))

    table = pq.read_table(outputs.cells_geoparquet_path)
    assert table.num_rows == 3
    assert table.column_names == [
        "active_index",
        "canonical_index",
        "geometry",
        "lon",
        "lat",
        "x",
        "y",
        "z",
        "surface_flags",
        "surface_class",
        "is_street_surface",
        "is_building_footprint",
        "include_in_public_realm_stats",
        "include_in_outdoor_surface_stats",
        "shading_index",
        *[f"utci_{hour:02d}" for hour in hours],
    ]
    assert table.schema.field("geometry").type == pa.binary()
    assert table.schema.field("surface_flags").type == pa.uint8()
    assert table.schema.field("surface_class").type == pa.string()
    assert table.schema.field("is_street_surface").type == pa.bool_()
    assert table.schema.field("is_building_footprint").type == pa.bool_()
    assert table.schema.field("include_in_public_realm_stats").type == pa.bool_()
    assert table.schema.field("include_in_outdoor_surface_stats").type == pa.bool_()

    geo = _geo_metadata(table)
    assert geo["primary_column"] == "geometry"
    assert geo["columns"]["geometry"]["encoding"] == "WKB"
    assert geo["columns"]["geometry"]["geometry_types"] == ["Point"]
    assert geo["columns"]["geometry"]["crs"]["id"]["authority"] == "EPSG"
    assert geo["columns"]["geometry"]["crs"]["id"]["code"] == 4326

    payload = table.to_pydict()
    assert payload["active_index"] == [0, 1, 2]
    assert payload["canonical_index"] == [2, 5, 8]
    assert payload["canonical_index"] != list(range(12))
    assert payload["x"] == pytest.approx([180723.5, 180725.5, 180727.5])
    assert payload["y"] == pytest.approx([575888.0, 575890.0, 575892.0])
    assert payload["z"] == pytest.approx([1.5, 1.5, 1.5])
    assert payload["surface_flags"] == [1, 6, 5]
    assert payload["surface_class"] == [
        "ground",
        "building_footprint",
        "building_footprint",
    ]
    assert payload["is_street_surface"] == [False, True, False]
    assert payload["is_building_footprint"] == [False, True, True]
    assert payload["include_in_public_realm_stats"] == [False, False, False]
    assert payload["include_in_outdoor_surface_stats"] == [True, False, False]
    assert payload["shading_index"] == pytest.approx([0.25, 0.75, 0.5])
    assert payload["utci_00"] == pytest.approx([20.0, 44.0, 68.0])
    assert payload["utci_22"] == pytest.approx([42.0, 66.0, 90.0])
    assert payload["utci_23"][0] is None
    assert payload["utci_23"][1:] == pytest.approx([67.0, 91.0])

    first_lon, first_lat = _wkb_point_lon_lat(payload["geometry"][0])
    assert first_lon == pytest.approx(payload["lon"][0])
    assert first_lat == pytest.approx(payload["lat"][0])
    assert 34.7 < first_lon < 34.9
    assert 31.1 < first_lat < 31.4

    assert [f["properties"]["active_index"] for f in debug_geojson["features"]] == [0, 2]
    assert debug_geojson["properties"]["sampleStrategy"] == "evenly-spaced-active-rows"
    assert debug_geojson["properties"]["sourceRowCount"] == 3
    assert debug_geojson["properties"]["featureCount"] == 2
    feature = debug_geojson["features"][0]
    assert feature["geometry"]["type"] == "Point"
    assert feature["properties"]["canonical_index"] == 2
    assert feature["properties"]["surface_flags"] == 1
    assert feature["properties"]["surface_class"] == "ground"
    assert feature["properties"]["is_street_surface"] is False
    assert feature["properties"]["is_building_footprint"] is False
    assert feature["properties"]["include_in_public_realm_stats"] is False
    assert feature["properties"]["include_in_outdoor_surface_stats"] is True
    assert feature["properties"]["utci_by_hour"]["0"] == pytest.approx(20.0)
    assert feature["properties"]["utci_by_hour"]["23"] is None
    assert feature["properties"]["shading_index"] == pytest.approx(0.25)

    assert manifest["bundle"]["name"] == "2025-08-15_2m"
    assert manifest["bundle"]["layout"] == {
        "raw": str(bundle_dir / "raw"),
        "cellsGeoparquet": str(bundle_dir / "cells.geoparquet"),
        "manifest": str(bundle_dir / "manifest.json"),
        "qa": str(bundle_dir / "qa"),
        "optional": str(bundle_dir / "optional"),
    }
    assert manifest["raw"]["sourceLayout"] == "legacy-flat"
    assert manifest["raw"]["metadata"]["path"] == str(metadata_path)
    assert manifest["raw"]["hours"] == hours
    assert manifest["raw"]["arrays"]["utci"]["path"].endswith(
        "2025-08-15_2m_active-cells.utci.f32.bin"
    )
    assert manifest["raw"]["arrays"]["surfaceFlags"]["path"].endswith(
        "2025-08-15_2m_active-cells.surface-flags.u8.bin"
    )
    assert manifest["raw"]["arrays"]["surfaceFlags"]["dtype"] == "u8"
    assert manifest["raw"]["arrays"]["surfaceFlags"]["shape"] == [3]
    assert manifest["raw"]["arrays"]["surfaceFlags"]["layout"] == "point-major"
    assert manifest["crs"] == {
        "sourceProjected": "EPSG:2039",
        "geometry": "EPSG:4326",
        "lonLatColumns": "EPSG:4326",
    }
    assert manifest["geoParquet"]["file"] == str(outputs.cells_geoparquet_path)
    assert manifest["geoParquet"]["primaryColumn"] == "geometry"
    assert manifest["geoParquet"]["encoding"] == "WKB"
    assert manifest["geoParquet"]["geometryTypes"] == ["Point"]
    assert "GeoParquet metadata" in manifest["geoParquet"]["note"]
    assert manifest["geoParquet"]["schema"] == manifest["outputs"]["cellsGeoparquet"]["schema"]
    assert manifest["surfaceClassification"] == {
        "rawField": "surfaceFlags",
        "pythonField": "surface_flags",
        "bitFlags": {
            "ground": 1,
            "street_surface": 2,
            "building_footprint": 4,
        },
        "semanticNotes": {
            "streetSurfaceFamily": (
                "Street, road, and sidewalk-family sampled surfaces are grouped as street_surface."
            ),
            "buildingFootprintOverlay": (
                "Building footprint is an exclusion flag for downstream maps and stats."
            ),
            "outdoorSurfaceStats": (
                "include_in_outdoor_surface_stats includes active rows excluding building footprints."
            ),
            "publicRealmStats": (
                "include_in_public_realm_stats includes street_surface rows excluding "
                "building footprints."
            ),
            "classifiedActiveRows": (
                "Classified active export rows always have sampled-surface provenance; "
                "unknown is not a legal active-row class in this contract."
            ),
        },
    }
    assert manifest["outputs"]["cellsGeoparquet"]["path"] == str(outputs.cells_geoparquet_path)
    assert manifest["outputs"]["cellsGeoparquet"]["rows"] == 3
    assert manifest["outputs"]["manifest"]["path"] == str(outputs.manifest_path)
    assert manifest["outputs"]["qaDebugSampleGeojson"]["path"] == str(outputs.debug_geojson_path)
    assert manifest["qa"]["debugSampleGeojson"]["rows"] == 2
    assert manifest["qa"]["spatialCoverageExpectations"] == [
        {
            "key": "street_surface_family_sample",
            "expectation": (
                "If source geometry makes it available, spatial QA should include at least one "
                "street, road, or sidewalk-family sample."
            ),
        },
        {
            "key": "building_footprint_overlap_sample",
            "expectation": (
                "If source geometry makes it available, spatial QA should include at least one "
                "building-footprint overlap sample."
            ),
        },
        {
            "key": "building_only_non_active_location",
            "expectation": (
                "If source geometry makes it available, spatial QA should include at least one "
                "building-only non-active location proving no classified export row was created."
            ),
        },
    ]
    assert manifest["counts"]["activeRows"] == 3
    assert manifest["counts"]["canonicalRows"] == 12
    assert manifest["counts"]["rowCount"] == 3
    assert manifest["counts"]["rowCountEqualsActiveRows"] is True
    assert manifest["counts"]["hourCount"] == 24
    assert manifest["counts"]["activeRatio"] == pytest.approx(0.25)
    assert manifest["counts"]["streetSurfaceRows"] == 1
    assert manifest["counts"]["buildingFootprintRows"] == 2
    assert manifest["counts"]["publicRealmRows"] == 0
    assert manifest["counts"]["outdoorSurfaceRows"] == 1
    assert manifest["activeMask"] == {
        "source": "base+road",
        "checksum": "active-mask-checksum",
        "signature": "active-mask-signature",
        "expectedSource": "base+road",
        "validation": "passed",
    }
    assert manifest["downstreamBoundary"]["note"] == (
        "Downstream consumers derive secondary map/chart artifacts from cells.geoparquet."
    )
    assert "dashboardBoundary" not in manifest
    assert "geojson" not in manifest.get("derived", {})
    assert outputs.legacy_geojson_path is None
    assert manifest["collectorTimingsMs"] == {
        "collectorSetup": pytest.approx(12.5),
        "scanActiveMask": pytest.approx(30.0),
        "total": pytest.approx(1250.0),
    }
    assert manifest["postprocessorTimingsMs"]["binaryLoadValidation"] >= 0
    assert manifest["postprocessorTimingsMs"]["epsg2039ToWgs84"] >= 0
    assert manifest["postprocessorTimingsMs"]["cellsGeoparquetWrite"] >= 0
    assert manifest["postprocessorTimingsMs"]["debugSampleWrite"] >= 0
    assert manifest["postprocessorTimingsMs"]["manifestWrite"] >= 0
    assert manifest["postprocessorTimingsMs"]["totalPostprocessorRuntime"] >= 0
    assert "summaryWrite" not in manifest["postprocessorTimingsMs"]
    assert "finalGeojsonWrite" not in manifest["postprocessorTimingsMs"]
    assert manifest["totalExportRuntimeMs"] == pytest.approx(
        manifest["collectorTimingsMs"]["total"]
        + manifest["postprocessorTimingsMs"]["totalPostprocessorRuntime"]
    )


def test_postprocess_active_cells_writes_null_float_columns_for_absent_hours(tmp_path):
    metadata_path, georef_path = write_tiny_raw_fixture(tmp_path)

    outputs = postprocess_active_cells(
        metadata_path=metadata_path,
        georef_path=georef_path,
        out_dir=tmp_path / "out",
    )

    payload = pq.read_table(outputs.cells_geoparquet_path).to_pydict()

    assert payload["surface_flags"] == [1, 6]
    assert payload["surface_class"] == ["ground", "building_footprint"]
    assert payload["is_street_surface"] == [False, True]
    assert payload["is_building_footprint"] == [False, True]
    assert payload["include_in_public_realm_stats"] == [False, False]
    assert payload["include_in_outdoor_surface_stats"] == [True, False]
    assert payload["utci_00"] == [None, None]
    assert payload["utci_09"] == [None, None]
    assert payload["utci_10"] == pytest.approx([32.5, 40.25])
    assert payload["utci_11"] == [None, pytest.approx(28.0)]
    assert payload["utci_12"] == [None, None]


def test_postprocess_active_cells_preserves_street_family_overlap_flags_and_excludes_overlap_rows_from_public_realm_stats(
    tmp_path,
):
    metadata_path, georef_path = write_tiny_raw_fixture(
        tmp_path,
        positions=np.array(
            [
                [180723.5, 575888.0, 1.5],
                [180725.5, 575890.0, 1.5],
                [180727.5, 575892.0, 1.5],
            ],
            dtype="<f4",
        ),
        canonical=np.array([2, 5, 8], dtype="<u4"),
        canonical_count=12,
        utci=np.array(
            [
                [32.5, np.nan],
                [40.25, 28.0],
                [35.0, 31.5],
            ],
            dtype="<f4",
        ),
        shading=np.array([0.25, 0.75, 0.5], dtype="<f4"),
        surface_flags=np.array(
            [
                1,
                3,
                6,
            ],
            dtype=np.uint8,
        ),
        hours=[10, 11],
    )

    outputs = postprocess_active_cells(
        metadata_path=metadata_path,
        georef_path=georef_path,
        out_dir=tmp_path / "out",
    )

    payload = pq.read_table(outputs.cells_geoparquet_path).to_pydict()
    manifest = json.loads(outputs.manifest_path.read_text(encoding="utf-8"))

    assert payload["surface_flags"] == [1, 3, 6]
    assert payload["surface_class"] == [
        "ground",
        "street_surface",
        "building_footprint",
    ]
    assert payload["is_street_surface"] == [False, True, True]
    assert payload["is_building_footprint"] == [False, False, True]
    assert payload["include_in_public_realm_stats"] == [False, True, False]
    assert payload["include_in_outdoor_surface_stats"] == [True, True, False]
    assert manifest["counts"]["streetSurfaceRows"] == 2
    assert manifest["counts"]["buildingFootprintRows"] == 1
    assert manifest["counts"]["publicRealmRows"] == 1
    assert manifest["counts"]["outdoorSurfaceRows"] == 2


def test_postprocess_active_cells_rejects_zero_sampled_surface_flags_instead_of_emitting_unknown_surface_class(
    tmp_path,
):
    metadata_path, georef_path = write_tiny_raw_fixture(
        tmp_path,
        surface_flags=np.array([1, 0], dtype=np.uint8),
    )

    with pytest.raises(ValueError, match="sampled-surface|surface_flags|unknown"):
        postprocess_active_cells(
            metadata_path=metadata_path,
            georef_path=georef_path,
            out_dir=tmp_path / "out",
        )


def test_postprocess_active_cells_writes_legacy_geojson_only_when_explicitly_requested(tmp_path):
    metadata_path, georef_path = write_tiny_raw_fixture(tmp_path)
    out_dir = tmp_path / "out"

    outputs = postprocess_active_cells(
        metadata_path=metadata_path,
        georef_path=georef_path,
        out_dir=out_dir,
        include_legacy_geojson=True,
    )

    assert outputs.legacy_geojson_path is not None
    assert outputs.legacy_geojson_path.exists()

    manifest = json.loads(outputs.manifest_path.read_text(encoding="utf-8"))

    assert manifest["optional"]["legacyAllHoursGeojson"]["path"] == str(outputs.legacy_geojson_path)


def test_postprocess_active_cells_guards_legacy_geojson_row_count(tmp_path):
    metadata_path, georef_path = write_tiny_raw_fixture(tmp_path)

    with pytest.raises(ValueError, match="max legacy row guard"):
        postprocess_active_cells(
            metadata_path=metadata_path,
            georef_path=georef_path,
            out_dir=tmp_path / "guarded",
            include_legacy_geojson=True,
            legacy_geojson_max_rows=1,
        )

    outputs = postprocess_active_cells(
        metadata_path=metadata_path,
        georef_path=georef_path,
        out_dir=tmp_path / "forced",
        include_legacy_geojson=True,
        legacy_geojson_max_rows=1,
        force_legacy_geojson=True,
    )

    assert outputs.legacy_geojson_path is not None
    assert outputs.legacy_geojson_path.exists()
