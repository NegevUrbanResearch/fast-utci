import sys
import subprocess
import json
from pathlib import Path

import pytest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

pytest.importorskip("pyproj")

from fast_utci.innovation_district_gis.transforms import epsg2039_to_wgs84
from test_innovation_district_gis_fixtures import write_tiny_raw_fixture


def test_epsg2039_to_wgs84_lands_near_beer_sheva():
    lon, lat = epsg2039_to_wgs84(180723.4887, 575888.0147)

    assert 34.7 < lon < 34.9
    assert 31.1 < lat < 31.4


def test_cli_wrapper_runs_from_script_path():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "postprocess_innovation_district_gis.py"), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--metadata" in result.stdout
    assert "--debug-geojson-limit" in result.stdout


def test_package_root_import_does_not_require_pyarrow():
    code = """
import builtins
import sys
import fast_utci

for module_name in list(sys.modules):
    if module_name == "pyarrow" or module_name.startswith("pyarrow."):
        del sys.modules[module_name]

real_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "pyarrow" or name.startswith("pyarrow."):
        raise AssertionError(f"pyarrow import blocked during package root import: {name}")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import
import fast_utci.innovation_district_gis as gis
assert "postprocess_active_cells" in gis.__all__
assert gis.ActiveCellArtifacts is not None
print("ok")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_cli_stdout_omits_default_geojson_when_not_written(tmp_path):
    metadata_path, georef_path = write_tiny_raw_fixture(
        tmp_path,
        positions=np.array(
            [
                [180723.5, 575888.0, 1.5],
                [180725.5, 575890.0, 1.5],
            ],
            dtype="<f4",
        ),
    )
    out_dir = tmp_path / "out"

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "postprocess_innovation_district_gis.py"),
            "--metadata",
            str(metadata_path),
            "--georef",
            str(georef_path),
            "--out-dir",
            str(out_dir),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert "geojson_path" not in payload
    assert "legacy_geojson_path" not in payload
    assert "summary_path" not in payload
    assert "geometry_parquet_path" not in payload
    assert "values_parquet_path" not in payload
    cells_path = payload["cells_geoparquet_path"].replace("\\", "/")
    debug_path = payload["debug_geojson_path"].replace("\\", "/")
    manifest_path = payload["manifest_path"].replace("\\", "/")
    assert cells_path.endswith("2025-08-15_2m/cells.geoparquet")
    assert debug_path.endswith("2025-08-15_2m/qa/debug-sample.geojson")
    assert manifest_path.endswith("2025-08-15_2m/manifest.json")
