import json

import numpy as np
import pytest

pytest.importorskip("pyproj")

from fast_utci.innovation_district_gis.raw import load_active_cell_artifacts
from test_innovation_district_gis_fixtures import write_tiny_raw_fixture


def test_load_active_cell_artifacts_validates_binary_metadata_and_arrays(tmp_path):
    metadata_path, georef_path = write_tiny_raw_fixture(tmp_path)

    loaded = load_active_cell_artifacts(metadata_path, georef_path)

    assert loaded.metadata["activeMask"]["source"] == "base+road"
    assert loaded.metadata["activeMask"]["signature"] == "active-mask-signature"
    assert loaded.positions.dtype == np.dtype("<f4")
    assert loaded.canonical_indices.dtype == np.dtype("<u4")
    assert loaded.positions.shape == (2, 3)
    assert loaded.utci.shape == (2, 2)
    assert loaded.shading_index.shape == (2,)
    assert loaded.canonical_indices.tolist() == [2, 5]
    assert loaded.positions[0].tolist() == pytest.approx([180723.5, 575888.0, 1.5])
    assert loaded.utci[0, 0] == pytest.approx(32.5)
    assert np.isnan(loaded.utci[0, 1])
    assert loaded.shading_index[1] == pytest.approx(0.75)


def test_load_active_cell_artifacts_rejects_contract_mismatches(tmp_path):
    metadata_path, georef_path = write_tiny_raw_fixture(tmp_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["arrays"]["utci"]["byteLength"] += 4
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="byteLength"):
        load_active_cell_artifacts(metadata_path, georef_path)

    metadata_path, georef_path = write_tiny_raw_fixture(tmp_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["activeMask"]["source"] = "base"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="base\\+road"):
        load_active_cell_artifacts(metadata_path, georef_path)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda metadata, georef: metadata["arrays"]["positions"].__setitem__("dtype", "u32"),
            "arrays\\.positions\\.dtype",
        ),
        (
            lambda metadata, georef: metadata["arrays"]["utci"].__setitem__("endianness", "big"),
            "arrays\\.utci\\.endianness",
        ),
        (
            lambda metadata, georef: metadata["layout"].__setitem__("utci", "hour-major-point"),
            "layout\\.utci",
        ),
        (
            lambda metadata, georef: metadata["arrays"]["shadingIndex"].__setitem__("shape", [2, 1]),
            "arrays\\.shadingIndex\\.shape",
        ),
        (
            lambda metadata, georef: metadata["activeMask"].pop("checksum"),
            "activeMask\\.checksum",
        ),
        (
            lambda metadata, georef: metadata["activeMask"].pop("signature"),
            "activeMask\\.signature",
        ),
        (
            lambda metadata, georef: metadata["files"]["positions"].pop("checksum"),
            "files\\.positions\\.checksum",
        ),
        (
            lambda metadata, georef: metadata["files"]["canonicalIndices"].pop("checksum"),
            "files\\.canonicalIndices\\.checksum",
        ),
        (
            lambda metadata, georef: metadata["files"]["utci"].pop("checksum"),
            "files\\.utci\\.checksum",
        ),
        (
            lambda metadata, georef: metadata["files"]["shadingIndex"].pop("checksum"),
            "files\\.shadingIndex\\.checksum",
        ),
        (
            lambda metadata, georef: metadata["files"]["utci"].__setitem__("checksum", "0" * 64),
            "files\\.utci\\.checksum",
        ),
        (
            lambda metadata, georef: metadata.__setitem__("hourCount", 3),
            "hours length must match hourCount",
        ),
        (
            lambda metadata, georef: metadata["arrays"]["utci"].__setitem__("shape", [2, 3]),
            "arrays\\.utci\\.shape",
        ),
        (
            lambda metadata, georef: georef.__setitem__("declared_crs", "EPSG:4326"),
            'georef\\.declared_crs must be "EPSG:2039"',
        ),
    ],
)
def test_load_active_cell_artifacts_rejects_validation_surface(tmp_path, mutate, match):
    metadata_path, georef_path = write_tiny_raw_fixture(tmp_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    georef = json.loads(georef_path.read_text(encoding="utf-8"))

    mutate(metadata, georef)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    georef_path.write_text(json.dumps(georef), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        load_active_cell_artifacts(metadata_path, georef_path)
