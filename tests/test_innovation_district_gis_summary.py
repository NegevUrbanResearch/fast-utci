import numpy as np
import pytest

pytest.importorskip("pyproj")

from fast_utci.innovation_district_gis.raw import load_active_cell_artifacts
from fast_utci.innovation_district_gis.summary import build_summary
from test_innovation_district_gis_fixtures import write_tiny_raw_fixture


def test_build_summary_preserves_aggregate_no_data_and_per_hour_stats(tmp_path):
    metadata_path, georef_path = write_tiny_raw_fixture(tmp_path)
    artifacts = load_active_cell_artifacts(metadata_path, georef_path)

    summary = build_summary(artifacts)

    assert summary["counts"]["activeRows"] == 2
    assert summary["counts"]["hourCount"] == 2
    assert summary["utci"]["valueCount"] == 4
    assert summary["utci"]["validCount"] == 3
    assert summary["utci"]["invalidCount"] == 1
    assert sum(summary["utci"]["histogram"].values()) == 3
    assert sum(summary["utci"]["comfortBreakdown"].values()) == 3
    assert summary["utci"]["hotAreaShareUtciGe32"] == pytest.approx(2.0 / 3.0)
    assert summary["shadingIndex"]["validCount"] == 2
    assert summary["shadingIndex"]["percentShaded"] == pytest.approx(50.0)

    per_hour = summary["utci"]["perHour"]
    assert [hour["hour"] for hour in per_hour] == [10, 11]
    assert per_hour[0]["validCount"] == 2
    assert per_hour[0]["invalidCount"] == 0
    assert per_hour[0]["min"] == pytest.approx(32.5)
    assert per_hour[0]["max"] == pytest.approx(40.25)
    assert per_hour[0]["comfortBreakdown"]["strong_heat_stress"] == 1
    assert per_hour[0]["comfortBreakdown"]["very_strong_heat_stress"] == 1
    assert per_hour[0]["hotAreaShareUtciGe32"] == pytest.approx(1.0)
    assert per_hour[1]["validCount"] == 1
    assert per_hour[1]["invalidCount"] == 1
    assert per_hour[1]["mean"] == pytest.approx(28.0)
    assert per_hour[1]["hotAreaShareUtciGe32"] == pytest.approx(0.0)


def test_build_summary_uses_null_metrics_when_every_value_is_no_data(tmp_path):
    metadata_path, georef_path = write_tiny_raw_fixture(
        tmp_path,
        utci=np.array([[np.nan, np.nan], [np.nan, np.nan]], dtype="<f4"),
    )
    artifacts = load_active_cell_artifacts(metadata_path, georef_path)

    summary = build_summary(artifacts)

    assert summary["utci"]["validCount"] == 0
    assert summary["utci"]["min"] is None
    assert summary["utci"]["max"] is None
    assert summary["utci"]["mean"] is None
    assert summary["utci"]["hotAreaShareUtciGe32"] is None
    assert [hour["hotAreaShareUtciGe32"] for hour in summary["utci"]["perHour"]] == [None, None]
