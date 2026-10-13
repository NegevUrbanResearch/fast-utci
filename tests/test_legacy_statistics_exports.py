from pathlib import Path
import json
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from fast_utci.utci.statistics import (
    calculate_hour_statistics,
    calculate_utci_statistics,
    compute_summary_statistics,
)
from scripts import export_for_viewer


def results(values, mrt=None):
    return {'position_0': {'position': (0.0, 1.0, 2.0), 'utci': values,
                           'mrt': values if mrt is None else mrt}}


@pytest.mark.parametrize('data', [{}, results([]), results([np.nan]),
                                 results([np.inf, -np.inf])])
@pytest.mark.parametrize('calculate', [calculate_utci_statistics, compute_summary_statistics])
def test_unavailable_overall_utci_fails_clearly(data, calculate):
    with pytest.raises(ValueError, match='No finite UTCI values'):
        calculate(data)


def test_mixed_nonfinite_metrics_use_only_finite_values():
    data = results([20.0, np.nan, np.inf, -np.inf, 30.0], [40.0, np.nan, np.inf, 50.0])
    expected = {'min': 20.0, 'max': 30.0, 'mean': 25.0, 'std': 5.0}
    assert calculate_utci_statistics(data) == expected
    summary = compute_summary_statistics(data)
    assert summary['utci_stats'] == expected
    assert summary['valid_utci_values'] == 2
    assert summary['mrt_stats'] == {'min': 40.0, 'max': 50.0, 'mean': 45.0, 'std': 5.0}


def test_hour_statistics_preserve_requested_sequence_and_finite_values():
    data = results([20.0, np.nan, 30.0])
    data['position_1'] = {'utci': [np.inf, 25.0, 40.0]}
    assert calculate_hour_statistics(data, [8, 10, 12]) == [
        {'hour': 8, 'min': 20.0, 'max': 20.0, 'mean': 20.0},
        {'hour': 10, 'min': 25.0, 'max': 25.0, 'mean': 25.0},
        {'hour': 12, 'min': 30.0, 'max': 40.0, 'mean': 35.0},
    ]


@pytest.mark.parametrize('values', [[20.0, np.nan, 30.0], [20.0, np.inf, 30.0], [20.0]])
def test_unavailable_hour_identifies_hour(values):
    with pytest.raises(ValueError, match='hour 10'):
        calculate_hour_statistics(results(values), [8, 10, 12])


@pytest.fixture
def metadata_export(monkeypatch):
    monkeypatch.setattr(export_for_viewer, '_extract_location_from_epw', lambda _: {'city': 'Test'})
    monkeypatch.setattr(export_for_viewer, '_calculate_sun_positions', lambda *args: [])
    def export(path, **kwargs):
        export_for_viewer.export_metadata_json(
            results([20.0, np.nan, 30.0]) if kwargs.get('analysis_type') != 'full_day'
            else results([20.0, 25.0, 30.0]),
            'test', kwargs.pop('analysis_type', 'single_hour'), 2.0, '20250815',
            [8, 10, 12], 'model.glb', 'weather.epw',
            kwargs.pop('runtime_seconds', 1.0), path, **kwargs)
    return export


@pytest.mark.parametrize('kwargs', [{'runtime_seconds': np.nan},
                                   {'runtime_seconds': np.inf},
                                   {'shading_indices': np.array([0.5, np.inf])}])
def test_nonfinite_metadata_does_not_overwrite_existing_file(tmp_path, metadata_export, kwargs):
    path = tmp_path / 'metadata.json'
    original = '{"preserved": true}\n'
    path.write_text(original)
    with pytest.raises(ValueError, match='JSON'):
        metadata_export(path, **kwargs)
    assert path.read_text() == original


def test_valid_metadata_is_strict_json_with_historical_statistics(tmp_path, metadata_export):
    path = tmp_path / 'metadata.json'
    metadata_export(path, analysis_type='full_day')
    def reject_constant(value):
        raise ValueError(value)
    metadata = json.loads(path.read_text(), parse_constant=reject_constant)
    assert metadata['utci_range'] == {'min': 20.0, 'max': 30.0, 'mean': 25.0,
                                      'std': pytest.approx(np.std([20.0, 25.0, 30.0]))}
    assert [row['hour'] for row in metadata['hour_statistics']] == [8, 10, 12]
    assert path.read_text().startswith('{\n  "analysis_id": "test",')
