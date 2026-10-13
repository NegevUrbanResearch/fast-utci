"""Regression coverage for legacy imports and weather boundary indexing."""

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest


def test_gis_and_mesh_imports_do_not_load_thermal_dependencies():
    script = '''
import importlib.abc
import sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'ladybug', 'ladybug_comfort', 'pythermalcomfort'}:
            raise AssertionError('thermal import: ' + fullname)
sys.meta_path.insert(0, Guard())
import fast_utci
import fast_utci.innovation_district_gis
import fast_utci.shared.io
from fast_utci import get_combined_mesh, get_ground_bounds
assert callable(get_combined_mesh) and callable(get_ground_bounds)
assert not any(name.startswith(('ladybug', 'pythermalcomfort', 'fast_utci.mrt', 'fast_utci.utci')) for name in sys.modules)
try:
    fast_utci.missing_export
except AttributeError:
    pass
else:
    raise AssertionError('unknown attributes must fail')
'''
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture
def calculation():
    from fast_utci.utci import calculation
    return calculation


def weather(air):
    return {'air_temp': np.array(air, dtype=float),
            'wind_speed': np.full(len(air), 1.0),
            'relative_humidity': np.full(len(air), 50.0)}


def mrt(n=3):
    return {'position': (0, 0, 1), 'mrt0': np.full(n, 30.), 'mrt1': np.full(n, 32.)}


@pytest.mark.parametrize('air,expected', [
    ([10, 20, 30, 40], [15, 25, 35]),
    ([10, 20, 30], [15, 25, 30]),
    ([25], [25, 25, 25]),
    ([10, 20], [15, 20, 20]),
])
def test_weather_padding_matches_both_paths(calculation, monkeypatch, air, expected):
    monkeypatch.setattr(calculation, 'utci', lambda **args: args['tdb'])
    vector = calculation.BoundaryAveragingCalculator().calculate(mrt(), weather(air))
    loop = calculation.BoundaryAveragingCalculator(False).calculate(mrt(), pd.DataFrame(weather(air)))
    np.testing.assert_allclose(vector.utci, expected)
    np.testing.assert_allclose(loop.utci, expected)
    np.testing.assert_allclose(vector.air_temp, loop.air_temp)


@pytest.mark.parametrize('vectorized', [True, False])
def test_empty_weather_fails_before_calculation(calculation, monkeypatch, vectorized):
    monkeypatch.setattr(calculation, 'utci', lambda **args: pytest.fail('UTCI called before validation'))
    data = weather([])
    if not vectorized:
        data = pd.DataFrame(data)
    with pytest.raises(ValueError, match='weather.*empty|empty.*weather'):
        calculation.BoundaryAveragingCalculator(vectorized).calculate(mrt(), data)


@pytest.mark.parametrize('vectorized', [True, False])
def test_mismatched_weather_lengths_fail(calculation, vectorized):
    data = weather([20, 25])
    data['wind_speed'] = np.array([1.])
    with pytest.raises(ValueError, match='weather.*length|length.*weather'):
        calculation.BoundaryAveragingCalculator(vectorized).calculate(mrt(), data)


@pytest.mark.parametrize('vectorized', [True, False])
@pytest.mark.parametrize('length', [2, 4])
def test_inconsistent_mrt_lengths_fail(calculation, vectorized, length):
    data = mrt()
    data['mrt1'] = np.full(length, 32.)
    with pytest.raises(ValueError, match='mrt0.*mrt1|MRT'):
        calculation.BoundaryAveragingCalculator(vectorized).calculate(data, weather([20, 25, 30, 35]))


@pytest.mark.parametrize('vectorized', [True, False])
@pytest.mark.parametrize('field', ['mrt0', 'mrt1'])
@pytest.mark.parametrize('values', [np.array(30.), np.full((3, 1), 30.)], ids=['scalar', '2d'])
def test_malformed_mrt_shapes_fail_before_calculation(calculation, monkeypatch, vectorized, field, values):
    monkeypatch.setattr(calculation, 'utci', lambda **args: pytest.fail('UTCI called before validation'))
    data = mrt()
    data[field] = values
    with pytest.raises(ValueError, match='MRT.*one-dimensional'):
        calculation.BoundaryAveragingCalculator(vectorized).calculate(data, weather([20, 25, 30, 35]))


@pytest.mark.parametrize('vectorized', [True, False])
@pytest.mark.parametrize('field', ['air_temp', 'wind_speed', 'relative_humidity'])
@pytest.mark.parametrize('values', [np.array(25.), np.full((3, 1), 25.)], ids=['scalar', '2d'])
def test_malformed_weather_shapes_fail_before_calculation(calculation, monkeypatch, vectorized, field, values):
    monkeypatch.setattr(calculation, 'utci', lambda **args: pytest.fail('UTCI called before validation'))
    data = weather([20, 25, 30])
    data[field] = values
    with pytest.raises(ValueError, match='weather.*one-dimensional'):
        calculation.BoundaryAveragingCalculator(vectorized).calculate(mrt(), data)


def test_iterative_dictionary_repeats_single_weather_row(calculation, monkeypatch):
    monkeypatch.setattr(calculation, 'utci', lambda **args: args['tdb'])
    result = calculation.BoundaryAveragingCalculator(False).calculate(mrt(), weather([25]))
    np.testing.assert_allclose(result.utci, [25, 25, 25])
    np.testing.assert_allclose(result.air_temp, [25, 25, 25])
    np.testing.assert_allclose(result.wind_speed, [1, 1, 1])
    np.testing.assert_allclose(result.relative_humidity, [50, 50, 50])


def test_real_utci_paths_agree(calculation):
    data = weather([25, 27, 29, 30])
    vector = calculation.BoundaryAveragingCalculator().calculate(mrt(), data)
    loop = calculation.BoundaryAveragingCalculator(False).calculate(mrt(), pd.DataFrame(data))
    assert np.isfinite(vector.utci).all()
    np.testing.assert_allclose(vector.utci, loop.utci)


def test_public_exports_resolve():
    from fast_utci import (MRTCalculator, MRTConfig, UTCICalculator,
                           read_project_data, get_combined_mesh, get_ground_bounds)
    assert all(callable(item) for item in (MRTCalculator, MRTConfig, UTCICalculator,
               read_project_data, get_combined_mesh, get_ground_bounds))


@pytest.mark.parametrize('first', ['UTCICalculator', 'MRTCalculator', 'read_project_data'])
def test_public_exports_resolve_in_fresh_import_orders(first):
    script = f'''
from fast_utci import {first}
import fast_utci
for name in fast_utci.__all__:
    value = getattr(fast_utci, name)
    assert callable(value), name
    assert getattr(fast_utci, name) is value
from fast_utci.mrt import EPWAdapter, DataFrameAdapter, WeatherDataSource, create_weather_adapter
from fast_utci.shared.io import read_project_data
assert read_project_data is fast_utci.read_project_data
'''
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
