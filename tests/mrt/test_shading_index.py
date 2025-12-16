"""
Tests for Shading Index calculation.

Shading Index measures the proportion of sunlight hours during which
each point is fully shaded (no direct solar radiation).
"""

import numpy as np
import pytest
from fast_utci.mrt.exposure import ExposureResult
from fast_utci.mrt.solar import SunData
from fast_utci.mrt.shading_index import calculate_shading_index


def test_shading_index_fully_shaded():
    """Test Shading Index for a point that is always shaded during sunlight hours."""
    # 10 hours total, 8 hours with sun up, all shaded
    n_hours = 10
    n_positions = 1
    
    # Sun is up for hours 1-8 (indices 1-8)
    is_sun_up = np.array([False, True, True, True, True, True, True, True, True, False])
    
    # All sunlight hours are fully shaded (fract_body_exp == 0.0)
    fract_body_exp = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    exposure_result = ExposureResult(
        fract_body_exp=fract_body_exp,
        sky_exposure=0.5,
        position=np.array([0.0, 0.0, 0.0]),
        sample_points=np.array([[0.0, 0.0, 1.7]])
    )
    
    sun_data = SunData(
        sun_vectors=np.zeros((n_hours, 3)),
        is_sun_up=is_sun_up,
        solar_times=[],
        hoys=np.arange(n_hours)
    )
    
    shading_indices = calculate_shading_index([exposure_result], sun_data)
    
    assert len(shading_indices) == 1
    assert shading_indices[0] == 1.0  # 100% shaded during all sunlight hours


def test_shading_index_fully_exposed():
    """Test Shading Index for a point that is always exposed during sunlight hours."""
    # 10 hours total, 8 hours with sun up, all exposed
    n_hours = 10
    n_positions = 1
    
    # Sun is up for hours 1-8 (indices 1-8)
    is_sun_up = np.array([False, True, True, True, True, True, True, True, True, False])
    
    # All sunlight hours are fully exposed (fract_body_exp == 1.0)
    fract_body_exp = np.array([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0])
    
    exposure_result = ExposureResult(
        fract_body_exp=fract_body_exp,
        sky_exposure=1.0,
        position=np.array([0.0, 0.0, 0.0]),
        sample_points=np.array([[0.0, 0.0, 1.7]])
    )
    
    sun_data = SunData(
        sun_vectors=np.zeros((n_hours, 3)),
        is_sun_up=is_sun_up,
        solar_times=[],
        hoys=np.arange(n_hours)
    )
    
    shading_indices = calculate_shading_index([exposure_result], sun_data)
    
    assert len(shading_indices) == 1
    assert shading_indices[0] == 0.0  # 0% shaded (always exposed)


def test_shading_index_partial_shading():
    """Test Shading Index for a point with partial shading."""
    # 10 hours total, 8 hours with sun up, half shaded
    n_hours = 10
    n_positions = 1
    
    # Sun is up for hours 1-8 (indices 1-8)
    is_sun_up = np.array([False, True, True, True, True, True, True, True, True, False])
    
    # Half of sunlight hours are shaded (4 out of 8)
    fract_body_exp = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
    
    exposure_result = ExposureResult(
        fract_body_exp=fract_body_exp,
        sky_exposure=0.5,
        position=np.array([0.0, 0.0, 0.0]),
        sample_points=np.array([[0.0, 0.0, 1.7]])
    )
    
    sun_data = SunData(
        sun_vectors=np.zeros((n_hours, 3)),
        is_sun_up=is_sun_up,
        solar_times=[],
        hoys=np.arange(n_hours)
    )
    
    shading_indices = calculate_shading_index([exposure_result], sun_data)
    
    assert len(shading_indices) == 1
    assert shading_indices[0] == 0.5  # 4 out of 8 sunlight hours shaded = 0.5


def test_shading_index_multiple_positions():
    """Test Shading Index calculation for multiple positions."""
    n_hours = 10
    n_positions = 3
    
    # Sun is up for hours 1-8
    is_sun_up = np.array([False, True, True, True, True, True, True, True, True, False])
    
    exposure_results = []
    # Position 0: fully shaded
    exposure_results.append(ExposureResult(
        fract_body_exp=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        sky_exposure=0.5,
        position=np.array([0.0, 0.0, 0.0]),
        sample_points=np.array([[0.0, 0.0, 1.7]])
    ))
    # Position 1: fully exposed
    exposure_results.append(ExposureResult(
        fract_body_exp=np.array([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]),
        sky_exposure=1.0,
        position=np.array([1.0, 0.0, 0.0]),
        sample_points=np.array([[1.0, 0.0, 1.7]])
    ))
    # Position 2: half shaded
    exposure_results.append(ExposureResult(
        fract_body_exp=np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]),
        sky_exposure=0.5,
        position=np.array([2.0, 0.0, 0.0]),
        sample_points=np.array([[2.0, 0.0, 1.7]])
    ))
    
    sun_data = SunData(
        sun_vectors=np.zeros((n_hours, 3)),
        is_sun_up=is_sun_up,
        solar_times=[],
        hoys=np.arange(n_hours)
    )
    
    shading_indices = calculate_shading_index(exposure_results, sun_data)
    
    assert len(shading_indices) == 3
    assert shading_indices[0] == 1.0  # Fully shaded
    assert shading_indices[1] == 0.0  # Fully exposed
    assert shading_indices[2] == 0.5  # Half shaded


def test_shading_index_no_sunlight_hours():
    """Test Shading Index when there are no sunlight hours (e.g., polar night)."""
    n_hours = 10
    n_positions = 1
    
    # No sun up hours
    is_sun_up = np.array([False] * n_hours)
    
    fract_body_exp = np.array([0.0] * n_hours)
    
    exposure_result = ExposureResult(
        fract_body_exp=fract_body_exp,
        sky_exposure=0.5,
        position=np.array([0.0, 0.0, 0.0]),
        sample_points=np.array([[0.0, 0.0, 1.7]])
    )
    
    sun_data = SunData(
        sun_vectors=np.zeros((n_hours, 3)),
        is_sun_up=is_sun_up,
        solar_times=[],
        hoys=np.arange(n_hours)
    )
    
    shading_indices = calculate_shading_index([exposure_result], sun_data)
    
    assert len(shading_indices) == 1
    # When no sunlight hours, point is considered fully shaded (by definition)
    assert shading_indices[0] == 1.0


def test_shading_index_partial_exposure():
    """Test Shading Index with partial exposure values (not just 0.0 or 1.0)."""
    # Test that only fract_body_exp == 0.0 counts as shaded
    n_hours = 10
    n_positions = 1
    
    # Sun is up for hours 1-8
    is_sun_up = np.array([False, True, True, True, True, True, True, True, True, False])
    
    # Some hours have partial exposure (0.5, 0.3) - these should NOT count as shaded
    fract_body_exp = np.array([0.0, 0.0, 0.5, 0.3, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0])
    
    exposure_result = ExposureResult(
        fract_body_exp=fract_body_exp,
        sky_exposure=0.5,
        position=np.array([0.0, 0.0, 0.0]),
        sample_points=np.array([[0.0, 0.0, 1.7]])
    )
    
    sun_data = SunData(
        sun_vectors=np.zeros((n_hours, 3)),
        is_sun_up=is_sun_up,
        solar_times=[],
        hoys=np.arange(n_hours)
    )
    
    shading_indices = calculate_shading_index([exposure_result], sun_data)
    
    assert len(shading_indices) == 1
    # Only hours with fract_body_exp == 0.0 count as shaded
    # Hours 1, 4, 8 are shaded (indices 1, 4, 8) = 3 out of 8 = 0.375
    assert shading_indices[0] == pytest.approx(3.0 / 8.0, rel=1e-6)


def test_shading_index_array_length_mismatch():
    """Test that function handles array length mismatches gracefully."""
    n_hours = 10
    n_positions = 1
    
    is_sun_up = np.array([False, True, True, True, True, True, True, True, True, False])
    
    # Mismatched length - should raise error or handle gracefully
    fract_body_exp = np.array([0.0, 0.0, 0.0])  # Only 3 hours instead of 10
    
    exposure_result = ExposureResult(
        fract_body_exp=fract_body_exp,
        sky_exposure=0.5,
        position=np.array([0.0, 0.0, 0.0]),
        sample_points=np.array([[0.0, 0.0, 1.7]])
    )
    
    sun_data = SunData(
        sun_vectors=np.zeros((n_hours, 3)),
        is_sun_up=is_sun_up,
        solar_times=[],
        hoys=np.arange(n_hours)
    )
    
    # Should raise ValueError for length mismatch
    with pytest.raises(ValueError, match="length"):
        calculate_shading_index([exposure_result], sun_data)


def test_shading_index_epsilon_threshold():
    """Test that very small floating point values near 0.0 are treated as shaded."""
    n_hours = 10
    n_positions = 1
    
    is_sun_up = np.array([False, True, True, True, True, True, True, True, True, False])
    
    # Very small values that should be treated as 0.0 (shaded) due to floating point precision
    fract_body_exp = np.array([
        0.0,           # hour 0: sun down
        1e-10,         # hour 1: tiny value (should be shaded)
        0.0,           # hour 2: exactly 0.0 (shaded)
        1e-15,         # hour 3: even tinier (should be shaded)
        0.5,           # hour 4: partial exposure (not shaded)
        1.0,           # hour 5: full exposure (not shaded)
        0.0,           # hour 6: shaded
        0.0,           # hour 7: shaded
        0.0,           # hour 8: shaded
        0.0            # hour 9: sun down
    ])
    
    exposure_result = ExposureResult(
        fract_body_exp=fract_body_exp,
        sky_exposure=0.5,
        position=np.array([0.0, 0.0, 0.0]),
        sample_points=np.array([[0.0, 0.0, 1.7]])
    )
    
    sun_data = SunData(
        sun_vectors=np.zeros((n_hours, 3)),
        is_sun_up=is_sun_up,
        solar_times=[],
        hoys=np.arange(n_hours)
    )
    
    shading_indices = calculate_shading_index([exposure_result], sun_data)
    
    assert len(shading_indices) == 1
    # Hours 1, 2, 3, 6, 7, 8 are shaded during sunlight = 6 out of 8 = 0.75
    # Note: epsilon threshold should treat 1e-10 and 1e-15 as shaded
    assert shading_indices[0] == pytest.approx(6.0 / 8.0, rel=1e-6)


def test_shading_index_building_scenario():
    """Test Shading Index for a point under a building (should be fully shaded)."""
    n_hours = 24
    n_positions = 1
    
    # Simulate August day: sun up from hour 6 to 19 (14 hours)
    is_sun_up = np.array([
        False, False, False, False, False, False,  # 0-5: night
        True, True, True, True, True, True, True, True, True, True, True, True, True, True,  # 6-19: day
        False, False, False, False  # 20-23: night
    ])
    
    # Point under building: always shaded when sun is up
    fract_body_exp = np.array([
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # night
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # day: all shaded
        0.0, 0.0, 0.0, 0.0  # night
    ])
    
    exposure_result = ExposureResult(
        fract_body_exp=fract_body_exp,
        sky_exposure=0.1,  # Low sky exposure (under building)
        position=np.array([0.0, 0.0, 0.0]),
        sample_points=np.array([[0.0, 0.0, 1.7]])
    )
    
    sun_data = SunData(
        sun_vectors=np.zeros((n_hours, 3)),
        is_sun_up=is_sun_up,
        solar_times=[],
        hoys=np.arange(n_hours)
    )
    
    shading_indices = calculate_shading_index([exposure_result], sun_data)
    
    assert len(shading_indices) == 1
    assert shading_indices[0] == 1.0  # 100% shaded during all sunlight hours


def test_shading_index_open_field_scenario():
    """Test Shading Index for a point in open field (should be fully exposed)."""
    n_hours = 24
    n_positions = 1
    
    # Simulate August day: sun up from hour 6 to 19 (14 hours)
    is_sun_up = np.array([
        False, False, False, False, False, False,  # 0-5: night
        True, True, True, True, True, True, True, True, True, True, True, True, True, True,  # 6-19: day
        False, False, False, False  # 20-23: night
    ])
    
    # Point in open field: always exposed when sun is up
    fract_body_exp = np.array([
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # night
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # day: all exposed
        0.0, 0.0, 0.0, 0.0  # night
    ])
    
    exposure_result = ExposureResult(
        fract_body_exp=fract_body_exp,
        sky_exposure=1.0,  # Full sky exposure
        position=np.array([0.0, 0.0, 0.0]),
        sample_points=np.array([[0.0, 0.0, 1.7]])
    )
    
    sun_data = SunData(
        sun_vectors=np.zeros((n_hours, 3)),
        is_sun_up=is_sun_up,
        solar_times=[],
        hoys=np.arange(n_hours)
    )
    
    shading_indices = calculate_shading_index([exposure_result], sun_data)
    
    assert len(shading_indices) == 1
    assert shading_indices[0] == 0.0  # 0% shaded (always exposed during sunlight)

