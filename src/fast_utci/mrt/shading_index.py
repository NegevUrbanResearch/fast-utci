"""
Shading Index calculation for thermal comfort analysis.

Shading Index measures the proportion of sunlight hours during which each point
is fully shaded (no direct solar radiation). This metric follows Israeli
shading metrics guidelines for urban planning.

Current Implementation:
- Point-based calculation: Each point's Shading Index = (hours shaded) / (total sunlight hours)
- A point is considered "shaded" when fract_body_exp == 0.0 (no direct solar radiation)
- Uses all sunlight hours from the analysis period (auto-detected from sun_data.is_sun_up)

TODO: Future enhancement - Sidewalk area aggregation
When sidewalk layers are available in the model:
1. Group points by sidewalk segment/identifier
2. For each hour, calculate % of sidewalk area that is shaded
   - Count shaded points in segment
   - Calculate: (shaded_points / total_points) * 100
3. Count hours where >50% of sidewalk area is shaded
4. Calculate area-based Shading Index: (hours with >50% shaded) / (total sunlight hours)
5. Export both point-based and area-based indices
See: Israeli Shading Metrics Guide, Section on sidewalk-level calculation

This enhancement will require:
- Sidewalk layer identification in model geometry
- Point-to-sidewalk mapping/grouping
- Area-based aggregation logic
- Updated export format to include area-based indices
"""

import numpy as np
from typing import List
from .exposure import ExposureResult
from .solar import SunData


def calculate_shading_index(
    exposure_results: List[ExposureResult],
    sun_data: SunData
) -> np.ndarray:
    """
    Calculate Shading Index for each position.
    
    Shading Index = (hours shaded during sunlight) / (total sunlight hours)
    
    A point is considered "shaded" when fract_body_exp == 0.0 (no direct
    solar radiation). Only sunlight hours (is_sun_up == True) are considered
    in the calculation.
    
    Args:
        exposure_results: List of ExposureResult objects, one per position
        sun_data: SunData object with sun vectors and is_sun_up mask
        
    Returns:
        Array of shape (n_positions,) with Shading Index values (0.0-1.0)
        where:
        - 0.0 = always exposed during sunlight hours
        - 1.0 = always shaded during sunlight hours
        - Values in between = proportion of sunlight hours that are shaded
        
    Raises:
        ValueError: If exposure array lengths don't match sun_data length
    """
    n_positions = len(exposure_results)
    n_hours = len(sun_data.is_sun_up)
    
    if n_positions == 0:
        return np.array([], dtype=np.float64)
    
    # Validate array lengths
    for i, exp_result in enumerate(exposure_results):
        if len(exp_result.fract_body_exp) != n_hours:
            raise ValueError(
                f"ExposureResult[{i}] fract_body_exp length ({len(exp_result.fract_body_exp)}) "
                f"doesn't match sun_data length ({n_hours})"
            )
    
    # Get sunlight hours mask
    is_sun_up = np.asarray(sun_data.is_sun_up, dtype=bool)
    n_sunlight_hours = np.sum(is_sun_up)
    
    # Handle edge case: no sunlight hours
    if n_sunlight_hours == 0:
        # By definition, if there are no sunlight hours, all points are fully shaded
        return np.ones(n_positions, dtype=np.float64)
    
    # Calculate Shading Index for each position
    shading_indices = np.zeros(n_positions, dtype=np.float64)
    
    for i, exp_result in enumerate(exposure_results):
        fract_body_exp = exp_result.fract_body_exp
        
        # Filter to sunlight hours only
        sunlight_exposure = fract_body_exp[is_sun_up]
        
        # Count hours where fract_body_exp == 0.0 (fully shaded)
        # Use small epsilon for floating point comparison
        shaded_hours = np.sum(sunlight_exposure == 0.0)
        
        # Calculate Shading Index
        shading_indices[i] = shaded_hours / n_sunlight_hours
    
    return shading_indices

