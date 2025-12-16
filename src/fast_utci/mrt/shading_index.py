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
    # Epsilon threshold for floating point comparison (treat very small values as 0.0)
    EPSILON = 1e-6
    
    shading_indices = np.zeros(n_positions, dtype=np.float64)
    
    # Debug: Track statistics
    fully_shaded_count = 0
    fully_exposed_count = 0
    sample_exposure_values = []
    
    for i, exp_result in enumerate(exposure_results):
        fract_body_exp = np.asarray(exp_result.fract_body_exp, dtype=np.float64)
        
        # Filter to sunlight hours only
        sunlight_exposure = fract_body_exp[is_sun_up]
        
        # Count hours where fract_body_exp <= EPSILON (fully shaded)
        # Use epsilon threshold to handle floating point precision issues
        # Values very close to 0.0 are treated as fully shaded
        shaded_hours = np.sum(sunlight_exposure <= EPSILON)
        
        # Calculate Shading Index
        shading_indices[i] = shaded_hours / n_sunlight_hours
        
        # Debug: Track detailed stats for first few positions
        if i < 5:
            sample_exposure_values.append({
                'position': i,
                'position_coords': exp_result.position.tolist(),
                'fract_body_exp_all': fract_body_exp.tolist(),
                'is_sun_up': is_sun_up.tolist(),
                'sunlight_exposure': sunlight_exposure.tolist(),
                'shaded_hours': int(shaded_hours),
                'total_sunlight_hours': int(n_sunlight_hours),
                'shading_index': float(shading_indices[i]),
                'min_fract_sunlight': float(np.min(sunlight_exposure)) if len(sunlight_exposure) > 0 else 0.0,
                'max_fract_sunlight': float(np.max(sunlight_exposure)) if len(sunlight_exposure) > 0 else 0.0,
                'mean_fract_sunlight': float(np.mean(sunlight_exposure)) if len(sunlight_exposure) > 0 else 0.0,
            })
        
        if shading_indices[i] >= 0.99:
            fully_shaded_count += 1
        elif shading_indices[i] <= 0.01:
            fully_exposed_count += 1
    
    # Debug output (can be removed later or made conditional)
    import logging
    logger = logging.getLogger(__name__)
    logger.debug(f"Shading Index calculation summary:")
    logger.debug(f"  Fully shaded positions (>=0.99): {fully_shaded_count}/{n_positions}")
    logger.debug(f"  Fully exposed positions (<=0.01): {fully_exposed_count}/{n_positions}")
    logger.debug(f"  Sample exposure values: {sample_exposure_values[:3]}")
    
    return shading_indices

