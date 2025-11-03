"""
Boundary averaging utilities for MRT calculations.

Provides shared helper functions for creating boundary-averaged MRT arrays
used in both serial and parallel computation paths.
"""

import numpy as np
from typing import Tuple


def create_boundary_arrays(mrt_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create mrt0 and mrt1 arrays for boundary averaging.
    
    For boundary averaging UTCI calculations:
    - mrt0[i] uses hour i's data
    - mrt1[i] uses hour i+1's data
    For the last hour, mrt1 duplicates the final value (no next-day wrap).
    
    Args:
        mrt_array: MRT values for N hours
        
    Returns:
        Tuple of (mrt0, mrt1) both with length N
    """
    if len(mrt_array) > 1:
        return mrt_array, np.concatenate([mrt_array[1:], [mrt_array[-1]]])
    return mrt_array, mrt_array

