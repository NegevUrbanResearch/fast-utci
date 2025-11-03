"""
Statistical analysis and thermal comfort classification for UTCI results.

Provides functions for analyzing UTCI values, classifying thermal comfort,
and computing summary statistics.

This module provides shared utilities used by both export modules and analysis scripts
to avoid code duplication.
"""

import numpy as np
from typing import Dict, Any, Tuple, List


# UTCI thermal comfort thresholds (°C)
UTCI_COMFORT_THRESHOLDS = {
    'extreme_cold': (-float('inf'), -40),
    'very_cold': (-40, -27),
    'cold': (-27, -13),
    'cool': (-13, 9),
    'comfortable': (9, 26),
    'warm': (26, 32),
    'hot': (32, 38),
    'very_hot': (38, 46),
    'extreme_hot': (46, float('inf'))
}


# ============================================================================
# Shared Utility Functions (used by export modules and scripts)
# ============================================================================

def extract_all_utci_values(utci_results: Dict[str, Any]) -> np.ndarray:
    """
    Extract all UTCI values from results as a flat numpy array.
    
    This is a shared utility used by export_for_viewer.py, run_analysis.py,
    and other scripts to avoid code duplication.
    
    Args:
        utci_results: Dictionary of UTCI results from compute_utci()
        
    Returns:
        Flat numpy array of all UTCI values across all positions and hours
    """
    all_utci = []
    for data in utci_results.values():
        utci_vals = data['utci']
        if isinstance(utci_vals, (list, np.ndarray)):
            all_utci.extend(utci_vals)
        else:
            all_utci.append(utci_vals)
    return np.array(all_utci)


def extract_positions(utci_results: Dict[str, Any]) -> np.ndarray:
    """
    Extract all positions from results as Nx3 numpy array.
    
    Args:
        utci_results: Dictionary of UTCI results from compute_utci()
        
    Returns:
        Numpy array of shape (N, 3) with x, y, z coordinates
    """
    return np.array([data['position'] for data in utci_results.values()])


def calculate_utci_statistics(utci_results: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate overall UTCI statistics (NaN-safe).
    
    This function uses nanmin/nanmax/nanmean to handle NaN values gracefully,
    ensuring valid JSON output for web viewers.
    
    Args:
        utci_results: Dictionary of UTCI results from compute_utci()
        
    Returns:
        Dictionary with 'min', 'max', 'mean', 'std' statistics
    """
    all_utci = extract_all_utci_values(utci_results)
    return {
        'min': float(np.nanmin(all_utci)),
        'max': float(np.nanmax(all_utci)),
        'mean': float(np.nanmean(all_utci)),
        'std': float(np.nanstd(all_utci))
    }


def calculate_hour_statistics(
    utci_results: Dict[str, Any],
    hours: List[int]
) -> List[Dict[str, Any]]:
    """
    Calculate per-hour UTCI statistics for full day analysis (NaN-safe).
    
    Consolidates logic from export_for_viewer.py to avoid duplication.
    Uses NaN-safe numpy functions to prevent invalid JSON output.
    
    Args:
        utci_results: Dictionary of UTCI results from compute_utci()
        hours: List of hours analyzed (e.g., [0, 1, 2, ..., 23])
        
    Returns:
        List of dictionaries with hour, min, max, mean for each hour
    """
    num_hours = len(hours)
    sorted_keys = sorted(utci_results.keys())
    
    # Organize UTCI values by hour
    utci_by_hour = [[] for _ in range(num_hours)]
    
    for pos_key in sorted_keys:
        data = utci_results[pos_key]
        utci_vals = data['utci']
        
        if isinstance(utci_vals, (list, np.ndarray)):
            for hour_idx, utci_val in enumerate(utci_vals[:num_hours]):
                utci_by_hour[hour_idx].append(float(utci_val))
    
    # Calculate statistics for each hour (NaN-safe with all-NaN protection)
    hour_stats = []
    for hour_idx, hour in enumerate(hours):
        if utci_by_hour[hour_idx]:
            hour_utci = np.array(utci_by_hour[hour_idx])
            
            # Check if all values are NaN
            valid_values = hour_utci[~np.isnan(hour_utci)]
            if len(valid_values) > 0:
                hour_stats.append({
                    'hour': hour,
                    'min': float(np.nanmin(hour_utci)),
                    'max': float(np.nanmax(hour_utci)),
                    'mean': float(np.nanmean(hour_utci))
                })
            # Skip hours with all NaN values
    
    return hour_stats


# ============================================================================
# Thermal Comfort Classification
# ============================================================================

def classify_thermal_comfort(utci_values: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Classify UTCI values into thermal comfort categories.
    
    Args:
        utci_values: Array of UTCI values in °C
        
    Returns:
        Tuple of (comfort_categories, category_counts)
        - comfort_categories: Array of category strings
        - category_counts: Dictionary mapping category to count
    """
    categories = np.full(utci_values.shape, 'unknown', dtype=object)
    
    # Apply UTCI classification using thresholds
    for category, (min_val, max_val) in UTCI_COMFORT_THRESHOLDS.items():
        if min_val == -float('inf'):
            categories[utci_values < max_val] = category
        elif max_val == float('inf'):
            categories[utci_values >= min_val] = category
        else:
            categories[(utci_values >= min_val) & (utci_values < max_val)] = category
    
    # Count categories
    unique, counts = np.unique(categories, return_counts=True)
    category_counts = dict(zip(unique, counts))
    
    return categories, category_counts


def compute_summary_statistics(utci_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute summary statistics for UTCI results.
    
    Uses shared utility functions for data extraction and NaN-safe calculations.
    
    Args:
        utci_results: Dictionary of UTCI results from compute_utci()
        
    Returns:
        Dictionary with summary statistics including:
        - total_positions, total_hours, valid_utci_values
        - utci_stats: mean, min, max, std
        - mrt_stats: mean, min, max, std
        - comfort_distribution: counts per category
        - position_bounds: spatial extent
    """
    # Use shared utilities for data extraction
    all_utci = extract_all_utci_values(utci_results)
    positions = extract_positions(utci_results)
    
    # Extract MRT values
    all_mrt = []
    for data in utci_results.values():
        all_mrt.extend(data.get('mrt', data.get('mrt0', [])))
    all_mrt = np.array(all_mrt)
    
    # Remove NaN values for statistics
    valid_utci = all_utci[~np.isnan(all_utci)]
    valid_mrt = all_mrt[~np.isnan(all_mrt)]
    
    # Compute comfort categories
    if len(valid_utci) > 0:
        _, comfort_counts = classify_thermal_comfort(valid_utci)
    else:
        comfort_counts = {}
    
    summary = {
        'total_positions': len(utci_results),
        'total_hours': len(all_utci),
        'valid_utci_values': len(valid_utci),
        'utci_stats': {
            'mean': float(np.mean(valid_utci)) if len(valid_utci) > 0 else np.nan,
            'min': float(np.min(valid_utci)) if len(valid_utci) > 0 else np.nan,
            'max': float(np.max(valid_utci)) if len(valid_utci) > 0 else np.nan,
            'std': float(np.std(valid_utci)) if len(valid_utci) > 0 else np.nan
        },
        'mrt_stats': {
            'mean': float(np.mean(valid_mrt)) if len(valid_mrt) > 0 else np.nan,
            'min': float(np.min(valid_mrt)) if len(valid_mrt) > 0 else np.nan,
            'max': float(np.max(valid_mrt)) if len(valid_mrt) > 0 else np.nan,
            'std': float(np.std(valid_mrt)) if len(valid_mrt) > 0 else np.nan
        },
        'comfort_distribution': comfort_counts,
        'position_bounds': {
            'x_min': float(positions[:, 0].min()) if len(positions) > 0 else np.nan,
            'x_max': float(positions[:, 0].max()) if len(positions) > 0 else np.nan,
            'y_min': float(positions[:, 1].min()) if len(positions) > 0 else np.nan,
            'y_max': float(positions[:, 1].max()) if len(positions) > 0 else np.nan,
            'z_min': float(positions[:, 2].min()) if len(positions) > 0 else np.nan,
            'z_max': float(positions[:, 2].max()) if len(positions) > 0 else np.nan
        }
    }
    
    return summary


def print_summary(summary: Dict[str, Any]) -> None:
    """
    Print formatted summary statistics.
    
    Args:
        summary: Summary dictionary from compute_summary_statistics()
    """
    print("\n=== UTCI Calculation Summary ===")
    print(f"Positions: {summary['total_positions']}")
    print(f"Hours: {summary['total_hours']}")
    print(f"UTCI range: {summary['utci_stats']['min']:.1f} to {summary['utci_stats']['max']:.1f} °C")
    print(f"UTCI mean: {summary['utci_stats']['mean']:.1f} °C")
    print(f"MRT range: {summary['mrt_stats']['min']:.1f} to {summary['mrt_stats']['max']:.1f} °C")
    print(f"MRT mean: {summary['mrt_stats']['mean']:.1f} °C")
    print(f"Comfort distribution: {summary['comfort_distribution']}")

