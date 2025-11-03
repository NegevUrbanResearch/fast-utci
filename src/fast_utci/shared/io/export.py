"""
Export functionality for fast-utci results.

Provides unified functions for exporting MRT and UTCI calculation results
to CSV and JSON formats.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def export_mrt_results(results: Dict[str, Any],
                      csv_path: str,
                      grasshopper_format: bool = False,
                      csv_encoding: str = 'utf-8',
                      csv_index: bool = False) -> None:
    """
    Export MRT results to CSV file.
    
    Args:
        results: Results dictionary from MRTCalculator.compute_mrt()
        csv_path: Path for CSV file
        grasshopper_format: If True, use Grasshopper-compatible format for validation
        csv_encoding: CSV file encoding
        csv_index: Whether to include row indices in CSV
    """
    if grasshopper_format:
        _export_mrt_grasshopper_csv(results, csv_path)
    else:
        _export_mrt_standard_csv(results, csv_path, csv_encoding, csv_index)


def _export_mrt_grasshopper_csv(results: Dict[str, Any], csv_path: str):
    """Export in Grasshopper validation format."""
    rows = []
    
    for pos_key, pos_data in results.items():
        # Extract position index from key (e.g., 'position_0' -> 0)
        try:
            pos_idx = int(pos_key.split('_')[1])
        except (IndexError, ValueError):
            pos_idx = 0
        
        mrt_values = pos_data['mrt']
        
        for hour_idx, mrt in enumerate(mrt_values):
            # Grid index format: pixel10*10
            pixel_id = pos_idx
            
            # Duplicate MRT in both columns for GH format
            mrt_0 = mrt
            mrt_1 = mrt
            
            # Placeholder values
            utci = 30.0  # Placeholder - compute separately
            color = "255,255,255"  # Placeholder
            
            rows.append([pixel_id, mrt_0, mrt_1, utci, color])
    
    # Create DataFrame and export
    df = pd.DataFrame(rows, columns=['pixel10*10', 'mrt 0', 'mrt 1', 'utci', 'color'])
    df.to_csv(csv_path, index=False)
    logger.info(f"Exported Grasshopper format CSV: {csv_path}")


def _export_mrt_standard_csv(results: Dict[str, Any], csv_path: str, csv_encoding: str, csv_index: bool):
    """Export in standard detailed format."""
    rows = []
    
    for pos_key, pos_data in results.items():
        position = pos_data['position']
        mrt_values = pos_data['mrt']
        fract_exp = pos_data['fract_body_exp']
        sky_exposure = pos_data['sky_exposure']
        
        # Create rows for each timestep
        for i, (mrt, fexp) in enumerate(zip(mrt_values, fract_exp)):
            rows.append({
                'position_id': pos_key,
                'x': position[0],
                'y': position[1], 
                'z': position[2],
                'hour': i,  # Simplified - no datetime handling
                'mrt': mrt,
                'fract_body_exp': fexp,
                'sky_exposure': sky_exposure,
                'short_erf': pos_data['short_erf'][i],
                'long_erf': pos_data['long_erf'][i],
                'short_dmrt': pos_data['short_dmrt'][i],
                'long_dmrt': pos_data['long_dmrt'][i]
            })
    
    # Export to CSV
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=csv_index, encoding=csv_encoding)
    logger.info(f"Exported CSV: {csv_path}")


def export_utci_results(utci_results: Dict[str, Any],
                       csv_path: str,
                       include_weather: bool = True,
                       include_comfort_categories: bool = True,
                       csv_encoding: str = 'utf-8',
                       csv_index: bool = False) -> None:
    """
    Export UTCI results to CSV file.
    
    Args:
        utci_results: Dictionary from UTCICalculator.compute_utci()
        csv_path: Output CSV file path
        include_weather: Whether to include weather variables (air_temp, wind_speed, RH)
        include_comfort_categories: Whether to include thermal comfort categories
        csv_encoding: CSV file encoding
        csv_index: Whether to include row indices in CSV
    """
    from fast_utci.utci.statistics import classify_thermal_comfort
    
    rows = []
    
    for pos_key, data in utci_results.items():
        position = data['position']
        utci_vals = data['utci']
        mrt_vals = data.get('mrt', data.get('mrt0'))
        
        # Get comfort categories if requested
        if include_comfort_categories:
            comfort_categories, _ = classify_thermal_comfort(utci_vals)
        
        for i, (utci_val, mrt_val) in enumerate(zip(utci_vals, mrt_vals)):
            row = {
                'position_id': pos_key,
                'x': position[0],
                'y': position[1],
                'z': position[2],
                'hour': i,
                'utci': utci_val,
                'mrt': mrt_val
            }
            
            # Add datetime if available
            if data.get('datetime') is not None and i < len(data['datetime']):
                dt = data['datetime'][i]
                row['datetime'] = dt
                if hasattr(dt, 'hour'):
                    row['hour'] = dt.hour
            
            # Add weather data if available and requested
            if include_weather:
                if 'air_temp' in data and i < len(data['air_temp']):
                    row['air_temp'] = data['air_temp'][i]
                if 'wind_speed' in data and i < len(data['wind_speed']):
                    row['wind_speed'] = data['wind_speed'][i]
                if 'relative_humidity' in data and i < len(data['relative_humidity']):
                    row['relative_humidity'] = data['relative_humidity'][i]
            
            # Add comfort category if requested
            if include_comfort_categories:
                row['comfort_category'] = comfort_categories[i]
            
            rows.append(row)
    
    # Create DataFrame and export
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=csv_index, encoding=csv_encoding)
    
    logger.info(f"Exported UTCI results to: {csv_path}")
    logger.info(f"Records: {len(df)}, Positions: {len(utci_results)}")
    
    if include_comfort_categories and len(df) > 0:
        comfort_summary = df['comfort_category'].value_counts()
        logger.info(f"Comfort distribution: {dict(comfort_summary)}")


def export_utci_results_json(utci_results: Dict[str, Any],
                             json_path: str,
                             include_numpy_arrays: bool = False) -> None:
    """
    Export UTCI results to JSON file.
    
    Args:
        utci_results: Dictionary from UTCICalculator.compute_utci()
        json_path: Output JSON file path
        include_numpy_arrays: Whether to convert numpy arrays to lists
    """
    import json
    
    # Convert numpy arrays to lists if requested
    if include_numpy_arrays:
        export_data = {}
        for pos_key, data in utci_results.items():
            export_data[pos_key] = {}
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    export_data[pos_key][key] = value.tolist()
                elif isinstance(value, (tuple, list)):
                    export_data[pos_key][key] = list(value)
                else:
                    export_data[pos_key][key] = value
    else:
        export_data = utci_results
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, default=str)
    
    logger.info(f"Exported UTCI results to: {json_path}")
    logger.info(f"Positions: {len(utci_results)}")

