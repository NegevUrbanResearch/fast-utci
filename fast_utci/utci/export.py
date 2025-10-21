"""
Export functionality for UTCI results.

Provides functions for exporting UTCI calculation results to various formats
(CSV, JSON, etc.) for analysis and visualization.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from pathlib import Path

from .statistics import classify_thermal_comfort


def to_csv(utci_results: Dict[str, Any],
          csv_path: str,
          include_weather: bool = True,
          include_comfort_categories: bool = True,
          csv_encoding: str = 'utf-8',
          csv_index: bool = False) -> None:
    """
    Export UTCI results to CSV file.
    
    Args:
        utci_results: Dictionary from compute_utci()
        csv_path: Output CSV file path
        include_weather: Whether to include weather variables (air_temp, wind_speed, RH)
        include_comfort_categories: Whether to include thermal comfort categories
        csv_encoding: CSV file encoding
        csv_index: Whether to include row indices in CSV
    """
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
    
    print(f"Exported UTCI results to: {csv_path}")
    print(f"  Records: {len(df)}")
    print(f"  Positions: {len(utci_results)}")
    
    if include_comfort_categories and len(df) > 0:
        comfort_summary = df['comfort_category'].value_counts()
        print(f"  Comfort distribution: {dict(comfort_summary)}")


def to_json(utci_results: Dict[str, Any],
           json_path: str,
           include_numpy_arrays: bool = False) -> None:
    """
    Export UTCI results to JSON file.
    
    Args:
        utci_results: Dictionary from compute_utci()
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
    
    print(f"Exported UTCI results to: {json_path}")
    print(f"  Positions: {len(utci_results)}")

