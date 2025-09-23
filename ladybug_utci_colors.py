"""
Ladybug UTCI Color Scale Implementation

This module provides the standard Ladybug Tools 11-point UTCI color scale
following the official Ladybug documentation and color specifications.
"""

import numpy as np
from typing import List, Tuple, Dict, Any
import plotly.graph_objects as go

# Ladybug UTCI 11-point scale categories and temperature ranges
LADYBUG_UTCI_CATEGORIES = {
    -5: {'range': (-float('inf'), -40), 'label': 'Extreme Cold Stress', 'abbrev': 'Extreme Cold'},
    -4: {'range': (-40, -27), 'label': 'Very Strong Cold Stress', 'abbrev': 'Very Cold'},
    -3: {'range': (-27, -13), 'label': 'Strong Cold Stress', 'abbrev': 'Strong Cold'},
    -2: {'range': (-13, 0), 'label': 'Moderate Cold Stress', 'abbrev': 'Moderate Cold'},
    -1: {'range': (0, 9), 'label': 'Slight Cold Stress', 'abbrev': 'Slight Cold'},
    0: {'range': (9, 26), 'label': 'No Thermal Stress', 'abbrev': 'Comfortable'},
    1: {'range': (26, 28), 'label': 'Slight Heat Stress', 'abbrev': 'Slight Heat'},
    2: {'range': (28, 32), 'label': 'Moderate Heat Stress', 'abbrev': 'Moderate Heat'},
    3: {'range': (32, 38), 'label': 'Strong Heat Stress', 'abbrev': 'Strong Heat'},
    4: {'range': (38, 46), 'label': 'Very Strong Heat Stress', 'abbrev': 'Very Strong Heat'},
    5: {'range': (46, float('inf')), 'label': 'Extreme Heat Stress', 'abbrev': 'Extreme Heat'}
}

# Ladybug "Nuanced" gradient colors (11-point scale)
# Based on official Ladybug Tools Colorset.nuanced() RGB values
# Reference: https://www.ladybug.tools/ladybug/docs/_modules/ladybug/color.html#Colorset
LADYBUG_NUANCED_COLORS = [
    '#313695',  # (49, 54, 149) - Extreme Cold
    '#4575B4',  # (69, 117, 180) - Very Strong Cold
    '#74ADD1',  # (116, 173, 209) - Strong Cold
    '#ABD9E9',  # (171, 217, 233) - Moderate Cold
    '#E0F3F8',  # (224, 243, 248) - Slight Cold
    '#FFFFBF',  # (255, 255, 191) - Comfortable
    '#FEE090',  # (254, 224, 144) - Slight Heat
    '#FDAE61',  # (253, 174, 97) - Moderate Heat
    '#F46D43',  # (244, 109, 67) - Strong Heat
    '#D73027',  # (215, 48, 39) - Very Strong Heat
    '#A50026'   # (165, 0, 38) - Extreme Heat
]

# Create color mapping for each category
LADYBUG_COLOR_MAP = {}
for i, (category, info) in enumerate(LADYBUG_UTCI_CATEGORIES.items()):
    LADYBUG_COLOR_MAP[category] = {
        'color': LADYBUG_NUANCED_COLORS[i],
        'label': info['label'],
        'abbrev': info['abbrev'],
        'range': info['range']
    }


class LadybugUTCIColors:
    """
    Ladybug UTCI color scale implementation with 11-point thermal stress categories.
    
    Follows the official Ladybug Tools specification for UTCI visualization
    using the "nuanced ladybug" gradient.
    """
    
    def __init__(self):
        self.categories = LADYBUG_UTCI_CATEGORIES
        self.colors = LADYBUG_NUANCED_COLORS
        self.color_map = LADYBUG_COLOR_MAP
    
    def get_utci_category(self, utci_value: float) -> int:
        """
        Get the UTCI thermal stress category (-5 to +5) for a given UTCI value.
        
        Args:
            utci_value: UTCI temperature in Celsius
            
        Returns:
            Integer category from -5 (extreme cold) to +5 (extreme heat)
        """
        for category, info in self.categories.items():
            min_val, max_val = info['range']
            if min_val <= utci_value < max_val:
                return category
        return 0  # Default to comfortable if outside range
    
    def get_utci_color(self, utci_value: float) -> str:
        """
        Get the Ladybug color for a UTCI value.
        
        Args:
            utci_value: UTCI temperature in Celsius
            
        Returns:
            Hex color string
        """
        category = self.get_utci_category(utci_value)
        return self.color_map[category]['color']
    
    def get_utci_label(self, utci_value: float) -> str:
        """
        Get the full thermal stress label for a UTCI value.
        
        Args:
            utci_value: UTCI temperature in Celsius
            
        Returns:
            Full label string (e.g., "No Thermal Stress")
        """
        category = self.get_utci_category(utci_value)
        return self.color_map[category]['label']
    
    def get_utci_abbrev(self, utci_value: float) -> str:
        """
        Get the abbreviated thermal stress label for a UTCI value.
        
        Args:
            utci_value: UTCI temperature in Celsius
            
        Returns:
            Abbreviated label string (e.g., "Comfortable")
        """
        category = self.get_utci_category(utci_value)
        return self.color_map[category]['abbrev']
    
    def create_static_colorscale(self, utci_min: float = 0, utci_max: float = 50) -> List[List]:
        """
        Create a static Plotly colorscale using the full 11-point Ladybug spectrum.
        
        Maps the 0-50°C range across all 11 colors from the Ladybug nuanced gradient,
        ensuring the full color spectrum is used even for limited temperature ranges.
        
        Args:
            utci_min: Minimum UTCI value for colorscale (default 0°C)
            utci_max: Maximum UTCI value for colorscale (default 50°C)
            
        Returns:
            Plotly colorscale as list of [position, color] pairs
        """
        colorscale = []
        
        # Use all 11 Ladybug colors, mapping them across the 0-50°C range
        for i in range(11):
            position = i / 10.0  # 0.0, 0.1, 0.2, ..., 1.0
            utci_val = utci_min + (utci_max - utci_min) * position
            color = self.get_utci_color(utci_val)
            colorscale.append([position, color])
        
        return colorscale
    
    def get_colorscale_bounds(self) -> Tuple[float, float]:
        """
        Get the recommended bounds for the UTCI colorscale.
        
        Returns:
            Tuple of (min_utci, max_utci) for the colorscale
        """
        # Focus on relevant UTCI range for better visualization of small differences
        return (0.0, 50.0)
    
    def create_legend_data(self) -> List[Dict[str, Any]]:
        """
        Create legend data for the 11-point UTCI scale.
        
        Returns:
            List of dictionaries with legend information
        """
        legend_data = []
        
        for category in sorted(self.categories.keys()):
            info = self.color_map[category]
            legend_data.append({
                'category': category,
                'color': info['color'],
                'label': info['label'],
                'abbrev': info['abbrev'],
                'range': info['range']
            })
        
        return legend_data
    
    def create_dynamic_colorscale(self, utci_min: float, utci_max: float) -> List[List]:
        """
        Create a dynamic Plotly colorscale using the full color spectrum mapped to actual data range.
        
        Maps the full 11-color Ladybug spectrum across the actual data range (utci_min to utci_max),
        providing maximum contrast for single-hour analysis while using the complete blue-to-red range.
        
        Args:
            utci_min: Minimum UTCI value in the data
            utci_max: Maximum UTCI value in the data
            
        Returns:
            Plotly colorscale as list of [position, color] pairs
        """
        colorscale = []
        
        # Use all 11 Ladybug colors directly, mapping them across the actual data range
        # This provides maximum contrast while using the full color spectrum from blue to red
        for i in range(11):
            position = i / 10.0  # 0.0, 0.1, 0.2, ..., 1.0
            # Use the Ladybug colors directly instead of get_utci_color()
            color = LADYBUG_NUANCED_COLORS[i]
            colorscale.append([position, color])
        
        return colorscale


def create_ladybug_utci_colorscale() -> List[List]:
    """
    Convenience function to create a static Ladybug UTCI colorscale.
    
    Returns:
        Plotly colorscale for UTCI visualization
    """
    utci_colors = LadybugUTCIColors()
    return utci_colors.create_static_colorscale()


def get_utci_color_mapping() -> Dict[int, Dict[str, Any]]:
    """
    Get the complete UTCI color mapping for reference.
    
    Returns:
        Dictionary mapping category numbers to color information
    """
    return LADYBUG_COLOR_MAP.copy()


# Example usage and validation
if __name__ == "__main__":
    # Test the color scale
    utci_colors = LadybugUTCIColors()
    
    # Test values
    test_values = [-45, -30, -15, -5, 5, 15, 25, 30, 35, 40, 50]
    
    print("Ladybug UTCI Color Scale Test:")
    print("=" * 50)
    
    for utci_val in test_values:
        category = utci_colors.get_utci_category(utci_val)
        color = utci_colors.get_utci_color(utci_val)
        label = utci_colors.get_utci_label(utci_val)
        
        print(f"UTCI {utci_val:3.0f}°C -> Category {category:+2d}: {label} ({color})")
    
    print("\nStatic Colorscale created successfully!")
    colorscale = utci_colors.create_static_colorscale()
    print(f"Colorscale has {len(colorscale)} color points")
