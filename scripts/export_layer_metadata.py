"""
Export model layer metadata to JSON for web viewer.

This script extracts layer information from the model_reader.py
MATERIAL_TYPE_MAPPING and exports it to JSON format.
"""

import json
from pathlib import Path
import sys

# Add parent directory to path to import fast_utci
sys.path.insert(0, str(Path(__file__).parent.parent))

from fast_utci.model_reader import MATERIAL_TYPE_MAPPING


def export_layer_metadata(output_path: Path) -> None:
    """
    Export layer metadata to JSON.
    
    Args:
        output_path: Path to output JSON file
    """
    layers = []
    
    for material_type, material_info in MATERIAL_TYPE_MAPPING.items():
        layer_data = {
            "type": material_type,
            "color": material_info['color'],
            "opacity": material_info['opacity'],
            "name": material_info['name']
        }
        layers.append(layer_data)
    
    metadata = {
        "layers": layers,
        "description": "Material layer definitions for 3D model rendering",
        "version": "1.0"
    }
    
    # Write JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"[SAVE] Layer metadata: {output_path}")
    print(f"[INFO] Exported {len(layers)} layer types")


def main():
    """Export layer metadata."""
    print("=" * 60)
    print("EXPORT LAYER METADATA")
    print("=" * 60)
    
    output_path = Path("data/models/model_layers.json")
    export_layer_metadata(output_path)
    
    print(f"\n[OK] Layer metadata exported successfully!")
    print(f"  File: {output_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())
