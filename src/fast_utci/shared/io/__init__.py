"""
I/O operations for fast-utci.

This package provides all file input/output operations including:
- Model loading (GLB/GLTF files)
- Scene graph operations
- Result export (CSV/JSON)
"""

from .glb import load_glb_safe, _is_binary_gltf
from .scene import get_ground_bounds, get_combined_mesh, get_layer_name_from_scene_graph
from .export import export_mrt_results, export_utci_results, export_utci_results_json

__all__ = [
    "load_glb_safe",
    "_is_binary_gltf",
    "get_ground_bounds",
    "get_combined_mesh",
    "get_layer_name_from_scene_graph",
    "read_project_data",
    "export_mrt_results",
    "export_utci_results",
    "export_utci_results_json",
]


def __getattr__(name):
    """Defer model/weather loading when callers need only mesh or export helpers."""
    if name == "read_project_data":
        from .model import read_project_data

        globals()[name] = read_project_data
        return read_project_data
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

