from .manifest import PostprocessOutputs
from .raw import ActiveCellArtifacts, load_active_cell_artifacts
from .transforms import epsg2039_to_wgs84

__all__ = [
    "ActiveCellArtifacts",
    "PostprocessOutputs",
    "epsg2039_to_wgs84",
    "load_active_cell_artifacts",
    "postprocess_active_cells",
]


def __getattr__(name: str):
    if name == "postprocess_active_cells":
        from .orchestrator import postprocess_active_cells

        return postprocess_active_cells
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
