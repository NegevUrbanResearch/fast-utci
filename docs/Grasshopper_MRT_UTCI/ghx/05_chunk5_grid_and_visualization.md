### GHX Analysis — Chunk 5 (Lines ~22040–23240)

**Scope**: Point grid generation for analysis surfaces, run toggles, and post-processing/visualization utilities.

- **LB Generate Point Grid (1.8.0)**
  - Generates an analysis mesh and test points from an input `Surface`/`Brep` or `Mesh`.
  - Inputs: `_geometry` (from a hidden `Surface` ref), `_grid_size` (Number Slider set to 10), optional `_offset_dist_`, `quad_only_`.
  - Outputs: `points` (face centroids), `vectors` (face normals), `face_areas`, `mesh`.
  - These points likely feed the `_position` input of `LB Human to Sky Relation` for many locations across a surface.

- **Analysis period filtering**
  - Additional `LB Apply Analysis Period` instances filter `fract_body_exp` before deconstruction, consistent with applying analysis windows to exposure results.

- **Deconstruct/average pipeline**
  - `LB Deconstruct Data` extracts `values` from filtered `fract_body_exp`.
  - A Grasshopper `Average` component computes the arithmetic mean; result shown in a `Panel`. This is likely for QA or summary metrics.

- **Run toggles**
  - Separate toggles for running Human-to-Sky and Outdoor Solar MRT allow staged execution.

- **Implications**
  - The GH workflow supports both single-point and grid-based analysis by connecting generated points as positions for exposure calculation.
  - Our Python tool should support:
    - Generating sampling grids from polygonal surfaces/meshes.
    - Batch computing exposure and MRT over many points with parallelization.
    - Optional period filtering and simple statistics (mean) over time series.

Next: finalize end-to-end GH data flow and produce the MRT calculator design aligned for parity and performance.


