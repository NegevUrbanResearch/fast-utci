### GHX Analysis — Chunk 4 (Lines ~16520–17600)

**Scope**: `LB Human to Sky Relation` component and context geometry inputs.

- **LB Human to Sky Relation (1.8.0)**
  - Computes two key parameters used by `Outdoor Solar MRT`:
    - `fract_body_exp` (time series, 0–1): fraction of human body exposed to direct sun each sun-up hour, via ray intersections from solar vectors against context mesh.
    - `sky_exposure` (scalar 0–1): fraction of visible sky vault using Tregenza dome patches and weighted intersection test.
  - Inputs:
    - `north_`, `_location` (from EPW), `_position` (point(s) at human feet), `_context` (Breps/Meshes), `_pt_count_`, `_height_`, `_cpu_count_`, `_run`.
  - Internals:
    - Builds human vertical line and sample points; generates sun vectors for all 8760 hours via `Sunpath.from_location` and calculates intersections with joined context mesh.
    - For sky exposure, uses `view_sphere.tregenza_dome_vectors` and weights, intersected with context.
  - Outputs are then connected to `Outdoor Solar MRT`’s `fract_body_exp_` and `sky_exposure_`.

- **Context geometry**
  - A hidden `Brep` parameter contains 949 referenced Breps, likely the urban environment.
  - These feed `_context` for occlusion tests.

- **Implications**
  - The GH script accounts for occlusions using ray tests but does not include reflected shortwave from context; ground reflectance is parameterized.
  - For Python parity, we should implement fast ray-geometry intersection for:
    - Sun vectors per hour at each pedestrian sample point.
    - Tregenza sky vectors for view factor to sky.
  - Multi-CPU execution is supported; we should parallelize intersection tests.

Next: scan for analysis period selection, sunpath/direct sun components used for visualization, and any grid generation for multiple positions.


