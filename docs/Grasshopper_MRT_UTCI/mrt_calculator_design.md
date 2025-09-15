### MRT Calculator Design (Parity with Grasshopper Script, optimized for speed)

**Goal**: Reproduce the Grasshopper pipeline’s MRT used for UTCI with equivalent results but faster, modular Python. GH pipeline uses Ladybug’s SolarCal-based `OutdoorSolarMRT` plus `Human to Sky Relation` (sun occlusion and sky exposure) and EPW radiative series. No Radiance raytracing is used; occlusion is computed via ray tests against context meshes.

## Inputs
- **Weather/EPW**:
  - `location` (lat, lon, time zone)
  - `dry_bulb_temperature` (°C) [proxy for `_surface_temp`]
  - `direct_normal_rad` (W/m²)
  - `diffuse_horizontal_rad` (W/m²)
  - `horizontal_infrared_rad` (W/m²)
- **Geometry & sampling**:
  - Context geometry: meshes or triangle soups (city/buildings/trees)
  - Analysis surfaces or polygons to generate sampling points (grid size param)
  - Optional explicit `positions` list
- **Exposure parameters**:
  - `pt_count` per human vertical (default 1)
  - `height` (default 1.8 m)
  - `north` degrees or vector (default 0)
  - `ground_reflectance` (default 0.25)
  - Optional Solar Body Parameters (skin/clothing absorptivity, SHARP)
- **Time window**:
  - AnalysisPeriod-like filter (start/end month/day/hour)

## Outputs
- For each position (or grid face center):
  - `mrt` time series (°C), plus `short_erf`, `long_erf`, `short_dmrt`, `long_dmrt`
  - Optional aggregates (mean, percentiles) per position
  - Optional exposure diagnostics: `fract_body_exp` series, `sky_exposure` scalar
  - CSV export for 1:1 comparison with Grasshopper CSV (`data/15th_Aug_MRT.csv`):
    - Columns (exact header): `Hour`, `pixel10*10`, `mrt 0`, `mrt 1`, `utci`, `color`.
    - `Hour`: for testing, use fixed `13-14` (we will filter to local hour 13).
    - `pixel10*10`: integer grid index (row order) to match GH.
    - `mrt 0`, `mrt 1`: duplicate computed MRT (°C) into both columns to match GH format.
    - `utci`: placeholder (e.g., empty or -999) until MRT parity is validated; we will compute UTCI later.
    - `color`: numeric color code from GH; optional in our output. If not reproducing GH color mapping, write a placeholder (e.g., NA) or implement a simple mapping later.

## Architecture (modules and location)
- Keep existing EPW reader: `reader.py` (no new EPW loader).
- All MRT-related code under `MRT/` with `mrt_calculator.py` as the orchestrator:
  - `MRT/mrt_calculator.py`: orchestrates EPW → exposure → SolarCal → outputs; supports single points and grids; parallel where sensible.
  - `MRT/solar.py`: Ladybug sunpath helpers to compute sun-up mask and sun vectors (HOYs), apply `north`.
  - `MRT/mesh.py`: context mesh ingestion; BVH/KD-tree acceleration (pyembree/trimesh.ray, rtree fallback).
  - `MRT/exposure.py`:
    - Build human vertical sample points per position.
    - Compute `fract_body_exp` (ray tests along sun vectors) and `sky_exposure` (Tregenza dome with weights).
  - `MRT/solarcal.py`:
    - Call Ladybug OutdoorSolarCal-equivalent routines to compute `short_erf`, `long_erf`, `short_dmrt`, `long_dmrt`, and `mrt`.
  - `MRT/grid.py`: generate grid points and normals over surfaces/meshes (grid size, offset).
  - `MRT/period.py`: analysis-period filtering utilities.

## Algorithms & Parity Considerations
- **Sun vectors**: Use Ladybug `sunpath` (parity with GH) to generate HOY sun positions/vectors; apply `north` rotation; compute `sun.is_during_day` mask.
- **fract_body_exp (time series)**:
  - For each position, for each sun-up HOY, cast rays from each human sample point along reversed sun vector; mark hit/miss against context mesh via BVH.
  - Fraction per HOY = visible_points / pt_count; assign 0 when sun-down.
- **sky_exposure (scalar)**:
  - Use Tregenza dome 145 patch vectors; intersect once per point; weight by patch weights; average across points.
- **SolarCal MRT**:
  - Follow Ladybug’s OutdoorSolarCal equations to combine:
    - Shortwave ERF from `dir_norm`, `diff_horiz`, `ground_reflectance`, body params, and `fract_body_exp`.
    - Longwave ERF from `horizontal_infrared_rad`, `surface_temp`, and `sky_exposure`.
  - Convert ERF deltas to `dmrt` and final `mrt` in °C; ensure unit consistency.
- **Time alignment**:
  - All arrays indexed by HOY. Apply period filter early; keep consistent lengths between weather and exposure series.

## Performance Strategy
- Ray casting:
  - Pre-build one BVH per context mesh set.
  - Vectorize ray batches per HOY and per point; use numba and/or pyembree if available.
  - Parallelize across positions: multiprocessing with shared read-only BVH; chunk HOYs to balance load.
- Memory:
  - Avoid storing full 8760×num_points matrices when unnecessary; stream compute per chunk and write partial results.
- Caching:
  - Cache sun vectors per location/north.
  - Cache Tregenza vectors/weights.
  - Optional on-disk cache for exposure matrices.

## Public API (Python)
```python
class MRTCalculator:
    def __init__(self, context_meshes, location, north=0.0,
                 ground_reflectance=0.25, body_params=None, cpu_count=None):
        ...

    def compute_exposure(self, positions, pt_count=1, height_m=1.8,
                          analysis_period=None) -> tuple[np.ndarray, float]:
        """Returns (fract_body_exp_series, sky_exposure_scalar) per position."""

    def compute_mrt(self, epw_data, exposure, analysis_period=None) -> dict:
        """Returns dict with time series arrays: mrt, short_erf, long_erf, short_dmrt, long_dmrt."""

    def run_grid(self, surface_or_mesh, grid_size, offset=0.0, **kwargs) -> dict:
        """Generates positions from surface, computes exposure+MRT for all points, returns per-point results."""

    def to_csv(self, results: dict, timeseries_path: str, summary_path: str | None = None,
               long_format: bool = True) -> None:
        """Write results to CSV files for GH comparison.
        - timeseries CSV: one row per (position_id, datetime/HOY) with MRT, ERFs, deltas, exposure, and key EPW fields.
        - summary CSV (optional): per-position statistics and sky_exposure.
        """
```

`epw_data` structure (from existing `reader.py` outputs or EPW object):
```python
class EPWData:
    location: Location
    dry_bulb: np.ndarray  # °C
    dir_norm: np.ndarray  # W/m²
    diff_horiz: np.ndarray  # W/m²
    horiz_ir: np.ndarray  # W/m²
    timestamps: pd.DatetimeIndex  # length 8760
```

## Validation Plan (match GH results)
- Unit tests on a small scene and on user-supplied EPW `ISR_Beer.Sheva.401900_MSI.epw`:
  - Compare `fract_body_exp` series against GH output at several positions and HOYs.
  - Compare `sky_exposure` scalar within 1e-3.
  - Compare `mrt` series within tolerance (e.g., RMSE < 0.2 °C) across a week and the analysis period.
  - For initial dev/testing, restrict to local hour 13–14 (i.e., HOYs where local hour == 13) to speed runs.

## Implementation Notes
- Dependencies: `numpy`, `pandas`, `trimesh` (+ `pyembree` optional), `numba` optional, `ladybug` (sunpath, comfort.solarcal), `shapely` optional for preprocessing.
- Keep functions pure and well-typed; document inputs/outputs; avoid hidden globals.
- Provide CLI for batch run (read EPW, read OBJ/GLB/mesh, surface sampling, write CSV/Parquet).
  - CLI flags: `--timeseries-csv`, `--summary-csv`, `--hour 13` (for testing window), `--grid-size`, `--offset`, `--north`, `--pt-count`, `--height`.

## Tooling choices (clarification)
- Use Ladybug libraries for parity with the GH definition: sunpath, viewsphere, and OutdoorSolarCal. No Honeybee-Radiance.
- We are not using Radiance raytracing; occlusion is computed via fast ray tests against the context mesh (as in GH Human-to-Sky).
- For UTCI, ensure wind is at 10 m (convert near-ground by ×1.5 if needed) and enforce minimum 0.5 m/s as in GH docs.


