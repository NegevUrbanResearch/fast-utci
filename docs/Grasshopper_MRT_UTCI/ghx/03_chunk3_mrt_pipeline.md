### GHX Analysis — Chunk 3 (Around Lines 15340–16600)

**Scope**: MRT computation chain using Ladybug’s SolarCal-based `LB Outdoor Solar MRT`, plus data preparation via `LB Apply Analysis Period` and `LB Deconstruct Data`.

- **MRT component: `LB Outdoor Solar MRT` (1.8.0)**
  - Purpose: Computes MRT combining shortwave (SolarCal) and longwave sky exchange.
  - Inputs:
    - `_location`: from EPW `location`.
    - `_surface_temp`: surface temperature proxy (often EPW dry bulb) for longwave exchange.
    - `_dir_norm_rad`, `_diff_horiz_rad`, `_horiz_infrared`: EPW radiative fluxes (W/m²).
    - `fract_body_exp_`: fraction of body exposed to direct sun (0–1), from `LB Human to Sky Relation` or scalar.
    - `sky_exposure_`: fraction of sky vault visible (0–1), from `LB Human to Sky Relation` or scalar.
    - `_ground_ref_`: ground reflectance (default 0.25) optionally provided.
    - `_solar_body_par_`: optional Solar Body Parameters (skin/clothing absorptivity, SHARP angle).
    - `_run`: boolean.
  - Outputs:
    - `short_erf`, `long_erf` (W/m²), `short_dmrt`, `long_dmrt` (°C), and final `mrt` (°C).
  - Implementation uses `ladybug_comfort.collection.solarcal.OutdoorSolarCal`.

- **Data preparation**:
  - Multiple `LB Apply Analysis Period` components filter EPW series (`direct_normal_rad`, `diffuse_horizontal_rad`, `horizontal_infrared_rad`) to a selected period, then feed MRT inputs.
  - `LB Deconstruct Data` is used downstream of `UTCI` results for header/values extraction (likely for visualization/export).

- **Connectivity observed (by GUID linkages)**:
  - `_location` ← `LB Import EPW.location`.
  - `_surface_temp` ← EPW `dry_bulb_temperature` (used as proxy for surrounding surface temps).
  - `_dir_norm_rad`/`_diff_horiz_rad`/`_horiz_infrared` ← EPW radiative series after `LB Apply Analysis Period`.
  - `fract_body_exp_` and `sky_exposure_` are connected to upstream components (likely `LB Human to Sky Relation`).

- **Interpretation**:
  - The GH script computes MRT via SolarCal plus sky exposure, not through full Radiance raytracing. Geometry effects enter through `Human to Sky Relation` scalars (body and sky visibility) rather than detailed shortwave reflections.
  - Therefore, replicating results in Python should rely on the same SolarCal formulation and exposure fractions, matched per timestep.

- **Implications for Python MRT module**:
  - Implement SolarCal using `ladybug-comfort` APIs or equivalent formulas to maintain parity.
  - Accept time-series inputs for `dir_norm`, `diff_horiz`, `horiz_infrared`, `surface_temp`, and scalar or series for exposure fractions.
  - Provide seamless filtering by analysis periods and ensure timestamp alignment with EPW.

Next: identify `LB Human to Sky Relation`, `LB SunPath`, and grid/point generation to see how exposure fractions and analysis points are computed.


