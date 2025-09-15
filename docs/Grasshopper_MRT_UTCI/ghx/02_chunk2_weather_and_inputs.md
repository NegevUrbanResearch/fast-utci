### GHX Analysis — Chunk 2 (Lines 401–1500)

**Scope**: Completion of `LB UTCI Comfort` IO wiring and the `LB Import EPW` component with its upstream `File Path`, plus `_run` toggle and a display `Panel`.

- **UTCI component IO wiring**:
  - Inputs link to sources:
    - `_air_temp` source GUID `6c4437bd-...` — likely EPW dry bulb.
    - `_mrt_` source GUID `d54ecbcb-...` — computed elsewhere (MRT pipeline, not yet seen here).
    - `_rel_humid` source GUID `c090f5ff-...` — EPW relative humidity.
    - `_wind_vel_` source GUID `c9f7f685-...` — EPW wind speed or adjusted value.
    - `utci_par_` — unconnected; default parameters used unless present.
    - `_run` source GUID `15b9e0f8-...` — Boolean Toggle.
  - Outputs: `utci`, `comfort`, `condition`, `category`, `comf_obj`, `out` available (likely wired to panels/geometry later).

- **Weather import (LB Import EPW 1.8.0)**:
  - Reads `_epw_file` and exposes multiple `DataCollection` series:
    - Key fields for UTCI: `dry_bulb_temperature`, `relative_humidity`, `wind_speed` (10 m met wind), `wind_direction`.
    - Solar: `direct_normal_rad`, `diffuse_horizontal_rad`, `global_horizontal_rad`, `horizontal_infrared_rad`.
    - Metadata: `location`, `barometric_pressure`, `model_year`, `total_sky_cover`, `ground_temperature` series.
  - Hidden on canvas; suggests scripted/automated workflow.

- **EPW file path**:
  - `File Path` component labeled `Weather .epw File` contains a persistent value:
    - `C:\Users\tuval\Downloads\ISR_Beer.Sheva.401900_MSI.epw`
  - This feeds `_epw_file` of `LB Import EPW`.

- **Run control**:
  - `Boolean Toggle` is present and connected to `_run` of UTCI component; initial value `false`.

- **Implications for MRT pipeline**:
  - MRT must be computed externally to EPW using geometry and solar; EPW provides radiative fluxes and atmospheric state but not view-dependent MRT.
  - We should align our MRT computation with EPW timestep indexing and timezone/location from `location` to ensure solar position parity with Ladybug.
  - Wind speed from EPW is meteorological at 10 m. If we simulate near-ground wind, convert to 10 m via ×1.5 for UTCI, or inversely adjust EPW if mapping to pedestrian-level fields.
  - Relative humidity and air temperature will be direct EPW series passed through; our outputs must share datatree structure or timestamps.

Next steps: identify MRT generation chain (Ladybug/Honeybee Radiance or custom GhPython) and sampling of analysis points/meshes.


