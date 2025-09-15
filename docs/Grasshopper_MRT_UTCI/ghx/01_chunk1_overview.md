### GHX Analysis — Chunk 1 (Lines 1–400)

**Scope**: File header, document metadata, libraries, and the `LB UTCI Comfort` component (GhPython script) metadata and core logic signature.

- **Grasshopper Document**:
  - Name: `1_15th of month.ghx`
  - GH version: 8.22.25217.12451; plugin version 1.8.0 present on `LB UTCI Comfort` component.
  - Units system: `SI`; K3D settings indicate `UnitLength=auto`.

- **Libraries referenced**:
  - Core Grasshopper library entries (duplicated records).
  - `GhPython` present (Version=8.22.25217.12451), implying scriptable Python components.

- **Key Component Identified**: `LB UTCI Comfort` (GhPython-based Ladybug component)
  - Category: `Ladybug > 1 :: Analyze Data`
  - Version message: `1.8.0`
  - Hidden in canvas (Hidden=true), likely wrapped or used as a black-box step.
  - Inputs per docstring:
    - `_air_temp` (°C)
    - `_mrt_` (°C)
    - `_rel_humid` (%)
    - `_wind_vel_` (m/s at 10 m AGL, note: original UTCI expects 10 m; 10 m wind ≈ 1.5 × near-ground speed)
    - `utci_par_` (optional parameters: default comfort range 9–26 °C)
    - `_run` (boolean)
  - Outputs per docstring:
    - `report`, `utci`, `comfort` (0/1 for stress), `condition` (-1/0/+1), `category` (-5..+5), `comf_obj` (full object)
  - Imports:
    - `ladybug.datatype.temperature.Temperature`
    - `ladybug.datacollection.BaseCollection`
    - `ladybug_comfort.collection.utci.UTCI`, `ladybug_comfort.parameter.utci.UTCIParameter`, `ladybug_comfort.utci.universal_thermal_climate_index`
    - `ladybug_rhino.grasshopper.all_required_inputs`
  - Input handling pattern:
    - If all are scalar values: calls `universal_thermal_climate_index(ta, mrt, vel10m, rh)` and computes categories via `UTCIParameter()`.
    - If any are `DataCollection`: aligns `_air_temp` collection; builds `UTCI(_air_temp, _rel_humid, _mrt_, _wind_vel_, utci_par)` and accesses computed properties.
    - Default `_wind_vel_` is 0.5 m/s if not supplied; doc suggests not below 0.5 m/s.
  - Notable assumptions:
    - Meteorological wind at 10 m expected (convert from near-ground via ×1.5 if needed).
    - MRT defaults to air temp if not provided.

- **Implications for our MRT calculator**:
  - Our MRT module must output MRT in °C aligned temporally and spatially with air temperature and humidity inputs.
  - Velocity input should either be at 10 m AGL or we must apply the 1.5 factor to convert ground-level speed to 10 m per UTCI assumptions.
  - Support both scalar and time-series (`DataCollection`) patterns and ensure alignment routines if we model series.
  - Keep outputs compatible with Ladybug/LB-Comfort downstream usage when possible.

No Radiance/Honeybee configuration appears in this chunk; those likely appear later in the GHX.


