# fast_utci Legacy Python / Ladybug Package

`fast_utci` is the legacy Python/Ladybug CPU pipeline for UTCI/SAI reference calculations, old artifact reproduction, exports, parity checks, and GIS support scripts. It was an intermediate improvement over the original Grasshopper/Ladybug workflow by moving analysis into Python with Embree-backed ray tracing and parallel CPU execution.

The main path is now the interactive WebGPU/Three.js application in `../../viewer/`, which is the preferred route for production-scale analysis and review. Use this package only for maintaining legacy scripts, generating reference/parity data, or producing GIS/support artifacts that still depend on the Ladybug-backed CPU pipeline.

This pathway keeps the project tied to its Ladybug source: it uses `ladybug` / `lbt-ladybug` for EPW weather data, sun paths, and sky geometry; `ladybug-comfort` for Outdoor SolarCal MRT; and `pythermalcomfort` for UTCI.

## What It Does

```text
3D model + weather
  -> CPU ray tracing / exposure
  -> Ladybug sun / sky / SolarCal MRT
  -> pythermalcomfort UTCI
  -> CSV / JSON / .bin outputs
```

The package also includes the reference implementation for the shade metric presented by the app as Shading Availability Index (SAI). The implementation module is named `shading_index.py` for compatibility with existing data and tests.

## Legacy Reference Example

The example below documents the Python/Ladybug reference path. For new production analysis, prefer the WebGPU/Three.js viewer path unless you specifically need parity data or a legacy export.

```python
from ladybug.epw import EPW
import numpy as np

from fast_utci import MRTCalculator, UTCICalculator
from fast_utci.mrt import create_analysis_period, create_rectangular_grid
from fast_utci.mrt.shading_index import calculate_shading_index
from fast_utci.shared import load_config
from fast_utci.shared.io import read_project_data, get_combined_mesh, get_ground_bounds

cfg = load_config()

scene, _, _ = read_project_data("buildings.glb", "weather.epw")
model = get_combined_mesh(scene)

mrt_calc = MRTCalculator(context_meshes=[model], config=cfg.mrt)
mrt_calc.set_location_from_epw("weather.epw")

model_bounds = get_ground_bounds(scene)
grid = create_rectangular_grid(
    bounds_min=model_bounds[0][:2],
    bounds_max=model_bounds[1][:2],
    grid_size=2.0,
    z_height=1.5,
)

period = create_analysis_period(
    start_month=8,
    start_day=15,
    start_hour=0,
    end_month=8,
    end_day=15,
    end_hour=23,
)

epw = EPW("weather.epw")
exposure_results = mrt_calc.compute_exposure(
    positions=grid.points,
    analysis_period=period,
)
mrt_results = mrt_calc.compute_mrt(epw, exposure_results, period)

utci_calc = UTCICalculator(weather_data="weather.epw", config=cfg.utci)
utci_results = utci_calc.compute_utci(mrt_results)

sun_data = mrt_calc.get_sun_data(period)
sai = calculate_shading_index(exposure_results, sun_data)
print(f"SAI range: {np.min(sai):.3f} to {np.max(sai):.3f}")

utci_calc.to_csv(utci_results, "utci_results.csv")
summary = utci_calc.summary_statistics(utci_results)
print(f"UTCI range: {summary['utci_stats']['min']:.1f} to {summary['utci_stats']['max']:.1f} C")
```

## Modules

- `fast_utci.mrt`: CPU exposure, ray tracing, Ladybug sun/sky helpers, Ladybug SolarCal MRT, and SAI reference logic.
- `fast_utci.utci`: UTCI calculation, boundary averaging, classification, and export, using MRT/weather inputs from the Ladybug-backed path.
- `fast_utci.shared`: TOML config loading, parallel utilities, Ladybug EPW weather adapters, and shared helpers.
- `fast_utci.innovation_district_gis`: GIS postprocessing and validation for viewer collector output.

## Configuration

Load configuration from the repo-level `fast_utci.toml`:

```python
from fast_utci.shared import load_config

cfg = load_config()
mrt_config = cfg.mrt
utci_config = cfg.utci
```

Pass `config=cfg.mrt` or `config=cfg.utci` explicitly when creating calculators.

## Legacy Reference Commands

Export Ben-Gurion reference intermediates:

```powershell
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe scripts/export_ben_gurion_intermediates.py --base-path data/analyses/Ben-Gurion/20250815_grid_2m_fullday --model data/3d_models/Ben-Gurion/original_with_layers.glb
```

Run focused Python validation for legacy/reference code:

```powershell
python -m pytest tests/mrt/test_shading_index.py tests/test_export_ben_gurion_intermediates.py tests/test_innovation_district_gis_raw.py tests/test_innovation_district_gis_qa_manifest.py tests/test_postprocess_innovation_district_gis.py
```
