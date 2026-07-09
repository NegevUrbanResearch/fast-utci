# fast-utci

fast-utci is an interactive WebGPU application for urban thermal-comfort and shade-availability analysis from 3D city models.

The app loads GLB models, weather inputs, and analysis metadata from `data/`, computes UTCI and Shading Availability Index (SAI) in the browser, and renders the result in an interactive 3D viewer.

Live demo: [3D UTCI Viewer](https://negevurbanresearch.github.io/fast-utci/)

## Project Status

The current product path is the WebGPU viewer. New projects should be added as GLB + weather + metadata and analyzed through the viewer route (`/`).

The project grew through three steps:

1. Ladybug/Grasshopper gave us the original environmental-analysis baseline.
2. The Python/Ladybug pathway improved that baseline by moving the workflow into code, using Embree-accelerated ray tracing and parallel CPU processing.
3. The current WebGPU/Three.js pathway moves the heavy exposure, MRT, UTCI, and SAI work onto the GPU in the browser. This is the main route because it is dramatically faster and gives an interactive workflow instead of a precompute-first workflow.

The Python/Ladybug code remains in the repo for provenance, parity checks, old artifact reproduction, and support utilities. It is not the recommended path for new high-throughput analysis when the WebGPU route can run the project.

## What It Shows

### UTCI

UTCI (Universal Thermal Climate Index) estimates how hot or cold it feels outdoors. It combines air temperature, wind speed, humidity, and Mean Radiant Temperature (MRT).

### Shading Availability Index (SAI)

SAI measures how often each point is shaded during sun-up hours. fast-utci uses SAI as the public shade metric, aligned with the Derech Tzel shade-availability methodology.

Methodology source: [Derech Tzel Shading Metrics Guide](https://tzel.org.il/wp-content/uploads/2025/02/Shading-Metrics-Guide.pdf)

Current viewer buckets:

- `0.0-0.5`: poor shade availability
- `0.5-0.7`: acceptable shade availability
- `0.7-0.9`: good shade availability
- `0.9-1.0`: excellent shade availability

The code and data schema may still use `shading_index` as an internal field name for SAI.

## How It Works

```text
GLB model + weather + analysis metadata
  -> WebGPU exposure calculation
  -> WebGPU MRT / UTCI / SAI
  -> GPU-backed interactive 3D visualization
```

The main route (`/`) is the normal app. The debug route (`/debug`) is for parity checks, collectors, `.bin` comparison, and diagnostics.

The viewer can still load precomputed `.bin` analyses for debug, parity, and legacy compatibility, but new projects should use live WebGPU metadata and the main WebGPU route.

## Quick Start

```powershell
cd viewer
npm install
npm run dev
```

Open the local URL printed by Vite, usually `http://localhost:5173`.

Useful checks:

```powershell
npm run check
npm test
npm run build
```

## Data

Project data lives under `data/`:

- `data/3d_models/`: GLB project models and georeference sidecars.
- `data/weather/`: EPW/weather inputs.
- `data/analyses/`: analysis metadata, manifests, live WebGPU metadata, and legacy/debug `.bin` reference payloads.
- `data/gis/`: GIS exports generated from verified viewer runs.
- `data/performance-results/`: raw collector output used by performance notes.

For the full data workflow, see [docs/data-onboarding.md](docs/data-onboarding.md).

## Legacy Python / Ladybug Reference Path

The repository also includes the earlier Python/Ladybug CPU analysis pathway. It uses Ladybug Tools and `ladybug-comfort` for EPW handling, sun paths, sky vectors, and SolarCal MRT, then uses `pythermalcomfort` for UTCI.

This pathway was the first major improvement over the default Grasshopper/Ladybug workflow: it made the analysis scriptable, added Embree-accelerated ray tracing, and parallelized the CPU work. The WebGPU pathway is the next step and is now the preferred route for new analysis because it runs the heavy work on the GPU inside the Three.js viewer.

Use the Python/Ladybug pathway when you need to reproduce legacy `.bin` outputs, compare against the original Ladybug-derived reference, debug parity, or run support scripts that have not moved to the viewer. Do not treat it as the normal path for new high-throughput runs.

```powershell
.\.venv\Scripts\Activate.ps1
pip install -e .[dev,gis]
```

Reference, parity, and legacy export commands for maintaining old artifacts:

```powershell
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe scripts/export_ben_gurion_intermediates.py --base-path data/analyses/Ben-Gurion/20250815_grid_2m_fullday --model data/3d_models/Ben-Gurion/original_with_layers.glb
```

## Project Layout

```text
fast-utci/
  viewer/                  SvelteKit + Three.js/WebGPU app
  viewer/src/routes/       Main and debug routes
  viewer/src/lib/compute/  WebGPU and selected-hour compute logic
  viewer/scripts/          Metadata, parity, GIS, and performance tools
  data/                    Models, weather, analyses, performance, GIS output
  docs/                    Data, GIS, architecture, and performance notes
  src/fast_utci/           Legacy Python/Ladybug reference and export package
  scripts/                 Python utility scripts for legacy exports and GIS
```

## Documentation

- [viewer/README.md](viewer/README.md): viewer development, routes, build, and tests.
- [docs/data-onboarding.md](docs/data-onboarding.md): how to add models, weather, metadata, and projects.
- [docs/innovation-district-gis-handoff.md](docs/innovation-district-gis-handoff.md): GIS export bundle, GeoParquet, CRS, and active-cell policy.
- [docs/webgpu_strategy_analysis.md](docs/webgpu_strategy_analysis.md): WebGPU architecture and performance boundaries.
- [src/fast_utci/README.md](src/fast_utci/README.md): legacy Python/Ladybug API and reference/export tools.

## License

fast-utci is licensed under the GNU Affero General Public License v3.0 or later (`AGPL-3.0-or-later`). See [LICENSE](LICENSE).

The license choice follows the Ladybug Tools / Ladybug Comfort licensing used by the thermal-comfort and environmental-analysis code this project depends on and adapts. See [NOTICE.md](NOTICE.md) for attribution.
