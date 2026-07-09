# Adding Data To fast-utci

This guide explains how to add a model, weather file, or analysis to the viewer.

```text
GLB model + EPW/weather + analysis metadata
  -> project registration
  -> viewer route
  -> WebGPU UTCI / SAI analysis
```

## File Locations

- `data/3d_models/<Project>/`: GLB files and optional georeference sidecars.
- `data/weather/`: EPW/weather inputs.
- `data/analyses/<Project>/`: metadata JSON, manifests, and precomputed `.bin` artifacts when needed.
- `viewer/src/lib/config/projects.ts`: the project list shown by the app.

Keep project and analysis IDs stable. They are used by URLs, manifests, caches, tests, and downstream exports.

## Add A Project

1. Put the GLB in `data/3d_models/<Project>/`.
2. Add or confirm the weather input in `data/weather/`.
3. Generate an analysis metadata JSON under `data/analyses/<Project>/`.
4. Register the project in `viewer/src/lib/config/projects.ts`.
5. Start the viewer:

```powershell
cd viewer
npm run dev
```

1. Open `/`, select the analysis, and verify the model, layers, time controls, UTCI, and SAI.



## Generate Metadata

Analysis metadata tells the viewer how to build the analysis grid and which model/weather files to use. It should include:

- `model_file`
- `epw_file`
- `bounds`
- `grid_size`
- `num_positions`
- `hours`
- `analysis_type`
- `coordinate_system`

For GLB-backed WebGPU projects, use `viewer/scripts/generate-live-webgpu-metadata.ts`.

Example:

```powershell
cd viewer
node --import tsx ./scripts/generate-live-webgpu-metadata.ts `
  --model ../data/3d_models/Innovation-District/innovation_district.glb `
  --out ../data/analyses/Innovation-District/innovation_district_webgpu.json `
  --analysis-id Innovation-District/innovation_district_webgpu `
  --project-id Innovation-District `
  --grid-size 2 `
  --date 2025-08-15 `
  --coordinate-system xy_ground `
  --sample-height 0.9 `
  --weather-profile beer-sheva
```

Required flags:

- `--model`: source GLB path.
- `--out`: output metadata JSON path.
- `--analysis-id`: stable app analysis id.
- `--project-id`: stable app project id.
- `--grid-size`: grid spacing in meters.
- `--date`: representative analysis date.
- `--coordinate-system`: `xy_ground` or `xz_ground`.
- `--sample-height`: analysis height above the sampled surface.
- `--weather-profile`: currently `beer-sheva`.

Optional layer flags:

- `--sampling-layers`: comma-separated layer names used to define the sampled analysis bounds.
- `--occluder-layers`: comma-separated layer names included in the compute BVH.
- `--ignored-layers`: comma-separated layer names excluded from compute.

`has_shading_index: false` means there is no baked SAI array in a precomputed `.bin` payload. It does not prevent the WebGPU route from computing SAI live.

## Weather

The viewer resolves weather from analysis metadata first. Prefer setting `epw_file` in the metadata so each analysis is self-contained. Project-level defaults are useful as fallback, but metadata is easier to audit.

## Precomputed `.bin` Analyses

Precomputed `.bin` + `.json` analyses are legacy/debug artifacts. They are supported for reference, parity, old export reproduction, and compatibility workflows. They are not the recommended path for new high-throughput analysis.

These artifacts usually come from the intermediate Python/Ladybug path, which improved on the default Grasshopper/Ladybug workflow with Embree-backed ray tracing and parallel CPU execution. The current WebGPU/Three.js path is the next step and should be used for new viewer projects whenever possible.

New interactive projects should use GLB + metadata + weather unless there is a specific reason to ship precomputed `.bin` data.

## Project Registration

Add the project to `viewer/src/lib/config/projects.ts`. At minimum, provide:

- a stable project id
- a display label
- a `defaultAnalysisId`
- one or more analyses

If the project has scenarios, define scenario categories in the project config. If it does not, leave scenarios out so the sidebar stays clean.

## Verification

For a normal data/project onboarding change:

```powershell
cd viewer
npm run check
npm test
```

For route-boundary changes or performance claims, collect evidence from `/`, not `/debug`, and include the relevant proof fields:

- `rendererBackend=webgpu`
- `utciSurfaceSource=compute-buffer-selected-hour`
- `baseRenderTransport=compute-buffer-selected-hour`
- `dataTextureBuildCount=0`
- `visibleSelectedHourReadbackCount=0` for the visible selected-hour path



