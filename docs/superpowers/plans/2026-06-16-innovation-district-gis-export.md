# Innovation District GIS Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Do not create commits, branches, git worktrees, pushes, or PRs unless the user explicitly asks.

**Goal:** Export shape-aware Innovation District UTCI and Shading Index artifacts from `fast-utci` that can be handed off to the `innovation_dashboard` project for GIS map and chart integration.

**Architecture:** Keep Rhino/georef probing, live WebGPU collection, raw collector outputs, normalized GeoParquet post-processing, and dashboard handoff as separate units. Reuse the existing collector script, viewer GIS helper, collector seam, Python postprocessor, and tests instead of inventing a parallel greenfield path; this slice is about carrying the current implementation forward from the completed raw-collection work into one clean reusable handoff package. The canonical final handoff is a per-export bundle under `data/gis/Innovation-District/<date>_<resolution>/` centered on `cells.geoparquet`, `manifest.json`, and `qa/debug-sample.geojson`; PMTiles, vector/raster tiles, and Chart.js JSON are owned by `innovation_dashboard` and must not be required final artifacts from this repo.

**Tech Stack:** Python 3.11, base `numpy`, `pyproj`, required `pyarrow` for direct GeoParquet writing and metadata, SvelteKit/Vite viewer scripts, Playwright/Vitest, WebGPU, TypeScript, and the existing `fast-utci` active-mask plus selected-hour live WebGPU modules. `rhino3dm` remains relevant only to the separate Rhino georef extractor.

---

## Source Documents And Current Evidence

- Design record: `docs/superpowers/specs/2026-06-16-innovation-district-gis-extraction-design.md`
- Rhino extractor: `scripts/extract_rhino_georef.py`
- Rhino extraction output: `data/3d_models/Innovation-District/innovation_district.georef.json`
- Visual alignment proof: `data/3d_models/Innovation-District/innovation_district_map_check.html`
- Innovation District live metadata: `data/analyses/Innovation-District/innovation_district_webgpu.json`
- Key runtime source: `viewer/src/lib/compute/selected-hour/liveUtciSelectedHourSession.ts`
- Active-mask source: `viewer/src/lib/compute/core/studyAreaMask.ts`

Current validated facts:

- The `.3dm` model units are meters.
- Projected model coordinates align visually when transformed as EPSG:2039 to WGS84.
- `district_outline` is the current visual GIS anchor.
- Rhino `EarthAnchorPoint` reports latitude/longitude `0,0` and must not be used for placement.
- Valid GIS UTCI/shading results must come from active sampled cells, not the rectangular canonical grid.
- `train_tracks` participated as a metadata sampling hint historically, but must not become UTCI result cells unless the runtime active-mask policy includes it.

Implementation status:

- Tasks 1-3 are already completed by prior workers and are retained below as historical implementation context.
- Task 4 is the first live execution task in this revised plan. Do not rewrite completed history as greenfield work; inspect only when needed to preserve the current contract.
- Keep SDD/no commits/no worktrees constraints in force for every remaining task.

Coordinate-source contract:

- The raw export source is the generated live `Analysis.data.positions` returned through the existing selected-hour session path.
- Those positions are already in projected analysis coordinates for the `xy_ground` analysis. Do not apply `worldToAnalysisCoords`, Rhino transforms, GLB origin transforms, or another Y/Z swap in the GIS export helper.
- Python post-processing owns the only CRS transform: EPSG:2039 projected `x,y` to WGS84 `lon,lat`.
- The current active-mask policy expected for Innovation District is `base+road`. Any export metadata with a missing or unexpected active-mask source must fail validation.

## File Structure

Start from current repo reality, not a greenfield mental model. Before changing behavior, inspect the existing collector script, viewer GIS helper, collector seam, Python postprocessor, and tests; identify what is missing from the reviewed contract; add or adjust targeted tests; then modify the existing files.

Modify existing runtime surfaces first:

- `viewer/src/lib/gis/innovationDistrictExport.ts`
  - Keep it as the pure active-cell raw array and raw metadata helper.
  - No browser lifecycle, no file writes, no CRS conversion, no summary statistics, and no dashboard-facing metric computation.

- `viewer/tests/gis/innovationDistrictExport.test.ts`
  - Adjust existing helper tests to cover raw array layout, active-only validation, metadata contract, and collector-timing payload shape.

- `viewer/src/routes/main/collectorExportSeam.ts`
  - Modify only if the existing query-gated seam needs contract tightening.
  - Keep GIS-specific orchestration out of compute-core modules.

- `viewer/tests/gis/innovationDistrictCollectorSeam.test.ts`
  - Adjust only if seam lifecycle or contract behavior changes.

- `viewer/scripts/export-innovation-district-gis.ts`
  - Extend the existing Playwright collector rather than replacing it.
  - Owns CLI parsing, bundle output paths, collector orchestration, and raw artifact writes only.

- `viewer/package.json`
  - The `gis:export-innovation-district` script already exists; update only if its arguments or wrapper command need to change.

- `scripts/postprocess_innovation_district_gis.py`
  - Keep as the thin CLI wrapper and compatibility entrypoint.
  - It should delegate real work into a focused package instead of remaining a god-script.

- `tests/test_postprocess_innovation_district_gis.py`
  - Keep only thin wrapper or CLI-smoke coverage here once the Python logic is split by seam.

- `pyproject.toml`
  - Extend the existing `gis` optional dependency group with `pyarrow`.
  - The GIS export workflow should require `pip install -e ".[gis]"`; keep heavy GIS stacks out unless evidence later forces them in.

- `docs/innovation-district-gis-handoff.md`
  - Handoff note for the `innovation_dashboard` agent: bundle layout, CRS, schema, timing fields, chart/map guidance, and known caveats.

Prefer decomposing Python postprocessing into a focused package:

- `src/fast_utci/innovation_district_gis/`
  - `raw.py`: raw load and validation
  - `transforms.py`: CRS transform and active-cell table building
  - `summary.py`: optional cheap summary/statistics helper only if clearly useful
  - `geojson_outputs.py`: deterministic QA sampling and QA GeoJSON
  - `manifest.py`: manifest and output inventory
  - `orchestrator.py`: thin orchestration seam used by the CLI wrapper
  - `parquet_outputs.py`: `cells.geoparquet` writing with GeoParquet metadata via `pyarrow`

Keep the CLI wrapper thin and dispatch into these package modules instead of reintroducing `scripts/innovation_district_gis/` as a second ownership surface.

Split Python tests by seam instead of growing one god-test file:

- `tests/test_innovation_district_gis_raw.py`
- `tests/test_innovation_district_gis_summary.py`
- `tests/test_innovation_district_gis_qa_manifest.py`
- `tests/test_postprocess_innovation_district_gis.py`

Add more seam-specific tests only if the current files cannot cover the reviewed contract without turning back into a monolith.

Keep generated/manual QA artifacts outside app runtime:

- `data/3d_models/Innovation-District/innovation_district_map_check.html`
- `data/gis/Innovation-District/<date>_<resolution>/qa/*`

Avoid:

- Do not modify `D:\Projects\Nur\innovation_dashboard` in this plan.
- Do not add dashboard UI code here.
- Do not generate rectangular-grid GeoJSON or uncapped debug GeoJSON.
- Do not generate a monolithic all-hours active-cell GeoJSON by default.
- Do not make 24 per-hour GeoJSON files the default handoff.
- Do not generate PMTiles, vector tiles, raster tiles, or Chart.js JSON as required final artifacts in this repo.
- Optional hourly GeoJSON, `geojsonseq.gz`, or `summary.json` conversion is allowed only behind explicit opt-in flags or cheap postprocessing paths with a clear use; none of these are the handoff contract.
- Do not export inactive canonical cells as valid UTCI/shading results.

## Artifact Contract

The canonical final output is a small reusable handoff bundle, not a flat directory of prefixed files and not a dashboard-specific display package:

- `data/gis/Innovation-District/2025-08-15_2m/cells.geoparquet`
- `data/gis/Innovation-District/2025-08-15_2m/manifest.json`
- `data/gis/Innovation-District/2025-08-15_2m/qa/debug-sample.geojson`

Raw collector inputs may still live under `data/gis/Innovation-District/2025-08-15_2m/raw/` during extraction and post-processing:

- `raw/active-cells.metadata.json`
  - Raw active-cell export metadata from the live WebGPU collector.
  - Contains schema, CRS, counts, dimensions, hours, checksums, binary file names, active-mask provenance, and collector timings.
  - Keep the collector timing source shape as `metadata.timingsMs.total`; the manifest should copy that payload through rather than inventing a renamed collector-total field.

- `raw/active-cells.positions.f32.bin`
  - `Float32Array`, compact active rows, projected analysis coordinates as `x,y,z`.

- `raw/active-cells.canonical.u32.bin`
  - `Uint32Array`, one canonical index per active row.

- `raw/active-cells.utci.f32.bin`
  - `Float32Array`, active rows by hour. Metadata layout must be `point-major-hour`.

- `raw/active-cells.shading.f32.bin`
  - `Float32Array`, one Shading Index value per active row.

Final reusable handoff contents:

- `cells.geoparquet`
  - One row per active cell only. Do not include inactive canonical grid rows.
  - Geometry column is WKB point geometry with GeoParquet metadata, CRS EPSG:4326, written directly with `pyarrow`.
  - Required columns:
    - `active_index`
    - `canonical_index`
    - `geometry`
    - `lon`
    - `lat`
    - source projected `x`, `y`, `z` in EPSG:2039
    - `shading_index`
    - exactly one UTCI column for each hour: `utci_00` through `utci_23`
  - Optional compact provenance columns such as `grid_size_m`, `active_mask_signature`, or source IDs are allowed only if they are useful for row-level filtering or debugging. Avoid repeated wide text provenance when `manifest.json` can own it once.

- `manifest.json`
  - CRS details: source EPSG:2039, derived point geometry EPSG:4326, and the GeoParquet metadata/schema note.
  - Active and canonical counts, active ratio, hours, row count, and confirmation that row count equals active count.
  - Source raw file checksums and byte sizes.
  - Active-mask source, checksum/signature, expected source policy, and validation result.
  - Collector timing copied from raw `timingsMs`, postprocessor timing, total runtime, and a clear timing breakdown.
  - Output file sizes and checksums for `cells.geoparquet`, `manifest.json`, `qa/debug-sample.geojson`, and any optional `summary.json`.
  - Explicit dashboard boundary note: `innovation_dashboard` derives PMTiles/vector/raster tiles for MapLibre and summary/chart JSON for Chart.js from `cells.geoparquet`.

- `qa/debug-sample.geojson`
  - Small visual QA sample only. Must be capped and deterministic.
  - Sampling strategy must be named and honest, for example seeded stratification by grid quadrant plus UTCI/shading bins.
  - This file is for local map-alignment QA only and must not be treated as production dashboard data.

Optional output:

- `summary.json`
  - May be generated by this repo only if cheap and clearly useful.
  - It is not part of the handoff contract; the dashboard agent can derive Chart.js JSON later from `cells.geoparquet`.

Dashboard-owned display products:

- `cells.geoparquet -> PMTiles/vector/raster tiles for MapLibre`
- `cells.geoparquet -> summary/chart JSON for Chart.js`
- These products belong in `innovation_dashboard` or a downstream GIS/display pipeline, not as required outputs from this repo.

Migration and cleanup policy:

- Existing root-level `data/gis/Innovation-District/2025-08-15_2m_active-cells.*` files and any older `final-*` binaries should be treated as migration leftovers, not the desired steady-state layout.
- The next implementation should write and consume the bundle layout above.
- During migration, it is acceptable to read legacy root-level files so existing data is not stranded, but the implementation should move them into `raw/`, replace them with explicit bundle references, or clean stale root outputs after a successful bundle write.
- Do not leave `data/gis/Innovation-District/` root as the canonical home for ambiguous flat exports.

Opt-in conversion artifacts remain out of the default handoff:

- Hourly GeoJSON or `geojsonseq.gz` only when explicitly requested with output-size guardrails.
- PMTiles should be generated in the relevant dashboard or GIS project from `cells.geoparquet`.
- Cloud Optimized GeoTIFF or raster tile outputs are future optional work only.

Do not commit generated data unless the user explicitly asks.

---

### Task 1: Historical Re-Baseline Of GIS Export Surfaces And Dependencies

**Status:** Completed before this revision. Keep this section as historical context unless current implementation evidence proves a dependency or baseline assumption has changed.

**Files:**
- Inspect: `pyproject.toml`
- Inspect: `viewer/package.json`
- Inspect: `viewer/src/lib/gis/innovationDistrictExport.ts`
- Inspect: `viewer/src/routes/main/collectorExportSeam.ts`
- Inspect: `viewer/scripts/export-innovation-district-gis.ts`
- Inspect: `scripts/postprocess_innovation_district_gis.py`
- Inspect: `tests/test_postprocess_innovation_district_gis.py`
- Inspect: `viewer/tests/gis/innovationDistrictExport.test.ts`
- Inspect: `viewer/tests/gis/innovationDistrictCollectorSeam.test.ts`
- Read-only reference: `data/3d_models/Innovation-District/innovation_district.georef.json`

Completed outcomes retained for context:

- Re-baselined the existing collector script, viewer helper, collector seam, Python postprocessor, and current tests against the reviewed contract.
- Confirmed the important deltas were bundle layout clarity, manifest timing naming, Python package ownership, and keeping default outputs away from dashboard-specific artifacts.
- Reaffirmed that the `gis` extra should carry `pyarrow` for GeoParquet writing while avoiding heavier GIS stacks unless evidence later forces them.
- Established that existing tests should be extended in place rather than replaced with a speculative new `scripts/innovation_district_gis/` test tree.

### Task 2: Historical Tightening Of Viewer Raw Export Helper And Collector Seam

**Status:** Completed before this revision. Keep this section as historical context unless current implementation evidence proves the raw collector contract has changed.

**Files:**
- Modify: `viewer/src/lib/gis/innovationDistrictExport.ts`
- Modify: `viewer/tests/gis/innovationDistrictExport.test.ts`
- Modify if needed: `viewer/src/routes/main/collectorExportSeam.ts`
- Modify if needed: `viewer/tests/gis/innovationDistrictCollectorSeam.test.ts`
- Modify: `viewer/scripts/export-innovation-district-gis.ts`
- Modify only if needed: `viewer/package.json`
- Reference: `viewer/src/lib/types/analysis.ts`
- Reference: `viewer/src/lib/compute/selected-hour/liveUtciSelectedHourSession.ts`

Completed outcomes retained for context:

- Confirmed the TypeScript side should stay limited to raw arrays plus raw metadata and should not own GeoParquet, manifest, summary, or QA artifact generation.
- Locked the raw collector contract around active-only rows, projected analysis coordinates, preserved active/canonical identity, `point-major-hour` UTCI layout, and validation of `activeMask.source`.
- Kept the collector on the existing selected-hour session and query-gated seam, with raw outputs targeting bundle `raw/` rather than a new parallel runtime path.
- Scoped viewer verification to the existing helper and seam tests instead of inventing a dashboard-facing test surface here.

### Task 3: Historical Python Package Decomposition

**Status:** Completed before this revision. Keep this section as historical context unless current implementation evidence proves a seam must change.

**Files:**
- Modify: `scripts/postprocess_innovation_district_gis.py`
- Modify: `src/fast_utci/innovation_district_gis/raw.py`
- Modify: `src/fast_utci/innovation_district_gis/transforms.py`
- Modify: `src/fast_utci/innovation_district_gis/summary.py`
- Modify: `src/fast_utci/innovation_district_gis/geojson_outputs.py`
- Modify: `src/fast_utci/innovation_district_gis/manifest.py`
- Modify: `src/fast_utci/innovation_district_gis/orchestrator.py`
- Modify: `tests/test_postprocess_innovation_district_gis.py`
- Create or modify: seam-specific tests under `tests/`
- Reference: `data/3d_models/Innovation-District/innovation_district.georef.json`

Completed outcomes retained for context:

- Split the Python ownership into the reusable `src/fast_utci/innovation_district_gis/` package while keeping `scripts/postprocess_innovation_district_gis.py` as the thin CLI wrapper.
- Kept seam ownership focused: `raw.py`, `transforms.py`, `summary.py`, `geojson_outputs.py`, `manifest.py`, and `orchestrator.py`, with GeoParquet work reserved for `parquet_outputs.py`.
- Preserved Python ownership for CRS conversion, final bundle writing, QA GeoJSON, manifest generation, and any clearly optional non-contract summaries.
- Rejected default all-hours GeoJSON, default 24-file hourly GeoJSON, PMTiles, vector/raster tiles, and Chart.js JSON as outputs of this repo.

### Task 4: Write Combined `cells.geoparquet` Handoff Package

**Status:** Implemented in the Task 4 worker slice on 2026-06-16.

**Files:**
- Modify: `viewer/scripts/export-innovation-district-gis.ts`
- Modify: `scripts/postprocess_innovation_district_gis.py`
- Modify: `src/fast_utci/innovation_district_gis/transforms.py`
- Modify or create: `src/fast_utci/innovation_district_gis/parquet_outputs.py`
- Modify: `src/fast_utci/innovation_district_gis/manifest.py`
- Modify: `src/fast_utci/innovation_district_gis/geojson_outputs.py`
- Modify if optional summary remains: `src/fast_utci/innovation_district_gis/summary.py`
- Modify related tests under `tests/test_innovation_district_gis_raw.py`, `tests/test_innovation_district_gis_summary.py`, `tests/test_innovation_district_gis_qa_manifest.py`, and `tests/test_postprocess_innovation_district_gis.py`
- Modify related viewer tests under `viewer/tests/gis/` only if raw collector paths or metadata shape changed

- [x] **Step 1: Inspect current output-path behavior and legacy artifacts**

Check the current `data/gis/Innovation-District/` layout and note whether existing root-level `*_active-cells.*` files or `final-*` binaries are present. Decide which are migration leftovers and which, if any, still need compatibility reads.

- [x] **Step 2: Add targeted tests for the combined active-cell table contract**

Add or adjust tests that prove:

- `cells.geoparquet` has one row per active cell, not one row per canonical grid cell
- inactive canonical cells are absent
- required columns exist: `active_index`, `canonical_index`, `geometry`, `lon`, `lat`, `x`, `y`, `z`, `shading_index`, `utci_00` through `utci_23`
- `geometry` is WKB point data with GeoParquet metadata declaring EPSG:4326
- `x`, `y`, `z` preserve source projected coordinates in EPSG:2039
- UTCI values preserve the raw `point-major-hour` layout in column form
- compact provenance columns are useful and bounded, or else provenance lives only in `manifest.json`

- [x] **Step 3: Implement or tighten direct `pyarrow` GeoParquet writing in `parquet_outputs.py`**

Use `pyarrow` directly:

- build a `pyarrow.Table` with the active-cell columns
- encode point geometry as WKB using WGS84 `lon`, `lat`
- attach GeoParquet metadata to the schema under the standard `geo` metadata key
- write `data/gis/Innovation-District/<date>_<resolution>/cells.geoparquet`
- avoid adding GeoPandas or a heavy GIS stack unless direct `pyarrow` writing is proven insufficient

- [x] **Step 4: Write the simple final bundle layout**

New postprocessor output should be:

- `data/gis/Innovation-District/<date>_<resolution>/cells.geoparquet`
- `data/gis/Innovation-District/<date>_<resolution>/manifest.json`
- `data/gis/Innovation-District/<date>_<resolution>/qa/debug-sample.geojson`

Raw collector artifacts may remain under `raw/` in the same bundle. Do not create a required `derived/` split just to hold the final package. Do not introduce `geometry.parquet`, `values.parquet`, or any other split-final-parquet contract.

- [x] **Step 5: Preserve migration behavior without over-cleaning**

During migration, the implementation may ingest legacy root-level `*_active-cells.*` or `final-*` artifacts only when they are explicitly present, but it should then:

- move them under `raw/`, or
- replace them with explicit bundle references, or
- leave them absent and regenerate fresh `raw/` inputs through the collector for new runs

The flat root-level layout should stop being the canonical contract for new runs. Do not delete source `.3dm`, `.glb`, `.georef.json`, or other georeferencing/model inputs as part of cleanup.

- [x] **Step 6: Verify there are no dashboard-specific required outputs**

Confirm that:

- no monolithic all-hours GeoJSON is produced by default
- no 24-file hourly GeoJSON set becomes the default handoff
- no `geometry.parquet` or `values.parquet` split artifact becomes part of the contract
- no PMTiles, vector tiles, raster tiles, or Chart.js JSON are generated as required outputs
- optional conversion outputs require explicit flags plus size guardrails
- optional `summary.json`, if retained, is clearly marked non-contractual

### Task 5: Fix Manifest Timing, Provenance, And Dashboard Boundary

**Status:** Completed in the Task 5 worker slice on 2026-06-16 with real bundle evidence under `data/gis/Innovation-District/2025-08-15_2m/`.

**Files:**
- Modify: `viewer/src/lib/gis/innovationDistrictExport.ts`
- Modify: `viewer/scripts/export-innovation-district-gis.ts`
- Modify: `src/fast_utci/innovation_district_gis/manifest.py`
- Modify: `tests/test_innovation_district_gis_qa_manifest.py`
- Modify: related viewer tests if raw metadata fields change

- [x] **Step 1: Add tests for the reviewed timing contract**

Use raw metadata shaped like:

```json
{
  "timingsMs": {
    "routeLoad": 1,
    "utciCollection": 2,
    "shadingCollection": 3,
    "binarySerialization": 4,
    "total": 10
  }
}
```

Required manifest behavior:

- `collectorTimingsMs` is copied from `metadata.timingsMs`
- the collector total is exposed as `collectorTimingsMs.total`
- `postprocessorTimingsMs.totalPostprocessorRuntime` records Python runtime
- `totalExportRuntimeMs = collectorTimingsMs.total + postprocessorTimingsMs.totalPostprocessorRuntime`

- [x] **Step 2: Add tests for manifest inventory and provenance**

Add or adjust tests to confirm `manifest.json` includes:

- CRS: source EPSG:2039, output geometry EPSG:4326, and GeoParquet schema/metadata note
- active count, canonical count, active ratio, hours, and row count
- source raw checksums and byte sizes
- active-mask source, checksum/signature, expected policy, and validation result
- output file sizes and checksums for `cells.geoparquet`, `manifest.json`, `qa/debug-sample.geojson`, and optional `summary.json`
- dashboard boundary note that `innovation_dashboard` derives PMTiles/vector/raster tiles and summary/chart JSON from `cells.geoparquet`

- [x] **Step 3: Remove any lingering artifact ownership confusion**

The TypeScript collector side owns raw metadata and raw binary artifacts only. The Python side owns `cells.geoparquet`, `manifest.json`, capped QA GeoJSON, and optional non-contract `summary.json` or conversion outputs. The dashboard repo owns PMTiles/vector/raster tile generation and summary/chart JSON derivation.

- [x] **Step 4: Run collector plus postprocessor smoke verification**

Run:

```powershell
cd viewer
npm run gis:export-innovation-district -- --out-dir ..\data\gis\Innovation-District
```

Then run:

```powershell
.\.venv\Scripts\python.exe scripts\postprocess_innovation_district_gis.py `
  --metadata data\gis\Innovation-District\2025-08-15_2m\raw\2025-08-15_2m_active-cells.metadata.json `
  --georef data\3d_models\Innovation-District\innovation_district.georef.json `
  --out-dir data\gis\Innovation-District\2025-08-15_2m `
  --debug-geojson-limit 5000
```

Expected:

- raw artifacts land in `raw/`
- `cells.geoparquet` and `manifest.json` land at the bundle root
- debug sample lands in `qa/`
- optional `summary.json`, if generated, is marked non-contractual
- no monolithic all-hours GeoJSON is written by default
- no PMTiles, vector/raster tiles, or Chart.js JSON are required outputs
- manifest timing fields match the reviewed contract
- real evidence captured: `cells.geoparquet` size `61874032` bytes, `manifest.json` size `11689` bytes, `qa/debug-sample.geojson` row count `5000`, `collectorTimingsMs.total = 55923.0294`, `postprocessorTimingsMs.totalPostprocessorRuntime = 3286.7725`, `totalExportRuntimeMs = 59209.8019`

### Task 6: Produce Dashboard Handoff Documentation

**Status:** Completed in this worker slice on 2026-06-16.

**Files:**
- Create: `docs/innovation-district-gis-handoff.md`
- Reference: generated `data/gis/Innovation-District/<date>_<resolution>/*`
- Reference: `docs/superpowers/specs/2026-06-16-innovation-district-gis-extraction-design.md`

- [x] **Step 1: Write the handoff doc**

Include:

- source `.3dm` path and generated georef path
- CRS: EPSG:2039 source `x,y,z`; EPSG:4326 `lon,lat` and GeoParquet point geometry
- EarthAnchorPoint caveat: reports `0,0`, do not use
- canonical final bundle layout: `cells.geoparquet`, `manifest.json`, `qa/debug-sample.geojson`
- raw collector inputs under `raw/` if retained
- raw artifact list and schema
- `cells.geoparquet` table schema, including WKB `geometry`, `active_index`, `canonical_index`, `lon`, `lat`, `x`, `y`, `z`, `shading_index`, and `utci_00` through `utci_23` exactly once
- optional `summary.json` note if generated, clearly marked non-contractual
- active vs canonical grid explanation
- timing breakdown for collector runtime, Python post-processing runtime, and total wall-clock export runtime
- timing contract: collector raw metadata keeps `timingsMs.total`; manifest reports it as `collectorTimingsMs.total`; total export runtime is `collectorTimingsMs.total + postprocessorTimingsMs.totalPostprocessorRuntime`
- debug GeoJSON is sampled local QA only and not production dashboard data
- optional hourly GeoJSON/GeoJSONSeq conversion is not part of the default handoff and requires explicit opt-in plus output-size guardrails
- migration note for legacy root-level `*_active-cells.*` or `final-*` artifacts, including whether they were moved, referenced, or cleaned
- recommended dashboard display path: derive PMTiles/vector/raster tiles for MapLibre and summary/chart JSON from `cells.geoparquet` inside `innovation_dashboard`
- exact map validation notes

- [x] **Step 2: Add a pasteable handoff section**

Include a short block another agent can paste into the `innovation_dashboard` session.

- [x] **Step 3: Verify docs mention dashboard scope boundary**

Confirm the doc says this repo does not implement dashboard UI.

### Task 7: Final Verification And Review

**Files:**
- All files touched by Tasks 1-6

- [ ] **Step 1: Run targeted Python tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_postprocess_innovation_district_gis.py tests\test_innovation_district_gis_*.py -q
```

Expected: pass.

- [ ] **Step 2: Run targeted viewer tests**

Run:

```powershell
cd viewer
npm run test -- tests/gis/innovationDistrictExport.test.ts tests/gis/innovationDistrictCollectorSeam.test.ts
```

Expected: pass for the files touched in this slice.

- [ ] **Step 3: Run type/check if collector changed app-visible TypeScript**

Run:

```powershell
cd viewer
npm run check
```

Expected: pass, or report exact failures if unrelated existing issues block.

- [ ] **Step 4: Run real export smoke**

Run the real collector and postprocessor commands from Tasks 4 and 5.

Expected:

- bundle directory exists under `data/gis/Innovation-District/<date>_<resolution>/`
- raw active-cell artifacts exist under `raw/`
- binary array byte lengths match metadata dimensions
- `cells.geoparquet` exists at the bundle root
- `cells.geoparquet` has GeoParquet metadata declaring EPSG:4326 WKB point geometry
- `cells.geoparquet` contains `active_index`, `canonical_index`, `lon`, `lat`, `x`, `y`, `z`, `shading_index`, and `utci_00` through `utci_23`
- optional `summary.json`, if generated, is marked non-contractual
- sampled debug GeoJSON exists under `qa/`
- manifest exists at the bundle root
- manifest enumerates raw inputs and final outputs with paths, file sizes, checksums, schemas, and row counts
- manifest includes collector, postprocessor, and total timing fields
- manifest uses collector `timingsMs.total` as `collectorTimingsMs.total`
- `totalExportRuntimeMs = collectorTimingsMs.total + postprocessorTimingsMs.totalPostprocessorRuntime`
- row count equals active result count
- `cells.geoparquet` row count equals `activeCount`, not `canonicalCount`
- `cells.geoparquet` contains `utci_00` through `utci_23` exactly once each
- inactive canonical cells are absent
- no monolithic all-hours GeoJSON is produced by default
- no PMTiles, vector/raster tiles, or Chart.js JSON are produced as required outputs
- optional hourly GeoJSON/GeoJSONSeq conversion, if implemented, requires opt-in and size budget or `--allow-large-output`
- legacy root-level `*_active-cells.*` or `final-*` leftovers are either moved, referenced explicitly, or cleaned so the root directory is not the canonical output surface
- `git status --short data/gis/Innovation-District` shows generated bundle contents are not staged or committed unless the user explicitly approved generated artifacts

- [ ] **Step 5: Run final review**

Run deslop/spec/architecture review before final report.

Required bar:

- no rectangular-grid GIS result leak
- no dashboard repo edits
- no duplicated live WebGPU compute architecture
- no full active-cell all-hours GeoJSON as a default exchange artifact
- no 24-file hourly GeoJSON default
- `cells.geoparquet` is the default reusable handoff artifact
- PMTiles/vector/raster tiles and Chart.js JSON are owned by `innovation_dashboard`
- debug QA GeoJSON is capped and deterministic
- optional conversion artifacts are guarded by opt-in and output budget/allow-large-output checks
- no rectangular-grid or inactive-cell GeoJSON
- no unbounded generated files committed by accident

- [ ] **Step 6: Fix review findings and rerun targeted verification**

Repeat until reviewers pass or the remaining issue is explicitly escalated to the user.

- [ ] **Step 7: Final report**

Report:

- changed files
- generated files
- exact verification commands and outcomes
- active/canonical counts
- raw, final package, QA, and optional artifact paths for dashboard handoff
- confirmation that GeoParquet metadata was written and what CRS it declares
- whether optional hourly GeoJSON conversion exists and, if so, what guardrails were verified
- residual risks

Do not commit.
