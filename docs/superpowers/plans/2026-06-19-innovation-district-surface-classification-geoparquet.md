# Innovation District Surface Classification GeoParquet Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Do not create commits, branches, git worktrees, pushes, or PRs unless the user explicitly asks.

**Goal:** Add truthful per-active-cell surface classification to the Innovation District GIS export so downstream maps and statistics can exclude building footprints and switch between all valid outdoor surfaces and street/road/sidewalk-family surfaces.

**Architecture:** Preserve classification at the runtime active-mask boundary before `base+road` provenance is collapsed, carry an aligned compact `surfaceFlags` raw bitset array through the collector, and write downstream-friendly columns into `cells.geoparquet`. Treat street, road, and sidewalk as one public-realm street-surface family; treat building footprint membership as a separate multi-hot flag because it can overlap other surface families. Keep the sister `innovation_dashboard` out of scope.

**Tech Stack:** SvelteKit/Vite viewer runtime, Three.js mesh/layer metadata, TypeScript typed array raw export helpers, Playwright collector script, Python 3.11, `numpy`, `pyarrow`, `pyproj`, existing `fast_utci.innovation_district_gis` postprocessor, Vitest, pytest.

---

## Hard Constraints

- No commits.
- No git worktrees.
- Do not edit `D:\Projects\Nur\innovation_dashboard`.
- Do not implement dashboard UI in this repo.
- Do not add PMTiles, raster tiles, vector tiles, or Chart.js JSON as required outputs.
- Do not export inactive canonical grid cells as valid GIS/stat rows.
- Do not infer `is_building_footprint` from UTCI/shading values.
- Do not add cosmetic abstraction layers or one-off flags scattered through busy runtime code.
- Do not add code comments beyond existing local style; prefer clear types, helpers, and tests.
- Preserve dirty-tree safety: inspect current file state before editing and do not revert unrelated user changes.
- User reviews this plan before implementation begins.

## Current Evidence

- GeoParquet rows are assembled in `src/fast_utci/innovation_district_gis/parquet_outputs.py` via `write_cells_geoparquet()`.
- Raw Python artifacts are loaded in `src/fast_utci/innovation_district_gis/raw.py` and currently include `canonicalIndices`, `positions`, `utci`, and `shadingIndex`.
- Raw TypeScript export contracts live in `viewer/src/lib/gis/innovationDistrictExport.ts`.
- The collector script is `viewer/scripts/export-innovation-district-gis.ts`.
- The browser collector seam is `viewer/src/routes/main/collectorExportSeam.ts`.
- Runtime active-mask creation lives in `viewer/src/lib/compute/selected-hour/liveUtciSelectedHourSession.ts`.
- Study-area rasterization lives in `viewer/src/lib/compute/core/studyAreaMask.ts`.
- Existing active-mask source is aggregate `base+road`; it does not preserve row-level ground-vs-road provenance.
- Layer semantics already identify `ground`, `street`/`road`, and `existing_buildings`, but buildings are occluders/context, not current sampled result surfaces.
- `liveUtciSelectedHourSession.ts` is already over 1,000 lines. Decompose classification out of it before implementation rather than adding new conditional growth inline.

## Design Decision

Use both a primary class and booleans:

- `surface_class`: derived primary label for convenience.
- `is_street_surface`: true for street/road/sidewalk-family sampled cells.
- `is_building_footprint`: true when the active cell sample center falls inside a projected building-footprint triangle.
- `include_in_public_realm_stats`: true for rows that should count in street/road/sidewalk-only downstream statistics.
- `include_in_outdoor_surface_stats`: true for rows that should count in the broad/default outdoor-surface downstream statistics.

Initial semantic contract:

```text
street / road / sidewalk -> is_street_surface = true
building footprint -> is_building_footprint = true
outdoor-surface stats -> active cells excluding building footprints
public-realm stats -> street/road/sidewalk-family cells excluding building footprints
```

Derived primary class values:

```text
ground
street_surface
building_footprint
unknown
```

If a cell is both street-family and building footprint, preserve both raw bits and booleans. Set `surface_class = building_footprint` only because the primary-class priority is useful for simple one-column consumers. The booleans are authoritative for statistics.

Priority for `surface_class`:

```text
building_footprint > street_surface > ground > unknown
```

This priority is only a display/filter convenience. It must not erase the booleans.

## Perspective Ensemble Gates

### Panel A - Council

- **Data contract:** downstream needs stable denominator fields more than raw layer names. Counter-move: expose `include_in_*_stats` columns and document them.
- **Geometry correctness:** street provenance is lost today when triangles merge into one mask. Counter-move: tag triangles before rasterization and emit aligned class bits with active canonical indices.
- **Building semantics:** buildings are occluders, not sampled surfaces. Counter-move: add a separate building-footprint classification pass over the canonical grid, then compact to active rows.
- **Maintainability:** live session code is already large. Counter-move: keep extraction/classification helpers small and pure, preferably by extending `studyAreaMask.ts` or a focused sibling helper rather than adding ad-hoc branches throughout the session.
- **Downstream ergonomics:** booleans are easy to filter in DuckDB/GeoParquet/MapLibre preprocessing. Counter-move: write ordinary Parquet boolean/string columns and update manifest schema/docs.

### Panel B - Red Cell

- **Attack target:** a plan that adds plausible-looking classification columns without proving they correspond to layer geometry.
- **Failure mode:** `is_building_footprint` gets confused with shade caused by buildings. Mitigation: derive only from building footprint rasterization, never from UTCI/shading.
- **Failure mode:** `surface_class` hides overlaps and makes stats wrong. Mitigation: booleans are authoritative; primary class is secondary.
- **Failure mode:** special cases leak into compute/session orchestration. Mitigation: isolate classification in pure helpers and typed contracts.
- **Failure mode:** downstream silently uses old GeoParquet without new fields. Mitigation: bump raw schema, manifest surface-class contract, docs, and tests that assert columns exist.

Falsifiers / early warnings:

- `is_street_surface` count is zero or nearly all active rows for Innovation District.
- `is_building_footprint` is derived without reading building-layer geometry.
- `include_in_outdoor_surface_stats` includes rows with `is_building_footprint = true`.
- `cells.geoparquet` row count changes without a deliberate active-mask policy change.
- New code pushes `liveUtciSelectedHourSession.ts` further into unreviewable conditional sprawl.

## File Structure

Modify existing runtime/classification surfaces:

- `viewer/src/lib/types/analysis.ts`
  - Keep `AnalysisActiveMask` minimal.
  - Add a narrow `ClassifiedAnalysisActiveMask extends AnalysisActiveMask` contract with required `surfaceFlagsByActiveCell: Uint8Array`.
  - Classified Innovation District export must require `ClassifiedAnalysisActiveMask`; do not make export code rely on optional classification fields.

- `viewer/src/lib/compute/core/studyAreaMask.ts`
  - Keep point-in-triangle/grid-axis logic centralized.
  - Add pure rasterization helpers only if the new focused module needs reusable point-in-triangle/grid scanning.
  - Do not turn this into a mixed policy object.

- `viewer/src/lib/compute/selected-hour/activeMaskSurfaceClassification.ts`
  - Create this focused module before Task 2 implementation.
  - Own layer-name/type mapping, tagged projected-triangle extraction, sampled-surface active-mask construction, and classification overlay compaction.
  - Split the builder into two phases: sampled-surface mask from ground/street-family triangles, then classification overlay from tagged triangles compacted against `activeCanonicalIndices`.
  - Use the same projection/origin/coordinate-system handling for sampled surfaces and building footprints.

- `viewer/src/lib/compute/selected-hour/liveUtciSelectedHourSession.ts`
  - Orchestrate only: import the focused classification helper, call it, and attach the returned active mask/classified mask to the generated analysis.
  - Do not add new layer-name condition chains or rasterization logic inline.

- `viewer/src/lib/gis/innovationDistrictExport.ts`
  - Add `surfaceFlags` raw array validation, layout, metadata, and raw export construction.
  - Keep it pure: no browser lifecycle, no file writes, no CRS conversion.
  - Decomposition gate: if raw array descriptor/validation changes start duplicating the existing pattern, extract one small pure helper only if it deletes duplication. Do not create a broad raw-schema subsystem.

- `viewer/src/routes/main/collectorExportSeam.ts`
  - Require the classified active-mask contract in the query-gated collector return shape for Innovation District classified export.

- `viewer/scripts/export-innovation-district-gis.ts`
  - Write the new `.surface-flags.u8.bin` raw artifact and metadata/checksum entries.

Modify existing Python postprocessor surfaces:

- `src/fast_utci/innovation_district_gis/raw.py`
  - Load and validate `surfaceFlags` when raw schema requires it.

- `src/fast_utci/innovation_district_gis/parquet_outputs.py`
  - Write classification columns into `cells.geoparquet`.

- `src/fast_utci/innovation_district_gis/manifest.py`
  - Include raw classification artifact inventory and semantic notes.

- `src/fast_utci/innovation_district_gis/geojson_outputs.py`
  - Include classification properties in QA samples.

- `docs/innovation-district-gis-handoff.md`
  - Update handoff with new columns and downstream filter examples.

Tests:

- `viewer/tests/compute/studyAreaMask.test.ts`
- `viewer/tests/compute/live-selected-hour-session.test.ts`
- `viewer/tests/gis/innovationDistrictExport.test.ts`
- `viewer/tests/gis/innovationDistrictCollectorSeam.test.ts`
- `tests/test_innovation_district_gis_fixtures.py`
- `tests/test_innovation_district_gis_raw.py`
- `tests/test_innovation_district_gis_qa_manifest.py`
- `tests/test_postprocess_innovation_district_gis.py`

Do not create a broad new subsystem. If `studyAreaMask.ts` becomes too broad, create one focused sibling module for surface-classification rasterization and keep `liveUtciSelectedHourSession.ts` as orchestration only.

## Raw And GeoParquet Contract

Raw artifact:

- `raw/2025-08-15_2m_active-cells.surface-flags.u8.bin`
- dtype: `u8`
- layout: `point-major`
- shape: `[activeCount]`
- aligned by row with `canonicalIndices`, `positions`, `utci`, and `shadingIndex`

Bit values:

```text
bit 0 = ground
bit 1 = street_surface
bit 2 = building_footprint
```

GeoParquet columns:

- `surface_class` string or dictionary-encoded string if direct PyArrow support stays simple.
- `surface_flags` uint8.
- `is_street_surface` bool.
- `is_building_footprint` bool.
- `include_in_public_realm_stats` bool.
- `include_in_outdoor_surface_stats` bool.

Boolean derivation:

```text
is_street_surface = active cell sample center falls inside street/road/sidewalk-family projected surface triangle
is_building_footprint = active cell sample center falls inside projected building-footprint triangle
include_in_public_realm_stats = is_street_surface and not is_building_footprint
include_in_outdoor_surface_stats = not is_building_footprint
```

Use bit flags as the raw contract. Do not use a primary-class enum as the raw source of truth because it cannot represent overlap without loss.

Classified Innovation District export invariants:

- Every active classified row must have at least one sampled-surface bit set: `ground` or `street_surface`.
- `surface_class = unknown` is not legal for classified active export rows in this plan. Unknown remains a generic-display fallback only for non-classified contexts unless a later user-approved policy change says otherwise.
- `surface_class` must be derived from `surface_flags`; raw metadata/artifacts must not introduce a primary enum/class array.
- Triangle boundary inclusion must be defined once via the shared sample-center point-in-triangle rule and reused by both sampled-surface rasterization and building-footprint overlays.

Layer policy:

- `ground` / `base`: valid sampled surface, ground flag.
- `street`, `streets`, `road`, `roads`, `sidewalk`, `sidewalks`: valid sampled surface if present in the active sampled surface geometry, street-surface flag.
- `train_track`, `train_tracks`: do not become valid sampled UTCI/shading cells in this pass. They should remain excluded from active-mask construction unless a separate user-approved policy change adds them.
- `building`, `existing_buildings`, `new_buildings`: classification overlay only. Building-only cells must not enter `activeCanonicalIndices`; building footprint flags are compacted only against existing active sampled cells.

## Task 0: Preflight Plan Review Loop

**Files:**
- Review: `docs/superpowers/plans/2026-06-19-innovation-district-surface-classification-geoparquet.md`
- Review: `docs/superpowers/specs/2026-06-16-innovation-district-gis-extraction-design.md`

- [ ] **Step 1: Run plan reviewer before implementation**

Dispatch a separate reviewer subagent with:

- this plan path
- relevant design/spec path
- hard constraints: no commits, no worktrees, no dashboard repo edits
- ask it to find missing tasks, ambiguous semantics, and verification gaps

If issues are found, update this plan and rerun the plan reviewer. Stop after 3 failed review iterations and ask the user.

- [ ] **Step 2: Confirm user review gate**

Implementation must not begin until the user reviews and approves this plan.

- [ ] **Step 3: Confirm durable design-record treatment**

This review pass must leave the classification contract in one of two explicit states before implementation:

- minimally updated in `docs/superpowers/specs/2026-06-16-innovation-district-gis-extraction-design.md`, or
- explicitly recorded in this plan as intentionally plan-local with a reason the durable design record should not carry it.

For this plan, prefer the minimal spec update because the active/export classification contract is durable handoff behavior, not temporary implementation bookkeeping.

## Task 1: Lock The Classification Contract With Failing Tests

**Files:**
- Modify: `viewer/tests/compute/studyAreaMask.test.ts`
- Modify: `viewer/tests/gis/innovationDistrictExport.test.ts`
- Modify: `tests/test_innovation_district_gis_fixtures.py`
- Modify: `tests/test_innovation_district_gis_raw.py`
- Modify: `tests/test_innovation_district_gis_qa_manifest.py`

- [ ] **Step 1: Add a study-area classification test**

Create a small rectangular grid fixture with overlapping triangles:

```text
ground covers cells 0,1,2
street_surface covers cell 1
building_footprint covers cell 2
```

Expected active rows stay `[0, 1, 2]`; expected flags preserve overlap.

Use sample-center point-in-triangle semantics. Include one boundary case and document the expected result in the test name so future tolerance changes are visible.

Also assert the shared boundary rule is the same one used later for building-footprint overlay rasterization.

- [ ] **Step 2: Run the study-area test and verify it fails**

Run:

```powershell
cd viewer
npm run test -- tests/compute/studyAreaMask.test.ts
```

Expected: fail because classification output does not exist yet.

- [ ] **Step 3: Add TypeScript raw export tests**

Add expectations that `buildActiveCellArrays()`:

- requires surface classification for the Innovation District classified export path
- preserves active row order
- validates `surfaceFlags` length equals active row count
- rejects raw metadata that attempts to introduce a primary enum/class array alongside `surfaceFlags`
- includes layout metadata for the new raw array

- [ ] **Step 4: Run GIS helper tests and verify they fail**

Run:

```powershell
cd viewer
npm run test -- tests/gis/innovationDistrictExport.test.ts
```

Expected: fail because raw classification support does not exist.

- [ ] **Step 5: Add Python fixture and postprocessor tests**

Extend `write_tiny_raw_fixture()` so it can write classification bytes. Add assertions that:

- raw loader validates dtype, shape, byte length, and checksum
- manifest inventories the classification raw artifact
- `cells.geoparquet` includes the six classification/stat columns
- GeoParquet schema types stay `surface_flags: uint8`, `surface_class: string`, derived inclusion/classification fields: bool
- QA GeoJSON includes classification properties

- [ ] **Step 6: Run Python tests and verify they fail**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_innovation_district_gis_raw.py tests\test_innovation_district_gis_qa_manifest.py -q
```

Expected: fail because Python raw and parquet layers do not load/write classification yet.

## Task 2: Preserve Layer Provenance During Active-Mask Rasterization

**Files:**
- Modify: `viewer/src/lib/types/analysis.ts`
- Create: `viewer/src/lib/compute/selected-hour/activeMaskSurfaceClassification.ts`
- Modify: `viewer/src/lib/compute/core/studyAreaMask.ts`
- Modify: `viewer/src/lib/compute/selected-hour/liveUtciSelectedHourSession.ts`
- Test: `viewer/tests/compute/studyAreaMask.test.ts`
- Test: `viewer/tests/compute/live-selected-hour-session.test.ts`

- [ ] **Step 1: Add typed surface classification model**

Add a narrow type for surface flags/classes. Do not let arbitrary numbers flow through classification code.

Candidate:

```ts
declare const surfaceFlagsBrand: unique symbol;
export type SurfaceFlags = number & { readonly [surfaceFlagsBrand]: true };

export const SURFACE_FLAGS = {
	ground: 1 << 0,
	streetSurface: 1 << 1,
	buildingFootprint: 1 << 2
} as const;
```

Add a helper-owned validated `uint8` constructor or parser for `SurfaceFlags`; all raw `number` values must enter through that boundary.

Add `ClassifiedAnalysisActiveMask extends AnalysisActiveMask` with required `surfaceFlagsByActiveCell: Uint8Array`. Do not use unsafe casts, `as any`, or casts that bypass type errors in the Innovation District classified export path. Literal `as const` declarations are fine.

- [ ] **Step 2: Add tagged triangle input**

Add a helper type such as:

```ts
export interface ClassifiedProjectedTriangle2D {
	triangle: ProjectedTriangle2D;
	flags: SurfaceFlags;
}
```

Keep the existing unclassified API if other callers use it.

- [ ] **Step 3: Create focused active-mask classification module**

Create `viewer/src/lib/compute/selected-hour/activeMaskSurfaceClassification.ts`.

It should own:

- normalizing layer names
- mapping layer names/types to `SurfaceFlags`
- extracting sampled-surface projected triangles
- extracting building-footprint projected triangles in a separate traversal
- calling pure rasterization helpers
- returning `activeMask` and `surfaceFlagsByActiveCell` as separate outputs

`liveUtciSelectedHourSession.ts` may only call this module and attach the result.

- [ ] **Step 4: Implement two-phase classification rasterization**

Extend or add a focused helper that returns:

```ts
{
	activeMask: StudyAreaMask;
	surfaceFlagsByActiveCell: Uint8Array;
}
```

Phase 1 builds the sampled-surface mask from ground/street-family triangles only, preserving the current `base+road` active policy. Phase 2 rasterizes classification overlays, including building footprints, and compacts flags strictly against `activeMask.activeCanonicalIndices`. Building-only cells must never enter `activeCanonicalIndices`.

Both phases must use the same sample-center point-in-triangle semantics, including the same boundary inclusion rule.

- [ ] **Step 5: Preserve street/road/sidewalk family tags**

In the focused classification module, map layer names/types:

```text
ground/base -> ground flag
street/streets/road/roads/sidewalk/sidewalks -> streetSurface flag
building/existing_buildings/new_buildings -> buildingFootprint flag
train_track/train_tracks -> excluded from active sampled surfaces in this pass
```

Street/road/sidewalk are treated the same only when they are part of the active sampled surface geometry. Sidewalk names should be supported by the classifier, but adding sidewalk support must not broaden the active mask unless matching geometry is actually present.

- [ ] **Step 6: Keep active mask source stable**

Do not change `activeMask.source` from `base+road` unless the implementation deliberately changes active-cell policy. The classification fields are an added contract, not a row-count change.

- [ ] **Step 7: Cover overlap and exclusion cases in viewer tests**

Add or extend viewer tests so they explicitly prove:

- building-only cells outside sampled ground/street geometry stay inactive
- `train_tracks` geometry stays excluded from active sampled cells
- street/road/sidewalk plus building-footprint overlap preserves multi-hot flags
- overlapping street/building rows remain excluded from `include_in_public_realm_stats`

- [ ] **Step 8: Run targeted tests**

Run:

```powershell
cd viewer
npm run test -- tests/compute/studyAreaMask.test.ts tests/compute/live-selected-hour-session.test.ts
```

Expected: pass.

## Task 3: Carry Classification Through Collector And Raw Export

**Files:**
- Modify: `viewer/src/lib/gis/innovationDistrictExport.ts`
- Modify: `viewer/src/routes/main/collectorExportSeam.ts`
- Modify: `viewer/scripts/export-innovation-district-gis.ts`
- Modify: `viewer/tests/gis/innovationDistrictExport.test.ts`
- Modify: `viewer/tests/gis/innovationDistrictCollectorSeam.test.ts`

- [ ] **Step 1: Add collector return field**

Expose `surfaceFlags` from `window.__fastUtciCollectorExport()` aligned with `canonicalIndices`.

- [ ] **Step 2: Validate raw classification array**

In `viewer/src/lib/gis/innovationDistrictExport.ts`, validate:

- required for classified Innovation District export
- length equals active row count
- dtype is `Uint8Array`
- values contain only known bit flags
- no unsafe casts, `as any`, silent fallback, or missing-classification coercion is used for the classified export path

Raw value validation belongs at collector/raw artifact ingress and Python load time. Do not add repeated known-value defensive checks inside trusted typed runtime paths.

- [ ] **Step 3: Extend raw metadata**

Add layout and array descriptors:

```json
"layout": {
  "surfaceFlags": "point-major"
},
"arrays": {
  "surfaceFlags": {
    "dtype": "u8",
    "endianness": "little",
    "shape": [activeCount],
    "byteLength": activeCount
  }
}
```

Do not add an enum raw array in this pass; derive primary class in Python from bit flags.

Add an explicit test/assertion that metadata/array descriptors contain `surfaceFlags` only and no raw primary class enum/list field.

- [ ] **Step 4: Write raw binary artifact**

In `viewer/scripts/export-innovation-district-gis.ts`, write:

```text
2025-08-15_2m_active-cells.surface-flags.u8.bin
```

Include checksum in metadata.

- [ ] **Step 5: Run targeted viewer GIS tests**

Run:

```powershell
cd viewer
npm run test -- tests/gis/innovationDistrictExport.test.ts tests/gis/innovationDistrictCollectorSeam.test.ts
```

Expected: pass.

## Task 4: Load Classification In Python And Write GeoParquet Columns

**Files:**
- Modify: `src/fast_utci/innovation_district_gis/raw.py`
- Modify: `src/fast_utci/innovation_district_gis/parquet_outputs.py`
- Modify: `src/fast_utci/innovation_district_gis/manifest.py`
- Modify: `src/fast_utci/innovation_district_gis/geojson_outputs.py`
- Modify: `tests/test_innovation_district_gis_fixtures.py`
- Modify: `tests/test_innovation_district_gis_raw.py`
- Modify: `tests/test_innovation_district_gis_qa_manifest.py`

- [ ] **Step 1: Extend `ActiveCellArtifacts`**

Add a `surface_flags` `np.ndarray` field.

- [ ] **Step 2: Validate raw array**

Validate dtype, shape, byte length, file existence, checksum, and known values. Keep validation errors explicit, matching current style.

Follow the existing raw loader validation style. Do not add broad `try/catch`, fallback-to-unknown, or permissive coercion for trusted classified export paths. Python raw load is the canonical postprocessor validation boundary; downstream Python code should not repeat defensive known-value checks after successful load.

- [ ] **Step 3: Derive final columns**

In `parquet_outputs.py`, derive:

```text
surface_flags
surface_class
is_street_surface
is_building_footprint
include_in_public_realm_stats
include_in_outdoor_surface_stats
```

Use `pa.uint8()`, `pa.string()` or dictionary encoding if simple, and `pa.bool_()`.

Enforce:

- every active export row has `surface_flags & (ground | street_surface) != 0`
- no classified active export row derives `surface_class = unknown`
- overlap rows keep both booleans true where applicable while public-realm inclusion stays false

- [ ] **Step 4: Add manifest semantic notes**

Document:

- street/road/sidewalk are grouped as `street_surface`
- building footprint is an exclusion flag for downstream maps/stats
- outdoor-surface stats exclude building footprints
- public-realm stats include street-surface rows excluding building footprints
- classified active export rows always have sampled-surface provenance; `unknown` is not a legal active-row class in this contract

- [ ] **Step 5: Add QA GeoJSON classification properties**

Include classification fields for sampled features so map spot checks can inspect whether flags look plausible.

- [ ] **Step 6: Add spatial QA coverage expectations**

Spatial QA must cover, if source geometry makes them available:

- one street/road/sidewalk-family sample
- one building-footprint overlap sample
- one building-only non-active location proving no classified export row was created

- [ ] **Step 7: Run targeted Python tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_innovation_district_gis_raw.py tests\test_innovation_district_gis_qa_manifest.py tests\test_postprocess_innovation_district_gis.py -q
```

Expected: pass.

## Task 5: Update Handoff Documentation And Downstream Usage Examples

**Files:**
- Modify: `docs/innovation-district-gis-handoff.md`
- Modify: `docs/superpowers/specs/2026-06-16-innovation-district-gis-extraction-design.md` only if the durable design record needs the new classification contract; otherwise leave it unchanged and document the decision in the implementation report.

- [ ] **Step 1: Document new raw artifact**

Add raw classification artifact name, dtype, shape, layout, and row alignment.

- [ ] **Step 2: Document new GeoParquet columns**

List all new classification/stat fields and their exact semantics.

- [ ] **Step 3: Make the durable design record explicit**

Update `docs/superpowers/specs/2026-06-16-innovation-district-gis-extraction-design.md` with the durable classification contract:

- classified export is sampled-surface-only
- building footprints are overlay/exclusion flags, not active-row creators
- street/road/sidewalk-family classification depends on matching sampled geometry
- `unknown` is not legal for classified active export rows
- shared sample-center boundary semantics apply to both sampled-surface and building-footprint rasterization

- [ ] **Step 4: Add downstream filter examples**

Include examples:

```sql
WHERE include_in_outdoor_surface_stats = true

WHERE include_in_public_realm_stats = true

WHERE is_building_footprint = false
```

These examples live in docs only. Do not copy explanatory SQL/Python comments into implementation code.

- [ ] **Step 5: Preserve dashboard boundary**

Confirm docs still say `innovation_dashboard` derives tiles/maps/charts from `cells.geoparquet`.

## Task 6: Real Export Smoke And Artifact Verification

**Files:**
- Generated only unless bugs are found.

- [ ] **Step 1: Run real collector**

Run:

```powershell
cd viewer
npm run gis:export-innovation-district -- --out-dir ..\data\gis\Innovation-District
```

Expected:

- raw classification artifact exists under `data/gis/Innovation-District/2025-08-15_2m/raw/`
- metadata references the classification artifact with checksum and shape `[activeCount]`
- active count remains consistent with prior active-mask policy unless intentionally changed
- active/canonical counts remain comparable to the known prior export baseline: `638,688` active rows and `1,492,551` canonical rows, unless a deliberate policy change is recorded in the implementation report

- [ ] **Step 2: Run Python postprocessor**

Run:

```powershell
.\.venv\Scripts\python.exe scripts\postprocess_innovation_district_gis.py `
  --metadata data\gis\Innovation-District\2025-08-15_2m\raw\2025-08-15_2m_active-cells.metadata.json `
  --georef data\3d_models\Innovation-District\innovation_district.georef.json `
  --out-dir data\gis\Innovation-District\2025-08-15_2m `
  --debug-geojson-limit 5000
```

Expected:

- `cells.geoparquet` contains classification columns
- `manifest.json` inventories classification raw artifact
- `qa/debug-sample.geojson` includes classification properties

- [ ] **Step 3: Inspect GeoParquet schema**

Use a short Python/PyArrow check:

```powershell
@'
import pyarrow.parquet as pq
path = r"data\gis\Innovation-District\2025-08-15_2m\cells.geoparquet"
table = pq.read_table(path)
print(table.schema)
print(table.num_rows)
'@ | .\.venv\Scripts\python.exe -
```

Expected: schema includes `surface_flags`, `surface_class`, `is_street_surface`, `is_building_footprint`, `include_in_public_realm_stats`, and `include_in_outdoor_surface_stats`.

Also confirm the schema types are `surface_flags: uint8`, `surface_class: string` (or dictionary-encoded string), and derived columns `bool`.

- [ ] **Step 4: Inspect counts**

Run a small count script:

```powershell
@'
import pyarrow.parquet as pq
path = r"data\gis\Innovation-District\2025-08-15_2m\cells.geoparquet"
d = pq.read_table(path, columns=[
    "is_street_surface",
    "is_building_footprint",
    "include_in_public_realm_stats",
    "include_in_outdoor_surface_stats",
]).to_pydict()
print({k: sum(bool(x) for x in v) for k, v in d.items()})
'@ | .\.venv\Scripts\python.exe -
```

Expected:

- street count is non-zero
- building footprint count is plausible, not all rows
- public-realm count excludes building footprints
- outdoor-surface count excludes building footprints
- `rowCount == activeCount`
- `surfaceFlags.length == activeCount`
- `include_in_public_realm_stats <= is_street_surface`
- `include_in_outdoor_surface_stats == activeRows - buildingFootprintRows`
- `publicRealmRows == streetSurfaceRows - streetAndBuildingOverlapRows`
- no inactive canonical rows are exported
- no building-only canonical index enters `activeCanonicalIndices`
- classification arrays align row-for-row with `canonicalIndices`

- [ ] **Step 5: Perform spatial QA spot checks**

Inspect against the source model/layers or QA map:

- at least one known street/road/sidewalk-family location
- at least one known building-footprint overlap location, if available
- at least one known building-only non-active location, if available

Record the sampled `canonical_index`, `x/y`, `is_street_surface`, and `is_building_footprint` result in the implementation report, plus the non-active finding for any building-only check.

## Task 7: Full Verification

**Files:**
- All modified files.

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
npm run test -- tests/compute/studyAreaMask.test.ts tests/compute/live-selected-hour-session.test.ts tests/gis/innovationDistrictExport.test.ts tests/gis/innovationDistrictCollectorSeam.test.ts
```

Expected: pass.

- [ ] **Step 3: Run type/check**

Run:

```powershell
cd viewer
npm run check
```

Expected: pass, or report exact unrelated blockers.

- [ ] **Step 4: Verify classification invariants**

Run a PowerShell-native Python check that asserts:

```text
rowCount == activeCount
surfaceFlags.length == activeCount
include_in_public_realm_stats <= is_street_surface
include_in_outdoor_surface_stats == activeRows - buildingFootprintRows
publicRealmRows == streetSurfaceRows - streetAndBuildingOverlapRows
no inactive canonical rows are exported
no building-only canonical index enters `activeCanonicalIndices`
classification arrays align with canonicalIndices
```

Expected: pass.

- [ ] **Step 5: Check generated file status**

Run:

```powershell
git status --short
```

Expected: no staged files; generated artifacts are not committed unless user explicitly asks.

## Task 8: Separate Post-Implementation Review Loop

**Files:**
- All implementation diffs.

- [ ] **Step 1: Deslop review after implementation**

Dispatch a separate reviewer using `deslop` criteria:

- remove unnecessary comments
- remove abnormal defensive checks
- remove cast-heavy or `any` shortcuts
- simplify nested conditionals
- ensure style matches surrounding files

Fix findings and rerun until pass or explicit disagreement is documented.

- [ ] **Step 2: Thermonuclear maintainability review after implementation**

Dispatch a separate reviewer using `thermo-nuclear-code-quality-review` criteria:

- no structural regression
- no file pushed past 1000 lines without decomposition
- no ad-hoc branching in busy runtime paths
- no feature logic leaked into wrong layers
- no unnecessary wrappers or casts
- clear canonical helper ownership

Fix findings and rerun until pass or explicit disagreement is documented.

- [ ] **Step 3: Final implementation report**

Report:

- modified files
- generated files
- verification commands and outcomes
- active/canonical counts
- classification counts
- GeoParquet schema confirmation
- reviewer loop outcomes
- residual risks

Do not commit.
