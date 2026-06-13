# Innovation District WebGPU Project Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `data/3d_models/Innovation-District/innovation_district.glb` as a selectable viewer project that works with live WebGPU Shading Index and UTCI, without running the Python analysis/export path.

**Architecture:** Treat Innovation District as a metadata-backed live WebGPU project: the viewer gets a normal `analysisId`, generated metadata, model path, bounds, and Beer Sheva EPW weather, while live WebGPU generates the grid and computes Shading/UTCI at runtime. Generate metadata programmatically from GLB structure so future models can follow the same path without Python export. Separate visual layer identity from compute eligibility so street/train/ground layers can render distinctly without becoming solar occluders.

**Tech Stack:** SvelteKit/Svelte 5, Three.js/Threlte GLTF loading, browser WebGPU compute, Vitest, Playwright only if manual browser verification is needed.

---

## Confirmed Input Facts

- New model path: `data/3d_models/Innovation-District/innovation_district.glb`.
- GLB raw layers found: `trees_canopy`, `existing_buildings`, `ground`, `district_outline`, `street`, `train_tracks`.
- `trees_point` is not present in the GLB JSON and there are no glTF `POINTS` primitives.
- `district_outline` is line-only and should be ignored.
- `street` and `train_tracks` are visual ground-family layers. They should not occlude Shading/UTCI for now.
- `trees_canopy` and `existing_buildings` should be compute occluders.
- Weather should use the same Beer Sheva EPW as `Ben-Gurion`.
- Do not commit. Do not create a git worktree.

## File Structure

- Modify `viewer/src/lib/config/projects.ts`: register Innovation District as a project/model.
- Modify `viewer/src/lib/compute/weather/projectWeather.ts`: map Innovation District to the Beer Sheva EPW.
- Modify or create `viewer/src/lib/types/layerMaterials.ts`: add stable visual layer ids/materials/default visibility for canopy, train tracks, and possibly project-aware/default semantics.
- Modify `viewer/src/lib/services/modelLoaderService.ts`: normalize Innovation District raw names and attach compute eligibility metadata to meshes/merged meshes.
- Create `viewer/scripts/generate-live-webgpu-metadata.ts`: generate metadata-only analysis JSON from a GLB, project id, weather profile, grid size, and layer-bound policy.
- Modify `viewer/src/lib/compute/gpu/mergeAndBvhWorkerClient.ts`: filter BVH payload to explicitly ineligible meshes only, while preserving old behavior for meshes without eligibility metadata.
- Create `data/analyses/Innovation-District/innovation_district_webgpu.json`: generated metadata-only analysis for live WebGPU.
- Modify `data/analyses/manifest.json` only if the existing app/test surface requires it for project discovery.
- Modify tests:
  - `viewer/tests/projectsConfig.test.ts`
  - `viewer/tests/compute/weather-index-alignment.test.ts` or a focused project-weather test if present
  - `viewer/tests/services/modelLoaderService.test.ts`
  - `viewer/tests/compute/mergeAndBvhWorkerClient.test.ts` if present, otherwise add a focused unit test near compute tests
  - Route/project-selection tests only if existing tests assert the full project list

## Task 1: Add Programmatic Metadata Generation For Live WebGPU

**Files:**
- Create: `viewer/scripts/generate-live-webgpu-metadata.ts`
- Create: `data/analyses/Innovation-District/innovation_district_webgpu.json`
- Test: `viewer/tests/analysisPaths.test.ts` or create a focused script/unit test if the repo has a scripts test pattern
- Possibly modify: `data/analyses/manifest.json`

- [ ] **Step 1: Add a metadata generator script**

Create `viewer/scripts/generate-live-webgpu-metadata.ts`. It should:

- read binary `.glb` files directly,
- parse the JSON chunk,
- traverse scene/node/mesh/accessor relationships,
- group primitive bounds by raw top-level layer name,
- infer default layer roles from raw layer names so a prepared GLB can be added with minimal configuration,
- support optional overrides for ambiguous edge cases,
- compute metadata `bounds`,
- compute exact `num_positions` for the selected grid size using the same canonical grid logic as the viewer,
- emit a validation report listing inferred sampling-surface layers, compute-occluder layers, ignored layers, unknown layers, bounds, and point count,
- write formatted JSON to `data/analyses/<Project>/<analysis>.json`.

Use `tsx` for execution, matching existing viewer scripts.

- [ ] **Step 2: Add generator options for reusable future models**

Required CLI options:

```powershell
Set-Location viewer
npx tsx scripts/generate-live-webgpu-metadata.ts `
  --model ../data/3d_models/Innovation-District/innovation_district.glb `
  --out ../data/analyses/Innovation-District/innovation_district_webgpu.json `
  --analysis-id innovation_district_webgpu `
  --project-id Innovation-District `
  --grid-size 2 `
  --date 20250815 `
  --coordinate-system xy_ground `
  --sample-height 1.5 `
  --weather-profile beer-sheva
```

Optional override example for unusual models:

```powershell
  --sampling-layers ground,street,train_tracks `
  --occluder-layers existing_buildings,trees_canopy `
  --ignored-layers district_outline
```

The normal path should not need those overrides when layer names follow the supported naming vocabulary.

The `beer-sheva` profile should expand to:

```json
{
  "epw_file": "data/weather/ISR_D_Beer.Sheva.401900_TMYx/ISR_D_Beer.Sheva.401900_TMYx.epw",
  "location": {
    "latitude": 31.2515,
    "longitude": 34.7995,
    "timezone": 2.0,
    "city": "Beer.Sheva"
  }
}
```

- [ ] **Step 3: Infer sampling bounds automatically**

The generator should infer sampling-surface layers from names such as `ground`, `terrain`, `street`, `road`, `roads`, `sidewalk`, `sidewalks`, `parking`, `walkway`, `train_tracks`, and `train track(s)`.

It should infer compute-occluder layers from names such as `existing_buildings`, `building(s)`, `new_building(s)`, `trees_canopy`, `tree_canopy`, `trees`, `vegetation`, `new_trees`, and future `trees_point` only if that future layer is represented as actual occluding geometry.

It should infer ignored layers from names such as `district_outline`, `outline`, and line-only primitives that have no sampling/occlusion role.

For Innovation District, the inferred sampling-surface layers should be `ground`, `street`, and `train_tracks`; this is a validation expectation, not a manual setup requirement. Do not use `district_outline` for bounds unless explicitly overridden later.

Expected full-envelope bounds from automatic inference:

```json
{
  "x_min": 180591.05,
  "x_max": 183188.48,
  "y_min": 573608.4,
  "y_max": 575905.44,
  "z": 1.5
}
```

Rationale: `z` is pedestrian sampling height in original Z-up metadata terms, matching existing analyses. `x/y` bounds cover the ground/street envelope.

Important performance note: the full-envelope bounds at `grid_size: 2.0` produce roughly `1.49M` points. That is larger than Ness Tziona's current 2m metadata count (`511,840`) and much larger than Ben-Gurion's (`104,445`). Keep `2m` as the default because that is the project default, but require runtime verification before calling the slice acceptable.

- [ ] **Step 4: Generate metadata JSON**

Expected generated output shape:

```json
{
  "analysis_id": "innovation_district_webgpu",
  "date": "20250815",
  "grid_size": 2.0,
  "analysis_type": "full_day",
  "hours": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
  "bounds": {
    "x_min": 180591.05,
    "x_max": 183188.48,
    "y_min": 573608.4,
    "y_max": 575905.44,
    "z": 1.5
  },
  "utci_range": { "min": 0, "max": 0, "mean": 0, "std": 0 },
  "num_positions": 1492551,
  "model_file": "data/3d_models/Innovation-District/innovation_district.glb",
  "epw_file": "data/weather/ISR_D_Beer.Sheva.401900_TMYx/ISR_D_Beer.Sheva.401900_TMYx.epw",
  "generation_date": "<current ISO timestamp>",
  "runtime_seconds": 0,
  "coordinate_system": "xy_ground",
  "hour_statistics": [],
  "has_shading_index": false,
  "shading_index_range": { "min": 0, "max": 1 },
  "location": {
    "latitude": 31.2515,
    "longitude": 34.7995,
    "timezone": 2.0,
    "city": "Beer.Sheva"
  }
}
```

The exact `num_positions` must come from the same grid-counting logic the viewer uses. Do not hand-maintain it. If the bounds policy or grid size changes, regenerate metadata.

- [ ] **Step 5: Add generator regression tests**

Add tests around the generator's pure helpers:

- GLB layer inventory returns expected Innovation District raw layers.
- Bounds from `ground,street,train_tracks` match expected full-envelope values within a small tolerance.
- `grid_size: 2` produces the expected derived point count.
- Missing requested bounds layers throw a clear error.

- [ ] **Step 6: Verify metadata loading still enforces consistency**

Run:

```powershell
Set-Location viewer
npm test -- tests/stores/analysisStore.test.ts tests/projectsConfig.test.ts
```

Expected: existing metadata consistency behavior still passes. Because generated metadata contains exact `num_positions`, `loadAnalysisMetadataOnly()` does not need to accept placeholder counts for this project.

## Task 2: Register Project And Weather

**Files:**
- Modify: `viewer/src/lib/config/projects.ts`
- Modify: `viewer/src/lib/compute/weather/projectWeather.ts`
- Test: `viewer/tests/projectsConfig.test.ts`
- Test: existing weather/project test or add a focused one if none exists

- [ ] **Step 1: Add failing project config test**

Add expectations:

```ts
it('includes Innovation District as a live WebGPU project', () => {
  const project = projects.find((p) => p.id === 'Innovation-District');
  expect(project?.label).toBe('Innovation District');
  expect(project?.defaultAnalysisId).toBe('Innovation-District/innovation_district_webgpu');
  expect(project?.models).toEqual([
    {
      id: 'base',
      label: 'Base',
      analysisId: 'Innovation-District/innovation_district_webgpu'
    }
  ]);
});
```

- [ ] **Step 2: Add project config**

Append a new `ProjectConfig` entry after Ness Tziona:

```ts
{
  id: 'Innovation-District',
  label: 'Innovation District',
  defaultAnalysisId: 'Innovation-District/innovation_district_webgpu',
  models: [
    {
      id: 'base',
      label: 'Base',
      analysisId: 'Innovation-District/innovation_district_webgpu'
    }
  ]
}
```

- [ ] **Step 3: Add weather mapping**

Add to `PROJECT_EPW_PATHS`:

```ts
'Innovation-District':
  '/data/weather/ISR_D_Beer.Sheva.401900_TMYx/ISR_D_Beer.Sheva.401900_TMYx.epw'
```

- [ ] **Step 4: Run focused tests**

Run:

```powershell
Set-Location viewer
npm test -- tests/projectsConfig.test.ts tests/compute/weather-index-alignment.test.ts
```

Expected: project and weather tests pass.

## Task 3: Add Visual Layer Semantics

**Files:**
- Modify: `viewer/src/lib/types/layerMaterials.ts`
- Modify: `viewer/src/lib/services/materialPool.ts` only if existing material config cannot express needed styles
- Test: `viewer/tests/services/modelLoaderService.test.ts`

- [ ] **Step 1: Add failing layer mapping tests**

Add expectations:

```ts
expect(mapLayerNameToType('existing_buildings')).toBe('building');
expect(mapLayerNameToType('trees_canopy')).toBe('vegetation');
expect(mapLayerNameToType('street')).toBe('road');
expect(mapLayerNameToType('train_tracks')).toBe('train_track');
expect(mapLayerNameToType('district_outline')).toBe('ignored');
```

Also include a future-compatible nonblocking alias:

```ts
expect(mapLayerNameToType('trees_point')).toBe('new_vegetation');
```

- [ ] **Step 2: Add material entries**

Add `train_track` and `ignored` to `LAYER_MATERIALS`.

Suggested visuals:

```ts
train_track: {
  color: '#2f3437',
  opacity: 0.85,
  displayName: 'Train Tracks',
  materialType: 'standard',
  polygonOffset: true
},
ignored: {
  color: '#000000',
  opacity: 0,
  displayName: 'Ignored',
  materialType: 'standard'
}
```

Keep `street` mapped to existing `road`, and keep `ground` mapped to `base`.

- [ ] **Step 3: Add mapping entries**

Add:

```ts
'existing_buildings': 'building',
'trees_canopy': 'vegetation',
'trees_camopy': 'vegetation',
'tree_canopy': 'vegetation',
'train_tracks': 'train_track',
'train track': 'train_track',
'train tracks': 'train_track',
'district_outline': 'ignored',
'trees_point': 'new_vegetation'
```

The `trees_camopy` typo alias is deliberate because the original request mentioned it.

- [ ] **Step 4: Update UI layer order**

Add `train_track` to `STANDARD_LAYER_TYPES` near roads/ground:

```ts
{ id: 'train_track', displayName: 'Train Tracks', defaultVisible: true }
```

Do not expose `ignored` in the normal UI unless tests show the layer manager needs every discovered id. Prefer to remove ignored objects or keep them hidden.

- [ ] **Step 5: Run focused tests**

Run:

```powershell
Set-Location viewer
npm test -- tests/services/modelLoaderService.test.ts tests/services/layerManagerService.test.ts
```

Expected: layer mapping and layer manager tests pass.

## Task 4: Separate Display Layers From Compute Occluders

**Files:**
- Modify: `viewer/src/lib/services/modelLoaderService.ts`
- Modify: `viewer/src/lib/compute/gpu/mergeAndBvhWorkerClient.ts`
- Modify: `viewer/src/lib/compute/selected-hour/liveUtciSelectedHourSession.ts` only if needed to pass metadata bounds/estimated point counts into payload preflight cleanly
- Test: `viewer/tests/services/modelLoaderService.test.ts`
- Test: existing or new compute payload test near `viewer/tests/compute/mergeAndBvhWorkerClient.test.ts`

- [ ] **Step 1: Add a raw-layer-aware compute eligibility helper**

In `modelLoaderService.ts`, add:

```ts
export function resolveComputeBvhEligibility(params: {
  rawLayerName: string;
  layerType: string;
}): boolean | undefined {
  const raw = params.rawLayerName.toLowerCase();
  if (raw === 'ground' || raw === 'street' || raw === 'train_tracks' || raw === 'district_outline') {
    return false;
  }
  if (
    params.layerType === 'building' ||
    params.layerType === 'new_building' ||
    params.layerType === 'vegetation' ||
    params.layerType === 'new_vegetation'
  ) {
    return true;
  }
  return undefined;
}
```

This avoids a broad global rule that would make every `base`, `road`, or `unknown` layer in older models non-occluding. Only known Innovation District visual-only raw layers get explicit `false`. Missing eligibility must preserve old behavior in compute payload helpers.

- [ ] **Step 2: Attach compute metadata during material application**

When processing each mesh:

```ts
const computeEligibility = resolveComputeBvhEligibility({ rawLayerName: layerName, layerType });
if (computeEligibility !== undefined) {
  child.userData.includeInComputeBvh = computeEligibility;
}
```

When creating merged meshes in both merge functions:

```ts
const eligibilityValues = meshes
  .map((mesh) => mesh.userData.includeInComputeBvh)
  .filter((value) => value !== undefined);
if (eligibilityValues.length > 0) {
  mergedMesh.userData.includeInComputeBvh = eligibilityValues.some(Boolean);
}
```

When unknown layers are remapped to `base`, leave `includeInComputeBvh` unset unless the raw layer name explicitly resolved to `false`.

- [ ] **Step 3: Remove ignored visual objects**

During traversal, if `layerType === 'ignored'`, queue the object for removal and do not include it in merge groups.

This makes `district_outline` ignored by policy, not just incidentally removed because it is a line.

- [ ] **Step 4: Add tests for compute metadata**

Add model-loader tests proving:

```ts
expect(buildingMesh.userData.includeInComputeBvh).toBe(true);
expect(treeCanopyMesh.userData.includeInComputeBvh).toBe(true);
expect(streetMesh.userData.includeInComputeBvh).toBe(false);
expect(trainTrackMesh.userData.includeInComputeBvh).toBe(false);
expect(groundMesh.userData.includeInComputeBvh).toBe(false);
expect(legacyUnknownRemappedToBase.userData.includeInComputeBvh).toBeUndefined();
```

- [ ] **Step 5: Filter BVH payload**

In both `prepareMeshPayloadForWorkerAsync()` and the synchronous `prepareMeshPayloadForWorker()` helper in `mergeAndBvhWorkerClient.ts`, skip meshes where:

```ts
mesh.userData.includeInComputeBvh === false
```

Important:

- Only an explicit `false` should skip.
- Missing metadata should preserve old behavior for existing tests/unprocessed models.
- The skip must happen before mesh list insertion, triangle counting, and compute mesh bounds accumulation.

- [ ] **Step 6: Keep grid/budget preflight tied to metadata bounds**

Do not let filtered occluder bounds shrink the live grid estimate. The live selected-hour path builds the grid from metadata `analysisBounds`, so budget/preflight should use either:

- explicit metadata bounds passed into payload preparation, or
- an explicit estimated grid point count derived from metadata bounds and selected grid resolution.

Add a test where a ground/street mesh has a larger extent than the building/canopy occluder mesh. Expected:

- BVH mesh payload excludes ground/street triangles.
- Preflight/grid estimate still reflects the metadata/full analysis bounds, not just the occluder mesh bounds.

- [ ] **Step 7: Add compute payload filter test**

Create a small scene with one building mesh and one road/train/ground mesh. Assert payload triangles include only the building when `includeInComputeBvh === false` is set on the ground-family mesh.

- [ ] **Step 8: Add cache/clone preservation test**

Add a focused test covering the processed scene clone path used by `Model.svelte` cache behavior:

```ts
const clone = processedScene.clone(true);
expect(findLayer(clone, 'building')?.userData.includeInComputeBvh).toBe(true);
expect(findLayer(clone, 'vegetation')?.userData.includeInComputeBvh).toBe(true);
expect(findLayer(clone, 'road')?.userData.includeInComputeBvh).toBe(false);
expect(findLayer(clone, 'train_track')?.userData.includeInComputeBvh).toBe(false);
expect(findLayer(clone, 'base')?.userData.includeInComputeBvh).toBe(false);
```

- [ ] **Step 9: Add line primitive guard test**

Create a `THREE.LineSegments` or `THREE.Line` under a `district_outline` group. Assert after `applyLayerMaterials()`:

- the line object is removed from the model,
- `ignored` is not discovered by the layer manager,
- the compute payload has no geometry from the outline.

- [ ] **Step 10: Run focused tests**

Run:

```powershell
Set-Location viewer
npm test -- tests/services/modelLoaderService.test.ts tests/compute/mergeAndBvhWorkerClient.test.ts
```

If the exact compute test file does not exist, run the closest focused compute tests around mesh payload/BVH.

## Task 5: Route And Runtime Verification

**Files:**
- No implementation files unless tests expose gaps.
- Test: `viewer/tests/routes/main-route-model-selection.test.ts`
- Optional test: add a route test for selecting Innovation District if existing harness makes that cheap.

- [ ] **Step 1: Verify model path resolution**

Add/adjust a test asserting:

```ts
expect(resolveAnalysisModelPath({
  model_file: 'data/3d_models/Innovation-District/innovation_district.glb',
  source_analysis_id: 'Innovation-District/innovation_district_webgpu'
})).toBe('data/3d_models/Innovation-District/innovation_district.glb');
```

- [ ] **Step 2: Run route/model selection tests**

Run:

```powershell
Set-Location viewer
npm test -- tests/routes/main-route-model-selection.test.ts tests/projectsConfig.test.ts
```

Expected: selection and model-path behavior pass.

- [ ] **Step 3: Start dev server for manual verification**

Run:

```powershell
Set-Location viewer
npm run dev -- --host 127.0.0.1
```

Open:

```text
http://127.0.0.1:5173/?analysis=Innovation-District%2Finnovation_district_webgpu&gridResolution=2&utciRender=auto&utciRenderDiagnostics=1
```

- [ ] **Step 4: Verify visual loading**

Expected:

- Model loads without missing metadata errors.
- Camera frames the Innovation District model after normalization.
- Layers include buildings, trees, roads/street, train tracks, and ground.
- `district_outline` is not visible.
- No `unknown` layer appears for the expected Innovation District layers.

- [ ] **Step 5: Verify live Shading Index**

Switch metric to Shading Index.

Expected:

- WebGPU live route starts.
- Shading surface appears.
- Diagnostics show `rendererBackend=webgpu` where available.
- Diagnostics show a reasonable derived point count for the selected `gridResolution`.
- Diagnostics expose `effectiveGridResolution` and first selected-hour readiness timings.
- No error about missing Beer Sheva EPW mapping.

- [ ] **Step 6: Verify live UTCI**

Switch metric to UTCI.

Expected:

- UTCI surface appears for the selected hour.
- No CPU `.bin` fetch is required for the Innovation District route.
- If WebGPU is unavailable, the app shows the existing capability/fallback behavior rather than a project-specific crash.
- Treat the slice as acceptable only after recording derived `num_positions`, `effectiveGridResolution`, and first selected-hour readiness from diagnostics.

## Task 6: Guardrails And Documentation Notes

**Files:**
- Modify: `docs/webgpu_strategy_analysis.md` only if the implementation materially changes the live WebGPU contract.
- Optional create: `docs/superpowers/plans/2026-06-11-innovation-district-webgpu-project-results.md` after implementation, if useful.

- [ ] **Step 1: Document the new compute contract if code changes are non-obvious**

Add a short note somewhere appropriate:

```md
Innovation District uses visual ground-family layers (`ground`, `street`, `train_tracks`) that are rendered for context but excluded from the compute BVH. Buildings and canopy vegetation remain compute occluders.
```

- [ ] **Step 2: Record the `trees_point` finding**

If adding comments/docs, state that `trees_point` was requested but not present in the current GLB; `trees_point` is supported as a future alias only.

- [ ] **Step 3: Final verification commands**

Run:

```powershell
Set-Location viewer
npm test -- tests/projectsConfig.test.ts tests/services/modelLoaderService.test.ts tests/routes/main-route-model-selection.test.ts
```

Run any added compute payload test.

If Svelte files were edited, also run:

```powershell
Set-Location viewer
npx @sveltejs/mcp svelte-autofixer ./src/lib/components/ui/ProjectSelector.svelte --svelte-version 5
```

Only run broader Playwright smoke tests if manual browser verification or route tests expose a UI integration risk.

## Non-Goals

- No Python export path.
- No `.bin` UTCI/Shading payload for Innovation District.
- No scenario variants for Innovation District.
- No generated 3D tree instances for `trees_point` until the layer exists in a future GLB.
- No commits or git worktrees.

## Review Checklist Before Implementation

- [ ] Separate reviewer confirms the plan does not accidentally make street/train/ground compute occluders.
- [ ] Separate reviewer confirms generated metadata has exact `bounds` and `num_positions` for the 2m live WebGPU route.
- [ ] Separate reviewer confirms Three.js loader semantics survive merge/cache paths and `includeInComputeBvh` is preserved on merged meshes.

---

## Follow-Up Remediation Plan: Active Base Mask, Layer Rendering, Then BG/ID Ray Diagnostics

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this follow-up task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct Innovation District live WebGPU so it computes and renders only real base/ground study cells, keeps street/train visual context above UTCI, and defers BG-vs-ID mismatch investigation until after the mask/layer fixes pass manual review.

**Architecture:** Keep the metadata rectangular grid as the canonical address space for Ladybug-style parity, but add a base-layer active-cell mask that compacts compute to real study cells and maps active outputs back into the canonical grid for rendering/tooltips. Treat road/street as an outline annotation over UTCI, keep train tracks as ordinary geometry above UTCI for now, and only after manual review add a ray-oracle diagnostic for BG-vs-ID differences.

**Tech Stack:** SvelteKit/Svelte 5, Three.js, browser WebGPU compute, WGSL storage buffers, Vitest, optional Playwright/browser diagnostics on the main route.

### Follow-Up Confirmed Decisions

- Use the `ground`/base layer as the single active study-area mask source for Innovation District.
- Do not include `street` or `train_tracks` in the active mask if they are contained by the base footprint.
- Keep the rectangular metadata bounds and `num_positions` for static metadata consistency.
- Compact live compute to active cells only; render inactive canonical cells transparent and ignore them in tooltips/ranges.
- Do not try to fix the BG-vs-ID mismatch before a human manually reviews the mask and layer rendering fixes.
- Do not start BG-vs-ID diagnostics until the user explicitly approves moving to that phase.
- Do not commit until the user reviews and explicitly approves the first fix slice.
- No git worktrees.

### Follow-Up File Structure

- Create: `viewer/src/lib/compute/core/studyAreaMask.ts`
- Test: `viewer/tests/compute/studyAreaMask.test.ts`
- Modify: `viewer/src/lib/compute/core/canonicalGrid.ts`
- Modify: `viewer/src/lib/compute/compute-manager.ts`
- Modify: `viewer/src/lib/compute/selected-hour/liveUtciSelectedHourSession.ts`
- Modify: `viewer/src/lib/types/analysis.ts`
- Modify: `viewer/src/lib/services/pointCloudService.ts`
- Modify: `viewer/src/lib/services/tooltipService.ts`
- Modify: `viewer/src/lib/services/gpuUtciRenderBridge.ts` only if the existing render bridge cannot consume active-to-canonical mapping cleanly.
- Modify: `viewer/src/lib/components/ui/LayerControls.svelte`
- Modify: `viewer/src/lib/types/layerMaterials.ts`
- Modify: `viewer/src/lib/services/materialPool.ts`
- Modify: `viewer/src/lib/services/modelLoaderService.ts`
- Optional create: `viewer/src/lib/services/layerOutlineService.ts`
- Optional diagnostic create, only after manual review: `viewer/scripts/compare-bg-id-ray-oracle.ts`
- Optional diagnostic output create, only after manual review: `docs/superpowers/plans/2026-06-11-innovation-district-bg-id-ray-diagnostics.md`

## Task 7: Base-Layer Study Area Mask

**Files:**
- Create: `viewer/src/lib/compute/core/studyAreaMask.ts`
- Test: `viewer/tests/compute/studyAreaMask.test.ts`
- Modify: `viewer/src/lib/compute/core/canonicalGrid.ts`
- Modify: `viewer/src/lib/types/analysis.ts`

- [ ] **Step 1: Write failing unit tests for canonical active masks**

Create `viewer/tests/compute/studyAreaMask.test.ts` with small synthetic footprints:

```ts
import { describe, expect, it } from 'vitest';
import { buildStudyAreaMaskFromProjectedTriangles } from '$lib/compute/core/studyAreaMask';

describe('buildStudyAreaMaskFromProjectedTriangles', () => {
	it('marks only canonical grid cells inside the base footprint', () => {
		const result = buildStudyAreaMaskFromProjectedTriangles({
			bounds: { x_min: 0, x_max: 4, y_min: 0, y_max: 4, z: 1.5 },
			gridSize: 2,
			coordinateSystem: 'xy_ground',
			triangles: [
				[0, 0, 4, 0, 0, 4],
				[4, 0, 4, 4, 0, 4]
			]
		});

		expect(result.canonicalPointCount).toBe(9);
		expect([...result.activeCanonicalIndices]).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8]);
		expect(result.activePointCount).toBe(9);
	});

	it('keeps holes/outside rectangle cells inactive', () => {
		const result = buildStudyAreaMaskFromProjectedTriangles({
			bounds: { x_min: 0, x_max: 4, y_min: 0, y_max: 4, z: 1.5 },
			gridSize: 2,
			coordinateSystem: 'xy_ground',
			triangles: [[0, 0, 2, 0, 0, 2]]
		});

		expect(result.canonicalPointCount).toBe(9);
		expect(result.activePointCount).toBeLessThan(9);
		expect(result.mask.some((active) => !active)).toBe(true);
	});
});
```

- [ ] **Step 2: Run the failing mask tests**

Run:

```powershell
Set-Location viewer
npm test -- tests/compute/studyAreaMask.test.ts
```

Expected: FAIL because `studyAreaMask.ts` does not exist.

- [ ] **Step 3: Implement the mask helper**

Create `viewer/src/lib/compute/core/studyAreaMask.ts`.

Requirements:

- Accept metadata `bounds`, `gridSize`, `coordinateSystem`, and projected base-layer triangles.
- Preserve the same canonical point ordering as `canonicalGridPoints()`.
- Return:
  - `canonicalPointCount`
  - `activePointCount`
  - `mask: Uint8Array`
  - `activeCanonicalIndices: Uint32Array`
  - `width`
  - `height`
  - a deterministic checksum or signature fields suitable for diagnostics/cache keys.
- Treat boundary points as active with a small epsilon.
- Keep the helper independent from Three.js if practical; pass plain projected triangles so it stays unit-testable.

- [ ] **Step 4: Add canonical grid support for active index projection**

Modify `viewer/src/lib/compute/core/canonicalGrid.ts` with a helper that can generate compact grid points from active canonical indices:

```ts
export function canonicalGridPointsForActiveIndices(params: CanonicalGridParams & {
	activeCanonicalIndices: Uint32Array;
}): CanonicalGridResult
```

Expected behavior:

- Same coordinate transform and origin offset as `canonicalGridPoints()`.
- Output length is `activeCanonicalIndices.length * 3`.
- No change to existing `canonicalGridPoints()` behavior.

- [ ] **Step 5: Run focused tests**

Run:

```powershell
Set-Location viewer
npm test -- tests/compute/studyAreaMask.test.ts tests/compute/core/analysisGridFromBounds.test.ts
```

Expected: PASS.

## Task 8: Wire Active Mask Into Live WebGPU Compute And Rendering

**Files:**
- Modify: `viewer/src/lib/compute/selected-hour/liveUtciSelectedHourSession.ts`
- Modify: `viewer/src/lib/compute/compute-manager.ts`
- Modify: `viewer/src/lib/types/analysis.ts`
- Modify: `viewer/src/lib/services/pointCloudService.ts`
- Modify: `viewer/src/lib/services/tooltipService.ts`
- Modify: `viewer/src/lib/services/gpuUtciRenderBridge.ts` only if required by the render layout.
- Test: add/modify focused selected-hour and render-layout tests.

- [ ] **Step 1: Add failing tests for active-count diagnostics and inactive-cell transparency**

Add focused tests near existing selected-hour/render tests asserting:

- Live selected-hour diagnostics expose `canonicalPointCount`, `activePointCount`, and `activeMaskSource: 'base'`.
- Active compute values remain compact.
- Render layout can place active values back into canonical cells.
- Inactive cells render transparent/no-data.
- Tooltip lookup ignores inactive cells.

- [ ] **Step 2: Run the focused tests to verify failure**

Run the smallest relevant set after adding tests, for example:

```powershell
Set-Location viewer
npm test -- tests/compute/live-selected-hour-session.test.ts tests/scene/utciComputeBufferRenderBridge.test.ts
```

Expected: FAIL on missing active mask metadata/mapping behavior.

- [ ] **Step 3: Extract base-layer projected triangles during live session preparation**

In `viewer/src/lib/compute/selected-hour/liveUtciSelectedHourSession.ts`, during the model/payload preparation phase:

- Traverse loaded model meshes.
- Select meshes whose `userData.layerType === 'base'` or raw normalized layer name is `ground`.
- Apply each mesh `matrixWorld`.
- Project triangles into analysis XY footprint coordinates.
- Feed those triangles into `buildStudyAreaMaskFromProjectedTriangles()`.
- Do not include `road`, `train_track`, `street`, `district_outline`, buildings, or vegetation in the active mask.

- [ ] **Step 4: Compact compute grid to active cells**

Modify `viewer/src/lib/compute/compute-manager.ts` so `initFromModelAndWeather()` accepts optional `activeCanonicalIndices`.

Expected:

- Without `activeCanonicalIndices`, behavior is unchanged.
- With `activeCanonicalIndices`, GPU buffers and exposure dispatch operate on active points only.
- Diagnostics still preserve the full canonical count.
- The active output count is used for UTCI/Shading calculations and range summaries.

- [ ] **Step 5: Preserve canonical render placement**

Modify render layout code so active outputs are mapped back to canonical grid cells.

Expected:

- `activeIndex -> canonicalIndex` controls cell placement.
- Inactive canonical cells have alpha `0`.
- The UTCI/Shading surface no longer appears outside the base footprint.
- Tooltips return no value for inactive cells.
- Range summaries ignore inactive cells because inactive cells are never computed.

- [ ] **Step 6: Add runtime diagnostics**

Expose these fields in existing selected-hour diagnostics:

```ts
activeMaskSource: 'base',
canonicalPointCount: number,
activePointCount: number,
inactivePointCount: number,
activePointRatio: number,
activeMaskChecksum: string
```

Also include active mask counts in console diagnostics only if the route already logs selected-hour diagnostics.

- [ ] **Step 7: Verify 2m and 0.5m behavior without claiming 0.5m success prematurely**

Open:

```text
http://localhost:5173/?analysis=Innovation-District%2Finnovation_district_webgpu&gridResolution=2&utciRender=auto&utciRenderDiagnostics=1
```

Expected:

- Active point count is lower than canonical point count.
- UTCI/Shading does not render outside the base footprint.
- Existing 2m route remains stable.

Then try:

```text
http://localhost:5173/?analysis=Innovation-District%2Finnovation_district_webgpu&gridResolution=0.5&utciRender=auto&utciRenderDiagnostics=1
```

Expected:

- If active count is still too large for this device, the route fails gracefully or falls back clearly.
- Do not treat 0.5m as accepted unless diagnostics prove active buffers fit the current WebGPU limits and the first selected hour renders.

## Task 9: Road Outline And Train Track Render Order

**Files:**
- Modify: `viewer/src/lib/types/layerMaterials.ts`
- Modify: `viewer/src/lib/components/ui/LayerControls.svelte`
- Modify: `viewer/src/lib/services/materialPool.ts`
- Modify: `viewer/src/lib/services/modelLoaderService.ts`
- Optional create: `viewer/src/lib/services/layerOutlineService.ts`
- Test: `viewer/tests/layerControls.test.ts`
- Test: add/update focused model loader/material tests if existing helpers make this cheap.

- [ ] **Step 1: Write failing tests for road visibility and train track visibility policy**

Tests should assert:

- `road` is not hidden from the layer UI for Innovation District.
- Road/street is visible as context by default only if the product decision is to show the road outline on first load.
- Train tracks remain visible by default.
- Train track meshes get a render policy that can appear above UTCI.

- [ ] **Step 2: Run focused layer tests**

Run:

```powershell
Set-Location viewer
npm test -- tests/layerControls.test.ts tests/services/modelLoaderService.test.ts
```

Expected: FAIL until layer policy is implemented.

- [ ] **Step 3: Implement road/street outline rendering**

Preferred approach:

- Keep raw `street` mapped to standard layer type `road`.
- Convert road/street visual mesh to an outline/edge overlay, or add a sibling line overlay and hide the filled road material.
- Set road/street outline render behavior deliberately so the outline appears above UTCI without filling or hiding the UTCI field.
- Do not add road/street to the compute BVH.
- Do not let road/street fill hide UTCI.
- Keep outline color and opacity readable over the UTCI palette.

Only create `viewer/src/lib/services/layerOutlineService.ts` if the outline conversion would otherwise bloat `modelLoaderService.ts`.

- [ ] **Step 4: Keep train tracks ordinary geometry above UTCI**

Train track policy for this slice:

- Keep train tracks as geometry, not outline-only.
- Keep train tracks excluded from compute BVH.
- Set render behavior so train tracks appear above UTCI for now.
- Prefer render ordering/depth policy over arbitrary large position offsets.
- If coplanar base depth still hides train tracks, add a tiny documented project/layer-specific lift only after a visual diagnostic proves it is necessary.

- [ ] **Step 5: Verify tree/UTCI shallow-angle behavior after masking**

Manual check:

- Angle camera shallow over Innovation District trees.
- Confirm UTCI no longer appears as a giant rectangular surface cutting over non-base areas.
- If UTCI still appears above tree canopy within the real base mask, document the exact camera/view and defer a separate depth/sorting fix rather than bundling it with mask/road/train work.

- [ ] **Step 6: Run focused tests**

Run:

```powershell
Set-Location viewer
npm test -- tests/layerControls.test.ts tests/services/modelLoaderService.test.ts tests/compute/mergeAndBvhWorkerClient.test.ts
```

Expected: PASS.

## Task 10: Manual Review Gate Before Commit Or BG/ID Mismatch Work

**Files:**
- Optional create: `docs/superpowers/plans/2026-06-11-innovation-district-remediation-results.md`

- [ ] **Step 1: Run the focused verification set**

Run:

```powershell
Set-Location viewer
npm test -- tests/compute/studyAreaMask.test.ts tests/layerControls.test.ts tests/services/modelLoaderService.test.ts tests/compute/mergeAndBvhWorkerClient.test.ts tests/compute/live-selected-hour-session.test.ts
```

If Svelte files changed, also run the repo's normal Svelte check/autofix command used for this workspace.

- [ ] **Step 2: Browser verify the main route**

Use the already-running dev server if available.

Open:

```text
http://localhost:5173/?analysis=Innovation-District%2Finnovation_district_webgpu&gridResolution=2&utciRender=auto&utciRenderDiagnostics=1
```

Expected:

- Active base mask diagnostics are present.
- UTCI/Shading is clipped to the base footprint.
- Road/street appears as an outline over UTCI.
- Train tracks appear above UTCI.
- Trees do not show an obvious rectangular UTCI sheet passing over them outside the base mask.
- Existing BG/NZ route behavior is not visibly regressed by the shared layer/render changes.

- [ ] **Step 3: Stop and request user review**

Do not commit and do not start BG/ID mismatch work.

Report:

- changed files,
- focused tests run,
- browser diagnostics,
- unresolved visual caveats,
- whether 0.5m succeeded, failed gracefully, or still requires tiling/device-limit work.

Wait for the user to review manually.

- [ ] **Step 4: Commit only after explicit approval**

If and only if the user approves the mask/layer slice, prepare a concise commit message and commit the approved changes.

## Task 11: BG-vs-ID Ray Oracle Diagnostics After Manual Review

**Files:**
- Optional create: `viewer/scripts/compare-bg-id-ray-oracle.ts`
- Optional create: `docs/superpowers/plans/2026-06-11-innovation-district-bg-id-ray-diagnostics.md`
- Modify: no production files unless the diagnostic proves a root cause and the user approves a follow-up implementation plan.

- [ ] **Step 1: Confirm the first fix slice was manually reviewed**

Do not run or implement this task until:

- the base-mask/layer fixes have passed manual review,
- the user has explicitly approved moving to mismatch investigation,
- and, if the user makes commit part of the gate, the first slice has been committed.

- [ ] **Step 2: Write a ray-oracle diagnostic script**

Create `viewer/scripts/compare-bg-id-ray-oracle.ts` only after the manual gate.

The script should accept:

```powershell
npx tsx scripts/compare-bg-id-ray-oracle.ts `
  --bg-analysis Ben-Gurion/20250815_grid_2m_fullday `
  --id-analysis Innovation-District/innovation_district_webgpu `
  --bg-x -3220.617 `
  --bg-y -265.286 `
  --id-x 181077.047 `
  --id-y 574578.375 `
  --month 8 `
  --hour 16
```

It should report:

- nearest canonical/active point for each model,
- distance from requested point to grid point,
- exact sample origin used for raycast,
- sun vector and EPW time index,
- CPU BVH first hit layer/distance for buildings and trees,
- comparison at `z = 1.5` and at the live compute height,
- whether current WebGPU solar exposure bit agrees with the CPU oracle where available.

- [ ] **Step 3: Use the diagnostic to classify the mismatch**

Classify the root cause as one of:

- sample-height mismatch,
- grid phase/nearest-point mismatch,
- tree opacity/occluder semantics mismatch,
- transform/normalization mismatch,
- WGSL BVH/raycast parity issue,
- visualization/range/color-mode issue,
- or unresolved.

- [ ] **Step 4: Stop and discuss before implementing a mismatch fix**

Write findings to:

```text
docs/superpowers/plans/2026-06-11-innovation-district-bg-id-ray-diagnostics.md
```

Do not change production compute/raycast behavior until the diagnostic evidence points to one root cause and the user approves the next fix plan.
