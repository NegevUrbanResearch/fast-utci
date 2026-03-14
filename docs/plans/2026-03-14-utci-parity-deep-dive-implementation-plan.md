# UTCI Parity Deep Dive (Extra Intermediates + Optional Rectangular Grid) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add MRT and weather alignment as extractable intermediates for diagnosis, and add an optional rectangular (same-grid-as-.bin) path so we can separate grid vs formula effects and get comparable "inside building" behaviour.

**Architecture:** (1) Python exports MRT and a small weather sample for the Ben-Gurion base case; WebGPU pipeline adds an MRT readback buffer and exposes it plus weather sample in the debug page; add a loader and optional statistical comparison for MRT. (2) Viewer compute path accepts an optional "use rectangular grid" mode: when analysis metadata has bounds and coordinate_system, generate grid via `createRectangularGridFromBounds` (mapping bounds to viewer Y-up) instead of `generateGridFromMesh`, and pass that grid to the existing pipeline so exposure and MRT/UTCI run on the same point set as .bin (allowing point-level or distribution comparison and fixing "inside = outside").

**Tech Stack:** Python (fast_utci, ladybug-comfort), Node (Playwright, fs), Svelte viewer, WebGPU (WGSL), TypeScript.

---

## Reference

- Design rationale: `docs/plans/2026-03-14-intermediate-stage-validation-design.md` (debugging zero intermediates); grid vs formula separation and "inside building" behaviour from conversation.
- Existing intermediates: `docs/plans/2026-03-14-python-vs-webgpu-exposure-paths.md`; `viewer/src/lib/parity/compareIntermediates.ts`, `loadReferenceIntermediatesFromFs.ts`; `scripts/export_ben_gurion_intermediates.py`.
- Grid: `viewer/src/lib/compute/grid-generator.ts` (`createRectangularGridFromBounds`, `generateGridFromMesh`); `viewer/src/lib/parity/analysisToWorld.ts` (xy_ground → world); analysis metadata `bounds`: `x_min`, `x_max`, `y_min`, `y_max`, `z`.

---

## Part A: Extra intermediates (MRT + weather alignment)

### Task A1: Python – export MRT for Ben-Gurion base

**Files:**
- Modify: `scripts/export_ben_gurion_intermediates.py` (add `--stage mrt`, MRT computation and write)
- Modify: `src/fast_utci/mrt/solarcal.py` or use existing MRT path (see `src/fast_utci/mrt/mrt_calculator.py` / exposure flow)

**Step 1: Identify how Python gets MRT for the analysis**

Inspect how the .bin is produced: `run_analysis.py` uses `MRTCalculator`; trace where MRT is computed (e.g. `compute_mrt_solarcal` in `solarcal.py`) and what inputs it needs (exposure, weather, location, datetimes). Confirm the same code path can be called from an export script with positions from .bin and exposure already exported.

**Step 2: Add MRT export to export script**

In `scripts/export_ben_gurion_intermediates.py`:
- Add `--stage mrt` to the argument parser (alongside `solar` and `sky`).
- When `mrt` is requested: load positions from .bin, load solar/sky reference (or recompute exposure if preferred), load EPW/weather for the analysis day, call the same MRT computation path used by the analysis (e.g. `compute_mrt_solarcal` with `fract_body_exp` from solar export and sky exposure from sky export), then write `{ "numPositions", "numHours", "mrt": number[] }` to `{base_path}_mrt.json`. MRT array layout: point-major flat, same as solar (`[p0_h0, p0_h1, ..., pN_h23]`).

**Step 3: Run export and commit reference file (optional)**

From repo root:  
`python scripts/export_ben_gurion_intermediates.py --base-path data/analyses/Ben-Gurion/20250815_grid_2m_fullday --model data/3d_models/Ben-Gurion/original_with_layers.glb --stage mrt`  
Verify `data/analyses/Ben-Gurion/20250815_grid_2m_fullday_mrt.json` exists and has expected length `numPositions * numHours`. Do not commit unless user wants reference in repo.

---

### Task A2: WebGPU pipeline – add MRT buffer and readback

**Files:**
- Modify: `viewer/src/lib/compute/shaders/mrt_utci.wgsl` (add `mrt_results` storage buffer, write MRT per (point, time))
- Modify: `viewer/src/lib/compute/webgpuUtciPipeline.ts` (create MRT buffer with COPY_SRC, bind in MRT pass, add `readMrtFull`, staging and copy like solar/sky)
- Modify: `viewer/src/lib/compute/gpu-pipeline.ts` (add `mrtBufferSize` to config; add optional `readMrtFull?` to interface)

**Step 1: Add MRT buffer and write in shader**

In `mrt_utci.wgsl`:
- Add `@group(0) @binding(6) var<storage, read_write> mrt_results: array<f32>;` (use next free binding; if 6 conflicts, renumber bindings).
- In `main`, after `let mrt0 = compute_outdoor_mrt(...)`, add `mrt_results[flat_index] = mrt0;` (write the per-hour MRT used for UTCI; no need to write boundary-averaged MRT for diagnosis).

**Step 2: Create and bind MRT buffer in pipeline**

In `webgpuUtciPipeline.ts`:
- Add `private mrtBuffer: GPUBuffer | null = null;` and `private mrtStagingBuffer: GPUBuffer | null = null;`.
- In `uploadStaticData` (or wherever other buffers are sized): compute `mrtBytes = numPoints * totalTimeSteps * 4`; create `mrtBuffer` with `GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC`, same size as UTCI buffer for the same layout.
- In the MRT pass bind group (the one that uses `mrtPipeline`), add binding for `mrtBuffer` (binding index must match the shader).
- After the MRT dispatch, ensure the encoder writes to `mrtBuffer` (the shader already does; no extra copy needed).
- Implement `async readMrtFull(params: { numPoints, numHours, numMonths }): Promise<Float32Array>`: same pattern as `readSolarExposureFull` (onSubmittedWorkDone, create staging buffer if needed, copy from mrtBuffer to staging, submit, mapAsync, copy out, unmap). Add `readMrtFull` to the pipeline interface in `gpu-pipeline.ts`.

**Step 3: Extend pipeline config for MRT buffer size**

In `gpu-pipeline.ts`, add `mrtResultBufferSize: number` to `PipelineConfig` (same as `utciResultBufferSize` for point×time layout) and set it in `createPipelineConfig`. Use it in the WebGPU implementation for buffer creation if needed.

**Step 4: Verify MRT readback**

In the debug page, after exposing `__parityIntermediates__`, if `pipeline.readMrtFull` exists, call it and add `mrt: Array.from(mrtArray)` to `__parityIntermediates__` (or `__parityDebug__.mrt`). Reload debug page and run inspect test; confirm no zeros and plausible range (e.g. 20–60 °C).

---

### Task A3: Weather alignment – expose and optionally compare

**Files:**
- Modify: `viewer/src/routes/debug-webgpu-utci/+page.svelte` (expose first few weather rows used by pipeline)
- Create or modify: `scripts/export_ben_gurion_intermediates.py` or a small script to write `*_weather_sample.json` (first 3 hours: air_temp, direct_normal, diffuse_horizontal, horiz_infrared, wind_speed, rel_humidity)

**Step 1: Expose weather sample on debug page**

In the block where we set `__parityIntermediates__` / `__parityDebug__`, get the weather data that was passed to the pipeline (e.g. from the same place `createLiveUtciAnalysisFromCompute` gets EPW → parsed weather). Expose `window.__parityDebug__.weatherSample = [ first 3 hours ]` as an array of objects with keys `air_temp`, `direct_normal`, `diffuse_horizontal`, `horiz_infrared`, `wind_speed`, `rel_humidity` so tests or manual comparison can check alignment with Python.

**Step 2: Python – export weather sample**

In `scripts/export_ben_gurion_intermediates.py`, add `--stage weather` (or append to a single export): for the same analysis day (e.g. Aug 15) and hours 0,1,2, read the EPW rows used by the Python pipeline and write `{ "numHours": 3, "weather": [ { "air_temp", "direct_normal", "diffuse_horizontal", "horiz_infrared", "wind_speed", "rel_humidity" }, ... ] }` to `{base_path}_weather_sample.json`. Use the same EPW path as in analysis metadata.

**Step 3: Optional test or doc**

Add a short note in `docs/plans/2026-03-14-intermediate-stage-validation-design.md` (or parity harness doc): to check weather alignment, compare `__parityDebug__.weatherSample` with `*_weather_sample.json` (manual or one-off script). No mandatory Playwright assertion unless you add a small tolerance compare.

---

### Task A4: Loader and optional MRT statistical comparison

**Files:**
- Modify: `viewer/src/lib/parity/loadReferenceIntermediatesFromFs.ts` (add `stage: 'mrt'`, load `*_mrt.json`, return `{ numPositions, numHours, mrt: Float32Array }`)
- Modify: `viewer/src/lib/parity/compareIntermediates.ts` (add `compareIntermediatesStats` for MRT: same signature as solar/sky, compare mean/max with tolerance; MRT in °C so use e.g. toleranceMean 1.0, toleranceMax 2.0)
- Modify: `viewer/tests/e2e/parity-intermediates.spec.ts` (if reference `*_mrt.json` exists, read WebGPU MRT from `__parityIntermediates__.mrt` or `__parityDebug__.mrt`, run `compareIntermediatesStats`, assert pass with a clear message)

**Step 1: Loader for MRT reference**

In `loadReferenceIntermediatesFromFs.ts`, add type `MrtReference { numPositions: number; numHours: number; mrt: Float32Array }`. In the function, when `stage === 'mrt'`, load `basePath + '_mrt.json'`, parse, validate shape and length `numPositions * numHours`, return `{ numPositions, numHours, mrt: new Float32Array(arr) }`.

**Step 2: Compare helper for MRT**

In `compareIntermediates.ts`, add a wrapper or reuse `compareIntermediatesStats` for two number arrays (ref mrt, webgpu mrt) with configurable tolerances (e.g. mean 1.0 °C, max 2.0 °C). Reuse the same stats structure (mean, max, pass/fail).

**Step 3: E2E assert MRT when reference exists**

In `parity-intermediates.spec.ts`, after solar/sky checks: if `existsSync(BASE_PATH + '_mrt.json')` and `__parityIntermediates__.mrt` or `__parityDebug__.mrt` is present, load MRT reference, run comparison, `expect(result.pass, ...).toBe(true)`.

---

## Part B: Optional rectangular grid (same-grid-as-.bin)

### Task B1: Map analysis bounds to viewer rectangular grid

**Files:**
- Modify: `viewer/src/lib/utils/coordinates.ts` or add `viewer/src/lib/compute/analysisGridFromBounds.ts` (function that takes metadata bounds + grid_size + coordinate_system and returns { points, normals } in viewer world)
- Reference: `viewer/src/lib/parity/analysisToWorld.ts` (xy_ground: (x,y,z) → (x, z, -y)); `viewer/src/lib/compute/grid-generator.ts` (`createRectangularGridFromBounds(bounds, gridSize, zHeight)` with `bounds.min = [minX, minZ]`, `bounds.max = [maxX, maxZ]`)

**Step 1: Implement analysisBoundsToRectangularGrid**

Create `viewer/src/lib/compute/analysisGridFromBounds.ts` (or add to an existing util). Export a function:

```ts
export function analysisBoundsToRectangularGrid(params: {
  bounds: { x_min: number; x_max: number; y_min: number; y_max: number; z?: number };
  gridSize: number;
  coordinateSystem: 'xy_ground' | 'xz_ground';
}): { points: THREE.Vector3[]; normals: THREE.Vector3[] }
```

- For `xy_ground`: analysis grid is (x, y) with z = bounds.z (fixed). Viewer world is (x, z, -y). So X_world = x, Z_world = -y, Y_world = bounds.z. So bounds in world: minX = bounds.x_min, maxX = bounds.x_max; minZ = -bounds.y_max, maxZ = -bounds.y_min; zHeight = bounds.z ?? 0.
- For `xz_ground`: bounds can be used as (x_min, x_max), (z_min, z_max), Y = bounds.z or similar; confirm from existing usage and match.
- Call `createRectangularGridFromBounds({ min: [minX, minZ], max: [maxX, maxZ] }, gridSize, zHeight)` and return its `points` and `normals` (both already in viewer frame).

**Step 2: Unit test (optional but recommended)**

In `viewer/src/lib/compute/analysisGridFromBounds.test.ts`, call `analysisBoundsToRectangularGrid` with a tiny bounds (e.g. 0,2,0,2,z=1) and gridSize 1, xy_ground; expect 9 points (3×3), first point (0, 1, 0) in world (x=0, y=z_height=1, z=0), last (2, 1, -2). Run: `cd viewer && npx vitest run src/lib/compute/analysisGridFromBounds.test.ts -v`.

---

### Task B2: Compute manager – optional rectangular grid input

**Files:**
- Modify: `viewer/src/lib/compute/compute-manager.ts` (accept optional `useRectangularGridFromBounds: true` and `analysisBounds` + `gridSize` + `coordinateSystem`; when set, call `analysisBoundsToRectangularGrid` and convert points/normals to `Float32Array` gridPoints; do not use mesh or worker grid)
- Modify: `viewer/src/lib/compute/liveUtciAnalysis.ts` (pass through option from analysis metadata when in "parity grid" or debug mode)

**Step 1: Extend initFromModelAndWeather params**

In `compute-manager.ts`, add optional:

```ts
useRectangularGridFromBounds?: boolean;
analysisBounds?: { x_min: number; x_max: number; y_min: number; y_max: number; z?: number };
gridSize?: number;
coordinateSystem?: 'xy_ground' | 'xz_ground';
```

When `useRectangularGridFromBounds === true` and `analysisBounds` and `gridSize` and `coordinateSystem` are present: call `analysisBoundsToRectangularGrid({ bounds: analysisBounds, gridSize, coordinateSystem })`, then fill `gridPoints` from the returned points (same format as mesh path: Float32Array of length numPoints*3), set `numPoints = points.length`. Do not require `mesh` or `workerGridPoints` in this branch. BVH is still required for exposure; so the caller must still provide a mesh (or serializedBvh) for the pipeline. So the flow is: rectangular grid points + mesh/BVH for raycasting. Update the method to allow this combination.

**Step 2: Wire from live analysis**

In `liveUtciAnalysis.ts`, when creating the compute run (e.g. for the debug page or when a flag like `useParityGrid` is true), read `baseMetadata.bounds`, `baseMetadata.coordinate_system`, and `gridResolution`; pass `useRectangularGridFromBounds: true`, `analysisBounds: baseMetadata.bounds`, `gridSize: gridResolution`, `coordinateSystem`. The pipeline needs both grid points (rectangular) and BVH (from mesh). Minimal approach: when the flag is set, still run the worker (or main-thread merge+BVH) to get `serializedBvh`; in addition, compute `gridPoints` via `analysisBoundsToRectangularGrid` and pass that into `initFromModelAndWeather` as the grid (e.g. add a branch: if `useRectangularGridFromBounds` and `analysisBounds`, set `gridPoints` from bounds and do not use the worker’s grid; still pass `serializedBvh` from the worker). So the worker returns BVH + optionally a mesh grid that we ignore; we override grid with the rectangular one from bounds.

---

### Task B3: Debug page – toggle or query for rectangular grid

**Files:**
- Modify: `viewer/src/routes/debug-webgpu-utci/+page.svelte` (add query param e.g. `?rectangularGrid=1` or use a store/checkbox; when set, pass `useRectangularGridFromBounds` and bounds from analysis metadata into the live analysis / compute manager)

**Step 1: Add query or UI flag**

In `debug-webgpu-utci/+page.svelte`, read `rectangularGrid=1` from the URL (or a checkbox). When building the params for `createLiveUtciAnalysisFromCompute` / ComputeManager, if the flag is set and `$analysisStore?.metadata?.bounds` exists, set `useRectangularGridFromBounds: true`, `analysisBounds: $analysisStore.metadata.bounds`, `gridSize: $analysisStore.metadata.grid_size ?? 2`, `coordinateSystem: $analysisStore.metadata.coordinate_system ?? 'xy_ground'`.

**Step 2: Pass into compute init**

Ensure the compute manager’s `initFromModelAndWeather` is called with these params. The merge/BVH worker or main-thread path still needs to provide the mesh for BVH; only the grid source changes. So when `useRectangularGridFromBounds` is true, the grid passed to the pipeline is the rectangular one; the mesh is still used to build BVH for solar/sky rays. Document in the plan or in-app tooltip: "Rectangular grid uses analysis bounds so points match .bin layout (including under buildings)."

---

### Task B4: Document and validate

**Files:**
- Modify: `docs/plans/2026-03-14-webgpu-bin-parity-validation-harness.md` (add section: Optional rectangular grid; how to run with `?rectangularGrid=1`; that grid count may match .bin for same bounds/grid_size)
- Modify: `docs/plans/2026-03-14-intermediate-stage-validation-design.md` (add line: MRT and weather sample are optional intermediates; rectangular grid option for same-grid comparison)

**Step 1: Update parity harness doc**

Add a subsection "Optional rectangular grid (same-grid-as-.bin)". Describe: the debug page supports `?rectangularGrid=1`; when set, the WebGPU grid is generated from analysis metadata bounds and grid_size in viewer coordinates, so point count and positions align with the .bin analysis. Use this to compare distributions or point-wise when needed, and to get distinct "inside building" values when hiding the building layer.

**Step 2: Update design doc**

In the intermediate-stage validation design doc, add one bullet: "MRT and weather sample can be exported/compared for diagnosis; rectangular grid option allows same-grid runs for grid vs formula separation."

---

## Execution order

- Part A (A1 → A2 → A3 → A4) can be done first; A2 depends on A1 only for having a reference format (A1 can be done in parallel with A2 if you define the JSON format up front).
- Part B (B1 → B2 → B3 → B4) can be done in parallel with A; B2 depends on B1, B3 on B2, B4 anytime.

---

## Verification

- After A: Run `npx playwright test tests/e2e/parity-intermediates.spec.ts` (with `*_mrt.json` and optionally `*_weather_sample.json`); run `npx playwright test tests/e2e/inspect-intermediates.spec.ts` and confirm MRT and weather sample in output.
- After B: Load debug page with `?rectangularGrid=1&analysis=Ben-Gurion/20250815_grid_2m_fullday`, confirm point count is close to 104445 (or exact for same bounds/grid_size), and that hiding the building layer shows variation inside building footprints.

---

Plan complete and saved to `docs/plans/2026-03-14-utci-parity-deep-dive-implementation-plan.md`.

**Two execution options:**

1. **Subagent-Driven (this session)** – I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Parallel Session (separate)** – Open a new session with executing-plans; batch execution with checkpoints.

Which approach?
