# Offline Parity Workflow Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the three-command offline parity workflow (Python collect, WebGPU collect, Compare) with sky normalization, UTCI min/max/mean comparison, and rectangular-grid fix so parity tests are fast and deterministic.

**Architecture:** (1) Viewer exposes parity data with sky already normalized 0–1. (2) WebGPU collect is a Playwright spec that loads the debug page, waits for results, reads from the page, and writes one JSON file per stage to disk. (3) Compare is a Node script that loads ref and WebGPU files and asserts solar/sky/MRT stats and UTCI min/max/mean. (4) Rectangular grid bug is diagnosed and fixed so the live UTCI layer shows when `?rectangularGrid=1`. No git commits or worktrees in this plan; user manages git.

**Tech Stack:** SvelteKit viewer, Playwright, Node `fs`, TypeScript, existing parity modules (`compareIntermediates`, `loadReferenceIntermediatesFromFs`, etc.). Python export script (existing).

**Design reference:** `docs/plans/2026-03-14-offline-parity-and-fixes-design.md`

---

## Part A: Sky normalization (WebGPU → 0–1)

### Task A1: Add sky normalization constant and use it when exposing parity intermediates

**Files:**
- Create: `viewer/src/lib/parity/skyScale.ts`
- Modify: `viewer/src/routes/debug-webgpu-utci/+page.svelte` (use constant when building `skyExposure` for `__parityIntermediates__`)

**Step 1: Add the constant and helper**

Create `viewer/src/lib/parity/skyScale.ts`:

```ts
/**
 * Total Tregenza dome weight used in MRT shader (mrt_utci.wgsl).
 * WebGPU exposure shader writes raw weight sum; divide by this to get 0–1 sky view factor for parity.
 */
export const TOTAL_TREGENZA_WEIGHT = 145.2488;

export function normalizeSkyExposureToViewFactor(rawSky: number[] | Float32Array): number[] {
	const out: number[] = [];
	for (let i = 0; i < rawSky.length; i++) {
		out.push(Math.max(0, Math.min(1, rawSky[i] / TOTAL_TREGENZA_WEIGHT)));
	}
	return out;
}
```

**Step 2: Write a quick unit test**

Create `viewer/tests/parity/skyScale.test.ts`:

```ts
import { describe, it, expect } from 'vitest';
import { TOTAL_TREGENZA_WEIGHT, normalizeSkyExposureToViewFactor } from '$lib/parity/skyScale';

describe('skyScale', () => {
	it('normalizes raw sky to 0–1', () => {
		const raw = [0, 72.6244, 145.2488, 200];
		const out = normalizeSkyExposureToViewFactor(raw);
		expect(out[0]).toBe(0);
		expect(out[1]).toBeCloseTo(0.5);
		expect(out[2]).toBeCloseTo(1);
		expect(out[3]).toBeCloseTo(1);
	});
});
```

Run: `cd viewer && npx vitest run tests/parity/skyScale.test.ts -v`  
Expected: PASS (after implementing the module).

**Step 3: Use normalization in the debug page**

In `viewer/src/routes/debug-webgpu-utci/+page.svelte`, in the block where `win.__parityIntermediates__` is set (around line 383–391):

- Import: `import { normalizeSkyExposureToViewFactor } from '$lib/parity/skyScale';`
- Replace the line that sets `skyExposure: Array.from(results[1])` with:
  `skyExposure: normalizeSkyExposureToViewFactor(results[1])`

So the value exposed on `window` is already 0–1 for sky.

**Step 4: Run tests**

Run: `cd viewer && npx vitest run tests/parity/skyScale.test.ts -v`  
Expected: PASS.

**Step 5: Checkpoint**

Sky normalization is in place for parity exposure; unit test passes. No staging or commit.

---

## Part B: WebGPU collect (Playwright → write files)

### Task B1: Playwright spec that waits for parity data and writes WebGPU JSON files

**Files:**
- Create: `viewer/tests/e2e/collect-webgpu-parity.spec.ts`
- Test: run once manually to confirm files are written

**Step 1: Create the collect spec**

Create `viewer/tests/e2e/collect-webgpu-parity.spec.ts`:

- Use `baseURL` from Playwright config (e.g. `http://localhost:5173`). Analysis base path comes from env `PARITY_BASE_PATH` (e.g. `data/analyses/Ben-Gurion/20250815_grid_2m_fullday`). Resolve full path from repo root: if `process.cwd()` ends with `viewer`, repo root is `..`, else `.`; then `path.join(repoRoot, basePathFromEnv)`.
- Single test: `goto` `/debug-webgpu-utci?analysis=Ben-Gurion/20250815_grid_2m_fullday` (or build URL from env so it’s configurable). Set `test.setTimeout(120_000)` (120 s).
- `waitForFunction`: wait until `(window as any).__parityResults__ != null && (window as any).__parityIntermediates__ != null`, with timeout 100_000 ms. If `(window as any).__parityIntermediatesError__` is set, fail the test with that message.
- `page.evaluate`: return `{ parityResults: (window as any).__parityResults__, parityIntermediates: (window as any).__parityIntermediates__ }`.
- In Node: require `fs` and `path`. Compute `basePath = resolve(repoRoot, process.env.PARITY_BASE_PATH || 'data/analyses/Ben-Gurion/20250815_grid_2m_fullday')`. Normalize sky: use `TOTAL_TREGENZA_WEIGHT` from `$lib/parity/skyScale` (or inline 145.2488) and divide each element of `parityIntermediates.skyExposure` by it, clamp 0–1 (if the page already sends normalized sky, this is idempotent).
- Write four files:
  - `{basePath}_webgpu_solar.json`: `{ numPositions: parityIntermediates.numPoints, numHours: parityIntermediates.numHours, solarExposure: parityIntermediates.solarExposure }`.
  - `{basePath}_webgpu_sky.json`: `{ numPositions: parityIntermediates.numPoints, skyExposure: normalizedSky }`.
  - `{basePath}_webgpu_mrt.json`: if `parityIntermediates.mrt` exists, `{ numPositions, numHours, mrt: parityIntermediates.mrt }`.
  - `{basePath}_webgpu_utci.json`: from `parityResults` compute `utci_range: { min, max, mean }` by iterating `parityResults.utciByHour` (array of number[]), then `{ numPoints: parityResults.numPoints, numHours: parityResults.numHours, utciByHour: parityResults.utciByHour, utci_range: { min, max, mean } }`.
- Use `writeFileSync` with `JSON.stringify(..., null, 0)` or compact. Do not fail the test if MRT is missing; still write solar, sky, utci.

**Step 2: Add npm script**

In `viewer/package.json`, add:

`"parity:collect-webgpu": "playwright test tests/e2e/collect-webgpu-parity.spec.ts"`

Optionally document that `PARITY_BASE_PATH` can be set (relative to repo root) to change output directory.

**Step 3: Run collect once**

From repo root: `cd viewer && set PARITY_BASE_PATH=data/analyses/Ben-Gurion/20250815_grid_2m_fullday&& npx playwright test tests/e2e/collect-webgpu-parity.spec.ts` (Windows). Or from viewer: `PARITY_BASE_PATH=../data/analyses/Ben-Gurion/20250815_grid_2m_fullday npx playwright test tests/e2e/collect-webgpu-parity.spec.ts` (Unix). Ensure dev server can start (playwright.config has webServer). After run, verify that `data/analyses/Ben-Gurion/20250815_grid_2m_fullday_webgpu_solar.json` (and _sky, _mrt, _utci) exist and have expected shape.

**Step 4: Checkpoint**

WebGPU collect spec runs and writes four JSON files. No staging or commit.

---

## Part C: Compare script (Node, offline)

### Task C2: Load WebGPU JSON files from disk

**Files:**
- Create: `viewer/src/lib/parity/loadWebgpuCollectedFromFs.ts`
- Create: `viewer/tests/parity/loadWebgpuCollectedFromFs.test.ts`

**Step 1: Write the failing test**

Create `viewer/tests/parity/loadWebgpuCollectedFromFs.test.ts`:

- Test: `loadWebgpuCollectedFromFs(basePath)` returns an object with optional `solar`, `sky`, `mrt`, `utci` (each matching the shapes from the design). If a file is missing, that key is undefined.
- Use a fixture path or skip when file does not exist. Example: for a fixture `tests/parity/fixtures/webgpu_solar.json` with `{ "numPositions": 2, "numHours": 24, "solarExposure": [0,1,...] }`, expect `loadWebgpuCollectedFromFs(fixtureBase).solar` to be defined and have `numPositions === 2`.

**Step 2: Run test to verify it fails**

Run: `cd viewer && npx vitest run tests/parity/loadWebgpuCollectedFromFs.test.ts`  
Expected: FAIL (module or function not found).

**Step 3: Implement loader**

Create `viewer/src/lib/parity/loadWebgpuCollectedFromFs.ts`:

- `loadWebgpuCollectedFromFs(basePath: string): Promise<{ solar?: { numPositions, numHours, solarExposure }; sky?: { numPositions, skyExposure }; mrt?: { numPositions, numHours, mrt }; utci?: { numPoints, numHours, utciByHour, utci_range } }>`.
- Use dynamic `import('node:fs')` and `readFileSync` for `{basePath}_webgpu_solar.json`, `_webgpu_sky.json`, `_webgpu_mrt.json`, `_webgpu_utci.json`. Parse JSON; validate minimal shape; return only present files. Missing file → that key omitted (no throw).

**Step 4: Run test to verify it passes**

Run: `cd viewer && npx vitest run tests/parity/loadWebgpuCollectedFromFs.test.ts`  
Expected: PASS.

**Step 5: Checkpoint**

Loader for WebGPU collected files exists and is tested. No staging or commit.

---

### Task C3: UTCI range comparison helper

**Files:**
- Create or modify: `viewer/src/lib/parity/compareUtciRange.ts`
- Create: `viewer/tests/parity/compareUtciRange.test.ts`

**Step 1: Write the failing test**

Create `viewer/tests/parity/compareUtciRange.test.ts`:

- `compareUtciRange({ ref: { min: 22, max: 39, mean: 28 }, webgpu: { min: 22.5, max: 38.5, mean: 28.2 }, toleranceMin: 1, toleranceMax: 1, toleranceMean: 0.5 })` returns `{ pass: true }`.
- `compareUtciRange({ ref: { min: 22, max: 39, mean: 28 }, webgpu: { min: 20, max: 39, mean: 28 }, ... })` returns `{ pass: false }` (min diff 2 > 1).

**Step 2: Run test to verify it fails**

Run: `cd viewer && npx vitest run tests/parity/compareUtciRange.test.ts`  
Expected: FAIL.

**Step 3: Implement**

Create `viewer/src/lib/parity/compareUtciRange.ts`:

- `compareUtciRange(params: { ref: { min, max, mean }; webgpu: { min, max, mean }; toleranceMin?, toleranceMax?, toleranceMean? }): { pass: boolean; minDiff?: number; maxDiff?: number; meanDiff?: number }`. Default tolerances e.g. 2 °C for min/max, 1 °C for mean. Pass iff all absolute differences ≤ tolerances.

**Step 4: Run test to verify it passes**

Run: `cd viewer && npx vitest run tests/parity/compareUtciRange.test.ts`  
Expected: PASS.

**Step 5: Checkpoint**

UTCI range comparison helper and tests in place. No staging or commit.

---

### Task C4: Compare script (CLI)

**Files:**
- Create: `viewer/scripts/compare-parity.ts`
- Add script to `viewer/package.json` to run it (e.g. `npx tsx scripts/compare-parity.ts` or `node` with compiled output)

**Step 1: Implement compare script**

Create `viewer/scripts/compare-parity.ts`:

- Parse CLI arg or env for base path (e.g. `--base-path data/analyses/Ben-Gurion/20250815_grid_2m_fullday`). Resolve to absolute from repo root (when run from viewer, cwd may be viewer, so resolve with `path.join(process.cwd(), '..', basePath)` if cwd ends with `viewer`).
- Load Python refs: `loadReferenceIntermediatesFromFs(basePath, 'solar')` etc. (try/catch per stage; missing ref → skip or fail that stage).
- Load WebGPU: `loadWebgpuCollectedFromFs(basePath)`.
- For each stage (solar, sky, mrt): if both ref and webgpu present, call `compareIntermediatesStats` with tolerances (solar/sky: mean 0.02, max 0.05; MRT: mean 1, max 2). Record pass/fail.
- For UTCI: load `{basePath}.json` (metadata) and get `metadata.utci_range`; load webgpu utci from `loadWebgpuCollectedFromFs`; call `compareUtciRange` with tolerances (e.g. 2 °C min/max, 1 °C mean). Record pass/fail.
- Print results per stage; set `process.exit(failCount > 0 ? 1 : 0)`.

**Step 2: Add npm script**

In `viewer/package.json`, add:

`"parity:compare": "tsx scripts/compare-parity.ts"` (or `node --loader ts-node/esm scripts/compare-parity.ts` if tsx not available; add `tsx` as devDependency if needed).

Script should accept `--base-path <path>`; default base path can be `data/analyses/Ben-Gurion/20250815_grid_2m_fullday` when run from repo root.

**Step 3: Run compare after collect**

Ensure WebGPU collect has been run so that `_webgpu_*.json` files exist. Run: `cd viewer && npx tsx scripts/compare-parity.ts --base-path ../data/analyses/Ben-Gurion/20250815_grid_2m_fullday` (or equivalent). Expect exit code 0 if all stages pass, 1 if any fail. Verify output lists each stage and pass/fail.

**Step 4: Checkpoint**

Compare script runs offline and reports pass/fail per stage. No staging or commit.

---

## Part D: Rectangular grid — diagnose and fix (UTCI layer never shows)

### Task D5: Add diagnostics and fix rectangular grid path

**Files:**
- Modify: `viewer/src/routes/debug-webgpu-utci/+page.svelte` (log when using rectangular grid; ensure liveError is visible)
- Modify: `viewer/src/lib/compute/liveUtciAnalysis.ts` or `viewer/src/lib/compute/compute-manager.ts` (if diagnosis shows init or pipeline fails for rectangular)

**Step 1: Add logging when rectangular grid is used**

In `viewer/src/routes/debug-webgpu-utci/+page.svelte`, immediately after `const useRectangularGrid = ...` (around line 302), add:

```ts
if (useRectangularGrid) {
  console.log('[parity] Using rectangular grid from bounds; grid will match .bin point count.');
}
```

Before calling `createLiveUtciAnalysisFromCompute`, if `useRectangularGrid` is true, log the analysis id and that we're passing `useRectangularGridFromBounds: true`. This confirms the path is taken.

**Step 2: Confirm liveError is shown in UI**

Verify that when compute fails, the sidebar shows "Failed to compute live UTCI: {liveError}". No code change needed if already present; if not, ensure the `{:else if liveError}` block is visible and not hidden by overlay. If the full-screen overlay hides the sidebar, ensure overlay is dismissed when `liveError` is set (design: `showFullLoadOverlay = modelLoading || (model != null && liveAnalysis === null && liveError === null)` — so when liveError is set, overlay should hide). Confirm in code.

**Step 3: Diagnose rectangular path in compute-manager**

In `viewer/src/lib/compute/compute-manager.ts`, in the branch `if (useRectangularGridFromBounds && analysisBounds && (serializedBvh || mesh))`, after building `gridPoints` from `analysisBoundsToRectangularGrid`, assert or log `numPoints > 0`. If `numPoints === 0`, throw a clear error so we get `liveError` instead of silent fail. Check that `epwContent` and other params are passed correctly in that branch (they are in the current code; no change if already correct).

**Step 4: Check for thrown errors in createLiveUtciAnalysisFromCompute**

In `viewer/src/lib/compute/liveUtciAnalysis.ts`, the rectangular branch (lines 137–148) passes `serializedBvh` or `mesh`, `useRectangularGridFromBounds: true`, `analysisBounds`, `coordinateSystem`, `epwContent`, `gridResolution`, `zHeight`, `signal`. Ensure when `workerResult` is present we pass `serializedBvh: workerResult.serializedBvh` and do not pass `gridPoints` (so the manager uses bounds). Re-read the flow; if any required param is missing in the rectangular branch, add it.

**Step 5: Verify UTCIPointCloud receives liveAnalysis**

The live layer is rendered with `{#if liveAnalysis}` and `<UTCIPointCloud analysis={liveAnalysis} ... />`. Ensure that when using rectangular grid, `liveAnalysis` is set the same way (same assignment `liveAnalysis = result` after `createLiveUtciAnalysisFromCompute`). No difference in assignment for rectangular vs mesh. If the result has a different shape (e.g. missing `metadata.utci_range`), `createUtciSurfaceMesh` or `resolveUtciRange` might throw or return invalid range; the live analysis built in `liveUtciAnalysis.ts` already sets `utci_range: { min: globalMin, max: globalMax }` and `hour_statistics`. So rectangular result should have the same shape. If diagnosis shows an exception inside `UTCIPointCloud` or `createUtciSurfaceMesh` for rectangular only (e.g. due to layout or NaN positions), fix that (e.g. guard in `buildUtciGridLayout` or ensure positions are finite).

**Step 6: Run manually with rectangular grid**

Open `http://localhost:5173/debug-webgpu-utci?analysis=Ben-Gurion/20250815_grid_2m_fullday&rectangularGrid=1`. Open browser console. Confirm log "[parity] Using rectangular grid from bounds...". Wait for compute to finish (or for an error). If error appears in sidebar or console, fix the root cause (missing param, throw in manager, or invalid layout). If no error but no layer, check that `liveAnalysis` is set (e.g. temporary `console.log(liveAnalysis?.data?.numPositions)` after the try block). Iterate until the rectangular grid run sets `liveAnalysis` and the UTCI layer is visible.

**Step 7: Checkpoint**

Rectangular grid path is diagnosed; fixes applied so that UTCI layer shows when `?rectangularGrid=1`. No staging or commit.

---

## Part E: Documentation and scripts summary

### Task E6: Document the three commands

**Files:**
- Modify: `docs/plans/2026-03-14-offline-parity-and-fixes-design.md` (add "How to run" section) or add a short README in `viewer/scripts/` or update `docs/plans/2026-03-14-webgpu-bin-parity-validation-harness.md`

**Step 1: Add "How to run" section**

In the design doc or harness doc, add:

- **Python collect:** From repo root, `python scripts/export_ben_gurion_intermediates.py --base-path data/analyses/Ben-Gurion/20250815_grid_2m_fullday --model data/3d_models/Ben-Gurion/original_with_layers.glb --stage solar --stage sky --stage mrt` (and optionally `--stage weather`). Produces `_solar.json`, `_sky.json`, `_mrt.json`, etc.
- **WebGPU collect:** From repo root, `cd viewer && npx playwright test tests/e2e/collect-webgpu-parity.spec.ts`. Optionally set `PARITY_BASE_PATH=data/analyses/Ben-Gurion/20250815_grid_2m_fullday` (relative to repo root). Produces `_webgpu_solar.json`, `_webgpu_sky.json`, `_webgpu_mrt.json`, `_webgpu_utci.json`.
- **Compare:** From repo root, `cd viewer && npx tsx scripts/compare-parity.ts --base-path ../data/analyses/Ben-Gurion/20250815_grid_2m_fullday` (or from viewer with `--base-path` relative to repo root). Exits 0 if all stages pass, 1 otherwise.

**Step 2: Checkpoint**

Docs updated. No staging or commit.

---

## Execution order

- **Part A** (sky normalization) first — required so WebGPU collect and compare see 0–1 sky.
- **Part B** (WebGPU collect spec) after A — depends on normalized sky in the page.
- **Part C** (load WebGPU files, UTCI range helper, compare script) can be done in parallel with B after A; C2 and C3 before C4.
- **Part D** (rectangular grid) can be done in parallel with B/C; do after A so parity exposure is correct.
- **Part E** (docs) last.

---

## Verification (after all tasks)

1. Run Python collect (if not already done) to ensure ref files exist.
2. Run WebGPU collect: `cd viewer && npx playwright test tests/e2e/collect-webgpu-parity.spec.ts`. Confirm four `_webgpu_*.json` files exist.
3. Run Compare: `cd viewer && npx tsx scripts/compare-parity.ts --base-path ../data/analyses/Ben-Gurion/20250815_grid_2m_fullday`. Confirm exit 0 or expected failures with clear per-stage output.
4. Open debug page with `?rectangularGrid=1` and confirm live UTCI layer appears after compute.

---

Plan complete and saved to `docs/plans/2026-03-14-offline-parity-implementation-plan.md`.

**Two execution options:**

1. **Subagent-Driven (this session)** — Dispatch a fresh subagent per task (or execute task-by-task in this session), review between tasks, fast iteration.
2. **Parallel Session (separate)** — Open a new session with executing-plans; batch execution with checkpoints.

Which approach?
