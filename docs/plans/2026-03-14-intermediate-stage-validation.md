# Intermediate-Stage Validation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add validation for WebGPU pipeline intermediate stages (solar exposure, sky exposure) against one-time Python-exported reference files, so we can isolate discrepancies. Ben-Gurion base case only; no point-to-point UTCI assertion.

**Architecture:** (1) One reference file per stage (`*_solar.json`, `*_sky.json`) produced once by a Python script. (2) Node loader and comparison helper in the parity module. (3) WebGPU pipeline exposes readback for solar and sky; debug viewer in parity mode sets `window.__parityIntermediates__`. (4) Playwright or Node script loads reference, gets WebGPU results from the page, compares per stage and fails if beyond tolerance.

**Tech Stack:** TypeScript (viewer parity module, Playwright), Python (export script), Node fs for reference loading. No new runtimes.

**Design reference:** `docs/plans/2026-03-14-intermediate-stage-validation-design.md`

---

## Task 1: Reference loader for intermediate stages (Node)

**Files:**
- Create: `viewer/src/lib/parity/loadReferenceIntermediatesFromFs.ts`
- Create: `viewer/tests/parity/loadReferenceIntermediatesFromFs.test.ts`

**Step 1: Write the failing test**

Use base path without suffix; loader appends `_solar.json` or `_sky.json`. Resolve path from repo root (same as `loadReferenceFromFs.test.ts`: `process.cwd()` with optional `..` when cwd is viewer).

```typescript
// viewer/tests/parity/loadReferenceIntermediatesFromFs.test.ts
import { describe, it, expect } from 'vitest';
import { resolve } from 'node:path';
import { loadReferenceIntermediatesFromFs } from '$lib/parity/loadReferenceIntermediatesFromFs';

const REPO_ROOT = resolve(process.cwd(), process.cwd().endsWith('viewer') ? '..' : '.');

describe('loadReferenceIntermediatesFromFs', () => {
	it('loads solar reference from basePath_solar.json', async () => {
		// Create a minimal fixture or skip if file does not exist yet
		const basePath = resolve(REPO_ROOT, 'data/analyses/Ben-Gurion/20250815_grid_2m_fullday');
		await expect(
			loadReferenceIntermediatesFromFs(basePath, 'solar')
		).rejects.toThrow(); // expect file missing until Task 5 creates it
	});

	it('when solar file exists, returns numPositions, numHours, solarExposure Float32Array', async () => {
		// After fixture exists: expect(ref.numPositions).toBeGreaterThan(0); ref.solarExposure.length === ref.numPositions * ref.numHours
	});
});
```

**Step 2: Run test to verify it fails**

Run: `cd viewer && npx vitest run tests/parity/loadReferenceIntermediatesFromFs.test.ts`  
Expected: FAIL (module or function not found, or test throws for missing file).

**Step 3: Implement loadReferenceIntermediatesFromFs**

- Signature: `loadReferenceIntermediatesFromFs(basePath: string, stage: 'solar' | 'sky'): Promise<...>`.
- Use dynamic `import('node:fs')` so the module can load in browser (and throw a clear error when called in browser).
- **Solar:** Read `${basePath}_solar.json`, parse JSON, expect `numPositions`, `numHours`, `solarExposure: number[]`. Convert `solarExposure` to Float32Array and return `{ numPositions, numHours, solarExposure: Float32Array }`.
- **Sky:** Read `${basePath}_sky.json`, expect `numPositions`, `skyExposure: number[]`. Return `{ numPositions, skyExposure: Float32Array }`.
- Throw if file not found or shape invalid.

**Step 4: Add a small fixture and run test to verify it passes**

Create `viewer/tests/parity/fixtures/ben_gurion_solar.json` with `{ "numPositions": 2, "numHours": 24, "solarExposure": [0,0,...,0] }` (48 numbers). In test, use fixture path for the "when solar file exists" case. Run: `cd viewer && npx vitest run tests/parity/loadReferenceIntermediatesFromFs.test.ts`. Expected: PASS.

**Step 5: Commit**

```bash
git add viewer/src/lib/parity/loadReferenceIntermediatesFromFs.ts viewer/tests/parity/loadReferenceIntermediatesFromFs.test.ts viewer/tests/parity/fixtures/ben_gurion_solar.json
git commit -m "feat(parity): add loadReferenceIntermediatesFromFs for solar/sky reference files"
```

---

## Task 2: Comparison helper for intermediates

**Files:**
- Create: `viewer/src/lib/parity/compareIntermediates.ts`
- Create: `viewer/tests/parity/compareIntermediates.test.ts`

**Step 1: Write the failing test**

```typescript
// viewer/tests/parity/compareIntermediates.test.ts
import { describe, it, expect } from 'vitest';
import { compareIntermediates } from '$lib/parity/compareIntermediates';

describe('compareIntermediates', () => {
	it('returns pass when arrays match within tolerance', () => {
		const ref = new Float32Array([0.0, 0.5, 1.0]);
		const webgpu = new Float32Array([0.0, 0.50001, 1.0]);
		const r = compareIntermediates({ ref, webgpu, tolerance: 1e-4 });
		expect(r.pass).toBe(true);
		expect(r.maxError).toBeLessThanOrEqual(1e-4);
	});

	it('returns fail when any value exceeds tolerance', () => {
		const ref = new Float32Array([0.5]);
		const webgpu = new Float32Array([0.6]);
		const r = compareIntermediates({ ref, webgpu, tolerance: 0.05 });
		expect(r.pass).toBe(false);
		expect(r.maxError).toBeCloseTo(0.1);
	});

	it('throws when lengths differ', () => {
		expect(() =>
			compareIntermediates({ ref: new Float32Array(3), webgpu: new Float32Array(5), tolerance: 0.01 })
		).toThrow(/length/);
	});
});
```

**Step 2: Run test to verify it fails**

Run: `cd viewer && npx vitest run tests/parity/compareIntermediates.test.ts`  
Expected: FAIL.

**Step 3: Implement compareIntermediates**

- Signature: `compareIntermediates(params: { ref: Float32Array; webgpu: Float32Array; tolerance?: number }): { pass: boolean; rmse: number; maxError: number; numPoints: number }`.
- Default tolerance 1e-5. If `ref.length !== webgpu.length`, throw with message containing "length". Compute RMSE and max absolute error; pass iff maxError <= tolerance. Pure function, Node/browser safe.

**Step 4: Run test to verify it passes**

Run: `cd viewer && npx vitest run tests/parity/compareIntermediates.test.ts`  
Expected: PASS.

**Step 5: Commit**

```bash
git add viewer/src/lib/parity/compareIntermediates.ts viewer/tests/parity/compareIntermediates.test.ts
git commit -m "feat(parity): add compareIntermediates for solar/sky array comparison"
```

---

## Task 3: WebGPU pipeline readback for solar and sky

**Files:**
- Modify: `viewer/src/lib/compute/gpu-pipeline.ts` (add optional methods to interface)
- Modify: `viewer/src/lib/compute/webgpuUtciPipeline.ts` (implement readback)
- Test: `viewer/tests/compute/exposure-pipeline.test.ts` or new test file if needed

**Step 1: Extend UTCIComputePipeline interface**

In `viewer/src/lib/compute/gpu-pipeline.ts`, add optional methods after `readUtcisSlice`:

```typescript
/**
 * Read full solar exposure buffer (point-major: [p0_h0..p0_h23, p1_h0..], one month only).
 * Optional; only WebGPU implementation provides this (for intermediate parity).
 */
readSolarExposureFull?(params: { numPoints: number; numHours: number; numMonths: number }): Promise<Float32Array>;

/**
 * Read full sky exposure buffer (one value per point).
 * Optional; only WebGPU implementation provides this (for intermediate parity).
 */
readSkyExposure?(params: { numPoints: number }): Promise<Float32Array>;
```

**Step 2: Implement readback in WebgpuUtciComputePipeline**

In `viewer/src/lib/compute/webgpuUtciPipeline.ts`:
- **readSolarExposureFull:** Create a staging buffer with MAP_READ | COPY_DST, size = numPoints * numHours * numMonths * 4. Encode copyBufferToBuffer from solarExposureBuffer to staging, submit, mapAsync(READ), copy Float32Array from getMappedRange(), unmap, return. Require lastConfig and solarExposureBuffer.
- **readSkyExposure:** Same pattern for skyExposureBuffer (size numPoints * 4). Require lastConfig and skyExposureBuffer.
- Both must run after runAll() has been called (lastConfig set).

**Step 3: Add unit test for readback (optional, or skip in Node)**

If Vitest runs in Node, WebGPU is unavailable; add a test that only runs when pipeline is a fake that implements readSolarExposureFull/readSkyExposure and returns fixed arrays; or document that readback is exercised in Playwright. For minimal scope, skip automated unit test for WebGPU readback and rely on Playwright.

**Step 4: Expose readback via ComputeManager (optional)**

If callers use ComputeManager only, add `getSolarExposureFull()` and `getSkyExposure()` that delegate to `this.pipeline.readSolarExposureFull?.()` and `this.pipeline.readSkyExposure?.()`. Debug viewer can instead call pipeline directly if it keeps a reference.

**Step 5: Commit**

```bash
git add viewer/src/lib/compute/gpu-pipeline.ts viewer/src/lib/compute/webgpuUtciPipeline.ts
git commit -m "feat(compute): add readSolarExposureFull and readSkyExposure to WebGPU pipeline"
```

---

## Task 4: Debug viewer exposes __parityIntermediates__ in parity mode

**Files:**
- Modify: `viewer/src/routes/debug-webgpu-utci/+page.svelte`

**Step 1: After setting __parityResults__, read intermediates when pipeline supports it**

In the block where `win.__parityResults__` is set (after `createLiveUtciAnalysisFromCompute` resolves), add:
- If `lastPipeline` is non-null and `lastPipeline.readSolarExposureFull` and `lastPipeline.readSkyExposure` exist, await `lastPipeline.readSolarExposureFull({ numPoints: result.data.numPositions, numHours: result.data.utciByHour.length, numMonths: 1 })` and `lastPipeline.readSkyExposure({ numPoints: result.data.numPositions })`.
- Set `(window as any).__parityIntermediates__ = { solarExposure: Array.from(solarArray), skyExposure: Array.from(skyArray), numPoints: result.data.numPositions, numHours: result.data.utciByHour.length }` (JSON-serializable for Playwright).

**Step 2: Type declaration for __parityIntermediates__**

Add to the same `win` cast or a small `.d.ts`: `__parityIntermediates__?: { solarExposure: number[]; skyExposure: number[]; numPoints: number; numHours: number }`.

**Step 3: Manual check**

Run viewer, open `/debug-webgpu-utci?parity=1&analysis=Ben-Gurion/20250815_grid_2m_fullday`, wait for compute to finish, in console run `window.__parityIntermediates__` and confirm structure and lengths (solarExposure.length === numPoints * numHours, skyExposure.length === numPoints).

**Step 4: Commit**

```bash
git add viewer/src/routes/debug-webgpu-utci/+page.svelte
git commit -m "feat(debug): expose __parityIntermediates__ in parity mode for solar/sky"
```

---

## Task 5: Python export script for Ben-Gurion solar and sky

**Files:**
- Create: `scripts/export_ben_gurion_intermediates.py`
- Reference: `src/fast_utci/mrt/exposure.py` (compute_solar_exposure, compute_sky_exposure), analysis loading (grid positions from .bin or from same pipeline that generated the .bin)

**Step 1: Implement script that loads analysis + model and runs exposure**

- Parse args: `--base-path` (e.g. `data/analyses/Ben-Gurion/20250815_grid_2m_fullday`), `--stage solar` and/or `--stage sky`, `--model` (e.g. `data/3d_models/Ben-Gurion/original_with_layers.glb`).
- Load positions from the .bin (same format as existing loader: num_positions, positions float32 xyz). Load metadata from .json (num_positions, hours, coordinate_system).
- Load mesh from GLB; build mesh context for ray tests. Get sun data (EPW from metadata or fixed path) for the same date/location as the analysis.
- For each position (or batch): call Python `compute_solar_exposure` and `compute_sky_exposure` (from `fast_utci.mrt.exposure`). Solar returns shape (n_hours,); sky returns scalar per position.
- **Solar output:** Build point-major flat array: for each point, append 24 values (one per hour). Write JSON: `{ "numPositions": N, "numHours": 24, "solarExposure": [ ... ] }` to `{base_path}_solar.json`.
- **Sky output:** Build array of length numPositions. Write JSON: `{ "numPositions": N, "skyExposure": [ ... ] }` to `{base_path}_sky.json`.

**Step 2: Document coordinate system and sun vectors**

Ensure script uses the same coordinate system (xy_ground) and same sun vectors (e.g. from analysis .json sun_positions or recompute from EPW) as the WebGPU path so reference is comparable. Document in script or in `docs/plans/2026-03-14-intermediate-stage-validation-design.md`.

**Step 3: Run script once and commit reference files (optional)**

Run from repo root: `python scripts/export_ben_gurion_intermediates.py --base-path data/analyses/Ben-Gurion/20250815_grid_2m_fullday --stage solar --stage sky --model data/3d_models/Ben-Gurion/original_with_layers.glb`. Commit the generated `*_solar.json` and `*_sky.json` if desired, or add to .gitignore and document "run script to generate."

**Step 4: Commit**

```bash
git add scripts/export_ben_gurion_intermediates.py
git commit -m "feat(scripts): add Python script to export solar/sky intermediates for Ben-Gurion"
```

---

## Task 6: Playwright test for solar and sky parity (one stage at a time)

**Files:**
- Create or modify: `viewer/tests/e2e/parity-intermediates.spec.ts` (or add to existing parity spec)
- Reference: `viewer/src/lib/parity/loadReferenceIntermediatesFromFs.ts`, `compareIntermediates.ts`

**Step 1: Solar parity test**

- Load reference: `loadReferenceIntermediatesFromFs(basePath, 'solar')` where basePath = `data/analyses/Ben-Gurion/20250815_grid_2m_fullday`. Skip test if file does not exist (e.g. `test.skip(!refFileExists)` or try/catch and skip).
- Goto debug page with `?parity=1&analysis=Ben-Gurion/20250815_grid_2m_fullday`, wait for `window.__parityIntermediates__` (poll with timeout, e.g. 90s).
- Get `const webgpu = await page.evaluate(() => window.__parityIntermediates__)`. Convert webgpu.solarExposure to Float32Array; ref already Float32Array from loader.
- Assert lengths match: ref.solarExposure.length === webgpu.solarExposure.length.
- Call `compareIntermediates({ ref: ref.solarExposure, webgpu: new Float32Array(webgpu.solarExposure), tolerance: 1e-4 })`. Assert `result.pass` (or log and fail with message including maxError).

**Step 2: Sky parity test**

- Same flow for sky: load `loadReferenceIntermediatesFromFs(basePath, 'sky')`, get `__parityIntermediates__.skyExposure`, compare with `compareIntermediates`, assert pass.

**Step 3: Run tests**

Run: `cd viewer && npx playwright test tests/e2e/parity-intermediates.spec.ts` (or the chosen path). If reference files are missing, tests skip. If reference exists and WebGPU runs, tests assert. Expected: PASS when reference and WebGPU match within tolerance; FAIL with clear message when they don’t.

**Step 4: Commit**

```bash
git add viewer/tests/e2e/parity-intermediates.spec.ts
git commit -m "test(e2e): add solar and sky intermediate parity tests for Ben-Gurion base"
```

---

## Task 7: Update harness plan and design docs

**Files:**
- Modify: `docs/plans/2026-03-14-webgpu-bin-parity-validation-harness.md`
- Optionally: `docs/plans/2026-03-14-intermediate-stage-validation-design.md` (add "How to run" section if not already there)

**Step 1: Add section "Intermediate-stage validation"**

In the harness plan, after "Validation strategy" (or in a new section), add:
- Intermediate validation: solar and sky exposure are validated against reference files produced once by Python (see `docs/plans/2026-03-14-intermediate-stage-validation-design.md`).
- Reference files: one per stage, `*_solar.json` and `*_sky.json`, for Ben-Gurion base only.
- How to generate reference: run `python scripts/export_ben_gurion_intermediates.py ...` (document args).
- How to run checks: Playwright test `parity-intermediates.spec.ts` (or the script name); smoke e2e remains "compute completes and __parityResults__ set"; point-to-point UTCI still out of scope.

**Step 2: Commit**

```bash
git add docs/plans/2026-03-14-webgpu-bin-parity-validation-harness.md docs/plans/2026-03-14-intermediate-stage-validation-design.md
git commit -m "docs: document intermediate-stage validation in parity harness plan"
```

---

## Execution handoff

Plan complete and saved to `docs/plans/2026-03-14-intermediate-stage-validation.md`.

Two execution options:

1. **Subagent-driven (this session)** – Dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Parallel session (separate)** – Open a new session with executing-plans and run task-by-task with checkpoints.

Which approach do you prefer?
