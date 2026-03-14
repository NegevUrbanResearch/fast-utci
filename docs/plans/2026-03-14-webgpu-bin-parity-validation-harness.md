# WebGPU vs .bin Parity Validation Harness Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a programmatic parity harness that compares Python-generated UTCI in `.bin`/`.json` files with live WebGPU results using the same grid and inputs (Ben-Gurion base case first), with shared comparison logic, Playwright-based assertions, and a batch runner for all scenarios. No manual browser checks required; no commit or worktree steps.

**Architecture:** (1) Shared pure-TS comparison module (metrics + pass/fail). (2) Node-friendly reference loader to read `.bin`/`.json` from disk for tests and scripts. (3) Worker supports BVH-only mode so the viewer can supply grid points from `.bin`. (4) Debug viewer parity mode: use .bin grid + expose `window.__parityResults__` for Playwright. (5) Playwright test runs one scenario and asserts via the comparison module. (6) Batch runner reads `data/analyses/manifest.json`, runs each scenario (or a filtered list), collects metrics, writes a single report file.

**Tech Stack:** SvelteKit, TypeScript, Web Workers, WebGPU, Vitest, Playwright, Node `fs` for reference loading. No new runtimes.

**Primary inputs:** `viewer/src/lib/services/dataLoader.ts` (parseFullDayBinary, loadAnalysis), `viewer/src/lib/compute/liveUtciAnalysis.ts`, `viewer/src/lib/compute/mergeAndBvhWorkerClient.ts`, `viewer/src/lib/compute/mergeAndBvh.worker.ts`, `viewer/src/routes/debug-webgpu-utci/+page.svelte`, `data/analyses/Ben-Gurion/20250815_grid_2m_fullday.bin` + `.json`, `data/analyses/manifest.json`.

**Constraints:** No commit or git worktree steps; user manages git.

**Validation strategy:** **Abandon same-grid point-to-point UTCI test for now.** Focus on validating **intermediate stages**: solar exposure, sky exposure, MRT, then UTCI. That isolates where discrepancies come from (BVH/raycast vs weather vs UTCI formula). The e2e test is a smoke test (compute completes, __parityResults__ set); point-count match and UTCI RMSE are not asserted. Revisit point-to-point UTCI once intermediates are validated and the .bin grid path works if needed.

**Intermediate-stage validation:** Solar and sky exposure are validated against reference files produced **once** by Python (see `docs/plans/2026-03-14-intermediate-stage-validation-design.md`). One reference file per stage: `*_solar.json` and `*_sky.json`. **No parity mode:** the debug viewer always exposes `__parityIntermediates__` after compute; grid sizes are not required to match. Validation is **statistical only** (mean and max of exposure within tolerance). **Generate reference:** from repo root run `python scripts/export_ben_gurion_intermediates.py --base-path data/analyses/Ben-Gurion/20250815_grid_2m_fullday --model data/3d_models/Ben-Gurion/original_with_layers.glb` (re-run when Python pipeline or model changes). **Run checks:** `cd viewer && npx playwright test tests/e2e/parity-intermediates.spec.ts`. Tests skip if reference files are missing. Point-to-point UTCI remains out of scope.

**Optional rectangular grid (same-grid-as-.bin):** The debug page supports `?rectangularGrid=1`. When set (and analysis metadata has `bounds`), the WebGPU grid is generated from analysis bounds and `grid_size` in viewer coordinates, so point count and positions align with the .bin analysis. Use this to compare distributions or point-wise when needed, and to get distinct "inside building" values when hiding the building layer. Load the debug page with `?rectangularGrid=1&analysis=Ben-Gurion/20250815_grid_2m_fullday` and confirm point count ~104445 for the same bounds/grid_size.

---

## Task 1: Shared parity comparison module (pure TS)

**Files:**
- Create: `viewer/src/lib/parity/compareParity.ts`
- Create: `viewer/tests/parity/compareParity.test.ts`

**Step 1: Write the failing test**

```typescript
// viewer/tests/parity/compareParity.test.ts
import { describe, it, expect } from 'vitest';
import { compareParity, type ParityResult } from '$lib/parity/compareParity';

describe('compareParity', () => {
	it('returns pass when reference and webgpu UTCI match within tolerance', () => {
		const ref = new Float32Array([25, 26, 27]);
		const webgpu = new Float32Array([25.1, 25.9, 27.05]);
		const result = compareParity({ utciRef: ref, utciWebgpu: webgpu, toleranceC: 0.5 });
		expect(result.pass).toBe(true);
		expect(result.maxError).toBeLessThanOrEqual(0.5);
	});

	it('returns fail when any point exceeds tolerance', () => {
		const ref = new Float32Array([25, 26]);
		const webgpu = new Float32Array([25, 30]);
		const result = compareParity({ utciRef: ref, utciWebgpu: webgpu, toleranceC: 1 });
		expect(result.pass).toBe(false);
		expect(result.maxError).toBe(4);
	});

	it('throws when lengths differ', () => {
		const ref = new Float32Array(3);
		const webgpu = new Float32Array(5);
		expect(() => compareParity({ utciRef: ref, utciWebgpu: webgpu })).toThrow(/length/);
	});

	it('computes rmse and withinTolerancePct', () => {
		const ref = new Float32Array([20, 22, 24]);
		const webgpu = new Float32Array([20, 22.5, 25]);
		const result = compareParity({ utciRef: ref, utciWebgpu: webgpu, toleranceC: 1 });
		expect(result.rmse).toBeGreaterThan(0);
		expect(result.withinTolerancePct).toBeGreaterThanOrEqual(0);
		expect(result.withinTolerancePct).toBeLessThanOrEqual(100);
	});
});
```

**Step 2: Run test to verify it fails**

Run: `cd viewer && npx vitest run tests/parity/compareParity.test.ts`
Expected: FAIL (module not found or compareParity not defined).

**Step 3: Implement compareParity**

```typescript
// viewer/src/lib/parity/compareParity.ts

export interface ParityOptions {
	toleranceC?: number;
}

export interface ParityResult {
	pass: boolean;
	rmse: number;
	maxError: number;
	withinTolerancePct: number;
	numPoints: number;
}

/**
 * Compare reference UTCI (e.g. from .bin) with WebGPU result for one hour.
 * Pure function; safe to run in Node or browser.
 */
export function compareParity(params: {
	utciRef: Float32Array;
	utciWebgpu: Float32Array;
	toleranceC?: number;
}): ParityResult {
	const { utciRef, utciWebgpu, toleranceC = 1 } = params;
	if (utciRef.length !== utciWebgpu.length) {
		throw new Error(`Length mismatch: ref ${utciRef.length} vs webgpu ${utciWebgpu.length}`);
	}
	const n = utciRef.length;
	if (n === 0) {
		return { pass: true, rmse: 0, maxError: 0, withinTolerancePct: 100, numPoints: 0 };
	}
	let sumSq = 0;
	let maxError = 0;
	let within = 0;
	for (let i = 0; i < n; i++) {
		const d = utciWebgpu[i] - utciRef[i];
		sumSq += d * d;
		const absD = Math.abs(d);
		if (absD > maxError) maxError = absD;
		if (absD <= toleranceC) within++;
	}
	const rmse = Math.sqrt(sumSq / n);
	const withinTolerancePct = (100 * within) / n;
	const pass = maxError <= toleranceC;
	return {
		pass,
		rmse,
		maxError,
		withinTolerancePct,
		numPoints: n
	};
}

/**
 * Compare full-day reference vs WebGPU (all hours). Returns one result per hour and an overall pass.
 */
export function compareParityFullDay(params: {
	utciRefByHour: Float32Array[];
	utciWebgpuByHour: Float32Array[];
	toleranceC?: number;
}): { byHour: ParityResult[]; overallPass: boolean; worstHour: number } {
	const { utciRefByHour, utciWebgpuByHour, toleranceC = 1 } = params;
	if (utciRefByHour.length !== utciWebgpuByHour.length) {
		throw new Error(`Hour count mismatch: ref ${utciRefByHour.length} vs webgpu ${utciWebgpuByHour.length}`);
	}
	const byHour: ParityResult[] = [];
	let worstHour = 0;
	let worstMax = -1;
	for (let h = 0; h < utciRefByHour.length; h++) {
		const r = compareParity({
			utciRef: utciRefByHour[h],
			utciWebgpu: utciWebgpuByHour[h],
			toleranceC
		});
		byHour.push(r);
		if (r.maxError > worstMax) {
			worstMax = r.maxError;
			worstHour = h;
		}
	}
	const overallPass = byHour.every((r) => r.pass);
	return { byHour, overallPass, worstHour };
}
```

**Step 4: Run test to verify it passes**

Run: `cd viewer && npx vitest run tests/parity/compareParity.test.ts`
Expected: PASS.

**Step 5: Review checkpoint**

Summary: `compareParity.ts` and tests in place; no staging/commit.

---

## Task 2: Node-friendly reference loader for .bin + .json

**Files:**
- Create: `viewer/src/lib/parity/loadReferenceFromFs.ts`
- Create: `viewer/tests/parity/loadReferenceFromFs.test.ts`
- Reference: `viewer/src/lib/services/dataLoader.ts` (`parseFullDayBinary` signature and `FullDayData` type).

**Step 1: Write the failing test**

Use a path relative to project root so it works from `viewer/`: e.g. `../../data/analyses/Ben-Gurion/20250815_grid_2m_fullday`. The test will run from viewer root; resolve path from `process.cwd()` (which may be repo root when running from repo) or from `import.meta.dirname` / `path.join`. Prefer resolving from repo root: e.g. `path.join(process.cwd(), 'data/analyses/Ben-Gurion/20250815_grid_2m_fullday')` when cwd is repo root; when running from viewer, cwd might be viewer, so use `path.join(process.cwd(), '..', 'data/analyses/...')` or accept a base path option.

```typescript
// viewer/tests/parity/loadReferenceFromFs.test.ts
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { loadReferenceFromFs } from '$lib/parity/loadReferenceFromFs';

const REPO_ROOT = resolve(process.cwd(), process.cwd().endsWith('viewer') ? '..' : '.');

describe('loadReferenceFromFs', () => {
	it('loads Ben-Gurion base .bin + .json and returns positions and utciByHour', async () => {
		const basePath = resolve(REPO_ROOT, 'data/analyses/Ben-Gurion/20250815_grid_2m_fullday');
		const ref = await loadReferenceFromFs(basePath);
		expect(ref.metadata.num_positions).toBe(ref.data.numPositions);
		expect(ref.data.positions.length).toBe(ref.data.numPositions * 3);
		expect(ref.data.utciByHour.length).toBe(24);
		expect(ref.data.utciByHour[0].length).toBe(ref.data.numPositions);
	});

	it('throws when .bin does not exist', async () => {
		await expect(
			loadReferenceFromFs(resolve(REPO_ROOT, 'data/analyses/nonexistent'))
		).rejects.toThrow();
	});
});
```

**Step 2: Run test to verify it fails**

Run: `cd viewer && npx vitest run tests/parity/loadReferenceFromFs.test.ts`
Expected: FAIL.

**Step 3: Implement loadReferenceFromFs**

Reuse `parseFullDayBinary` from dataLoader; use Node `fs.readFileSync` only when running in Node (check `typeof readFileSync === 'function'` or use a small wrapper that imports `fs` only in Node). Use dynamic import of `fs` and `path` so the module can still load in browser (and no-op or throw with a clear message when called in browser).

```typescript
// viewer/src/lib/parity/loadReferenceFromFs.ts
import { parseFullDayBinary } from '$lib/services/dataLoader';

export interface ReferenceMetadata {
	num_positions: number;
	hours: number[];
	analysis_type: string;
	coordinate_system?: string;
	[key: string]: unknown;
}

export interface ReferenceData {
	metadata: ReferenceMetadata;
	data: {
		numPositions: number;
		numHours: number;
		positions: Float32Array;
		utciByHour: Float32Array[];
	};
}

/**
 * Load reference analysis from filesystem (.bin + .json). Node only.
 * @param basePath - Full path without extension, e.g. .../Ben-Gurion/20250815_grid_2m_fullday
 */
export async function loadReferenceFromFs(basePath: string): Promise<ReferenceData> {
	const readFileSync = (await import('node:fs')).readFileSync;
	const metadataPath = `${basePath}.json`;
	const binaryPath = `${basePath}.bin`;
	let metadata: ReferenceMetadata;
	try {
		metadata = JSON.parse(readFileSync(metadataPath, 'utf8'));
	} catch (e) {
		throw new Error(`Failed to load metadata from ${metadataPath}: ${e}`);
	}
	let buffer: Buffer;
	try {
		buffer = readFileSync(binaryPath);
	} catch (e) {
		throw new Error(`Failed to load binary from ${binaryPath}: ${e}`);
	}
	const data = parseFullDayBinary(buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength), metadata);
	return {
		metadata,
		data: {
			numPositions: data.numPositions,
			numHours: data.numHours,
			positions: data.positions,
			utciByHour: data.utciByHour
		}
	};
}
```

Note: `parseFullDayBinary` is in dataLoader and may use `$app/paths`; if it has SSR/browser-only deps, call it from a context that has no `base` dependency for the buffer parse only, or ensure the test runs in Node where `base` is available. If dataLoader imports cause issues, copy the minimal parsing logic into the parity module to keep it Node-safe. Prefer reusing and fixing imports if needed.

**Step 4: Run test to verify it passes**

Run: `cd viewer && npx vitest run tests/parity/loadReferenceFromFs.test.ts`
Expected: PASS. If path is wrong (e.g. cwd is viewer and path is `../data/...`), adjust REPO_ROOT in test or basePath so the test finds the real Ben-Gurion files.

**Step 5: Review checkpoint**

Summary: Reference loader works from Node; no staging/commit.

---

## Task 3: Convert .bin positions (xy_ground) to pipeline (Three.js Y-up) frame

**Files:**
- Create: `viewer/src/lib/parity/analysisToWorld.ts`
- Create: `viewer/tests/parity/analysisToWorld.test.ts`
- Reference: `viewer/src/lib/services/pointCloudService.ts` (xy_ground → world: (x,y,z) → (x, z, -y)).

**Step 1: Write the failing test**

```typescript
// viewer/tests/parity/analysisToWorld.test.ts
import { describe, it, expect } from 'vitest';
import { analysisPositionsToWorld } from '$lib/parity/analysisToWorld';

describe('analysisPositionsToWorld', () => {
	it('converts xy_ground (x,y,z) to world (x, z, -y)', () => {
		const analysis = new Float32Array([10, 20, 1.5]); // one point
		const world = analysisPositionsToWorld(analysis, 'xy_ground');
		expect(world[0]).toBe(10);
		expect(world[1]).toBe(1.5);
		expect(world[2]).toBe(-20);
	});

	it('converts multiple points', () => {
		const analysis = new Float32Array([0, 0, 0, 1, 1, 1]);
		const world = analysisPositionsToWorld(analysis, 'xy_ground');
		expect(world.length).toBe(6);
		expect(world[0]).toBe(0);
		expect(world[1]).toBe(0);
		expect(world[2]).toBe(0);
		expect(world[3]).toBe(1);
		expect(world[4]).toBe(1);
		expect(world[5]).toBe(-1);
	});
});
```

**Step 2: Run test to verify it fails**

Run: `cd viewer && npx vitest run tests/parity/analysisToWorld.test.ts`
Expected: FAIL.

**Step 3: Implement analysisPositionsToWorld**

```typescript
// viewer/src/lib/parity/analysisToWorld.ts

/**
 * Convert positions from analysis (xy_ground) to Three.js world (Y-up).
 * xy_ground: (x, y, z) → world: (x, z, -y). In-place safe if out === positions.
 */
export function analysisPositionsToWorld(
	positions: Float32Array,
	coordinateSystem: 'xy_ground' | 'xz_ground'
): Float32Array {
	const n = positions.length / 3;
	const out = new Float32Array(positions.length);
	if (coordinateSystem === 'xy_ground') {
		for (let i = 0; i < n; i++) {
			const x = positions[i * 3];
			const y = positions[i * 3 + 1];
			const z = positions[i * 3 + 2];
			out[i * 3] = x;
			out[i * 3 + 1] = z;
			out[i * 3 + 2] = -y;
		}
	} else {
		out.set(positions);
	}
	return out;
}
```

**Step 4: Run test to verify it passes**

Run: `cd viewer && npx vitest run tests/parity/analysisToWorld.test.ts`
Expected: PASS.

**Step 5: Review checkpoint**

Summary: Conversion helper and tests in place; no staging/commit.

---

## Task 4: Worker BVH-only mode and client support

**Files:**
- Modify: `viewer/src/lib/compute/mergeAndBvh.worker.ts`
- Modify: `viewer/src/lib/compute/mergeAndBvhWorkerClient.ts`
- Test: `viewer/tests/compute/mergeAndBvhWorkerClient.test.ts` or `viewer/tests/compute/compute-manager.test.ts`

**Step 1: Write the failing test**

Add a test that runs the worker with `bvhOnly: true` and asserts the result has `serializedBvh` and either no grid points or zero-length gridPoints; and that the client can call `runMergeAndBvhInWorker` with `bvhOnly: true` and receive a result that has only serializedBvh (gridPoints length 0 or not used). Prefer an existing compute or worker test file.

```typescript
// In viewer/tests/compute/mergeAndBvhWorkerClient.test.ts or a new test file
it('when bvhOnly is true, returns serializedBvh and empty gridPoints', async () => {
	// Use minimal mesh payload if available from fixtures
	const result = await runMergeAndBvhInWorker({
		meshes: [/* minimal payload */],
		gridResolution: 2,
		zHeight: 0.9,
		bvhOnly: true
	});
	expect(result.serializedBvh).toBeDefined();
	expect(result.gridPoints.length).toBe(0);
});
```

If no minimal mesh fixture exists, create a tiny payload (e.g. one triangle) in the test or skip the test when Worker is undefined and document that the contract is "bvhOnly returns gridPoints.length === 0".

**Step 2: Run test to verify it fails**

Run: `cd viewer && npx vitest run tests/compute/mergeAndBvhWorkerClient.test.ts -t bvhOnly` (or the test file you added to).
Expected: FAIL or test not found.

**Step 3: Implement worker bvhOnly**

In `mergeAndBvh.worker.ts`:
- Extend `StartRequest` with optional `bvhOnly?: boolean`.
- When `bvhOnly === true`: after merge and creating `mesh`, skip `generateGridFromMesh` and building `gridPoints`. Build BVH from `mesh` (same as now), serialize, then `postMessage({ serializedBvh, gridPoints: new Float32Array(0) }, [serialized buffers])`. Transfer list cannot include the empty Float32Array's buffer (or use a 1-element buffer for compatibility). Check: transfer list with empty array — use `gridPoints = new Float32Array(0)`; `gridPoints.buffer` is still valid but byteLength 0; some runtimes allow transferring it. Safer: send a dummy 1-float array so transfer list is valid, and document that client must ignore gridPoints when bvhOnly.

Simpler: when bvhOnly, do not add gridPoints.buffer to transfer list; send `gridPoints: new Float32Array(0)`. Transfer list: only serializedBvh buffers. Then in client, when bvhOnly was requested, treat returned gridPoints as empty and use caller-supplied grid points later.

**Step 4: Implement client support**

In `mergeAndBvhWorkerClient.ts`:
- Add `bvhOnly?: boolean` to the options passed to the worker.
- In the request object sent to the worker, include `bvhOnly: true` when set.
- When `bvhOnly` was true, the response will have `gridPoints` length 0; the client returns the result as-is. The caller (parity flow) will merge worker result (serializedBvh) with .bin grid points and pass both to initFromModelAndWeather.

**Step 5: Run test to verify it passes**

Run the same vitest command. Expected: PASS.

**Step 6: Review checkpoint**

Summary: Worker can return BVH only; client can request it; no staging/commit.

---

## Task 5: Parity mode in debug viewer (use .bin grid, expose __parityResults__)

**Files:**
- Modify: `viewer/src/routes/debug-webgpu-utci/+page.svelte`
- Reference: `viewer/src/lib/parity/loadReferenceFromFs.ts` (browser cannot use it; load reference in viewer via existing loadAnalysis which uses fetch), `viewer/src/lib/parity/analysisToWorld.ts`, `viewer/src/lib/compute/liveUtciAnalysis.ts`.

**Context:** In the browser we already have `loadAnalysis(analysisId)` which fetches `.json` and `.bin` and returns `{ metadata, data }` with `data.positions` and `data.utciByHour`. So for parity mode we do not use loadReferenceFromFs (that is Node-only); we use the existing `loadAnalysisData` / `loadAnalysis` that the page already uses to load the .bin. So parity mode: (1) Detect parity mode (e.g. URL param `parity=1` or `parityMode=true`). (2) When running live compute for that analysis, use grid points from the already-loaded base analysis (same as left side): convert `base.data.positions` (analysis coords) to world via `analysisPositionsToWorld`, then we need BVH from the model — so run worker with `bvhOnly: true`, then pass `gridPoints: analysisPositionsToWorld(base.data.positions, base.metadata.coordinate_system || 'xy_ground')` and `serializedBvh` from worker. (3) After createLiveUtciAnalysisFromCompute resolves, set `window.__parityResults__ = { utciByHour: result.data.utciByHour, positions: result.data.positions, numPoints: result.data.numPositions, numHours: result.data.utciByHour.length }`. Use a type declaration for `window.__parityResults__` in a small `.d.ts` or in the page.

**Step 1: Add parity mode detection and branch in run logic**

In `+page.svelte`, where the live compute is triggered (reactive block that calls `prepareMeshPayloadForWorkerAsync` then `runMergeAndBvhInWorker` then `createLiveUtciAnalysisFromCompute`):
- Define `parityMode = $page.url.searchParams.get('parity') === '1' || $page.url.searchParams.get('parityMode') === 'true'`.
- When `parityMode` and we have `base` (analysis store with the current analysis loaded, i.e. base.data has positions and utciByHour):
  - Convert base.data.positions to world: `const worldGridPoints = analysisPositionsToWorld(base.data.positions, base.metadata.coordinate_system || 'xy_ground')`.
  - Call worker with same `meshes` and options but `bvhOnly: true`. Await result.
  - Build `workerResult` as `{ gridPoints: worldGridPoints, serializedBvh: result.serializedBvh }` (so we're reusing the worker's BVH but supplying .bin grid).
  - Call `createLiveUtciAnalysisFromCompute` with this `workerResult` (and same analysisId, baseMetadata, epwContent, etc.).
- After `createLiveUtciAnalysisFromCompute` resolves, if `parityMode`, set:
  - `(window as any).__parityResults__ = { utciByHour: result.data.utciByHour.map((a) => a.slice(0)), positions: result.data.positions.slice(0), numPoints: result.data.numPositions, numHours: result.data.utciByHour.length }` (copy arrays so Playwright gets serializable data; or expose references if Playwright will read from same-origin page and only need to pass back to Node for comparison — then copy on the test side when calling page.evaluate). For Playwright we need to get arrays to Node; Float32Array is not JSON-serializable. So either: (A) expose as lists: `utciByHour: result.data.utciByHour.map(arr => Array.from(arr))`, `positions: Array.from(result.data.positions)`, or (B) expose a getter that returns JSON. Use (A) so Playwright can `page.evaluate(() => window.__parityResults__)` and get a plain object with number arrays.

**Step 2: Implement worker call with bvhOnly when parityMode**

- When parityMode, call `runMergeAndBvhInWorker({ meshes, gridResolution, zHeight, signal, bvhOnly: true })`. Then build `workerResult = { gridPoints: worldGridPoints, serializedBvh: result.serializedBvh }` and pass to createLiveUtciAnalysisFromCompute. Ensure numPoints from .bin matches (worldGridPoints.length / 3 === base.data.numPositions); otherwise pipeline and reference will mismatch.

**Step 3: Expose __parityResults__ after run**

After `liveAnalysis = result` (and comparisonStore update), if parityMode:

```typescript
(window as any).__parityResults__ = {
	utciByHour: result.data.utciByHour.map((arr) => Array.from(arr)),
	positions: Array.from(result.data.positions),
	numPoints: result.data.numPositions,
	numHours: result.data.utciByHour.length
};
```

**Step 4: Manual check**

Run viewer, open debug page with `?parity=1` and analysis Ben-Gurion/20250815_grid_2m_fullday, wait for compute to finish, in console run `window.__parityResults__` and confirm structure and numPoints/numHours.

**Step 5: Review checkpoint**

Summary: Parity mode uses .bin grid and exposes results; no staging/commit.

---

## Task 6: Playwright parity test (Ben-Gurion base case)

**Files:**
- Create: `viewer/tests/e2e/parity-ben-gurion.spec.ts` (or under `viewer/tests/` with Playwright config)
- Ensure Playwright is configured for viewer (see existing e2e if any); if not, add minimal Playwright config to run one browser test.
- Reference: `viewer/src/lib/parity/compareParity.ts`, `viewer/src/lib/parity/loadReferenceFromFs.ts`, `viewer/src/lib/parity/compareParityFullDay`

**Step 1: Add Playwright dependency if missing**

If `viewer/package.json` does not have `@playwright/test`, add it as devDependency and add a minimal `playwright.config.ts` (baseURL to viewer dev server, one project Chromium). Skip if already present.

**Step 2: Write the test**

- In the test: load reference from disk with `loadReferenceFromFs(pathToBenGurionBase)`.
- Launch browser, goto debug page with `?parity=1&analysis=Ben-Gurion/20250815_grid_2m_fullday` (or whatever URL and query the debug route uses).
- Wait for `window.__parityResults__` to be defined (poll page.evaluate until it exists or timeout after e.g. 120s).
- Get results: `const webgpu = await page.evaluate(() => window.__parityResults__)`.
- Convert webgpu.utciByHour (array of number[]) to Float32Array[] and compare with reference using `compareParityFullDay({ utciRefByHour: ref.data.utciByHour, utciWebgpuByHour: webgpuFloat32Arrays, toleranceC: 2 })`. Use tolerance 2°C initially so the test does not block; we can tighten later.
- Assert `overallPass` or at least assert `worstHour` maxError is below a threshold (e.g. 3) and log byHour metrics.

**Step 3: Run test**

Run: `cd viewer && npx playwright test tests/e2e/parity-ben-gurion.spec.ts` (or the path you used). Ensure dev server is not required if Playwright is configured to start it; otherwise start dev server and run test. Expected: Test runs; it may FAIL on parity (maxError > tolerance) until pipeline is fixed; the test must complete and perform the assertion.

**Step 4: Review checkpoint**

Summary: One Playwright test runs Ben-Gurion base case and asserts programmatically; no staging/commit.

---

## Task 7: Batch runner script for all scenarios

**Files:**
- Create: `viewer/scripts/run-parity-batch.ts` (or `.js` with ts-node/tsx, or add to package.json script)
- Read: `data/analyses/manifest.json` (list of analyses with `id`, `path`, `project`).

**Step 1: Implement batch runner**

- Read manifest: parse `data/analyses/manifest.json` (path relative to repo root; script may run from viewer or repo root).
- For each entry (or a filtered list, e.g. only `path` that have corresponding .bin/.json on disk), run the same flow as the Playwright test: launch browser, open debug URL with parity mode and that analysis id, wait for __parityResults__, load reference from fs for that analysis (basePath = resolve(repoRoot, 'data/analyses', entry.path)), compare, record result (analysisId, pass, rmse, maxError, worstHour).
- Write a report (e.g. `parity-report.json` or `parity-report.md`) with one line/block per scenario and overall summary. Optionally support a `--filter` flag to run only analyses whose id matches a pattern (e.g. Ben-Gurion only).

**Step 2: Add npm script**

In `viewer/package.json` add a script, e.g. `"parity:batch": "tsx scripts/run-parity-batch.ts"` (or node + compiled, or playwright test with a config that runs multiple specs). Prefer reusing Playwright: one spec that accepts an env var or config for which analysis id to run, then the batch script invokes Playwright multiple times with different env vars and aggregates results into a report. That avoids implementing a second browser automation path.

**Step 3: Run batch for a small subset**

Run the batch with a filter so only 1–2 scenarios run (e.g. Ben-Gurion base only). Confirm report is generated.

**Step 4: Review checkpoint**

Summary: Batch runner produces a report for multiple scenarios; no staging/commit.

---

## Execution handoff

Plan complete and saved to `docs/plans/2026-03-14-webgpu-bin-parity-validation-harness.md`.

Two execution options:

1. **Subagent-driven (this session)** – I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Parallel session (separate)** – Open a new session with executing-plans and run task-by-task with checkpoints.

Which approach do you prefer?
