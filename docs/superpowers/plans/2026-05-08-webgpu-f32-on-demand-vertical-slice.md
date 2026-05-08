# WebGPU F32 On-Demand Vertical Slice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render one selected UTCI hour produced by the `f32` WebGPU on-demand path in the app, without the all-hours UTCI/MRT readback or CPU `DataTexture` rebuild path.

**Architecture:** Keep the existing `.bin` and `runAll()` paths as the default baseline. Add an explicitly flagged `f32` on-demand path that captures device/runtime diagnostics, runs exposure-only precompute, dispatches one-hour UTCI, and renders selected-hour colors through a clearly labeled render transport. The preferred transport is a direct compute-buffer bridge; if Three/WebGPU interop prevents that, use a selected-hour `Float32Array` upload and label it `cpu-uploaded-selected-hour` rather than zero-copy. Treat packed output and broader memory compression as intentionally deferred until this vertical slice is app-visible and user-verified.

**Tech Stack:** SvelteKit, Svelte 5, Three.js r175 `WebGPURenderer`, Three node materials / `StorageBufferAttribute`, WGSL, WebGPU compute buffers, Vitest, Playwright.

**User Constraints:** Do not create git worktrees. Do not commit. This plan intentionally contains no commit steps. Do not implement this plan until the user reviews it and chooses an execution path.

---

## Current State

Implemented:

- `viewer/src/lib/compute/webgpuUtciPipeline.ts` has `runExposurePrecompute()`, `runUtciForTimeIndex()`, and `readOnDemandUtciForDebug()`.
- `viewer/src/lib/compute/shaders/mrt_utci_on_demand.wgsl` contains real one-hour MRT/UTCI math and preserves current boundary averaging semantics.
- `viewer/tests/e2e/webgpu-on-demand-prototype.spec.ts` proves focused one-hour `f32` parity for `timeIndex=12` against the all-hours WebGPU slice.
- `viewer/src/lib/services/gpuUtciRenderBridge.ts` proves Three-owned storage-backed rendering can affect visible color.
- `docs/superpowers/plans/2026-05-07-webgpu-compute-on-demand-prototype-results.md` records the prototype status and correctly marks several gates as partial.

Not yet proven:

- Renderer and compute resources are not confirmed to share one concrete `GPUDevice`.
- The render bridge does not consume `runUtciForTimeIndex()` output directly.
- The main app path still builds UTCI display from CPU `Analysis` data.
- Exposure-only precompute is guarded mostly by source-lock tests, not runtime allocation/timing evidence.
- One-hour parity is narrow: `timeIndex=12` only, against all-hours WebGPU, not a multi-hour app-visible baseline.

## Scope And Hard Gates

In scope:

- Capture WebGPU adapter/device/limit diagnostics where the browser exposes them.
- Prove an exposure-only path runs without allocating all-hours UTCI/MRT buffers.
- Render one selected `f32` UTCI hour in the app/debug route behind a feature flag.
- Instrument the path so all-hours readback and `DataTexture` rebuilds are visibly absent from the selected-hour route.
- Classify the selected-hour render transport as `compute-buffer-selected-hour` or `cpu-uploaded-selected-hour`.
- Validate selected-hour output against the current baseline for hours `12`, `23`, `16`, and `17`, including the known point `31079` when that dataset is loaded.
- Preserve the default production path and allow easy fallback with `utciRender=data` or by omitting the on-demand flag.

Out of scope:

- `pack2x16float` / packed MRT+UTCI output.
- Spatial tiling and 0.5m full production scaling.
- Wind/sun-rights/future derived layers.
- Replacing the default `.bin` or `runAll()` path.
- Broad repo type-check cleanup unrelated to this feature.

Hard gates:

1. **Diagnostics gate:** the route records renderer backend, `navigator.gpu`, adapter info if available, WebGPU limits, model/scenario, point count, grid resolution, and selected time indices.
2. **Exposure-only runtime gate:** a strict route path reaches one-hour compute via `runExposurePrecompute()`, not `runAll()`, and diagnostics show no all-hours UTCI/MRT output allocation for that path.
3. **F32 render gate:** a selected hour visibly renders through the WebGPU renderer path under an explicit flag.
4. **No-all-hours-hot-path gate:** rendering the selected hour does not perform all-hours UTCI/MRT readback and does not rebuild a CPU `DataTexture`. If selected-hour CPU upload is used, diagnostics must report exactly one selected-hour transfer and must not call it zero-copy.
5. **Multi-hour parity gate:** the selected-hour path matches a separate baseline source for hours `12`, `23`, `16`, and `17` within the existing `f32` tolerance, and reports the exact values for point `31079` when present.
6. **Fallback gate:** the user can switch back to the current CPU/data-texture path without reloading code or changing data files.

## File Structure

Modify:

- `viewer/src/lib/compute/gpu-pipeline.ts`  
  Add diagnostics/allocation metadata to the on-demand interfaces.

- `viewer/src/lib/compute/webgpuUtciPipeline.ts`  
  Track whether the latest on-demand run used exposure-only precompute, capture buffer allocation sizes, expose runtime diagnostics, and keep debug readback opt-in.

- `viewer/src/lib/compute/compute-manager.ts`  
  Add thin accessors for on-demand diagnostics and preserve existing all-hours behavior.

- `viewer/src/lib/services/gpuUtciRenderBridge.ts`  
  Add an app-facing `f32` selected-hour render bridge contract. Initially it may upload an `f32` selected-hour array into a Three-owned storage attribute if raw compute-buffer sharing is impossible, but diagnostics must label that as "GPU-rendered, CPU-uploaded" rather than "zero-copy".

- `viewer/src/lib/services/pointCloudService.ts`  
  Route selected-hour on-demand surface creation/update through the bridge while preserving the existing data-texture backend.

- `viewer/src/lib/utciRenderMode.ts`  
  Keep the explicit render-mode/fallback contract and add an on-demand mode flag only if it keeps the existing `auto | gpu | data` semantics clear.

- `viewer/src/lib/components/scene/UTCIPointCloud.svelte`  
  Accept selected-hour on-demand surface data behind an explicit prop or store signal; keep the existing `Analysis` path intact.

- `viewer/src/routes/+page.svelte`  
  Add main-route diagnostics and a guarded on-demand experiment flag without changing the default route behavior.

- `viewer/src/routes/debug-webgpu-utci/+page.svelte`  
  Add strict exposure-only and multi-hour comparison paths.

- `viewer/tests/e2e/webgpu-on-demand-prototype.spec.ts`  
  Extend existing prototype E2E coverage.

- `docs/superpowers/plans/2026-05-07-webgpu-compute-on-demand-prototype-results.md`  
  Append verified vertical-slice results after implementation.

Create:

- `viewer/src/lib/compute/onDemandDiagnostics.ts`  
  Shared types and helpers for diagnostics, timing, allocation, and selected-hour comparison summaries.

- `viewer/tests/compute/onDemandDiagnostics.test.ts`

- `viewer/tests/e2e/webgpu-f32-on-demand-vertical-slice.spec.ts`

## Milestone 1: Make Diagnostics Real

### Task 1: Add On-Demand Diagnostics Types

**Files:**
- Create: `viewer/src/lib/compute/onDemandDiagnostics.ts`
- Test: `viewer/tests/compute/onDemandDiagnostics.test.ts`

- [ ] **Step 1: Write failing tests**

Create `viewer/tests/compute/onDemandDiagnostics.test.ts`:

```ts
import { describe, expect, it } from 'vitest';
import {
	createEmptyOnDemandDiagnostics,
	recordOnDemandTiming,
	type OnDemandRuntimeDiagnostics
} from '$lib/compute/onDemandDiagnostics';

describe('on-demand runtime diagnostics', () => {
	it('starts with conservative defaults', () => {
		const diagnostics = createEmptyOnDemandDiagnostics();

		expect(diagnostics.navigatorGpu).toBe(false);
		expect(diagnostics.rendererBackend).toBe('unknown');
		expect(diagnostics.path).toBe('idle');
		expect(diagnostics.debugReadbackCount).toBe(0);
		expect(diagnostics.dataTextureBuildCount).toBe(0);
		expect(diagnostics.usedRunAllForSelectedHour).toBe(false);
		expect(diagnostics.usedExposureOnlyPrecompute).toBe(false);
		expect(diagnostics.allHoursUtciBytesAllocated).toBe(0);
		expect(diagnostics.allHoursMrtBytesAllocated).toBe(0);
	});

	it('records timing entries without dropping existing metadata', () => {
		const diagnostics: OnDemandRuntimeDiagnostics = {
			...createEmptyOnDemandDiagnostics(),
			path: 'exposure-only-f32',
			pointCount: 511_840
		};

		const next = recordOnDemandTiming(diagnostics, 'oneHourDispatchMs', 12.5);

		expect(next.path).toBe('exposure-only-f32');
		expect(next.pointCount).toBe(511_840);
		expect(next.timings.oneHourDispatchMs).toBe(12.5);
	});
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```powershell
cd viewer
npm test -- tests/compute/onDemandDiagnostics.test.ts
```

Expected: fails because `onDemandDiagnostics.ts` does not exist.

- [ ] **Step 3: Implement diagnostics helpers**

Create `viewer/src/lib/compute/onDemandDiagnostics.ts`:

```ts
export type OnDemandRendererBackend = 'webgpu' | 'unknown';
export type OnDemandPath = 'idle' | 'run-all-baseline' | 'exposure-only-f32' | 'error';

export interface OnDemandTimings {
	exposurePrecomputeMs?: number;
	oneHourDispatchMs?: number;
	renderUpdateMs?: number;
	debugReadbackMs?: number;
}

export interface OnDemandRuntimeDiagnostics {
	navigatorGpu: boolean;
	rendererBackend: OnDemandRendererBackend;
	path: OnDemandPath;
	adapterInfo?: string;
	maxStorageBufferBindingSize?: number;
	maxBufferSize?: number;
	maxStorageBuffersPerShaderStage?: number;
	modelId?: string;
	scenarioId?: string;
	gridResolution?: number;
	pointCount?: number;
	timeIndices: number[];
	usedRunAllForSelectedHour: boolean;
	usedExposureOnlyPrecompute: boolean;
	allHoursUtciBytesAllocated: number;
	allHoursMrtBytesAllocated: number;
	oneHourOutputBytes: number;
	selectedHourTransferCount: number;
	renderTransport: 'none' | 'compute-buffer-selected-hour' | 'cpu-uploaded-selected-hour';
	debugReadbackCount: number;
	dataTextureBuildCount: number;
	timings: OnDemandTimings;
	error?: string;
}

export function createEmptyOnDemandDiagnostics(): OnDemandRuntimeDiagnostics {
	return {
		navigatorGpu: false,
		rendererBackend: 'unknown',
		path: 'idle',
		timeIndices: [],
		usedRunAllForSelectedHour: false,
		usedExposureOnlyPrecompute: false,
		allHoursUtciBytesAllocated: 0,
		allHoursMrtBytesAllocated: 0,
		oneHourOutputBytes: 0,
		selectedHourTransferCount: 0,
		renderTransport: 'none',
		debugReadbackCount: 0,
		dataTextureBuildCount: 0,
		timings: {}
	};
}

export function recordOnDemandTiming<K extends keyof OnDemandTimings>(
	diagnostics: OnDemandRuntimeDiagnostics,
	key: K,
	value: number
): OnDemandRuntimeDiagnostics {
	return {
		...diagnostics,
		timings: {
			...diagnostics.timings,
			[key]: value
		}
	};
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run:

```powershell
cd viewer
npm test -- tests/compute/onDemandDiagnostics.test.ts
```

Expected: test passes.

### Task 2: Surface Runtime Diagnostics From The Pipeline

**Files:**
- Modify: `viewer/src/lib/compute/gpu-pipeline.ts`
- Modify: `viewer/src/lib/compute/webgpuUtciPipeline.ts`
- Modify: `viewer/src/lib/compute/compute-manager.ts`
- Test: `viewer/tests/compute/compute-manager-on-demand.test.ts`

- [ ] **Step 1: Add failing tests for diagnostics access**

In `viewer/tests/compute/compute-manager-on-demand.test.ts`, add:

```ts
it('exposes on-demand diagnostics when the pipeline supports them', async () => {
	const pipeline = createMockPipeline({
		getOnDemandDiagnostics: () => ({
			navigatorGpu: true,
			rendererBackend: 'webgpu',
			path: 'exposure-only-f32',
			timeIndices: [12],
			usedRunAllForSelectedHour: false,
			usedExposureOnlyPrecompute: true,
			allHoursUtciBytesAllocated: 0,
			allHoursMrtBytesAllocated: 0,
			oneHourOutputBytes: 400,
			selectedHourTransferCount: 0,
			renderTransport: 'none',
			debugReadbackCount: 0,
			dataTextureBuildCount: 0,
			timings: {}
		})
	});
	const manager = new ComputeManager(pipeline);

	expect(manager.getOnDemandDiagnostics()?.path).toBe('exposure-only-f32');
	expect(manager.getOnDemandDiagnostics()?.oneHourOutputBytes).toBe(400);
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```powershell
cd viewer
npm test -- tests/compute/compute-manager-on-demand.test.ts
```

Expected: fails because the interface/manager accessor does not exist.

- [ ] **Step 3: Add interface and manager accessor**

In `viewer/src/lib/compute/gpu-pipeline.ts`, import diagnostics:

```ts
import type { OnDemandRuntimeDiagnostics } from '$lib/compute/onDemandDiagnostics';
```

Add to `UTCIComputePipeline`:

```ts
getOnDemandDiagnostics?(): OnDemandRuntimeDiagnostics;
```

In `viewer/src/lib/compute/compute-manager.ts`, add:

```ts
getOnDemandDiagnostics(): import('$lib/compute/onDemandDiagnostics').OnDemandRuntimeDiagnostics | undefined {
	return this.pipeline.getOnDemandDiagnostics?.();
}
```

- [ ] **Step 4: Track pipeline allocation state**

In `viewer/src/lib/compute/webgpuUtciPipeline.ts`, add a field:

```ts
private onDemandDiagnostics = createEmptyOnDemandDiagnostics();
```

Import helpers:

```ts
import {
	createEmptyOnDemandDiagnostics,
	recordOnDemandTiming,
	type OnDemandRuntimeDiagnostics
} from '$lib/compute/onDemandDiagnostics';
```

Add:

```ts
getOnDemandDiagnostics(): OnDemandRuntimeDiagnostics {
	return {
		...this.onDemandDiagnostics,
		timeIndices: [...this.onDemandDiagnostics.timeIndices],
		timings: { ...this.onDemandDiagnostics.timings }
	};
}
```

In `runAll()`, after `const utciBytes = numPoints * totalTimeSteps * 4;` and `const mrtBytes = numPoints * totalTimeSteps * 4;`, record:

```ts
this.onDemandDiagnostics = {
	...this.onDemandDiagnostics,
	path: 'run-all-baseline',
	usedRunAllForSelectedHour: true,
	usedExposureOnlyPrecompute: false,
	allHoursUtciBytesAllocated: utciBytes,
	allHoursMrtBytesAllocated: mrtBytes
};
```

In `runExposurePrecompute()`, put `const started = performance.now();` as the first statement after `{ numPoints, numHours, numMonths }` are read from `params`, and put the elapsed-time update immediately after `this.queue.submit([encoder.finish()]);`:

```ts
const started = performance.now();
// Keep the existing exposure-only buffer allocation and encoder submission here.
const elapsed = performance.now() - started;
this.onDemandDiagnostics = recordOnDemandTiming(
	{
		...this.onDemandDiagnostics,
		path: 'exposure-only-f32',
		usedRunAllForSelectedHour: false,
		usedExposureOnlyPrecompute: true,
		allHoursUtciBytesAllocated: 0,
		allHoursMrtBytesAllocated: 0
	},
	'exposurePrecomputeMs',
	elapsed
);
```

In `runUtciForTimeIndex()`, put `const started = performance.now();` immediately before `const encoder = this.device.createCommandEncoder();`, keep the existing compute dispatch, then put the elapsed-time update immediately after `this.queue.submit([encoder.finish()]);`:

```ts
const started = performance.now();
// Keep the existing one-hour compute dispatch here.
const elapsed = performance.now() - started;
this.onDemandDiagnostics = recordOnDemandTiming(
	{
		...this.onDemandDiagnostics,
		path: 'exposure-only-f32',
		timeIndices: Array.from(new Set([...this.onDemandDiagnostics.timeIndices, timeIndex])),
		oneHourOutputBytes: outputBytes
	},
	'oneHourDispatchMs',
	elapsed
);
```

In `readOnDemandUtciForDebug()`, put `const started = performance.now();` immediately before `const encoder = this.device.createCommandEncoder();`, keep the existing copy/map/unmap logic, then put the elapsed-time update immediately before `return out;`:

```ts
const started = performance.now();
// Keep the existing debug readback copy/map/unmap here.
const elapsed = performance.now() - started;
this.onDemandDiagnostics = recordOnDemandTiming(
	{
		...this.onDemandDiagnostics,
		debugReadbackCount: this.onDemandDiagnostics.debugReadbackCount + 1
	},
	'debugReadbackMs',
	elapsed
);
```

- [ ] **Step 5: Run focused tests**

Run:

```powershell
cd viewer
npm test -- tests/compute/onDemandDiagnostics.test.ts tests/compute/compute-manager-on-demand.test.ts
```

Expected: both files pass.

## Milestone 2: Prove Exposure-Only Runtime Path

### Task 3: Add A Strict Static-Upload Entry Point

**Files:**
- Modify: `viewer/src/routes/debug-webgpu-utci/+page.svelte`
- Test: `viewer/tests/e2e/webgpu-f32-on-demand-vertical-slice.spec.ts`

- [ ] **Step 1: Write failing E2E**

Create `viewer/tests/e2e/webgpu-f32-on-demand-vertical-slice.spec.ts`:

```ts
import { expect, test } from '@playwright/test';

test('strict exposure-only f32 path does not construct the all-hours live Analysis', async ({ page }) => {
	await page.goto('/debug-webgpu-utci?onDemandPrototype=1&strictExposureOnly=1&timeIndex=12');

	const hasWebGpu = await page.evaluate(() => Boolean(navigator.gpu));
	test.skip(!hasWebGpu && process.env.REQUIRE_WEBGPU_ON_DEMAND !== '1', 'WebGPU unavailable.');

	await expect(page.getByTestId('on-demand-prototype-status')).toContainText(/ready|error/i, {
		timeout: 120_000
	});

	const diagnostics = await page.evaluate(() => {
		return (window as Window & { __onDemandPrototypeDiagnostics__?: any })
			.__onDemandPrototypeDiagnostics__;
	});

	expect(diagnostics).toBeTruthy();
	expect(diagnostics.path).toBe('exposure-only-f32');
	expect(diagnostics.usedExposureOnlyPrecompute).toBe(true);
	expect(diagnostics.usedRunAllForSelectedHour).toBe(false);
	expect(diagnostics.liveAnalysisConstructedForSelectedHour).toBe(false);
	expect(diagnostics.allHoursUtciBytesAllocated).toBe(0);
	expect(diagnostics.allHoursMrtBytesAllocated).toBe(0);
	expect(diagnostics.oneHourOutputBytes).toBeGreaterThan(0);
});
```

- [ ] **Step 2: Run E2E to verify it fails**

Run:

```powershell
cd viewer
$env:REQUIRE_WEBGPU_ON_DEMAND='1'
npm run test:e2e -- tests/e2e/webgpu-f32-on-demand-vertical-slice.spec.ts --project=chromium
Remove-Item Env:\REQUIRE_WEBGPU_ON_DEMAND
```

Expected: fails because the debug route still reaches the current live-analysis path before on-demand comparison.

- [ ] **Step 3: Add the strict branch before live analysis construction**

In `viewer/src/routes/debug-webgpu-utci/+page.svelte`, add this branch before the code path that calls `createLiveUtciAnalysisFromCompute()`:

```ts
if (onDemandPrototypeEnabled && $page.url.searchParams.get('strictExposureOnly') === '1') {
	onDemandPrototypeStatus = 'diagnostics';
	const timeIndex = Number($page.url.searchParams.get('timeIndex') ?? '12');

	const setup = await prepareWebgpuDebugInputsForCurrentSelection();
	const computeManager = setup.computeManager;
	const { numPoints, numHours, numMonths } = setup;

	await computeManager.runExposurePrecompute({ numPoints, numHours, numMonths });
	await computeManager.runUtciForTimeIndex({
		timeIndex,
		numPoints,
		numHours,
		numMonths,
		format: 'f32-utci'
	});

	const pipelineDiagnostics = computeManager.getOnDemandDiagnostics();
	window.__onDemandPrototypeDiagnostics__ = {
		...window.__onDemandPrototypeDiagnostics__,
		...pipelineDiagnostics,
		navigatorGpu: Boolean(navigator.gpu),
		rendererBackend,
		modelId: selectedProject,
		pointCount: numPoints,
		timeIndices: pipelineDiagnostics?.timeIndices ?? [timeIndex],
		liveAnalysisConstructedForSelectedHour: false
	};
	onDemandPrototypeStatus = 'ready';
	return;
}
```

If `prepareWebgpuDebugInputsForCurrentSelection()` does not exist, extract it from the existing debug-route setup code that loads model geometry, weather, sun vectors, dome vectors, grid points, and BVH data, but stop before `runAll()` or `createLiveUtciAnalysisFromCompute()` is called. The helper must return:

```ts
{
	computeManager: ComputeManager;
	numPoints: number;
	numHours: number;
	numMonths: number;
	gridResolution?: number;
}
```

- [ ] **Step 4: Run strict static-upload E2E**

Run:

```powershell
cd viewer
$env:REQUIRE_WEBGPU_ON_DEMAND='1'
npm run test:e2e -- tests/e2e/webgpu-f32-on-demand-vertical-slice.spec.ts --project=chromium
Remove-Item Env:\REQUIRE_WEBGPU_ON_DEMAND
```

Expected: test passes.

### Task 4: Add Strict Exposure-Only Timing E2E

**Files:**
- Modify: `viewer/src/routes/debug-webgpu-utci/+page.svelte`
- Modify: `viewer/tests/e2e/webgpu-f32-on-demand-vertical-slice.spec.ts`

- [ ] **Step 1: Write failing E2E**

Append to `viewer/tests/e2e/webgpu-f32-on-demand-vertical-slice.spec.ts`:

```ts
import { expect, test } from '@playwright/test';

test('strict exposure-only f32 path records runtime timings', async ({ page }) => {
	await page.goto('/debug-webgpu-utci?onDemandPrototype=1&strictExposureOnly=1&timeIndex=12');

	const hasWebGpu = await page.evaluate(() => Boolean(navigator.gpu));
	test.skip(!hasWebGpu && process.env.REQUIRE_WEBGPU_ON_DEMAND !== '1', 'WebGPU unavailable.');

	await expect(page.getByTestId('on-demand-prototype-status')).toContainText(/ready|error/i, {
		timeout: 120_000
	});

	const diagnostics = await page.evaluate(() => {
		return (window as Window & { __onDemandPrototypeDiagnostics__?: any })
			.__onDemandPrototypeDiagnostics__;
	});

	expect(diagnostics).toBeTruthy();
	expect(diagnostics.timings.exposurePrecomputeMs).toBeGreaterThan(0);
	expect(diagnostics.timings.oneHourDispatchMs).toBeGreaterThanOrEqual(0);
});
```

- [ ] **Step 2: Run E2E to verify it fails**

Run:

```powershell
cd viewer
$env:REQUIRE_WEBGPU_ON_DEMAND='1'
npm run test:e2e -- tests/e2e/webgpu-f32-on-demand-vertical-slice.spec.ts --project=chromium
Remove-Item Env:\REQUIRE_WEBGPU_ON_DEMAND
```

Expected: fails if timings are not recorded.

- [ ] **Step 3: Implement strict route branch**

In `viewer/src/routes/debug-webgpu-utci/+page.svelte`, add derived flags:

```ts
$: strictExposureOnlyEnabled =
	onDemandPrototypeEnabled && $page.url.searchParams.get('strictExposureOnly') === '1';
```

In the strict branch from Task 3, keep the existing `runExposurePrecompute()` and `runUtciForTimeIndex()` sequence:

```ts
await computeManager.runExposurePrecompute({ numPoints, numHours, numMonths });
await computeManager.runUtciForTimeIndex({
	timeIndex,
	numPoints,
	numHours,
	numMonths,
	format: 'f32-utci'
});
```

After the branch, merge diagnostics into `window.__onDemandPrototypeDiagnostics__`:

```ts
const pipelineDiagnostics = computeManager.getOnDemandDiagnostics?.();
window.__onDemandPrototypeDiagnostics__ = {
	...window.__onDemandPrototypeDiagnostics__,
	...pipelineDiagnostics,
	navigatorGpu: Boolean(navigator.gpu),
	rendererBackend,
	modelId: selectedProject,
	pointCount: numPoints,
	timeIndices: pipelineDiagnostics?.timeIndices ?? [timeIndex]
};
```

- [ ] **Step 4: Run strict exposure-only E2E**

Run:

```powershell
cd viewer
$env:REQUIRE_WEBGPU_ON_DEMAND='1'
npm run test:e2e -- tests/e2e/webgpu-f32-on-demand-vertical-slice.spec.ts --project=chromium
Remove-Item Env:\REQUIRE_WEBGPU_ON_DEMAND
```

Expected: timing test passes.

## Milestone 3: Render The Selected Hour In The App Path

### Task 5: Add Explicit Selected-Hour Render Contract

**Files:**
- Modify: `viewer/src/lib/services/gpuUtciRenderBridge.ts`
- Modify: `viewer/src/lib/services/pointCloudService.ts`
- Test: `viewer/tests/services/pointCloudService.surface.test.ts`

- [ ] **Step 1: Write failing service test**

In `viewer/tests/services/pointCloudService.surface.test.ts`, add:

```ts
it('creates a gpuNative selected-hour surface without a DataTexture map', () => {
	const analysis = createTinyAnalysis();
	const mesh = createUtciSurfaceMesh({
		analysis,
		hourIndex: 0,
		monthIndex: 0,
		backend: 'gpuNative'
	});

	const material = mesh.material as THREE.Material & { map?: THREE.Texture | null };
	expect(mesh.userData.utciSurfaceBackend).toBe('gpuNative');
	expect(mesh.userData.utciSurfaceSource).toBe('cpu-uploaded-selected-hour');
	expect(material.map ?? null).toBeNull();

	disposeUtciSurfaceMesh(mesh);
});
```

- [ ] **Step 2: Run test to verify it fails or captures current behavior**

Run:

```powershell
cd viewer
npm test -- tests/services/pointCloudService.surface.test.ts
```

Expected: either fails because the test helper needs updating, or passes if the current `gpuNative` service already satisfies this contract. If it passes, keep it as a regression guard.

- [ ] **Step 3: Add selected-hour bridge metadata**

In `viewer/src/lib/services/gpuUtciRenderBridge.ts`, add to `GpuNativeUtciSurfaceState`:

```ts
source: 'cpu-uploaded-selected-hour' | 'compute-buffer-selected-hour';
```

When creating the current storage-backed mesh, set:

```ts
source: 'cpu-uploaded-selected-hour'
```

Expose:

```ts
export function getGpuNativeUtciSurfaceSource(mesh: THREE.Mesh): string | undefined {
	return (mesh.userData[GPU_NATIVE_SURFACE_STATE_KEY] as GpuNativeUtciSurfaceState | undefined)?.source;
}
```

- [ ] **Step 4: Preserve honest diagnostics**

In `viewer/src/lib/services/pointCloudService.ts`, when backend is `gpuNative`, set:

```ts
mesh.userData.utciSurfaceSource = getGpuNativeUtciSurfaceSource(mesh);
```

Do not label this as zero-copy unless a later task actually connects the compute output buffer to the renderer.

- [ ] **Step 5: Count selected-hour render transfers**

In `viewer/src/lib/services/pointCloudService.ts`, increment a diagnostic counter when `gpuNative` updates a selected-hour storage attribute from a CPU array:

```ts
mesh.userData.selectedHourTransferCount =
	((mesh.userData.selectedHourTransferCount as number | undefined) ?? 0) + 1;
```

Do not increment this counter for the `dataTexture` backend.

- [ ] **Step 6: Run service test**

Run:

```powershell
cd viewer
npm test -- tests/services/pointCloudService.surface.test.ts
```

Expected: test passes.

### Task 6: Add Main-Route On-Demand Flag And Fallback

**Files:**
- Modify: `viewer/src/routes/+page.svelte`
- Modify: `viewer/src/lib/components/scene/UTCIPointCloud.svelte`
- Test: `viewer/tests/e2e/webgpu-f32-on-demand-vertical-slice.spec.ts`

- [ ] **Step 1: Add failing E2E for feature flag and fallback**

Append to `viewer/tests/e2e/webgpu-f32-on-demand-vertical-slice.spec.ts`:

```ts
test('main route exposes f32 on-demand render diagnostics behind a flag and keeps data fallback', async ({ page }) => {
	await page.goto('/?utciRenderDiagnostics=1&utciOnDemand=f32&utciRender=gpu');

	const hasWebGpu = await page.evaluate(() => Boolean(navigator.gpu));
	test.skip(!hasWebGpu && process.env.REQUIRE_WEBGPU_ON_DEMAND !== '1', 'WebGPU unavailable.');

	await page.waitForFunction(() => {
		const diagnostics = (window as Window & { __utciRenderDiagnostics__?: any })
			.__utciRenderDiagnostics__;
		return diagnostics?.utciOnDemand === 'f32' && diagnostics?.utciRenderResolved === 'gpuNative';
	});

	let diagnostics = await page.evaluate(() => {
		return (window as Window & { __utciRenderDiagnostics__?: any }).__utciRenderDiagnostics__;
	});
	expect(diagnostics.utciSurfaceSource).toMatch(/selected-hour/);
	expect(diagnostics.dataTextureBuildCount ?? 0).toBe(0);
	expect(diagnostics.selectedHourTransferCount ?? 0).toBeGreaterThanOrEqual(0);

	await page.goto('/?utciRenderDiagnostics=1&utciOnDemand=f32&utciRender=data');
	await page.waitForFunction(() => {
		const diagnostics = (window as Window & { __utciRenderDiagnostics__?: any })
			.__utciRenderDiagnostics__;
		return diagnostics?.utciRenderResolved === 'dataTexture';
	});

	diagnostics = await page.evaluate(() => {
		return (window as Window & { __utciRenderDiagnostics__?: any }).__utciRenderDiagnostics__;
	});
	expect(diagnostics.utciRenderResolved).toBe('dataTexture');
});
```

- [ ] **Step 2: Run E2E to verify it fails**

Run:

```powershell
cd viewer
$env:REQUIRE_WEBGPU_ON_DEMAND='1'
npm run test:e2e -- tests/e2e/webgpu-f32-on-demand-vertical-slice.spec.ts --project=chromium
Remove-Item Env:\REQUIRE_WEBGPU_ON_DEMAND
```

Expected: fails because `utciOnDemand=f32` diagnostics are not implemented on the main route.

- [ ] **Step 3: Add main-route flag parsing**

In `viewer/src/routes/+page.svelte`, add:

```ts
type UtciOnDemandMode = 'off' | 'f32';

$: utciOnDemandMode = $page.url.searchParams.get('utciOnDemand') === 'f32' ? 'f32' : 'off';
```

Extend `MainRouteUtciRenderDiagnostics`:

```ts
utciOnDemand: UtciOnDemandMode;
utciSurfaceSource?: string;
selectedHourTransferCount?: number;
dataTextureBuildCount?: number;
```

Add these fields in `updateUtciRenderDiagnostics()`.

- [ ] **Step 4: Thread selected-hour backend metadata from `UTCIPointCloud`**

In `viewer/src/lib/components/scene/UTCIPointCloud.svelte`, add an exported callback:

```ts
export let onUtciSurfaceDiagnostics:
	| ((diagnostics: { utciSurfaceSource?: string }) => void)
	| undefined = undefined;
```

After creating/updating `utciSurface`, call:

```ts
onUtciSurfaceDiagnostics?.({
	utciSurfaceSource: utciSurface?.userData.utciSurfaceSource,
	selectedHourTransferCount: utciSurface?.userData.selectedHourTransferCount,
	dataTextureBuildCount: utciSurface?.userData.utciSurfaceBackend === 'dataTexture' ? 1 : 0
});
```

In `viewer/src/routes/+page.svelte`, store that metadata and include it in diagnostics.

- [ ] **Step 5: Run main-route E2E**

Run:

```powershell
cd viewer
$env:REQUIRE_WEBGPU_ON_DEMAND='1'
npm run test:e2e -- tests/e2e/webgpu-f32-on-demand-vertical-slice.spec.ts --project=chromium
Remove-Item Env:\REQUIRE_WEBGPU_ON_DEMAND
```

Expected: feature-flag/fallback test passes.

## Milestone 4: Multi-Hour Parity And Known-Point Evidence

### Task 7: Compare Multiple Hours Against A Separate Baseline

**Files:**
- Modify: `viewer/src/routes/debug-webgpu-utci/+page.svelte`
- Modify: `viewer/tests/e2e/webgpu-f32-on-demand-vertical-slice.spec.ts`

- [ ] **Step 1: Add failing multi-hour E2E**

Append:

```ts
test('strict f32 on-demand matches a separate baseline for boundary and solar-edge hours', async ({ page }) => {
	await page.goto('/debug-webgpu-utci?onDemandPrototype=1&strictExposureOnly=1&compareHours=12,23,16,17&baseline=separateRunAll');

	const hasWebGpu = await page.evaluate(() => Boolean(navigator.gpu));
	test.skip(!hasWebGpu && process.env.REQUIRE_WEBGPU_ON_DEMAND !== '1', 'WebGPU unavailable.');

	await expect(page.getByTestId('on-demand-prototype-status')).toContainText(/ready|error/i, {
		timeout: 180_000
	});

	const result = await page.evaluate(() => {
		return (window as Window & { __onDemandMultiHourComparison__?: any })
			.__onDemandMultiHourComparison__;
	});

	expect(result).toBeTruthy();
	expect(result.strictPath.usedRunAllForSelectedHour).toBe(false);
	expect(result.baselineSource).toBe('separateRunAll');
	expect(result.hours).toEqual([12, 23, 16, 17]);
	for (const hour of result.hourResults) {
		expect(hour.numCompared).toBeGreaterThan(0);
		expect(hour.maxAbsDiff).toBeLessThanOrEqual(1e-5);
	}

	if (result.knownPoint31079) {
		expect(result.knownPoint31079.pointIndex).toBe(31079);
		expect(result.knownPoint31079.hours.map((entry: any) => entry.hour)).toEqual([16, 17]);
	}
});
```

- [ ] **Step 2: Run E2E to verify it fails**

Run:

```powershell
cd viewer
$env:REQUIRE_WEBGPU_ON_DEMAND='1'
npm run test:e2e -- tests/e2e/webgpu-f32-on-demand-vertical-slice.spec.ts --project=chromium
Remove-Item Env:\REQUIRE_WEBGPU_ON_DEMAND
```

Expected: fails because `compareHours` is not implemented.

- [ ] **Step 3: Implement strict-vs-baseline comparison without contaminating strict diagnostics**

In `viewer/src/routes/debug-webgpu-utci/+page.svelte`, parse:

```ts
function parseCompareHours(value: string | null): number[] {
	if (!value) return [];
	return value
		.split(',')
		.map((entry) => Number(entry.trim()))
		.filter((entry) => Number.isInteger(entry) && entry >= 0);
}
```

Create a separate baseline manager/pipeline that may run the current all-hours path. Do not reuse the strict exposure-only `computeManager` for baseline slices, because `readUtcisSlice()` depends on the all-hours `utciBuffer`.

```ts
const strictManager = strictSetup.computeManager;
const baselineSetup = await prepareWebgpuDebugInputsForCurrentSelection();
const baselineManager = baselineSetup.computeManager;
await baselineManager.runAll({ numPoints, numHours, numMonths });
```

For each hour, compute on-demand output on `strictManager` and baseline slices on `baselineManager`:

```ts
const hourResults = [];
for (const hour of compareHours) {
	await strictManager.runUtciForTimeIndex({
		timeIndex: hour,
		numPoints,
		numHours,
		numMonths,
		format: 'f32-utci'
	});
	const onDemand = await strictManager.getPipeline().readOnDemandUtciForDebug?.({ numPoints });
	if (!onDemand) {
		throw new Error('Strict on-demand debug readback is unavailable for comparison.');
	}
	const baseline = await baselineManager.readUtcisSlice({
		monthIndex: 0,
		hourIndex: hour,
		numPoints,
		numHours,
		numMonths
	});
	hourResults.push(compareFloatArrays(hour, onDemand, baseline));
}
```

Add the comparison helper in the debug route:

```ts
function compareFloatArrays(hour: number, onDemand: Float32Array, baseline: Float32Array) {
	let maxAbsDiff = 0;
	let sumSq = 0;
	const numCompared = Math.min(onDemand.length, baseline.length);
	for (let index = 0; index < numCompared; index += 1) {
		const diff = onDemand[index] - baseline[index];
		const abs = Math.abs(diff);
		if (abs > maxAbsDiff) maxAbsDiff = abs;
		sumSq += diff * diff;
	}
	const pointIndex = 31079;
	const hasKnownPoint = pointIndex < numCompared;
	return {
		hour,
		numCompared,
		maxAbsDiff,
		rmse: Math.sqrt(sumSq / Math.max(1, numCompared)),
		onDemandAt31079: hasKnownPoint ? onDemand[pointIndex] : undefined,
		baselineAt31079: hasKnownPoint ? baseline[pointIndex] : undefined,
		diffAt31079: hasKnownPoint ? onDemand[pointIndex] - baseline[pointIndex] : undefined
	};
}
```

Record known point values when `numPoints > 31079`:

```ts
const knownPoint31079 =
	numPoints > 31079
		? {
				pointIndex: 31079,
				hours: hourResults
					.filter((entry) => entry.hour === 16 || entry.hour === 17)
					.map((entry) => ({
						hour: entry.hour,
						onDemand: entry.onDemandAt31079,
						baseline: entry.baselineAt31079,
						diff: entry.diffAt31079
					}))
			}
		: undefined;
```

Set:

```ts
window.__onDemandMultiHourComparison__ = {
	baselineSource: 'separateRunAll',
	strictPath: strictManager.getOnDemandDiagnostics(),
	hours: compareHours,
	hourResults,
	knownPoint31079
};
```

- [ ] **Step 4: Run multi-hour E2E**

Run:

```powershell
cd viewer
$env:REQUIRE_WEBGPU_ON_DEMAND='1'
npm run test:e2e -- tests/e2e/webgpu-f32-on-demand-vertical-slice.spec.ts --project=chromium
Remove-Item Env:\REQUIRE_WEBGPU_ON_DEMAND
```

Expected: multi-hour comparison passes and reports exact point `31079` values when applicable.

## Milestone 5: Record Results And Keep The Decision Narrow

### Task 8: Update Prototype Results With Vertical-Slice Evidence

**Files:**
- Modify: `docs/superpowers/plans/2026-05-07-webgpu-compute-on-demand-prototype-results.md`
- Modify: `docs/webgpu_strategy_analysis.md`

- [ ] **Step 1: Append vertical-slice result section from actual diagnostics**

After strict E2E passes, read `window.__onDemandPrototypeDiagnostics__` and `window.__onDemandMultiHourComparison__` from the passing browser run and append a completed section to `docs/superpowers/plans/2026-05-07-webgpu-compute-on-demand-prototype-results.md`.

Use this helper in the debug route or a temporary local script to build the section from actual captured objects. It has no blank fields; `valueOrUnavailable()` converts missing runtime fields to `not exposed by runtime`.

```ts
function valueOrUnavailable(value: unknown): string {
	return value === undefined || value === null || value === ''
		? 'not exposed by runtime'
		: String(value);
}

function passFail(value: boolean): 'pass' | 'fail' {
	return value ? 'pass' : 'fail';
}

function buildVerticalSliceResultsSection(params: {
	browserName: string;
	diagnostics: any;
	comparison: any;
	fallbackDiagnostics: any;
}): string {
	const { browserName, diagnostics, comparison, fallbackDiagnostics } = params;
	const maxDiffs = comparison.hourResults
		.map((entry: any) => `hour ${entry.hour}: ${entry.maxAbsDiff}`)
		.join('; ');
	const known16 = comparison.knownPoint31079?.hours.find((entry: any) => entry.hour === 16);
	const known17 = comparison.knownPoint31079?.hours.find((entry: any) => entry.hour === 17);
	const noAllHours =
		diagnostics.allHoursUtciBytesAllocated === 0 && diagnostics.allHoursMrtBytesAllocated === 0;
	const multiHourPass = comparison.hourResults.every((entry: any) => entry.maxAbsDiff <= 1e-5);

	return `## 2026-05-08 F32 Vertical Slice Follow-Up

### Environment

- Browser: ${browserName}
- WebGPU available: ${valueOrUnavailable(diagnostics.navigatorGpu)}
- GPU adapter: ${valueOrUnavailable(diagnostics.adapterInfo)}
- Renderer/backend: ${valueOrUnavailable(diagnostics.rendererBackend)}
- \`maxStorageBufferBindingSize\`: ${valueOrUnavailable(diagnostics.maxStorageBufferBindingSize)}
- \`maxBufferSize\`: ${valueOrUnavailable(diagnostics.maxBufferSize)}
- \`maxStorageBuffersPerShaderStage\`: ${valueOrUnavailable(diagnostics.maxStorageBuffersPerShaderStage)}
- Model/scenario: ${valueOrUnavailable(diagnostics.modelId)} / ${valueOrUnavailable(diagnostics.scenarioId)}
- Grid resolution: ${valueOrUnavailable(diagnostics.gridResolution)}
- Point count: ${valueOrUnavailable(diagnostics.pointCount)}
- Compared hours: ${comparison.hours.join(', ')}
- Render transport: ${valueOrUnavailable(diagnostics.renderTransport)}
- Selected-hour transfers: ${valueOrUnavailable(diagnostics.selectedHourTransferCount)}

### Gate Results

| Gate | Result | Evidence |
| --- | --- | --- |
| Diagnostics captured | ${passFail(Boolean(diagnostics.navigatorGpu && diagnostics.rendererBackend))} | rendererBackend=${valueOrUnavailable(diagnostics.rendererBackend)}, pointCount=${valueOrUnavailable(diagnostics.pointCount)} |
| Exposure-only runtime path | ${passFail(Boolean(diagnostics.usedExposureOnlyPrecompute && !diagnostics.usedRunAllForSelectedHour && diagnostics.liveAnalysisConstructedForSelectedHour === false))} | usedExposureOnlyPrecompute=${diagnostics.usedExposureOnlyPrecompute}, usedRunAllForSelectedHour=${diagnostics.usedRunAllForSelectedHour}, liveAnalysisConstructedForSelectedHour=${diagnostics.liveAnalysisConstructedForSelectedHour} |
| F32 selected-hour render | ${passFail(diagnostics.renderTransport === 'compute-buffer-selected-hour' || diagnostics.renderTransport === 'cpu-uploaded-selected-hour')} | renderTransport=${valueOrUnavailable(diagnostics.renderTransport)} |
| No all-hours readback | ${passFail(noAllHours)} | allHoursUtciBytesAllocated=${diagnostics.allHoursUtciBytesAllocated}, allHoursMrtBytesAllocated=${diagnostics.allHoursMrtBytesAllocated} |
| No hot-path \`DataTexture\` rebuild | ${passFail((diagnostics.dataTextureBuildCount ?? 0) === 0)} | dataTextureBuildCount=${diagnostics.dataTextureBuildCount ?? 0} |
| Multi-hour parity | ${passFail(multiHourPass)} | ${maxDiffs} |
| Fallback to data path | ${passFail(fallbackDiagnostics?.utciRenderResolved === 'dataTexture')} | utciRenderResolved=${valueOrUnavailable(fallbackDiagnostics?.utciRenderResolved)} |

### Timings

| Phase | ms |
| --- | ---: |
| Exposure-only precompute | ${valueOrUnavailable(diagnostics.timings?.exposurePrecomputeMs)} |
| One-hour f32 dispatch | ${valueOrUnavailable(diagnostics.timings?.oneHourDispatchMs)} |
| Render update | ${valueOrUnavailable(diagnostics.timings?.renderUpdateMs)} |
| Debug readback only | ${valueOrUnavailable(diagnostics.timings?.debugReadbackMs)} |

### Decision

- [x] Keep \`f32\` on-demand behind a debug flag only.
- [ ] Allow user verification of \`f32\` on-demand in the main app behind \`utciOnDemand=f32\`.
- [ ] Continue to a later packed-output plan after user verification.

### Notes

- Point \`31079\` hour \`16\`: on-demand ${valueOrUnavailable(known16?.onDemand)}, baseline ${valueOrUnavailable(known16?.baseline)}, diff ${valueOrUnavailable(known16?.diff)}.
- Point \`31079\` hour \`17\`: on-demand ${valueOrUnavailable(known17?.onDemand)}, baseline ${valueOrUnavailable(known17?.baseline)}, diff ${valueOrUnavailable(known17?.diff)}.
`;
}
```

Before saving this section, search the generated markdown for empty table cells (`|  |`) and empty bullet values (`: ` at line end). If either exists, do not save the section; fix the captured diagnostic source first.

- [ ] **Step 2: Link the follow-up plan from strategy doc**

Add to `docs/webgpu_strategy_analysis.md` near the existing prototype links:

```md
F32 vertical-slice follow-up plan: `docs/superpowers/plans/2026-05-08-webgpu-f32-on-demand-vertical-slice.md`.
```

- [ ] **Step 3: Run focused verification**

Run:

```powershell
cd viewer
npm test -- tests/compute/onDemandDiagnostics.test.ts tests/compute/onDemandSizing.test.ts tests/compute/onDemandOutputFormat.test.ts tests/compute/compute-manager-on-demand.test.ts tests/compute/webgpu-on-demand-source-locks.test.ts tests/compute/gpu-pipeline.test.ts tests/services/pointCloudService.surface.test.ts
```

Expected: focused Vitest files pass.

Run strict browser verification:

```powershell
cd viewer
$env:REQUIRE_WEBGPU_ON_DEMAND='1'
npm run test:e2e -- tests/e2e/webgpu-on-demand-prototype.spec.ts --project=chromium
npm run test:e2e -- tests/e2e/webgpu-f32-on-demand-vertical-slice.spec.ts --project=chromium
Remove-Item Env:\REQUIRE_WEBGPU_ON_DEMAND
```

Expected: both E2E files pass on a WebGPU-capable browser. If they cannot run, do not mark any runtime gate as proven.

## Self-Review Checklist

- [ ] No git worktree steps.
- [ ] No commit steps.
- [ ] Packed output remains out of scope.
- [ ] Default production path remains available.
- [ ] The plan distinguishes Three storage rendering from compute-output zero-copy.
- [ ] Exposure-only evidence is runtime-based, not source-lock only.
- [ ] Debug readback is validation-only and explicitly counted.
- [ ] DataTexture fallback remains available.
- [ ] Multi-hour checks include hour `23` and hours `16`/`17`.
- [ ] Known point `31079` is reported when present.
- [ ] Results doc cannot be completed with blank or aspirational evidence.

## Execution Options

Plan complete and saved to `docs/superpowers/plans/2026-05-08-webgpu-f32-on-demand-vertical-slice.md`. Two execution options:

1. **Subagent-Driven (recommended)** - Dispatch a fresh subagent per milestone or task, review between tasks, fast iteration.
2. **Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints.
