# WebGPU On-Demand GPU-Resident Debug Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the debug WebGPU side fully on-demand for month/hour scrubbing, keep UTCI parity against Python `.bin` where available and WebGPU `runAll()` elsewhere, capture enough timing to understand responsiveness, and replace the transitional CPU-uploaded selected-hour bridge with a GPU-resident render path.

**Architecture:** Keep the current `f32` on-demand compute path as the correctness base: precompute exposure once, compute only the selected month/hour with `runUtciForTimeIndex()`, and preserve `.bin`, `runAll()`, `readUtciBulk()`, and `dataTexture` fallback paths. Add month scrub verification and WebGPU `runAll()` comparisons before changing rendering. Then add a GPU-resident render bridge that shades from GPU-owned selected-hour UTCI data and reports `renderTransport='compute-buffer-selected-hour'` only after the render path no longer depends on selected-hour CPU readback/color upload.

**Tech Stack:** SvelteKit, Playwright, Vitest, WebGPU, Three r175 WebGPURenderer, Three TSL, WGSL, existing `debug-webgpu-utci` diagnostics.

**Workflow Constraints:** No commits. No git worktrees. The worktree may be dirty; do not touch unrelated files such as `data/batch-parity-results/parity_performance_report.md` unless the active task explicitly requires it.

---

## Current Evidence And Boundaries

- Current debug on-demand route is behind `?onDemandPrototype=1&utciOnDemand=f32&utciRender=gpu`.
- Current selected-hour path is `exposure-only-f32` and avoids `runAll()` for selected-hour scrubbing.
- Current render transport is still `cpu-uploaded-selected-hour`; this is transitional and should not be called zero-copy.
- Current month handling now has direct debug-route E2E proof for `monthIndex=7&timeIndex=12`: the on-demand WebGPU side completes `selectedTimeIndex = 7 * 24 + 12`, while the Python `.bin` comparison remains hour-local.
- Python `.bin` parity is only available for the August representative dataset. For other months, compare on-demand results to a separate WebGPU `runAll()` baseline.
- Existing batch parity timing files such as `data/batch-parity-results/Ben-Gurion_20250815_grid_2m_fullday_timing.json` are old/full `runAll()` timing baselines, not on-demand timing captures.
- The known hour 16/17 ray-flip issue is allowed as existing WebGPU alignment debt. Do not broaden this plan to fix it.

## Files And Responsibilities

- `viewer/src/routes/debug-webgpu-utci/+page.svelte`
  - Owns debug-route query flags, month/hour selection, on-demand scheduling, diagnostics publishing, Python `.bin` comparison, separate `runAll()` baseline comparison, and E2E-facing window objects.
- `viewer/src/lib/compute/onDemandDiagnostics.ts`
  - Owns typed runtime diagnostics and timing fields. Extend only with fields that will be asserted by tests or displayed in captured results.
- `viewer/src/lib/compute/webgpuUtciPipeline.ts`
  - Owns WebGPU exposure precompute, one-time-step UTCI compute, debug readback, allocation counters, and future GPU-resident output handle lifecycle.
- `viewer/src/lib/components/scene/Scene.svelte`
  - Owns Three `WebGPURenderer` creation and renderer backend diagnostics. This is the only place that can reliably expose the renderer-owned `GPUDevice`.
- `viewer/src/lib/components/scene/UTCIPointCloud.svelte`
  - Owns scene-level UTCI surface creation/update through `useThrelte()`. GPU-resident surface wiring belongs here or in a sibling scene component, not directly in the route.
- `viewer/src/lib/services/gpuUtciRenderBridge.ts`
  - Owns the Three WebGPU/TSL surface implementation. Replace CPU-uploaded color storage with a GPU-resident selected-hour render path after feasibility is proven.
- `viewer/src/lib/services/pointCloudService.ts`
  - Owns surface backend selection and mesh update metadata. Preserve `dataTexture`; keep `gpuNative` honest about whether it is CPU-uploaded or compute-buffer-backed.
- `viewer/tests/e2e/webgpu-f32-on-demand-vertical-slice.spec.ts`
  - Add month scrub, WebGPU `runAll()` comparison, timing capture, and GPU-resident render assertions.
- `viewer/tests/compute/onDemandDiagnostics.test.ts`
  - Add coverage for new timing and render-transport diagnostics.
- `viewer/tests/compute/webgpu-on-demand-source-locks.test.ts`
  - Add guardrails against accidental CPU-readback/color-upload claims on the GPU-resident path.
- `viewer/tests/services/pointCloudService.surface.test.ts`
  - Update backend metadata tests for `compute-buffer-selected-hour`.
- `docs/superpowers/plans/2026-05-07-webgpu-compute-on-demand-prototype-results.md`
  - Append final captured evidence after implementation passes; do not rewrite older evidence.

---

## Milestone 1: Month Scrubber Proof On The Current Bridge

### Task 1: Add A Month Scrub E2E For Debug On-Demand

**Files:**
- Modify: `viewer/tests/e2e/webgpu-f32-on-demand-vertical-slice.spec.ts`
- Verify: `viewer/src/routes/debug-webgpu-utci/+page.svelte`

**Status:** Completed 2026-05-08.

**Implementation note:** The final committed regression is intentionally narrower than the original scrub-storm sketch below. It asserts that `monthIndex=7&timeIndex=12` completes as full-year WebGPU time index `7 * 24 + 12`, confirms Python `.bin` comparison stays hour-local at `12`, and navigates to `about:blank` during teardown to avoid the local WebGPU/Playwright hang seen with the broader rapid-scrub test.

**Verification:** `REQUIRE_WEBGPU_ON_DEMAND=1 npm run test:e2e -- tests/e2e/webgpu-f32-on-demand-vertical-slice.spec.ts --project=chromium --grep "honors the selected month" --workers=1` completed successfully with the vertical-slice spec reporting `11 passed (58.2s)`. The npm wrapper parsed the grep text as positional args in this run, so Playwright ran the full spec file rather than only the single test; this still verifies the new regression plus neighboring f32 on-demand coverage.

- [x] **Step 1: Write the failing month scrub test**

Append this test near the existing stale scrub test:

```ts
test('debug on-demand discards stale month scrub results and ends on the final selected month/hour', async ({
	page
}) => {
	test.setTimeout(180_000);
	await page.goto(
		'/debug-webgpu-utci?onDemandPrototype=1&utciOnDemand=f32&utciRender=gpu&timeIndex=12'
	);

	const hasWebGpu = await page.evaluate(() => Boolean(navigator.gpu));
	test.skip(!hasWebGpu && process.env.REQUIRE_WEBGPU_ON_DEMAND !== '1', 'WebGPU unavailable.');

	await page.waitForFunction(() => {
		const diagnostics = (window as Window & { __onDemandPrototypeDiagnostics__?: any })
			.__onDemandPrototypeDiagnostics__;
		return diagnostics?.path === 'exposure-only-f32' && diagnostics?.completedTimeIndex === 12;
	}, undefined, { timeout: 180_000 });

	await page.evaluate(async () => {
		const pushScrubSelection = async (monthIndex: number, hourIndex: number) => {
			await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
			const url = new URL(window.location.href);
			url.searchParams.set('monthIndex', String(monthIndex));
			url.searchParams.set('timeIndex', String(hourIndex));
			url.searchParams.set('forceOnDemandOverlapMs', '25');
			window.history.pushState({}, '', url);
			window.dispatchEvent(new PopStateEvent('popstate'));
		};

		await pushScrubSelection(1, 9);
		await pushScrubSelection(4, 15);
		await pushScrubSelection(10, 18);
	});

	await page.waitForFunction(() => {
		const diagnostics = (window as Window & { __onDemandPrototypeDiagnostics__?: any })
			.__onDemandPrototypeDiagnostics__;
		return (
			diagnostics?.selectedMonthIndex === 10 &&
			diagnostics?.completedMonthIndex === 10 &&
			diagnostics?.selectedTimeIndex === 10 * 24 + 18 &&
			diagnostics?.completedTimeIndex === 10 * 24 + 18 &&
			diagnostics?.inFlightCount === 0 &&
			diagnostics?.pendingReadbackRequestId == null
		);
	}, undefined, { timeout: 180_000 });

	const diagnostics = await page.evaluate(() => {
		return (window as Window & { __onDemandPrototypeDiagnostics__?: any })
			.__onDemandPrototypeDiagnostics__;
	});

	expect(diagnostics.usedRunAllForSelectedHour).toBe(false);
	expect(diagnostics.usedExposureOnlyPrecompute).toBe(true);
	expect(diagnostics.allHoursUtciBytesAllocated).toBe(0);
	expect(diagnostics.allHoursMrtBytesAllocated).toBe(0);
	expect(diagnostics.trackedGpuAllocationBytes.allHoursOutputBytes).toBe(0);
	expect(diagnostics.dataTextureBuildCount).toBe(0);
	expect(diagnostics.scrubSampleCount).toBeGreaterThanOrEqual(3);
	expect(diagnostics.staleResultDiscardCount).toBeGreaterThan(0);
});
```

- [x] **Step 2: Run the test to verify current behavior**

Run:

```powershell
cd viewer
$env:REQUIRE_WEBGPU_ON_DEMAND='1'
npm run test:e2e -- tests/e2e/webgpu-f32-on-demand-vertical-slice.spec.ts --project=chromium --grep "month scrub"
```

Expected before fixes: either PASS if month query/state is already fully wired, or FAIL because the route does not yet parse `monthIndex` from the query / does not recompute on month changes.

- [x] **Step 3: If the test fails, wire `monthIndex` query input into the debug on-demand selection**

In `viewer/src/routes/debug-webgpu-utci/+page.svelte`, add a helper next to `getStrictExposureOnlyTimeIndex()`:

```ts
function getDebugQueryMonthIndex(defaultMonthIndex: number): number {
	const raw = Number($page.url.searchParams.get("monthIndex") ?? String(defaultMonthIndex));
	if (!Number.isInteger(raw)) return defaultMonthIndex;
	return Math.min(Math.max(raw, 0), 11);
}
```

Then ensure the debug on-demand selection uses the query month when present:

```ts
$: debugOnDemandMonthIndex = getDebugQueryMonthIndex(comparisonSelection.monthIndex);
$: debugOnDemandSelection = getDebugOnDemandSelection({
	monthIndex: debugOnDemandMonthIndex,
	hourIndex: getStrictExposureOnlyTimeIndex(),
	parityMode,
});
```

Keep existing store-driven UI behavior intact. The query override exists to make month scrub E2E deterministic.

- [x] **Step 4: Run the month scrub E2E again**

Run:

```powershell
cd viewer
$env:REQUIRE_WEBGPU_ON_DEMAND='1'
npm run test:e2e -- tests/e2e/webgpu-f32-on-demand-vertical-slice.spec.ts --project=chromium --grep "month scrub"
```

Expected: PASS, with final diagnostics on month `10` and hour `18`.

### Task 2: Compare Month/Hour On-Demand Against A Separate WebGPU `runAll()` Baseline

**Files:**
- Modify: `viewer/src/routes/debug-webgpu-utci/+page.svelte`
- Modify: `viewer/tests/e2e/webgpu-f32-on-demand-vertical-slice.spec.ts`

- [ ] **Step 1: Write the failing multi-month comparison E2E**

Append:

```ts
test('strict exposure-only month/hour outputs match a separate WebGPU runAll baseline', async ({
	page
}) => {
	test.setTimeout(240_000);
	await page.goto(
		'/debug-webgpu-utci?onDemandPrototype=1&strictExposureOnly=1&compareMonthHours=0:12,3:15,7:23,10:18&baseline=separateRunAll'
	);

	const hasWebGpu = await page.evaluate(() => Boolean(navigator.gpu));
	test.skip(!hasWebGpu && process.env.REQUIRE_WEBGPU_ON_DEMAND !== '1', 'WebGPU unavailable.');

	await page.waitForFunction(() => {
		const result = (window as Window & { __onDemandMonthHourComparison__?: any })
			.__onDemandMonthHourComparison__;
		return result?.status === 'complete' || result?.status === 'error';
	}, undefined, { timeout: 240_000 });

	const result = await page.evaluate(() => {
		return (window as Window & { __onDemandMonthHourComparison__?: any })
			.__onDemandMonthHourComparison__;
	});

	expect(result?.status).toBe('complete');
	expect(result?.baselineSource).toBe('separateRunAll');
	expect(result?.pairs).toHaveLength(4);
	for (const pair of result.pairs) {
		expect(pair.numCompared).toBeGreaterThan(1000);
		expect(pair.maxAbsDiff).toBeLessThanOrEqual(1e-5);
		expect(pair.rmse).toBeLessThanOrEqual(1e-6);
	}
});
```

- [ ] **Step 2: Add parsing for `compareMonthHours`**

In `viewer/src/routes/debug-webgpu-utci/+page.svelte`, add:

```ts
function getCompareMonthHourPairs(): Array<{ monthIndex: number; hourIndex: number; timeIndex: number }> {
	const raw = $page.url.searchParams.get("compareMonthHours");
	if (!raw) return [];

	return raw
		.split(",")
		.map((entry) => entry.trim())
		.filter(Boolean)
		.map((entry) => {
			const [monthRaw, hourRaw] = entry.split(":");
			const monthIndex = Number(monthRaw);
			const hourIndex = Number(hourRaw);
			if (
				!Number.isInteger(monthIndex) ||
				!Number.isInteger(hourIndex) ||
				monthIndex < 0 ||
				monthIndex > 11 ||
				hourIndex < 0 ||
				hourIndex > 23
			) {
				throw new Error(`Invalid compareMonthHours entry "${entry}". Expected month:hour.`);
			}
			return { monthIndex, hourIndex, timeIndex: monthIndex * 24 + hourIndex };
		});
}
```

- [ ] **Step 3: Add the window result type**

Extend the local `Window` declaration with:

```ts
__onDemandMonthHourComparison__?: {
	status: "idle" | "running" | "complete" | "error";
	baselineSource: "separateRunAll";
	pairs: Array<{
		monthIndex: number;
		hourIndex: number;
		timeIndex: number;
		numCompared: number;
		maxAbsDiff: number;
		rmse: number;
		onDemandAt31079?: number;
		baselineAt31079?: number;
		diffAt31079?: number;
	}>;
	error?: string;
};
```

- [ ] **Step 4: Implement the strict month/hour comparison runner**

Add a function beside the existing strict `compareHours` runner:

```ts
async function runStrictMonthHourComparison(): Promise<void> {
	if (!browser || !onDemandPrototypeEnabled || !strictExposureOnlyEnabled) return;
	const pairs = getCompareMonthHourPairs();
	if (pairs.length === 0) return;

	const win = getParityWindow();
	win.__onDemandMonthHourComparison__ = {
		status: "running",
		baselineSource: "separateRunAll",
		pairs: [],
	};

	try {
		const base = $analysisStore;
		if (!base) throw new Error("Analysis store is not loaded.");

		const prepared = await ensureStrictExposureOnlyPrepared(base);
		const baselineManager = await createSeparateRunAllBaselineManager(base);
		const results = [];

		for (const pair of pairs) {
			await prepared.computeManager.runUtciForTimeIndex({
				timeIndex: pair.timeIndex,
				numPoints: prepared.numPoints,
				numHours: prepared.numHours,
				numMonths: prepared.numMonths,
				format: "f32-utci",
			});
			const onDemandUtci = await prepared.pipeline.readOnDemandUtciForDebug?.({
				numPoints: prepared.numPoints,
			});
			if (!onDemandUtci) {
				throw new Error("Strict on-demand debug readback API is unavailable.");
			}

			const baselineUtci = await baselineManager.getUtcisForMonthHour({
				monthIndex: pair.monthIndex,
				hourIndex: pair.hourIndex,
				numPoints: prepared.numPoints,
				numHours: prepared.numHours,
				numMonths: prepared.numMonths,
			});

			results.push({
				monthIndex: pair.monthIndex,
				hourIndex: pair.hourIndex,
				timeIndex: pair.timeIndex,
				...compareFloatArrays(pair.timeIndex, onDemandUtci, baselineUtci),
			});
		}

		win.__onDemandMonthHourComparison__ = {
			status: "complete",
			baselineSource: "separateRunAll",
			pairs: results,
		};
	} catch (error) {
		win.__onDemandMonthHourComparison__ = {
			status: "error",
			baselineSource: "separateRunAll",
			pairs: [],
			error: error instanceof Error ? error.message : String(error),
		};
	}
}
```

If helper names such as `ensureStrictExposureOnlyPrepared` or `createSeparateRunAllBaselineManager` do not already exist, extract the equivalent code from the existing strict `compareHours` runner without changing behavior.

- [ ] **Step 5: Trigger the runner in the existing reactive section**

Add:

```ts
$: if (browser && onDemandPrototypeEnabled && strictExposureOnlyEnabled) {
	void runStrictMonthHourComparison();
}
```

Guard it with the same run-token pattern used by the existing comparison runner so navigation cannot publish stale results.

- [ ] **Step 6: Run focused E2E**

Run:

```powershell
cd viewer
$env:REQUIRE_WEBGPU_ON_DEMAND='1'
npm run test:e2e -- tests/e2e/webgpu-f32-on-demand-vertical-slice.spec.ts --project=chromium --grep "month/hour outputs"
```

Expected: PASS with `maxAbsDiff <= 1e-5` for each month/hour pair.

---

## Milestone 2: Minimal Timing Capture

### Task 3: Extend Timing Diagnostics Without Turning This Into A Profiling Project

**Files:**
- Modify: `viewer/src/lib/compute/onDemandDiagnostics.ts`
- Modify: `viewer/tests/compute/onDemandDiagnostics.test.ts`
- Modify: `viewer/src/routes/debug-webgpu-utci/+page.svelte`

- [ ] **Step 1: Add timing fields**

Extend `OnDemandTimings`:

```ts
export interface OnDemandTimings {
	exposurePrecomputeMs?: number;
	oneHourDispatchMs?: number;
	renderUpdateMs?: number;
	debugReadbackMs?: number;
	selectedHourReadbackMs?: number;
	selectedHourAnalysisBuildMs?: number;
	cpuColorBuildMs?: number;
	gpuSurfaceUpdateMs?: number;
}
```

- [ ] **Step 2: Add a unit test for preserving timing fields**

In `viewer/tests/compute/onDemandDiagnostics.test.ts`, add:

```ts
it('records selected-hour timing attribution fields without clearing existing timings', () => {
	const diagnostics = createEmptyOnDemandDiagnostics();
	const withReadback = recordOnDemandTiming(diagnostics, 'selectedHourReadbackMs', 11.5);
	const withSurface = recordOnDemandTiming(withReadback, 'gpuSurfaceUpdateMs', 7.25);

	expect(withSurface.timings.selectedHourReadbackMs).toBe(11.5);
	expect(withSurface.timings.gpuSurfaceUpdateMs).toBe(7.25);
});
```

- [ ] **Step 3: Capture selected-hour readback timing**

In `runDebugOnDemandSelectedHour()`, wrap `readOnDemandUtciForDebug`:

```ts
const selectedHourReadbackStart = performance.now();
const selectedHourUtci = params.readbackForComparison
	? await prepared.pipeline.readOnDemandUtciForDebug?.({
			numPoints: prepared.numPoints,
		})
	: undefined;
const selectedHourReadbackMs = params.readbackForComparison
	? performance.now() - selectedHourReadbackStart
	: undefined;
```

Include the value in `updateOnDemandPrototypeDiagnostics()`:

```ts
timings: {
	...pipelineDiagnostics?.timings,
	selectedHourReadbackMs,
},
```

- [ ] **Step 4: Capture selected-hour analysis build timing**

Around `buildSelectedHourLiveAnalysis()`:

```ts
const selectedHourAnalysisBuildStart = performance.now();
const selectedHourAnalysis = buildSelectedHourLiveAnalysis({
	base: prepared.base,
	utciValues: selectedHourUtci,
	monthIndex: params.monthIndex,
	timeIndex: params.timeIndex,
});
const selectedHourAnalysisBuildMs = performance.now() - selectedHourAnalysisBuildStart;
```

Merge `selectedHourAnalysisBuildMs` into diagnostics.

- [ ] **Step 5: Capture surface update timing from existing render diagnostics**

In the debug route handler that reads `selectedHourTransferCount` / `dataTextureBuildCount` from the rendered mesh, measure the elapsed time already represented by `pendingRenderUpdate.startedAt`. Keep the existing `renderUpdateMs` and add:

```ts
gpuSurfaceUpdateMs: performance.now() - pendingRenderUpdate.startedAt,
```

Do not over-interpret this as pure GPU time. It is an app-visible surface update timing.

- [ ] **Step 6: Run timing diagnostics tests**

Run:

```powershell
cd viewer
npm test -- tests/compute/onDemandDiagnostics.test.ts
```

Expected: PASS.

### Task 4: Capture Timing Output In E2E

**Files:**
- Modify: `viewer/tests/e2e/webgpu-f32-on-demand-vertical-slice.spec.ts`
- Modify: `docs/superpowers/plans/2026-05-07-webgpu-compute-on-demand-prototype-results.md`

- [ ] **Step 1: Add timing assertions to the existing debug on-demand test**

In the "debug route can use f32 on-demand" test, add:

```ts
expect(diagnostics.timings.oneHourDispatchMs).toBeGreaterThanOrEqual(0);
expect(diagnostics.timings.selectedHourReadbackMs).toBeGreaterThanOrEqual(0);
expect(diagnostics.timings.selectedHourAnalysisBuildMs).toBeGreaterThanOrEqual(0);
```

In the stale scrub test, add:

```ts
expect(diagnostics.timings.renderUpdateMs).toBeGreaterThan(0);
expect(diagnostics.timings.gpuSurfaceUpdateMs).toBeGreaterThan(0);
```

- [ ] **Step 2: Run focused E2E timing tests**

Run:

```powershell
cd viewer
$env:REQUIRE_WEBGPU_ON_DEMAND='1'
npm run test:e2e -- tests/e2e/webgpu-f32-on-demand-vertical-slice.spec.ts --project=chromium --grep "debug route can use|stale scrub"
```

Expected: PASS and diagnostics contain the new timing fields.

- [ ] **Step 3: Record timing numbers after full verification**

After all implementation tasks pass, append a short section to `docs/superpowers/plans/2026-05-07-webgpu-compute-on-demand-prototype-results.md`:

```md
## YYYY-MM-DD GPU-Resident Debug Follow-Up Capture

Route:
`/debug-webgpu-utci?onDemandPrototype=1&utciOnDemand=f32&utciRender=gpu`

| Field | Value |
| --- | --- |
| `timings.exposurePrecomputeMs` | copy from `diagnostics.timings.exposurePrecomputeMs` |
| `timings.oneHourDispatchMs` | copy from `diagnostics.timings.oneHourDispatchMs` |
| `timings.selectedHourReadbackMs` | copy from `diagnostics.timings.selectedHourReadbackMs` |
| `timings.selectedHourAnalysisBuildMs` | copy from `diagnostics.timings.selectedHourAnalysisBuildMs` |
| `timings.gpuSurfaceUpdateMs` | copy from `diagnostics.timings.gpuSurfaceUpdateMs` |
| `timings.renderUpdateMs` | copy from `diagnostics.timings.renderUpdateMs` |
| `renderTransport` | copy from `diagnostics.renderTransport` |
| `selectedMonthIndex/completedMonthIndex` | copy as `${diagnostics.selectedMonthIndex}/${diagnostics.completedMonthIndex}` |
| `selectedTimeIndex/completedTimeIndex` | copy as `${diagnostics.selectedTimeIndex}/${diagnostics.completedTimeIndex}` |
| `trackedGpuAllocationBytes` | paste compact JSON from `diagnostics.trackedGpuAllocationBytes` |
```

Use actual values captured from the passing E2E/browser run.

---

## Milestone 3: GPU-Resident Render Bridge Feasibility

### Task 5: Add Hard Same-Device And Render-Storage Feasibility Gates

**Files:**
- Modify: `viewer/src/lib/compute/onDemandDiagnostics.ts`
- Modify: `viewer/src/lib/components/scene/Scene.svelte`
- Modify: `viewer/src/routes/debug-webgpu-utci/+page.svelte`
- Modify: `viewer/tests/e2e/webgpu-f32-on-demand-vertical-slice.spec.ts`

This task must pass before any code reports `renderTransport='compute-buffer-selected-hour'`.

- [ ] **Step 1: Add fail-closed diagnostics fields**

Extend `OnDemandRuntimeDiagnostics`:

```ts
gpuResidentRenderAvailable: boolean;
sameDeviceForComputeAndRender: boolean | null;
gpuResidentCopyStatus: 'idle' | 'pending' | 'complete' | 'failed';
gpuResidentCopyError?: string;
```

Set conservative defaults in `createEmptyOnDemandDiagnostics()`:

```ts
gpuResidentRenderAvailable: false,
sameDeviceForComputeAndRender: null,
gpuResidentCopyStatus: 'idle',
```

- [ ] **Step 2: Expose renderer device identity from `Scene.svelte`**

Update the diagnostics callback type:

```ts
export let onRendererDiagnostics:
	| ((diagnostics: {
			rendererBackend: 'webgpu' | 'unknown';
			rendererDevice?: GPUDevice;
			error?: string;
		}) => void)
	| undefined = undefined;
```

After renderer init resolves, publish:

```ts
const initializedBackend =
	(renderer as unknown as { backend?: { isWebGPUBackend?: boolean; device?: GPUDevice } }).backend;
onRendererDiagnostics?.({
	rendererBackend: initializedBackend?.isWebGPUBackend ? 'webgpu' : 'unknown',
	rendererDevice: initializedBackend?.device,
});
```

Do not serialize the device into window diagnostics. Store only an internal reference in the route.

- [ ] **Step 3: Expose compute device identity**

In `viewer/src/lib/compute/webgpuUtciPipeline.ts`, add:

```ts
getDeviceForDebug(): GPUDevice {
	return this.device;
}
```

Add a wrapper in `ComputeManager`:

```ts
getDeviceForDebug(): GPUDevice | undefined {
	return this.pipeline.getDeviceForDebug?.();
}
```

If the interface does not currently include this method, extend the local `UTCIComputePipeline` type with optional `getDeviceForDebug?: () => GPUDevice`.

- [ ] **Step 4: Add same-device assertion before GPU-resident render**

In the debug route, keep an internal `rendererDeviceForDebug: GPUDevice | undefined`.

Update `handleRendererDiagnostics()`:

```ts
rendererBackend = diagnostics.rendererBackend;
rendererDeviceForDebug = diagnostics.rendererDevice;
```

Add:

```ts
function canUseGpuResidentRender(computeManager: ComputeManager): {
	available: boolean;
	sameDevice: boolean | null;
	error?: string;
} {
	const computeDevice = computeManager.getDeviceForDebug();
	if (!rendererDeviceForDebug || !computeDevice) {
		return { available: false, sameDevice: null, error: 'Missing renderer or compute GPUDevice.' };
	}
	if (rendererDeviceForDebug !== computeDevice) {
		return {
			available: false,
			sameDevice: false,
			error: 'Compute and render WebGPU devices differ; cannot copy buffers across devices.',
		};
	}
	return { available: true, sameDevice: true };
}
```

- [ ] **Step 5: Add E2E that fails closed when the same-device gate is not satisfied**

Append:

```ts
test('debug on-demand does not claim GPU-resident rendering unless compute and render share a GPUDevice', async ({
	page
}) => {
	test.setTimeout(180_000);
	await page.goto(
		'/debug-webgpu-utci?onDemandPrototype=1&utciOnDemand=f32&utciRender=gpu&timeIndex=12'
	);

	const hasWebGpu = await page.evaluate(() => Boolean(navigator.gpu));
	test.skip(!hasWebGpu && process.env.REQUIRE_WEBGPU_ON_DEMAND !== '1', 'WebGPU unavailable.');

	await page.waitForFunction(() => {
		const diagnostics = (window as Window & { __onDemandPrototypeDiagnostics__?: any })
			.__onDemandPrototypeDiagnostics__;
		return diagnostics?.path === 'exposure-only-f32';
	}, undefined, { timeout: 180_000 });

	const diagnostics = await page.evaluate(() => {
		return (window as Window & { __onDemandPrototypeDiagnostics__?: any })
			.__onDemandPrototypeDiagnostics__;
	});

	if (diagnostics.sameDeviceForComputeAndRender !== true) {
		expect(diagnostics.renderTransport).not.toBe('compute-buffer-selected-hour');
		expect(diagnostics.gpuResidentRenderAvailable).toBe(false);
		expect(diagnostics.gpuResidentCopyStatus).not.toBe('complete');
	} else {
		expect(diagnostics.gpuResidentRenderAvailable).toBe(true);
	}
});
```

- [ ] **Step 6: Run the feasibility gate test**

Run:

```powershell
cd viewer
$env:REQUIRE_WEBGPU_ON_DEMAND='1'
npm run test:e2e -- tests/e2e/webgpu-f32-on-demand-vertical-slice.spec.ts --project=chromium --grep "same GPUDevice"
```

Expected: PASS. It may still report the transitional CPU bridge if devices differ or storage is not ready.

### Task 6: Prove A Three-Compatible GPU-Resident UTCI Surface Contract

**Files:**
- Modify: `viewer/src/lib/services/gpuUtciRenderBridge.ts`
- Modify: `viewer/tests/services/pointCloudService.surface.test.ts`
- Modify: `viewer/tests/compute/webgpu-on-demand-source-locks.test.ts`

- [ ] **Step 1: Add the render source type**

In `viewer/src/lib/services/gpuUtciRenderBridge.ts`, expose:

```ts
export type GpuNativeUtciSurfaceSource =
	| 'cpu-uploaded-selected-hour'
	| 'compute-buffer-selected-hour';
```

Use this type in `GpuNativeUtciSurfaceState.source`.

- [ ] **Step 2: Add behavioral source tests that prevent false zero-copy claims**

In `viewer/tests/services/pointCloudService.surface.test.ts`, add a CPU-uploaded bridge assertion:

```ts
it('labels the existing gpuNative surface as CPU-uploaded selected-hour', () => {
	const layout = createTestUtciGridLayout({ width: 2, height: 2, numPositions: 4 });
	const mesh = createGpuNativeUtciSurfaceMesh({
		layout,
		colors: new Float32Array(4 * 3),
	});

	expect(getGpuNativeUtciSurfaceSource(mesh)).toBe('cpu-uploaded-selected-hour');
	disposeGpuNativeUtciSurfaceMesh(mesh);
});
```

Add a compute-buffer bridge assertion after the new constructor exists:

```ts
it('labels only compute-buffer surfaces as compute-buffer selected-hour', () => {
	const layout = createTestUtciGridLayout({ width: 2, height: 2, numPositions: 4 });
	const fakeBuffer = {} as GPUBuffer;
	const mesh = createComputeBufferUtciSurfaceMesh({
		layout,
		utciBuffer: fakeBuffer,
		utciRange: { min: 0, max: 50 },
	});

	expect(getGpuNativeUtciSurfaceSource(mesh)).toBe('compute-buffer-selected-hour');
	disposeGpuNativeUtciSurfaceMesh(mesh);
});
```

- [ ] **Step 3: Add a new constructor name for compute-buffer-backed surfaces**

Add a stub that throws until the GPU-resident path is implemented:

```ts
export interface ComputeBufferUtciSurfaceMeshOptions {
	layout: UtciGridLayout;
	utciBuffer: GPUBuffer;
	utciRange: { min: number; max: number };
	opacity?: number;
}

export function createComputeBufferUtciSurfaceMesh(
	_options: ComputeBufferUtciSurfaceMeshOptions
): THREE.Mesh {
	throw new Error('compute-buffer-selected-hour rendering is not implemented yet.');
}
```

- [ ] **Step 4: Run service/source tests**

Run:

```powershell
cd viewer
npm test -- tests/compute/webgpu-on-demand-source-locks.test.ts tests/services/pointCloudService.surface.test.ts
```

Expected: PASS for the source-lock tests and existing surface tests.

### Task 7: Implement The Compute-Buffer-Backed Surface

**Files:**
- Modify: `viewer/src/lib/services/gpuUtciRenderBridge.ts`
- Modify: `viewer/src/lib/services/pointCloudService.ts`

This task is the core architecture step. Use Three r175 WebGPU/TSL APIs already present in the repo. If direct wrapping of an existing raw `GPUBuffer` is not supported by Three, use a Three-owned `StorageBufferAttribute` as the render-owned UTCI storage and copy the compute output into it on GPU with `copyBufferToBuffer`. Do not map/read selected-hour UTCI to CPU on the render path.

- [ ] **Step 1: Add a render-owned UTCI storage attribute**

In `gpuUtciRenderBridge.ts`, define state for compute-buffer surfaces:

```ts
interface ComputeBufferUtciSurfaceState extends GpuNativeUtciSurfaceState {
	source: 'compute-buffer-selected-hour';
	utciStorageAttribute: StorageBufferAttribute;
	vertexToPointStorageAttribute: StorageBufferAttribute;
	utciRange: { min: number; max: number };
	minUniform: ReturnType<typeof uniform>;
	maxUniform: ReturnType<typeof uniform>;
}
```

- [ ] **Step 2: Add a GPU color mapping node**

Update the TSL imports in `gpuUtciRenderBridge.ts`:

```ts
import { clamp, float, mix, storage, uniform, vec3, vertexIndex } from 'three/tsl';
```

Create a helper in `gpuUtciRenderBridge.ts`. This first GPU-resident color ramp is intentionally simple and deterministic: blue at low UTCI, yellow/white at high UTCI. UTCI numeric parity is tested separately against Python/WebGPU values.

```ts
function createUtciColorNode(
	utciStorageAttribute: StorageBufferAttribute,
	vertexToPointStorageAttribute: StorageBufferAttribute,
	minUniform: ReturnType<typeof uniform>,
	maxUniform: ReturnType<typeof uniform>
) {
	const utciStorage = storage(utciStorageAttribute, 'float', utciStorageAttribute.count).toReadOnly();
	const vertexToPointStorage = storage(
		vertexToPointStorageAttribute,
		'uint',
		vertexToPointStorageAttribute.count
	).toReadOnly();

	const pointIndex = vertexToPointStorage.element(vertexIndex);
	const value = utciStorage.element(pointIndex);
	const t = clamp(
		value.sub(minUniform).div(maxUniform.sub(minUniform).max(float(0.001))),
		0,
		1
	);

	return {
		colorNode: mix(vec3(0.08, 0.25, 0.95), vec3(1.0, 0.92, 0.25), t),
		opacityNode: t.mul(0).add(DEFAULT_SURFACE_OPACITY),
	};
}
```

If the existing app color ramp must be matched exactly later, add a follow-up task after GPU-resident rendering passes. Do not block this plan on exact color-ramp parity because UTCI numeric parity is independent of the visual ramp. Use uniforms so month/hour range changes do not require shader reconstruction.

- [ ] **Step 3: Build vertex-to-point mapping once**

Add a helper that maps each surface vertex to the source point index:

```ts
function createVertexToPointIndexArray(layout: UtciGridLayout): Uint32Array {
	const cellCount = layout.width * layout.height;
	const fallbackPointIndex = 0;
	const cellToPoint = new Uint32Array(cellCount);
	cellToPoint.fill(fallbackPointIndex);

	for (let pointIndex = 0; pointIndex < layout.numPositions; pointIndex += 1) {
		const row = layout.indexToRow[pointIndex];
		const column = layout.indexToColumn[pointIndex];
		if (row >= layout.height || column >= layout.width) continue;
		cellToPoint[row * layout.width + column] = pointIndex;
	}

	const indices = new Uint32Array(cellCount * SURFACE_VERTICES_PER_CELL);
	let offset = 0;
	for (let cellIndex = 0; cellIndex < cellCount; cellIndex += 1) {
		const pointIndex = cellToPoint[cellIndex];
		for (let i = 0; i < SURFACE_VERTICES_PER_CELL; i += 1) {
			indices[offset++] = pointIndex;
		}
	}
	return indices;
}
```

Add a unit test with shuffled `indexToRow` / `indexToColumn` so this cannot regress to row-major point order.

- [ ] **Step 4: Implement `createComputeBufferUtciSurfaceMesh()`**

Implement the constructor so it:

```ts
const geometry = createGpuNativeSurfaceGeometry(options.layout);
const utciArray = new Float32Array(options.layout.numPositions);
const utciStorageAttribute = new StorageBufferAttribute(utciArray, 1);
const vertexToPointArray = createVertexToPointIndexArray(options.layout);
const vertexToPointStorageAttribute = new StorageBufferAttribute(vertexToPointArray, 1);
const minUniform = uniform(options.utciRange.min);
const maxUniform = uniform(options.utciRange.max);
const { colorNode, opacityNode } = createUtciColorNode(
	utciStorageAttribute,
	vertexToPointStorageAttribute,
	minUniform,
	maxUniform
);

const material = new MeshBasicNodeMaterial({
	side: THREE.DoubleSide,
	transparent: true,
	depthTest: true,
	depthWrite: false,
});
material.colorNode = colorNode;
material.opacityNode = opacityNode;
material.toneMapped = false;

const mesh = new THREE.Mesh(geometry, material);
mesh.name = 'UTCI GPU Resident Surface Overlay';
mesh.frustumCulled = false;
mesh.renderOrder = 2;
mesh.userData[GPU_NATIVE_SURFACE_STATE_KEY] = {
	colorStorageAttribute: utciStorageAttribute,
	utciStorageAttribute,
	vertexToPointStorageAttribute,
	width: options.layout.width,
	height: options.layout.height,
	gridSize: options.layout.gridSize,
	vertexCount: geometry.getAttribute('position').count,
	source: 'compute-buffer-selected-hour',
	utciRange: options.utciRange,
	minUniform,
	maxUniform,
} satisfies ComputeBufferUtciSurfaceState;
```

Do not call `utciArray.set()` from selected-hour readback data in this path.

- [ ] **Step 5: Add a GPU-only update function**

Add:

```ts
export function updateComputeBufferUtciSurfaceMesh(
	mesh: THREE.Mesh,
	options: ComputeBufferUtciSurfaceMeshOptions
): boolean {
	const state = mesh.userData[GPU_NATIVE_SURFACE_STATE_KEY] as
		| ComputeBufferUtciSurfaceState
		| undefined;
	if (!state || state.source !== 'compute-buffer-selected-hour') return false;
	if (
		state.width !== options.layout.width ||
		state.height !== options.layout.height ||
		state.gridSize !== options.layout.gridSize
	) {
		return false;
	}

	state.utciRange = options.utciRange;
	state.minUniform.value = options.utciRange.min;
	state.maxUniform.value = options.utciRange.max;
	mesh.userData.pendingComputeBufferUtciSource = options.utciBuffer;
	mesh.userData.utciSurfaceSource = 'compute-buffer-selected-hour';
	return true;
}
```

The actual GPU buffer copy will be triggered from the render integration task, where renderer/device access is available.

- [ ] **Step 6: Extend `pointCloudService` backend metadata without switching default behavior**

Add a new internal backend mode only if needed:

```ts
export type UtciSurfaceBackendType = 'dataTexture' | 'gpuNative';
```

Keep the public type unchanged if the debug route can call the compute-buffer constructor directly. Do not make production default behavior depend on the new path yet.

- [ ] **Step 7: Run unit tests**

Run:

```powershell
cd viewer
npm test -- tests/services/pointCloudService.surface.test.ts tests/compute/webgpu-on-demand-source-locks.test.ts
```

Expected: PASS.

### Task 8: Wire Compute Output Through The Scene-Owned GPU-Resident Surface

**Files:**
- Modify: `viewer/src/routes/debug-webgpu-utci/+page.svelte`
- Modify: `viewer/src/lib/components/scene/UTCIPointCloud.svelte`
- Modify: `viewer/src/lib/compute/webgpuUtciPipeline.ts`
- Modify: `viewer/tests/e2e/webgpu-f32-on-demand-vertical-slice.spec.ts`

- [ ] **Step 1: Preserve compute output handle and metadata**

Ensure `runUtciForTimeIndex()` returns:

```ts
return {
	format,
	numPoints,
	timeIndex,
	gpuBuffer: onDemandOutputBuffer,
	debugLabel: 'webgpu-on-demand-f32-utci',
};
```

This already exists; do not remove it.

- [ ] **Step 2: Add diagnostics for GPU-resident render path after copy completion only**

Do not publish `compute-buffer-selected-hour` when `runDebugOnDemandSelectedHour()` merely receives `output`. Publish it only after the scene-owned component reports:

```ts
sameDeviceForComputeAndRender === true
gpuResidentCopyStatus === "complete"
utciSurfaceSource === "compute-buffer-selected-hour"
```

Then publish:

```ts
renderTransport: "compute-buffer-selected-hour",
selectedHourReadbackCount: 0,
debugReadbackCount: pipelineDiagnostics?.debugReadbackCount ?? 0,
selectedHourTransferCount: 0,
```

Only do this on a route branch where no selected-hour CPU readback is used for rendering. Python `.bin` sample comparison may still perform a separate diagnostic readback if explicitly requested; label it as debug-only and keep it out of render transport counts.

- [ ] **Step 3: Use CPU readback only for comparison mode**

Split render and comparison:

```ts
const shouldReadbackForComparison = params.readbackForComparison && pythonBinComparisonEnabled;
const selectedHourUtciForComparison = shouldReadbackForComparison
	? await prepared.pipeline.readOnDemandUtciForDebug?.({ numPoints: prepared.numPoints })
	: undefined;
```

Do not build `selectedHourAnalysis` from this array for the GPU-resident render path.

- [ ] **Step 4: Pass accepted compute output into a scene-owned surface path**

In the debug route, create a small accepted-output state:

```ts
type AcceptedGpuResidentUtciOutput = {
	requestId: number;
	monthIndex: number;
	hourIndex: number;
	timeIndex: number;
	output: Awaited<ReturnType<ComputeManager["runUtciForTimeIndex"]>>;
	utciRange: { min: number; max: number };
};

let acceptedGpuResidentUtciOutput: AcceptedGpuResidentUtciOutput | null = null;
```

Set it only after `completed.accepted === true` and the request still owns the selection. Pass it as a prop to `UTCIPointCloud` or a sibling `GpuResidentUTCISurface` component inside the `<Scene>` slot. The component must use `useThrelte()` because it owns `scene`, `invalidate`, and renderer lifecycle access.

- [ ] **Step 5: Initialize render storage before copying**

In the scene-owned component:

1. Create or update the compute-buffer surface mesh.
2. Add it to the scene.
3. Call `invalidate()`.
4. Wait one animation frame.
5. Look up the Three-owned storage buffer.

If the storage buffer is still missing, publish:

```ts
{
	gpuResidentCopyStatus: "failed",
	gpuResidentCopyError: "Three storage buffer was not initialized after render invalidation.",
	renderTransport: "cpu-uploaded-selected-hour"
}
```

and keep the transitional CPU path active.

- [ ] **Step 6: Implement GPU buffer copy into the render-owned storage**

If Three does not support binding our existing compute `GPUBuffer` directly as a TSL storage node, copy the selected-hour compute output into the Three-owned `StorageBufferAttribute` buffer on GPU:

```ts
const rendererBackend = renderer.backend as unknown as {
	device?: GPUDevice;
	get?: (attribute: StorageBufferAttribute) => { buffer?: GPUBuffer };
};
const target = rendererBackend.get?.(state.utciStorageAttribute)?.buffer;
if (!rendererBackend.device || !target) {
	throw new Error('Three WebGPU storage buffer is not initialized for UTCI render copy.');
}
if (rendererBackend.device !== computeDevice) {
	throw new Error('Compute and render WebGPU devices differ; cannot copy selected-hour buffer.');
}
const encoder = rendererBackend.device.createCommandEncoder();
encoder.copyBufferToBuffer(output.gpuBuffer, 0, target, 0, output.numPoints * 4);
rendererBackend.device.queue.submit([encoder.finish()]);
await rendererBackend.device.queue.onSubmittedWorkDone();
```

If backend access differs in Three r175, inspect `viewer/node_modules/three/src/renderers/webgpu` and adjust to the local API. Keep this code isolated in one helper, for example `copyComputeUtciBufferToRenderStorage(...)`.

- [ ] **Step 7: Protect against overwrite/stale-copy hazards**

Use the accepted request id on both sides:

```ts
if (currentAcceptedOutput.requestId !== copyRequestId) {
	return { copied: false, stale: true };
}
```

If a newer scrub request arrives before copy completion, discard the older copy result and do not publish `gpuResidentCopyStatus='complete'` for it. Keep the existing stale-result counters.

- [ ] **Step 8: Add E2E assertion for GPU-resident transport**

In the debug on-demand test, once the new branch is active:

```ts
expect(diagnostics.renderTransport).toBe('compute-buffer-selected-hour');
expect(diagnostics.selectedHourReadbackCount).toBe(0);
expect(diagnostics.selectedHourTransferCount).toBe(0);
expect(diagnostics.dataTextureBuildCount).toBe(0);
expect(diagnostics.utciSurfaceSource).toBe('compute-buffer-selected-hour');
```

Keep a separate test for `utciRender=data` fallback.

- [ ] **Step 9: Run focused E2E**

Run:

```powershell
cd viewer
$env:REQUIRE_WEBGPU_ON_DEMAND='1'
npm run test:e2e -- tests/e2e/webgpu-f32-on-demand-vertical-slice.spec.ts --project=chromium --grep "debug route can use f32 on-demand"
```

Expected: PASS with `renderTransport='compute-buffer-selected-hour'`.

---

## Milestone 4: Full Debug Experience Verification

### Task 9: Verify Month, Hour, Parity, Timing, And Fallback Together

**Files:**
- Modify: `viewer/tests/e2e/webgpu-f32-on-demand-vertical-slice.spec.ts`
- Modify: `docs/superpowers/plans/2026-05-07-webgpu-compute-on-demand-prototype-results.md`

- [ ] **Step 1: Run compute/unit verification**

Run:

```powershell
cd viewer
npm test -- tests/compute/onDemandDiagnostics.test.ts tests/compute/onDemandSizing.test.ts tests/compute/onDemandOutputFormat.test.ts tests/compute/compute-manager-on-demand.test.ts tests/compute/webgpu-on-demand-source-locks.test.ts tests/services/pointCloudService.surface.test.ts
```

Expected: PASS.

- [ ] **Step 2: Run full on-demand E2E verification**

Run:

```powershell
cd viewer
$env:REQUIRE_WEBGPU_ON_DEMAND='1'
npm run test:e2e -- tests/e2e/webgpu-on-demand-prototype.spec.ts --project=chromium
npm run test:e2e -- tests/e2e/webgpu-f32-on-demand-vertical-slice.spec.ts --project=chromium
```

Expected: PASS.

- [ ] **Step 3: Run explicit fallback and August Python comparison checks**

Run:

```powershell
cd viewer
$env:REQUIRE_WEBGPU_ON_DEMAND='1'
npm run test:e2e -- tests/e2e/webgpu-f32-on-demand-vertical-slice.spec.ts --project=chromium --grep "dataTexture fallback|python-bin comparison"
```

Expected: PASS. `utciRender=data` must still report `dataTexture`, and August sampled Python `.bin` comparison must remain active.

- [ ] **Step 4: Compare against old full-run timing context**

Read:

```powershell
Get-Content ..\data\batch-parity-results\Ben-Gurion_20250815_grid_2m_fullday_timing.json
```

Record these old full-run reference fields in the results note:

```md
| Old `webgpu_1m.compute_s` | `1.828` |
| Old `webgpu_12m.compute_s` | `2.923` |
| Old `webgpu_12m` phases | `runAll`, `readback` |
```

Do not overwrite the JSON timing file from this plan unless the user explicitly asks to refresh batch parity artifacts.

- [ ] **Step 5: Capture final debug diagnostics**

In a browser/E2E evaluation, capture:

```ts
const diagnostics = await page.evaluate(() => {
	return (window as Window & { __onDemandPrototypeDiagnostics__?: any })
		.__onDemandPrototypeDiagnostics__;
});
const monthHourComparison = await page.evaluate(() => {
	return (window as Window & { __onDemandMonthHourComparison__?: any })
		.__onDemandMonthHourComparison__;
});
```

Append actual values to `docs/superpowers/plans/2026-05-07-webgpu-compute-on-demand-prototype-results.md`.

- [ ] **Step 6: Final sanity checks**

Confirm:

```md
- [ ] Debug route hour scrub works.
- [ ] Debug route month scrub works.
- [ ] On-demand selected month/hour matches separate WebGPU `runAll()` baseline.
- [ ] August Python `.bin` comparison remains active where available.
- [ ] `utciRender=data` fallback remains available.
- [ ] `runAll()`, `readUtciBulk()`, and `.bin` paths remain callable.
- [ ] GPU path reports `renderTransport='compute-buffer-selected-hour'`.
- [ ] `sameDeviceForComputeAndRender=true` before GPU-resident transport is claimed.
- [ ] `gpuResidentCopyStatus='complete'` before GPU-resident transport is claimed.
- [ ] CPU-uploaded bridge is not described as zero-copy.
- [ ] Timing numbers are captured but not over-interpreted as total browser/OS VRAM or pure GPU time.
- [ ] No commits were made.
- [ ] No git worktrees were created.
```

---

## Risks And Design Notes

- **Three raw `GPUBuffer` interop may be limited.** If direct binding is not possible, use a Three-owned storage buffer and GPU-side `copyBufferToBuffer`. This still removes selected-hour CPU readback/upload from the render path.
- **Color parity is not UTCI parity.** UTCI values must match Python/WebGPU baselines; GPU color ramp only needs to preserve visual semantics and avoid flat/incorrect output.
- **Debug comparison readback can remain.** A debug-only readback for sampled comparison is acceptable if diagnostics label it separately from render transport.
- **Month parity uses WebGPU baseline.** Python `.bin` remains the reference only for the available August dataset.
- **Do not remove fallbacks.** Keep `.bin`, `runAll()`, `readUtciBulk()`, and `dataTexture` until a later production-switch plan.

## Suggested Verification Agents

After drafting or changing this plan, run two read-only review agents:

1. **Architecture reviewer:** check whether the GPU-resident bridge steps are feasible with Three r175 and whether they avoid false zero-copy claims.
2. **Test reviewer:** check whether month scrub, WebGPU `runAll()` comparison, timing capture, fallback preservation, and ray-flip allowances are all covered.

Revise this plan before implementation if either reviewer finds a blocker.
