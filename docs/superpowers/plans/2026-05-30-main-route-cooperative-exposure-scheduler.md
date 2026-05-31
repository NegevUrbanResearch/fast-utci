# Main Route Cooperative Exposure Scheduler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. No commits. No git worktrees.

**Goal:** Add a flag-gated cooperative/tiled exposure scheduler for the main-route WebGPU cold path and prove whether it removes the NZ 0.5m desktop visual freeze without regressing first visible, first scrub, memory, or GPU-native proof boundaries.

**Architecture:** Keep the default exposure path unchanged throughout this plan. Thread an explicit main-route query flag into the selected-hour session, split exposure precompute into bounded point slices, wait/yield between slices, and record slice-level diagnostics. Compare control vs chunked runs with headed freeze-map and cold-start waterfall collectors; passing evidence only permits a recommendation, not a default-behavior change.

**Tech Stack:** Svelte 5, TypeScript, Vitest, Playwright, WebGPU/WGSL, Three.js WebGPU, existing `window.__utciRenderDiagnostics__`, `main-route-visual-freeze-map`, and selected-hour runtime diagnostics.

---

## Hard Constraints

- No commits.
- No git worktrees.
- Preserve unrelated dirty files.
- Keep `/` as the proof surface, not `/debug`.
- Keep default behavior unchanged throughout this plan; passing gates only permits an evidence recommendation.
- Keep GPU-native proof boundaries: `rendererBackend=webgpu`, `compute-buffer-selected-hour`, same compute/render device, no visible selected-hour readback fallback.
- Do not reintroduce the old lazy-loading regression shape: first visible cannot improve by stealing responsiveness from first post-visible scrub.
- Do not claim success from total wall-clock alone; max rAF gap and first post-visible scrub are hard gates.

## Review Gate After Every Task

- After each file-changing task, run a fresh spec-compliance review subagent first.
- Do not start code-quality review until spec compliance is clean.
- Then run a fresh code-quality review subagent focused on maintainability, stale-session safety, diagnostics overhead, and unrelated formatting.
- Fix review findings before moving to the next task.
- Do not commit and do not create a git worktree at any point.

## Evidence Baseline

Current committed diagnostics (`38291b1`) show the NZ 0.5m pressure case:

- `pointCount`: `8,171,761`
- `drawIndices`: `49,030,566`
- app-owned GPU memory: `685.9 MiB`
- top-level first selected-hour publication clock: `26.84s`
- `exposurePrecomputeMs`: `16.32s`
- `exposureQueueWaitMs`: `16.32s`
- exposure encode time: about `0.2ms`
- `staticUploadTrace.totalMs`: `417ms`
- `rangeResolve`: about `2.57s`
- `renderUpdateMs`: about `4.75s`
- top rAF gap: about `3.02s`
- top interval gap: about `1.87s`
- top long task: about `1.56s`

Interpretation: exposure queue execution is the lead freeze source. Static upload and mesh creation are secondary. Render/range work remains a second-stage tail after exposure is fixed.

## File Structure

- Create `viewer/src/lib/compute/gpu/exposureScheduling.ts`
  - Owns scheduler option types, query parsing, defaults, and point-slice construction helpers.
  - Contains pure functions so chunk sizing, defaults, and query parsing are testable outside WebGPU.
- Modify `viewer/src/lib/compute/gpu/gpu-pipeline.ts`
  - Add optional `exposureScheduling` and `signal` to `ExposurePrecomputeParams`.
- Modify `viewer/src/lib/compute/on-demand/onDemandDiagnostics.ts`
  - Add scheduler diagnostics fields: mode, slice counts, queue wait totals/max, yield counts, and configured slice size.
- Modify `viewer/src/lib/compute/gpu/webgpuUtciPipeline.ts`
  - Keep current single-submit path for default mode.
  - Add chunked branch in `runExposurePrecompute`.
  - Add `encodeExposurePassesForChunks` helper so the single and chunked branches share pass-encoding logic.
  - Add `yieldToBrowserFrame` helper for browser-safe yielding.
  - Add abort/dispose guards so stale sessions stop between scheduler slices.
- Modify `viewer/src/lib/compute/selected-hour/liveUtciSelectedHourSession.ts`
  - Accept and store `exposureScheduling`.
  - Pass it to `computeManager.runExposurePrecompute`.
- Modify `viewer/src/lib/compute/selected-hour/liveSelectedHourRouteHost.ts`
  - Add `exposureScheduling` to route inputs and session config creation.
- Modify `viewer/src/routes/+page.svelte`
  - Parse main-route query params and pass scheduling options to the live route host.
  - Include scheduling fields in `window.__utciRenderDiagnostics__`.
- Modify `viewer/tests/compute/exposureScheduling.test.ts`
  - New tests for defaults, query parsing, slice construction, and invalid values.
- Modify `viewer/tests/compute/compute-manager-on-demand.test.ts`
  - Prove `ExposurePrecomputeParams.exposureScheduling` is delegated unchanged.
- Modify `viewer/tests/compute/live-selected-hour-session.test.ts`
  - Prove the session passes scheduler options into exposure precompute.
- Modify `viewer/tests/compute/live-selected-hour-route-host.test.ts`
  - Prove route-host session config includes scheduler options and that `buildControllerIdentity` / `sessionKey` changes when options change.
- Modify `viewer/tests/compute/webgpuUtciPipeline.behavior.test.ts`
  - Add behavior-level proof that default scheduling submits once, chunked scheduling submits/waits per scheduler slice, abort stops before later slices, and scheduler diagnostics are recorded.
- Modify `viewer/tests/compute/webgpu-on-demand-source-locks.test.ts`
  - Keep only narrow architectural source locks for default single-submit branch and chunked helper presence.
- Modify `viewer/tests/e2e/main-route-visual-freeze-map.spec.ts`
  - Add an NZ 0.5m chunked variant next to the existing control case.
  - Persist scheduler fields and compare max rAF gap, exposure time, first visible, and proof contract.
- Modify `viewer/tests/e2e/main-route-cold-start-waterfall.spec.ts`
  - Add an explicit NZ 0.5m chunked variant that records initial first visible and first post-visible scrub in the same artifact.
  - This is mandatory evidence before any success claim or default-behavior recommendation.

---

## Task 1: Add Scheduler Types, Defaults, Query Parsing, And Slice Helpers

**Files:**
- Create: `viewer/src/lib/compute/gpu/exposureScheduling.ts`
- Test: `viewer/tests/compute/exposureScheduling.test.ts`

- [ ] **Step 1: Write failing unit tests for scheduler defaults and query parsing**

Add `viewer/tests/compute/exposureScheduling.test.ts`:

```ts
import { describe, expect, it } from 'vitest';
import {
	DEFAULT_EXPOSURE_MAX_WORKGROUPS_PER_SLICE,
	buildExposurePointSlices,
	parseExposureSchedulingFromSearchParams
} from '../../src/lib/compute/gpu/exposureScheduling';

describe('exposureScheduling', () => {
	it('defaults to single-submit exposure scheduling', () => {
		expect(parseExposureSchedulingFromSearchParams(new URLSearchParams(''))).toEqual({
			mode: 'single-submit',
			maxWorkgroupsPerSlice: DEFAULT_EXPOSURE_MAX_WORKGROUPS_PER_SLICE,
			yieldBetweenSlices: true
		});
	});

	it('parses the chunked query flag and clamps slice size', () => {
		const params = new URLSearchParams(
			'utciExposureSchedule=chunked&utciExposureMaxWorkgroupsPerSlice=8192'
		);

		expect(parseExposureSchedulingFromSearchParams(params)).toEqual({
			mode: 'chunked',
			maxWorkgroupsPerSlice: 8192,
			yieldBetweenSlices: true
		});
	});

	it('ignores invalid mode and invalid slice sizes', () => {
		const params = new URLSearchParams(
			'utciExposureSchedule=banana&utciExposureMaxWorkgroupsPerSlice=-1'
		);

		expect(parseExposureSchedulingFromSearchParams(params)).toEqual({
			mode: 'single-submit',
			maxWorkgroupsPerSlice: DEFAULT_EXPOSURE_MAX_WORKGROUPS_PER_SLICE,
			yieldBetweenSlices: true
		});
	});

	it('builds bounded point slices without dropping points', () => {
		const slices = buildExposurePointSlices({
			numPoints: 1_000,
			workgroupSize: 64,
			maxWorkgroupsPerSlice: 4
		});

		expect(slices).toEqual([
			{ pointOffset: 0, pointCount: 256, workgroupsX: 4 },
			{ pointOffset: 256, pointCount: 256, workgroupsX: 4 },
			{ pointOffset: 512, pointCount: 256, workgroupsX: 4 },
			{ pointOffset: 768, pointCount: 232, workgroupsX: 4 }
		]);
	});
});
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```powershell
cd viewer
npm run test -- --run tests/compute/exposureScheduling.test.ts
```

Expected: FAIL because `exposureScheduling.ts` does not exist.

- [ ] **Step 3: Implement the pure scheduler helper**

Create `viewer/src/lib/compute/gpu/exposureScheduling.ts`:

```ts
import type { PointDispatchChunk } from '$lib/compute/gpu/gpu-pipeline';

export type ExposureSchedulingMode = 'single-submit' | 'chunked';

export interface ExposureSchedulingOptions {
	mode: ExposureSchedulingMode;
	maxWorkgroupsPerSlice: number;
	yieldBetweenSlices: boolean;
}

export const DEFAULT_EXPOSURE_MAX_WORKGROUPS_PER_SLICE = 8192;
export const MIN_EXPOSURE_MAX_WORKGROUPS_PER_SLICE = 1;
export const MAX_EXPOSURE_MAX_WORKGROUPS_PER_SLICE = 65_535;

export const DEFAULT_EXPOSURE_SCHEDULING: ExposureSchedulingOptions = {
	mode: 'single-submit',
	maxWorkgroupsPerSlice: DEFAULT_EXPOSURE_MAX_WORKGROUPS_PER_SLICE,
	yieldBetweenSlices: true
};

export function parseExposureSchedulingFromSearchParams(
	params: URLSearchParams
): ExposureSchedulingOptions {
	const mode =
		params.get('utciExposureSchedule') === 'chunked'
			? 'chunked'
			: 'single-submit';
	const rawMaxWorkgroups = Number(params.get('utciExposureMaxWorkgroupsPerSlice'));
	const maxWorkgroupsPerSlice =
		Number.isFinite(rawMaxWorkgroups) && rawMaxWorkgroups >= MIN_EXPOSURE_MAX_WORKGROUPS_PER_SLICE
			? Math.min(Math.floor(rawMaxWorkgroups), MAX_EXPOSURE_MAX_WORKGROUPS_PER_SLICE)
			: DEFAULT_EXPOSURE_MAX_WORKGROUPS_PER_SLICE;
	const yieldBetweenSlices = params.get('utciExposureYieldBetweenSlices') !== '0';

	return {
		mode,
		maxWorkgroupsPerSlice,
		yieldBetweenSlices
	};
}

export function areExposureSchedulingOptionsEqual(
	left: ExposureSchedulingOptions | undefined,
	right: ExposureSchedulingOptions | undefined
): boolean {
	const resolvedLeft = left ?? DEFAULT_EXPOSURE_SCHEDULING;
	const resolvedRight = right ?? DEFAULT_EXPOSURE_SCHEDULING;
	return (
		resolvedLeft.mode === resolvedRight.mode &&
		resolvedLeft.maxWorkgroupsPerSlice === resolvedRight.maxWorkgroupsPerSlice &&
		resolvedLeft.yieldBetweenSlices === resolvedRight.yieldBetweenSlices
	);
}

export function buildExposurePointSlices(params: {
	numPoints: number;
	workgroupSize: number;
	maxWorkgroupsPerSlice: number;
}): PointDispatchChunk[] {
	const { numPoints, workgroupSize, maxWorkgroupsPerSlice } = params;
	if (numPoints <= 0 || workgroupSize <= 0 || maxWorkgroupsPerSlice <= 0) {
		throw new Error('numPoints, workgroupSize, and maxWorkgroupsPerSlice must be positive');
	}

	const maxPointsPerSlice = workgroupSize * maxWorkgroupsPerSlice;
	const slices: PointDispatchChunk[] = [];
	for (let pointOffset = 0; pointOffset < numPoints; pointOffset += maxPointsPerSlice) {
		const pointCount = Math.min(maxPointsPerSlice, numPoints - pointOffset);
		slices.push({
			pointOffset,
			pointCount,
			workgroupsX: Math.ceil(pointCount / workgroupSize)
		});
	}
	return slices;
}
```

- [ ] **Step 4: Run the helper tests**

Run:

```powershell
cd viewer
npm run test -- --run tests/compute/exposureScheduling.test.ts
```

Expected: PASS.

---

## Task 2: Thread Scheduler Options And Abort Signal From The Main Route To The Live Session

**Files:**
- Modify: `viewer/src/lib/compute/gpu/gpu-pipeline.ts`
- Modify: `viewer/src/lib/compute/selected-hour/liveUtciSelectedHourSession.ts`
- Modify: `viewer/src/lib/compute/selected-hour/liveSelectedHourRouteHost.ts`
- Modify: `viewer/src/routes/+page.svelte`
- Test: `viewer/tests/compute/compute-manager-on-demand.test.ts`
- Test: `viewer/tests/compute/live-selected-hour-session.test.ts`
- Test: `viewer/tests/compute/live-selected-hour-route-host.test.ts`

- [ ] **Step 1: Extend `ExposurePrecomputeParams` with scheduler and abort options**

In `viewer/src/lib/compute/gpu/gpu-pipeline.ts`, add the import:

```ts
import type { ExposureSchedulingOptions } from '$lib/compute/gpu/exposureScheduling';
```

Then update the interface:

```ts
export interface ExposurePrecomputeParams {
	numPoints: number;
	numHours: number;
	numMonths: number;
	exposureScheduling?: ExposureSchedulingOptions;
	signal?: AbortSignal;
}
```

- [ ] **Step 2: Add delegation tests in `compute-manager-on-demand.test.ts`**

Update the existing `runExposurePrecompute delegates when supported` test so the params include scheduling:

```ts
const params = {
	numPoints: 10,
	numHours: 24,
	numMonths: 12,
	signal: new AbortController().signal,
	exposureScheduling: {
		mode: 'chunked' as const,
		maxWorkgroupsPerSlice: 8192,
		yieldBetweenSlices: true
	}
};
```

Expected assertion stays:

```ts
expect(runExposurePrecompute).toHaveBeenCalledWith(params);
```

- [ ] **Step 3: Store scheduler options in selected-hour session state**

In `viewer/src/lib/compute/selected-hour/liveUtciSelectedHourSession.ts`, import the type:

```ts
import type { ExposureSchedulingOptions } from '$lib/compute/gpu/exposureScheduling';
```

Add the optional field to `PreparedSessionState`:

```ts
exposureScheduling?: ExposureSchedulingOptions;
```

Add the optional field to `prepareSelectedHourLiveSession` params:

```ts
exposureScheduling?: ExposureSchedulingOptions;
```

When creating the session, pass it into state:

```ts
return createSelectedHourLiveSession({
	base: sessionBase,
	pipeline: activePipeline,
	computeManager,
	numPoints: initResult.numPoints,
	numHours,
	numMonths,
	deviceSource: preferredDevice ? 'renderer' : 'standalone',
	signal,
	exposureReady: false,
	exposurePrecomputePromise: null,
	requestSequence: 0,
	selectedDayRangeCache: new Map(),
	lifecycleTimings,
	coldStartStartedAt,
	exposureScheduling
});
```

Update `ensureExposurePrecompute`:

```ts
state.exposurePrecomputePromise = state.computeManager
	.runExposurePrecompute({
		numPoints: state.numPoints,
		numHours: state.numHours,
		numMonths: state.numMonths,
		exposureScheduling: state.exposureScheduling,
		signal: state.signal
	})
```

- [ ] **Step 4: Add session test proof**

In `viewer/tests/compute/live-selected-hour-session.test.ts`, add or update a test so the fake `runExposurePrecompute` receives:

```ts
const signal = new AbortController().signal;

const session = await prepareSelectedHourLiveSession({
	analysisId: 'analysis-a',
	base: createFullDayBaseAnalysis(),
	model: {} as Group,
	epwUrl: '/test.epw',
	signal,
	preferredDevice: mockState.rendererDevice,
	exposureScheduling: {
		mode: 'chunked',
		maxWorkgroupsPerSlice: 8192,
		yieldBetweenSlices: true
	}
});
await session.runForSelectedHour({
	monthIndex: 0,
	hourIndex: 0,
	timeIndex: 0,
	selectionKey: 'analysis-a|0|0',
	colorMode: 'normalized',
	preferGpuResident: true
});

const computeManager = mockState.constructors[0];
expect(computeManager.runExposurePrecompute).toHaveBeenCalledWith(
	expect.objectContaining({
		exposureScheduling: {
			mode: 'chunked',
			maxWorkgroupsPerSlice: 8192,
			yieldBetweenSlices: true
		},
		signal
	})
);
```

If this test file uses a mock pipeline rather than `computeManager`, assert against the existing mock `runExposurePrecompute` function. The assertion must prove the session's existing `AbortSignal` is passed into exposure precompute, because chunked exposure can now stop safely between scheduler slices.

- [ ] **Step 5: Thread route-host inputs and session config**

In `viewer/src/lib/compute/selected-hour/liveSelectedHourRouteHost.ts`, import:

```ts
import type { ExposureSchedulingOptions } from '$lib/compute/gpu/exposureScheduling';
import { areExposureSchedulingOptionsEqual } from '$lib/compute/gpu/exposureScheduling';
```

Add to `LiveSelectedHourRouteInputs`:

```ts
exposureScheduling?: ExposureSchedulingOptions;
```

Add to `createSessionConfig` params and return object:

```ts
exposureScheduling?: ExposureSchedulingOptions;
```

```ts
return {
	analysisId: params.analysisId,
	base: params.analysis,
	model: params.model,
	epwUrl: resolveEpwUrl({
		analysisId: params.analysisId,
		fallbackProjectId: params.fallbackProjectId
	}),
	preferredDevice: params.preferredDevice,
	gridResolution: params.gridResolutionMeters,
	exposureScheduling: params.exposureScheduling
};
```

When building session identity/config keys, include the scheduler mode and slice size. This must be added to `buildControllerIdentity(...)`, not only `buildSelectionTriggerKey(...)`, because `createLiveSelectedHourController` reuses the existing session when `request.sessionKey` is unchanged. Extend `buildControllerIdentity` params:

```ts
exposureScheduling?: ExposureSchedulingOptions;
```

Add this segment to the returned identity array:

```ts
`exposure:${params.exposureScheduling?.mode ?? 'single-submit'}:${params.exposureScheduling?.maxWorkgroupsPerSlice ?? 'default'}:${params.exposureScheduling?.yieldBetweenSlices !== false ? 'yield' : 'no-yield'}`
```

Because `buildControllerIdentity` is called inside `buildSelectionPlan(...)`, also extend `buildSelectionPlan` params:

```ts
exposureScheduling?: ExposureSchedulingOptions;
```

When calling `buildControllerIdentity` inside `buildSelectionPlan`, pass:

```ts
exposureScheduling: params.exposureScheduling
```

Update every `buildSelectionPlan(...)` call site. There are four call sites in this file:

- `publishState()` base currentness check
- `publishState()` comparison currentness check
- `reconcileBase(...)`
- `reconcileComparison(...)`

For the two `publishState()` currentness calls, pass:

```ts
exposureScheduling: currentInputs.exposureScheduling
```

For the two reconcile calls, pass:

```ts
exposureScheduling: inputs.exposureScheduling
```

When calling `createSessionConfig` in both base and comparison paths, pass:

```ts
exposureScheduling: inputs.exposureScheduling
```

Keep `buildSelectionTriggerKey(...)` including `controllerIdentity`, so the trigger naturally changes when the controller/session identity changes. Do not rely on `sessionConfig` changes alone; they are not enough.

If the host has an equality helper for route inputs, use `areExposureSchedulingOptionsEqual` instead of object identity.

- [ ] **Step 6: Add route-host session identity tests**

In `viewer/tests/compute/live-selected-hour-route-host.test.ts`, add tests after `threads selected grid resolution into live session identity and config`:

```ts
it('threads exposure scheduling into base live session identity and config', async () => {
	const factory = createControllerFactory();
	const host = createLiveSelectedHourRouteHost(makeHostDeps(factory));
	const baseAnalysis = createFullDayAnalysis({
		label: 'base',
		sourceAnalysisId: 'Ben-Gurion/base',
		baseMin: 18,
		baseMax: 30
	});
	const baseModel = {} as Group;
	const initialScheduling = {
		mode: 'chunked' as const,
		maxWorkgroupsPerSlice: 8192,
		yieldBetweenSlices: true
	};

	host.setRouteInputs(
		makeBaseInputs({
			baseAnalysis,
			baseModel,
			exposureScheduling: initialScheduling
		})
	);
	await host.flush();

	const firstRequest = factory.records[0].requests[0];
	expect(firstRequest?.sessionConfig.exposureScheduling).toEqual(initialScheduling);
	expect(firstRequest?.sessionKey).toContain('exposure:chunked:8192:yield');

	const replacementScheduling = {
		mode: 'chunked' as const,
		maxWorkgroupsPerSlice: 4096,
		yieldBetweenSlices: false
	};
	host.setRouteInputs(
		makeBaseInputs({
			baseAnalysis,
			baseModel,
			exposureScheduling: replacementScheduling
		})
	);
	await host.flush();

	const replacementRequest = factory.records[2].requests[0];
	expect(factory.records[0].dispose).toHaveBeenCalledTimes(1);
	expect(replacementRequest?.sessionConfig.exposureScheduling).toEqual(replacementScheduling);
	expect(replacementRequest?.sessionKey).toContain('exposure:chunked:4096:no-yield');
	expect(replacementRequest?.sessionKey).not.toBe(firstRequest?.sessionKey);
});

it('threads exposure scheduling into comparison live session identity and config', async () => {
	const factory = createControllerFactory();
	const host = createLiveSelectedHourRouteHost(makeHostDeps(factory));
	const scheduling = {
		mode: 'chunked' as const,
		maxWorkgroupsPerSlice: 8192,
		yieldBetweenSlices: true
	};

	host.setRouteInputs(
		makeComparisonInputs({
			utciSurfaceBackend: 'gpuNative',
			exposureScheduling: scheduling
		})
	);
	await host.flush();

	const comparisonRequest = factory.records[1].requests[0];
	expect(comparisonRequest?.sessionConfig.exposureScheduling).toEqual(scheduling);
	expect(comparisonRequest?.sessionKey).toContain('exposure:chunked:8192:yield');
	expect(host.getState().comparisonSceneSurfaceIdentity?.controllerIdentity).toBe(
		comparisonRequest?.sessionKey
	);
});
```

- [ ] **Step 7: Parse query params on `/`**

In `viewer/src/routes/+page.svelte`, import:

```ts
import { parseExposureSchedulingFromSearchParams } from '$lib/compute/gpu/exposureScheduling';
```

Add the reactive parse near `utciRenderDiagnosticsEnabled`:

```svelte
$: exposureScheduling = parseExposureSchedulingFromSearchParams($page.url.searchParams);
```

In the reactive `liveRouteHost.setRouteInputs({ ... })` object, add `exposureScheduling` after `gridResolutionMeters`:

```ts
$: liveRouteHost.setRouteInputs({
	enabled: useLiveUtciOnMainRoute,
	analysisId,
	baseAnalysis: $analysisStore,
	baseModel: modelLoading ? null : model,
	selection: {
		monthIndex: selectedMonthIndex,
		hourIndex: selectedHourIndex,
		timeIndex: selectedTimeIndex,
		selectionKey: [analysisId, selectedMonthIndex, selectedHourIndex].join("|"),
	},
	gridResolutionMeters: selectedGridResolutionMeters,
	exposureScheduling,
	colorMode: $viewerStore.colorMode,
	utciRenderMode,
	rendererBackend,
	rendererDevice: rendererDeviceForMain,
	utciSurfaceBackend: resolvedUtciSurfaceBackend,
	comparison: {
		active: isComparing,
		analysisId: $comparisonStore.comparisonAnalysisId,
		sourceAnalysis: $comparisonStore.isLoading ? null : $comparisonAnalysis,
		model: comparisonModelForLiveCompute,
		rendererDevice: rendererDeviceForMain,
	},
});
```

Add `exposureScheduling` to the `updateUtciRenderDiagnostics({ ... })` call after `baseMetadataGridSize`:

```ts
updateUtciRenderDiagnostics({
	enabled: utciRenderDiagnosticsEnabled,
	utciOnDemand: utciOnDemandMode,
	utciRenderRequested: utciRenderMode,
	utciRenderResolved: resolvedUtciSurfaceBackend,
	rendererBackend,
	rendererRequiredLimits,
	rendererDeviceLimits,
	liveRouteState,
	lastBaseGpuResidentCopyFailure,
	baseLiveReady,
	comparisonLiveReady,
	selectedMonthIndex,
	selectedHourIndex,
	selectedTimeIndex,
	baseColorMode: $viewerStore.colorMode,
	basePointCount: liveRouteState.base.analysis?.metadata.num_positions ?? null,
	baseMetadataGridSize: liveRouteState.base.analysis?.metadata.grid_size ?? null,
	exposureScheduling,
	baseSceneRenderContextTimeIndex: baseSceneRenderContext?.timeIndex,
	baseAcceptedUtciRange: basePendingGpuResidentOutput?.utciRange ?? undefined,
	tooltipInteraction: tooltipInteractionDiagnostics,
	cameraInteraction: cameraInteractionDiagnostics,
	timingsOverride: mainRouteRenderPublicationProjectionTracker.apply({
		enabled: useLiveUtciOnMainRoute,
		timings: liveRouteState.base.runtimeDiagnostics?.timings,
		projectedSceneSurfaceIdentity: baseSceneSurfaceIdentity,
		publishedSurfaceIdentity: liveRouteState.baseSurfaceIdentity,
		sceneRenderContextTimeIndex: baseSceneRenderContext?.timeIndex,
		selectedTimeIndex,
	}),
});
```

In `viewer/src/routes/main/liveSelectedHour.ts`, import the scheduler type:

```ts
import type { ExposureSchedulingOptions } from '$lib/compute/gpu/exposureScheduling';
```

Add it to `MainRouteLiveSelectedHourDiagnosticsParams`:

```ts
exposureScheduling?: ExposureSchedulingOptions;
```

Forward it from `buildMainRouteLiveSelectedHourDiagnosticsInputs(...)`:

```ts
exposureScheduling: params.exposureScheduling,
```

In `viewer/src/lib/diagnostics/mainRouteUtciDiagnostics.ts`, import the scheduler type:

```ts
import type { ExposureSchedulingOptions } from '$lib/compute/gpu/exposureScheduling';
```

Add it to both `MainRouteUtciDiagnosticsPayload` and `MainRouteUtciDiagnosticsInputs`:

```ts
exposureScheduling?: ExposureSchedulingOptions;
```

Add it to the object returned by `buildMainRouteUtciDiagnostics(...)`:

```ts
exposureScheduling: inputs.exposureScheduling
	? { ...inputs.exposureScheduling }
	: undefined,
```

- [ ] **Step 8: Run threaded-option tests**

Run:

```powershell
cd viewer
npm run test -- --run tests/compute/compute-manager-on-demand.test.ts tests/compute/live-selected-hour-session.test.ts tests/compute/live-selected-hour-route-host.test.ts tests/compute/exposureScheduling.test.ts
```

Expected: PASS.

---

## Task 3: Add Chunked Exposure Execution In The WebGPU Pipeline

**Files:**
- Modify: `viewer/src/lib/compute/gpu/webgpuUtciPipeline.ts`
- Modify: `viewer/src/lib/compute/on-demand/onDemandDiagnostics.ts`
- Test: `viewer/tests/compute/webgpuUtciPipeline.behavior.test.ts`
- Test: `viewer/tests/compute/webgpu-on-demand-source-locks.test.ts`

- [ ] **Step 1: Add diagnostics fields**

In `viewer/src/lib/compute/on-demand/onDemandDiagnostics.ts`, add optional timing fields. Use `scheduler` for submission slices and `pointDispatch` for shader dispatch chunks so evidence review does not confuse existing WebGPU point chunks with the new scheduler slices:

```ts
exposureSchedulerMode?: 'single-submit' | 'chunked';
exposureSchedulerSliceCount?: number;
exposurePointDispatchChunkCount?: number;
exposureSchedulerMaxWorkgroupsPerSlice?: number;
exposureSchedulerQueueWaitTotalMs?: number;
exposureSchedulerQueueWaitMaxMs?: number;
exposureSchedulerQueueWaitMinMs?: number;
exposureSchedulerYieldCount?: number;
exposureSchedulerSubmitCount?: number;
```

- [ ] **Step 2: Write behavior tests for default and chunked submissions**

In `viewer/tests/compute/webgpuUtciPipeline.behavior.test.ts`, extend the fake device helper so it can run exposure precompute without real WebGPU:

```ts
function createFakeComputePass() {
	return {
		setPipeline: vi.fn(),
		setBindGroup: vi.fn(),
		dispatchWorkgroups: vi.fn(),
		end: vi.fn()
	};
}

function createFakePipeline() {
	return {
		getBindGroupLayout: vi.fn(() => ({}))
	};
}

function createFakeDeviceForExposure() {
	const pass = createFakeComputePass();
	return {
		limits: {
			maxStorageBuffersPerShaderStage: 8
		},
		queue: {
			writeBuffer: vi.fn(),
			submit: vi.fn(),
			onSubmittedWorkDone: vi.fn().mockResolvedValue(undefined)
		},
		createBuffer: vi.fn(({ size }: { size: number }) => createFakeBuffer(size)),
		createBindGroup: vi.fn(() => ({})),
		createCommandEncoder: vi.fn(() => ({
			beginComputePass: vi.fn(() => pass),
			finish: vi.fn(() => ({}))
		})),
		pass
	};
}
```

Add tests:

```ts
it('uses one queue submit for default exposure scheduling', async () => {
	const device = createFakeDeviceForExposure();
	const pipeline = new __TEST_ONLY_WebgpuUtciComputePipeline(device as any, false) as any;
	pipeline.solarPipeline = createFakePipeline();
	pipeline.skyPipeline = createFakePipeline();

	await pipeline.uploadStaticData({
		gridPoints: new Float32Array(1024 * 3),
		sunVectors: new Float32Array([1, 0, 0]),
		sunAltitudes: new Float32Array([0.5]),
		weather: new Float32Array([1, 2, 3, 4, 5, 6, 7]),
		domeVectors: new Float32Array(145 * 3),
		domeWeights: new Float32Array(145),
		serializedBvh: {
			bvhNodeBuffer: new ArrayBuffer(32),
			bvhIndexBuffer: new ArrayBuffer(4),
			vertexBuffer: new Float32Array([0, 0, 0]),
			indexBuffer: new Uint32Array([0])
		}
	});

	await pipeline.runExposurePrecompute({
		numPoints: 1024,
		numHours: 1,
		numMonths: 1
	});

	expect(device.queue.submit).toHaveBeenCalledTimes(1);
	expect(device.queue.onSubmittedWorkDone).toHaveBeenCalledTimes(1);
	expect(pipeline.getOnDemandDiagnostics().timings).toMatchObject({
		exposureSchedulerMode: 'single-submit',
		exposureSchedulerSliceCount: 1,
		exposureSchedulerSubmitCount: 1
	});
});

it('uses one queue submit per scheduler slice in chunked exposure mode', async () => {
	const device = createFakeDeviceForExposure();
	const pipeline = new __TEST_ONLY_WebgpuUtciComputePipeline(device as any, false) as any;
	pipeline.solarPipeline = createFakePipeline();
	pipeline.skyPipeline = createFakePipeline();

	await pipeline.uploadStaticData({
		gridPoints: new Float32Array(1024 * 3),
		sunVectors: new Float32Array([1, 0, 0]),
		sunAltitudes: new Float32Array([0.5]),
		weather: new Float32Array([1, 2, 3, 4, 5, 6, 7]),
		domeVectors: new Float32Array(145 * 3),
		domeWeights: new Float32Array(145),
		serializedBvh: {
			bvhNodeBuffer: new ArrayBuffer(32),
			bvhIndexBuffer: new ArrayBuffer(4),
			vertexBuffer: new Float32Array([0, 0, 0]),
			indexBuffer: new Uint32Array([0])
		}
	});

	await pipeline.runExposurePrecompute({
		numPoints: 1024,
		numHours: 1,
		numMonths: 1,
		exposureScheduling: {
			mode: 'chunked',
			maxWorkgroupsPerSlice: 4,
			yieldBetweenSlices: false
		}
	});

	expect(device.queue.submit).toHaveBeenCalledTimes(4);
	expect(device.queue.onSubmittedWorkDone).toHaveBeenCalledTimes(4);
	expect(pipeline.getOnDemandDiagnostics().timings).toMatchObject({
		exposureSchedulerMode: 'chunked',
		exposureSchedulerSliceCount: 4,
		exposureSchedulerSubmitCount: 4,
		exposureSchedulerYieldCount: 0,
		exposureSchedulerMaxWorkgroupsPerSlice: 4
	});
});

it('stops chunked exposure before submitting later slices when aborted', async () => {
	const controller = new AbortController();
	const device = createFakeDeviceForExposure();
	let waitCount = 0;
	device.queue.onSubmittedWorkDone.mockImplementation(async () => {
		waitCount += 1;
		if (waitCount === 1) controller.abort();
	});
	const pipeline = new __TEST_ONLY_WebgpuUtciComputePipeline(device as any, false) as any;
	pipeline.solarPipeline = createFakePipeline();
	pipeline.skyPipeline = createFakePipeline();

	await pipeline.uploadStaticData({
		gridPoints: new Float32Array(1024 * 3),
		sunVectors: new Float32Array([1, 0, 0]),
		sunAltitudes: new Float32Array([0.5]),
		weather: new Float32Array([1, 2, 3, 4, 5, 6, 7]),
		domeVectors: new Float32Array(145 * 3),
		domeWeights: new Float32Array(145),
		serializedBvh: {
			bvhNodeBuffer: new ArrayBuffer(32),
			bvhIndexBuffer: new ArrayBuffer(4),
			vertexBuffer: new Float32Array([0, 0, 0]),
			indexBuffer: new Uint32Array([0])
		}
	});

	await expect(
		pipeline.runExposurePrecompute({
			numPoints: 1024,
			numHours: 1,
			numMonths: 1,
			signal: controller.signal,
			exposureScheduling: {
				mode: 'chunked',
				maxWorkgroupsPerSlice: 4,
				yieldBetweenSlices: true
			}
		})
	).rejects.toMatchObject({ name: 'AbortError' });
	expect(device.queue.submit).toHaveBeenCalledTimes(1);
	expect(device.queue.onSubmittedWorkDone).toHaveBeenCalledTimes(1);
});
```

Run:

```powershell
cd viewer
npm run test -- --run tests/compute/webgpuUtciPipeline.behavior.test.ts tests/compute/webgpu-on-demand-source-locks.test.ts
```

Expected: FAIL before implementation. The behavior tests fail because `ExposurePrecomputeParams` does not accept `exposureScheduling` or `signal`, chunked mode does not submit once per scheduler slice, abort cannot stop later slices, and scheduler timing fields are not recorded. The source-lock tests fail because `runChunkedExposurePrecompute`, `buildExposurePointSlices`, and `yieldToBrowserFrame` are not wired yet.

- [ ] **Step 3: Add narrow source-lock tests for branch presence**

In `viewer/tests/compute/webgpu-on-demand-source-locks.test.ts`, add assertions:

```ts
it('keeps default exposure precompute on the single-submit path', () => {
	expect(exposureMethodSource.includes("mode === 'chunked'")).toBe(true);
	expect(exposureMethodSource.includes('runChunkedExposurePrecompute')).toBe(true);
	expect(exposureMethodSource.includes('this.queue.submit([encoder.finish()])')).toBe(true);
});

it('chunked exposure keeps bounded scheduling helpers wired', () => {
	const chunkedSource = getSection(
		'private async runChunkedExposurePrecompute',
		'\n\n\tasync runExposurePrecompute'
	);
	expect(chunkedSource.includes('buildExposurePointSlices')).toBe(true);
	expect(chunkedSource.includes('await yieldToBrowserFrame')).toBe(true);
	expect(chunkedSource.includes('assertExposurePrecomputeActive')).toBe(true);
	expect(chunkedSource.includes('maxWorkgroupsPerSlice')).toBe(true);
	expect(chunkedSource.includes('exposureSchedulerQueueWaitMaxMs')).toBe(true);
});
```

These source locks are not the main proof; the behavior tests above are.

- [ ] **Step 4: Split pass encoding by caller-provided chunks**

In `viewer/src/lib/compute/gpu/webgpuUtciPipeline.ts`, import:

```ts
import {
	DEFAULT_EXPOSURE_SCHEDULING,
	buildExposurePointSlices,
	type ExposureSchedulingOptions
} from '$lib/compute/gpu/exposureScheduling';
import type { PointDispatchChunk } from '$lib/compute/gpu/gpu-pipeline';
```

Refactor the existing `encodeExposurePasses` into a helper accepting prebuilt chunks. The helper body is the current solar/sky pass encoding with the local `createPointDispatchChunks(...)` call replaced by the caller-provided `pointChunks` array:

```ts
private async encodeExposurePassesForChunks(params: {
	encoder: GPUCommandEncoder;
	numPoints: number;
	totalTimeSteps: number;
	pointChunks: PointDispatchChunk[];
	solarPipeline: GPUComputePipeline;
	skyPipeline: GPUComputePipeline;
	daylightTimeSteps?: number;
}): Promise<{ transientUniformBuffers: GPUBuffer[]; trace: ExposureEncodeTrace }> {
	const {
		encoder,
		numPoints,
		totalTimeSteps,
		pointChunks,
		solarPipeline,
		skyPipeline,
		daylightTimeSteps
	} = params;
	const commandEncodeStartedAt = performance.now();
	const scheduledPointCount = pointChunks.reduce((sum, chunk) => sum + chunk.pointCount, 0);
	const transientUniformBuffers: GPUBuffer[] = [];
	const hasBvh =
		this.bvhNodeBuffer && this.bvhIndexBuffer && this.bvhVertexBuffer && this.bvhParamsBuffer;
	let solarEncodeMs: number | undefined;
	let skyEncodeMs: number | undefined;
	let solarDispatchCount = 0;
	let skyDispatchCount = 0;

	if (hasBvh && this.gridPointsBuffer && this.sunVectorsBuffer && this.solarExposureBuffer) {
		const solarEncodeStartedAt = performance.now();
		this.ranExposurePassesThisRun = true;
		const solarPass = encoder.beginComputePass();
		solarPass.setPipeline(solarPipeline);
		solarPass.setBindGroup(1, this.createBvhBindGroup(solarPipeline));
		for (const chunk of pointChunks) {
			const solarParamsBuffer = this.createUintParamsBuffer(
				new Uint32Array([numPoints, totalTimeSteps, chunk.pointOffset, 0]),
				transientUniformBuffers
			);
			const solarBindGroup0 = this.device.createBindGroup({
				layout: solarPipeline.getBindGroupLayout(0),
				entries: [
					{ binding: 0, resource: { buffer: this.gridPointsBuffer } },
					{ binding: 1, resource: { buffer: this.sunVectorsBuffer } },
					{ binding: 2, resource: { buffer: this.solarExposureBuffer } },
					{ binding: 3, resource: { buffer: solarParamsBuffer } }
				]
			});
			solarPass.setBindGroup(0, solarBindGroup0);
			solarPass.dispatchWorkgroups(chunk.workgroupsX, totalTimeSteps, 1);
			solarDispatchCount += 1;
		}
		solarPass.end();
		solarEncodeMs = performance.now() - solarEncodeStartedAt;
	}

	const numPatches = 145;
	if (
		hasBvh &&
		this.gridPointsBuffer &&
		this.domeVectorsBuffer &&
		this.domeWeightsBuffer &&
		this.skyExposureBuffer
	) {
		const skyEncodeStartedAt = performance.now();
		this.ranExposurePassesThisRun = true;
		const skyPass = encoder.beginComputePass();
		skyPass.setPipeline(skyPipeline);
		skyPass.setBindGroup(1, this.createBvhBindGroup(skyPipeline));
		for (const chunk of pointChunks) {
			const skyParamsBuffer = this.createUintParamsBuffer(
				new Uint32Array([numPoints, numPatches, chunk.pointOffset, 0]),
				transientUniformBuffers
			);
			const skyBindGroup0 = this.device.createBindGroup({
				layout: skyPipeline.getBindGroupLayout(0),
				entries: [
					{ binding: 0, resource: { buffer: this.gridPointsBuffer } },
					{ binding: 1, resource: { buffer: this.domeVectorsBuffer } },
					{ binding: 2, resource: { buffer: this.domeWeightsBuffer } },
					{ binding: 3, resource: { buffer: this.skyExposureBuffer } },
					{ binding: 4, resource: { buffer: skyParamsBuffer } }
				]
			});
			skyPass.setBindGroup(0, skyBindGroup0);
			skyPass.dispatchWorkgroups(chunk.workgroupsX, 1, 1);
			skyDispatchCount += 1;
		}
		skyPass.end();
		skyEncodeMs = performance.now() - skyEncodeStartedAt;
	}

	return {
		transientUniformBuffers,
		trace: {
			commandEncodeTotalMs: performance.now() - commandEncodeStartedAt,
			solarEncodeMs,
			skyEncodeMs,
			pointChunks: pointChunks.length,
			solarDispatchCount,
			skyDispatchCount,
			solarRayBudget:
				solarDispatchCount > 0
					? scheduledPointCount * (daylightTimeSteps ?? totalTimeSteps)
					: 0,
			skyRayBudget: skyDispatchCount > 0 ? scheduledPointCount * numPatches : 0
		}
	};
}
```

Keep `encodeExposurePasses` as the default wrapper:

```ts
private async encodeExposurePasses(params: {
	encoder: GPUCommandEncoder;
	numPoints: number;
	totalTimeSteps: number;
	workgroupSize: number;
	solarPipeline: GPUComputePipeline;
	skyPipeline: GPUComputePipeline;
	daylightTimeSteps?: number;
}) {
	return this.encodeExposurePassesForChunks({
		...params,
		pointChunks: createPointDispatchChunks(params.numPoints, params.workgroupSize)
	});
}
```

- [ ] **Step 5: Add a safe post-frame yield helper**

Add this top-level helper in `webgpuUtciPipeline.ts`. The `setTimeout` inside the rAF callback prevents the async continuation from immediately submitting the next GPU chunk in the same pre-paint turn:

```ts
function yieldToBrowserFrame(): Promise<void> {
	return new Promise((resolve) => {
		if (typeof globalThis.requestAnimationFrame === 'function') {
			globalThis.requestAnimationFrame(() => {
				setTimeout(resolve, 0);
			});
			return;
		}
		setTimeout(resolve, 0);
	});
}
```

- [ ] **Step 6: Add abort/dispose guards**

In `viewer/src/lib/compute/gpu/webgpuUtciPipeline.ts`, add this helper near `yieldToBrowserFrame`:

```ts
function createExposureAbortError(message: string): Error {
	if (typeof DOMException === 'function') {
		return new DOMException(message, 'AbortError');
	}
	const error = new Error(message);
	error.name = 'AbortError';
	return error;
}
```

Add a disposed flag to `WebgpuUtciComputePipeline`:

```ts
private disposed = false;
```

Set it in `dispose()` before destroying buffers:

```ts
this.disposed = true;
```

Add this private guard method:

```ts
private assertExposurePrecomputeActive(signal?: AbortSignal): void {
	if (signal?.aborted) {
		throw createExposureAbortError('WebGPU UTCI exposure precompute aborted');
	}
	if (this.disposed) {
		throw createExposureAbortError('WebGPU UTCI exposure precompute disposed');
	}
}
```

Call this guard before every chunk submit, after every `queue.onSubmittedWorkDone()`, and after every browser yield. This is required so a stale selected-hour session cannot continue feeding the WebGPU queue after route/session replacement.

- [ ] **Step 7: Implement chunked exposure precompute branch**

Add a private method before `runExposurePrecompute`:

```ts
private async runChunkedExposurePrecompute(params: {
	numPoints: number;
	totalTimeSteps: number;
	solarPipeline: GPUComputePipeline;
	skyPipeline: GPUComputePipeline;
	exposureScheduling: ExposureSchedulingOptions;
	signal?: AbortSignal;
}): Promise<{
	trace: ExposureEncodeTrace;
	queueWaitTotalMs: number;
	queueWaitMaxMs: number;
	queueWaitMinMs: number;
	yieldCount: number;
	submitCount: number;
	sliceCount: number;
}> {
	const workgroupSize = 64;
	const pointSlices = buildExposurePointSlices({
		numPoints: params.numPoints,
		workgroupSize,
		maxWorkgroupsPerSlice: params.exposureScheduling.maxWorkgroupsPerSlice
	});
	let mergedTrace: ExposureEncodeTrace = {
		commandEncodeTotalMs: 0,
		pointChunks: 0,
		solarDispatchCount: 0,
		skyDispatchCount: 0,
		solarRayBudget: 0,
		skyRayBudget: 0
	};
	let queueWaitTotalMs = 0;
	let queueWaitMaxMs = 0;
	let queueWaitMinMs = Number.POSITIVE_INFINITY;
	let yieldCount = 0;
	let submitCount = 0;

	for (let index = 0; index < pointSlices.length; index += 1) {
		this.assertExposurePrecomputeActive(params.signal);
		const slice = pointSlices[index];
		const encoder = this.device.createCommandEncoder();
		const { transientUniformBuffers, trace } = await this.encodeExposurePassesForChunks({
			encoder,
			numPoints: params.numPoints,
			totalTimeSteps: params.totalTimeSteps,
			pointChunks: [slice],
			solarPipeline: params.solarPipeline,
			skyPipeline: params.skyPipeline,
			daylightTimeSteps: this.lastDaylightTimeStepCount ?? undefined
		});

		mergedTrace = {
			commandEncodeTotalMs: mergedTrace.commandEncodeTotalMs + trace.commandEncodeTotalMs,
			solarEncodeMs: (mergedTrace.solarEncodeMs ?? 0) + (trace.solarEncodeMs ?? 0),
			skyEncodeMs: (mergedTrace.skyEncodeMs ?? 0) + (trace.skyEncodeMs ?? 0),
			pointChunks: mergedTrace.pointChunks + trace.pointChunks,
			solarDispatchCount: mergedTrace.solarDispatchCount + trace.solarDispatchCount,
			skyDispatchCount: mergedTrace.skyDispatchCount + trace.skyDispatchCount,
			solarRayBudget: mergedTrace.solarRayBudget + trace.solarRayBudget,
			skyRayBudget: mergedTrace.skyRayBudget + trace.skyRayBudget
		};

		try {
			this.assertExposurePrecomputeActive(params.signal);
			this.queue.submit([encoder.finish()]);
			submitCount += 1;
			const waitStartedAt = performance.now();
			await this.queue.onSubmittedWorkDone();
			const waitMs = performance.now() - waitStartedAt;
			queueWaitTotalMs += waitMs;
			queueWaitMaxMs = Math.max(queueWaitMaxMs, waitMs);
			queueWaitMinMs = Math.min(queueWaitMinMs, waitMs);
			this.assertExposurePrecomputeActive(params.signal);
		} finally {
			this.destroyTransientUniformBuffers(transientUniformBuffers);
		}

		if (params.exposureScheduling.yieldBetweenSlices && index < pointSlices.length - 1) {
			yieldCount += 1;
			await yieldToBrowserFrame();
			this.assertExposurePrecomputeActive(params.signal);
		}
	}

	return {
		trace: mergedTrace,
		queueWaitTotalMs,
		queueWaitMaxMs,
		queueWaitMinMs: Number.isFinite(queueWaitMinMs) ? queueWaitMinMs : 0,
		yieldCount,
		submitCount,
		sliceCount: pointSlices.length
	};
}
```

- [ ] **Step 8: Branch inside `runExposurePrecompute`**

Inside `runExposurePrecompute`, replace the current single-path encoder/submit section from `this.ranExposurePassesThisRun = false;` through the `timings` object with this branch structure. Keep the existing allocation and `lastConfig` code after the timings object unchanged:

```ts
const exposureScheduling = params.exposureScheduling ?? DEFAULT_EXPOSURE_SCHEDULING;
this.assertExposurePrecomputeActive(params.signal);

this.ranExposurePassesThisRun = false;
const exposurePrecomputeStart = performance.now();
let exposureEncodeTrace: ExposureEncodeTrace;
let exposureQueueWaitMs = 0;
let exposureSchedulerSliceCount = 1;
let exposureSchedulerQueueWaitTotalMs: number | undefined;
let exposureSchedulerQueueWaitMaxMs: number | undefined;
let exposureSchedulerQueueWaitMinMs: number | undefined;
let exposureSchedulerYieldCount = 0;
let exposureSchedulerSubmitCount = 1;

if (exposureScheduling.mode === 'chunked') {
	const chunkedResult = await this.runChunkedExposurePrecompute({
		numPoints,
		totalTimeSteps,
		solarPipeline,
		skyPipeline,
		exposureScheduling,
		signal: params.signal
	});
	exposureEncodeTrace = chunkedResult.trace;
	exposureQueueWaitMs = chunkedResult.queueWaitTotalMs;
	exposureSchedulerSliceCount = chunkedResult.sliceCount;
	exposureSchedulerQueueWaitTotalMs = chunkedResult.queueWaitTotalMs;
	exposureSchedulerQueueWaitMaxMs = chunkedResult.queueWaitMaxMs;
	exposureSchedulerQueueWaitMinMs = chunkedResult.queueWaitMinMs;
	exposureSchedulerYieldCount = chunkedResult.yieldCount;
	exposureSchedulerSubmitCount = chunkedResult.submitCount;
} else {
	const encoder = this.device.createCommandEncoder();
	const {
		transientUniformBuffers: exposureUniformBuffers,
		trace
	} = await this.encodeExposurePasses({
		encoder,
		numPoints,
		totalTimeSteps,
		workgroupSize: 64,
		solarPipeline,
		skyPipeline,
		daylightTimeSteps: this.lastDaylightTimeStepCount ?? undefined
	});
	exposureEncodeTrace = trace;
	try {
		this.assertExposurePrecomputeActive(params.signal);
		this.queue.submit([encoder.finish()]);
		const exposureQueueWaitStartedAt = performance.now();
		await this.queue.onSubmittedWorkDone();
		exposureQueueWaitMs = performance.now() - exposureQueueWaitStartedAt;
		this.assertExposurePrecomputeActive(params.signal);
	} finally {
		this.destroyTransientUniformBuffers(exposureUniformBuffers);
	}
}

const exposurePrecomputeMs = performance.now() - exposurePrecomputeStart;
const solarExposureBytes = Math.ceil((numPoints * numHours * numMonths) / 32) * 4;
const skyExposureBytes = numPoints * 4;
this.onDemandDiagnostics = {
	...mergeTrackedGpuAllocationBytes(
		{
			...this.onDemandDiagnostics,
			path: 'exposure-only-f32',
			timeIndices: [],
			usedRunAllForSelectedHour: false,
			usedExposureOnlyPrecompute: true,
			allHoursUtciBytesAllocated: 0,
			allHoursMrtBytesAllocated: 0,
			oneHourOutputBytes: 0
		},
		{
			persistentExposureBytes: solarExposureBytes + skyExposureBytes,
			allHoursOutputBytes: 0
		}
	),
	timings: {
		...this.onDemandDiagnostics.timings,
		exposurePrecomputeMs,
		exposureWeatherBufferEnsureMs,
		exposureCommandEncodeTotalMs: exposureEncodeTrace.commandEncodeTotalMs,
		exposureSolarEncodeMs: exposureEncodeTrace.solarEncodeMs,
		exposureSkyEncodeMs: exposureEncodeTrace.skyEncodeMs,
		exposureQueueWaitMs,
		exposurePointCount: numPoints,
		exposureTotalTimeSteps: totalTimeSteps,
		exposureDaylightTimeSteps: this.lastDaylightTimeStepCount ?? undefined,
		exposurePointChunks: exposureEncodeTrace.pointChunks,
		exposureSolarDispatchCount: exposureEncodeTrace.solarDispatchCount,
		exposureSkyDispatchCount: exposureEncodeTrace.skyDispatchCount,
		exposureSolarRayBudget: exposureEncodeTrace.solarRayBudget,
		exposureSkyRayBudget: exposureEncodeTrace.skyRayBudget,
		exposureSchedulerMode: exposureScheduling.mode,
		exposureSchedulerSliceCount,
		exposurePointDispatchChunkCount: exposureEncodeTrace.pointChunks,
		exposureSchedulerMaxWorkgroupsPerSlice: exposureScheduling.maxWorkgroupsPerSlice,
		exposureSchedulerQueueWaitTotalMs: exposureSchedulerQueueWaitTotalMs ?? exposureQueueWaitMs,
		exposureSchedulerQueueWaitMaxMs: exposureSchedulerQueueWaitMaxMs ?? exposureQueueWaitMs,
		exposureSchedulerQueueWaitMinMs: exposureSchedulerQueueWaitMinMs ?? exposureQueueWaitMs,
		exposureSchedulerYieldCount,
		exposureSchedulerSubmitCount
	}
};
```

- [ ] **Step 9: Run behavior, source-lock, and targeted tests**

Run:

```powershell
cd viewer
npm run test -- --run tests/compute/webgpuUtciPipeline.behavior.test.ts tests/compute/webgpu-on-demand-source-locks.test.ts tests/compute/exposureScheduling.test.ts
```

Expected: PASS.

---

## Task 4: Extend The Freeze And Scrub Collectors To Compare Control Vs Chunked

**Files:**
- Modify: `viewer/tests/e2e/main-route-visual-freeze-map.spec.ts`
- Modify: `viewer/tests/e2e/main-route-cold-start-waterfall.spec.ts`

- [ ] **Step 1: Add optional query params to case config**

Extend `AnalysisCase`:

```ts
queryParams?: Record<string, string>;
```

Add a second NZ 0.5m case:

```ts
{
	caseId: 'ness-tziona-0_5m-chunked-8192',
	projectLabel: 'Ness-Tziona',
	analysisId: 'Ness-Tziona/exploded/nes_tziona_unblock_2',
	expectedSelectionKey: 'Ness-Tziona/exploded/nes_tziona_unblock_2|7|0',
	gridResolutionMeters: 0.5,
	queryParams: {
		utciExposureSchedule: 'chunked',
		utciExposureMaxWorkgroupsPerSlice: '8192'
	}
}
```

- [ ] **Step 2: Append case query params in `buildSourceUrl`**

Update `buildSourceUrl`:

```ts
for (const [key, value] of Object.entries(caseConfig.queryParams ?? {})) {
	params.set(key, value);
}
```

- [ ] **Step 3: Persist scheduler summary fields**

Ensure `summarizeDiagnostics` includes:

```ts
exposureSchedulerMode: value.timings?.exposureSchedulerMode ?? null,
exposureSchedulerSliceCount: numberOrNull(value.timings?.exposureSchedulerSliceCount),
exposurePointDispatchChunkCount: numberOrNull(value.timings?.exposurePointDispatchChunkCount),
exposureSchedulerMaxWorkgroupsPerSlice: numberOrNull(
	value.timings?.exposureSchedulerMaxWorkgroupsPerSlice
),
exposureSchedulerQueueWaitTotalMs: numberOrNull(
	value.timings?.exposureSchedulerQueueWaitTotalMs
),
exposureSchedulerQueueWaitMaxMs: numberOrNull(value.timings?.exposureSchedulerQueueWaitMaxMs),
exposureSchedulerYieldCount: numberOrNull(value.timings?.exposureSchedulerYieldCount),
exposureSchedulerSubmitCount: numberOrNull(value.timings?.exposureSchedulerSubmitCount)
```

- [ ] **Step 4: Add the same chunked case to the cold-start waterfall collector**

In `viewer/tests/e2e/main-route-cold-start-waterfall.spec.ts`, extend `AnalysisCase`:

```ts
queryParams?: Record<string, string>;
```

Add a second NZ 0.5m case immediately after the existing NZ 0.5m control case:

```ts
{
	projectLabel: 'Ness-Tziona',
	analysisId: 'Ness-Tziona/exploded/nes_tziona_unblock_2',
	metadataPath: 'data/analyses/Ness-Tziona/exploded/nes_tziona_unblock_2.json',
	expectedSelectionKey: 'Ness-Tziona/exploded/nes_tziona_unblock_2|7|0',
	gridResolutionMeters: 0.5,
	queryParams: {
		utciExposureSchedule: 'chunked',
		utciExposureMaxWorkgroupsPerSlice: '8192'
	}
}
```

Add a `caseId` field to `CollectedColdCase` so the artifact can distinguish control vs chunked:

```ts
caseId: string;
```

Build `caseId` in `collectCase`:

```ts
const caseId = [
	caseConfig.projectLabel.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, ''),
	`${caseConfig.gridResolutionMeters}m`.replace('.', '_'),
	caseConfig.queryParams?.utciExposureSchedule ?? 'single-submit'
].join('-');
```

Append query params in `collectCase` after the base params are built:

```ts
const params = new URLSearchParams({
	analysis: caseConfig.analysisId,
	utciRender: 'auto',
	utciRenderDiagnostics: '1'
});
if (caseConfig.gridResolutionMeters === 0.5) {
	params.set('gridResolution', String(caseConfig.gridResolutionMeters));
}
for (const [key, value] of Object.entries(caseConfig.queryParams ?? {})) {
	params.set(key, value);
}
const sourceUrl = `/?${params.toString()}`;
```

In `extractTimings`, add scheduler fields immediately after `exposurePointChunks` so initial and first-scrub phases persist the same timing names:

```ts
function exposureSchedulerModeOrNull(value: unknown): 'single-submit' | 'chunked' | null {
	return value === 'single-submit' || value === 'chunked' ? value : null;
}
```

```ts
function extractTimings(timings: Record<string, unknown> | undefined) {
	return {
		payloadPrepareMs: numberOrNull(timings?.payloadPrepareMs),
		workerBvhMs: numberOrNull(timings?.workerBvhMs),
		pipelineUploadMs: numberOrNull(timings?.pipelineUploadMs),
		exposurePrecomputeMs: numberOrNull(timings?.exposurePrecomputeMs),
		exposureCommandEncodeTotalMs: numberOrNull(timings?.exposureCommandEncodeTotalMs),
		exposureEncodeMs: numberOrNull(timings?.exposureEncodeMs),
		exposureSolarEncodeMs: numberOrNull(timings?.exposureSolarEncodeMs),
		exposureSkyEncodeMs: numberOrNull(timings?.exposureSkyEncodeMs),
		exposureQueueWaitMs: numberOrNull(timings?.exposureQueueWaitMs),
		exposurePointCount: numberOrNull(timings?.exposurePointCount),
		exposureTotalTimeSteps: numberOrNull(timings?.exposureTotalTimeSteps),
		exposureDaylightTimeSteps: numberOrNull(timings?.exposureDaylightTimeSteps),
		exposurePointChunks: numberOrNull(timings?.exposurePointChunks),
		exposureSchedulerMode: exposureSchedulerModeOrNull(timings?.exposureSchedulerMode),
		exposureSchedulerSliceCount: numberOrNull(timings?.exposureSchedulerSliceCount),
		exposurePointDispatchChunkCount: numberOrNull(timings?.exposurePointDispatchChunkCount),
		exposureSchedulerMaxWorkgroupsPerSlice: numberOrNull(
			timings?.exposureSchedulerMaxWorkgroupsPerSlice
		),
		exposureSchedulerQueueWaitTotalMs: numberOrNull(
			timings?.exposureSchedulerQueueWaitTotalMs
		),
		exposureSchedulerQueueWaitMaxMs: numberOrNull(
			timings?.exposureSchedulerQueueWaitMaxMs
		),
		exposureSchedulerQueueWaitMinMs: numberOrNull(
			timings?.exposureSchedulerQueueWaitMinMs
		),
		exposureSchedulerYieldCount: numberOrNull(timings?.exposureSchedulerYieldCount),
		exposureSchedulerSubmitCount: numberOrNull(timings?.exposureSchedulerSubmitCount),
		exposureSolarDispatchCount: numberOrNull(timings?.exposureSolarDispatchCount),
		exposureSkyDispatchCount: numberOrNull(timings?.exposureSkyDispatchCount),
		exposureSolarRayBudget: numberOrNull(timings?.exposureSolarRayBudget),
		exposureSkyRayBudget: numberOrNull(timings?.exposureSkyRayBudget),
		oneHourDispatchMs: numberOrNull(timings?.oneHourDispatchMs),
		firstSelectedHourReadyMs: numberOrNull(timings?.firstSelectedHourReadyMs),
		firstSelectedHourVisibleMs: numberOrNull(timings?.firstSelectedHourVisibleMs),
		renderUpdateMs: numberOrNull(timings?.renderUpdateMs),
		renderSceneSyncStartDelayMs: numberOrNull(timings?.renderSceneSyncStartDelayMs),
		renderSceneSyncTotalMs: numberOrNull(timings?.renderSceneSyncTotalMs),
		renderLayoutBuildMs: numberOrNull(timings?.renderLayoutBuildMs),
		renderSurfaceMeshMs: numberOrNull(timings?.renderSurfaceMeshMs),
		renderStorageInitWaitMs: numberOrNull(timings?.renderStorageInitWaitMs),
		renderBufferCopyMs: numberOrNull(timings?.renderBufferCopyMs),
		renderQueueDrainMs: numberOrNull(timings?.renderQueueDrainMs)
	};
}
```

- [ ] **Step 5: Run collector list checks**

Run:

```powershell
cd viewer
npx playwright test --config=playwright.collect.config.ts tests/e2e/main-route-visual-freeze-map.spec.ts --list
npx playwright test --config=playwright.collect.config.ts tests/e2e/main-route-cold-start-waterfall.spec.ts --list
```

Expected: one collector test listed for each file.

---

## Task 5: Run The Evidence Matrix And Decide The Evidence Outcome

**Files:**
- Modify: `data/performance-results/main-route-visual-freeze-map.json`
- Modify: `data/performance-results/main-route-cold-start-waterfall.json`
- Optional modify: `docs/performance/main-route-exposure-scheduler.md`

- [ ] **Step 1: Run the headed freeze-map collector**

Run:

```powershell
cd viewer
npx playwright test --config=playwright.collect.config.ts tests/e2e/main-route-visual-freeze-map.spec.ts --project=chromium --workers=1 --reporter=list --timeout=900000
```

Expected: PASS and write `data/performance-results/main-route-visual-freeze-map.json`.

- [ ] **Step 2: Run the cold-start waterfall scrub collector**

Run:

```powershell
cd viewer
npx playwright test --config=playwright.collect.config.ts tests/e2e/main-route-cold-start-waterfall.spec.ts --project=chromium --workers=1 --reporter=list --timeout=900000
```

Expected: PASS and write `data/performance-results/main-route-cold-start-waterfall.json` with both NZ 0.5m control and chunked cases.

- [ ] **Step 3: Parse control vs chunked NZ 0.5m freeze evidence**

Run:

```powershell
cd D:\Projects\Nur\Shade\fast-utci
@'
const fs = require('fs');
const artifact = JSON.parse(fs.readFileSync('data/performance-results/main-route-visual-freeze-map.json', 'utf8'));
for (const caseId of ['ness-tziona-0_5m', 'ness-tziona-0_5m-chunked-8192']) {
  const entry = artifact.cases.find((item) => item.caseId === caseId);
  const timing = entry.summary.finalTimingBuckets;
  console.log(caseId, {
    publicationReached: entry.publicationReached,
    firstSelectedHourVisibleMs: entry.summary.firstSelectedHourVisibleMs,
    topRafGapMs: entry.summary.topRafGapMs,
    topIntervalGapMs: entry.summary.topIntervalGapMs,
    topLongTaskMs: entry.summary.topLongTaskMs,
    exposurePrecomputeMs: timing.exposurePrecomputeMs,
    exposureQueueWaitMs: timing.exposureQueueWaitMs,
    exposureSchedulerMode: timing.exposureSchedulerMode,
    exposureSchedulerSliceCount: timing.exposureSchedulerSliceCount,
    exposureSchedulerQueueWaitMaxMs: timing.exposureSchedulerQueueWaitMaxMs,
    renderUpdateMs: timing.renderUpdateMs,
    ownedGpuMemoryBytes: entry.summary.ownedGpuMemoryBytes
  });
}
'@ | node -
```

Expected: print both cases and the scheduler fields for the chunked variant.

- [ ] **Step 4: Parse control vs chunked NZ 0.5m first-scrub evidence**

Run from repo root:

```powershell
@'
const fs = require('fs');
const artifact = JSON.parse(fs.readFileSync('data/performance-results/main-route-cold-start-waterfall.json', 'utf8'));
const nzCases = artifact.cases.filter((entry) =>
  entry.analysisId === 'Ness-Tziona/exploded/nes_tziona_unblock_2' &&
  entry.gridResolutionMeters === 0.5
);
for (const entry of nzCases) {
  const schedulerMode = entry.initial.timings.exposureSchedulerMode ?? 'single-submit';
  console.log(`${entry.caseId ?? entry.projectLabel}-${schedulerMode}`, {
    firstVisibleMs: entry.initial.firstVisibleMs,
    firstPostVisibleScrubMs: entry.firstPostVisibleScrub.visibleMs,
    scrubSurfaceRequestId: entry.firstPostVisibleScrub.surfaceRequestId,
    initialExposureQueueWaitMs: entry.initial.timings.exposureQueueWaitMs,
    initialSchedulerMode: entry.initial.timings.exposureSchedulerMode,
    scrubSchedulerMode: entry.firstPostVisibleScrub.timings.exposureSchedulerMode,
    memoryScope: entry.assertions.memoryScope
  });
}
'@ | node -
```

Expected: print at least two NZ 0.5m cases: one `single-submit` control and one `chunked` case, each with first visible and first post-visible scrub values.

- [ ] **Step 5: Apply pass, tune, or fail gates**

Keep chunked scheduling strictly query-gated throughout this plan. Use the gates below only to decide pass, tune, or fail evidence:

- `publicationReached === true`
- main-route GPU-native proof still holds
- NZ 0.5m `topRafGapMs` is below `500ms`
- `topIntervalGapMs` improves and does not stay in multi-second territory
- first selected-hour visible is not worse by more than `10%`
- first post-visible scrub in `main-route-cold-start-waterfall.json` is not worse by more than `10%`
- app-owned GPU memory does not increase beyond `10%`
- no page errors, request failures, crashes, or WebGPU device-lost events

If `topRafGapMs` improves but remains above `500ms`, or total time regresses while responsiveness improves, this is not an evidence pass. Keep the feature query-gated and tune one more slice size (`4096`, `16384`, or `32768`) before abandoning or proposing a second plan.

- [ ] **Step 6: Keep control and chunked explicit**

Do not infer chunked scheduling from `utciRenderDiagnostics=1`, `gridResolution=0.5`, or any other existing flag. That would silently destroy the collector's control case because both control and chunked evidence run with diagnostics enabled.

If all gates pass, leave the implementation query-gated and write a recommendation in the evidence note. Making chunked the product or diagnostics default requires a separate plan/review after this prototype evidence.

- [ ] **Step 7: Write a short evidence note**

Create `docs/performance/main-route-exposure-scheduler.md` only after the collector runs. Include:

```md
# Main Route Exposure Scheduler Evidence

## Baseline

- artifact: `data/performance-results/main-route-visual-freeze-map.json`
- scrub artifact: `data/performance-results/main-route-cold-start-waterfall.json`
- control case: `ness-tziona-0_5m`
- chunked case: `ness-tziona-0_5m-chunked-8192`

## Result

Write one row for `ness-tziona-0_5m` and one row for `ness-tziona-0_5m-chunked-8192` with these exact measured fields:

- first publication ms: `summary.firstSelectedHourVisibleMs` from `main-route-visual-freeze-map.json`
- first scrub ms: `firstPostVisibleScrub.visibleMs` from `main-route-cold-start-waterfall.json`
- top rAF gap ms: `summary.topRafGapMs`
- exposure queue ms: `summary.finalTimingBuckets.exposureQueueWaitMs`
- render update ms: `summary.finalTimingBuckets.renderUpdateMs`
- owned GPU MiB: `summary.ownedGpuMemoryBytes / 1048576`

## Decision

Keep query-gated / tune slice size / propose a separate default-behavior plan.
```

---

## Task 6: Final Verification And Review

**Files:**
- All changed files from Tasks 1-5.

- [ ] **Step 1: Run compile/type checks**

Run:

```powershell
cd viewer
npm run check
```

Expected: `svelte-check found 0 errors and 0 warnings`.

- [ ] **Step 2: Run focused unit tests**

Run:

```powershell
cd viewer
npm run test -- --run tests/compute/exposureScheduling.test.ts tests/compute/compute-manager-on-demand.test.ts tests/compute/live-selected-hour-session.test.ts tests/compute/live-selected-hour-route-host.test.ts tests/compute/webgpu-on-demand-source-locks.test.ts tests/compute/webgpuUtciPipeline.behavior.test.ts tests/diagnostics/main-route-utci-diagnostics.test.ts
```

Expected: PASS.

- [ ] **Step 3: Run selected-hour quality suite**

Run:

```powershell
cd viewer
npm run test:quality:selected-hour
```

Expected: PASS.

- [ ] **Step 4: Run headed freeze-map proof**

Run:

```powershell
cd viewer
npx playwright test --config=playwright.collect.config.ts tests/e2e/main-route-visual-freeze-map.spec.ts --project=chromium --workers=1 --reporter=list --timeout=900000
```

Expected: PASS and updated artifact.

- [ ] **Step 5: Run headed cold-start scrub proof**

Run:

```powershell
cd viewer
npx playwright test --config=playwright.collect.config.ts tests/e2e/main-route-cold-start-waterfall.spec.ts --project=chromium --workers=1 --reporter=list --timeout=900000
```

Expected: PASS and updated artifact containing NZ 0.5m control and chunked first post-visible scrub values.

- [ ] **Step 6: Run whitespace check**

Run:

```powershell
git diff --check
```

Expected: no whitespace errors. LF-to-CRLF warnings are acceptable if no whitespace errors are reported.

- [ ] **Step 7: Subagent review order**

Use SDD review exactly in this order:

1. Fresh spec-compliance reviewer:
   - Checks no commits/worktrees.
	- Checks default path remains unchanged throughout this plan; passing gates only supports an evidence recommendation.
   - Checks freeze-map artifact includes control and chunked cases.
   - Checks first visible, first scrub, rAF gaps, memory, and GPU-native proof boundaries.

2. Fresh code-quality reviewer, only after spec reviewer is clean:
   - Checks scheduler isolation and option naming.
   - Checks instrumentation overhead.
   - Checks stale request/session safety.
   - Checks no broad refactors or unrelated formatting.

Do not ask for a final completion claim until both reviewers are clean and verification commands have passed.

---

## Self-Review

- Spec coverage: The plan covers a query-gated tiled exposure scheduler, no-commit/no-worktree workflow, rAF/freeze proof, GPU-native boundaries, first visible, first scrub, and memory guardrails.
- Placeholder scan: No placeholder tokens or deferred-work phrases are present.
- Type consistency: `ExposureSchedulingOptions`, `ExposurePrecomputeParams.exposureScheduling`, and timing field names are consistent across tasks.
- Risk control: Default path remains `single-submit` throughout this plan; chunked mode is query-gated for evidence collection and any default/product promotion requires a separate plan.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-30-main-route-cooperative-exposure-scheduler.md`.

Recommended execution mode: Subagent-Driven Development.

- Fresh implementation subagent per task.
- After each file-changing task, run a fresh spec-compliance review first.
- Only after spec compliance is clean, run a fresh code-quality review.
- No commits.
- No git worktrees.
