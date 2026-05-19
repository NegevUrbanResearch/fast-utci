# Main Route Cold Start Waterfall Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a diagnostics-only cold-start waterfall for the main route so we can observe first-visible timing by discrete phase before choosing cold-path optimizations.

**Architecture:** Keep this as observability, not optimization. Add a cold-start timeline to the existing main-route diagnostics payload, stamp route/session/model boundaries only when `utciRenderDiagnostics=1`, collect a sibling cold-start artifact, and write a current-state evidence note separate from warm-scrub artifacts.

**Tech Stack:** Svelte 5, TypeScript, Vitest, Playwright, WebGPU, Three.js, existing `window.__utciRenderDiagnostics__` proof surface.

---

## Hard Constraints

- No commits.
- No git worktrees.
- Preserve unrelated dirty files.
- Do not change product overlay copy or UX in this plan.
- Do not implement performance optimizations in this plan.
- Keep the warm-path proof artifacts separate:
  - `docs/performance/main-route-selected-hour-render-diagnostics-next.md`
  - `data/performance-results/main-route-selected-hour-render-diagnostics-next.json`
- Create a sibling cold artifact instead of overwriting warm or old baseline artifacts:
  - `data/performance-results/main-route-cold-start-waterfall.json`
  - `docs/performance/main-route-cold-start-waterfall.md`

## Proof Boundary

The cold collector must continue to prove:

- route is `/`, not `/debug`
- `utciRenderResolved='gpuNative'`
- `utciSurfaceSource='compute-buffer-selected-hour'`
- `baseRenderTransport='compute-buffer-selected-hour'`
- `dataTextureBuildCount=0`
- `selectedHourRuntimeContract.route='main'`
- `selectedHourRuntimeContract.readbackInstrumentation='instrumented'`
- `selectedHourRuntimeContract.visibleSelectedHourReadbackCount=0`
- `selectedHourRuntimeContract.strongVisibleGpuPath=true`
- no `.bin`, Python reference, parity, or debug comparison requests

## File Structure

- Modify `viewer/src/lib/diagnostics/mainRouteUtciDiagnostics.ts`
  - Own the typed cold-start waterfall payload and defensive copies.
- Modify `viewer/src/routes/main/liveSelectedHour.ts`
  - Thread cold-start diagnostics through the existing main-route diagnostics builder.
- Modify `viewer/src/routes/+page.svelte`
  - Stamp route analysis/model/session lifecycle boundaries and pass them into diagnostics.
- Modify `viewer/src/routes/main/MainRouteViewport.svelte`
  - Bridge model cold-start timing callbacks from the route shell to `Model.svelte`.
- Modify `viewer/src/lib/components/scene/Model.svelte`
  - Emit model load/process timing events without changing visible loading behavior.
- Modify `viewer/src/lib/compute/selected-hour/liveUtciSelectedHourSession.ts`
  - Split cold session lifecycle timing into explicit phases already present in the code path.
- Modify `viewer/src/lib/compute/on-demand/onDemandDiagnostics.ts`
  - Add the shared nested cold-start timing contract used by runtime diagnostics.
- Modify `viewer/tests/diagnostics/main-route-utci-diagnostics.test.ts`
  - Unit-test cloning and publication of cold-start waterfall fields.
- Modify `viewer/tests/compute/live-selected-hour-session.test.ts`
  - Unit-test lifecycle timing fields on session preparation.
- Modify `viewer/tests/e2e/main-route-performance-baseline.spec.ts`
  - Collect 2m cold-start waterfall cases to `main-route-cold-start-waterfall.json`.
- Modify `viewer/tests/e2e/main-route-performance-0_5m.spec.ts`
  - Reuse extraction helpers or add cold-only collection mode for 0.5m initial cases.
- Create `docs/performance/main-route-cold-start-waterfall.md`
  - Summarize current cold evidence after collection.

## Cold Artifact Schema

Only `data/performance-results/main-route-cold-start-waterfall.json` may receive cold-start waterfall output in this plan. Do not write cold-start fields into:

- `data/performance-results/main-route-selected-hour-render-diagnostics-next.json`
- `data/performance-results/main-route-selected-hour-0_5m-base.json`
- `data/performance-results/main-route-selected-hour-current-head.json`

The cold artifact root must use this shape:

```ts
type MainRouteColdStartWaterfallArtifact = {
	collectedOn: string;
	sourceRoute: '/';
	collectionMethod: string;
	cases: Array<{
		projectLabel: string;
		analysisId: string;
		gridResolutionMeters: 2 | 0.5;
		colorMode: 'normalized' | 'discrete';
		phase: 'cold-initial';
		pointCount: number;
		firstSelectedHourVisibleAtMs: number | null;
		firstSelectedHourVisibleProvenance: string;
		sourceUrl: string;
		timings: Record<string, number | null>;
		coldStart: Record<string, number | null>;
		renderPublication: Record<string, unknown> | null;
		trackedGpuAllocationBytes: number | null;
		ownedGpuMemoryBytes: number | null;
		proof: Record<string, unknown>;
		assertions: {
			pythonBinDebugComparisonFieldsAbsent: true;
			forbiddenComparisonFieldsPresent: string[];
			forbiddenRequestUrls: string[];
			memoryScope: 'utci-owned-webgpu-buffers';
		};
	}>;
};
```

---

### Task 1: Add Typed Cold-Start Diagnostics Payload

**Files:**
- Modify: `viewer/src/lib/diagnostics/mainRouteUtciDiagnostics.ts`
- Modify: `viewer/src/lib/compute/on-demand/onDemandDiagnostics.ts`
- Test: `viewer/tests/diagnostics/main-route-utci-diagnostics.test.ts`

- [ ] **Step 1: Write the failing diagnostics unit test**

Add this test near the other `buildMainRouteUtciDiagnostics` tests:

```ts
it('defensively exposes main-route cold-start waterfall timings', () => {
	const coldStart = {
		routeAnalysisLoadStartedAtMs: 10,
		routeAnalysisLoadCompletedAtMs: 20,
		modelLoadStartedAtMs: 30,
		modelLoadCompletedAtMs: 40,
		modelProcessingStartedAtMs: 41,
		modelProcessingCompletedAtMs: 55,
		sessionPrepareStartedAtMs: 60,
		sessionPayloadPrepareStartedAtMs: 61,
		sessionPayloadPrepareCompletedAtMs: 70,
		sessionWorkerBvhStartedAtMs: 71,
		sessionWorkerBvhCompletedAtMs: 80,
		sessionPipelineUploadStartedAtMs: 81,
		sessionPipelineUploadCompletedAtMs: 90,
		exposurePrecomputeStartedAtMs: 91,
		exposurePrecomputeCompletedAtMs: 120,
		firstSelectedHourDispatchStartedAtMs: 121,
		firstSelectedHourDispatchCompletedAtMs: 130,
		firstSelectedHourReadyAtMs: 131,
		firstSelectedHourVisibleAtMs: 150
	};

	const diagnostics = buildMainRouteUtciDiagnostics({
		enabled: true,
		utciOnDemand: 'f32',
		utciRenderRequested: 'auto',
		utciRenderResolved: 'gpuNative',
		rendererBackend: 'webgpu',
		baseSurfaceDiagnostics: {
			utciSurfaceSource: 'compute-buffer-selected-hour',
			selectedHourTransferCount: 0,
			dataTextureBuildCount: 0,
			gpuResidentCopyStatus: 'complete',
			gpuResidentCopyRequestId: 1
		},
		comparisonSurfaceDiagnostics: {},
		baseRenderTransport: 'compute-buffer-selected-hour',
		comparisonRenderTransport: 'idle',
		baseLiveReady: true,
		comparisonLiveReady: false,
		baseSurfaceRequestId: 1,
		baseSelectionKey: 'analysis|7|0',
		baseSceneSurfaceRequestId: 1,
		baseSceneSelectionKey: 'analysis|7|0',
		baseSameDeviceForComputeAndRender: true,
		baseSelectedMonthIndex: 7,
		baseSelectedHourIndex: 0,
		baseSelectedTimeIndex: 168,
		comparisonSameDeviceForComputeAndRender: null,
		visibleSelectedHourReadbackCount: 0,
		readbackInstrumentation: 'instrumented',
		coldStart
	});

	expect(diagnostics?.coldStart).toEqual(coldStart);
	expect(diagnostics?.coldStart).not.toBe(coldStart);
	coldStart.firstSelectedHourVisibleAtMs = 999;
	expect(diagnostics?.coldStart?.firstSelectedHourVisibleAtMs).toBe(150);
});
```

- [ ] **Step 2: Run the failing test**

Run:

```powershell
cd viewer
npx vitest run tests/diagnostics/main-route-utci-diagnostics.test.ts --no-file-parallelism --maxWorkers=1
```

Expected: fail with a TypeScript or assertion error because `coldStart` is not part of `MainRouteUtciDiagnosticsInputs`.

- [ ] **Step 3: Add cold-start types and copy helper**

In `viewer/src/lib/compute/on-demand/onDemandDiagnostics.ts`, add this exported type near `OnDemandTimings`:

```ts
export type ColdStartWaterfallTimings = {
	routeAnalysisLoadStartedAtMs?: number;
	routeAnalysisLoadCompletedAtMs?: number;
	modelLoadStartedAtMs?: number;
	modelLoadCompletedAtMs?: number;
	modelProcessingStartedAtMs?: number;
	modelProcessingCompletedAtMs?: number;
	sessionPrepareStartedAtMs?: number;
	sessionPrepareCompletedAtMs?: number;
	sessionPayloadPrepareStartedAtMs?: number;
	sessionPayloadPrepareCompletedAtMs?: number;
	sessionWorkerBvhStartedAtMs?: number;
	sessionWorkerBvhCompletedAtMs?: number;
	sessionPipelineUploadStartedAtMs?: number;
	sessionPipelineUploadCompletedAtMs?: number;
	exposurePrecomputeStartedAtMs?: number;
	exposurePrecomputeCompletedAtMs?: number;
	firstSelectedHourDispatchStartedAtMs?: number;
	firstSelectedHourDispatchCompletedAtMs?: number;
	firstSelectedHourReadyAtMs?: number;
	firstSelectedHourVisibleAtMs?: number;
};
```

Add this field to `OnDemandTimings`:

```ts
coldStart?: ColdStartWaterfallTimings;
```

In `viewer/src/lib/diagnostics/mainRouteUtciDiagnostics.ts`, import and alias the shared type:

```ts
import type {
	ColdStartWaterfallTimings,
	OnDemandTimings,
	TrackedGpuAllocationBytes
} from '$lib/compute/on-demand/onDemandDiagnostics';

export type MainRouteColdStartDiagnostics = ColdStartWaterfallTimings;
```

Add `coldStart?: MainRouteColdStartDiagnostics;` to both `MainRouteUtciDiagnosticsPayload` and `MainRouteUtciDiagnosticsInputs`.

Add this helper below `copyDiagnosticsTimings`:

```ts
function copyColdStartDiagnostics(
	coldStart: MainRouteColdStartDiagnostics | undefined
): MainRouteColdStartDiagnostics | undefined {
	if (!coldStart) return undefined;
	return { ...coldStart };
}
```

In `buildMainRouteUtciDiagnostics`, add:

```ts
coldStart: copyColdStartDiagnostics(inputs.coldStart),
```

- [ ] **Step 4: Run the diagnostics test**

Run:

```powershell
cd viewer
npx vitest run tests/diagnostics/main-route-utci-diagnostics.test.ts --no-file-parallelism --maxWorkers=1
```

Expected: pass.

---

### Task 2: Stamp Session Cold Lifecycle Boundaries

**Files:**
- Modify: `viewer/src/lib/compute/selected-hour/liveUtciSelectedHourSession.ts`
- Test: `viewer/tests/compute/live-selected-hour-session.test.ts`

- [ ] **Step 1: Write the failing session lifecycle test**

Add a test next to the existing session preparation timing tests:

```ts
it('records cold lifecycle phase boundaries during live session preparation', async () => {
	const session = await prepareSelectedHourLiveSession({
		analysisId: 'analysis',
		base: baseAnalysis,
		model: modelGroup,
		epwUrl: '/weather.epw',
		signal: new AbortController().signal,
		preferredDevice: mockDevice as unknown as GPUDevice,
		gridResolution: 2
	});
	const result = await session.runSelectedHour({
		monthIndex: 7,
		hourIndex: 0,
		timeIndex: 168,
		colorMode: 'normalized',
		preferGpuResident: true,
		rendererDevice: mockDevice as unknown as GPUDevice
	});

	expect(result.diagnostics.timings.coldStart).toMatchObject({
		sessionPrepareStartedAtMs: expect.any(Number),
		sessionPayloadPrepareStartedAtMs: expect.any(Number),
		sessionPayloadPrepareCompletedAtMs: expect.any(Number),
		sessionWorkerBvhStartedAtMs: expect.any(Number),
		sessionWorkerBvhCompletedAtMs: expect.any(Number),
		sessionPipelineUploadStartedAtMs: expect.any(Number),
		sessionPipelineUploadCompletedAtMs: expect.any(Number),
		sessionPrepareCompletedAtMs: expect.any(Number)
	});
	const coldStart = result.diagnostics.timings.coldStart;
	expect(coldStart.sessionPayloadPrepareStartedAtMs).toBeGreaterThanOrEqual(
		coldStart.sessionPrepareStartedAtMs
	);
	expect(coldStart.sessionPayloadPrepareCompletedAtMs).toBeGreaterThanOrEqual(
		coldStart.sessionPayloadPrepareStartedAtMs
	);
	expect(coldStart.sessionWorkerBvhCompletedAtMs).toBeGreaterThanOrEqual(
		coldStart.sessionWorkerBvhStartedAtMs
	);
	expect(coldStart.sessionPipelineUploadCompletedAtMs).toBeGreaterThanOrEqual(
		coldStart.sessionPipelineUploadStartedAtMs
	);
	expect(coldStart.sessionPrepareCompletedAtMs).toBeGreaterThanOrEqual(
		coldStart.sessionPipelineUploadCompletedAtMs
	);
});
```

If the existing test fixture uses different local names than `baseAnalysis`, `modelGroup`, or `mockDevice`, use the same fixture names already used in nearby `prepareSelectedHourLiveSession(...)` tests.

- [ ] **Step 2: Run the failing session test**

Run:

```powershell
cd viewer
npx vitest run tests/compute/live-selected-hour-session.test.ts --no-file-parallelism --maxWorkers=1
```

Expected: fail because `session.diagnostics.timings.coldStart` does not exist.

- [ ] **Step 3: Add session lifecycle timing storage**

In `PreparedSessionState['lifecycleTimings']`, add a `coldStart` object compatible with `MainRouteColdStartDiagnostics`. If the type is local, define:

```ts
type LiveSelectedHourColdStartTimings = {
	sessionPrepareStartedAtMs?: number;
	sessionPrepareCompletedAtMs?: number;
	sessionPayloadPrepareStartedAtMs?: number;
	sessionPayloadPrepareCompletedAtMs?: number;
	sessionWorkerBvhStartedAtMs?: number;
	sessionWorkerBvhCompletedAtMs?: number;
	sessionPipelineUploadStartedAtMs?: number;
	sessionPipelineUploadCompletedAtMs?: number;
	exposurePrecomputeStartedAtMs?: number;
	exposurePrecomputeCompletedAtMs?: number;
	firstSelectedHourDispatchStartedAtMs?: number;
	firstSelectedHourDispatchCompletedAtMs?: number;
	firstSelectedHourReadyAtMs?: number;
	firstSelectedHourVisibleAtMs?: number;
};
```

At the top of `prepareSelectedHourLiveSession`, immediately after `coldStartStartedAt`, initialize:

```ts
const lifecycleTimings: PreparedSessionState['lifecycleTimings'] = {
	coldStart: {
		sessionPrepareStartedAtMs: coldStartStartedAt
	}
};
```

Replace the current empty lifecycle object if present.

Around payload preparation, stamp:

```ts
const payloadPrepareStartedAt = performance.now();
lifecycleTimings.coldStart.sessionPayloadPrepareStartedAtMs ??= payloadPrepareStartedAt;
try {
	// existing prepareMeshPayloadForWorkerAsync call
} finally {
	const payloadPrepareCompletedAt = performance.now();
	lifecycleTimings.payloadPrepareMs =
		(lifecycleTimings.payloadPrepareMs ?? 0) +
		(payloadPrepareCompletedAt - payloadPrepareStartedAt);
	lifecycleTimings.coldStart.sessionPayloadPrepareCompletedAtMs = payloadPrepareCompletedAt;
}
```

Around worker BVH:

```ts
const workerBvhStartedAt = performance.now();
lifecycleTimings.coldStart.sessionWorkerBvhStartedAtMs = workerBvhStartedAt;
workerResult = await runMergeAndBvhInWorker({
	meshes,
	gridResolution: effectiveGridResolution,
	zHeight,
	signal,
	maxGridPoints: LIVE_SELECTED_HOUR_MAX_GRID_POINTS,
	bvhOnly: true
});
const workerBvhCompletedAt = performance.now();
lifecycleTimings.workerBvhMs = workerBvhCompletedAt - workerBvhStartedAt;
lifecycleTimings.coldStart.sessionWorkerBvhCompletedAtMs = workerBvhCompletedAt;
```

Around upload:

```ts
const pipelineUploadStartedAt = performance.now();
lifecycleTimings.coldStart.sessionPipelineUploadStartedAtMs = pipelineUploadStartedAt;
const initResult = await uploadManager.initFromModelAndWeather({
	// existing params
});
ensureNotAborted(signal);
const pipelineUploadCompletedAt = performance.now();
lifecycleTimings.pipelineUploadMs = pipelineUploadCompletedAt - pipelineUploadStartedAt;
lifecycleTimings.coldStart.sessionPipelineUploadCompletedAtMs = pipelineUploadCompletedAt;
```

Before returning `createSelectedHourLiveSession`, stamp:

```ts
lifecycleTimings.coldStart.sessionPrepareCompletedAtMs = performance.now();
```

- [ ] **Step 4: Thread session cold timings into diagnostics**

Where session diagnostics are prepared from lifecycle timings, include:

```ts
coldStart: {
	...state.lifecycleTimings.coldStart
}
```

When first selected-hour compute starts and completes in the request path, stamp:

```ts
state.lifecycleTimings.coldStart.firstSelectedHourDispatchStartedAtMs ??= performance.now();
```

and after the selected-hour compute output returns:

```ts
state.lifecycleTimings.coldStart.firstSelectedHourDispatchCompletedAtMs ??= performance.now();
```

When `recordSelectedHourReadyTiming(...)` records first ready, also set:

```ts
state.lifecycleTimings.coldStart.firstSelectedHourReadyAtMs ??= performance.now();
```

- [ ] **Step 5: Run the session test**

Run:

```powershell
cd viewer
npx vitest run tests/compute/live-selected-hour-session.test.ts --no-file-parallelism --maxWorkers=1
```

Expected: pass.

---

### Task 3: Stamp Route Analysis and Model Boundaries

**Files:**
- Modify: `viewer/src/routes/+page.svelte`
- Modify: `viewer/src/routes/main/MainRouteViewport.svelte`
- Modify: `viewer/src/lib/components/scene/Model.svelte`
- Modify: `viewer/src/routes/main/liveSelectedHour.ts`
- Test: `viewer/tests/routes/main-route-live-selected-hour.test.ts`

- [ ] **Step 1: Add a route-level test for cold-start passthrough**

In `viewer/tests/routes/main-route-live-selected-hour.test.ts`, add a test near existing diagnostics input tests:

```ts
it('passes route cold-start boundaries into main-route diagnostics inputs', () => {
	const coldStart = {
		routeAnalysisLoadStartedAtMs: 1,
		routeAnalysisLoadCompletedAtMs: 2,
		modelLoadStartedAtMs: 3,
		modelLoadCompletedAtMs: 4,
		modelProcessingStartedAtMs: 5,
		modelProcessingCompletedAtMs: 6,
		sessionPrepareStartedAtMs: 7,
		sessionPrepareCompletedAtMs: 8
	};

	const inputs = buildMainRouteLiveSelectedHourDiagnosticsInputs({
		enabled: true,
		utciOnDemand: 'f32',
		utciRenderRequested: 'auto',
		utciRenderResolved: 'gpuNative',
		rendererBackend: 'webgpu',
		liveRouteState: makeLiveRouteState(),
		baseLiveReady: true,
		comparisonLiveReady: false,
		selectedMonthIndex: 7,
		selectedHourIndex: 0,
		selectedTimeIndex: 168,
		tooltipInteraction: makeTooltipInteractionDiagnostics(),
		cameraInteraction: makeCameraInteractionDiagnostics(),
		coldStart
	});

	expect(inputs.coldStart).toEqual(coldStart);
	expect(inputs.coldStart).not.toBe(coldStart);
});
```

Use the existing factory helpers in this test file. If the names differ, use the local helper that creates a valid `liveRouteState`, tooltip diagnostics, and camera diagnostics.

- [ ] **Step 2: Run the failing route test**

Run:

```powershell
cd viewer
npx vitest run tests/routes/main-route-live-selected-hour.test.ts --no-file-parallelism --maxWorkers=1
```

Expected: fail because `coldStart` is not accepted or copied by `buildMainRouteLiveSelectedHourDiagnosticsInputs`.

- [ ] **Step 3: Thread coldStart through `liveSelectedHour.ts`**

In `viewer/src/routes/main/liveSelectedHour.ts`, import the type:

```ts
import type {
	MainRouteColdStartDiagnostics,
	// existing imports
} from '$lib/diagnostics/mainRouteUtciDiagnostics';
```

Add to `MainRouteLiveSelectedHourDiagnosticsParams`:

```ts
coldStart?: MainRouteColdStartDiagnostics;
```

In `buildMainRouteLiveSelectedHourDiagnosticsInputs`, include:

```ts
coldStart: params.coldStart ? { ...params.coldStart } : undefined,
```

- [ ] **Step 4: Stamp route analysis load in `+page.svelte`**

Near the route state that owns selected analysis loading, add:

```ts
let mainRouteColdStartDiagnostics: MainRouteColdStartDiagnostics = {};
```

Import the type:

```ts
import type { MainRouteColdStartDiagnostics } from '$lib/diagnostics/mainRouteUtciDiagnostics';
```

Around the existing `loadAnalysis(analysisId)` path, stamp:

```ts
if (utciRenderDiagnosticsEnabled) {
	mainRouteColdStartDiagnostics = {
		...mainRouteColdStartDiagnostics,
		routeAnalysisLoadStartedAtMs: performance.now()
	};
}
try {
	// existing loadAnalysis call
} finally {
	if (utciRenderDiagnosticsEnabled) {
		mainRouteColdStartDiagnostics = {
			...mainRouteColdStartDiagnostics,
			routeAnalysisLoadCompletedAtMs: performance.now()
		};
	}
}
```

When the selected analysis/model identity changes enough to start a new cold lifecycle, reset:

```ts
mainRouteColdStartDiagnostics = {};
```

Do this only at the same route boundary that already resets model/live selected-hour state, so scrubs do not erase cold evidence.

- [ ] **Step 5: Add model timing callback without changing overlay copy**

In `viewer/src/lib/components/scene/Model.svelte`, export:

```ts
export let onModelColdStartTiming:
	| ((event: { phase: 'load-start' | 'load-complete' | 'processing-start' | 'processing-complete'; atMs: number }) => void)
	| undefined = undefined;
```

Before model loading begins:

```ts
onModelColdStartTiming?.({ phase: 'load-start', atMs: performance.now() });
```

After the GLB/model load resolves:

```ts
onModelColdStartTiming?.({ phase: 'load-complete', atMs: performance.now() });
```

Before material/layer/model post-processing:

```ts
onModelColdStartTiming?.({ phase: 'processing-start', atMs: performance.now() });
```

After post-processing completes and before the component reports ready:

```ts
onModelColdStartTiming?.({ phase: 'processing-complete', atMs: performance.now() });
```

In `+page.svelte`, pass a handler to `Model`:

```ts
function recordMainRouteModelColdStartTiming(event: {
	phase: 'load-start' | 'load-complete' | 'processing-start' | 'processing-complete';
	atMs: number;
}) {
	if (!utciRenderDiagnosticsEnabled) return;
	const fieldByPhase = {
		'load-start': 'modelLoadStartedAtMs',
		'load-complete': 'modelLoadCompletedAtMs',
		'processing-start': 'modelProcessingStartedAtMs',
		'processing-complete': 'modelProcessingCompletedAtMs'
	} as const;
	mainRouteColdStartDiagnostics = {
		...mainRouteColdStartDiagnostics,
		[fieldByPhase[event.phase]]: event.atMs
	};
}
```

- [ ] **Step 6: Bridge model timing through `MainRouteViewport.svelte`**

In `viewer/src/routes/main/MainRouteViewport.svelte`, export the same callback prop:

```ts
export let onModelColdStartTiming:
	| ((event: { phase: 'load-start' | 'load-complete' | 'processing-start' | 'processing-complete'; atMs: number }) => void)
	| undefined = undefined;
```

Pass it into the existing `<Model ... />` call:

```svelte
<Model
	...
	{onModelColdStartTiming}
/>
```

In `+page.svelte`, pass the handler into `<MainRouteViewport ... />`:

```svelte
<MainRouteViewport
	...
	onModelColdStartTiming={utciRenderDiagnosticsEnabled
		? recordMainRouteModelColdStartTiming
		: undefined}
/>
```

- [ ] **Step 7: Merge session cold timings and publish diagnostics**

Where `buildMainRouteLiveSelectedHourDiagnosticsInputs(...)` is called in `+page.svelte`, pass:

```ts
coldStart: utciRenderDiagnosticsEnabled
	? {
			...mainRouteColdStartDiagnostics,
			...liveRouteState.base.runtimeDiagnostics?.timings?.coldStart
	  }
	: undefined
```

If the controller state stores timings under a different local field, use the existing field that already feeds `timingsOverride`.

- [ ] **Step 8: Run route tests**

Run:

```powershell
cd viewer
npx vitest run tests/routes/main-route-live-selected-hour.test.ts --no-file-parallelism --maxWorkers=1
```

Expected: pass.

---

### Task 4: Collect Cold Waterfall Artifacts

**Files:**
- Modify: `viewer/tests/e2e/main-route-performance-baseline.spec.ts`
- Modify: `viewer/tests/e2e/main-route-performance-0_5m.spec.ts`
- Create: `data/performance-results/main-route-cold-start-waterfall.json`

- [ ] **Step 1: Add cold-start extraction to the 2m collector**

In `viewer/tests/e2e/main-route-performance-baseline.spec.ts`, update `CollectedCase`:

```ts
coldStart: Record<string, number | null>;
```

Add:

```ts
function extractColdStart(coldStart: Record<string, unknown> | undefined) {
	return {
		routeAnalysisLoadStartedAtMs: numberOrNull(coldStart?.routeAnalysisLoadStartedAtMs),
		routeAnalysisLoadCompletedAtMs: numberOrNull(coldStart?.routeAnalysisLoadCompletedAtMs),
		modelLoadStartedAtMs: numberOrNull(coldStart?.modelLoadStartedAtMs),
		modelLoadCompletedAtMs: numberOrNull(coldStart?.modelLoadCompletedAtMs),
		modelProcessingStartedAtMs: numberOrNull(coldStart?.modelProcessingStartedAtMs),
		modelProcessingCompletedAtMs: numberOrNull(coldStart?.modelProcessingCompletedAtMs),
		sessionPrepareStartedAtMs: numberOrNull(coldStart?.sessionPrepareStartedAtMs),
		sessionPrepareCompletedAtMs: numberOrNull(coldStart?.sessionPrepareCompletedAtMs),
		sessionPayloadPrepareStartedAtMs: numberOrNull(coldStart?.sessionPayloadPrepareStartedAtMs),
		sessionPayloadPrepareCompletedAtMs: numberOrNull(coldStart?.sessionPayloadPrepareCompletedAtMs),
		sessionWorkerBvhStartedAtMs: numberOrNull(coldStart?.sessionWorkerBvhStartedAtMs),
		sessionWorkerBvhCompletedAtMs: numberOrNull(coldStart?.sessionWorkerBvhCompletedAtMs),
		sessionPipelineUploadStartedAtMs: numberOrNull(coldStart?.sessionPipelineUploadStartedAtMs),
		sessionPipelineUploadCompletedAtMs: numberOrNull(coldStart?.sessionPipelineUploadCompletedAtMs),
		exposurePrecomputeStartedAtMs: numberOrNull(coldStart?.exposurePrecomputeStartedAtMs),
		exposurePrecomputeCompletedAtMs: numberOrNull(coldStart?.exposurePrecomputeCompletedAtMs),
		firstSelectedHourDispatchStartedAtMs: numberOrNull(coldStart?.firstSelectedHourDispatchStartedAtMs),
		firstSelectedHourDispatchCompletedAtMs: numberOrNull(coldStart?.firstSelectedHourDispatchCompletedAtMs),
		firstSelectedHourReadyAtMs: numberOrNull(coldStart?.firstSelectedHourReadyAtMs),
		firstSelectedHourVisibleAtMs: numberOrNull(coldStart?.firstSelectedHourVisibleAtMs)
	};
}
```

In the returned case object, include:

```ts
coldStart: extractColdStart(diagnostics.coldStart),
renderPublication: diagnostics.timings?.renderPublication ?? null,
```

Derive `Initial render publication ms` in docs from the existing coarse timing first:

```ts
const initialRenderPublicationMs = numberOrNull(diagnostics.timings?.renderUpdateMs);
```

Also persist `timings.renderPublication` so a later pass can split initial render publication further without rerunning the collection. Do not invent sub-buckets that are not present.

- [ ] **Step 2: Assert required proof and cold fields in the collector**

Add an explicit proof assertion block; do not rely on previous collector behavior:

```ts
expect(new URL(sourceUrl, 'http://localhost').pathname).toBe('/');
expect(diagnostics.rendererBackend).toBe('webgpu');
expect(diagnostics.utciRenderResolved).toBe('gpuNative');
expect(diagnostics.utciSurfaceSource).toBe('compute-buffer-selected-hour');
expect(diagnostics.baseRenderTransport).toBe('compute-buffer-selected-hour');
expect(diagnostics.dataTextureBuildCount).toBe(0);
expect(diagnostics.selectedHourRuntimeContract?.route).toBe('main');
expect(diagnostics.selectedHourRuntimeContract?.readbackInstrumentation).toBe('instrumented');
expect(diagnostics.selectedHourRuntimeContract?.visibleSelectedHourReadbackCount).toBe(0);
expect(diagnostics.selectedHourRuntimeContract?.strongVisibleGpuPath).toBe(true);
expect(forbiddenComparisonFieldsPresent).toEqual([]);
expect(forbiddenRequestUrls).toEqual([]);
```

Then add cold timing assertions:

```ts
expect(diagnostics.coldStart).toMatchObject({
	routeAnalysisLoadStartedAtMs: expect.any(Number),
	routeAnalysisLoadCompletedAtMs: expect.any(Number),
	modelLoadStartedAtMs: expect.any(Number),
	modelLoadCompletedAtMs: expect.any(Number),
	sessionPrepareStartedAtMs: expect.any(Number),
	sessionPrepareCompletedAtMs: expect.any(Number),
	exposurePrecomputeStartedAtMs: expect.any(Number),
	exposurePrecomputeCompletedAtMs: expect.any(Number),
	firstSelectedHourDispatchStartedAtMs: expect.any(Number),
	firstSelectedHourDispatchCompletedAtMs: expect.any(Number),
	firstSelectedHourReadyAtMs: expect.any(Number),
	firstSelectedHourVisibleAtMs: expect.any(Number)
});
expect(diagnostics.coldStart.routeAnalysisLoadCompletedAtMs).toBeGreaterThanOrEqual(
	diagnostics.coldStart.routeAnalysisLoadStartedAtMs
);
expect(diagnostics.coldStart.sessionPrepareCompletedAtMs).toBeGreaterThanOrEqual(
	diagnostics.coldStart.sessionPrepareStartedAtMs
);
expect(diagnostics.coldStart.exposurePrecomputeCompletedAtMs).toBeGreaterThanOrEqual(
	diagnostics.coldStart.exposurePrecomputeStartedAtMs
);
expect(diagnostics.coldStart.firstSelectedHourVisibleAtMs).toBeGreaterThanOrEqual(
	diagnostics.coldStart.firstSelectedHourReadyAtMs
);
```

Only assert `completed >= started` within the same owner/phase. Do not assert a single strict total ordering across route analysis load, model load, session preparation, scene publication, and Svelte reactive publication unless the implementation proves those stamps come from one causal chain.

- [ ] **Step 3: Write the sibling cold artifact**

Change the artifact path for this cold collection branch to:

```ts
const ARTIFACT_PATH = resolve(RESULTS_DIR, 'main-route-cold-start-waterfall.json');
const COLLECTED_ON = '2026-05-18';
```

The JSON root should include:

```ts
{
	collectedOn: COLLECTED_ON,
	sourceRoute: SOURCE_ROUTE,
	collectionMethod:
		'Main route cold-start waterfall: / with utciRender=auto&utciRenderDiagnostics=1, initial selected hour only, no debug route and no parity/.bin comparison.',
	cases
}
```

- [ ] **Step 4: Add 0.5m initial cold cases**

Do not write 0.5m cold output through the warm artifact path in `viewer/tests/e2e/main-route-performance-0_5m.spec.ts`.

Use one of these safe approaches:

1. Add a cold-only test in `main-route-performance-baseline.spec.ts` that visits the same 0.5m URLs and appends 0.5m `cold-initial` cases to `main-route-cold-start-waterfall.json`.
2. Or add a cold collection mode to `main-route-performance-0_5m.spec.ts` with a separate constant `COLD_ARTIFACT_PATH = resolve(RESULTS_DIR, 'main-route-cold-start-waterfall.json')` and a separate test title/grep.

The 0.5m cold entries must use the shared cold artifact schema:

```ts
{
	projectLabel,
	analysisId,
	gridResolutionMeters: 0.5,
	colorMode: 'normalized',
	phase: 'cold-initial',
	timings,
	coldStart,
	renderPublication,
	proof,
	assertions
}
```

Do not mix scrub rows into the cold headline table.

- [ ] **Step 5: Run the 2m cold collector**

Run:

```powershell
cd viewer
npx playwright test tests/e2e/main-route-performance-baseline.spec.ts --project=chromium --workers=1 --reporter=list
```

Expected: pass and write `data/performance-results/main-route-cold-start-waterfall.json` with BG and Ness Tziona 2m cold cases.

- [ ] **Step 6: Run the 0.5m collector**

Run:

```powershell
cd viewer
npx playwright test tests/e2e/main-route-performance-0_5m.spec.ts --project=chromium --workers=1 --reporter=list --grep "collects BG and Ness Tziona 0.5m main-route timing, memory, color-mode, and scrub samples"
```

Expected: pass and include cold-start fields for 0.5m initial cases without weakening the existing warm-scrub assertions.

---

### Task 5: Write the Cold-Start Evidence Note

**Files:**
- Create: `docs/performance/main-route-cold-start-waterfall.md`
- Read: `data/performance-results/main-route-cold-start-waterfall.json`

- [ ] **Step 1: Create the evidence note**

Create `docs/performance/main-route-cold-start-waterfall.md` with this structure:

```md
# Main Route Cold-Start Waterfall Evidence

Date: 2026-05-18

## Scope

This note measures the main route `/` cold-start path. It is separate from warm-scrub selected-hour render diagnostics and does not use `/debug`, `.bin`, Python reference data, or parity comparison.

JSON source: [data/performance-results/main-route-cold-start-waterfall.json](../../data/performance-results/main-route-cold-start-waterfall.json)

## Proof Boundary

- `rendererBackend=webgpu`
- `utciRenderResolved=gpuNative`
- `utciSurfaceSource=compute-buffer-selected-hour`
- `baseRenderTransport=compute-buffer-selected-hour`
- `dataTextureBuildCount=0`
- `selectedHourRuntimeContract.strongVisibleGpuPath=true`
- `selectedHourRuntimeContract.visibleSelectedHourReadbackCount=0`
- no python/bin/debug comparison fields
- no forbidden comparison requests

## Included Analyses

- `Ben-Gurion/20250815_grid_2m_fullday`
- `Ness-Tziona/exploded/nes_tziona_unblock_2`

## Timing Table

| Project | Grid m | First visible ms | Analysis load ms | Model load ms | Model processing ms | Payload prepare ms | Worker BVH ms | Upload ms | Exposure precompute ms | First-hour dispatch ms | Initial render publication ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |

## Diagnosis

- List the largest observed bucket for each project/grid.
- Separate confirmed timings from null/unavailable fields.
- State whether exposure precompute, model work, upload, or initial render publication is currently the largest target.

## Next Optimization Candidates

1. Candidate selected from the largest confirmed bucket.
2. Candidate selected from the second-largest confirmed bucket.
3. Candidate only if evidence shows it is real.

## Non-Goals

- No overlay copy changes in this pass.
- No warm-scrub optimization in this pass.
- No `.bin` or parity comparison in this pass.
```

- [ ] **Step 2: Fill the table from the collected JSON**

Use exact numbers from `data/performance-results/main-route-cold-start-waterfall.json`. Compute each phase duration by subtracting start from completed stamps. If a field is missing, write `null` and explain the gap in `Diagnosis`.

- [ ] **Step 3: Verify no stale warm-path claims leaked into the cold note**

Run:

```powershell
Select-String -Path docs\performance\main-route-cold-start-waterfall.md -Pattern "scrub|warm|layout reuse|Python|\\.bin|parity"
```

Expected: matches are allowed in scope, proof-boundary, and non-goal text. Manually inspect the output and confirm no warm-scrub diagnosis is presented as cold-start evidence.

---

### Task 6: Final Verification and Review Agents

**Files:**
- Verify all modified files.
- Do not commit.

- [ ] **Step 1: Run focused unit tests**

Run:

```powershell
cd viewer
npx vitest run tests/diagnostics/main-route-utci-diagnostics.test.ts tests/routes/main-route-live-selected-hour.test.ts tests/compute/live-selected-hour-session.test.ts --no-file-parallelism --maxWorkers=1
```

Expected: pass.

- [ ] **Step 2: Run cold and 0.5m collectors**

Run:

```powershell
cd viewer
npx playwright test tests/e2e/main-route-performance-baseline.spec.ts --project=chromium --workers=1 --reporter=list
```

Expected: pass.

Run:

```powershell
cd viewer
npx playwright test tests/e2e/main-route-performance-0_5m.spec.ts --project=chromium --workers=1 --reporter=list --grep "collects BG and Ness Tziona 0.5m main-route timing, memory, color-mode, and scrub samples"
```

Expected: pass.

- [ ] **Step 3: Run static checks**

Run:

```powershell
cd viewer
npm run check
```

Expected: pass, or report pre-existing unrelated failures separately with exact file/line evidence.

- [ ] **Step 4: Run review agents**

Dispatch a spec-compliance review agent first. It must check:

- no commits
- no git worktrees
- cold artifact is separate from warm artifacts
- route is `/`, not `/debug`
- proof boundary remains intact
- no overlay UX/copy changes
- no optimization changes

Only after spec compliance is clean, dispatch a code-quality review agent. It must check:

- diagnostics copies are defensive
- timestamp ordering assertions are meaningful but not over-constrained
- route/model/session stamps reset only on cold lifecycle boundaries
- e2e collectors do not mix warm and cold headline claims
- generated docs match collected JSON

- [ ] **Step 5: Report current status**

Report:

- files changed
- exact commands run and pass/fail
- cold-start largest buckets by project/grid
- any missing/null fields
- next optimization recommendation, if and only if the collected evidence supports it

Do not claim performance improvement. This plan only creates proof.

---

## Self-Review

- Spec coverage: The plan focuses first on breaking cold-start timing into discrete steps, observes via sibling artifacts, and defers optimization/overlay UX changes.
- Placeholder scan: No `TBD`, `TODO`, or open-ended "add tests" steps remain.
- Type consistency: `MainRouteColdStartDiagnostics` is introduced in diagnostics, passed through route diagnostics, merged from route/session stamps, and extracted by collectors.
- Constraint check: No commits and no git worktrees are required or suggested.
