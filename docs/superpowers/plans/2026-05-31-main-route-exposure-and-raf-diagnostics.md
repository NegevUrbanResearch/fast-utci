# Main Route Exposure And RAF Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and collect one complete diagnostics pass that separates Ness Tziona 0.5m desktop "breathing" pain during exposure from page-local rAF/render-publication pain after exposure.

**Architecture:** Keep the product proof surface on `/` and add diagnostics-only timing fields behind existing `utciRenderDiagnostics=1` collection paths. The plan does not change default scheduling, does not tune chunk size, does not add lazy/background exposure fill, and does not implement cooperative render-publication chunking. It produces one refreshed artifact and note that can decide whether the next implementation spike should target exposure pacing/system responsiveness or render-publication rAF stalls.

**Tech Stack:** Svelte, TypeScript, Three.js WebGPU, Playwright headed Chromium collectors, Vitest.

---

## Hard Constraints

- Do not commit.
- Do not create git worktrees.
- Preserve unrelated dirty/staged files.
- Proof surface is `/`, not `/debug`.
- Preserve `rendererBackend=webgpu`, `compute-buffer-selected-hour`, same-device compute/render proof, and `visibleSelectedHourReadbackCount=0`.
- Do not move load cost onto scrub.
- Do not implement lazy/background exposure fill.
- Do not reattempt cooperative render-publication chunking.
- Do not run slow collectors first during implementation. Update unit/source-lock coverage before running the headed collector.
- If timing gates fail or evidence is inconclusive, report that directly.

## Current Evidence Boundary

Existing artifacts already prove that smaller chunks are not the next answer:

- `docs/performance/main-route-exposure-scheduler.md`
- `data/performance-results/main-route-visual-freeze-map.json`
- `data/performance-results/main-route-cold-start-waterfall.json`

Current promoted `2048` chunking lowers the largest exposure queue wait into the `~381-425ms` band, but the browser still records `~1.3-1.5s` top rAF/long-task gaps. The new question is not "what chunk size is better?" It is:

1. During the `~17-20s` exposure window, is the GPU/driver/page loop still saturated enough that the desktop cannot breathe?
2. After exposure, is there a separate page-local render-publication rAF stall that should be fixed independently?

## File Structure

Modify:

- `viewer/src/lib/compute/gpu/webgpuUtciPipeline.ts`
  - Add exposure breathing trace fields for chunked exposure slices: per-slice queue wait, encode duration, submit timestamp, yield wait, and inter-slice page-frame delay.
  - Publish bounded aggregate and sample diagnostics, not unbounded per-slice logs.
- `viewer/src/lib/compute/on-demand/onDemandDiagnostics.ts`
  - Own the exported exposure breathing trace types and extend `OnDemandTimings`; do not create ad hoc local timing types in the pipeline.
- `viewer/src/lib/diagnostics/selectedHourRenderPublicationDiagnostics.ts`
  - Extend render publication timeline with the missing initial scene-publication split fields.
- `viewer/src/lib/components/scene/UTCIPointCloud.svelte`
  - Stamp the new render-publication split fields without adding yield behavior or changing publication order.
- `viewer/src/lib/components/scene/utciComputeBufferRenderBridge.ts`
  - Split copy submit from queue drain and expose queue-drain diagnostics already returned by `copyComputeBufferToRenderStorage()`.
- `viewer/tests/e2e/main-route-visual-freeze-map.spec.ts`
  - Extend the browser probe and artifact summary to correlate rAF/interval/long-task gaps with exposure slices and render-publication windows.
- `viewer/tests/e2e/main-route-cold-start-waterfall.spec.ts`
  - Preserve the new fields in summary output and proof checks if it also snapshots the same timing structure.
- `viewer/tests/compute/webgpuUtciPipeline.behavior.test.ts`
  - Add focused tests for exposure breathing trace bounds and aggregation.
- `viewer/tests/e2e/main-route-exposure-scheduler-collectors-source-lock.test.ts`
  - Lock the collector source to the required proof fields and new diagnostics fields.
- `docs/performance/main-route-exposure-and-raf-diagnostics.md`
  - Create the evidence note after collection.

Create:

- `data/performance-results/main-route-exposure-and-raf-diagnostics.json`
  - New focused artifact written by the extended visual-freeze collector or by a sibling collector if keeping old artifact compatibility is cleaner.

Do not modify:

- Runtime defaults in `viewer/src/lib/compute/gpu/exposureScheduling.ts` except source-lock-compatible query parsing if tests reveal existing fields are not carried through.
- Any deleted/unrelated plan files currently present in the dirty worktree.

---

## Perspective Ensemble Review

### Panel A - Council

- **Felt performance lens:** concern -> page rAF may not represent desktop freeze -> counter-move: make exposure breathing the lead diagnostic with per-slice queue/yield/frame-delay fields, and keep rAF as a correlated symptom rather than the only target.
- **Renderer correctness lens:** concern -> render-publication diagnostics can tempt a queue-drain removal without proof -> counter-move: collect copy/queue timing only; any queue-drain behavior change requires a later correctness plan.
- **Collector cost lens:** concern -> adding every slice verbatim can make artifacts huge and slow -> counter-move: collect aggregate stats plus first/last/top-N slice samples.
- **Repeat-work lens:** concern -> plan drifts back to smaller chunks, lazy exposure, analytic bounds, or cooperative publication yielding -> counter-move: mark these as excluded and only use them as regression context.
- **Proof-boundary lens:** concern -> diagnostics route or comparison mode pollutes evidence -> counter-move: collector waits only for strong main-route GPU-native publication and records forbidden comparison-field absence.

### Tensions

- Full per-slice evidence vs artifact size: keep bounded samples plus aggregate stats.
- Responsiveness metrics vs causal attribution: rAF/long tasks are symptoms; timeline fields and slice telemetry are owner evidence.
- Render-publication curiosity vs user pain: collect rAF/publication fields, but rank exposure breathing first if it explains the desktop freeze.

### Panel B - Adversarial Red Cell

- **Attack target:** A single diagnostics run that claims to distinguish system-level exposure saturation from page-local rAF stalls.
- **Metric mismatch:** vulnerability -> browser rAF and long-task data still cannot observe Windows desktop/GPU scheduler pressure directly -> failure scenario: artifact says "page breathed" while the desktop still feels frozen -> mitigation: include exposure slice cadence, queue wait, inter-slice yielded frame delay, and browser probe gaps, then label OS-level GPU saturation as inferred unless externally profiled.
- **Observer effect:** vulnerability -> extra instrumentation during exposure changes scheduling -> failure scenario: diagnostics makes the freeze better or worse -> mitigation: keep fields cheap, bounded, and diagnostics-only; compare first visible and first scrub against current artifact bands.
- **Wrong owner collapse:** vulnerability -> render-publication rAF gap is easier to explain than exposure breathing -> failure scenario: next plan fixes rAF but leaves felt load freeze -> mitigation: evidence note must rank exposure breathing and render-publication independently.
- **One missing field risk:** vulnerability -> plan adds many fields but misses the exact boundary between yielding to rAF and actual frame paint -> failure scenario: another diagnostic plan is needed -> mitigation: collector must explicitly compute overlap tables between exposure slices, yielded-frame windows, rAF gaps, interval gaps, long tasks, and scene publication windows.

### Conditional Recommendation

Create one diagnostics spike with two explicit outputs:

1. **Exposure breathing profile:** per-slice aggregate/samples plus browser gap overlap during `exposurePrecomputeMs`.
2. **Render-publication rAF profile:** initial publication split from controller acceptance through layout/proof/mesh/storage/copy/queue drain.

The next implementation plan is allowed only after the evidence note says which lane owns the felt pain.

---

## Task 1: Define Diagnostics Shapes And Source Locks

**Files:**

- Modify: `viewer/src/lib/compute/gpu/webgpuUtciPipeline.ts`
- Modify: `viewer/src/lib/compute/on-demand/onDemandDiagnostics.ts`
- Modify: `viewer/src/lib/diagnostics/selectedHourRenderPublicationDiagnostics.ts`
- Modify: `viewer/tests/e2e/main-route-exposure-scheduler-collectors-source-lock.test.ts`

- [ ] **Step 1: Add exposure breathing trace types**

In `viewer/src/lib/compute/on-demand/onDemandDiagnostics.ts`, add bounded trace types near `OnDemandTimings`. Import `ExposureSchedulingMode` from `viewer/src/lib/compute/gpu/exposureScheduling.ts`; do not redeclare it.

```ts
export type ExposureSchedulerSliceTraceSample = {
	sliceIndex: number;
	pointStart: number;
	pointCount: number;
	workgroupCount: number;
	encodeMs: number;
	submitAtMs: number;
	queueWaitMs: number;
	yieldStartedAtMs?: number;
	yieldRafCallbackAtMs?: number;
	yieldCompletedAtMs?: number;
	yieldWaitMs?: number;
	yieldPostRafTimeoutMs?: number;
};

export type ExposureSchedulerSliceWindow = {
	sliceIndex: number;
	startMs: number;
	endMs: number;
	queueWaitMs: number;
	yieldWaitMs?: number;
};

export type ExposureSchedulerBreathingTrace = {
	version: 1;
	mode: ExposureSchedulingMode;
	maxWorkgroupsPerSlice: number;
	sliceCount: number;
	submitCount: number;
	yieldCount: number;
	queueWaitTotalMs: number;
	queueWaitMaxMs: number;
	queueWaitMinMs: number;
	queueWaitAverageMs: number;
	encodeTotalMs: number;
	yieldWaitTotalMs: number;
	yieldWaitMaxMs: number;
	yieldWaitAverageMs: number;
	yieldPostRafTimeoutMaxMs: number;
	yieldPostRafTimeoutAverageMs: number;
	allSliceWindows: ExposureSchedulerSliceWindow[];
	firstSamples: ExposureSchedulerSliceTraceSample[];
	worstQueueWaitSamples: ExposureSchedulerSliceTraceSample[];
	worstYieldSamples: ExposureSchedulerSliceTraceSample[];
	lastSamples: ExposureSchedulerSliceTraceSample[];
};
```

Then extend `OnDemandTimings` in the same file:

```ts
exposureSchedulerBreathingTrace?: ExposureSchedulerBreathingTrace;
```

- [ ] **Step 2: Add render-publication split fields**

In `viewer/src/lib/diagnostics/selectedHourRenderPublicationDiagnostics.ts`, extend `SelectedHourRenderPublicationTimeline` with these optional fields:

```ts
renderPublicationPreStorageStartedAtMs?: number;
renderPublicationPreStorageCompletedAtMs?: number;
renderPublicationPreStorageMs?: number;
renderStoragePendingFlagStartedAtMs?: number;
renderStoragePendingFlagCompletedAtMs?: number;
renderStorageInvalidateRequestedAtMs?: number;
renderStorageFirstWaitFrameRequestedAtMs?: number;
renderStorageFirstWaitFrameCompletedAtMs?: number;
renderCopyQueueDrainStartedAtMs?: number;
renderCopyQueueDrainCompletedAtMs?: number;
renderCopyQueueDrainMs?: number;
```

Also update `copyRenderPublicationTimeline()` to preserve every new field explicitly, matching the surrounding pattern.

- [ ] **Step 3: Lock source expectations**

In `viewer/tests/e2e/main-route-exposure-scheduler-collectors-source-lock.test.ts`, add assertions that collector source or diagnostics summary includes:

```ts
expect(source).toContain('exposureSchedulerBreathingTrace');
expect(source).toContain('topRafGaps');
expect(source).toContain('longTasks');
expect(source).toContain('renderPublicationPreStorageMs');
expect(source).toContain('renderCopyQueueDrainMs');
expect(source).toContain('visibleSelectedHourReadbackCount');
expect(source).toContain('baseSameDeviceForComputeAndRender');
```

- [ ] **Step 4: Run focused source-lock test**

Run:

```powershell
cd viewer
npm test -- --run tests/e2e/main-route-exposure-scheduler-collectors-source-lock.test.ts
```

Expected: test fails until the collector preservation work in later tasks is complete, or passes if the lock is staged after collector edits. Do not commit.

---

## Task 2: Add Exposure Breathing Trace Collection

**Files:**

- Modify: `viewer/src/lib/compute/gpu/webgpuUtciPipeline.ts`
- Modify: `viewer/src/lib/compute/on-demand/onDemandDiagnostics.ts`
- Test: `viewer/tests/compute/webgpuUtciPipeline.behavior.test.ts`

- [ ] **Step 1: Add bounded sample helper**

In `viewer/src/lib/compute/gpu/webgpuUtciPipeline.ts`, add a helper near `runChunkedExposurePrecompute()`:

```ts
function buildBoundedExposureBreathingTrace(params: {
	mode: ExposureSchedulingMode;
	maxWorkgroupsPerSlice: number;
	samples: ExposureSchedulerSliceTraceSample[];
	queueWaitTotalMs: number;
	queueWaitMaxMs: number;
	queueWaitMinMs: number;
	encodeTotalMs: number;
	yieldWaitTotalMs: number;
	yieldWaitMaxMs: number;
	submitCount: number;
	yieldCount: number;
}): ExposureSchedulerBreathingTrace {
	const sliceCount = params.samples.length;
	const byQueueWait = [...params.samples]
		.sort((left, right) => right.queueWaitMs - left.queueWaitMs)
		.slice(0, 8);
	const byYieldWait = [...params.samples]
		.sort((left, right) => (right.yieldWaitMs ?? 0) - (left.yieldWaitMs ?? 0))
		.slice(0, 8);
	const yieldPostRafTimeouts = params.samples
		.map((sample) => sample.yieldPostRafTimeoutMs)
		.filter((value): value is number => typeof value === 'number' && Number.isFinite(value));
	const yieldPostRafTimeoutTotal = yieldPostRafTimeouts.reduce((sum, value) => sum + value, 0);
	return {
		version: 1,
		mode: params.mode,
		maxWorkgroupsPerSlice: params.maxWorkgroupsPerSlice,
		sliceCount,
		submitCount: params.submitCount,
		yieldCount: params.yieldCount,
		queueWaitTotalMs: params.queueWaitTotalMs,
		queueWaitMaxMs: params.queueWaitMaxMs,
		queueWaitMinMs: params.queueWaitMinMs,
		queueWaitAverageMs: sliceCount > 0 ? params.queueWaitTotalMs / sliceCount : 0,
		encodeTotalMs: params.encodeTotalMs,
		yieldWaitTotalMs: params.yieldWaitTotalMs,
		yieldWaitMaxMs: params.yieldWaitMaxMs,
		yieldWaitAverageMs: params.yieldCount > 0 ? params.yieldWaitTotalMs / params.yieldCount : 0,
		yieldPostRafTimeoutMaxMs:
			yieldPostRafTimeouts.length > 0 ? Math.max(...yieldPostRafTimeouts) : 0,
		yieldPostRafTimeoutAverageMs:
			yieldPostRafTimeouts.length > 0
				? yieldPostRafTimeoutTotal / yieldPostRafTimeouts.length
				: 0,
		allSliceWindows: params.samples.map((sample) => ({
			sliceIndex: sample.sliceIndex,
			startMs: sample.submitAtMs,
			endMs: sample.yieldCompletedAtMs ?? sample.submitAtMs + sample.queueWaitMs,
			queueWaitMs: sample.queueWaitMs,
			yieldWaitMs: sample.yieldWaitMs
		})),
		firstSamples: params.samples.slice(0, 8),
		worstQueueWaitSamples: byQueueWait,
		worstYieldSamples: byYieldWait,
		lastSamples: params.samples.slice(-8)
	};
}
```

- [ ] **Step 2: Stamp chunked slice samples**

Inside `runChunkedExposurePrecompute()`, create `const sliceSamples: ExposureSchedulerSliceTraceSample[] = [];`, `let yieldWaitTotalMs = 0;`, and `let yieldWaitMaxMs = 0;`.

For each slice, record:

```ts
const submitAtMs = performance.now();
this.queue.submit([encoder.finish()]);
submitCount += 1;
const queueWaitStartedAt = performance.now();
await this.queue.onSubmittedWorkDone();
const queueWaitMs = performance.now() - queueWaitStartedAt;

const sample: ExposureSchedulerSliceTraceSample = {
	sliceIndex,
	pointStart: pointSlice.pointOffset,
	pointCount: pointSlice.pointCount,
	workgroupCount: Math.ceil(pointSlice.pointCount / workgroupSize),
	encodeMs: trace.commandEncodeTotalMs,
	submitAtMs,
	queueWaitMs
};
```

First change `yieldToBrowserFrame()` so it returns honest split timing instead of a bare promise:

```ts
async function yieldToBrowserFrame(): Promise<{
	rafCallbackAtMs?: number;
	completedAtMs: number;
}> {
	let rafCallbackAtMs: number | undefined;
	if (typeof requestAnimationFrame === 'function') {
		await new Promise<void>((resolve) => {
			requestAnimationFrame(() => {
				rafCallbackAtMs = performance.now();
				setTimeout(resolve, 0);
			});
		});
		return { rafCallbackAtMs, completedAtMs: performance.now() };
	}
	await new Promise<void>((resolve) => setTimeout(resolve, 0));
	return { completedAtMs: performance.now() };
}
```

When `yieldBetweenSlices` is active, wrap `yieldToBrowserFrame()`:

```ts
const yieldStartedAtMs = performance.now();
const yieldResult = await yieldToBrowserFrame();
const yieldCompletedAtMs = yieldResult.completedAtMs;
const yieldWaitMs = yieldCompletedAtMs - yieldStartedAtMs;
sample.yieldStartedAtMs = yieldStartedAtMs;
sample.yieldRafCallbackAtMs = yieldResult.rafCallbackAtMs;
sample.yieldCompletedAtMs = yieldCompletedAtMs;
sample.yieldWaitMs = yieldWaitMs;
sample.yieldPostRafTimeoutMs =
	yieldResult.rafCallbackAtMs != null
		? Math.max(0, yieldCompletedAtMs - yieldResult.rafCallbackAtMs)
		: undefined;
yieldWaitTotalMs += yieldWaitMs;
yieldWaitMaxMs = Math.max(yieldWaitMaxMs, yieldWaitMs);
yieldCount += 1;
```

Push the sample after optional yield. Preserve abort checks exactly as they exist today.

- [ ] **Step 3: Publish breathing trace**

Extend `publishExposurePrecomputeDiagnostics()` params and its `OnDemandTimings` assignment to carry `exposureSchedulerBreathingTrace?: ExposureSchedulerBreathingTrace` from `onDemandDiagnostics.ts`.

For chunked mode, pass:

```ts
exposureSchedulerBreathingTrace: buildBoundedExposureBreathingTrace({
	mode: exposureScheduling.mode,
	maxWorkgroupsPerSlice: exposureScheduling.maxWorkgroupsPerSlice,
	samples: sliceSamples,
	queueWaitTotalMs,
	queueWaitMaxMs,
	queueWaitMinMs: queueWaitMinMs === Number.POSITIVE_INFINITY ? 0 : queueWaitMinMs,
	encodeTotalMs: commandEncodeTotalMs,
	yieldWaitTotalMs,
	yieldWaitMaxMs,
	submitCount,
	yieldCount
})
```

For single-submit mode, publish a one-slice trace with `yieldCount=0` so the artifact can compare modes without special casing.

- [ ] **Step 4: Add behavior tests**

In `viewer/tests/compute/webgpuUtciPipeline.behavior.test.ts`, add tests that verify the trace is bounded and stable:

```ts
it('keeps exposure breathing trace samples bounded', () => {
	const samples = Array.from({ length: 63 }, (_, sliceIndex) => ({
		sliceIndex,
		pointStart: sliceIndex * 1024,
		pointCount: 1024,
		workgroupCount: 16,
		encodeMs: 1,
		submitAtMs: sliceIndex * 10,
		queueWaitMs: sliceIndex,
		yieldWaitMs: sliceIndex % 3
	}));
	const trace = buildBoundedExposureBreathingTrace({
		mode: 'chunked',
		maxWorkgroupsPerSlice: 2048,
		samples,
		queueWaitTotalMs: samples.reduce((sum, sample) => sum + sample.queueWaitMs, 0),
		queueWaitMaxMs: 62,
		queueWaitMinMs: 0,
		encodeTotalMs: 63,
		yieldWaitTotalMs: samples.reduce((sum, sample) => sum + (sample.yieldWaitMs ?? 0), 0),
		yieldWaitMaxMs: 2,
		submitCount: 63,
		yieldCount: 62
	});
	expect(trace.firstSamples).toHaveLength(8);
	expect(trace.worstQueueWaitSamples).toHaveLength(8);
	expect(trace.lastSamples).toHaveLength(8);
	expect(trace.worstQueueWaitSamples[0]?.sliceIndex).toBe(62);
});
```

If the helper cannot be exported cleanly without widening production API, put the equivalent test on the public diagnostics object produced by an existing mocked pipeline test. Do not add broad new test harnesses.

- [ ] **Step 5: Run focused tests**

Run:

```powershell
cd viewer
npm test -- --run tests/compute/webgpuUtciPipeline.behavior.test.ts
```

Expected: pass. Do not commit.

---

## Task 3: Add Render-Publication Split Fields Without Behavior Changes

**Files:**

- Modify: `viewer/src/lib/components/scene/UTCIPointCloud.svelte`
- Modify: `viewer/src/lib/components/scene/utciComputeBufferRenderBridge.ts`
- Modify: `viewer/src/lib/diagnostics/selectedHourRenderPublicationDiagnostics.ts`

- [ ] **Step 1: Stamp pre-storage publication window**

In `UTCIPointCloud.svelte`, before layout-key work starts, define:

```ts
const renderPublicationPreStorageStartedAtMs = performance.now();
```

After `setComputeBufferSurfacePendingStorageInit(utciSurface)`, define:

```ts
const renderPublicationPreStorageCompletedAtMs = performance.now();
const renderPublicationPreStorageMs =
	renderPublicationPreStorageCompletedAtMs - renderPublicationPreStorageStartedAtMs;
```

Add these values to the `renderPublicationTimeline` object.

- [ ] **Step 2: Stamp storage pending flag and invalidate boundary**

Around `setComputeBufferSurfacePendingStorageInit(utciSurface)`, stamp:

```ts
const renderStoragePendingFlagStartedAtMs = performance.now();
setComputeBufferSurfacePendingStorageInit(utciSurface);
const renderStoragePendingFlagCompletedAtMs = performance.now();
```

Inside the existing `waitForRenderStorageBuffer()` call site in `copyComputeBufferIntoRenderOwnedStorage()`, stamp:

```ts
let renderStorageFirstWaitFrameRequestedAtMs: number | undefined;
let renderStorageFirstWaitFrameCompletedAtMs: number | undefined;
```

Then in the `waitForNextFrame` callback:

```ts
renderStorageFirstWaitFrameRequestedAtMs ??= performance.now();
await waitForNextFrame();
renderStorageFirstWaitFrameCompletedAtMs ??= performance.now();
```

Include these fields in the final `createRenderPublicationDiagnostics()` timeline.

- [ ] **Step 3: Stamp queue drain boundaries**

In `copyComputeBufferToRenderStorage()`, add returned fields:

```ts
queueDrainStartedAtMs: number;
queueDrainCompletedAtMs: number;
```

Set them around `await params.queue.onSubmittedWorkDone()` and return them:

```ts
const queueDrainStartedAtMs = params.now();
await params.queue.onSubmittedWorkDone();
const queueDrainCompletedAtMs = params.now();
```

In `UTCIPointCloud.svelte`, copy them into timeline fields:

```ts
renderCopyQueueDrainStartedAtMs: copyTimings.queueDrainStartedAtMs,
renderCopyQueueDrainCompletedAtMs: copyTimings.queueDrainCompletedAtMs,
renderCopyQueueDrainMs: copyTimings.queueDrainMs,
```

- [ ] **Step 4: Type-check**

Run:

```powershell
cd viewer
npm run check
```

Expected: `svelte-check found 0 errors and 0 warnings`. Do not commit.

---

## Task 4: Extend Collector Correlation And Artifact Output

**Files:**

- Modify: `viewer/tests/e2e/main-route-visual-freeze-map.spec.ts`
- Modify: `viewer/tests/e2e/main-route-cold-start-waterfall.spec.ts`
- Create: `data/performance-results/main-route-exposure-and-raf-diagnostics.json`

- [ ] **Step 1: Preserve new diagnostics in summaries**

In both collectors' `summarizeDiagnostics()` or equivalent timing extraction, preserve:

```ts
exposureSchedulerBreathingTrace: value.timings?.exposureSchedulerBreathingTrace ?? null,
renderPublicationPreStorageMs:
	numberOrNull(value.timings?.renderPublication?.renderPublicationTimeline?.renderPublicationPreStorageMs),
renderCopyQueueDrainMs:
	numberOrNull(value.timings?.renderPublication?.renderPublicationTimeline?.renderCopyQueueDrainMs)
```

Do not remove any existing proof fields.

- [ ] **Step 2: Add overlap computation helper**

In `main-route-visual-freeze-map.spec.ts`, add:

```ts
type TimingWindow = { label: string; startMs: number; endMs: number };

function overlapMs(leftStart: number, leftEnd: number, rightStart: number, rightEnd: number) {
	return Math.max(0, Math.min(leftEnd, rightEnd) - Math.max(leftStart, rightStart));
}

function buildTimingWindows(finalDiagnostics: DiagnosticsSnapshot | null): TimingWindow[] {
	const timeline = finalDiagnostics?.timings?.renderPublication?.renderPublicationTimeline;
	const exposureTrace = finalDiagnostics?.timings?.exposureSchedulerBreathingTrace;
	const windows: TimingWindow[] = [];
	for (const sample of exposureTrace?.allSliceWindows ?? []) {
		const startMs = numberOrNull(sample.startMs);
		const endMs = numberOrNull(sample.endMs);
		if (startMs != null && endMs != null) {
			windows.push({ label: `exposure-slice-${sample.sliceIndex}`, startMs, endMs });
		}
	}
	if (timeline?.controllerStatePublishedAtMs != null && timeline?.sceneSyncCompletedAtMs != null) {
		windows.push({
			label: 'render-publication-scene-sync',
			startMs: timeline.controllerStatePublishedAtMs,
			endMs: timeline.sceneSyncCompletedAtMs
		});
	}
	if (timeline?.renderCopyQueueDrainStartedAtMs && timeline?.renderCopyQueueDrainCompletedAtMs) {
		windows.push({
			label: 'render-copy-queue-drain',
			startMs: timeline.renderCopyQueueDrainStartedAtMs,
			endMs: timeline.renderCopyQueueDrainCompletedAtMs
		});
	}
	return windows;
}
```

Then add `gapOverlapSummary` to each case summary with top rAF, interval, and long-task overlaps by timing window.

- [ ] **Step 3: Write a focused sibling artifact**

Keep the existing `main-route-visual-freeze-map.json` behavior unless implementation review chooses to replace it. Add a sibling artifact path:

```ts
const EXPOSURE_AND_RAF_ARTIFACT_FILENAME = 'main-route-exposure-and-raf-diagnostics.json';
const EXPOSURE_AND_RAF_ARTIFACT_PATH = resolve(RESULTS_DIR, EXPOSURE_AND_RAF_ARTIFACT_FILENAME);
```

Write the same collected cases plus a top-level `diagnosticFocus`:

```ts
diagnosticFocus:
	'exposure breathing during cold load plus render-publication rAF correlation on main route /'
```

- [ ] **Step 4: Run source-lock and type tests**

Run:

```powershell
cd viewer
npm test -- --run tests/e2e/main-route-exposure-scheduler-collectors-source-lock.test.ts
npm run check
```

Expected: source-lock passes and type-check reports `0 errors and 0 warnings`. Do not commit.

---

## Task 5: Collect One Complete Artifact

**Files:**

- Create/refresh: `data/performance-results/main-route-exposure-and-raf-diagnostics.json`
- Refresh if intentionally preserved: `data/performance-results/main-route-visual-freeze-map.json`

- [ ] **Step 1: Run the headed collector only after unit/source-lock checks pass**

Run:

```powershell
cd viewer
npx playwright test --config=playwright.collect.config.ts tests/e2e/main-route-visual-freeze-map.spec.ts --project=chromium --workers=1 --reporter=list --timeout=900000
```

Expected:

- `1 passed`
- artifact includes Ness Tziona 0.5m on `/`
- no page errors, request failures, or crashes for the relevant case
- strong GPU-native proof remains true

- [ ] **Step 2: Validate artifact with a local parser**

Run:

```powershell
cd ..
node -e "const fs=require('fs'); const p='data/performance-results/main-route-exposure-and-raf-diagnostics.json'; const a=JSON.parse(fs.readFileSync(p,'utf8')); const nz=a.cases.find(c=>c.caseId.includes('ness-tziona')&&c.gridResolutionMeters===0.5); if(!nz) throw new Error('missing NZ 0.5m case'); const d=nz.raw.finalDiagnostics; const proof=d.selectedHourRuntimeContract||{}; const t=d.timings||{}; const timeline=t.renderPublication?.renderPublicationTimeline||{}; if(a.sourceRoute!=='/') throw new Error('wrong route'); if(d.rendererBackend!=='webgpu') throw new Error('not webgpu'); if(d.utciSurfaceSource!=='compute-buffer-selected-hour') throw new Error('wrong surface'); if(d.baseRenderTransport!=='compute-buffer-selected-hour') throw new Error('wrong transport'); if(d.baseSameDeviceForComputeAndRender!==true) throw new Error('not same device'); if(proof.visibleSelectedHourReadbackCount!==0) throw new Error('visible readback'); if(t.exposureSchedulerBreathingTrace==null) throw new Error('missing exposure breathing trace'); if(!Array.isArray(t.exposureSchedulerBreathingTrace.allSliceWindows)||t.exposureSchedulerBreathingTrace.allSliceWindows.length===0) throw new Error('missing all exposure slice windows'); if(timeline.renderPublicationPreStorageMs==null) throw new Error('missing render pre-storage split'); console.log(JSON.stringify({caseId:nz.caseId, firstVisible:nz.summary.firstSelectedHourVisibleMs, exposureTrace:true, sliceWindows:t.exposureSchedulerBreathingTrace.allSliceWindows.length, topRaf:nz.summary.topRafGapMs, topLong:nz.summary.topLongTaskMs}, null, 2));"
```

Expected: parses the collector's summarized `raw.finalDiagnostics` object, then prints case id, first visible, `exposureTrace: true`, slice-window count, `topRaf`, and `topLong`. Do not commit.

---

## Task 6: Write Evidence Note And Ranking

**Files:**

- Create: `docs/performance/main-route-exposure-and-raf-diagnostics.md`

- [ ] **Step 1: Create the evidence note**

Write `docs/performance/main-route-exposure-and-raf-diagnostics.md` with these sections:

```markdown
# Main Route Exposure And RAF Diagnostics

## Scope

- Route: `/`
- Analysis: Ness Tziona 0.5m is the decision case
- Purpose: separate desktop breathing during exposure from page-local render-publication rAF pain
- Non-goals: chunk-size tuning, lazy/background exposure fill, render-publication yielding, queue-drain removal

## Proof Boundary

List `rendererBackend`, `utciSurfaceSource`, `baseRenderTransport`, `baseSameDeviceForComputeAndRender`, `visibleSelectedHourReadbackCount`, `dataTextureBuildCount`, source route, and forbidden comparison status.

## Exposure Breathing Profile

Summarize `exposurePrecomputeMs`, scheduler mode, slice count, max/average queue wait, max/average yield wait, post-yield rAF delay, top exposure-overlapped rAF/interval/long-task gaps, and whether the desktop-breathing hypothesis remains plausible.

## Render-Publication RAF Profile

Summarize controller publish to scene sync complete, layout/proof, mesh, storage init, copy submit, queue drain, and overlap with top rAF/long-task gaps.

## Ranked Owners

Rank:

1. exposure GPU/driver saturation / breathing
2. early startup/data prep before controller run
3. layout/proof/key construction
4. mesh/surface creation
5. Three/WebGPU storage initialization
6. compute-buffer copy submit
7. queue drain
8. first render/backend init

For each owner include evidence fields, current values, confidence, and falsifier.

## Recommendation

State the next implementation spike only if the diagnostic evidence identifies a clear owner. Otherwise state that evidence is inconclusive.
```

- [ ] **Step 2: Run markdown/source sanity checks**

Run:

```powershell
rg -n "TBD|TODO|smaller chunks|lazy/background|cooperative render-publication" docs/performance/main-route-exposure-and-raf-diagnostics.md docs/superpowers/plans/2026-05-31-main-route-exposure-and-raf-diagnostics.md
git diff --check
```

Expected:

- No `TBD` or `TODO`.
- Any mention of excluded classes is explicitly in non-goals or already-tried context.
- `git diff --check` exits `0` or only reports known line-ending warnings. Do not commit.

---

## Task 7: Subagent Verification Gates

**Files:**

- No direct edits unless reviewers find gaps.

- [ ] **Step 1: Spec-compliance review subagent**

Dispatch a fresh review subagent with:

```text
Review docs/superpowers/plans/2026-05-31-main-route-exposure-and-raf-diagnostics.md and the resulting diff.
Check only spec compliance:
- no commits, no worktrees
- proof surface remains /
- proof preserves rendererBackend=webgpu, compute-buffer-selected-hour, same-device compute/render, visibleSelectedHourReadbackCount=0
- plan collects exposure breathing and rAF/render-publication diagnostics in one pass
- plan does not tune smaller chunks, lazy/background fill, or cooperative render-publication yielding
- plan does not move load cost onto scrub
- verification commands are exact and bounded
Return blocking issues first. Do not edit.
```

- [ ] **Step 2: Code-quality review subagent**

After spec compliance is clean, dispatch a fresh review subagent with:

```text
Review the diagnostics plan and diff for code quality and maintainability risks.
Focus on:
- bounded artifact size
- diagnostics-only runtime overhead
- type consistency
- collector flake risk
- preserving existing dirty files
- whether fields are sufficient to avoid another diagnostic run
Return blocking issues first. Do not edit.
```

- [ ] **Step 3: Apply only review fixes approved by the coordinator**

If reviewers find issues, patch the plan or implementation narrowly. Do not start behavior-changing optimization work.

---

## Final Verification Commands

Run these only after implementation is approved:

```powershell
cd viewer
npm test -- --run tests/compute/webgpuUtciPipeline.behavior.test.ts tests/e2e/main-route-exposure-scheduler-collectors-source-lock.test.ts
npm run check
npx playwright test --config=playwright.collect.config.ts tests/e2e/main-route-visual-freeze-map.spec.ts --project=chromium --workers=1 --reporter=list --timeout=900000
cd ..
git diff --check
```

If any command fails, report the exact failing command and do not claim the diagnostic pass is complete.

## Review Before Implementation

Before implementing, review these choices:

1. Whether to write a sibling artifact `main-route-exposure-and-raf-diagnostics.json` or extend only `main-route-visual-freeze-map.json`.
2. Whether bounded per-slice samples are enough, or whether Ness Tziona 0.5m should temporarily include all 63 slice records.
3. Whether this plan should include an optional external OS/GPU profiler note. The current plan does not, because it keeps proof inside the repo collector.
