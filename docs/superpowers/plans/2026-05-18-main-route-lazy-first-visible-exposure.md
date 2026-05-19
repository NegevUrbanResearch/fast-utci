# Main Route Lazy First-Visible Exposure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a diagnostics-gated lazy exposure path for the main route so the first selected hour can render before full solar exposure precompute completes, while proving that warm scrubbing does not regress.

**Architecture:** Keep the optimization behind an explicit query flag until proof passes. Split exposure readiness into first-visible and full-cache states: compute sky exposure plus the selected-hour solar bit first, publish the first selected hour, then complete full solar exposure in the background with diagnostics that measure first visible, first scrub during background fill, and scrub after full readiness. Preserve the existing full exposure path as the default fallback and proof baseline.

**Tech Stack:** Svelte 5, TypeScript, Vitest, Playwright, WebGPU/WGSL, Three.js, existing `window.__utciRenderDiagnostics__`, `selectedHourRuntimeContract`, and main-route performance artifacts.

---

## Required Skills For Execution

- `superpowers:using-superpowers`
- `superpowers:subagent-driven-development`
- `superpowers:test-driven-development`
- `superpowers:verification-before-completion`
- `svelte-code-writer` for any `.svelte` edits
- `webgpu` for WebGPU/WGSL pipeline changes
- `perspective-ensemble` before final strategy interpretation

## Hard Constraints

- No commits.
- No git worktrees.
- Preserve unrelated dirty files.
- Do not change overlay copy or UX text in this plan.
- Keep the existing full exposure path available and default until proof explicitly supports enabling lazy exposure.
- Do not weaken `/` route proof boundaries:
  - `utciRenderResolved='gpuNative'`
  - `utciSurfaceSource='compute-buffer-selected-hour'`
  - `baseRenderTransport='compute-buffer-selected-hour'`
  - `dataTextureBuildCount=0`
  - `selectedHourRuntimeContract.route='main'`
  - `selectedHourRuntimeContract.readbackInstrumentation='instrumented'`
  - `selectedHourRuntimeContract.visibleSelectedHourReadbackCount=0`
  - `selectedHourRuntimeContract.strongVisibleGpuPath=true`
- Do not write lazy-exposure measurements into existing warm-path artifacts:
  - `data/performance-results/main-route-selected-hour-render-diagnostics-next.json`
  - `data/performance-results/main-route-selected-hour-0_5m-base.json`
  - `data/performance-results/main-route-selected-hour-current-head.json`
- Create a sibling artifact and note:
  - `data/performance-results/main-route-lazy-first-visible-exposure.json`
  - `docs/performance/main-route-lazy-first-visible-exposure.md`

## Current Evidence

Source note: `docs/performance/main-route-cold-start-waterfall.md`

The latest cold-start waterfall shows exposure GPU queue wait is the dominant cold-path bottleneck:

| Project | Grid m | Exposure queue wait ms | Exposure span ms | First visible ms | Initial render publication ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ben-Gurion | 2 | 722.8 | 779.6 | 2632.5 | 354.2 |
| Ness-Tziona | 2 | 1460.3 | 1523.0 | 4770.7 | 844.6 |
| Ben-Gurion | 0.5 | 6053.2 | 6107.4 | 9902.4 | 2137.6 |
| Ness-Tziona | 0.5 | 16371.6 | 17738.0 | 30276.2 | 8946.5 |

CPU command encoding is negligible (`0.1-2.1 ms`). Optimizing TypeScript setup or bind-group construction is not the target.

Solar daylight ray budget and sky ray budget are equal in the current collection because both are `points * 145`. Lazy selected-hour solar can reduce the solar portion before first visible, but sky remains full-point work. This plan must measure the real improvement instead of assuming it.

## Perspective Ensemble Check

### Panel A - Council

- **First-visible performance lens:** The highest-ROI goal is reducing the blocking exposure queue wait before first selected-hour publication. Counter-move: compute only the selected-hour solar bit plus required sky exposure before first visible.
- **Scrub UX lens:** Lazy work can hurt scrubbing if background full exposure competes with early user interaction. Counter-move: measure scrub during background fill and after full readiness, and keep lazy exposure gated until those numbers are acceptable.
- **Correctness/parity lens:** Solar exposure is bit-packed point-major by total time step. Counter-move: add a solar time-offset shader path that writes into the existing full bitfield layout, so `runUtciForTimeIndex` reads the same contract.
- **Maintainability lens:** `webgpuUtciPipeline.ts` is already large. Counter-move: add narrow helper types/functions and keep collector extraction modular; do not fold strategy policy into rendering components.

### Panel B - Red Cell

- **Attack target:** A lazy first-visible exposure path that looks faster but damages scrub behavior.
- **GPU contention vector:** Background full solar fill can occupy the GPU immediately after first visible, delaying render publication or first scrub. Probe: collect `firstScrubDuringBackgroundMs`, render publication timeline, and proof counters.
- **State correctness vector:** A scrub may request an hour whose solar bit is not ready. Probe: explicit exposure coverage state and tests for selected-hour-ready vs full-ready transitions.
- **False win vector:** Sky may dominate enough that selected-hour solar does not materially reduce cold start. Probe: artifact must report first-visible exposure wait and background full-fill wait separately; if first-visible does not improve meaningfully, stop and discuss sky optimization.

### Recommendation

Build a gated prototype with proof artifacts first. Do not flip the default path in this plan. The success condition is faster first visible on 0.5m without CPU fallback, visible readback, or scrub regression during/after background fill.

## File Structure

- Modify `viewer/src/lib/compute/gpu/gpu-pipeline.ts`
  - Extend the pipeline interface with a narrow first-visible exposure entrypoint and full-exposure readiness metadata.
- Modify `viewer/src/lib/compute/compute-manager.ts`
  - Add wrappers for the new optional pipeline methods.
- Modify `viewer/src/lib/compute/gpu/shaders/exposure_solar.wgsl`
  - Add a solar time offset while preserving full point-major bitfield layout.
- Modify `viewer/src/lib/compute/gpu/webgpuUtciPipeline.ts`
  - Support selected-time solar dispatch, sky-only/full-solar modes, coverage diagnostics, and background full exposure completion.
- Modify `viewer/src/lib/compute/on-demand/onDemandDiagnostics.ts`
  - Add typed lazy exposure readiness/timing fields.
- Modify `viewer/src/lib/compute/selected-hour/liveUtciSelectedHourSession.ts`
  - Gate lazy exposure by parameter, request first-visible exposure for the initial hour, expose a background full exposure starter, and enforce scrub readiness.
- Modify `viewer/src/lib/compute/selected-hour/liveSelectedHourController.ts`
  - Start background full exposure only after render diagnostics acknowledge the compute-buffer surface as visible.
- Modify `viewer/src/routes/main/liveSelectedHour.ts`
  - Thread the route-level lazy exposure flag into live session creation.
- Modify `viewer/src/routes/+page.svelte`
  - Read `utciLazyExposure=1` from the query string and pass it to the main-route live selected-hour path.
- Modify `viewer/src/lib/components/ui/RadialTimePicker.svelte`
  - Add stable test ids for hour/month scrubbing in the e2e collector.
- Modify `viewer/tests/compute/compute-manager-on-demand.test.ts`
  - Unit-test optional wrapper delegation and fallback errors.
- Modify `viewer/tests/compute/webgpu-on-demand-source-locks.test.ts`
  - Source-guard shader offset and no-all-hours-output regression.
- Modify `viewer/tests/compute/live-selected-hour-session.test.ts`
  - Unit-test lazy exposure state transitions and no eager full exposure before first selected-hour dispatch.
- Create `viewer/tests/e2e/main-route-lazy-first-visible-exposure.spec.ts`
  - Collect baseline vs lazy cold-start and scrub proof.
- Create `docs/performance/main-route-lazy-first-visible-exposure.md`
  - Summarize the collected proof and final recommendation.

## Lazy Artifact Schema

Only `data/performance-results/main-route-lazy-first-visible-exposure.json` may receive lazy exposure output in this plan.

```ts
type MainRouteLazyFirstVisibleExposureArtifact = {
	collectedOn: string;
	sourceRoute: '/';
	collectionMethod: string;
	cases: Array<{
		projectLabel: string;
		analysisId: string;
		gridResolutionMeters: 2 | 0.5;
		mode: 'baseline-full-exposure' | 'lazy-first-visible-exposure';
		colorMode: 'normalized';
		pointCount: number;
		sourceUrl: string;
		firstVisibleMs: number | null;
		firstExposureQueueWaitMs: number | null;
		backgroundFullExposureQueueWaitMs: number | null;
		firstScrubAfterVisibleMs: number | null;
		firstScrubDuringBackgroundMs: number | null;
		firstScrubAfterFullExposureMs: number | null;
		visibleAcknowledgedAtMs: number | null;
		backgroundFullExposureStartedAtMs: number | null;
		backgroundStartDeltaMs: number | null;
		preScrubExposureCoverageState:
			| 'none'
			| 'selected-hour-ready'
			| 'background-full-running'
			| 'full-ready'
			| 'failed'
			| null;
		exposureCoverage: {
			state: 'none' | 'selected-hour-ready' | 'background-full-running' | 'full-ready' | 'failed';
			selectedTimeIndex: number | null;
			fullExposureReady: boolean;
			backgroundStartedAfterFirstVisible: boolean;
			backgroundFullExposureStartedAtMs?: number;
			backgroundFullExposureCompletedAtMs?: number;
			visibleAcknowledgedAtMs?: number;
		};
		timings: Record<string, number | null>;
		coldStart: Record<string, number | null>;
		renderPublication: Record<string, unknown> | null;
		proof: Record<string, unknown>;
		assertions: {
			pythonBinDebugComparisonFieldsAbsent: true;
			forbiddenComparisonFieldsPresent: string[];
			forbiddenRequestUrls: string[];
			memoryScope: 'utci-owned-webgpu-buffers';
			noCpuUploadFallback: true;
			noVisibleReadback: true;
		};
	}>;
};
```

---

### Task 1: Add Lazy Exposure Pipeline Contracts

**Files:**
- Modify: `viewer/src/lib/compute/gpu/gpu-pipeline.ts`
- Modify: `viewer/src/lib/compute/compute-manager.ts`
- Modify: `viewer/src/lib/compute/on-demand/onDemandDiagnostics.ts`
- Test: `viewer/tests/compute/compute-manager-on-demand.test.ts`
- Test: `viewer/tests/compute/onDemandDiagnostics.test.ts`

- [ ] **Step 1: Write failing ComputeManager wrapper tests**

Add these tests to `viewer/tests/compute/compute-manager-on-demand.test.ts`:

```ts
it('runExposureForTimeIndex delegates when supported', async () => {
	const params = {
		timeIndex: 168,
		numPoints: 8,
		numHours: 24,
		numMonths: 12
	};
	const runExposureForTimeIndex = vi.fn().mockResolvedValue(undefined);
	const pipeline: UTCIComputePipeline = {
		...basePipeline(),
		runExposureForTimeIndex
	};
	const manager = new ComputeManager(pipeline, { numMonths: 12, numHoursPerDay: 24 });

	await manager.runExposureForTimeIndex(params);

	expect(runExposureForTimeIndex).toHaveBeenCalledTimes(1);
	expect(runExposureForTimeIndex).toHaveBeenCalledWith(params);
});

it('runExposureForTimeIndex throws clearly when unsupported', async () => {
	const manager = new ComputeManager(basePipeline(), { numMonths: 12, numHoursPerDay: 24 });

	await expect(
		manager.runExposureForTimeIndex({
			timeIndex: 168,
			numPoints: 8,
			numHours: 24,
			numMonths: 12
		})
	).rejects.toThrow(/does not support selected-hour exposure precompute/i);
});

it('runFullSolarExposurePrecompute delegates when supported', async () => {
	const params = {
		numPoints: 8,
		numHours: 24,
		numMonths: 12
	};
	const runFullSolarExposurePrecompute = vi.fn().mockResolvedValue(undefined);
	const pipeline: UTCIComputePipeline = {
		...basePipeline(),
		runFullSolarExposurePrecompute
	};
	const manager = new ComputeManager(pipeline, { numMonths: 12, numHoursPerDay: 24 });

	await manager.runFullSolarExposurePrecompute(params);

	expect(runFullSolarExposurePrecompute).toHaveBeenCalledTimes(1);
	expect(runFullSolarExposurePrecompute).toHaveBeenCalledWith(params);
});
```

- [ ] **Step 2: Run the failing wrapper tests**

Run:

```powershell
cd viewer
npx vitest run tests/compute/compute-manager-on-demand.test.ts --no-file-parallelism --maxWorkers=1
```

Expected: fail because `runExposureForTimeIndex` and `runFullSolarExposurePrecompute` do not exist.

- [ ] **Step 3: Add pipeline interface types**

In `viewer/src/lib/compute/gpu/gpu-pipeline.ts`, add these exported types near `ExposurePrecomputeParams`:

```ts
export interface SelectedTimeExposurePrecomputeParams extends ExposurePrecomputeParams {
	timeIndex: number;
}

export interface FullSolarExposurePrecomputeParams extends ExposurePrecomputeParams {
	visibleAcknowledgedAtMs?: number;
}

export interface ExposureCoverageDiagnostics {
	state: 'none' | 'selected-hour-ready' | 'background-full-running' | 'full-ready' | 'failed';
	selectedTimeIndex: number | null;
	fullExposureReady: boolean;
	backgroundStartedAfterFirstVisible: boolean;
	visibleAcknowledgedAtMs?: number;
	backgroundFullExposureStartedAtMs?: number;
	backgroundFullExposureCompletedAtMs?: number;
	lastErrorMessage?: string;
}
```

Then add optional methods to `UTCIComputePipeline`:

```ts
runExposureForTimeIndex?(params: SelectedTimeExposurePrecomputeParams): Promise<void>;

runFullSolarExposurePrecompute?(params: FullSolarExposurePrecomputeParams): Promise<void>;
```

- [ ] **Step 4: Add ComputeManager wrappers**

In `viewer/src/lib/compute/compute-manager.ts`, import `SelectedTimeExposurePrecomputeParams` and `FullSolarExposurePrecomputeParams`, then add methods next to `runExposurePrecompute(...)`:

```ts
async runExposureForTimeIndex(params: SelectedTimeExposurePrecomputeParams): Promise<void> {
	if (!this.pipeline.runExposureForTimeIndex) {
		throw new Error('The configured UTCI pipeline does not support selected-hour exposure precompute.');
	}
	return this.pipeline.runExposureForTimeIndex(params);
}

async runFullSolarExposurePrecompute(params: FullSolarExposurePrecomputeParams): Promise<void> {
	if (!this.pipeline.runFullSolarExposurePrecompute) {
		throw new Error('The configured UTCI pipeline does not support full solar exposure precompute.');
	}
	return this.pipeline.runFullSolarExposurePrecompute(params);
}
```

- [ ] **Step 5: Add diagnostics fields**

In `viewer/src/lib/compute/on-demand/onDemandDiagnostics.ts`, add these optional fields to `OnDemandTimings`:

```ts
firstVisibleExposureQueueWaitMs?: number;
firstVisibleExposureCommandEncodeMs?: number;
backgroundFullSolarExposureQueueWaitMs?: number;
backgroundFullSolarExposureCommandEncodeMs?: number;
```

Add these optional fields to `OnDemandRuntimeDiagnostics`:

```ts
lazyExposureEnabled?: boolean;
exposureCoverage?: ExposureCoverageDiagnostics;
```

Import `ExposureCoverageDiagnostics` from `$lib/compute/gpu/gpu-pipeline`.

- [ ] **Step 6: Add diagnostics copy/reset tests**

In `viewer/tests/compute/onDemandDiagnostics.test.ts`, add:

```ts
it('preserves lazy exposure coverage diagnostics while copying runtime diagnostics', () => {
	const diagnostics = {
		...createEmptyOnDemandDiagnostics(),
		lazyExposureEnabled: true,
		exposureCoverage: {
			state: 'selected-hour-ready',
			selectedTimeIndex: 168,
			fullExposureReady: false,
			backgroundStartedAfterFirstVisible: false
		},
		timings: {
			firstVisibleExposureQueueWaitMs: 123,
			backgroundFullSolarExposureQueueWaitMs: 456
		}
	};

	expect(diagnostics.exposureCoverage).toMatchObject({
		state: 'selected-hour-ready',
		selectedTimeIndex: 168
	});
	expect(diagnostics.timings.firstVisibleExposureQueueWaitMs).toBe(123);
});
```

- [ ] **Step 7: Run wrapper and diagnostics tests**

Run:

```powershell
cd viewer
npx vitest run tests/compute/compute-manager-on-demand.test.ts tests/compute/onDemandDiagnostics.test.ts --no-file-parallelism --maxWorkers=1
```

Expected: pass.

---

### Task 2: Support Selected-Time Solar Exposure In WebGPU

**Files:**
- Modify: `viewer/src/lib/compute/gpu/shaders/exposure_solar.wgsl`
- Modify: `viewer/src/lib/compute/gpu/webgpuUtciPipeline.ts`
- Test: `viewer/tests/compute/webgpu-on-demand-source-locks.test.ts`

- [ ] **Step 1: Write source guard tests for solar time offset**

In `viewer/tests/compute/webgpu-on-demand-source-locks.test.ts`, read `exposure_solar.wgsl` like the existing on-demand shader source:

```ts
const solarExposureShaderSource = readFileSync(
	resolve(testDir, '../../src/lib/compute/gpu/shaders/exposure_solar.wgsl'),
	'utf8'
);
```

Add:

```ts
it('supports selected-time solar exposure without changing point-major bit layout', () => {
	expect(solarExposureShaderSource.includes('time_offset')).toBe(true);
	expect(solarExposureShaderSource.includes('let time_idx = local_time_idx + params.time_offset')).toBe(true);
	expect(solarExposureShaderSource.includes('let flat_index = point_idx * params.num_time_steps + time_idx')).toBe(true);
	expect(solarExposureShaderSource.includes('if (time_idx >= params.num_time_steps)')).toBe(true);
});

it('keeps lazy exposure entrypoints separate from all-hours UTCI allocation', () => {
	expect(source.includes('runExposureForTimeIndex')).toBe(true);
	expect(source.includes('runFullSolarExposurePrecompute')).toBe(true);
	const selectedExposureSource = getSection(
		'async runExposureForTimeIndex',
		'\n\n\tasync runFullSolarExposurePrecompute'
	);
	expect(selectedExposureSource.includes('this.utciBuffer')).toBe(false);
	expect(selectedExposureSource.includes('this.mrtBuffer')).toBe(false);
});
```

- [ ] **Step 2: Run the failing source guard**

Run:

```powershell
cd viewer
npx vitest run tests/compute/webgpu-on-demand-source-locks.test.ts --no-file-parallelism --maxWorkers=1
```

Expected: fail because shader offset and entrypoints do not exist.

- [ ] **Step 3: Change the solar shader params**

In `viewer/src/lib/compute/gpu/shaders/exposure_solar.wgsl`, change `Params` to:

```wgsl
struct Params {
	num_points: u32,
	num_time_steps: u32,
	point_offset: u32,
	time_offset: u32,
}
```

Then change the start of `main` to:

```wgsl
let point_idx = global_id.x + params.point_offset;
let local_time_idx = global_id.y;
let time_idx = local_time_idx + params.time_offset;

if (point_idx >= params.num_points || time_idx >= params.num_time_steps) {
	return;
}
```

Keep the flat index exactly:

```wgsl
let flat_index = point_idx * params.num_time_steps + time_idx;
```

- [ ] **Step 4: Add an exposure pass plan helper**

In `viewer/src/lib/compute/gpu/webgpuUtciPipeline.ts`, add local types near `ExposurePassEncodeMetrics`:

```ts
type SolarExposureDispatch = {
	timeOffset: number;
	timeSteps: number;
	diagnosticLabel: 'full-solar' | 'selected-time-solar';
};

type ExposurePassPlan = {
	solar: SolarExposureDispatch | null;
	sky: boolean;
};
```

Change `encodeExposurePasses` signature to accept `plan?: ExposurePassPlan` and replace the first lines of the function with:

```ts
private async encodeExposurePasses(params: {
	encoder: GPUCommandEncoder;
	numPoints: number;
	totalTimeSteps: number;
	workgroupSize: number;
	solarPipeline: GPUComputePipeline;
	skyPipeline: GPUComputePipeline;
	plan?: ExposurePassPlan;
}): Promise<ExposurePassEncodeMetrics> {
	const plan = params.plan ?? {
		solar: { timeOffset: 0, timeSteps: params.totalTimeSteps, diagnosticLabel: 'full-solar' },
		sky: true
	};
	const encodeStartedAt = performance.now();
	const { encoder, numPoints, totalTimeSteps, workgroupSize, solarPipeline, skyPipeline } = params;
	const pointChunks = createPointDispatchChunks(numPoints, workgroupSize);
	const transientUniformBuffers: GPUBuffer[] = [];
	const hasBvh = this.bvhNodeBuffer && this.bvhIndexBuffer && this.bvhVertexBuffer && this.bvhParamsBuffer;
	let solarEncodeMs = 0;
	let skyEncodeMs = 0;
	let solarDispatchCount = 0;
	let skyDispatchCount = 0;
}
```

- [ ] **Step 5: Update solar dispatch to use the plan**

Replace the solar pass block in `encodeExposurePasses(...)` with:

```ts
if (plan.solar && hasBvh && this.gridPointsBuffer && this.sunVectorsBuffer && this.solarExposureBuffer) {
	const solarEncodeStartedAt = performance.now();
	const solarTimeOffset = plan.solar.timeOffset;
	const solarDispatchTimeSteps = plan.solar.timeSteps;
	this.ranExposurePassesThisRun = true;
	const solarPass = encoder.beginComputePass();
	solarPass.setPipeline(solarPipeline);
	solarPass.setBindGroup(1, this.createBvhBindGroup(solarPipeline));
	for (const chunk of pointChunks) {
		const solarParamsBuffer = this.createUintParamsBuffer(
			new Uint32Array([numPoints, totalTimeSteps, chunk.pointOffset, solarTimeOffset]),
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
		solarPass.dispatchWorkgroups(chunk.workgroupsX, solarDispatchTimeSteps, 1);
		solarDispatchCount += 1;
	}
	solarPass.end();
	solarEncodeMs = performance.now() - solarEncodeStartedAt;
}
```

Do not change the default full solar call sites.

- [ ] **Step 6: Update sky dispatch to use the plan**

Replace the sky pass block in `encodeExposurePasses(...)` with:

```ts
if (plan.sky && hasBvh && this.gridPointsBuffer && this.domeVectorsBuffer && this.domeWeightsBuffer && this.skyExposureBuffer) {
	const skyEncodeStartedAt = performance.now();
	this.ranExposurePassesThisRun = true;
	const skyPass = encoder.beginComputePass();
	skyPass.setPipeline(skyPipeline);
	skyPass.setBindGroup(1, this.createBvhBindGroup(skyPipeline));
	for (const chunk of pointChunks) {
		const skyParamsBuffer = this.createUintParamsBuffer(
			new Uint32Array([numPoints, 145, chunk.pointOffset, 0]),
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
```

Keep the existing return object at the end of `encodeExposurePasses(...)`; it should still return `transientUniformBuffers`, `encodeMs`, `solarEncodeMs`, `skyEncodeMs`, `pointChunks`, `solarDispatchCount`, and `skyDispatchCount`.

- [ ] **Step 7: Add selected-time and full-solar methods**

In `webgpuUtciPipeline.ts`, add:

```ts
async runExposureForTimeIndex(params: {
	timeIndex: number;
	numPoints: number;
	numHours: number;
	numMonths: number;
}): Promise<void> {
	if (!this.weatherData) {
		throw new Error('WebGPU UTCI pipeline: weather data not uploaded');
	}
	const { timeIndex, numPoints, numHours, numMonths } = params;
	const totalTimeSteps = numHours * numMonths;
	const [solarPipeline, skyPipeline] = await Promise.all([
		this.ensureSolarPipeline(),
		this.ensureSkyPipeline()
	]);
	await this.ensureWeatherBuffer();
	this.ranExposurePassesThisRun = false;
	const encoder = this.device.createCommandEncoder();
	const startedAt = performance.now();
	const exposureMetrics = await this.encodeExposurePasses({
		encoder,
		numPoints,
		totalTimeSteps,
		workgroupSize: 64,
		solarPipeline,
		skyPipeline,
		plan: {
			solar: { timeOffset: timeIndex, timeSteps: 1, diagnosticLabel: 'selected-time-solar' },
			sky: true
		}
	});
	const commandEncodeMs = performance.now() - startedAt;
	const queueStartedAt = performance.now();
	this.queue.submit([encoder.finish()]);
	await this.queue.onSubmittedWorkDone();
	const queueWaitMs = performance.now() - queueStartedAt;
	this.destroyTransientUniformBuffers(exposureMetrics.transientUniformBuffers);
	this.onDemandDiagnostics = {
		...this.onDemandDiagnostics,
		lazyExposureEnabled: true,
		exposureCoverage: {
			state: 'selected-hour-ready',
			selectedTimeIndex: timeIndex,
			fullExposureReady: false,
			backgroundStartedAfterFirstVisible: false
		},
		timings: {
			...this.onDemandDiagnostics.timings,
			firstVisibleExposureQueueWaitMs: queueWaitMs,
			firstVisibleExposureCommandEncodeMs: commandEncodeMs
		}
	};
	this.lastConfig = { numPoints, numHours, numMonths };
}

async runFullSolarExposurePrecompute(params: {
	numPoints: number;
	numHours: number;
	numMonths: number;
	visibleAcknowledgedAtMs?: number;
}): Promise<void> {
	if (!this.weatherData) {
		throw new Error('WebGPU UTCI pipeline: weather data not uploaded');
	}
	const { numPoints, numHours, numMonths } = params;
	const totalTimeSteps = numHours * numMonths;
	const solarPipeline = await this.ensureSolarPipeline();
	const skyPipeline = await this.ensureSkyPipeline();
	const encoder = this.device.createCommandEncoder();
	const startedAt = performance.now();
	const exposureMetrics = await this.encodeExposurePasses({
		encoder,
		numPoints,
		totalTimeSteps,
		workgroupSize: 64,
		solarPipeline,
		skyPipeline,
		plan: {
			solar: { timeOffset: 0, timeSteps: totalTimeSteps, diagnosticLabel: 'full-solar' },
			sky: false
		}
	});
	const commandEncodeMs = performance.now() - startedAt;
	this.onDemandDiagnostics = {
		...this.onDemandDiagnostics,
		lazyExposureEnabled: true,
		exposureCoverage: {
			...(this.onDemandDiagnostics.exposureCoverage ?? {
				state: 'none',
				selectedTimeIndex: null,
				fullExposureReady: false,
				backgroundStartedAfterFirstVisible: false
			}),
			state: 'background-full-running',
			backgroundStartedAfterFirstVisible:
				typeof params.visibleAcknowledgedAtMs === 'number',
			visibleAcknowledgedAtMs: params.visibleAcknowledgedAtMs,
			backgroundFullExposureStartedAtMs: performance.now()
		}
	};
	const queueStartedAt = performance.now();
	this.queue.submit([encoder.finish()]);
	await this.queue.onSubmittedWorkDone();
	const queueWaitMs = performance.now() - queueStartedAt;
	this.destroyTransientUniformBuffers(exposureMetrics.transientUniformBuffers);
	this.onDemandDiagnostics = {
		...this.onDemandDiagnostics,
		exposureCoverage: {
			...(this.onDemandDiagnostics.exposureCoverage ?? {
				selectedTimeIndex: null,
				backgroundStartedAfterFirstVisible: true
			}),
			state: 'full-ready',
			fullExposureReady: true,
			backgroundStartedAfterFirstVisible: true,
			backgroundFullExposureCompletedAtMs: performance.now()
		},
		timings: {
			...this.onDemandDiagnostics.timings,
			backgroundFullSolarExposureQueueWaitMs: queueWaitMs,
			backgroundFullSolarExposureCommandEncodeMs: commandEncodeMs
		}
	};
	this.lastConfig = { numPoints, numHours, numMonths };
}
```

If TypeScript requires small local helpers to avoid repeating the default coverage object, add:

```ts
function defaultExposureCoverage(): ExposureCoverageDiagnostics {
	return {
		state: 'none',
		selectedTimeIndex: null,
		fullExposureReady: false,
		backgroundStartedAfterFirstVisible: false
	};
}
```

- [ ] **Step 8: Run source guard and TypeScript**

Run:

```powershell
cd viewer
npx vitest run tests/compute/webgpu-on-demand-source-locks.test.ts --no-file-parallelism --maxWorkers=1
npx tsc --noEmit --pretty false --project tsconfig.json
```

Expected: both pass.

---

### Task 3: Add Lazy Exposure Session State

**Files:**
- Modify: `viewer/src/lib/compute/selected-hour/liveUtciSelectedHourSession.ts`
- Test: `viewer/tests/compute/live-selected-hour-session.test.ts`

- [ ] **Step 1: Extend the existing ComputeManager mock**

In `viewer/tests/compute/live-selected-hour-session.test.ts`, update the mocked `ComputeManager` class to expose the new methods:

```ts
runExposurePrecompute = vi.fn(async () => undefined);
runExposureForTimeIndex = vi.fn(async () => undefined);
runFullSolarExposurePrecompute = vi.fn(async () => undefined);
runUtciForTimeIndex = vi.fn(async () => mockState.outputOverride ?? { gpuBuffer: mockState.gpuBuffer });
```

This replaces the current adjacent `runExposurePrecompute` and `runUtciForTimeIndex` mock field declarations.

- [ ] **Step 2: Write failing session tests**

Add these tests to `viewer/tests/compute/live-selected-hour-session.test.ts`:

```ts
it('uses selected-hour exposure before first lazy selected-hour run', async () => {
	const session = await prepareSelectedHourLiveSession({
		analysisId: 'analysis-a',
		base: createFullDayBaseAnalysis(),
		model: {} as Group,
		epwUrl: '/weather.epw',
		signal: new AbortController().signal,
		preferredDevice: mockState.rendererDevice,
		gridResolution: 2,
		lazyExposure: true
	});

	const result = await session.runSelectedHour({
		monthIndex: 7,
		hourIndex: 0,
		timeIndex: 168,
		colorMode: 'normalized',
		preferGpuResident: true,
		rendererDevice: mockState.rendererDevice
	});

	const manager = mockState.constructors[0];
	expect(manager.runExposureForTimeIndex).toHaveBeenCalledWith({
		timeIndex: 168,
		numPoints: expect.any(Number),
		numHours: 24,
		numMonths: 12
	});
	expect(manager.runExposurePrecompute).not.toHaveBeenCalled();
	expect(result.diagnostics.lazyExposureEnabled).toBe(true);
	expect(result.diagnostics.exposureCoverage).toMatchObject({
		state: 'selected-hour-ready',
		selectedTimeIndex: 168,
		fullExposureReady: false
	});
	expect(result.diagnostics.timings.firstVisibleExposureQueueWaitMs).toEqual(expect.any(Number));
});

it('runs full exposure for non-lazy selected-hour sessions', async () => {
	const session = await prepareSelectedHourLiveSession({
		analysisId: 'analysis-a',
		base: createFullDayBaseAnalysis(),
		model: {} as Group,
		epwUrl: '/weather.epw',
		signal: new AbortController().signal,
		preferredDevice: mockState.rendererDevice,
		gridResolution: 2,
		lazyExposure: false
	});

	await session.runSelectedHour({
		monthIndex: 7,
		hourIndex: 0,
		timeIndex: 168,
		colorMode: 'normalized',
		preferGpuResident: true,
		rendererDevice: mockState.rendererDevice
	});

	const manager = mockState.constructors[0];
expect(manager.runExposurePrecompute).toHaveBeenCalledTimes(1);
expect(manager.runExposureForTimeIndex).not.toHaveBeenCalled();
});
```

The assertions are the contract: lazy first selected hour uses selected-hour exposure, default path stays full exposure.

Also update `mockState.runtimeDiagnostics` in the lazy test before `runSelectedHour(...)`:

```ts
mockState.runtimeDiagnostics = {
	...mockState.runtimeDiagnostics,
	lazyExposureEnabled: true,
	exposureCoverage: {
		state: 'selected-hour-ready',
		selectedTimeIndex: 168,
		fullExposureReady: false,
		backgroundStartedAfterFirstVisible: false
	},
	timings: {
		...mockState.runtimeDiagnostics.timings,
		firstVisibleExposureQueueWaitMs: 42,
		firstVisibleExposureCommandEncodeMs: 1
	}
};
```

This proves the live-session diagnostics snapshot propagates lazy exposure fields before route-level diagnostics consume them.

- [ ] **Step 3: Run the failing tests**

Run:

```powershell
cd viewer
npx vitest run tests/compute/live-selected-hour-session.test.ts --no-file-parallelism --maxWorkers=1
```

Expected: fail because `lazyExposure` and selected-hour exposure state do not exist.

- [ ] **Step 4: Add session params and state**

In `liveUtciSelectedHourSession.ts`, add `lazyExposure?: boolean` to the prepare params type.

Add this method to `SelectedHourLiveSession`:

```ts
startBackgroundFullExposurePrecompute(params: { visibleAcknowledgedAtMs: number }): void;
```

Add to `PreparedSessionState`:

```ts
lazyExposure: boolean;
fullExposureReady: boolean;
fullExposurePromise: Promise<void> | null;
selectedExposureReadyTimeIndices: Set<number>;
```

Initialize in `prepareSelectedHourLiveSession`:

```ts
lazyExposure: params.lazyExposure === true,
fullExposureReady: false,
fullExposurePromise: null,
selectedExposureReadyTimeIndices: new Set<number>(),
```

- [ ] **Step 5: Split exposure ensure helpers**

Replace `ensureExposurePrecompute(state)` with three helpers:

```ts
async function ensureFullExposurePrecompute(state: PreparedSessionState): Promise<void> {
	if (state.fullExposureReady || state.exposureReady) return;
	if (!state.exposurePrecomputePromise) {
		const exposurePrecomputeStartedAt = performance.now();
		state.lifecycleTimings.coldStart ??= {};
		state.lifecycleTimings.coldStart.exposurePrecomputeStartedAtMs ??= exposurePrecomputeStartedAt;
		state.exposurePrecomputePromise = state.computeManager
			.runExposurePrecompute({
				numPoints: state.numPoints,
				numHours: state.numHours,
				numMonths: state.numMonths
			})
			.then(() => {
				state.exposureReady = true;
				state.fullExposureReady = true;
				state.lifecycleTimings.coldStart ??= {};
				state.lifecycleTimings.coldStart.exposurePrecomputeCompletedAtMs ??= performance.now();
			})
			.finally(() => {
				state.exposurePrecomputePromise = null;
			});
	}
	await state.exposurePrecomputePromise;
	ensureNotAborted(state.signal);
}

async function ensureSelectedHourExposurePrecompute(
	state: PreparedSessionState,
	timeIndex: number
): Promise<void> {
	if (state.fullExposureReady || state.selectedExposureReadyTimeIndices.has(timeIndex)) return;
	const exposurePrecomputeStartedAt = performance.now();
	state.lifecycleTimings.coldStart ??= {};
	state.lifecycleTimings.coldStart.exposurePrecomputeStartedAtMs ??= exposurePrecomputeStartedAt;
	await state.computeManager.runExposureForTimeIndex({
		timeIndex,
		numPoints: state.numPoints,
		numHours: state.numHours,
		numMonths: state.numMonths
	});
	state.selectedExposureReadyTimeIndices.add(timeIndex);
	state.lifecycleTimings.coldStart.exposurePrecomputeCompletedAtMs ??= performance.now();
}

function scheduleBackgroundFullExposurePrecompute(
	state: PreparedSessionState,
	params: { visibleAcknowledgedAtMs: number }
): void {
	if (!state.lazyExposure || state.fullExposureReady || state.fullExposurePromise) return;
	state.fullExposurePromise = state.computeManager
		.runFullSolarExposurePrecompute({
			numPoints: state.numPoints,
			numHours: state.numHours,
			numMonths: state.numMonths,
			visibleAcknowledgedAtMs: params.visibleAcknowledgedAtMs
		})
		.then(() => {
			state.fullExposureReady = true;
			state.exposureReady = true;
		})
		.catch((error) => {
			console.error('[live-selected-hour] background full exposure precompute failed.', error);
		})
		.finally(() => {
			state.fullExposurePromise = null;
		});
}
```

This helper must not be called from `runSelectedHour`. The controller will call it only after render diagnostics acknowledge the first compute-buffer surface as visible.

- [ ] **Step 6: Use lazy exposure in `runSelectedHour`**

Replace:

```ts
await ensureExposurePrecompute(state);
```

with:

```ts
if (state.lazyExposure) {
	await ensureSelectedHourExposurePrecompute(state, params.timeIndex);
} else {
	await ensureFullExposurePrecompute(state);
}
```

Expose the background starter on the returned session:

```ts
startBackgroundFullExposurePrecompute(params) {
	scheduleBackgroundFullExposurePrecompute(state, params);
}
```

Do not await this background promise in the first selected-hour path and do not call it from `runSelectedHour`.

- [ ] **Step 7: Add scrub policy for not-yet-ready hours**

At the same exposure gate, if `state.lazyExposure` is true and the requested `params.timeIndex` is not ready while background full exposure is running, compute that selected hour exposure first:

```ts
if (state.lazyExposure && !state.fullExposureReady) {
	await ensureSelectedHourExposurePrecompute(state, params.timeIndex);
} else {
	await ensureFullExposurePrecompute(state);
}
```

This policy avoids publishing incorrect scrub hours. It may make a very early scrub pay for its own selected-hour exposure, so the e2e collector must measure it.

- [ ] **Step 8: Run session tests**

Run:

```powershell
cd viewer
npx vitest run tests/compute/live-selected-hour-session.test.ts --no-file-parallelism --maxWorkers=1
```

Expected: pass.

---

### Task 4: Gate Lazy Exposure From The Main Route

**Files:**
- Modify: `viewer/src/lib/compute/selected-hour/liveSelectedHourController.ts`
- Modify: `viewer/src/routes/main/liveSelectedHour.ts`
- Modify: `viewer/src/routes/+page.svelte`
- Modify: `viewer/src/lib/components/ui/RadialTimePicker.svelte`
- Test: `viewer/tests/routes/main-route-live-selected-hour.test.ts`

- [ ] **Step 1: Add route diagnostics/input test**

In `viewer/tests/routes/main-route-live-selected-hour.test.ts`, add a test near the existing live selected-hour diagnostics input tests:

```ts
it('passes lazy exposure enablement through main-route diagnostics inputs', () => {
	const params = createDiagnosticsParams();
	params.liveRouteState.base.runtimeDiagnostics = {
		...params.liveRouteState.base.runtimeDiagnostics,
		lazyExposureEnabled: true,
		exposureCoverage: {
			state: 'selected-hour-ready',
			selectedTimeIndex: 168,
			fullExposureReady: false,
			backgroundStartedAfterFirstVisible: false
		},
		timings: {
			...(params.liveRouteState.base.runtimeDiagnostics?.timings ?? {}),
			firstVisibleExposureQueueWaitMs: 123
		},
		trackedGpuAllocationBytes:
			params.liveRouteState.base.runtimeDiagnostics?.trackedGpuAllocationBytes ?? {
				persistentExposureBytes: 0,
				allHoursOutputBytes: 0,
				selectedHourOutputBytes: 0,
				selectedHourOutputBytesHighWatermark: 0,
				trackingScope: 'utci-owned-webgpu-buffers'
			}
	} as NonNullable<typeof params.liveRouteState.base.runtimeDiagnostics>;

	const inputs = buildMainRouteLiveSelectedHourDiagnosticsInputs({
		...params,
		selectedTimeIndex: 168
	});

	expect(inputs.lazyExposureEnabled).toBe(true);
	expect(inputs.exposureCoverage).toMatchObject({
		state: 'selected-hour-ready',
		selectedTimeIndex: 168,
		fullExposureReady: false
	});
	expect(inputs.timingsOverride?.firstVisibleExposureQueueWaitMs).toBe(123);
});
```

- [ ] **Step 2: Run the failing route test**

Run:

```powershell
cd viewer
npx vitest run tests/routes/main-route-live-selected-hour.test.ts --no-file-parallelism --maxWorkers=1
```

Expected: fail until route diagnostics copy the new fields.

- [ ] **Step 3: Thread lazy exposure into live session creation**

In `viewer/src/routes/main/liveSelectedHour.ts`, add a `lazyExposure?: boolean` param to the live selected-hour session creation call that currently calls `prepareSelectedHourLiveSession(...)`.

Pass:

```ts
lazyExposure: params.lazyExposure === true
```

into `prepareSelectedHourLiveSession(...)`.

- [ ] **Step 4: Read the query flag in `+page.svelte`**

In `viewer/src/routes/+page.svelte`, add a diagnostics/prototype-only flag:

```ts
$: utciLazyExposureEnabled = $page.url.searchParams.get('utciLazyExposure') === '1';
```

Pass it into the main-route live selected-hour setup:

```ts
lazyExposure: utciLazyExposureEnabled
```

Do not enable lazy exposure without the query flag in this plan.

- [ ] **Step 5: Publish lazy exposure diagnostics**

In `viewer/src/routes/main/liveSelectedHour.ts`, add `lazyExposureEnabled?: boolean` and `exposureCoverage?: ExposureCoverageDiagnostics` to `MainRouteLiveSelectedHourDiagnosticsParams` and to the object returned by `buildMainRouteLiveSelectedHourDiagnosticsInputs(...)`:

```ts
lazyExposureEnabled: liveRouteState.base.runtimeDiagnostics?.lazyExposureEnabled,
exposureCoverage: liveRouteState.base.runtimeDiagnostics?.exposureCoverage
	? { ...liveRouteState.base.runtimeDiagnostics.exposureCoverage }
	: undefined
```

In `viewer/src/lib/diagnostics/mainRouteUtciDiagnostics.ts`, add matching fields to `MainRouteUtciDiagnosticsInputs` and `MainRouteUtciDiagnosticsPayload`, and copy `exposureCoverage` defensively:

```ts
function copyExposureCoverageDiagnostics(
	coverage: ExposureCoverageDiagnostics | undefined
): ExposureCoverageDiagnostics | undefined {
	if (!coverage) return undefined;
	return { ...coverage };
}
```

Then include:

```ts
lazyExposureEnabled: inputs.lazyExposureEnabled,
exposureCoverage: copyExposureCoverageDiagnostics(inputs.exposureCoverage),
```

- [ ] **Step 6: Start background full exposure after visible acknowledgement**

In `viewer/src/lib/compute/selected-hour/liveSelectedHourController.ts`, inside `handleRenderSurfaceDiagnostics(...)`, call the session background starter only in the `acceptsGpuCompletion` branch after `visibleAtMs` is known:

```ts
if (acceptsGpuCompletion) {
	const visibleAtMs = performance.now();
	currentSession?.startBackgroundFullExposurePrecompute({ visibleAcknowledgedAtMs: visibleAtMs });
}
```

Add only the `currentSession?.startBackgroundFullExposurePrecompute();` line. Keep the existing state update block in that branch unchanged. Do not call this from stale, failed, CPU fallback, or non-`compute-buffer-selected-hour` paths. The session method is internally idempotent, so repeated render diagnostics must not start duplicate background full exposure work.

- [ ] **Step 7: Add stable radial picker test ids**

In `viewer/src/lib/components/ui/RadialTimePicker.svelte`, add a data-testid to the `role="slider"` dial:

```svelte
data-testid={mode === "day" ? "main-route-hour-dial" : "main-route-month-dial"}
```

Place it on the same `<div class="radial-dial" ...>` that already has `role="slider"` and keyboard/pointer handlers. This does not change visible UI text.

- [ ] **Step 8: Run route tests**

Run:

```powershell
cd viewer
npx vitest run tests/routes/main-route-live-selected-hour.test.ts --no-file-parallelism --maxWorkers=1
```

Expected: pass.

---

### Task 5: Collect Lazy First-Visible And Scrub Proof

**Files:**
- Create: `viewer/tests/e2e/main-route-lazy-first-visible-exposure.spec.ts`
- Create: `data/performance-results/main-route-lazy-first-visible-exposure.json`

- [ ] **Step 1: Create the e2e collector skeleton**

Create `viewer/tests/e2e/main-route-lazy-first-visible-exposure.spec.ts` based on `viewer/tests/e2e/main-route-cold-start-waterfall.spec.ts`, but with:

```ts
const ARTIFACT_FILENAME = 'main-route-lazy-first-visible-exposure.json';
const COLLECTED_ON = '2026-05-18';
const SOURCE_ROUTE = '/';
```

Use the same four cases:

```ts
const CASES = [
	{
		projectLabel: 'Ben-Gurion',
		analysisId: 'Ben-Gurion/20250815_grid_2m_fullday',
		gridResolutionMeters: 2
	},
	{
		projectLabel: 'Ness-Tziona',
		analysisId: 'Ness-Tziona/exploded/nes_tziona_unblock_2',
		gridResolutionMeters: 2
	},
	{
		projectLabel: 'Ben-Gurion',
		analysisId: 'Ben-Gurion/20250815_grid_2m_fullday',
		gridResolutionMeters: 0.5
	},
	{
		projectLabel: 'Ness-Tziona',
		analysisId: 'Ness-Tziona/exploded/nes_tziona_unblock_2',
		gridResolutionMeters: 0.5
	}
] as const;
```

- [ ] **Step 2: Collect baseline and lazy mode URLs**

For each case, visit both URLs:

```ts
function buildSourceUrl(caseConfig: CaseConfig, mode: 'baseline-full-exposure' | 'lazy-first-visible-exposure') {
	const gridQuery =
		caseConfig.gridResolutionMeters === 0.5
			? `&gridResolution=${caseConfig.gridResolutionMeters}`
			: '';
	const lazyQuery = mode === 'lazy-first-visible-exposure' ? '&utciLazyExposure=1' : '';
	return `/?analysis=${encodeURIComponent(caseConfig.analysisId)}${gridQuery}&utciRender=auto&utciRenderDiagnostics=1${lazyQuery}`;
}
```

- [ ] **Step 3: Assert proof boundary for every row**

Copy the proof boundary assertions from `main-route-cold-start-waterfall.spec.ts` and keep these exact assertions:

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

- [ ] **Step 4: Measure scrub during background and after full readiness**

For baseline rows, measure `firstScrubAfterVisibleMs` immediately after first visible:

```ts
const baselineScrubStartedAt = performance.now();
const hourDial = page.getByTestId('main-route-hour-dial');
await hourDial.focus();
await page.keyboard.press('Home');
await page.keyboard.press('ArrowRight');
await waitForSelectedHourPublication(page, `${caseConfig.analysisId}|7|1`);
const firstScrubAfterVisibleMs = performance.now() - baselineScrubStartedAt;
```

After first visible in lazy mode:

```ts
const preScrubExposureCoverageState =
	diagnostics.exposureCoverage?.state ?? null;
if (preScrubExposureCoverageState !== 'background-full-running') {
	summaryNotes.push(
		`${caseConfig.projectLabel} ${caseConfig.gridResolutionMeters}m lazy scrub did not run during background fill; state=${preScrubExposureCoverageState}`
	);
}
const duringBackgroundStartedAt = performance.now();
const hourDial = page.getByTestId('main-route-hour-dial');
await hourDial.focus();
await page.keyboard.press('Home');
await page.keyboard.press('ArrowRight');
const duringBackgroundDiagnostics = await waitForSelectedHourPublication(
	page,
	`${caseConfig.analysisId}|7|1`
);
const firstScrubDuringBackgroundMs = performance.now() - duringBackgroundStartedAt;
```

Then wait for full exposure readiness:

```ts
await page.waitForFunction(
	() => window.__utciRenderDiagnostics__?.exposureCoverage?.state === 'full-ready',
	null,
	{ timeout: 60000 }
).catch(async (error) => {
	const lastDiagnostics = await page.evaluate(() => window.__utciRenderDiagnostics__);
	throw new Error(
		`Timed out waiting for lazy full exposure readiness: ${
			error instanceof Error ? error.message : String(error)
		}\nLast diagnostics:\n${JSON.stringify(lastDiagnostics, null, 2)}`
	);
});
```

If this wait times out because route diagnostics are not republished after background completion, do not increase the timeout. Add an explicit diagnostics propagation hook in Task 4: after `runFullSolarExposurePrecompute(...)` completes, the session/controller must publish updated runtime diagnostics to `liveRouteState.base.runtimeDiagnostics` without requiring a new selected-hour request. Re-run the bounded wait after adding that propagation.

Then scrub again:

```ts
const afterFullStartedAt = performance.now();
await hourDial.focus();
await page.keyboard.press('Home');
await page.keyboard.press('ArrowRight');
await page.keyboard.press('ArrowRight');
await waitForSelectedHourPublication(page, `${caseConfig.analysisId}|7|2`);
const firstScrubAfterFullExposureMs = performance.now() - afterFullStartedAt;
```

- [ ] **Step 5: Write artifact rows**

Each row must include:

```ts
{
	projectLabel,
	analysisId,
	gridResolutionMeters,
	mode,
	colorMode: 'normalized',
	pointCount,
	sourceUrl,
		firstVisibleMs,
		visibleAcknowledgedAtMs,
		firstExposureQueueWaitMs,
		backgroundFullExposureQueueWaitMs,
		backgroundFullExposureStartedAtMs,
		backgroundStartDeltaMs,
		firstScrubAfterVisibleMs,
		firstScrubDuringBackgroundMs,
		firstScrubAfterFullExposureMs,
		preScrubExposureCoverageState,
		exposureCoverage,
	timings,
	coldStart,
	renderPublication,
	proof,
	assertions
}
```

For baseline rows, set lazy-only fields to `null`:

```ts
backgroundFullExposureQueueWaitMs: null,
backgroundFullExposureStartedAtMs: null,
backgroundStartDeltaMs: null,
firstScrubDuringBackgroundMs: null,
firstScrubAfterFullExposureMs: null
```

- [ ] **Step 6: Add pass/fail thresholds as warnings, not hard failures**

The collector should assert proof boundaries hard, but should not fail solely because performance is not improved. Add a top-level `summary`:

```ts
summary: {
	lazyImprovedFirstVisible: boolean;
	lazyScrubRegressionDetected: boolean;
	notes: string[];
}
```

Compute:

```ts
lazyImprovedFirstVisible =
	lazyNess05.firstVisibleMs != null &&
	baselineNess05.firstVisibleMs != null &&
	lazyNess05.firstVisibleMs < baselineNess05.firstVisibleMs;

lazyScrubRegressionDetected =
	lazyNess05.firstScrubDuringBackgroundMs != null &&
	baselineNess05.firstScrubAfterVisibleMs != null &&
	lazyNess05.firstScrubDuringBackgroundMs > baselineNess05.firstScrubAfterVisibleMs * 1.25;
```

Also compute:

```ts
backgroundStartSequencingValid =
	lazyRows.every((row) =>
		row.backgroundStartDeltaMs == null ? false : row.backgroundStartDeltaMs >= 0
	);
```

Do not hide a regression or missing proof. Put it in `summary.notes`.

- [ ] **Step 7: Run the collector**

Run:

```powershell
cd viewer
npx playwright test tests/e2e/main-route-lazy-first-visible-exposure.spec.ts --project=chromium --workers=1 --reporter=list
```

Expected: pass if proof boundaries hold and write `data/performance-results/main-route-lazy-first-visible-exposure.json`.

---

### Task 6: Write Lazy Exposure Evidence Note

**Files:**
- Create: `docs/performance/main-route-lazy-first-visible-exposure.md`
- Read: `data/performance-results/main-route-lazy-first-visible-exposure.json`

- [ ] **Step 1: Create the note**

Create `docs/performance/main-route-lazy-first-visible-exposure.md`:

```md
# Main Route Lazy First-Visible Exposure Evidence

Date: 2026-05-18

## Scope

This note evaluates the diagnostics-gated lazy first-visible exposure path on the main route `/`. It compares default full exposure against `utciLazyExposure=1` and does not use `/debug`, `.bin`, Python reference data, or parity comparison.

JSON source: [data/performance-results/main-route-lazy-first-visible-exposure.json](../../data/performance-results/main-route-lazy-first-visible-exposure.json)

## Proof Boundary

- `rendererBackend=webgpu`
- `utciRenderResolved=gpuNative`
- `utciSurfaceSource=compute-buffer-selected-hour`
- `baseRenderTransport=compute-buffer-selected-hour`
- `dataTextureBuildCount=0`
- `selectedHourRuntimeContract.route=main`
- `selectedHourRuntimeContract.readbackInstrumentation=instrumented`
- `selectedHourRuntimeContract.strongVisibleGpuPath=true`
- `selectedHourRuntimeContract.visibleSelectedHourReadbackCount=0`
- no python/bin/debug comparison fields
- no forbidden comparison requests

## Timing Table

| Project | Grid m | Mode | First visible ms | First exposure queue wait ms | Background full exposure queue wait ms | First scrub during background ms | First scrub after full exposure ms |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |

## Diagnosis

Lazy exposure improved 0.5m first visible if the lazy `First visible ms` is lower than the matching baseline row for BG and Ness Tziona. Treat this as unproven until the table is filled from the JSON.

Scrub during background fill regressed if `First scrub during background ms` is more than `1.25x` the matching baseline `First scrub after visible ms`. Treat any such case as a blocker for default enablement.

Scrub was actually measured during background fill only if `Pre-scrub exposure coverage state` is `background-full-running`. If it is `full-ready`, the artifact did not exercise GPU contention and the recommendation must say so.

Background scheduling is proven only if `Background start delta ms` is non-negative for every lazy row. A negative or null value means full exposure started before visible acknowledgement was proven and lazy exposure must remain gated.

Background full exposure competed with render publication if lazy mode improves exposure wait but `Initial render publication ms` or first visible does not improve. In that case, keep `utciLazyExposure=1` gated and investigate scheduling/render publication before enabling.

The default path remains the full exposure path unless the artifact proves both first-visible improvement and clean scrub behavior.

## Recommendation

- `Proceed`: only if first visible improves for 0.5m and scrub proof remains clean during background fill and after full exposure.
- `Stop`: if first visible does not improve enough or scrub regression is detected.
- `Investigate sky`: if selected-hour solar laziness does not materially reduce the exposure queue wait.
```

- [ ] **Step 2: Fill exact numbers from JSON**

Use exact rounded numbers from the artifact. Do not summarize stale numbers from `main-route-cold-start-waterfall.md`.

- [ ] **Step 3: Run stale-claim scan**

Run:

```powershell
Select-String -Path docs\performance\main-route-lazy-first-visible-exposure.md -Pattern "debug|\\.bin|Python|parity|cpu-uploaded|readback|regression|default"
```

Expected: matches are allowed in scope, proof, and diagnosis. Manually confirm no proof boundary is weakened.

---

### Task 7: Final Verification And Review

**Files:**
- Verify all modified files.
- Do not commit.

- [ ] **Step 1: Run focused unit tests**

Run:

```powershell
cd viewer
npx vitest run tests/compute/compute-manager-on-demand.test.ts tests/compute/webgpu-on-demand-source-locks.test.ts tests/compute/onDemandDiagnostics.test.ts tests/compute/live-selected-hour-session.test.ts tests/routes/main-route-live-selected-hour.test.ts --no-file-parallelism --maxWorkers=1
```

Expected: pass.

- [ ] **Step 2: Run TypeScript and Svelte checks**

Run:

```powershell
cd viewer
npx tsc --noEmit --pretty false --project tsconfig.json
npm run check
```

Expected: pass.

- [ ] **Step 3: Run the lazy exposure collector**

Run:

```powershell
cd viewer
npx playwright test tests/e2e/main-route-lazy-first-visible-exposure.spec.ts --project=chromium --workers=1 --reporter=list
```

Expected: pass and write `data/performance-results/main-route-lazy-first-visible-exposure.json`.

- [ ] **Step 4: Confirm protected artifacts are untouched**

Run:

```powershell
git diff --name-only -- data/performance-results/main-route-selected-hour-render-diagnostics-next.json data/performance-results/main-route-selected-hour-0_5m-base.json data/performance-results/main-route-selected-hour-current-head.json viewer/tests/e2e/main-route-performance-0_5m.spec.ts viewer/tests/e2e/main-route-performance-baseline.spec.ts
```

Expected: no output.

- [ ] **Step 5: Run review agents**

Dispatch a spec-compliance reviewer first. It must check:

- no commits
- no git worktrees
- default full exposure path preserved
- lazy exposure gated behind `utciLazyExposure=1`
- no overlay copy/UX text changes
- first visible proof uses `/`, WebGPU, compute-buffer selected-hour path
- no CPU-upload fallback or visible readback
- scrub during background fill and after full exposure are measured
- protected warm artifacts are untouched

Only after spec compliance is clean, dispatch code-quality reviewer. It must check:

- solar shader offset preserves point-major full bitfield layout
- selected-hour exposure cannot publish an hour whose solar bit is not ready
- background full solar fill cannot corrupt selected-hour output handles
- diagnostics names distinguish first-visible exposure from full exposure
- route/component files are not bloated with compute policy
- e2e collector reports performance regressions instead of hiding them

- [ ] **Step 6: Run perspective-ensemble on the collected result**

Use `perspective-ensemble` after the artifact exists. The panel must answer:

- Did lazy exposure actually reduce the 0.5m first-visible freeze?
- Did it make early scrub slower?
- Is the next step enabling lazy exposure, splitting sky optimization, or attacking render publication?

- [ ] **Step 7: Report for discussion**

Report:

- files changed
- exact commands run and pass/fail
- baseline vs lazy first visible for BG/Ness at 0.5m
- scrub during background and after full exposure
- proof boundary status
- recommendation, explicitly marked as a recommendation for discussion, not an automatic default flip

Do not claim the optimization is safe unless the artifact proves first visible improves and scrub does not regress.

---

## Self-Review

- Spec coverage: The plan implements a gated lazy first-visible exposure path, measures first-visible and scrub behavior, and defers default enablement until proof exists.
- Placeholder scan: No `TBD`, `TODO`, or open-ended "add tests" steps remain.
- Type consistency: The new methods are `runExposureForTimeIndex` and `runFullSolarExposurePrecompute`; diagnostics use `firstVisibleExposureQueueWaitMs`, `backgroundFullSolarExposureQueueWaitMs`, and `exposureCoverage`.
- Constraint check: No commits or git worktrees are required. The default path remains full exposure unless `utciLazyExposure=1` is present.
- Risk check: The plan explicitly handles the main risk: background GPU exposure work can slow early scrubbing. The collector measures this before any default enablement.
