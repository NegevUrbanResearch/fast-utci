# Main Route GPU Selected-Hour Range Summary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task with a fresh implementation subagent per task when practical, then fresh spec-compliance review, then fresh code-quality review. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace per-hour selected-hour min/max discovery that scans full CPU selected-hour arrays with a compact GPU summary readback, while preserving CPU tooltip/analysis values and the main-route `compute-buffer-selected-hour` render path.

**Architecture:** Reuse the existing WebGPU compact range reducer, but add an API that reduces the already GPU-resident visible selected-hour output buffer instead of re-computing the hour into another buffer. The selected-hour live session will ask the GPU for one 16-byte range summary in per-hour/discrete mode, keep the existing tooltip/analysis CPU readback separately labeled, and fall back to the existing CPU scan only when compact output summaries are unavailable.

**Tech Stack:** SvelteKit/Svelte 5, TypeScript, Vitest, Playwright, WebGPU/WGSL.

---

## Hard Constraints

- Do not commit.
- Do not create git worktrees.
- Preserve unrelated dirty files.
- User instructions override any skill doc commit step. Replace commit steps with `git status --short --branch` and `git diff --check` checkpoints.
- Proof surface is `/`, not `/debug`.
- Preserve `rendererBackend=webgpu`.
- Preserve `compute-buffer-selected-hour`.
- Preserve same compute/render device.
- Do not clobber the visible selected-hour GPU buffer while computing summaries.
- Compact range path must not call debug full-value readback just for min/max.
- Tooltip/analysis CPU readbacks are allowed, but must be separate from range-discovery diagnostics.
- Final compact summary readback must be 16 bytes per summarized hour.

## Current File Map

- `viewer/src/lib/compute/gpu/webgpuUtciPipeline.ts`
  - Already owns selected-day compact range reduction via `runUtciRangeSummaryForTimeIndex()`.
  - Already has `rangeSummaryOutputBuffer`, partial buffers, final 16-byte staging buffer, reducer pipelines, `encodeRangeSummaryReduction()`, `parseRangeSummaryRecord()`, and `__TEST_ONLY_reduceRangeValuesForDebug()`.
  - New responsibility: reduce an existing `OnDemandUtciOutput` GPU buffer without writing to it.
- `viewer/src/lib/compute/gpu/gpu-pipeline.ts`
  - Already defines `RunUtciRangeSummaryForTimeIndexParams`, `UtciRangeSummary`, `OnDemandUtciOutput`, and `UTCIComputePipeline`.
  - New responsibility: define `RunUtciRangeSummaryForOutputParams` and optional `runUtciRangeSummaryForOutput`.
- `viewer/src/lib/compute/compute-manager.ts`
  - Already delegates `runUtciRangeSummaryForTimeIndex()`.
  - New responsibility: delegate `runUtciRangeSummaryForOutput()`.
- `viewer/src/lib/compute/on-demand/onDemandDiagnostics.ts`
  - Already has selected-day compact summary timing fields.
  - New responsibility: add selected-hour compact summary timing and byte/count fields.
- `viewer/src/lib/diagnostics/selectedHourRenderPublicationDiagnostics.ts`
  - Already preserves selected-day compact proof fields.
  - New responsibility: preserve selected-hour per-hour range proof fields.
- `viewer/src/lib/compute/selected-hour/liveUtciSelectedHourSession.ts`
  - Current per-hour/discrete mode does:
    `selectedHourUtci = readOnDemandUtciForDebug(...)`, then `getUtciValuesRange(selectedHourUtci)`.
  - New responsibility: use compact output summary for `selectedHourUtciRange` when available, and keep CPU values for tooltip/analysis.
- `viewer/tests/compute/live-selected-hour-session.test.ts`
  - Add red tests for compact per-hour range, tooltip readback labeling, and unavailable compact fallback.
- `viewer/tests/compute/compute-manager.test.ts`
  - Add API delegation and unsupported-path tests.
- `viewer/tests/compute/webgpuUtciPipeline.behavior.test.ts`
  - Add source locks for the output-buffer summary API.
- `viewer/tests/compute/onDemandDiagnostics.test.ts`
  - Add selected-hour compact summary diagnostics coverage.
- `viewer/tests/diagnostics/main-route-utci-diagnostics.test.ts`
  - Add selected-hour timeline preservation assertions.
- `viewer/tests/e2e/compact-range-summary-parity.spec.ts`
  - Extend proof to cover output-buffer summary min/max, not only raw value debug reduction.
- `viewer/tests/e2e/main-route-transition-scrub-diagnostics.spec.ts`
  - Extend collector so it compares full-day/month and per-hour/hour behavior and asserts non-vacuous compact per-hour proof.
- `viewer/tests/e2e/main-route-transition-scrub-diagnostics-source-lock.test.ts`
  - Lock collector proof fields.
- Create `viewer/scripts/assert-main-route-transition-compact-proof.ts`
  - Durable artifact proof parser for compact per-hour and selected-day proof.

## Design Contracts

Add this API in `gpu-pipeline.ts`:

```ts
export interface RunUtciRangeSummaryForOutputParams {
	timeIndex: number;
	numPoints: number;
	format: 'f32-utci';
	output: OnDemandUtciOutput;
	signal?: AbortSignal;
}
```

Add this optional method to `UTCIComputePipeline`:

```ts
runUtciRangeSummaryForOutput?(
	params: RunUtciRangeSummaryForOutputParams
): Promise<UtciRangeSummary>;
```

Output-buffer summary rules:

- It must read from `params.output.gpuOutputHandle?.buffer ?? params.output.gpuBuffer`.
- The returned selected-hour snapshot buffer from `runUtciForTimeIndex()` must be created with `GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST`; otherwise the reducer cannot bind it as read-only storage.
- It must never call `readOnDemandUtciForDebug()`.
- It must never call `runUtciForTimeIndex(params)`.
- It must never use `rangeSummaryOutputBuffer` as the input if a visible output buffer is present.
- It must not write to or destroy the visible output buffer.
- It must use `encodeRangeSummaryReduction()` and `RANGE_SUMMARY_RECORD_BYTES`.
- It must map exactly 16 bytes for the final summary.
- Keep command submission boundaries explicit:
  - selected-day `runUtciRangeSummaryForTimeIndex()` keeps its current one-encoder flow where UTCI calculation into `rangeSummaryOutputBuffer`, range reduction, final copy, submit, queue wait, and 16-byte map happen in order.
  - selected-hour output-buffer summary may use a separate encoder/submit because its source buffer already exists.
  - test-only output-buffer helpers must not call a serializing public summary method from inside `runRangeSummarySerial()`.

New selected-hour timeline fields:

```ts
sessionSelectedHourRangeResolutionPath?:
	| 'compact-gpu-summary'
	| 'cpu-scan-existing-values'
	| 'unavailable'
	| 'not-needed';
sessionSelectedHourRangeReadbackCount?: number;
sessionSelectedHourRangeCpuScanCount?: number;
sessionSelectedHourRangeSummaryReadbackCount?: number;
sessionSelectedHourRangeSummaryReadbackBytes?: number;
sessionSelectedHourRangeFullReadbackAvoidedCount?: number;
sessionSelectedHourRangeSummaryReductionPassCount?: number;
```

New `OnDemandTimings` fields:

```ts
selectedHourRangeSummaryMs?: number;
selectedHourRangeSummaryDispatchMs?: number;
selectedHourRangeSummaryReadbackMs?: number;
selectedHourRangeSummaryReadbackBytes?: number;
selectedHourRangeSummaryReadbackCount?: number;
selectedHourRangeSummaryReductionPassCount?: number;
selectedHourRangeFullReadbackAvoidedCount?: number;
```

Expected compact per-hour proof on an uncached discrete/per-hour selected-hour sample:

```txt
sessionSelectedHourRangeResolutionPath = compact-gpu-summary
sessionSelectedHourRangeReadbackCount = 0
sessionSelectedHourRangeCpuScanCount = 0
sessionSelectedHourRangeSummaryReadbackCount = 1
sessionSelectedHourRangeSummaryReadbackBytes = 16
sessionSelectedHourRangeFullReadbackAvoidedCount = 1
selectedHourReadbackReasons includes tooltip or comparison when CPU values are requested
selectedHourReadbackReasons does not include range for compact min/max
```

## Task 1: Output-Buffer Summary API

**Files:**
- Modify: `viewer/src/lib/compute/gpu/gpu-pipeline.ts`
- Modify: `viewer/src/lib/compute/compute-manager.ts`
- Modify: `viewer/src/lib/compute/gpu/webgpuUtciPipeline.ts`
- Test: `viewer/tests/compute/compute-manager.test.ts`
- Test: `viewer/tests/compute/webgpuUtciPipeline.behavior.test.ts`

- [x] **Step 1: Write the failing compute-manager API test**

In `viewer/tests/compute/compute-manager.test.ts`, add a fake output:

```ts
const output = {
	format: 'f32-utci' as const,
	numPoints: 4,
	timeIndex: 8,
	gpuBuffer: { size: 16 }
};
```

Add this fake pipeline method in the existing fake pipeline setup:

```ts
runUtciRangeSummaryForOutput: vi.fn(async (params) => ({
	timeIndex: params.timeIndex,
	range: { min: 2, max: 9 },
	validCount: params.numPoints,
	readbackBytes: 16,
	reductionPassCount: 1,
	debugLabel: 'webgpu-on-demand-f32-utci-range-summary' as const
}))
```

Add this test:

```ts
it('delegates compact UTCI output range summary requests to the pipeline', async () => {
	const { pipeline } = createFakePipeline();
	const manager = new ComputeManager(pipeline);
	const output = {
		format: 'f32-utci' as const,
		numPoints: 4,
		timeIndex: 8,
		gpuBuffer: { size: 16 }
	};

	const summary = await manager.runUtciRangeSummaryForOutput({
		timeIndex: 8,
		numPoints: 4,
		format: 'f32-utci',
		output
	});

	expect(pipeline.runUtciRangeSummaryForOutput).toHaveBeenCalledWith({
		timeIndex: 8,
		numPoints: 4,
		format: 'f32-utci',
		output
	});
	expect(summary).toMatchObject({
		timeIndex: 8,
		range: { min: 2, max: 9 },
		validCount: 4,
		readbackBytes: 16,
		reductionPassCount: 1,
		debugLabel: 'webgpu-on-demand-f32-utci-range-summary'
	});
});
```

Add unsupported-path coverage:

```ts
it('throws when the pipeline does not support compact output range summaries', async () => {
	const { pipeline } = createFakePipeline();
	pipeline.runUtciRangeSummaryForOutput = undefined;
	const manager = new ComputeManager(pipeline);

	await expect(
		manager.runUtciRangeSummaryForOutput({
			timeIndex: 8,
			numPoints: 4,
			format: 'f32-utci',
			output: {
				format: 'f32-utci',
				numPoints: 4,
				timeIndex: 8,
				gpuBuffer: { size: 16 }
			}
		})
	).rejects.toThrowError('UTCI pipeline does not support compact selected-hour output range summaries');
});
```

Run:

```powershell
cd viewer
npx vitest run tests/compute/compute-manager.test.ts
```

Expected before implementation: TypeScript/test failure because `runUtciRangeSummaryForOutput` does not exist.

- [x] **Step 2: Add the API types**

In `viewer/src/lib/compute/gpu/gpu-pipeline.ts`, add the `RunUtciRangeSummaryForOutputParams` interface from the Design Contracts section and add the optional method to `UTCIComputePipeline`.

Keep `format` narrowed to `'f32-utci'`.

- [x] **Step 3: Add compute-manager delegation**

In `viewer/src/lib/compute/compute-manager.ts`, import `RunUtciRangeSummaryForOutputParams` if needed and add:

```ts
async runUtciRangeSummaryForOutput(
	params: RunUtciRangeSummaryForOutputParams
): Promise<UtciRangeSummary> {
	if (params.format !== 'f32-utci') {
		throw new Error('UTCI range summaries support only f32-utci output format');
	}
	if (!this.pipeline.runUtciRangeSummaryForOutput) {
		throw new Error('UTCI pipeline does not support compact selected-hour output range summaries');
	}
	return this.pipeline.runUtciRangeSummaryForOutput(params);
}
```

- [x] **Step 4: Write WebGPU source-lock tests before implementation**

In `viewer/tests/compute/webgpuUtciPipeline.behavior.test.ts`, add:

```ts
it('output range summary method reduces the visible GPU output buffer without debug readback', () => {
	const source = readFileSync(
		resolve(__dirname, '../../src/lib/compute/gpu/webgpuUtciPipeline.ts'),
		'utf8'
	);
	const methodStart = source.indexOf('async runUtciRangeSummaryForOutput');
	expect(methodStart).toBeGreaterThanOrEqual(0);
	const methodEnd = source.indexOf('\n\n\tasync readOnDemandUtciForDebug', methodStart);
	expect(methodEnd).toBeGreaterThan(methodStart);
	const method = source.slice(methodStart, methodEnd);

	expect(method).toContain('runRangeSummarySerial');
	expect(method).toContain('encodeRangeSummaryReduction');
	expect(method).toContain('RANGE_SUMMARY_RECORD_BYTES');
	expect(method).toContain('copyBufferToBuffer');
	expect(method).toContain('unmap()');
	expect(method).not.toContain('readOnDemandUtciForDebug');
	expect(method).not.toContain('runUtciForTimeIndex(params)');
expect(method).not.toContain('rangeSummaryOutputBuffer =');
});
```

Add a source-lock test for storage-bindable selected-hour snapshots:

```ts
it('creates selected-hour snapshot buffers that can be reduced as storage input', () => {
	const source = readFileSync(
		resolve(__dirname, '../../src/lib/compute/gpu/webgpuUtciPipeline.ts'),
		'utf8'
	);
	const snapshotStart = source.indexOf('const snapshotBuffer = this.device.createBuffer');
	expect(snapshotStart).toBeGreaterThanOrEqual(0);
	const snapshotEnd = source.indexOf('});', snapshotStart);
	const snapshotBlock = source.slice(snapshotStart, snapshotEnd);

	expect(snapshotBlock).toContain('GPUBufferUsage.STORAGE');
	expect(snapshotBlock).toContain('GPUBufferUsage.COPY_SRC');
	expect(snapshotBlock).toContain('GPUBufferUsage.COPY_DST');
});
```

Run:

```powershell
cd viewer
npx vitest run tests/compute/webgpuUtciPipeline.behavior.test.ts
```

Expected before implementation: fail because the method does not exist.

- [x] **Step 5: Implement WebGPU output-buffer summary**

In `viewer/src/lib/compute/gpu/webgpuUtciPipeline.ts`, add a private resolver near the range summary helpers:

```ts
private resolveOutputRangeSummarySourceBuffer(output: OnDemandUtciOutput): GPUBuffer {
	const source = output.gpuOutputHandle?.buffer ?? output.gpuBuffer;
	if (!source || typeof (source as GPUBuffer).size !== 'number') {
		throw new Error('WebGPU UTCI pipeline: compact output range summary requires a GPU output buffer');
	}
	return source as GPUBuffer;
}
```

Add a shared reducer method:

```ts
private async reduceRangeSummaryFromValuesBuffer(params: {
	sourceValuesBuffer: GPUBuffer;
	valueCount: number;
	timeIndex: number;
	signal?: AbortSignal;
	recordDiagnostics?: 'selected-hour' | 'selected-day';
}): Promise<UtciRangeSummary> {
	this.ensureNotAborted(params.signal);
	const [valuesPipeline, rangesPipeline] = await Promise.all([
		this.ensureRangeReduceValuesPipeline(),
		this.ensureRangeReduceRangesPipeline()
	]);
	this.ensureNotAborted(params.signal);

	const rangeSummaryStartedAt = performance.now();
	const rangeSummaryFinalStagingBuffer = this.ensureRangeSummaryFinalStagingBuffer();
	const transientUniformBuffers: GPUBuffer[] = [];
	let mapped = false;
	try {
		const dispatchStartedAt = performance.now();
		const encoder = this.device.createCommandEncoder();
		const { finalBuffer, reductionPassCount } = this.encodeRangeSummaryReduction({
			encoder,
			sourceValuesBuffer: params.sourceValuesBuffer,
			valueCount: params.valueCount,
			valuesPipeline,
			rangesPipeline,
			transientUniformBuffers
		});
		encoder.copyBufferToBuffer(
			finalBuffer,
			0,
			rangeSummaryFinalStagingBuffer,
			0,
			RANGE_SUMMARY_RECORD_BYTES
		);
		this.queue.submit([encoder.finish()]);
		await this.queue.onSubmittedWorkDone();
		this.ensureNotAborted(params.signal);
		const dispatchMs = performance.now() - dispatchStartedAt;
		this.destroyTransientUniformBuffers(transientUniformBuffers);

		const readbackStartedAt = performance.now();
		await rangeSummaryFinalStagingBuffer.mapAsync(GPUMapMode.READ);
		mapped = true;
		this.ensureNotAborted(params.signal);
		const mappedRange = rangeSummaryFinalStagingBuffer.getMappedRange(
			0,
			RANGE_SUMMARY_RECORD_BYTES
		);
		const summaryBytes = mappedRange.slice(0);
		const parsed = parseRangeSummaryRecord(summaryBytes);
		const readbackMs = performance.now() - readbackStartedAt;
		const summaryMs = performance.now() - rangeSummaryStartedAt;

		if (params.recordDiagnostics === 'selected-hour') {
			this.onDemandDiagnostics = {
				...this.onDemandDiagnostics,
				timings: {
					...this.onDemandDiagnostics.timings,
					selectedHourRangeSummaryMs: summaryMs,
					selectedHourRangeSummaryDispatchMs: dispatchMs,
					selectedHourRangeSummaryReadbackMs: readbackMs,
					selectedHourRangeSummaryReadbackBytes:
						(this.onDemandDiagnostics.timings.selectedHourRangeSummaryReadbackBytes ?? 0) +
						RANGE_SUMMARY_RECORD_BYTES,
					selectedHourRangeSummaryReadbackCount:
						(this.onDemandDiagnostics.timings.selectedHourRangeSummaryReadbackCount ?? 0) + 1,
					selectedHourRangeSummaryReductionPassCount:
						(this.onDemandDiagnostics.timings.selectedHourRangeSummaryReductionPassCount ?? 0) +
						reductionPassCount,
					selectedHourRangeFullReadbackAvoidedCount:
						(this.onDemandDiagnostics.timings.selectedHourRangeFullReadbackAvoidedCount ?? 0) + 1
				}
			};
		}

		return {
			timeIndex: params.timeIndex,
			range: parsed.range,
			validCount: parsed.validCount,
			readbackBytes: RANGE_SUMMARY_RECORD_BYTES,
			reductionPassCount,
			debugLabel: 'webgpu-on-demand-f32-utci-range-summary'
		};
	} finally {
		this.destroyTransientUniformBuffers(transientUniformBuffers);
		if (mapped) {
			rangeSummaryFinalStagingBuffer.unmap();
		}
	}
}
```

This helper is for already-produced source buffers only. Do not call it from the middle of `runUtciRangeSummaryForTimeIndex()` before the selected-day UTCI compute encoder has been submitted. Keep `runUtciRangeSummaryForTimeIndex()` in its current single-submit shape, or split shared logic into non-submitting `encodeRangeSummaryReduction()` plus a small readback parser. The implementation must not submit the reducer before the source values exist.

Update the selected-hour snapshot buffer creation in `runUtciForTimeIndex()` from:

```ts
usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
```

to:

```ts
usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
```

Then add:

```ts
async runUtciRangeSummaryForOutput(
	params: RunUtciRangeSummaryForOutputParams
): Promise<UtciRangeSummary> {
	return this.runRangeSummarySerial(async () => {
		this.ensureNotAborted(params.signal);
		if (params.format !== 'f32-utci') {
			throw new Error(
				`WebGPU UTCI pipeline: unsupported compact range output format "${params.format}"`
			);
		}
		if (params.numPoints <= 0) {
			throw new Error(
				`WebGPU UTCI pipeline: invalid compact output range numPoints=${params.numPoints}`
			);
		}
		if (params.output.format !== 'f32-utci') {
			throw new Error('WebGPU UTCI pipeline: compact output range summary requires f32-utci output');
		}
		if (params.output.numPoints !== params.numPoints) {
			throw new Error(
				`WebGPU UTCI pipeline: compact output range numPoints mismatch output=${params.output.numPoints} requested=${params.numPoints}`
			);
		}
		const sourceValuesBuffer = this.resolveOutputRangeSummarySourceBuffer(params.output);
		return this.reduceRangeSummaryFromValuesBuffer({
			sourceValuesBuffer,
			valueCount: params.numPoints,
			timeIndex: params.timeIndex,
			signal: params.signal,
			recordDiagnostics: 'selected-hour'
		});
	});
}
```

Do not refactor `runUtciRangeSummaryForTimeIndex()` into a two-submit flow in this task. If shared code is extracted, keep selected-day's source-value dispatch and reduction in the same encoder before the single submit.

- [x] **Step 6: Add a test-only output-buffer parity helper**

In `webgpuUtciPipeline.ts`, add:

```ts
async __TEST_ONLY_reduceOutputRangeForDebug(values: Float32Array): Promise<UtciRangeSummary> {
	return this.runRangeSummarySerial(async () => {
		const buffer = this.device.createBuffer({
			size: values.byteLength,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
		});
		try {
			this.queue.writeBuffer(buffer, 0, values.buffer as ArrayBuffer, values.byteOffset, values.byteLength);
			return this.reduceRangeSummaryFromValuesBuffer({
				sourceValuesBuffer: buffer,
				valueCount: values.length,
				timeIndex: 0,
				recordDiagnostics: undefined
			});
		} finally {
			buffer.destroy();
		}
	});
}
```

The test helper must call `reduceRangeSummaryFromValuesBuffer()` directly, not `runUtciRangeSummaryForOutput()`, because both public summary APIs serialize access to shared reduction buffers.

- [x] **Step 7: Run Task 1 tests**

Run:

```powershell
cd viewer
npx vitest run tests/compute/compute-manager.test.ts tests/compute/webgpuUtciPipeline.behavior.test.ts
```

Expected after implementation: pass.

- [x] **Step 8: Review checkpoint**

Run a fresh spec-compliance review subagent first. Do not start code-quality review until spec compliance is clean. Then run a fresh code-quality review subagent. Fix findings before Task 2.

Run checkpoint after fixes:

```powershell
git status --short --branch
git diff --check
```

Expected: no commits, no whitespace errors.

## Task 2: Selected-Hour Session Per-Hour Range Resolver

**Files:**
- Modify: `viewer/src/lib/compute/on-demand/onDemandDiagnostics.ts`
- Modify: `viewer/src/lib/diagnostics/selectedHourRenderPublicationDiagnostics.ts`
- Modify: `viewer/src/lib/compute/selected-hour/liveUtciSelectedHourSession.ts`
- Test: `viewer/tests/compute/onDemandDiagnostics.test.ts`
- Test: `viewer/tests/diagnostics/main-route-utci-diagnostics.test.ts`
- Test: `viewer/tests/compute/live-selected-hour-session.test.ts`

- [x] **Step 1: Write failing diagnostics tests first**

In `viewer/tests/compute/onDemandDiagnostics.test.ts`, add:

```ts
it('exposes selected-hour compact range timing fields without changing readback reason accounting', () => {
	const diagnostics = createEmptyOnDemandDiagnostics();
	const next = recordOnDemandTiming(diagnostics, 'selectedHourRangeSummaryReadbackBytes', 16);

	expect(next.timings.selectedHourRangeSummaryReadbackBytes).toBe(16);
	expect(next.selectedHourReadbackReasons).toBeUndefined();
});
```

In `viewer/tests/diagnostics/main-route-utci-diagnostics.test.ts`, extend the existing render timeline preservation test with:

```ts
sessionSelectedHourRangeResolutionPath: 'compact-gpu-summary',
sessionSelectedHourRangeReadbackCount: 0,
sessionSelectedHourRangeCpuScanCount: 0,
sessionSelectedHourRangeSummaryReadbackCount: 1,
sessionSelectedHourRangeSummaryReadbackBytes: 16,
sessionSelectedHourRangeFullReadbackAvoidedCount: 1,
sessionSelectedHourRangeSummaryReductionPassCount: 2
```

Also add these fields as `undefined` in the second partial merge object and assert the original values survive.

Run:

```powershell
cd viewer
npx vitest run tests/compute/onDemandDiagnostics.test.ts tests/diagnostics/main-route-utci-diagnostics.test.ts
```

Expected before implementation: TypeScript/test failure because selected-hour compact timing/timeline fields do not exist.

- [x] **Step 2: Add diagnostics fields**

Add selected-hour timing fields to `OnDemandTimings` and selected-hour timeline fields to `SelectedHourRenderPublicationTimeline` exactly as listed in the Design Contracts section.

No Svelte files are touched in this task.

- [x] **Step 3: Write failing live-session compact per-hour test**

In `viewer/tests/compute/live-selected-hour-session.test.ts`, update the mocked `ComputeManager` class with:

```ts
runUtciRangeSummaryForOutput = vi.fn(async (params) =>
	this.pipeline.runUtciRangeSummaryForOutput(params)
);
```

Update `beforeEach()` fake pipeline with:

```ts
runUtciRangeSummaryForOutput: vi.fn(async (params: { timeIndex: number }) => ({
	timeIndex: params.timeIndex,
	range: { min: 11, max: 29 },
	validCount: 2,
	readbackBytes: 16,
	reductionPassCount: 1,
	debugLabel: 'webgpu-on-demand-f32-utci-range-summary' as const
})),
```

Add a focused red test:

```ts
it('uses compact GPU output summary for per-hour range while keeping tooltip CPU values separate', async () => {
	mockState.pipeline.runUtciRangeSummaryForOutput = vi.fn(async (params: { timeIndex: number }) => ({
		timeIndex: params.timeIndex,
		range: { min: 5, max: 35 },
		validCount: 2,
		readbackBytes: 16,
		reductionPassCount: 2,
		debugLabel: 'webgpu-on-demand-f32-utci-range-summary' as const
	}));
	const session = await prepareSelectedHourLiveSession({
		analysisId: 'analysis-a',
		base: createBaseAnalysis(),
		model: {} as Group,
		epwUrl: '/weather.epw',
		signal: new AbortController().signal,
		preferredDevice: mockState.rendererDevice
	});

	const result = await session.runSelectedHour({
		monthIndex: 0,
		hourIndex: 12,
		timeIndex: 12,
		colorMode: 'discrete',
		preferGpuResident: true,
		rendererDevice: mockState.rendererDevice
	});

	expect(result.renderTransport).toBe('compute-buffer-selected-hour');
	expect(result.analysis?.metadata.utci_range).toEqual({ min: 5, max: 35 });
	expect(result.gpuResidentOutput?.utciRange).toEqual({ min: 5, max: 35 });
	expect(result.gpuResidentOutput?.tooltipUtciValues).toEqual(new Float32Array([11, 29]));
	expect(mockState.pipeline.runUtciRangeSummaryForOutput).toHaveBeenCalledTimes(1);
	expect(mockState.pipeline.runUtciRangeSummaryForOutput).toHaveBeenCalledWith(
		expect.objectContaining({
			timeIndex: 12,
			numPoints: 2,
			format: 'f32-utci',
			output: expect.objectContaining({ gpuBuffer: mockState.gpuBuffer })
		})
	);
	expect(mockState.pipeline.readOnDemandUtciForDebug).toHaveBeenCalledTimes(1);
	expect(result.diagnostics.selectedHourReadbackReasons).toEqual(['tooltip']);
	expect(result.diagnostics.selectedHourReadbackReasonCounts).toEqual({ tooltip: 1 });
	expect(result.diagnostics.timings.renderPublication?.renderPublicationTimeline).toMatchObject({
		sessionSelectedHourRangeResolutionPath: 'compact-gpu-summary',
		sessionSelectedHourRangeReadbackCount: 0,
		sessionSelectedHourRangeCpuScanCount: 0,
		sessionSelectedHourRangeSummaryReadbackCount: 1,
		sessionSelectedHourRangeSummaryReadbackBytes: 16,
		sessionSelectedHourRangeFullReadbackAvoidedCount: 1,
		sessionSelectedHourRangeSummaryReductionPassCount: 2
	});
});
```

Run:

```powershell
cd viewer
npx vitest run tests/compute/live-selected-hour-session.test.ts
```

Expected before implementation: fail because discrete mode still scans `selectedHourUtci`.

Also update existing assertions in this file that currently expect CPU-derived discrete ranges:

- `attaches live selected-hour values for same-device GPU-resident range and tooltip data`
- `records one selected-hour range scan and reuses it for discrete GPU-resident range`

Those tests should expect the compact summary range from the mock, keep `tooltipUtciValues` from `readOnDemandUtciForDebug()`, and assert that readback reasons are `['tooltip']`, not `['range']`.

- [x] **Step 4: Write failing fallback test**

Add:

```ts
it('falls back to scanning existing tooltip values when compact per-hour summary is unavailable', async () => {
	mockState.pipeline.runUtciRangeSummaryForOutput = undefined;
	const session = await prepareSelectedHourLiveSession({
		analysisId: 'analysis-a',
		base: createBaseAnalysis(),
		model: {} as Group,
		epwUrl: '/weather.epw',
		signal: new AbortController().signal,
		preferredDevice: mockState.rendererDevice
	});

	const result = await session.runSelectedHour({
		monthIndex: 0,
		hourIndex: 12,
		timeIndex: 12,
		colorMode: 'discrete',
		preferGpuResident: true,
		rendererDevice: mockState.rendererDevice
	});

	expect(result.analysis?.metadata.utci_range).toEqual({ min: 11, max: 29 });
	expect(result.gpuResidentOutput?.utciRange).toEqual({ min: 11, max: 29 });
	expect(result.diagnostics.selectedHourReadbackReasons).toEqual(['tooltip']);
	expect(result.diagnostics.selectedHourReadbackReasonCounts).toEqual({ tooltip: 1 });
	expect(result.diagnostics.timings.renderPublication?.renderPublicationTimeline).toMatchObject({
		sessionSelectedHourRangeResolutionPath: 'cpu-scan-existing-values',
		sessionSelectedHourRangeReadbackCount: 0,
		sessionSelectedHourRangeCpuScanCount: 1,
		sessionSelectedHourRangeSummaryReadbackCount: 0,
		sessionSelectedHourRangeSummaryReadbackBytes: 0,
		sessionSelectedHourRangeFullReadbackAvoidedCount: 0
	});
});
```

Expected before implementation: fail because fallback path has no explicit diagnostics.

- [x] **Step 5: Implement `resolveSelectedHourUtciRange()`**

In `liveUtciSelectedHourSession.ts`, add near `resolveSelectedDayUtciRange()`:

```ts
async function resolveSelectedHourUtciRange(params: {
	state: PreparedSessionState;
	output: OnDemandUtciOutput;
	timeIndex: number;
	colorMode: 'normalized' | 'discrete';
	selectedHourUtci?: Float32Array;
}): Promise<{
	range: { min: number; max: number } | null;
	resolutionPath: 'compact-gpu-summary' | 'cpu-scan-existing-values' | 'unavailable' | 'not-needed';
	readbackCount: number;
	cpuScanCount: number;
	summaryReadbackCount: number;
	summaryReadbackBytes: number;
	fullReadbackAvoidedCount: number;
	reductionPassCount: number;
	timings: Pick<
		OnDemandTimings,
		| 'selectedHourRangeSummaryMs'
		| 'selectedHourRangeSummaryDispatchMs'
		| 'selectedHourRangeSummaryReadbackMs'
		| 'selectedHourRangeSummaryReadbackBytes'
		| 'selectedHourRangeSummaryReadbackCount'
		| 'selectedHourRangeSummaryReductionPassCount'
		| 'selectedHourRangeFullReadbackAvoidedCount'
	>;
}> {
	if (params.colorMode !== 'discrete') {
		return {
			range: null,
			resolutionPath: 'not-needed',
			readbackCount: 0,
			cpuScanCount: 0,
			summaryReadbackCount: 0,
			summaryReadbackBytes: 0,
			fullReadbackAvoidedCount: 0,
			reductionPassCount: 0,
			timings: {}
		};
	}

	if (params.state.pipeline.runUtciRangeSummaryForOutput) {
		const summary = await params.state.computeManager.runUtciRangeSummaryForOutput({
			timeIndex: params.timeIndex,
			numPoints: params.state.numPoints,
			format: 'f32-utci',
			output: params.output,
			signal: params.state.signal
		});
		return {
			range: summary.range,
			resolutionPath: 'compact-gpu-summary',
			readbackCount: 0,
			cpuScanCount: 0,
			summaryReadbackCount: 1,
			summaryReadbackBytes: summary.readbackBytes,
			fullReadbackAvoidedCount: 1,
			reductionPassCount: summary.reductionPassCount,
			timings: {
				...(params.state.computeManager.getOnDemandDiagnostics?.()?.timings ?? {})
			}
		};
	}

	if (params.selectedHourUtci) {
		return {
			range: getUtciValuesRange(params.selectedHourUtci),
			resolutionPath: 'cpu-scan-existing-values',
			readbackCount: 0,
			cpuScanCount: 1,
			summaryReadbackCount: 0,
			summaryReadbackBytes: 0,
			fullReadbackAvoidedCount: 0,
			reductionPassCount: 0,
			timings: {}
		};
	}

	return {
		range: null,
		resolutionPath: 'unavailable',
		readbackCount: 0,
		cpuScanCount: 0,
		summaryReadbackCount: 0,
		summaryReadbackBytes: 0,
		fullReadbackAvoidedCount: 0,
		reductionPassCount: 0,
		timings: {}
	};
}
```

Use this helper instead of:

```ts
const selectedHourUtciRange =
	params.colorMode === 'discrete' && selectedHourUtci
		? getUtciValuesRange(selectedHourUtci)
		: null;
```

Do not move or remove the existing tooltip/analysis readback. It should still record `tooltip`, `comparison`, or caller-provided reason.

- [x] **Step 6: Stamp selected-hour range timeline fields**

Replace the old scan timeline with compact resolution fields:

```ts
const sessionSelectedHourRangeResolveStartedAtMs = performance.now();
const selectedHourUtciRangeResult = await resolveSelectedHourUtciRange({
	state,
	output,
	timeIndex: params.timeIndex,
	colorMode: params.colorMode,
	selectedHourUtci
});
const selectedHourUtciRange = selectedHourUtciRangeResult.range;
diagnostics.timings = {
	...diagnostics.timings,
	...selectedHourUtciRangeResult.timings
};
stampSessionRenderTimeline(diagnostics, {
	sessionSelectedHourRangeScanStartedAtMs: sessionSelectedHourRangeResolveStartedAtMs,
	sessionSelectedHourRangeScanCompletedAtMs: performance.now(),
	sessionSelectedHourRangeResolutionPath: selectedHourUtciRangeResult.resolutionPath,
	sessionSelectedHourRangeReadbackCount: selectedHourUtciRangeResult.readbackCount,
	sessionSelectedHourRangeCpuScanCount: selectedHourUtciRangeResult.cpuScanCount,
	sessionSelectedHourRangeSummaryReadbackCount: selectedHourUtciRangeResult.summaryReadbackCount,
	sessionSelectedHourRangeSummaryReadbackBytes: selectedHourUtciRangeResult.summaryReadbackBytes,
	sessionSelectedHourRangeFullReadbackAvoidedCount:
		selectedHourUtciRangeResult.fullReadbackAvoidedCount,
	sessionSelectedHourRangeSummaryReductionPassCount:
		selectedHourUtciRangeResult.reductionPassCount
});
```

When merging these timing fields, preserve the existing `renderPublication` object; the subsequent `stampSessionRenderTimeline()` call should add timeline fields after the merge. If copying pipeline diagnostics wholesale would overwrite unrelated timing state, only copy the seven `selectedHourRangeSummary*` fields listed above.

Keep the old `sessionSelectedHourRangeScanStartedAtMs` names populated for downstream timing compatibility, but their meaning becomes "per-hour range resolve window."

- [x] **Step 7: Run Task 2 tests**

Run:

```powershell
cd viewer
npx vitest run tests/compute/onDemandDiagnostics.test.ts tests/diagnostics/main-route-utci-diagnostics.test.ts tests/compute/live-selected-hour-session.test.ts
```

Expected: pass.

- [x] **Step 8: Review checkpoint**

Fresh spec-compliance review first, then code-quality review after spec is clean. Fix findings before Task 3.

Run checkpoint:

```powershell
git status --short --branch
git diff --check
```

Expected: no commits, no whitespace errors.

## Task 3: E2E Parity And Collector Proof

**Files:**
- Modify: `viewer/tests/e2e/compact-range-summary-parity.spec.ts`
- Modify: `viewer/tests/e2e/main-route-transition-scrub-diagnostics.spec.ts`
- Modify: `viewer/tests/e2e/main-route-transition-scrub-diagnostics-source-lock.test.ts`
- Create: `viewer/scripts/assert-main-route-transition-compact-proof.ts`

- [x] **Step 1: Extend compact parity test first**

In `viewer/tests/e2e/compact-range-summary-parity.spec.ts`, add:

```ts
async function runGpuOutputCompactSummary(page: Page, values: number[]): Promise<RangeSummary> {
	await page.goto('/');
	return page.evaluate(async (rawValues) => {
		if (!navigator.gpu) throw new Error('WebGPU is not available in this browser context');
		const adapter = await navigator.gpu.requestAdapter();
		if (!adapter) throw new Error('WebGPU adapter is not available in this browser context');
		const device = await adapter.requestDevice();
		const modulePath = '/src/lib/compute/gpu/webgpuUtciPipeline.ts';
		const { __TEST_ONLY_WebgpuUtciComputePipeline } = (await import(
			/* @vite-ignore */ modulePath
		)) as typeof WebgpuUtciPipelineModule;
		const pipeline = new __TEST_ONLY_WebgpuUtciComputePipeline(device, false);
		try {
			const summary = await pipeline.__TEST_ONLY_reduceOutputRangeForDebug(
				new Float32Array(rawValues)
			);
			return summary;
		} finally {
			pipeline.dispose();
			device.destroy();
		}
	}, values);
}

async function expectGpuOutputMatchesCpu(page: Page, values: number[]) {
	const gpuSummary = await runGpuOutputCompactSummary(page, values);
	const expected = cpuRange(Array.from(new Float32Array(values)));

	expect(gpuSummary.readbackBytes).toBe(16);
	expect(gpuSummary.validCount).toBe(expected.validCount);
	expect(gpuSummary.debugLabel).toBe('webgpu-on-demand-f32-utci-range-summary');
	if (expected.range === null) {
		expect(gpuSummary.range).toBeNull();
		return;
	}
	expect(gpuSummary.range?.min).toBeCloseTo(expected.range.min, 5);
	expect(gpuSummary.range?.max).toBeCloseTo(expected.range.max, 5);
}
```

Add:

```ts
test('match CPU range when reducing an existing selected-hour output buffer', async ({ page }) => {
	const values = Array.from({ length: 768 }, (_, index) => Math.cos(index * 0.19) * 20 + index / 100);
	values[33] = -31.5;
	values[700] = 49.25;

	await expectGpuOutputMatchesCpu(page, values);
});
```

Run:

```powershell
cd viewer
npx playwright test --config=playwright.collect.config.ts tests/e2e/compact-range-summary-parity.spec.ts --project=chromium --workers=1 --reporter=list --timeout=120000
```

Expected before Task 1 helper implementation: fail because `__TEST_ONLY_reduceOutputRangeForDebug` does not exist. Expected after Task 1: pass.

- [x] **Step 2: Extend transition collector to collect per-hour mode**

In `viewer/tests/e2e/main-route-transition-scrub-diagnostics.spec.ts`, add:

```ts
type ColorMode = 'normalized' | 'discrete';
```

Add helper:

```ts
async function setColorScaleMode(page: Page, mode: 'full day' | 'per hour') {
	const button = page.getByRole('button', { name: new RegExp(`^${mode}$`, 'i') });
	await expect(button).toBeVisible();
	await button.click();
	await expect(button).toHaveAttribute('aria-pressed', 'true');
}
```

Update `waitForSelectedHourPublication()` options to accept `colorMode?: ColorMode` and require:

```ts
if (params.colorMode) {
	expect(value.baseColorMode).toBe(params.colorMode);
}
```

Update `buildSample()` input and output to include:

```ts
colorMode: params.diagnostics.baseColorMode ?? null,
```

For each case, after the existing normalized/full-day collection, collect a discrete/per-hour mini-sequence:

```ts
await setColorScaleMode(page, 'per hour');
const perHourDiagnostics = await waitForSelectedHourPublication(
	page,
	selectionKey(caseConfig.analysisId, stableMonthIndex, stableHourIndex),
	{
		minSurfaceRequestId: previousRequestId,
		expectedGridResolutionMeters: caseConfig.gridResolutionMeters,
		colorMode: 'discrete'
	}
);
assertStrongGpuProof(perHourDiagnostics, sourceUrl);
const perHourSample = buildSample({
	caseId: caseConfig.caseId,
	actionKind: 'hour-scrub',
	actionLabel: 'per-hour-mode',
	wallVisibleMs: null,
	diagnostics: perHourDiagnostics,
	entryUrl: entry.entryUrl,
	targetUrl: entry.targetUrl
});
assertPerHourRangeResolutionProof(perHourSample);
samples.push(perHourSample);
previousRequestId = perHourDiagnostics.baseSurfaceRequestId ?? previousRequestId;
```

Then scrub one hour in per-hour mode:

```ts
const perHourScrub = await collectInteractionSample({
	page,
	caseConfig,
	actionKind: 'hour-scrub',
	actionLabel: 'per-hour-hour-1',
	targetMonthIndex: stableMonthIndex,
	targetHourIndex: stableHourIndex + 1,
	previousRequestId,
	interact: () => setHourSelection(page, stableHourIndex + 1),
	entryUrl: entry.entryUrl,
	targetUrl: entry.targetUrl,
	colorMode: 'discrete'
});
assertStrongGpuProof(perHourScrub.diagnostics, sourceUrl);
assertPerHourRangeResolutionProof(perHourScrub.sample);
samples.push(perHourScrub.sample);
```

If `collectInteractionSample()` has no `colorMode` parameter, add it and pass it through to `waitForSelectedHourPublication()`.

- [x] **Step 3: Add per-hour proof assertion**

Add:

```ts
function assertPerHourRangeResolutionProof(sample: ReturnType<typeof buildSample>) {
	if (sample.colorMode !== 'discrete') return;
	const timeline = sample.renderPublication.timeline;
	const proofLabel = `${sample.caseId} ${sample.actionLabel}`;
	expect(timeline, `${proofLabel} should include timeline`).not.toBeNull();
	if (!timeline) return;
	expect(timeline.sessionSelectedHourRangeResolutionPath, `${proofLabel} path`).toBe(
		'compact-gpu-summary'
	);
	expect(timeline.sessionSelectedHourRangeReadbackCount, `${proofLabel} full readbacks`).toBe(0);
	expect(timeline.sessionSelectedHourRangeCpuScanCount, `${proofLabel} CPU scans`).toBe(0);
	expect(timeline.sessionSelectedHourRangeSummaryReadbackCount, `${proofLabel} summaries`).toBe(1);
	expect(timeline.sessionSelectedHourRangeSummaryReadbackBytes, `${proofLabel} bytes`).toBe(16);
	expect(timeline.sessionSelectedHourRangeFullReadbackAvoidedCount, `${proofLabel} avoided`).toBe(1);
}
```

At the end of each case:

```ts
const compactPerHourSamples = samples.filter((sample) => {
	const timeline = sample.renderPublication.timeline;
	return (
		sample.colorMode === 'discrete' &&
		timeline?.sessionSelectedHourRangeResolutionPath === 'compact-gpu-summary' &&
		timeline.sessionSelectedHourRangeSummaryReadbackBytes === 16
	);
});
expect(compactPerHourSamples.length).toBeGreaterThanOrEqual(2);
```

This is the non-vacuous proof for per-hour/hour behavior.

- [x] **Step 4: Pick timeline fields**

In `pickTimelineFields()`, add:

```ts
sessionSelectedHourRangeResolutionPath: stringOrNull(
	timeline.sessionSelectedHourRangeResolutionPath
),
sessionSelectedHourRangeReadbackCount: numberOrNull(
	timeline.sessionSelectedHourRangeReadbackCount
),
sessionSelectedHourRangeCpuScanCount: numberOrNull(
	timeline.sessionSelectedHourRangeCpuScanCount
),
sessionSelectedHourRangeSummaryReadbackCount: numberOrNull(
	timeline.sessionSelectedHourRangeSummaryReadbackCount
),
sessionSelectedHourRangeSummaryReadbackBytes: numberOrNull(
	timeline.sessionSelectedHourRangeSummaryReadbackBytes
),
sessionSelectedHourRangeFullReadbackAvoidedCount: numberOrNull(
	timeline.sessionSelectedHourRangeFullReadbackAvoidedCount
),
sessionSelectedHourRangeSummaryReductionPassCount: numberOrNull(
	timeline.sessionSelectedHourRangeSummaryReductionPassCount
),
```

- [x] **Step 5: Update source lock**

In `viewer/tests/e2e/main-route-transition-scrub-diagnostics-source-lock.test.ts`, add:

```ts
expect(source).toContain('sessionSelectedHourRangeResolutionPath');
expect(source).toContain('sessionSelectedHourRangeSummaryReadbackBytes');
expect(source).toContain('assertPerHourRangeResolutionProof');
expect(source).toContain('per-hour-hour-1');
expect(source).toContain('compactPerHourSamples.length');
```

Run:

```powershell
cd viewer
npx vitest run tests/e2e/main-route-transition-scrub-diagnostics-source-lock.test.ts
```

Expected: pass.

- [x] **Step 6: Create artifact proof parser**

Create `viewer/scripts/assert-main-route-transition-compact-proof.ts`:

```ts
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const repoRoot = resolve(process.cwd(), process.cwd().endsWith('viewer') ? '..' : '.');
const artifactPath = resolve(
	repoRoot,
	'data/performance-results/main-route-transition-scrub-diagnostics.json'
);

type Timeline = Record<string, unknown>;
type Sample = {
	actionKind?: string;
	actionLabel?: string;
	colorMode?: string | null;
	proof?: {
		rendererBackend?: string;
		utciSurfaceSource?: string;
		baseRenderTransport?: string;
		baseSameDeviceForComputeAndRender?: boolean;
		dataTextureBuildCount?: number | null;
		selectedHourRuntimeContract?: {
			visibleSelectedHourReadbackCount?: number | null;
			strongVisibleGpuPath?: boolean | null;
		};
	};
	forbiddenComparisonFieldsPresent?: string[];
	renderPublication?: {
		timeline?: Timeline | null;
	};
};
type Artifact = { cases?: Array<{ caseId?: string; samples?: Sample[] }> };

function numberField(timeline: Timeline, key: string): number | null {
	const value = timeline[key];
	return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function stringField(timeline: Timeline, key: string): string | null {
	const value = timeline[key];
	return typeof value === 'string' ? value : null;
}

const artifact = JSON.parse(readFileSync(artifactPath, 'utf8')) as Artifact;
const expectedUncachedMonthProofCount = Math.max(0, (artifact.cases?.length ?? 0) * 2);
const expectedPerHourProofCount = Math.max(0, (artifact.cases?.length ?? 0) * 2);
let badProof = 0;
let forbidden = 0;
let missingSelectedDayCompact = 0;
let missingPerHourCompact = 0;
let uncachedMonthProofCount = 0;
let perHourProofCount = 0;
let samples = 0;

for (const c of artifact.cases ?? []) {
	for (const s of c.samples ?? []) {
		samples += 1;
		const proof = s.proof ?? {};
		const contract = proof.selectedHourRuntimeContract ?? {};
		if (
			proof.rendererBackend !== 'webgpu' ||
			proof.utciSurfaceSource !== 'compute-buffer-selected-hour' ||
			proof.baseRenderTransport !== 'compute-buffer-selected-hour' ||
			proof.baseSameDeviceForComputeAndRender !== true ||
			proof.dataTextureBuildCount !== 0 ||
			contract.visibleSelectedHourReadbackCount !== 0 ||
			contract.strongVisibleGpuPath !== true
		) {
			badProof += 1;
		}
		forbidden += s.forbiddenComparisonFieldsPresent?.length ?? 0;
		const timeline = s.renderPublication?.timeline ?? null;
		if (!timeline) continue;
		if (
			s.actionKind === 'month-change' &&
			timeline.sessionSelectedDayRangeCacheHit === false
		) {
			uncachedMonthProofCount += 1;
			if (
				stringField(timeline, 'sessionSelectedDayRangeResolutionPath') !==
					'compact-gpu-summary' ||
				numberField(timeline, 'sessionSelectedDayRangeReadbackCount') !== 0 ||
				numberField(timeline, 'sessionSelectedDayRangeSummaryReadbackCount') !== 23 ||
				numberField(timeline, 'sessionSelectedDayRangeSummaryReadbackBytes') !== 23 * 16 ||
				numberField(timeline, 'sessionSelectedDayRangeFullReadbackAvoidedCount') !== 23
			) {
				missingSelectedDayCompact += 1;
			}
		}
		if (s.colorMode === 'discrete') {
			perHourProofCount += 1;
			if (
				stringField(timeline, 'sessionSelectedHourRangeResolutionPath') !==
					'compact-gpu-summary' ||
				numberField(timeline, 'sessionSelectedHourRangeReadbackCount') !== 0 ||
				numberField(timeline, 'sessionSelectedHourRangeCpuScanCount') !== 0 ||
				numberField(timeline, 'sessionSelectedHourRangeSummaryReadbackCount') !== 1 ||
				numberField(timeline, 'sessionSelectedHourRangeSummaryReadbackBytes') !== 16 ||
				numberField(timeline, 'sessionSelectedHourRangeFullReadbackAvoidedCount') !== 1
			) {
				missingPerHourCompact += 1;
			}
		}
	}
}

const result = {
	cases: artifact.cases?.length ?? 0,
	samples,
	badProof,
	forbidden,
	missingSelectedDayCompact,
	missingPerHourCompact,
	uncachedMonthProofCount,
	perHourProofCount,
	expectedUncachedMonthProofCount,
	expectedPerHourProofCount
};
console.log(JSON.stringify(result, null, 2));

if (
	result.cases < 4 ||
	badProof !== 0 ||
	forbidden !== 0 ||
	missingSelectedDayCompact !== 0 ||
	missingPerHourCompact !== 0 ||
	uncachedMonthProofCount < expectedUncachedMonthProofCount ||
	perHourProofCount < expectedPerHourProofCount
) {
	process.exitCode = 1;
}
```

- [x] **Step 7: Run Task 3 focused tests**

Run:

```powershell
cd viewer
npx vitest run tests/e2e/main-route-transition-scrub-diagnostics-source-lock.test.ts
npx playwright test --config=playwright.collect.config.ts tests/e2e/compact-range-summary-parity.spec.ts --project=chromium --workers=1 --reporter=list --timeout=120000
```

Expected: both pass.

- [x] **Step 8: Review checkpoint**

Fresh spec-compliance review first, then code-quality review after spec is clean. Fix findings before Task 4.

Run checkpoint:

```powershell
git status --short --branch
git diff --check
```

Expected: no commits, no whitespace errors.

## Task 4: Evidence Refresh And Final Verification

**Files:**
- Existing artifact refreshed: `data/performance-results/main-route-transition-scrub-diagnostics.json`
- Existing progress artifact refreshed: `data/performance-results/main-route-transition-scrub-diagnostics-progress.json`
- Modify only if evidence needs a note: `docs/webgpu_strategy_analysis.md`

- [x] **Step 1: Run Svelte check**

```powershell
cd viewer
npm run check
```

Expected: `svelte-check found 0 errors and 0 warnings`.

- [x] **Step 2: Run focused Vitest suite**

```powershell
cd viewer
npx vitest run tests/compute/compute-manager.test.ts tests/compute/webgpuUtciPipeline.behavior.test.ts tests/compute/onDemandDiagnostics.test.ts tests/diagnostics/main-route-utci-diagnostics.test.ts tests/compute/live-selected-hour-session.test.ts tests/e2e/main-route-transition-scrub-diagnostics-source-lock.test.ts
```

Expected: all tests pass.

- [x] **Step 3: Run selected-hour quality suite**

```powershell
cd viewer
npm run test:quality:selected-hour
```

Expected: all tests pass.

- [x] **Step 4: Run compact range parity Playwright proof**

```powershell
cd viewer
npx playwright test --config=playwright.collect.config.ts tests/e2e/compact-range-summary-parity.spec.ts --project=chromium --workers=1 --reporter=list --timeout=120000
```

Expected: pass and include the output-buffer compact summary case.

- [x] **Step 5: Run headed transition scrub collector**

```powershell
cd viewer
npx playwright test --config=playwright.collect.config.ts tests/e2e/main-route-transition-scrub-diagnostics.spec.ts --project=chromium --workers=1 --reporter=list --timeout=900000
```

Expected: one collector test passes and refreshes:

```txt
data/performance-results/main-route-transition-scrub-diagnostics.json
data/performance-results/main-route-transition-scrub-diagnostics-progress.json
```

This collector compares full-day/month and per-hour/hour behavior on `/`.

- [x] **Step 6: Run artifact proof parser**

```powershell
cd viewer
npx tsx scripts/assert-main-route-transition-compact-proof.ts
```

Expected JSON:

```json
{
  "badProof": 0,
  "forbidden": 0,
  "missingSelectedDayCompact": 0,
  "missingPerHourCompact": 0,
  "uncachedMonthProofCount": 8,
  "perHourProofCount": 8,
  "expectedUncachedMonthProofCount": 8,
  "expectedPerHourProofCount": 8
}
```

The expected counts are derived from the artifact case count. Counts may be greater than expected; they must not be `0`.

- [x] **Step 7: Run diff whitespace check**

```powershell
cd ..
git diff --check
```

Expected: exit code `0`.

- [x] **Step 8: Final git status**

```powershell
git status --short --branch
```

Expected: branch still `webGPU`; no commit created; no worktree created.

- [ ] **Step 9: Final response**

Report:

- files changed
- spec review outcomes
- code-quality review outcomes
- verification commands with pass/fail evidence
- performance/evidence outcome from the artifact parser and collector
- residual risks
- current `git status --short --branch`
- explicit note that no commit was made

## Self-Review Checklist

- [ ] Plan uses existing selected-day compact reducer instead of inventing a second reduction shader.
- [ ] Per-hour range summary reads the already GPU-resident visible output buffer.
- [ ] Per-hour summary does not write to, destroy, or replace the visible output buffer.
- [ ] Compact per-hour path does not call `readOnDemandUtciForDebug()` for min/max.
- [ ] Tooltip/analysis CPU readback stays available and is labeled `tooltip` or `comparison`, not `range`.
- [ ] Final compact summary readback is 16 bytes.
- [ ] Fallback compact-unavailable path is covered and visibly labeled.
- [ ] Diagnostics distinguish compact summary count/bytes/path, tooltip/analysis CPU readbacks, and forbidden full selected-hour range scan/readback.
- [ ] Collector asserts at least one uncached compact per-hour proof sample per case.
- [ ] Proof remains on `/`, not `/debug`.
- [ ] No commits and no worktrees.
