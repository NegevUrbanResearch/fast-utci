# Main Route GPU Selected-Day Range Reduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task with a fresh implementation subagent per task, then fresh spec-compliance review, then fresh code-quality review. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace full selected-hour value readbacks used for normalized selected-day/month range calculation with true compact GPU min/max summaries, preserving main-route GPU-native rendering and reducing first uncached month changes on Ness Tziona 0.5m.

**Architecture:** Keep visible selected-hour rendering on `compute-buffer-selected-hour`. Add an isolated range-summary path that computes non-visible hours into a separate summary source buffer, performs GPU reduction down to one final `RangeSummary` per hour, and maps only that final tiny summary. The selected-day range cache aggregates final per-hour summaries; the range path must not call `readOnDemandUtciForDebug()` and must not overwrite the visible selected-hour GPU output handle.

**Tech Stack:** SvelteKit/Svelte 5, TypeScript, Vitest, Playwright, WebGPU/WGSL.

**Hard Constraints For This Plan:**
- Do not commit.
- Do not create git worktrees.
- Preserve unrelated dirty files.
- Current investigation diagnostics are intentionally uncommitted and should remain part of the eventual fix commit.
- Keep default route behavior semantically unchanged: same UTCI calculation, same normalized coloring result within floating-point epsilon, same GPU-native proof.
- The proof surface is `/`, not `/debug`.

---

## Current Evidence And Target

The uncommitted investigation already proves:

- First explicit uncached visits to month `8` and month `1` spend about `1.8-2.4s` in selected-day range resolve.
- Those samples report `sessionSelectedDayRangeCacheHit=false`, `sessionSelectedDayRangeReadbackCount=23`, and `sessionSelectedDayRangeComputedHourCount=23`.
- Return visits report cache hits, `readbackCount=0`, `computedHourCount=0`, range resolve around `0ms`, and visible time around `74-118ms`.

The bad path is `resolveSelectedDayUtciRange()` in `viewer/src/lib/compute/selected-hour/liveUtciSelectedHourSession.ts`: it computes 23 other hours, reads back `Float32Array(numPoints)` for each hour, and scans min/max on CPU.

The proper fix is not prewarming. It is a compact GPU range summary:

```ts
const summary = await params.state.computeManager.runUtciRangeSummaryForTimeIndex({
	timeIndex,
	numPoints: params.state.numPoints,
	numHours: params.state.numHours,
	numMonths: params.state.numMonths,
	format: 'f32-utci',
	signal: params.state.signal
});
dayRange = accumulateUtciRangeSummary(dayRange, summary.range);
```

Target proof for uncached months:

```txt
sessionSelectedDayRangeResolutionPath = compact-gpu-summary
sessionSelectedDayRangeReadbackCount = 0
sessionSelectedDayRangeSummaryReadbackCount = 23
sessionSelectedDayRangeSummaryReadbackBytes = 23 * 16
sessionSelectedDayRangeFullReadbackAvoidedCount = 23
```

## Planned File Map

- Modify `viewer/src/lib/compute/gpu/gpu-pipeline.ts`: add summary API/types.
- Modify `viewer/src/lib/compute/compute-manager.ts`: add delegator.
- Modify `viewer/src/lib/compute/on-demand/onDemandDiagnostics.ts`: add compact range timing/count/byte fields.
- Modify `viewer/src/lib/compute/gpu/webgpuUtciPipeline.ts`: add isolated summary UTCI output buffer, explicit value-reduction and range-reduction GPU pipelines, final summary staging buffer, cleanup, and diagnostics.
- Create `viewer/src/lib/compute/gpu/shaders/utci_range_reduce.wgsl`: reusable reduction shader for source values and partial summaries.
- Modify `viewer/src/lib/compute/selected-hour/liveUtciSelectedHourSession.ts`: use compact summaries for normalized selected-day range cache.
- Modify `viewer/src/lib/diagnostics/selectedHourRenderPublicationDiagnostics.ts`: preserve new timeline fields.
- Modify tests and collectors listed below.

---

## Task 1: Contract And Diagnostics Types

**Files:**
- Modify: `viewer/src/lib/compute/gpu/gpu-pipeline.ts`
- Modify: `viewer/src/lib/compute/compute-manager.ts`
- Modify: `viewer/src/lib/compute/on-demand/onDemandDiagnostics.ts`
- Test: `viewer/tests/compute/compute-manager.test.ts`

- [ ] **Step 1: Write failing compute-manager test first**

In `viewer/tests/compute/compute-manager.test.ts`, add this method to the fake pipeline:

```ts
runUtciRangeSummaryForTimeIndex: vi.fn(async (params) => ({
	timeIndex: params.timeIndex,
	range: { min: 1, max: 4 },
	validCount: params.numPoints,
	readbackBytes: 16,
	reductionPassCount: 2,
	debugLabel: 'webgpu-on-demand-f32-utci-range-summary' as const
}))
```

Add this test:

```ts
it('delegates compact UTCI range summary requests to the pipeline', async () => {
	const { pipeline } = createFakePipeline();
	const manager = new ComputeManager(pipeline);

	const summary = await manager.runUtciRangeSummaryForTimeIndex({
		timeIndex: 8,
		numPoints: 4,
		numHours: 24,
		numMonths: 12,
		format: 'f32-utci'
	});

	expect(pipeline.runUtciRangeSummaryForTimeIndex).toHaveBeenCalledWith({
		timeIndex: 8,
		numPoints: 4,
		numHours: 24,
		numMonths: 12,
		format: 'f32-utci'
	});
	expect(summary).toMatchObject({
		timeIndex: 8,
		range: { min: 1, max: 4 },
		validCount: 4,
		readbackBytes: 16,
		reductionPassCount: 2,
		debugLabel: 'webgpu-on-demand-f32-utci-range-summary'
	});
});
```

Run:

```powershell
cd viewer
npx vitest run tests/compute/compute-manager.test.ts
```

Expected before implementation: TypeScript/test failure because the API does not exist.

- [ ] **Step 2: Add summary API types**

In `viewer/src/lib/compute/gpu/gpu-pipeline.ts`, add:

```ts
export interface RunUtciRangeSummaryForTimeIndexParams extends RunUtciForTimeIndexParams {
	signal?: AbortSignal;
}

export interface UtciRangeSummary {
	timeIndex: number;
	range: { min: number; max: number } | null;
	validCount: number;
	readbackBytes: number;
	reductionPassCount: number;
	debugLabel: 'webgpu-on-demand-f32-utci-range-summary';
}
```

Extend `UTCIComputePipeline`:

```ts
runUtciRangeSummaryForTimeIndex?(
	params: RunUtciRangeSummaryForTimeIndexParams
): Promise<UtciRangeSummary>;
```

- [ ] **Step 3: Add on-demand timing fields**

In `viewer/src/lib/compute/on-demand/onDemandDiagnostics.ts`, add:

```ts
selectedDayRangeSummaryMs?: number;
selectedDayRangeSummaryDispatchMs?: number;
selectedDayRangeSummaryReadbackMs?: number;
selectedDayRangeSummaryReadbackBytes?: number;
selectedDayRangeSummaryReadbackCount?: number;
selectedDayRangeSummaryComputedHourCount?: number;
selectedDayRangeSummaryReductionPassCount?: number;
selectedDayRangeFullReadbackAvoidedCount?: number;
```

- [ ] **Step 4: Add compute-manager delegator**

In `viewer/src/lib/compute/compute-manager.ts`, import `RunUtciRangeSummaryForTimeIndexParams` and `UtciRangeSummary`, then add:

```ts
async runUtciRangeSummaryForTimeIndex(
	params: RunUtciRangeSummaryForTimeIndexParams
): Promise<UtciRangeSummary> {
	if (!this.pipeline.runUtciRangeSummaryForTimeIndex) {
		throw new Error('UTCI pipeline does not support compact selected-hour range summaries');
	}
	return this.pipeline.runUtciRangeSummaryForTimeIndex(params);
}
```

- [ ] **Step 5: Run test green**

Run:

```powershell
cd viewer
npx vitest run tests/compute/compute-manager.test.ts
```

Expected: pass.

- [ ] **Step 6: Review checkpoint**

Run a fresh spec-compliance review subagent first. Do not start code-quality review until spec compliance is clean. Then run a fresh code-quality review subagent. Fix findings before Task 2.

---

## Task 2: Pure Summary Helpers And Source Locks

**Files:**
- Modify: `viewer/src/lib/compute/gpu/webgpuUtciPipeline.ts`
- Test: `viewer/tests/compute/webgpuUtciPipeline.behavior.test.ts`

- [ ] **Step 1: Write failing pure helper tests first**

Export helper functions with `__TEST_ONLY_` names from `webgpuUtciPipeline.ts` in this task:

```ts
type RangeSummaryRecord = {
	min: number;
	max: number;
	validCount: number;
};
```

Add tests in `webgpuUtciPipeline.behavior.test.ts`:

```ts
it('parses mixed f32/u32 range summary records correctly', () => {
	const bytes = new ArrayBuffer(16);
	const view = new DataView(bytes);
	view.setFloat32(0, -3.5, true);
	view.setFloat32(4, 42.25, true);
	view.setUint32(8, 8171761, true);
	view.setUint32(12, 0, true);

	expect(__TEST_ONLY_parseRangeSummaryRecord(bytes)).toEqual({
		range: { min: -3.5, max: 42.25 },
		validCount: 8171761
	});
});

it('keeps equal-valued valid range summaries instead of dropping them', () => {
	const bytes = new ArrayBuffer(16);
	const view = new DataView(bytes);
	view.setFloat32(0, 12.25, true);
	view.setFloat32(4, 12.25, true);
	view.setUint32(8, 4, true);
	view.setUint32(12, 0, true);

	expect(__TEST_ONLY_parseRangeSummaryRecord(bytes)).toEqual({
		range: { min: 12.25, max: 12.25 },
		validCount: 4
	});
});

it('returns null range when compact summary valid count is zero', () => {
	const bytes = new ArrayBuffer(16);
	const view = new DataView(bytes);
	view.setFloat32(0, 3.4028234663852886e38, true);
	view.setFloat32(4, -3.4028234663852886e38, true);
	view.setUint32(8, 0, true);
	view.setUint32(12, 0, true);

	expect(__TEST_ONLY_parseRangeSummaryRecord(bytes)).toEqual({
		range: null,
		validCount: 0
	});
});
```

Run:

```powershell
cd viewer
npx vitest run tests/compute/webgpuUtciPipeline.behavior.test.ts
```

Expected before implementation: fail because helper exports do not exist.

- [ ] **Step 2: Implement mixed-type parsing helper**

In `webgpuUtciPipeline.ts`, add:

```ts
const RANGE_SUMMARY_RECORD_BYTES = 16;

export function __TEST_ONLY_parseRangeSummaryRecord(buffer: ArrayBuffer): {
	range: { min: number; max: number } | null;
	validCount: number;
} {
	const view = new DataView(buffer);
	const min = view.getFloat32(0, true);
	const max = view.getFloat32(4, true);
	const validCount = view.getUint32(8, true);
	if (validCount === 0 || !Number.isFinite(min) || !Number.isFinite(max)) {
		return { range: null, validCount };
	}
	return { range: { min, max }, validCount };
}
```

Important: do not require `max > min`. A valid all-equal field must preserve `{ min, max }`.

- [ ] **Step 3: Add source-lock for no full debug readback in summary method**

In `webgpuUtciPipeline.behavior.test.ts`, import `readFileSync` and `resolve`, then add:

```ts
it('range summary method does not call full selected-hour debug readback', () => {
	const source = readFileSync(
		resolve(__dirname, '../../src/lib/compute/gpu/webgpuUtciPipeline.ts'),
		'utf8'
	);
	const methodStart = source.indexOf('async runUtciRangeSummaryForTimeIndex');
	expect(methodStart).toBeGreaterThanOrEqual(0);
	const methodEnd = source.indexOf('\n\n\tasync readOnDemandUtciForDebug', methodStart);
	const method = source.slice(methodStart, methodEnd);

	expect(method).toContain('ensureRangeReduceValuesPipeline');
	expect(method).toContain('ensureRangeReduceRangesPipeline');
	expect(method).not.toContain('readOnDemandUtciForDebug');
	expect(method).toContain('rangeSummaryOutputBuffer');
	expect(method).toContain('selectedDayRangeSummaryReadbackBytes');
});
```

Expected before runtime implementation: fail because method does not exist.

- [ ] **Step 4: Run helper/source tests**

Run:

```powershell
cd viewer
npx vitest run tests/compute/webgpuUtciPipeline.behavior.test.ts
```

Expected after helper implementation but before runtime method: parsing tests pass, source-lock fails. Leave source-lock red for Task 3.

- [ ] **Step 5: Review checkpoint**

Fresh spec-compliance review first, then code-quality review after spec is clean. Fix findings before Task 3.

---

## Task 3: True GPU Final Summary Reduction

**Files:**
- Create: `viewer/src/lib/compute/gpu/shaders/utci_range_reduce.wgsl`
- Modify: `viewer/src/lib/compute/gpu/webgpuUtciPipeline.ts`
- Test: `viewer/tests/compute/webgpuUtciPipeline.behavior.test.ts`

- [ ] **Step 1: Create WGSL shader with no invalid built-ins**

Create `viewer/src/lib/compute/gpu/shaders/utci_range_reduce.wgsl`:

```wgsl
const F32_MAX_VALUE = 3.4028234663852886e38;

struct Params {
	input_count: u32,
	input_offset: u32,
	output_offset: u32,
	input_stride: u32,
}

struct RangeSummary {
	min_value: f32,
	max_value: f32,
	valid_count: u32,
	_pad: u32,
}

@group(0) @binding(0)
var<storage, read> source_values: array<f32>;

@group(0) @binding(1)
var<storage, read> source_ranges: array<RangeSummary>;

@group(0) @binding(2)
var<storage, read_write> output_ranges: array<RangeSummary>;

@group(0) @binding(3)
var<uniform> params: Params;

var<workgroup> local_min: array<f32, 256>;
var<workgroup> local_max: array<f32, 256>;
var<workgroup> local_count: array<u32, 256>;

fn is_valid_value(value: f32) -> bool {
	return value == value && abs(value) <= F32_MAX_VALUE;
}

@compute @workgroup_size(256)
fn reduce_values(
	@builtin(local_invocation_id) local_id: vec3<u32>,
	@builtin(global_invocation_id) global_id: vec3<u32>,
	@builtin(workgroup_id) workgroup_id: vec3<u32>
) {
	let local_index = local_id.x;
	let source_index = params.input_offset + global_id.x;
	if (global_id.x < params.input_count) {
		let value = source_values[source_index];
		if (is_valid_value(value)) {
			local_min[local_index] = value;
			local_max[local_index] = value;
			local_count[local_index] = 1u;
		} else {
			local_min[local_index] = F32_MAX_VALUE;
			local_max[local_index] = -F32_MAX_VALUE;
			local_count[local_index] = 0u;
		}
	} else {
		local_min[local_index] = F32_MAX_VALUE;
		local_max[local_index] = -F32_MAX_VALUE;
		local_count[local_index] = 0u;
	}
	reduce_workgroup(local_index);
	if (local_index == 0u) {
		let out_index = params.output_offset + workgroup_id.x;
		output_ranges[out_index].min_value = local_min[0];
		output_ranges[out_index].max_value = local_max[0];
		output_ranges[out_index].valid_count = local_count[0];
		output_ranges[out_index]._pad = 0u;
	}
}

@compute @workgroup_size(256)
fn reduce_ranges(
	@builtin(local_invocation_id) local_id: vec3<u32>,
	@builtin(global_invocation_id) global_id: vec3<u32>,
	@builtin(workgroup_id) workgroup_id: vec3<u32>
) {
	let local_index = local_id.x;
	let source_index = params.input_offset + global_id.x;
	if (global_id.x < params.input_count) {
		let summary = source_ranges[source_index];
		if (summary.valid_count > 0u && is_valid_value(summary.min_value) && is_valid_value(summary.max_value)) {
			local_min[local_index] = summary.min_value;
			local_max[local_index] = summary.max_value;
			local_count[local_index] = summary.valid_count;
		} else {
			local_min[local_index] = F32_MAX_VALUE;
			local_max[local_index] = -F32_MAX_VALUE;
			local_count[local_index] = 0u;
		}
	} else {
		local_min[local_index] = F32_MAX_VALUE;
		local_max[local_index] = -F32_MAX_VALUE;
		local_count[local_index] = 0u;
	}
	reduce_workgroup(local_index);
	if (local_index == 0u) {
		let out_index = params.output_offset + workgroup_id.x;
		output_ranges[out_index].min_value = local_min[0];
		output_ranges[out_index].max_value = local_max[0];
		output_ranges[out_index].valid_count = local_count[0];
		output_ranges[out_index]._pad = 0u;
	}
}

fn reduce_workgroup(local_index: u32) {
	workgroupBarrier();
	var stride = 128u;
	loop {
		if (stride == 0u) { break; }
		if (local_index < stride) {
			local_min[local_index] = min(local_min[local_index], local_min[local_index + stride]);
			local_max[local_index] = max(local_max[local_index], local_max[local_index + stride]);
			local_count[local_index] = local_count[local_index] + local_count[local_index + stride];
		}
		workgroupBarrier();
		stride = stride / 2u;
	}
}
```

- [ ] **Step 2: Add pipeline and buffers**

In `webgpuUtciPipeline.ts`, import the shader and add fields:

```ts
private rangeReduceValuesPipeline: GPUComputePipeline | null = null;
private rangeReduceValuesPipelinePromise: Promise<GPUComputePipeline> | null = null;
private rangeReduceRangesPipeline: GPUComputePipeline | null = null;
private rangeReduceRangesPipelinePromise: Promise<GPUComputePipeline> | null = null;
private rangeSummaryOutputBuffer: GPUBuffer | null = null;
private rangeSummaryPartialBufferA: GPUBuffer | null = null;
private rangeSummaryPartialBufferB: GPUBuffer | null = null;
private rangeSummaryFinalStagingBuffer: GPUBuffer | null = null;
```

Add cleanup in `dispose()` or the existing equivalent cleanup path:

```ts
this.rangeSummaryOutputBuffer?.destroy();
this.rangeSummaryOutputBuffer = null;
this.rangeSummaryPartialBufferA?.destroy();
this.rangeSummaryPartialBufferA = null;
this.rangeSummaryPartialBufferB?.destroy();
this.rangeSummaryPartialBufferB = null;
this.rangeSummaryFinalStagingBuffer?.destroy();
this.rangeSummaryFinalStagingBuffer = null;
```

- [ ] **Step 3: Add explicit range pipeline ensure methods**

Create two compute pipelines. WebGPU pipelines are bound to one entry point, so `reduce_values` and `reduce_ranges` must not share one `GPUComputePipeline`.

```ts
private async ensureRangeReduceValuesPipeline(): Promise<GPUComputePipeline> {
	if (this.rangeReduceValuesPipeline) return this.rangeReduceValuesPipeline;
	if (!this.rangeReduceValuesPipelinePromise) {
		const module = this.device.createShaderModule({ code: utciRangeReduceShaderRaw });
		this.rangeReduceValuesPipelinePromise = this.device
			.createComputePipelineAsync({
				layout: 'auto',
				compute: { module, entryPoint: 'reduce_values' }
			})
			.then((pipeline) => {
				this.rangeReduceValuesPipeline = pipeline;
				return pipeline;
			});
	}
	return this.rangeReduceValuesPipelinePromise;
}

private async ensureRangeReduceRangesPipeline(): Promise<GPUComputePipeline> {
	if (this.rangeReduceRangesPipeline) return this.rangeReduceRangesPipeline;
	if (!this.rangeReduceRangesPipelinePromise) {
		const module = this.device.createShaderModule({ code: utciRangeReduceShaderRaw });
		this.rangeReduceRangesPipelinePromise = this.device
			.createComputePipelineAsync({
				layout: 'auto',
				compute: { module, entryPoint: 'reduce_ranges' }
			})
			.then((pipeline) => {
				this.rangeReduceRangesPipeline = pipeline;
				return pipeline;
			});
	}
	return this.rangeReduceRangesPipelinePromise;
}
```

- [ ] **Step 4: Implement isolated summary UTCI compute**

Add `rangeSummaryOutputBuffer`, separate from `onDemandOutputBuffer`. In `runUtciRangeSummaryForTimeIndex()`, do not call `runUtciForTimeIndex()` because that would overwrite the visible selected-hour output. Instead, duplicate the on-demand compute dispatch binding shape but bind output storage to `rangeSummaryOutputBuffer`.

Required source-lock facts:

```txt
runUtciRangeSummaryForTimeIndex
rangeSummaryOutputBuffer
ensureRangeReduceValuesPipeline
ensureRangeReduceRangesPipeline
not readOnDemandUtciForDebug
not runUtciForTimeIndex(params)
```

- [ ] **Step 5: Implement multi-pass GPU reduction to one final summary**

Algorithm and chunk invariants:

1. Compute non-visible hour UTCI values into `rangeSummaryOutputBuffer`.
2. Build first-pass chunks with `createPointDispatchChunks(numPoints, 256)`.
3. Allocate partial record capacity:

```ts
const firstPassChunks = createPointDispatchChunks(numPoints, 256);
const firstPassPartialCount = firstPassChunks.reduce(
	(total, chunk) => total + Math.ceil(chunk.pointCount / 256),
	0
);
const partialBytes = Math.max(1, firstPassPartialCount) * RANGE_SUMMARY_RECORD_BYTES;
```

4. First pass: for each chunk, dispatch `reduce_values` from `rangeSummaryOutputBuffer` into `rangeSummaryPartialBufferA`.
   - `input_count = chunk.pointCount`
   - `input_offset = chunk.pointOffset`
   - `output_offset = runningPartialOffset`
   - after each chunk, `runningPartialOffset += Math.ceil(chunk.pointCount / 256)`
   - assert `runningPartialOffset === firstPassPartialCount`
5. While partial count is greater than `1`:
   - run `reduce_ranges` from current partial buffer into the other partial buffer
   - update partial count to `ceil(partialCount / 256)`
   - swap A/B
6. Copy exactly `16` bytes from the final partial buffer to `rangeSummaryFinalStagingBuffer`.
7. Map `16` bytes only.
8. Copy the mapped bytes into a new `ArrayBuffer`, unmap in `finally`, then parse using `__TEST_ONLY_parseRangeSummaryRecord`.

Guard dispatch dimensions:

```ts
const workgroupCount = Math.ceil(inputCount / 256);
if (workgroupCount > this.device.limits.maxComputeWorkgroupsPerDimension) {
	// Split value input into chunks and write partial output offsets.
}
```

For the first implementation, use `createPointDispatchChunks(numPoints, 256)` for the first pass so X workgroups never exceed WebGPU limits. Then reduce the concatenated partial range count.

Use this staging lifecycle:

```ts
let mapped = false;
try {
	await this.rangeSummaryFinalStagingBuffer.mapAsync(GPUMapMode.READ);
	mapped = true;
	const mappedRange = this.rangeSummaryFinalStagingBuffer.getMappedRange(0, RANGE_SUMMARY_RECORD_BYTES);
	const summaryBytes = mappedRange.slice(0);
	return __TEST_ONLY_parseRangeSummaryRecord(summaryBytes);
} finally {
	if (mapped) this.rangeSummaryFinalStagingBuffer.unmap();
}
```

- [ ] **Step 6: Record diagnostics**

For each successful compact summary:

```ts
selectedDayRangeSummaryReadbackBytes += 16
selectedDayRangeSummaryReadbackCount += 1
selectedDayRangeSummaryReductionPassCount += reductionPassCount
selectedDayRangeFullReadbackAvoidedCount += 1
```

Also record:

```ts
selectedDayRangeSummaryMs
selectedDayRangeSummaryDispatchMs
selectedDayRangeSummaryReadbackMs
```

- [ ] **Step 7: Add abort and cleanup behavior**

Use a helper:

```ts
function ensureNotAborted(signal: AbortSignal | undefined): void {
	if (signal?.aborted) {
		throw new DOMException('Aborted', 'AbortError');
	}
}
```

Call it:
- before creating work
- after `queue.onSubmittedWorkDone()`
- before and after `mapAsync()`

Destroy transient uniform buffers in `finally`.

- [ ] **Step 8: Add behavior/source tests**

Extend `webgpuUtciPipeline.behavior.test.ts` source-lock:

```ts
expect(method).toContain('rangeSummaryOutputBuffer');
expect(method).not.toContain('runUtciForTimeIndex(params)');
expect(method).not.toContain('readOnDemandUtciForDebug');
expect(method).toContain('copyBufferToBuffer');
expect(method).toContain('RANGE_SUMMARY_RECORD_BYTES');
expect(method).toContain('ensureRangeReduceValuesPipeline');
expect(method).toContain('ensureRangeReduceRangesPipeline');
expect(method).toContain('unmap()');
```

Add cleanup source-lock:

```ts
it('destroys range summary buffers on disposal', () => {
	const source = readFileSync(
		resolve(__dirname, '../../src/lib/compute/gpu/webgpuUtciPipeline.ts'),
		'utf8'
	);
	expect(source).toContain('rangeSummaryOutputBuffer?.destroy()');
	expect(source).toContain('rangeSummaryPartialBufferA?.destroy()');
	expect(source).toContain('rangeSummaryPartialBufferB?.destroy()');
expect(source).toContain('rangeSummaryFinalStagingBuffer?.destroy()');
});
```

Add pure CPU reference helpers for expected min/max in the test file:

```ts
function cpuRange(values: readonly number[]) {
	let min = Number.POSITIVE_INFINITY;
	let max = Number.NEGATIVE_INFINITY;
	let validCount = 0;
	for (const value of values) {
		if (!Number.isFinite(value)) continue;
		min = Math.min(min, value);
		max = Math.max(max, value);
		validCount += 1;
	}
	return {
		range: validCount > 0 ? { min, max } : null,
		validCount
	};
}
```

Add a required real WebGPU parity test file or test block. If the repo's Playwright/WebGPU harness is more reliable than Vitest for real GPU execution, create the test under `viewer/tests/e2e/compact-range-summary-parity.spec.ts`; otherwise put it in `webgpuUtciPipeline.behavior.test.ts` behind the existing WebGPU test guards. The test must:

- write known UTCI-like f32 values into the summary source path or run the summary method with a tiny fixture
- cover a normal mixed range spanning more than one workgroup, for example `600` values
- cover equal values, for example `Array(300).fill(12.25)`
- cover invalid values if they can be injected into the source buffer: `NaN`, `Infinity`, `-Infinity`
- compare final compact summary to `cpuRange(values)` within `1e-5`
- assert final readback bytes are exactly `16`
- assert `validCount` matches the CPU reference

Required assertion shape:

```ts
expect(summary.readbackBytes).toBe(16);
expect(summary.validCount).toBe(expected.validCount);
expect(summary.range?.min).toBeCloseTo(expected.range!.min, 5);
expect(summary.range?.max).toBeCloseTo(expected.range!.max, 5);
```

If direct injection into `rangeSummaryOutputBuffer` needs a small test-only method, expose it as `__TEST_ONLY_reduceRangeValuesForDebug(values: Float32Array)` and keep it unavailable from production interfaces. The implementation must use the same `reduce_values` / `reduce_ranges` pipeline path as production.

If the parity test is implemented as Playwright, the file must be exactly:

```txt
viewer/tests/e2e/compact-range-summary-parity.spec.ts
```

- [ ] **Step 9: Run Task 2/3 tests**

Run:

```powershell
cd viewer
npx vitest run tests/compute/webgpuUtciPipeline.behavior.test.ts
```

Expected: pass.

If the parity test was implemented as Playwright, also run:

```powershell
cd viewer
npx playwright test --config=playwright.collect.config.ts tests/e2e/compact-range-summary-parity.spec.ts --project=chromium --workers=1 --reporter=list --timeout=120000
```

Expected: pass and prove compact summary min/max equals CPU reference for mixed, equal, and invalid-value fixtures.

- [ ] **Step 10: Review checkpoint**

Fresh spec-compliance review first, then code-quality review after spec is clean. Fix findings before Task 4.

---

## Task 4: Use Compact Summaries In Selected-Day Range Resolution

**Files:**
- Modify: `viewer/src/lib/compute/selected-hour/liveUtciSelectedHourSession.ts`
- Modify: `viewer/src/lib/diagnostics/selectedHourRenderPublicationDiagnostics.ts`
- Test: `viewer/tests/compute/live-selected-hour-session.test.ts`
- Test: `viewer/tests/diagnostics/main-route-utci-diagnostics.test.ts`

- [ ] **Step 1: Write failing live-session test first**

In `live-selected-hour-session.test.ts`, extend the fake pipeline with `runUtciRangeSummaryForTimeIndex`.

Add a test for uncached normalized range:

```ts
expect(coldTimeline).toMatchObject({
	sessionSelectedDayRangeCacheKey: '1:24',
	sessionSelectedDayRangeCacheHit: false,
	sessionSelectedDayRangeReadbackCount: 0,
	sessionSelectedDayRangeComputedHourCount: 23,
	sessionSelectedDayRangeResolutionPath: 'compact-gpu-summary',
	sessionSelectedDayRangeSummaryReadbackCount: 23,
	sessionSelectedDayRangeSummaryReadbackBytes: 23 * 16,
	sessionSelectedDayRangeFullReadbackAvoidedCount: 23
});
expect(fakePipeline.readOnDemandUtciForDebug).toHaveBeenCalledTimes(1);
```

The single `readOnDemandUtciForDebug()` is the existing tooltip/debug selected-hour readback, not a visible selected-hour fallback and not a range readback. The compact range path must not add 23 range readbacks.

Add a warm return assertion:

```ts
expect(warmTimeline).toMatchObject({
	sessionSelectedDayRangeCacheHit: true,
	sessionSelectedDayRangeReadbackCount: 0,
	sessionSelectedDayRangeComputedHourCount: 0,
	sessionSelectedDayRangeResolutionPath: 'cache-hit',
	sessionSelectedDayRangeSummaryReadbackCount: 0,
	sessionSelectedDayRangeSummaryReadbackBytes: 0
});
```

Run:

```powershell
cd viewer
npx vitest run tests/compute/live-selected-hour-session.test.ts
```

Expected before implementation: fail.

- [ ] **Step 2: Add timeline fields**

In `SelectedHourRenderPublicationTimeline`, add and preserve in copy:

```ts
sessionSelectedDayRangeResolutionPath?: 'full-readback' | 'compact-gpu-summary' | 'cache-hit' | 'unavailable';
sessionSelectedDayRangeSummaryReadbackCount?: number;
sessionSelectedDayRangeSummaryReadbackBytes?: number;
sessionSelectedDayRangeFullReadbackAvoidedCount?: number;
```

- [ ] **Step 3: Add summary accumulator helper**

In `liveUtciSelectedHourSession.ts`, add:

```ts
function accumulateUtciRangeSummary(
	current: { min: number; max: number } | null,
	range: { min: number; max: number } | null
): { min: number; max: number } | null {
	if (!range) return current;
	if (!current) return range;
	return {
		min: Math.min(current.min, range.min),
		max: Math.max(current.max, range.max)
	};
}
```

Do not overload `accumulateUtciRange()` for both arrays and summaries.

- [ ] **Step 4: Prefer compact summaries**

In `resolveSelectedDayUtciRange()`, add result fields:

```ts
resolutionPath: 'full-readback' | 'compact-gpu-summary' | 'cache-hit' | 'unavailable';
summaryReadbackCount: number;
summaryReadbackBytes: number;
fullReadbackAvoidedCount: number;
```

In the non-selected-hour loop:

```ts
if (params.state.pipeline.runUtciRangeSummaryForTimeIndex) {
	const summary = await params.state.computeManager.runUtciRangeSummaryForTimeIndex({
		timeIndex,
		numPoints: params.state.numPoints,
		numHours: params.state.numHours,
		numMonths: params.state.numMonths,
		format: 'f32-utci',
		signal: params.state.signal
	});
	computedHourCount += 1;
	summaryReadbackCount += 1;
	summaryReadbackBytes += summary.readbackBytes;
	fullReadbackAvoidedCount += 1;
	dayRange = accumulateUtciRangeSummary(dayRange, summary.range);
	continue;
}
```

Only fallback to `readOnDemandUtciForDebug()` if compact summary API is unavailable.

- [ ] **Step 5: Stamp timeline**

Add:

```ts
sessionSelectedDayRangeResolutionPath: selectedDayUtciRangeResult?.resolutionPath,
sessionSelectedDayRangeSummaryReadbackCount:
	selectedDayUtciRangeResult?.summaryReadbackCount,
sessionSelectedDayRangeSummaryReadbackBytes:
	selectedDayUtciRangeResult?.summaryReadbackBytes,
sessionSelectedDayRangeFullReadbackAvoidedCount:
	selectedDayUtciRangeResult?.fullReadbackAvoidedCount
```

- [ ] **Step 6: Update diagnostics preservation test**

In `main-route-utci-diagnostics.test.ts`, add the new fields to existing render-publication preservation/copy assertions.

- [ ] **Step 7: Run focused tests**

Run:

```powershell
cd viewer
npx vitest run tests/compute/live-selected-hour-session.test.ts tests/diagnostics/main-route-utci-diagnostics.test.ts
```

Expected: pass.

- [ ] **Step 8: Review checkpoint**

Fresh spec-compliance review first, then code-quality review after spec is clean. Fix findings before Task 5.

---

## Task 5: Collector Proof And Source Locks

**Files:**
- Modify: `viewer/tests/e2e/main-route-transition-scrub-diagnostics.spec.ts`
- Modify: `viewer/tests/e2e/main-route-transition-scrub-diagnostics-source-lock.test.ts`

- [ ] **Step 1: Capture new timeline fields**

In `pickTimelineFields()`, add:

```ts
sessionSelectedDayRangeResolutionPath: stringOrNull(
	timeline.sessionSelectedDayRangeResolutionPath
),
sessionSelectedDayRangeSummaryReadbackCount: numberOrNull(
	timeline.sessionSelectedDayRangeSummaryReadbackCount
),
sessionSelectedDayRangeSummaryReadbackBytes: numberOrNull(
	timeline.sessionSelectedDayRangeSummaryReadbackBytes
),
sessionSelectedDayRangeFullReadbackAvoidedCount: numberOrNull(
	timeline.sessionSelectedDayRangeFullReadbackAvoidedCount
)
```

- [ ] **Step 2: Assert compact proof for uncached months**

Add:

```ts
function assertRangeResolutionProof(sample: ReturnType<typeof buildSample>) {
	if (sample.actionKind !== 'month-change') return;
	const timeline = sample.renderPublication.timeline;
	expect(timeline).not.toBeNull();
	if (sample.actionLabel.endsWith('-return')) {
		expect(timeline!.sessionSelectedDayRangeCacheHit).toBe(true);
		expect(timeline!.sessionSelectedDayRangeReadbackCount).toBe(0);
		return;
	}
	if (timeline!.sessionSelectedDayRangeCacheHit === false) {
		expect(timeline!.sessionSelectedDayRangeResolutionPath).toBe('compact-gpu-summary');
		expect(timeline!.sessionSelectedDayRangeReadbackCount).toBe(0);
		expect(timeline!.sessionSelectedDayRangeSummaryReadbackCount).toBe(23);
		expect(timeline!.sessionSelectedDayRangeSummaryReadbackBytes).toBe(23 * 16);
		expect(timeline!.sessionSelectedDayRangeFullReadbackAvoidedCount).toBe(23);
	}
}
```

Call it before pushing each month-change sample.

At the end of each case, assert the case actually exercised uncached compact proof:

```ts
const uncachedCompactMonthSamples = samples.filter((sample) => {
	const timeline = sample.renderPublication.timeline;
	return (
		sample.actionKind === 'month-change' &&
		timeline?.sessionSelectedDayRangeCacheHit === false &&
		timeline.sessionSelectedDayRangeResolutionPath === 'compact-gpu-summary'
	);
});
expect(uncachedCompactMonthSamples.length).toBeGreaterThanOrEqual(2);
```

This prevents the collector from passing vacuously if all sampled months are already warm.

- [ ] **Step 3: Update source lock**

Add:

```ts
expect(source).toContain('sessionSelectedDayRangeResolutionPath');
expect(source).toContain('sessionSelectedDayRangeSummaryReadbackCount');
expect(source).toContain('sessionSelectedDayRangeSummaryReadbackBytes');
expect(source).toContain('sessionSelectedDayRangeFullReadbackAvoidedCount');
expect(source).toContain('compact-gpu-summary');
```

- [ ] **Step 4: Run source-lock test**

Run:

```powershell
cd viewer
npx vitest run tests/e2e/main-route-transition-scrub-diagnostics-source-lock.test.ts
```

Expected: pass.

- [ ] **Step 5: Review checkpoint**

Fresh spec-compliance review first, then code-quality review after spec is clean. Fix findings before Task 6.

---

## Task 6: Evidence Refresh And Docs

**Files:**
- Modify: `data/performance-results/main-route-transition-scrub-diagnostics.json`
- Modify: `data/performance-results/main-route-transition-scrub-diagnostics-progress.json`
- Modify: `docs/webgpu_strategy_analysis.md`

- [ ] **Step 1: Run headed transition collector**

Run:

```powershell
cd viewer
npx playwright test --config=playwright.collect.config.ts tests/e2e/main-route-transition-scrub-diagnostics.spec.ts --project=chromium --workers=1 --reporter=list --timeout=900000
```

Expected: one headed Chromium collector test passes and refreshes both transition scrub artifacts.

- [ ] **Step 2: Parse artifact proof**

Run:

```powershell
@'
const fs = require('fs');
const data = JSON.parse(fs.readFileSync('data/performance-results/main-route-transition-scrub-diagnostics.json','utf8'));
let badProof = 0, forbidden = 0, missingCompact = 0, uncachedMonthProofCount = 0;
for (const c of data.cases || []) {
  for (const s of c.samples || []) {
    const p = s.proof || {};
    if (p.rendererBackend !== 'webgpu' || p.utciSurfaceSource !== 'compute-buffer-selected-hour' || p.baseSameDeviceForComputeAndRender !== true || p.selectedHourRuntimeContract?.visibleSelectedHourReadbackCount !== 0 || p.selectedHourRuntimeContract?.strongVisibleGpuPath !== true) badProof++;
    forbidden += (s.forbiddenComparisonFieldsPresent || []).length;
    if (s.actionKind === 'month-change') {
      const t = s.renderPublication?.timeline || {};
      if (t.sessionSelectedDayRangeCacheHit === false) {
        uncachedMonthProofCount++;
        if (
          t.sessionSelectedDayRangeResolutionPath !== 'compact-gpu-summary' ||
          t.sessionSelectedDayRangeReadbackCount !== 0 ||
          t.sessionSelectedDayRangeSummaryReadbackCount !== 23 ||
          t.sessionSelectedDayRangeSummaryReadbackBytes !== 23 * 16 ||
          t.sessionSelectedDayRangeFullReadbackAvoidedCount !== 23
        ) missingCompact++;
      }
    }
  }
}
console.log(JSON.stringify({
  cases: data.cases?.length,
  samples: data.cases?.reduce((n,c)=>n+(c.samples?.length||0),0),
  badProof,
  forbidden,
  missingCompact,
  uncachedMonthProofCount
}, null, 2));
'@ | node -
```

Expected: `badProof=0`, `forbidden=0`, `missingCompact=0`, and `uncachedMonthProofCount>=8` across the four cases. If `uncachedMonthProofCount` is `0`, the collector did not prove the cold path and the run must not be accepted.

- [ ] **Step 3: Update strategy doc**

Add `#### GPU Compact Range Summary Follow-Up` under `First-Time Month Range Investigation` in `docs/webgpu_strategy_analysis.md`.

Include:
- command used to collect evidence
- before/after first uncached month timing summary
- proof that visible path remains `/`, WebGPU, `compute-buffer-selected-hour`, same device, no visible selected-hour fallback
- proof that uncached months use `compact-gpu-summary`, `fullReadbacks=0`, `compact=23`, `summaryBytes=368`
- proof count for uncached month samples, not only absence of failures
- residual risks: floating-point reduction ordering epsilon and remaining non-range publication time

- [ ] **Step 4: Review checkpoint**

Fresh spec-compliance review first, then code-quality review after spec is clean. Fix findings before Task 7.

---

## Task 7: Final Verification

**Files:**
- All changed files from Tasks 1-6.

- [ ] **Step 1: Run Svelte check**

```powershell
cd viewer
npm run check
```

Expected: `svelte-check found 0 errors and 0 warnings`.

- [ ] **Step 2: Run focused Vitest suite**

```powershell
cd viewer
npx vitest run tests/compute/compute-manager.test.ts tests/compute/webgpuUtciPipeline.behavior.test.ts tests/compute/live-selected-hour-session.test.ts tests/diagnostics/main-route-utci-diagnostics.test.ts tests/e2e/main-route-transition-scrub-diagnostics-source-lock.test.ts
```

Expected: all tests pass.

- [ ] **Step 3: Run compact range parity proof if implemented as Playwright**

If `viewer/tests/e2e/compact-range-summary-parity.spec.ts` exists, run:

```powershell
cd viewer
npx playwright test --config=playwright.collect.config.ts tests/e2e/compact-range-summary-parity.spec.ts --project=chromium --workers=1 --reporter=list --timeout=120000
```

Expected: pass.

- [ ] **Step 4: Run selected-hour quality suite**

```powershell
cd viewer
npm run test:quality:selected-hour
```

Expected: all tests pass.

- [ ] **Step 5: Run headed transition collector**

```powershell
cd viewer
npx playwright test --config=playwright.collect.config.ts tests/e2e/main-route-transition-scrub-diagnostics.spec.ts --project=chromium --workers=1 --reporter=list --timeout=900000
```

Expected: one test passes and artifacts are up to date.

- [ ] **Step 6: Run artifact proof parser from Task 6**

Expected: `badProof=0`, `forbidden=0`, `missingCompact=0`, and `uncachedMonthProofCount>=8`.

- [ ] **Step 7: Run diff whitespace check**

```powershell
git diff --check
```

Expected: exit code `0`. LF-to-CRLF warnings are acceptable if no whitespace errors are reported.

- [ ] **Step 8: Final git status**

```powershell
git status --short --branch
```

Expected: branch still `webGPU`, no commits created by this plan execution, no worktrees created, and dirty files limited to investigation/fix source, tests, docs, and refreshed artifacts.

- [ ] **Step 9: Final response**

Report files changed, review subagent outcomes, verification commands, evidence outcome, residual risks, current git status, and explicit note that no commit was made.

---

## Self-Review Checklist

- [ ] True final GPU reduction to 16 bytes per summarized hour, not per-workgroup partial readback.
- [ ] Explicit `reduce_values` and `reduce_ranges` compute pipelines; no single pipeline is reused for both entry points.
- [ ] No use of invalid WGSL `isFinite`.
- [ ] Mixed `{ f32, f32, u32, u32 }` summary parsed with `DataView`, not plain `Float32Array`.
- [ ] Equal-valued valid ranges are preserved.
- [ ] Compact range path does not clobber visible selected-hour output.
- [ ] Compact range path does not call `readOnDemandUtciForDebug()` or `runUtciForTimeIndex(params)`.
- [ ] Final staging buffer is copied out and unmapped in success/error/abort paths.
- [ ] Evidence parser asserts nonzero uncached compact month proofs, not just zero failures.
- [ ] Accuracy parity compares compact summary min/max to CPU reference on edge and multi-workgroup cases.
- [ ] Tests are red-green oriented before implementation in each task.
- [ ] No commits and no worktrees.
