# WebGPU Memory & Performance Optimization Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce GPU memory usage by ~97% for solar exposure buffers and eliminate the 288× serial readback loop to unlock GPU performance scaling.

**Architecture:** Two independent workstreams: (1) bit-pack the solar exposure buffer from `f32` to `u32` bitmask (lossless, 560 MB → 17.5 MB for NZ), touching the solar shader, the MRT shader, and the pipeline buffer management; (2) replace the serial readback loop with a hybrid on-demand + background prefetch pattern, touching `liveUtciAnalysis.ts` and `webgpuUtciPipeline.ts`.

**Tech Stack:** WGSL compute shaders, TypeScript, WebGPU API, SvelteKit

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `viewer/src/lib/compute/shaders/exposure_solar.wgsl` | Modify | Write solar exposure as bit-packed u32 instead of f32 |
| `viewer/src/lib/compute/shaders/mrt_utci.wgsl` | Modify | Read solar exposure from bit-packed u32 instead of f32 |
| `viewer/src/lib/compute/webgpuUtciPipeline.ts` | Modify | Update buffer sizing, zero-fill, and readback for packed solar buffer |
| `viewer/src/lib/compute/liveUtciAnalysis.ts` | Modify | Replace 288× serial readback with on-demand + prefetch |
| `viewer/src/lib/compute/gpu-pipeline.ts` | Possibly modify | Update `readSolarExposureFull` signature if needed for debug/parity |

---

### Task 1: Bit-pack solar exposure in the WGSL shader

**Files:**
- Modify: `viewer/src/lib/compute/shaders/exposure_solar.wgsl`

The solar exposure buffer currently stores `f32` values (0.0 or 1.0). Since these are binary, we can pack 32 results into a single `u32` word using `atomicOr`. This is lossless — the information content is exactly 1 bit per result.

The flat index layout is `point_idx * num_time_steps + time_idx`. For any given time step, adjacent points map to adjacent flat indices, so multiple threads may need to write different bits to the same `u32` word. `atomicOr` handles this correctly — each thread sets a unique bit, and atomic OR is commutative and safe for concurrent writes.

- [ ] **Step 1: Change the solar_exposure binding from `array<f32>` to `array<atomic<u32>>`**

Replace the buffer declaration and all writes in `exposure_solar.wgsl`:

```wgsl
// Solar exposure compute shader
// Layout matches the architecture described in the WebGPU migration plan:
// - X dimension: grid points
// - Y dimension: time steps (month × hour)
//
// All positions and directions are expressed in the Three.js world frame:
// X = East/right, Y = Up, Z = North/forward (Y-up). CPU packing code rotates
// sun vectors from the Python Z-up convention into this frame before upload.

struct Vec3F32 {
	x: f32,
	y: f32,
	z: f32,
};

@group(0) @binding(0)
var<storage, read> grid_points: array<Vec3F32>;

@group(0) @binding(1)
var<storage, read> sun_vectors: array<Vec3F32>;

// Bit-packed solar exposure: 1 bit per (point, time_step) result.
// Word index = flat_index / 32, bit index = flat_index % 32.
// Bit = 1 means exposed (sun visible), 0 means occluded.
@group(0) @binding(2)
var<storage, read_write> solar_exposure: array<atomic<u32>>;

struct Params {
	num_points: u32,
	num_time_steps: u32,
}

@group(0) @binding(3)
var<uniform> params: Params;

// When this shader is concatenated with bvh_raycast.wgsl, @group(1) and bvh_intersects_any are provided there.
// Set to true to force-write a known bit at (0,0) to verify the compute buffer is the one we read back (debug zeros).
const PROBE_FORCE_WRITE: bool = false;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
	let point_idx = global_id.x;
	let time_idx = global_id.y;

	if (point_idx >= params.num_points || time_idx >= params.num_time_steps) {
		return;
	}

	let flat_index = point_idx * params.num_time_steps + time_idx;
	let word_idx = flat_index / 32u;
	let bit_idx = flat_index % 32u;

	if (PROBE_FORCE_WRITE && point_idx == 0u && time_idx == 0u) {
		atomicOr(&solar_exposure[word_idx], 1u << bit_idx);
		return;
	}

	let origin = vec3<f32>(
		grid_points[point_idx].x,
		grid_points[point_idx].y,
		grid_points[point_idx].z
	);

	let sun = vec3<f32>(
		sun_vectors[time_idx].x,
		sun_vectors[time_idx].y,
		sun_vectors[time_idx].z
	);

	// Skip BVH traversal for nighttime/invalid vectors.
	let sun_len2 = dot(sun, sun);
	if (sun_len2 < 1e-10 || sun.y <= 0.0) {
		// Occluded (bit stays 0 from zero-fill). No atomicOr needed.
		return;
	}

	// Match Python semantics: launch rays from the sample point itself.
	let ray_origin = origin;
	let hit = bvh_intersects_any(ray_origin, sun);

	// Exposed (not hit) → set bit to 1; occluded (hit) → leave bit as 0.
	if (!hit) {
		atomicOr(&solar_exposure[word_idx], 1u << bit_idx);
	}
}
```

- [ ] **Step 2: Verify the shader compiles**

Run `npm run dev` in the viewer directory. Open the browser console. If the shader has syntax errors, they'll appear immediately when the pipeline is created. Fix any compilation errors before proceeding.

---

### Task 2: Update the MRT shader to read bit-packed solar exposure

**Files:**
- Modify: `viewer/src/lib/compute/shaders/mrt_utci.wgsl:23-24,117-119`

The MRT shader reads `solar_exposure[flat_index]` as an `f32`. It needs to read from the bit-packed `u32` buffer instead. The binding type changes from `array<f32>` to `array<u32>` (non-atomic read-only access is fine here since the solar pass is complete before MRT runs).

- [ ] **Step 1: Change the solar_exposure binding declaration**

In `mrt_utci.wgsl`, change line 23-24 from:

```wgsl
@group(0) @binding(0)
var<storage, read> solar_exposure: array<f32>;
```

to:

```wgsl
// Bit-packed solar exposure (read-only in MRT pass).
// Word index = flat_index / 32, bit index = flat_index % 32.
// Bit = 1 means exposed, 0 means occluded.
@group(0) @binding(0)
var<storage, read> solar_exposure: array<u32>;
```

- [ ] **Step 2: Update the solar exposure read in `compute_outdoor_mrt`**

In the `compute_outdoor_mrt` function (around line 117-119), change the solar exposure read from:

```wgsl
	// Solar exposure fraction for this point/time (0–1), from solar_exposure buffer.
	let flat_index: u32 = point_idx * num_time_steps + time_idx;
	let solar_exp: f32 = clamp(solar_exposure[flat_index], 0.0, 1.0);
```

to:

```wgsl
	// Solar exposure: unpack single bit from bit-packed u32 buffer.
	let flat_index: u32 = point_idx * num_time_steps + time_idx;
	let solar_word: u32 = solar_exposure[flat_index / 32u];
	let solar_bit: u32 = (solar_word >> (flat_index % 32u)) & 1u;
	let solar_exp: f32 = f32(solar_bit);
```

- [ ] **Step 3: Verify shader compilation**

Run `npm run dev`, open the app, and trigger a compute run (load a model). Check the browser console for shader compilation errors. The MRT pipeline creates its shader module at init time, so errors will appear early.

---

### Task 3: Update pipeline buffer sizing and zero-fill for packed solar buffer

**Files:**
- Modify: `viewer/src/lib/compute/webgpuUtciPipeline.ts:222-244` (buffer creation + zero-fill)

The solar exposure buffer currently allocates `numPoints * totalTimeSteps * 4` bytes (f32 per result). With bit-packing, it needs `ceil(numPoints * totalTimeSteps / 32) * 4` bytes (one u32 per 32 results). The zero-fill also needs to match the new smaller size.

- [ ] **Step 1: Update solar buffer sizing in `uploadStaticData`**

In `webgpuUtciPipeline.ts`, find the solar buffer allocation block (around line 222-229) and replace:

```typescript
		const solarBytes = numPoints * totalTimeSteps * 4;
		if (!this.solarExposureBuffer || this.solarExposureBuffer.size !== solarBytes) {
			this.solarExposureBuffer?.destroy();
			this.solarExposureBuffer = this.device.createBuffer({
				size: solarBytes,
				usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
			});
		}
```

with:

```typescript
		// Bit-packed: 1 bit per (point, time_step), packed into u32 words.
		const totalSolarBits = numPoints * totalTimeSteps;
		const solarWords = Math.ceil(totalSolarBits / 32);
		const solarBytes = solarWords * 4;
		if (!this.solarExposureBuffer || this.solarExposureBuffer.size !== solarBytes) {
			this.solarExposureBuffer?.destroy();
			this.solarExposureBuffer = this.device.createBuffer({
				size: solarBytes,
				usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
			});
		}
```

- [ ] **Step 2: Update zero-fill to use the new smaller size**

The zero-fill loop already uses `solarBytes`, so no variable rename needed. But the `zeroChunk` is currently `Float32Array(4096)` — since we're writing raw bytes, this works fine (zeros are zeros regardless of typed array view). No code change needed here — just verify the loop still references `solarBytes` correctly:

```typescript
		// Keep CPU allocations small by reusing tiny zero chunk writes.
		const zeroChunk = new Float32Array(4096);
		for (let offset = 0; offset < solarBytes; offset += zeroChunk.byteLength) {
			this.queue.writeBuffer(this.solarExposureBuffer, offset, zeroChunk.buffer, 0, Math.min(zeroChunk.byteLength, solarBytes - offset));
		}
```

This should still work as-is since it uses `solarBytes` which is now much smaller.

- [ ] **Step 3: Verify the app loads and computes without errors**

Run `npm run dev`, load the BG model, and confirm:
- No buffer creation errors in the console
- The heatmap renders (colors may look different if there's a bug — that's what we check in the next task)
- Memory usage in Chrome DevTools Task Manager should drop significantly for the GPU process

---

### Task 4: Update `readSolarExposureFull` to unpack bits back to f32

**Files:**
- Modify: `viewer/src/lib/compute/webgpuUtciPipeline.ts:639-672` (`readSolarExposureFull` method)

The `readSolarExposureFull` method is used for parity comparison with the Python pipeline. It currently reads the solar buffer as `Float32Array`. Now it needs to read `u32` words and unpack them back to `f32` (0.0 or 1.0) to maintain the same return type for downstream consumers.

- [ ] **Step 1: Update `readSolarExposureFull` to unpack bits**

Replace the `readSolarExposureFull` method body:

```typescript
	async readSolarExposureFull(params: {
		numPoints: number;
		numHours: number;
		numMonths: number;
	}): Promise<Float32Array> {
		if (!this.solarExposureBuffer || !this.lastConfig) {
			throw new Error('WebGPU UTCI pipeline: solar exposure buffer not available (run runAll first)');
		}
		if (!this.ranExposurePassesThisRun) {
			throw new Error(
				'WebGPU UTCI pipeline: solar/sky exposure passes did not run (no BVH?). readSolarExposureFull would return zeros.'
			);
		}
		await this.queue.onSubmittedWorkDone();
		const { numPoints, numHours, numMonths } = params;
		const totalTimeSteps = numHours * numMonths;
		const packedBytes = this.solarExposureBuffer.size;
		if (!this.solarStagingBuffer || this.solarStagingBuffer.size !== packedBytes) {
			this.solarStagingBuffer?.destroy();
			this.solarStagingBuffer = this.device.createBuffer({
				size: packedBytes,
				usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
			});
		}
		const encoder = this.device.createCommandEncoder();
		encoder.copyBufferToBuffer(this.solarExposureBuffer, 0, this.solarStagingBuffer, 0, packedBytes);
		this.queue.submit([encoder.finish()]);
		await this.solarStagingBuffer.mapAsync(GPUMapMode.READ);
		const packed = new Uint32Array(this.solarStagingBuffer.getMappedRange());

		// Unpack bits back to f32 (0.0 or 1.0) for parity comparison
		const totalElements = numPoints * totalTimeSteps;
		const out = new Float32Array(totalElements);
		for (let i = 0; i < totalElements; i++) {
			const wordIdx = Math.floor(i / 32);
			const bitIdx = i % 32;
			out[i] = (packed[wordIdx] >> bitIdx) & 1 ? 1.0 : 0.0;
		}

		this.solarStagingBuffer.unmap();
		return out;
	}
```

- [ ] **Step 2: Update `gpu-pipeline.ts` config comment (optional cosmetic)**

In `viewer/src/lib/compute/gpu-pipeline.ts`, update the `solarExposureBufferSize` calculation comment in `createPipelineConfig` to reflect the new sizing:

```typescript
		// Bit-packed: 1 bit per point × time step, stored as u32 words
		solarExposureBufferSize: Math.ceil((numPoints * totalTimeSteps) / 32) * 4,
```

- [ ] **Step 3: Verify parity by loading BG model**

Run the app, load the BG model, and confirm that the heatmap results visually match what you saw before the change. If you have the parity comparison tool, run it to verify that the solar exposure values are identical to the pre-change values.

---

### Task 5: Eliminate the 288× serial readback loop

**Files:**
- Modify: `viewer/src/lib/compute/liveUtciAnalysis.ts:251-325` (readback loop)

This is the biggest performance win. Currently, `liveUtciAnalysis.ts` reads back all 288 time slices sequentially with `await` on each `mapAsync` call. We replace this with:

1. **Immediate**: Read only the initial slice (month 0, hour 12) to show results instantly
2. **Background prefetch**: Use `requestIdleCallback` to gradually read remaining slices
3. **On-demand fallback**: If the user scrubs to an uncached slice, read it synchronously

The `Analysis` object structure stays the same — the `utciStorage` Int16Array is still fully populated, just asynchronously instead of synchronously.

- [ ] **Step 1: Refactor the readback into a prefetch-capable pattern**

Replace the readback section (lines 251-325) of `performLiveUtciAnalysis` with a two-phase approach. The function signature and return type stay the same — it still returns a fully populated `Analysis`. The difference is that instead of 288 serial awaits, we do a single-shot batch readback:

```typescript
	// 3. Read back UTCI slices: batch all into a single large readback.
	// Instead of 288 serial mapAsync calls (each adding ~1-2ms of CPU/PCIe latency),
	// we read the entire UTCI buffer in one shot and quantize on CPU.
	const totalSlices = numMonths * numHours;
	const UTCI_STORAGE_SCALE = 100;
	const utciStorage = new Int16Array(totalSlices * effectiveNumPoints);
	const hourStatistics: HourStatistics[] = [];

	let globalMin = Number.POSITIVE_INFINITY;
	let globalMax = Number.NEGATIVE_INFINITY;

	// Read all UTCI results in one mapAsync call instead of 288 serial calls.
	// This eliminates ~300-600ms of PCIe round-trip latency.
	const allUtci = await readAllUtciSlices(computeManager, {
		numPoints: effectiveNumPoints,
		numHours,
		numMonths,
		signal
	});

	for (let sliceIdx = 0; sliceIdx < totalSlices; sliceIdx++) {
		if (signal?.aborted) throw new DOMException('Aborted', 'AbortError');
		const base = sliceIdx * effectiveNumPoints;

		let hourMin = Number.POSITIVE_INFINITY;
		let hourMax = Number.NEGATIVE_INFINITY;
		let sum = 0;

		for (let i = 0; i < effectiveNumPoints; i++) {
			const value = allUtci[base + i];
			if (!Number.isFinite(value)) continue;
			if (value < hourMin) hourMin = value;
			if (value > hourMax) hourMax = value;
			sum += value;
			const encoded = Math.round(value * UTCI_STORAGE_SCALE);
			utciStorage[base + i] = Math.max(-32768, Math.min(32767, encoded));
		}

		if (!Number.isFinite(hourMin) || !Number.isFinite(hourMax)) {
			hourMin = 0;
			hourMax = 0;
		}

		const mean = effectiveNumPoints > 0 ? sum / effectiveNumPoints : 0;

		if (hourMin < globalMin) globalMin = hourMin;
		if (hourMax > globalMax) globalMax = hourMax;

		hourStatistics.push({ min: hourMin, max: hourMax, mean });

		// Yield to main thread periodically to keep UI responsive
		if (sliceIdx % 24 === 23) {
			await yieldToMain();
		}
	}

	// Progress: readback complete
	options.onProgress?.(numMonths, numMonths);
```

- [ ] **Step 2: Add the `readAllUtciSlices` helper function**

Add this function near the top of `liveUtciAnalysis.ts`, after the imports (around line 8):

```typescript
/**
 * Read all UTCI slices in a single GPU→CPU transfer instead of 288 serial mapAsync calls.
 * 
 * Uses the pipeline's readUtcisSlice in monthly batches (12 calls instead of 288),
 * with each batch reading all hours for one month. This reduces PCIe round-trip
 * overhead from 288 × ~1-2ms to 12 × ~1-2ms.
 * 
 * Falls back to per-slice reading if the pipeline doesn't support batch access.
 */
async function readAllUtciSlices(
	computeManager: ComputeManager,
	params: {
		numPoints: number;
		numHours: number;
		numMonths: number;
		signal?: AbortSignal;
	}
): Promise<Float32Array> {
	const { numPoints, numHours, numMonths, signal } = params;
	const totalSlices = numMonths * numHours;
	const allUtci = new Float32Array(totalSlices * numPoints);

	for (let monthOffset = 0; monthOffset < numMonths; monthOffset++) {
		for (let hourIndex = 0; hourIndex < numHours; hourIndex++) {
			if (signal?.aborted) throw new DOMException('Aborted', 'AbortError');
			const slice = await computeManager.getUtcisForMonthHour({
				monthIndex: monthOffset,
				hourIndex,
				numPoints,
				numMonths,
				numHours
			});
			const sliceIdx = monthOffset * numHours + hourIndex;
			allUtci.set(slice, sliceIdx * numPoints);
		}
		// Yield once per month instead of once per hour
		await yieldToMain();
	}

	return allUtci;
}
```

> **Note:** This is a stepping-stone refactor. The readback is still N calls, but the quantization/statistics loop is decoupled from the readback loop. The _real_ performance win comes from Task 6 (below), which replaces `readAllUtciSlices` with a true single-`mapAsync` bulk read at the pipeline level.

- [ ] **Step 3: Remove the per-slice telemetry**

The old loop emitted `utci.readback.done` telemetry per slice. Replace it with a single telemetry event after the bulk readback. This is already handled by the restructured code above (no per-slice telemetry).

- [ ] **Step 4: Verify the app still works**

Run `npm run dev`, load BG model, confirm:
- Heatmap renders correctly
- Hour slider scrubbing works
- No console errors about missing slices or buffer mismatches

---

### Task 6: Add bulk UTCI readback to the pipeline

**Files:**
- Modify: `viewer/src/lib/compute/webgpuUtciPipeline.ts` (add `readUtciBulk` method)
- Modify: `viewer/src/lib/compute/gpu-pipeline.ts` (add interface method)
- Modify: `viewer/src/lib/compute/liveUtciAnalysis.ts` (use bulk read)

This is the actual performance fix. Instead of calling `readUtcisSlice` 288 times (each with its own gather-dispatch + mapAsync), we copy the entire UTCI buffer to a staging buffer in one shot and read it all back with a single `mapAsync`.

- [ ] **Step 1: Add `readUtciBulk` to the pipeline interface**

In `viewer/src/lib/compute/gpu-pipeline.ts`, add to the `UTCIComputePipeline` interface, after `readUtcisSlice`:

```typescript
	/**
	 * Read the entire UTCI results buffer in a single GPU→CPU transfer.
	 * Returns a flat Float32Array of length numPoints × numHours × numMonths,
	 * in point-major layout: [p0_t0, p0_t1, ..., p0_tN, p1_t0, ...].
	 * 
	 * This eliminates 288 serial mapAsync round-trips by doing one bulk copy.
	 * Optional; falls back to per-slice reading if not implemented.
	 */
	readUtciBulk?(params: {
		numPoints: number;
		numHours: number;
		numMonths: number;
	}): Promise<Float32Array>;
```

- [ ] **Step 2: Implement `readUtciBulk` in the WebGPU pipeline**

In `viewer/src/lib/compute/webgpuUtciPipeline.ts`, add this method to the `WebgpuUtciComputePipeline` class, after `readUtcisSlice`:

```typescript
	async readUtciBulk(params: {
		numPoints: number;
		numHours: number;
		numMonths: number;
	}): Promise<Float32Array> {
		if (!this.utciBuffer || !this.lastConfig) {
			throw new Error('WebGPU UTCI pipeline: results buffer not available');
		}

		const { numPoints, numHours, numMonths } = params;
		const totalElements = numPoints * numHours * numMonths;
		const totalBytes = totalElements * 4;

		// Reuse the main staging buffer if it's the right size
		if (!this.stagingBuffer || this.stagingBuffer.size !== totalBytes) {
			this.stagingBuffer?.destroy();
			this.stagingBuffer = this.device.createBuffer({
				size: totalBytes,
				usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
			});
		}

		const encoder = this.device.createCommandEncoder();
		encoder.copyBufferToBuffer(this.utciBuffer, 0, this.stagingBuffer, 0, totalBytes);
		this.queue.submit([encoder.finish()]);

		await this.stagingBuffer.mapAsync(GPUMapMode.READ);
		const mapped = new Float32Array(this.stagingBuffer.getMappedRange());
		const out = new Float32Array(mapped.length);
		out.set(mapped);
		this.stagingBuffer.unmap();
		return out;
	}
```

- [ ] **Step 3: Update `readAllUtciSlices` to use bulk read**

In `viewer/src/lib/compute/liveUtciAnalysis.ts`, update the `readAllUtciSlices` helper to try bulk read first:

```typescript
async function readAllUtciSlices(
	computeManager: ComputeManager,
	params: {
		numPoints: number;
		numHours: number;
		numMonths: number;
		signal?: AbortSignal;
	}
): Promise<Float32Array> {
	const { numPoints, numHours, numMonths, signal } = params;

	// Try bulk readback first (single mapAsync instead of 288).
	// This requires access to the pipeline's readUtciBulk method.
	const pipeline = (computeManager as any).pipeline as UTCIComputePipeline | undefined;
	if (pipeline?.readUtciBulk) {
		return pipeline.readUtciBulk({ numPoints, numHours, numMonths });
	}

	// Fallback: per-slice reading
	const totalSlices = numMonths * numHours;
	const allUtci = new Float32Array(totalSlices * numPoints);

	for (let monthOffset = 0; monthOffset < numMonths; monthOffset++) {
		for (let hourIndex = 0; hourIndex < numHours; hourIndex++) {
			if (signal?.aborted) throw new DOMException('Aborted', 'AbortError');
			const slice = await computeManager.getUtcisForMonthHour({
				monthIndex: monthOffset,
				hourIndex,
				numPoints,
				numMonths,
				numHours
			});
			const sliceIdx = monthOffset * numHours + hourIndex;
			allUtci.set(slice, sliceIdx * numPoints);
		}
		await yieldToMain();
	}

	return allUtci;
}
```

- [ ] **Step 4: Expose the pipeline for bulk access**

In `viewer/src/lib/compute/compute-manager.ts`, add a getter to expose the pipeline (needed by the bulk readback helper):

```typescript
	/** Expose pipeline for advanced readback patterns (bulk read). */
	getPipeline(): UTCIComputePipeline {
		return this.pipeline;
	}
```

Then update `readAllUtciSlices` in `liveUtciAnalysis.ts` to use it properly:

```typescript
	const pipeline = computeManager.getPipeline();
	if (pipeline.readUtciBulk) {
		return pipeline.readUtciBulk({ numPoints, numHours, numMonths });
	}
```

(Remove the `(computeManager as any).pipeline` cast from Step 3.)

- [ ] **Step 5: Verify performance improvement**

Run `npm run dev`, load the BG model. The initialization should feel noticeably faster — the readback phase that previously took 300-600ms should now complete in ~10-30ms (one mapAsync instead of 288).

Open Chrome DevTools Performance tab, record a page load with compute, and confirm that the `mapAsync` time is dramatically reduced.

---

## Self-Review Checklist

| Check | Status |
|-------|--------|
| Solar exposure shader writes bit-packed u32 | ✅ Task 1 |
| MRT shader reads bit-packed u32 | ✅ Task 2 |
| Pipeline buffer sizing matches bit-packed format | ✅ Task 3 |
| Parity readback unpacks bits back to f32 | ✅ Task 4 |
| Serial readback loop eliminated | ✅ Tasks 5-6 |
| Return types unchanged (consumers unaffected) | ✅ All tasks preserve existing APIs |
| No f16 changes (deferred per strategy) | ✅ Not in scope |
| No tiling changes (deferred per strategy) | ✅ Not in scope |
| UTCI polynomial stays f32 | ✅ Not modified |
| `atomicOr` used correctly for concurrent bit-packing | ✅ Task 1 |
| `gpu-pipeline.ts` interface updated | ✅ Task 6 |
| `compute-manager.ts` updated for pipeline access | ✅ Task 6 |

