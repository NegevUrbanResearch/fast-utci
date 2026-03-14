# WebGPU Deep Code Review Action Plan (2026-03-14)

## 1. Executive Summary
- Crash risk is reduced but not eliminated: the current safety gate is triangle-only, while actual failure pressure is memory-bytes across CPU copies, worker merge/BVH, GPU buffers, and readback caches.
- The largest remaining freeze source near "Computing UTCI" is still main-thread heavy work: duplicate full-scene traversals, payload copying, synchronous pipeline compilation, and per-hour CPU slicing/statistics.
- `MAX_GRID_POINTS_GUARD` exists but is not enforced, so very dense grids can still explode memory/time even when triangle caps pass.
- Worker offload exists but is still mostly one-shot bursts (merge -> grid -> BVH), with limited cooperative cancellation and only coarse progress events.
- Solar compute currently dispatches for all hours and still traverses BVH at night (zero sun vectors), wasting substantial compute.
- WebGPU parity improvements were made (north sign, reflectance `0.25`, default `zHeight=0.9`), but parity is still not scientifically locked due to sun model/time sampling and fixture limitations.
- Sun vectors are generated with a custom NOAA-style implementation sampled at `hour + 0.5`, while Python uses Ladybug hourly datetimes; this is a likely systematic parity bias.
- Current "WebGPU vs Python" fixture test is not a runtime WebGPU parity execution; it validates stored constants and CPU UTCI, so it can pass while GPU pipeline behavior drifts.
- Device-loss recovery is missing (`GPUDevice.lost` handling), so OOM/device reset can leave the app in a non-recovering state.
- The architecture does not need a full rewrite; it needs a targeted partial rewrite of the compute orchestration path (memory-budgeted streaming + async pipeline compile + chunked readback/statistics + true parity harness).

## 2. Findings by Severity

### P0 Critical

#### P0-1: Memory budget is not enforced by bytes/points, only by triangles
- Evidence (file + line refs):
  - `viewer/src/routes/debug-webgpu-utci/+page.svelte:229-241` enforces triangle caps but no grid-point or byte cap.
  - `viewer/src/lib/compute/mergeAndBvhWorkerClient.ts:32` defines `MAX_GRID_POINTS_GUARD` but it is never used.
  - `viewer/src/lib/compute/mergeAndBvh.worker.ts:87-97` builds `gridPoints` with no cap.
  - `viewer/src/lib/compute/webgpuUtciPipeline.ts:108-125`, `371-387`, `507-517` allocate large exposure/result/staging/readback buffers.
  - `viewer/src/lib/compute/liveUtciAnalysis.ts:182-236` stores per-hour slices and stats in-memory.
- Why it matters: crashes/freezes in browsers are governed by peak memory and GC pressure, not triangle count alone.
- Repro/trigger conditions: large geometry with moderate triangles but huge area/dense grid; or high point count with 24h readback/materialization.
- Recommended fix:
  - Introduce a preflight byte-budget estimator and hard-fail before payload prep.
  - Enforce `maxGridPoints` before worker returns data.
  - Gate run by estimated total bytes: payload copies + worker merged/BVH + GPU buffers + readback cache + analysis arrays.
- Effort: M
- Confidence: High

#### P0-2: Peak-memory amplification remains high across main thread, worker, and readback
- Evidence:
  - Main-thread payload copies all mesh arrays before worker transfer: `viewer/src/lib/compute/mergeAndBvhWorkerClient.ts:119-166`.
  - Worker rebuilds geometries and merges in one shot: `viewer/src/lib/compute/mergeAndBvh.worker.ts:70-75`.
  - Large zero-array allocations on CPU before GPU write: `viewer/src/lib/compute/webgpuUtciPipeline.ts:127-144`.
  - Full UTCI cache copy for readback: `viewer/src/lib/compute/webgpuUtciPipeline.ts:510-517`.
  - Per-hour slices duplicated into final analysis: `viewer/src/lib/compute/liveUtciAnalysis.ts:231`, `267-272`.
- Why it matters: this can still push the tab/process into OOM territory for large models (especially when multiple copies coexist).
- Repro/trigger conditions: near upper model/grid sizes; repeated runs without full memory recovery.
- Recommended fix:
  - Stream/chunk payload-to-worker instead of building full copied payload in one array.
  - Remove CPU-side zero-array initialization in favor of GPU-side clear strategy.
  - Use hour-slice GPU gather/readback, not full-buffer cache for all workflows.
- Effort: L
- Confidence: High

#### P0-3: No device-loss recovery path after GPU reset/OOM
- Evidence:
  - Global cached device promise with no `device.lost` handling: `viewer/src/lib/compute/webgpuUtciPipeline.ts:609-616`.
- Why it matters: after device loss, subsequent runs may fail repeatedly or appear stuck.
- Repro/trigger conditions: memory pressure, driver reset, adapter instability.
- Recommended fix:
  - Subscribe to `device.lost`; invalidate `cachedDevicePromise`; recreate adapter/device and surface clear error telemetry.
- Effort: S
- Confidence: Medium

### P1 High

#### P1-1: Main-thread long tasks remain at compute start due duplicate traversals and sync prep
- Evidence:
  - First full traversal for triangle counting: `viewer/src/routes/debug-webgpu-utci/+page.svelte:223`.
  - Another full `updateMatrixWorld + traverse` in payload prep: `viewer/src/lib/compute/mergeAndBvhWorkerClient.ts:102-113`.
- Why it matters: this directly matches the observed multi-second freeze near "Computing UTCI".
- Repro/trigger conditions: large scene graphs, high mesh count.
- Recommended fix:
  - Single-pass scene walk that collects both counts and payload metadata.
  - Chunk traversal with periodic yielding and early abort checks.
- Effort: M
- Confidence: High

#### P1-2: Pipeline compilation is synchronous and re-done per run
- Evidence:
  - Synchronous `createComputePipeline` in all pipelines: `viewer/src/lib/compute/webgpuUtciPipeline.ts:65`, `324`, `333`.
  - Pipeline recreated each run: `viewer/src/routes/debug-webgpu-utci/+page.svelte:271-274`.
- Why it matters: shader/pipeline compilation on the main thread can introduce jank.
- Repro/trigger conditions: first run per model, repeated project switches.
- Recommended fix:
  - Switch to `createComputePipelineAsync` and warm pipelines once.
  - Reuse a long-lived pipeline object per page/session where safe.
- Effort: M
- Confidence: High

#### P1-3: Solar pass does unnecessary BVH traversal for nighttime hours
- Evidence:
  - Night sun vectors set to `[0,0,0]`: `viewer/src/lib/compute/sunpath.ts:121-123`.
  - Solar shader always calls `bvh_intersects_any`: `viewer/src/lib/compute/shaders/exposure_solar.wgsl:56-61`.
  - Dispatch includes all timesteps: `viewer/src/lib/compute/webgpuUtciPipeline.ts:430`.
- Why it matters: significant wasted compute for sun-down hours.
- Repro/trigger conditions: any day with substantial night period.
- Recommended fix:
  - Add sun-up mask/altitude guard in shader and short-circuit to `0` exposure.
  - Optionally compact dispatch to sun-up timesteps only.
- Effort: M
- Confidence: High

#### P1-4: Readback + slice/stats loops are main-thread O(points * hours) hot path
- Evidence:
  - Per-slice extraction loop: `viewer/src/lib/compute/webgpuUtciPipeline.ts:541-545`.
  - Per-hour statistics loop: `viewer/src/lib/compute/liveUtciAnalysis.ts:207-213`.
- Why it matters: large CPU loops can create visible long tasks and UI hitching.
- Repro/trigger conditions: large `numPoints`, 24-hour runs.
- Recommended fix:
  - Compute reductions on GPU (min/max/sum) or chunk JS loops with `await` yields.
  - Use direct hour-slice GPU gather to reduce copied working set.
- Effort: M
- Confidence: High

#### P1-5: Worker remains bursty, with limited cooperative cancellation
- Evidence:
  - Worker has only coarse yields and no abort checks during heavy stages: `viewer/src/lib/compute/mergeAndBvh.worker.ts:70-100`.
  - Timeout-based cancellation only from client side: `viewer/src/lib/compute/mergeAndBvhWorkerClient.ts:239-242`.
- Why it matters: long worker runs become opaque and can still stress the browser process.
- Repro/trigger conditions: big merge/BVH workloads.
- Recommended fix:
  - Implement staged worker protocol (`prepare -> merge -> grid -> bvh -> transfer`) with periodic cancellation polling.
  - Emit progress with timing and memory estimates at each stage.
- Effort: M
- Confidence: High

#### P1-6: Remaining parity gap from sun model/time sampling mismatch
- Evidence:
  - Web viewer uses custom NOAA-style solar model and half-hour sampling: `viewer/src/lib/compute/sunpath.ts:21`, `103-107`.
  - Python path uses Ladybug `Sunpath` on hourly datetimes: `src/fast_utci/mrt/solar.py:41`, `65-67`.
- Why it matters: directional and timing differences can produce systematic MRT/UTCI bias even if shader math is correct.
- Repro/trigger conditions: low solar altitude hours and seasonal shoulder periods.
- Recommended fix:
  - Add a parity mode that consumes Python-generated sun vectors/timestamps directly.
  - Validate ENU->Y-up transform with known timestamp/vector fixtures.
- Effort: M
- Confidence: High

#### P1-7: Current parity test can pass without exercising live WebGPU runtime parity
- Evidence:
  - `viewer/tests/compute/webgpu-python-parity.test.ts:28-45` compares stored fixture constants and CPU UTCI, not full WebGPU pipeline runtime outputs.
  - WebGPU integration checks are skipped without WebGPU env: `viewer/tests/compute/bvhGpuUpload.test.ts:89` and `viewer/tests/compute/mrt-utci-gpu-solarcal.test.ts:58`.
- Why it matters: false confidence; regressions in GPU path can slip through CI.
- Repro/trigger conditions: shader changes, data-layout drift, driver differences.
- Recommended fix:
  - Add browser-backed integration parity tests (Playwright + WebGPU-capable runner) with intermediate-buffer assertions.
- Effort: M
- Confidence: High

### P2 Medium

#### P2-1: Stale zHeight documentation can mislead parity/debugging
- Evidence:
  - Comment says default `1.5m`: `viewer/src/lib/compute/liveUtciAnalysis.ts:57-58`.
  - Actual default is `0.9`: `viewer/src/lib/compute/liveUtciAnalysis.ts:101`.
- Why it matters: review/debug confusion and incorrect assumptions.
- Repro/trigger conditions: audits and future refactors.
- Recommended fix: align comments/docs/tests with current parity default.
- Effort: S
- Confidence: High

#### P2-2: Abort cleanup on component destroy is incomplete
- Evidence:
  - `onDestroy` disposes pipeline but does not abort active run controller: `viewer/src/routes/debug-webgpu-utci/+page.svelte:453-461`.
- Why it matters: in-flight async work can continue post-unmount.
- Repro/trigger conditions: navigation during compute.
- Recommended fix: abort `liveAbortController` in `onDestroy` and clear handlers.
- Effort: S
- Confidence: Medium

#### P2-3: Intended grid guard is dead code
- Evidence:
  - `MAX_GRID_POINTS_GUARD` defined but unused: `viewer/src/lib/compute/mergeAndBvhWorkerClient.ts:32`.
- Why it matters: latent safety requirement exists but is not enforced.
- Repro/trigger conditions: dense grid inputs.
- Recommended fix: enforce it in worker result and preflight checks.
- Effort: S
- Confidence: High

## 3. Freeze Root-Cause Hypotheses Ranked

### 1) Main-thread compute bootstrap still creates long tasks (most likely)
- Why likely:
  - Duplicate full traversals + sync matrix updates + payload copy on main thread.
  - Evidence: `+page.svelte:223`, `mergeAndBvhWorkerClient.ts:102-113`, `119-166`.
- Data that proves/disproves:
  - Chrome Performance trace with long-task markers around `live.compute.start` and `payload.prepare.done` telemetry.
  - Main-thread flame chart showing `traverse`, `updateMatrixWorld`, typed-array copies.

### 2) Readback/slice/statistics CPU loops dominate post-GPU stage
- Why likely:
  - Per-hour extraction and stat loops are O(points * hours).
  - Evidence: `webgpuUtciPipeline.ts:541-545`, `liveUtciAnalysis.ts:207-213`.
- Data that proves/disproves:
  - Time budget per hour from telemetry + JS profile around readback loop.
  - Compare with temporary chunked-yield prototype to see freeze reduction.

### 3) Sync pipeline compilation + repeated pipeline recreation causes startup hitch
- Why likely:
  - Sync `createComputePipeline` and per-run recreate.
  - Evidence: `webgpuUtciPipeline.ts:65,324,333`; `+page.svelte:271-274`.
- Data that proves/disproves:
  - Capture first-run trace with shader compilation cost.
  - A/B with async pipeline creation and warm cache.

### 4) Worker burst merge/grid/BVH creates memory pressure and apparent freeze
- Why likely:
  - Worker does large sequential stages with limited yields.
  - Evidence: `mergeAndBvh.worker.ts:70-100`.
- Data that proves/disproves:
  - Chrome Memory timeline + worker thread profile.
  - Stage-level telemetry with resident memory snapshots.

### 5) Remaining model-load/GLTF parse cost spills into compute perception (least likely for this specific stage)
- Why likely:
  - GLTF parse is still synchronous in browser pipeline.
- Data that proves/disproves:
  - Distinguish timeline region before vs after first `live.compute.start` event.

## 4. Parity Gap Analysis

### Coordinate Frames
- Current state:
  - ENU -> Y-up transform in compute manager is now `[x, z, -y]` (`viewer/src/lib/compute/compute-manager.ts:14-17`).
- Remaining risk:
  - Sun model source differs (custom NOAA vs Ladybug), so frame transform parity alone is insufficient.

### Constants
- Current state:
  - Ground reflectance now `0.25` in WGSL (`viewer/src/lib/compute/shaders/mrt_utci.wgsl:72`) matching Python config (`fast_utci.toml:47`).
  - Live default `zHeight` is `0.9` (`+page.svelte:220`, `liveUtciAnalysis.ts:101`).
- Remaining risk:
  - Comment/document mismatch still says 1.5m in one location.

### Sampling Assumptions
- Current divergence:
  - Web sun vectors sampled at `hour + 0.5` (`sunpath.ts:106`), Python uses hourly Ladybug datetimes (`src/fast_utci/mrt/solar.py:65-67`).
  - Web grid uses raycast-over-mesh walkable surfaces (`viewer/src/lib/compute/grid-generator.ts:74-147`); Python can use rectangular/face-center style depending pipeline (`src/fast_utci/mrt/grid.py`).
- Impact:
  - Different point sets and sun vectors create systematic UTCI differences independent of GPU math.

### Data Layout
- Current state:
  - Point-major UTCI layout is consistent (`getUtciFlatIndex` and shader indexing).
- Remaining risk:
  - Runtime tests do not verify intermediate arrays (sun vectors, exposures, MRT) against Python fixtures.

### Precision Assumptions
- Current divergence:
  - WGSL pipeline is `f32`; Python/Ladybug stack often uses `float64` and different math-library implementations.
- Impact:
  - Small deterministic deltas are expected; parity must be tolerance-based and stage-wise, not bitwise.

### Concrete Parity Test Improvements
- Add stage fixtures containing:
  - Sun vectors per hour, sky weights normalization, solar exposure fractions, MRT, UTCI.
- Run WebGPU integration in browser CI and compare:
  - `sunVectors`, `sunAltitudes`, `solarExposure`, `skyExposure`, `mrt`, `utci`.
- Use tolerance tiers:
  - Stage-level tolerances (tight for geometry/indexing, looser for floating-point thermo outputs).
- Add explicit scenario matrix:
  - Midday clear sky, low-sun winter hour, shaded canyon, high-wind case.

## 5. Internet Research Notes

### Best-practice summary + links
- WebGPU pipeline creation:
  - Prefer async pipeline creation to avoid blocking compilation (`createComputePipelineAsync`).
  - Sources:
    - MDN: https://developer.mozilla.org/en-US/docs/Web/API/GPUDevice/createComputePipelineAsync
    - Khronos best-practices deck: https://www.khronos.org/assets/uploads/developers/library/2022-webinar/webgpu-and-webgl-best-practices-july22.pdf
- Readback patterns:
  - `mapAsync` is asynchronous and can queue; `onSubmittedWorkDone()` can coordinate scheduling and avoid over-queuing.
  - Sources:
    - MDN `mapAsync`: https://developer.mozilla.org/en-US/docs/Web/API/GPUBuffer/mapAsync
    - MDN `onSubmittedWorkDone`: https://developer.mozilla.org/en-US/docs/Web/API/GPUQueue/onSubmittedWorkDone
- Worker/off-main-thread architecture:
  - Move heavy work off main thread, but also avoid giant single tasks and optimize message payload handling.
  - Sources:
    - web.dev Off-main-thread: https://web.dev/articles/off-main-thread
    - MDN Using workers: https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API/Using_web_workers
- Transferable ownership:
  - Transfer `ArrayBuffer` ownership (not typed-array object itself) to avoid extra clone; sender buffer becomes detached.
  - Source:
    - MDN transferables: https://developer.mozilla.org/en-US/docs/Web/API/Transferable_objects
- Browser freeze mitigation:
  - Break up long tasks; tasks >50ms hurt responsiveness.
  - Source:
    - web.dev long tasks: https://web.dev/articles/optimize-long-tasks
- Scientific parity validation:
  - Floating-point differences across hardware/libraries are expected; compare with tolerances and controlled operation ordering.
  - Sources:
    - NVIDIA floating-point guidance (CPU vs GPU comparison caveats): https://docs.nvidia.com/cuda/archive/13.0.0/floating-point/index.html
    - NumPy `assert_allclose`: https://numpy.org/doc/stable/reference/generated/numpy.testing.assert_allclose
    - PyTorch numerical accuracy note: https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html

### What experts would say (synthesis)
- "Measure peak memory and task duration first; cap by bytes, not just triangles."
- "Compile pipelines asynchronously and reuse them."
- "Treat worker architecture as staged/streaming, not one giant message + one giant merge."
- "Use transferables correctly, but eliminate upstream copies where possible."
- "Parity should be stage-wise with tolerance contracts and fixed fixtures, not endpoint-only snapshots."

## 6. Prioritized Remediation Plan

### Phase 1 (Safety)
- Actions:
  - Add byte-budget preflight and hard fail before payload prep.
  - Enforce grid-point cap (`MAX_GRID_POINTS_GUARD`) and surface explicit user error.
  - Add `GPUDevice.lost` recovery and reset cached device promise.
- Acceptance criteria:
  - No crashes on NZ stress case; deterministic graceful failure when exceeding limits.
  - Telemetry includes `estimatedBytes`, `numPoints`, and fail reason.
- Rollback strategy:
  - Feature-flag byte-budget guard and fall back to current behavior if false positives appear.

### Phase 2 (Freeze/Perf)
- Actions:
  - Replace sync pipeline creation with async + warmup + reuse.
  - Merge triangle count + payload collection into one traversal.
  - Skip night-hour solar BVH traversal.
  - Chunk readback/stat loops or move reductions to GPU.
- Acceptance criteria:
  - No main-thread tasks >100ms during compute startup on target NZ/BGU runs.
  - `payload.prepare.done` and per-hour readback timings reduce by agreed threshold.
- Rollback strategy:
  - Keep old path behind debug toggle for A/B timing and emergency fallback.

### Phase 3 (Parity Hardening)
- Actions:
  - Introduce parity-mode sun vectors from Python/Ladybug fixtures.
  - Add browser WebGPU integration parity tests with intermediate stage assertions.
  - Formalize tolerance budgets per stage/output.
- Acceptance criteria:
  - Fixture suite passes in CI with documented tolerances.
  - Known parity deltas are explained by model assumptions, not unknown drift.
- Rollback strategy:
  - Keep legacy parity comparator as secondary report while new harness stabilizes.

## 7. Test/Telemetry Additions Needed

### Missing tests
- Worker client and worker lifecycle:
  - cancellation, timeout, progress ordering, transfer-list correctness.
- Safety guards:
  - byte-budget rejection, grid-point cap rejection, device-loss recovery path.
- Real WebGPU parity:
  - browser integration tests for intermediate buffers and final UTCI.

### Missing instrumentation
- Stage timings for:
  - triangle count traversal, payload traverse/copy, worker merge/grid/BVH, pipeline compile, readback extraction, statistics aggregation.
- Memory estimates:
  - predicted and observed bytes per stage.
- Long-task markers:
  - emit when a stage exceeds 50ms/100ms on main thread.

### Runtime guardrails
- Hard limits:
  - `maxEstimatedBytes`, `maxGridPoints`, `maxHours` in debug mode.
- Watchdogs:
  - abort stale runs aggressively and gate concurrent runs.
- Recovery:
  - automatic device reinit after `device.lost` with user-visible status.

## 8. Do We Need a Rewrite?
- Clear answer: No full rewrite.
- Partial rewrite needed:
  - Rewrite compute orchestration around memory-budgeted staged execution.
  - Rewrite readback/post-processing path to avoid large synchronous loops.
  - Rewrite parity verification as stage-wise, browser-executed harness.
- Why:
  - Core architecture (worker offload + GPU compute + adapter layer) is viable; bottlenecks are concentrated in orchestration and verification layers.

## 9. Quick Wins (next 1-2 days)
- Enforce `MAX_GRID_POINTS_GUARD` and add byte-estimate preflight before payload preparation.
- Convert pipeline creation to async (`createComputePipelineAsync`) and cache/reuse across runs.
- Add night-hour early exit in solar shader to skip BVH traversal when sun is down.
- Add `device.lost` handler and reset cached device promise.
- Fix stale zHeight comment/doc mismatch and add a parity metadata banner in debug UI.

## 10. Appendix

## 11. Implementation Status Snapshot (2026-03-14)
- Implemented safety preflight in payload preparation with estimated byte budgeting and grid-point cap enforcement before heavy copies.
- Current operational `MAX_GRID_POINTS_GUARD` is set to `600,000` (tuned for Nes Tziona runtime stability/performance).
- Worker flow now uses staged progress (`prepare`, `merge`, `grid`, `bvh`, `transfer`) with cooperative cancellation checks and cancel message handling.
- Solar exposure shader now short-circuits night hours (`sun.y <= 0` or zero-length vectors) before BVH traversal.
- WebGPU pipelines moved to `createComputePipelineAsync` with cached pipeline promises.
- UTCI readback changed from full-buffer CPU cache to hour-slice gather shader readback.
- Added `GPUDevice.lost` handling that clears cached device state for reinitialization on next run.
- Added parity fixture-mode support in `ComputeManager` for injecting Python-generated sun vectors/altitudes directly.
- Added runtime parity harness scaffolding (`viewer/tests/compute/webgpu-runtime-parity.test.ts`) and stage fixture contract.

### Command snippets
```bash
# Focused compute tests used in this review
cd viewer
npm run test -- tests/compute/meshMerger.test.ts tests/compute/compute-manager.test.ts tests/compute/live-utci-analysis.test.ts tests/compute/bvhGpuUpload.test.ts tests/compute/mrt-utci-gpu-solarcal.test.ts tests/compute/webgpu-python-parity.test.ts

# Locate worker/transfer/readback hotspots
rg -n "prepareMeshPayloadForWorkerAsync|mergeGeometries|mapAsync|copyBufferToBuffer|createComputePipeline|createComputePipelineAsync|MAX_GRID_POINTS_GUARD|device\.lost" viewer/src/lib/compute viewer/src/routes/debug-webgpu-utci/+page.svelte

# Confirm test coverage gaps around worker path
rg -n "mergeAndBvhWorkerClient|mergeAndBvh\.worker|runMergeAndBvhInWorker" viewer/tests/compute
```

### Profiling steps (Chrome)
1. Open `debug-webgpu-utci`, load BGU then NZ.
2. Record Performance trace from model load through live compute completion.
3. Enable screenshots + memory in Performance settings.
4. Correlate long tasks with telemetry stages (`payload.prepare.done`, `worker.*`, `utci.readback.done`).
5. Capture a Memory timeline to inspect peak RSS around worker merge/BVH and readback cache creation.
6. Repeat after each phase change and compare:
   - max main-thread task duration,
   - total compute wall-time,
   - peak memory usage,
   - parity delta summary.
