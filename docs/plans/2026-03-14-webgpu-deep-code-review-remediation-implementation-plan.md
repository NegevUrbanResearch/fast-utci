# WebGPU Deep Code Review Remediation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement all findings and recommendations from the 2026-03-14 deep WebGPU review so large-scene runs are safe, responsive, and parity-verifiable.

**Architecture:** Keep the existing viewer + worker + WebGPU pipeline architecture, but harden it with byte-budgeted preflight, staged worker execution, async pipeline lifecycle, chunked post-processing, and a true browser parity harness. Sequence work as Safety -> Freeze/Performance -> Parity -> Cleanup, with telemetry and tests added first for each change.

**Tech Stack:** SvelteKit, TypeScript, Web Workers, WebGPU/WGSL, three-mesh-bvh, Vitest, Playwright (WebGPU integration lane).

**Primary Inputs:**
- `docs/plans/2026-03-14-webgpu-deep-code-review-action-plan.md`
- `docs/webgpu-migration-analysis.md`
- `docs/plans/2026-03-13-webgpu-migration.md`

**Constraints Applied:** No git worktrees, no commit steps in this plan (per user request).

---

## Scope Matrix (Finding -> Task)

| Finding | Severity | Task(s) |
|---|---|---|
| Byte budget not enforced by points/bytes | P0 | 2, 3 |
| Peak-memory amplification across main/worker/readback | P0 | 4, 7 |
| No `GPUDevice.lost` recovery | P0 | 8 |
| Duplicate traversals and sync prep long task | P1 | 4 |
| Sync pipeline compile + recreate per run | P1 | 5 |
| Solar traversal at night | P1 | 6 |
| Readback + stats hot path on main thread | P1 | 7 |
| Worker bursty protocol + weak cancellation | P1 | 3 |
| Sun model/time sampling mismatch with Python | P1 | 9 |
| Parity test does not exercise live WebGPU runtime | P1 | 10 |
| Stale zHeight docs | P2 | 11 |
| Destroy cleanup misses active abort | P2 | 12 |
| `MAX_GRID_POINTS_GUARD` dead code | P2 | 2 |

---

## Phase 0: Baseline and Guardrails

### Task 1: Add timing + memory telemetry baseline before logic changes

**Files:**
- Modify: `viewer/src/lib/compute/telemetry.ts`
- Modify: `viewer/src/lib/compute/liveUtciAnalysis.ts`
- Modify: `viewer/src/lib/compute/mergeAndBvhWorkerClient.ts`
- Modify: `viewer/src/lib/compute/webgpuUtciPipeline.ts`
- Test: `viewer/tests/compute/live-utci-analysis.test.ts`

**Step 1: Write failing tests for required telemetry fields**
- Add assertions for stage markers: `payload.prepare.start/done`, `worker.merge.start/done`, `pipeline.compile.start/done`, `utci.readback.start/done`.
- Add assertions for `estimatedBytes`, `numPoints`, `numHours` presence.

**Step 2: Run focused tests (expect fail)**
- Run: `cd viewer && npx vitest run tests/compute/live-utci-analysis.test.ts -t telemetry`

**Step 3: Implement minimal telemetry fields/events**
- Add typed payload fields and stable event names used by all later tasks.

**Step 4: Re-run tests (expect pass)**
- Same command as Step 2.

**Step 5: Manual trace validation**
- Run app and capture one performance trace on `debug-webgpu-utci` for BGU and NZ models.
- Save baseline numbers in plan notes before optimization changes.

---

## Phase 1: Safety (P0 first)

### Task 2: Enforce grid-point + byte-budget preflight before worker prep

**Files:**
- Modify: `viewer/src/lib/compute/mergeAndBvhWorkerClient.ts`
- Modify: `viewer/src/lib/compute/liveUtciAnalysis.ts`
- Modify: `viewer/src/routes/debug-webgpu-utci/+page.svelte`
- Test: `viewer/tests/compute/compute-manager.test.ts`
- Test: `viewer/tests/compute/live-utci-analysis.test.ts`

**Step 1: Write failing tests for hard reject behavior**
- Exceeding `MAX_GRID_POINTS_GUARD` returns structured failure.
- Exceeding estimated bytes returns structured failure before payload copy starts.

**Step 2: Run tests (expect fail)**
- Run: `cd viewer && npx vitest run tests/compute/compute-manager.test.ts tests/compute/live-utci-analysis.test.ts -t "guard|budget"`

**Step 3: Implement preflight estimator + guard checks**
- Add estimator components:
  - payload copy bytes
  - worker merged geom + BVH bytes
  - GPU buffer bytes (exposure, results, staging)
  - readback/stat array bytes
- Enforce in `liveUtciAnalysis` before calling worker.
- Enforce `MAX_GRID_POINTS_GUARD` in worker result path (not just constant declaration).

**Step 4: Re-run tests (expect pass)**

**Step 5: Manual validation**
- Trigger dense-grid scenario and verify graceful user-visible failure without freeze/crash.

### Task 3: Convert worker flow to staged protocol with cooperative cancellation

**Files:**
- Modify: `viewer/src/lib/compute/mergeAndBvh.worker.ts`
- Modify: `viewer/src/lib/compute/mergeAndBvhWorkerClient.ts`
- Test: `viewer/tests/compute/meshMerger.test.ts`
- Test: `viewer/tests/compute/compute-manager.test.ts`

**Step 1: Write failing tests for staged progress and abort**
- Progress sequence must be ordered: `prepare -> merge -> grid -> bvh -> transfer`.
- Abort in any stage ends worker quickly with explicit stage label.

**Step 2: Run tests (expect fail)**
- Run: `cd viewer && npx vitest run tests/compute/meshMerger.test.ts tests/compute/compute-manager.test.ts -t "worker|abort|progress"`

**Step 3: Implement staged messages + cancellation polling**
- Add worker message protocol with stage IDs and elapsed ms.
- Add cancellation checks between chunked operations inside heavy stages.

**Step 4: Re-run tests (expect pass)**

**Step 5: Manual validation**
- Start large run, cancel mid-merge and mid-BVH, verify immediate stop and clean UI state.

### Task 4: Remove duplicate scene traversals and reduce payload-copy amplification

**Files:**
- Modify: `viewer/src/routes/debug-webgpu-utci/+page.svelte`
- Modify: `viewer/src/lib/compute/mergeAndBvhWorkerClient.ts`
- Test: `viewer/tests/compute/compute-manager.test.ts`

**Step 1: Write failing tests for single-pass traversal contract**
- Assert one traversal path produces triangle count + payload metadata.
- Assert abort short-circuits traversal chunk loop.

**Step 2: Run tests (expect fail)**
- Run: `cd viewer && npx vitest run tests/compute/compute-manager.test.ts -t "traverse|payload"`

**Step 3: Implement one-pass collector + chunked yielding**
- Move count + extraction into one collector.
- Yield periodically to avoid long tasks; check abort signal on each chunk.

**Step 4: Re-run tests (expect pass)**

**Step 5: Performance check**
- Confirm max startup main-thread task reduces against Phase 0 baseline.

### Task 5: Async pipeline compilation and run-to-run pipeline reuse

**Files:**
- Modify: `viewer/src/lib/compute/webgpuUtciPipeline.ts`
- Modify: `viewer/src/routes/debug-webgpu-utci/+page.svelte`
- Test: `viewer/tests/compute/gpu-pipeline.test.ts`

**Step 1: Write failing tests for async compile + cache reuse**
- Verify `createComputePipelineAsync` path is used.
- Verify repeated runs reuse cached compiled pipelines for same layout.

**Step 2: Run tests (expect fail)**
- Run: `cd viewer && npx vitest run tests/compute/gpu-pipeline.test.ts -t "pipeline|async|cache"`

**Step 3: Implement async compile/warm + cache keying**
- Cache by device + bind group layout signature + shader revision hash.
- Warm once on first run/page load.

**Step 4: Re-run tests (expect pass)**

**Step 5: Manual check**
- Validate first-run compile telemetry, then low/no compile time on second run.

### Task 6: Skip night-hour solar BVH traversal

**Files:**
- Modify: `viewer/src/lib/compute/shaders/exposure_solar.wgsl`
- Modify: `viewer/src/lib/compute/sunpath.ts`
- Modify: `viewer/src/lib/compute/webgpuUtciPipeline.ts`
- Test: `viewer/tests/compute/solar-altitude-packing.test.ts`
- Test: `viewer/tests/compute/exposure-pipeline.test.ts`

**Step 1: Write failing tests for night short-circuit semantics**
- Night hours must emit zero solar exposure.
- No BVH traversal should run for night indices (validated via instrumentation flag/counter).

**Step 2: Run tests (expect fail)**
- Run: `cd viewer && npx vitest run tests/compute/solar-altitude-packing.test.ts tests/compute/exposure-pipeline.test.ts`

**Step 3: Implement altitude/mask guard**
- Add sun-up mask/altitude threshold in packed input.
- Shader early-return sets exposure `0.0` before intersection call.
- Optional: compact dispatch to sun-up hours only if complexity is low.

**Step 4: Re-run tests (expect pass)**

---

## Phase 2: Freeze/Performance Hot Paths

### Task 7: Replace full-buffer readback with hour-slice gather + chunked stats

**Files:**
- Modify: `viewer/src/lib/compute/webgpuUtciPipeline.ts`
- Modify: `viewer/src/lib/compute/liveUtciAnalysis.ts`
- Test: `viewer/tests/compute/mrt-utci-gpu-solarcal.test.ts`
- Test: `viewer/tests/compute/live-utci-analysis.test.ts`

**Step 1: Write failing tests for bounded readback memory behavior**
- Assert hour-slice readback path does not allocate full UTCI cache for analysis mode.
- Assert statistics path yields/chunks for large arrays.

**Step 2: Run tests (expect fail)**
- Run: `cd viewer && npx vitest run tests/compute/mrt-utci-gpu-solarcal.test.ts tests/compute/live-utci-analysis.test.ts -t "readback|stats"`

**Step 3: Implement hour-slice gather path + chunked reducers**
- Add API for readback by hour (or small batch).
- Replace monolithic JS loops with chunked loops using `await` yield boundaries.

**Step 4: Re-run tests (expect pass)**

**Step 5: Performance check**
- Compare post-GPU main-thread long tasks vs baseline.

### Task 8: Add robust `GPUDevice.lost` recovery and retry flow

**Files:**
- Modify: `viewer/src/lib/compute/webgpuUtciPipeline.ts`
- Modify: `viewer/src/lib/compute/compute-manager.ts`
- Modify: `viewer/src/lib/compute/liveUtciAnalysis.ts`
- Test: `viewer/tests/compute/gpu-pipeline.test.ts`

**Step 1: Write failing tests for loss handling contract**
- When device lost is signaled, cached device/pipelines are invalidated.
- Next run attempts re-init once and surfaces deterministic error if re-init fails.

**Step 2: Run tests (expect fail)**
- Run: `cd viewer && npx vitest run tests/compute/gpu-pipeline.test.ts -t "device lost|reinit"`

**Step 3: Implement loss subscription and recovery**
- Subscribe to `device.lost` once per device.
- Invalidate device promise + pipelines + bind groups.
- Emit recovery telemetry and UI state update.

**Step 4: Re-run tests (expect pass)**

---

## Phase 3: Scientific Parity Hardening

### Task 9: Add parity mode for Python-generated sun vectors/timestamps

**Files:**
- Modify: `viewer/src/lib/compute/sunpath.ts`
- Modify: `viewer/src/lib/compute/compute-manager.ts`
- Create: `viewer/tests/fixtures/parity/sun-vectors/*.json`
- Test: `viewer/tests/compute/sunpath.test.ts`
- Test: `viewer/tests/compute/compute-manager.test.ts`

**Step 1: Write failing tests for parity-mode data source selection**
- Given fixture vectors, runtime must bypass NOAA generation and consume fixture vectors/timestamps directly.
- Verify ENU/Z-up fixture vectors are transformed once into chosen world convention.

**Step 2: Run tests (expect fail)**
- Run: `cd viewer && npx vitest run tests/compute/sunpath.test.ts tests/compute/compute-manager.test.ts -t "parity mode|fixture"`

**Step 3: Implement parity-mode loader and transform contract**
- Add explicit mode: `generated` vs `fixture`.
- Document one canonical world frame and transformation rules.

**Step 4: Re-run tests (expect pass)**

### Task 10: Build true browser WebGPU parity harness with intermediate buffer assertions

**Files:**
- Modify: `viewer/tests/compute/webgpu-python-parity.test.ts`
- Modify: `viewer/tests/compute/mrt-utci-gpu-solarcal.test.ts`
- Modify: `viewer/tests/compute/bvhGpuUpload.test.ts`
- Create: `viewer/tests/compute/webgpu-runtime-parity.test.ts`
- Create: `viewer/tests/fixtures/parity/stages/*.json`

**Step 1: Write failing integration tests (browser WebGPU lane)**
- Assert stage-level outputs against fixtures:
  - `sunVectors`
  - `solarExposure`
  - `skyExposure`
  - `mrt`
  - `utci`
- Use tiered tolerances per stage.

**Step 2: Run tests in normal lane (expect skip or fail where unsupported)**
- Run: `cd viewer && npx vitest run tests/compute/webgpu-runtime-parity.test.ts`

**Step 3: Add browser-backed lane (Playwright + WebGPU)**
- Add runner config and CI job restricted to WebGPU-capable host.
- Ensure failures are actionable with per-stage diff summary.

**Step 4: Re-run parity suite in WebGPU lane (expect pass)**

---

## Phase 4: Cleanup and Correctness

### Task 11: Fix stale docs/comments and surface parity assumptions in UI

**Files:**
- Modify: `viewer/src/lib/compute/liveUtciAnalysis.ts`
- Modify: `docs/webgpu-migration-analysis.md`
- Modify: `docs/plans/2026-03-14-webgpu-deep-code-review-action-plan.md`

**Step 1: Write failing docs consistency check (if docs test exists) or add lint script**
- Enforce `zHeight` default references are consistent (`0.9`).

**Step 2: Implement doc/comment corrections**
- Update stale comment in `liveUtciAnalysis.ts`.
- Add parity-mode assumptions and known tolerance notes.

**Step 3: Validate docs checks / run targeted lint**

### Task 12: Ensure `onDestroy` aborts active compute run and detaches handlers

**Files:**
- Modify: `viewer/src/routes/debug-webgpu-utci/+page.svelte`
- Test: `viewer/tests/compute/compute-manager.test.ts`

**Step 1: Write failing lifecycle test**
- Destroying component during active run must trigger abort and prevent stale callbacks/state writes.

**Step 2: Run test (expect fail)**
- Run: `cd viewer && npx vitest run tests/compute/compute-manager.test.ts -t "destroy|abort"`

**Step 3: Implement destroy cleanup**
- Abort `liveAbortController` in `onDestroy`.
- Remove listeners/handlers and clear transient state safely.

**Step 4: Re-run test (expect pass)**

---

## Verification Gates (must pass before completion claim)

> Runtime tuning note: `MAX_GRID_POINTS_GUARD` currently uses `600,000` for large-model viability (including Nes Tziona).

### Gate A: Safety
- `MAX_GRID_POINTS_GUARD` enforced in preflight and worker result path.
- Byte budget failure occurs before expensive allocation.
- No crash on stress model; graceful reject path verified.

### Gate B: Responsiveness
- No main-thread tasks >100ms during compute bootstrap on target scenarios.
- Startup and readback stages show reduction vs Phase 0 baseline.

### Gate C: Parity
- Browser WebGPU runtime parity suite passes with documented tolerances.
- Stage-level diffs are within thresholds and trend stable across reruns.

### Gate D: Recovery
- Simulated `device.lost` path recovers or fails deterministically with explicit UI message.
- Component destroy always aborts active run.

---

## Suggested Execution Order

1. Task 1 (baseline telemetry)
2. Tasks 2, 3, 8 (safety/recovery core)
3. Tasks 4, 5, 6, 7 (freeze/perf)
4. Tasks 9, 10 (parity hardening)
5. Tasks 11, 12 (cleanup/lifecycle)

This ordering reduces crash/freeze risk first, then improves speed, then locks scientific parity.
