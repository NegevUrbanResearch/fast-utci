# Selected-Hour Runtime Quality Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Workflow override:** Do **not** create commits. Do **not** create git worktrees. Preserve unrelated dirty files. This is a quality/refactor plan, so every task must report fresh verification before claiming completion.

**Goal:** Make the selected-hour WebGPU runtime maintainable by locking the current `/` and `/debug` proof surface, giving selected-hour compute/render one canonical contract, and shrinking Svelte route responsibilities without changing user-visible behavior.

**Architecture:** Treat `liveSelectedHourController` / `liveSelectedHourRouteHost` / route projection as the canonical selected-hour runtime spine. Keep `/debug` as the parity/proof lab, but remove duplicated normal f32 selected-hour ownership and make diagnostics request-scoped enough to explain performance and CPU fallback work. Extract only stable route/viewer and render-bridge seams; keep Threlte/WebGPU lifecycle code in scene-facing modules.

**Tech Stack:** SvelteKit, Svelte 5-compatible Svelte components and stores, Threlte/Three.js WebGPURenderer, WebGPU UTCI compute, Vitest, Playwright Chromium with `--enable-unsafe-webgpu`, PowerShell on Windows.

---

## Current State Snapshot

This plan starts after recent commits made `/` and `/debug` much closer.

- Current debug route file: `viewer/src/routes/debug/+page.svelte`
- Old route/file names such as `/debug-webgpu-utci` and `viewer/src/routes/debug-webgpu-utci/+page.svelte` are stale for this checkout unless a compatibility route is intentionally reintroduced.
- Main route selected-hour path: `viewer/src/routes/+page.svelte`
- Canonical selected-hour orchestration stack:
  - `viewer/src/lib/compute/liveSelectedHourController.ts`
  - `viewer/src/lib/compute/liveSelectedHourRouteHost.ts`
  - `viewer/src/lib/compute/liveSelectedHourRouteProjection.ts`
  - `viewer/src/lib/compute/liveUtciSelectedHourSession.ts`
- Render bridge pressure point:
  - `viewer/src/lib/components/scene/UTCIPointCloud.svelte`
  - `viewer/src/lib/services/gpuUtciRenderBridge.ts`
- Current route-level proof specs:
  - `viewer/tests/e2e/main-route-manual-diagnostics.spec.ts`
  - `viewer/tests/e2e/debug-route-baseline-diagnostics.spec.ts`
  - `viewer/tests/e2e/debug-route-shared-host-diagnostics.spec.ts`
  - `viewer/tests/e2e/debug-route-parity-runtime-diagnostics.spec.ts`

Known static-quality caveat:

- `npm run check` has known repo-wide inherited failures. Treat this as a tracked baseline until a dedicated static-cleanup plan pays it down.
- This plan must not add new `svelte-check` failures in touched files. If `npm run check` still fails, capture and compare the filtered errors for plan-touched files.

## Non-Goals

- Do not rewrite the app to Svelte runes as a broad syntax migration.
- Do not remove `dataTexture`, `.bin`, parity, collect, strict-exposure, or Python comparison fallback paths.
- Do not move Threlte scene ownership, renderer initialization, `scene.add`, or GPU disposal into route pages.
- Do not optimize 0.5m performance yet. First make the selected-hour runtime measurable and maintainable.
- Do not fix unrelated repo-wide `svelte-check` debt in this slice.

## Svelte 5 / Runes Position

The repo already uses Svelte 5, but this plan intentionally does **not** start with a broad runes migration.

Runes are useful here as a future readability tool, not as the first quality lever:

- `$state` is for component-local reactive state.
- `$derived` is for pure computed state. It should not mutate state or perform side effects.
- `$effect` is for browser/imperative side effects such as canvas listeners, Threlte/Three integration, diagnostics publication, and other DOM/runtime work.

Current route risk is not primarily "legacy syntax." It is unclear ownership across route pages, selected-hour runtime state, scene copy lifecycle, diagnostics, and fallback/readback paths. A syntax migration before ownership cleanup would make the diff larger without proving runtime behavior.

Preferred order:

1. Move selected-hour runtime contracts and lifecycle rules into tested plain TypeScript helpers.
2. Keep route pages as smaller Svelte composition roots.
3. Only then consider converting selected components or `.svelte.ts` helpers to runes where it clarifies local reactive state and derived values.

Do not convert `$:` blocks mechanically. For each candidate, classify it first:

| Current pattern | Better destination |
| --- | --- |
| Pure derived value | `$derived` or plain helper |
| Browser/Three/WebGPU side effect | `$effect`, component lifecycle, or scene helper |
| Async selected-hour state machine | plain TypeScript controller/session |
| Route diagnostics object shaping | pure diagnostics helper |
| Event listener attach/detach | action/controller helper or `$effect` with cleanup |

## Expected File Shape After This Plan

After this plan, the project should look directionally like this:

```text
viewer/src/routes/
  +page.svelte
    Main product viewer composition. Owns page-level wiring, UI choices, and route-facing diagnostics publication only.
  debug/+page.svelte
    Debug/parity lab composition. Owns parity, collect, strict-exposure, Python .bin comparison, and proof tooling only.

viewer/src/lib/diagnostics/
  mainRouteUtciDiagnostics.ts
  selectedHourRuntimeContract.ts
    Shared selected-hour proof vocabulary. Distinguishes visible render-path transport from range/tooltip/comparison/debug readbacks.

viewer/src/lib/debug/
  debugWebgpuUtciDiagnostics.ts
    Debug-only diagnostics shaping and parity validity. No main-route imports from this module.

viewer/src/lib/compute/
  liveSelectedHourController.ts
  liveSelectedHourRouteHost.ts
  liveSelectedHourRouteProjection.ts
  liveUtciSelectedHourSession.ts
  selectedHourOutputHandle.ts
    Canonical selected-hour runtime path and GPU output ownership.

viewer/src/lib/components/scene/
  UTCIPointCloud.svelte
    Still scene-owned. Delegates compute-buffer copy mechanics but keeps mesh/material lifecycle.
  utciComputeBufferRenderBridge.ts
    Testable compute-buffer-to-render-storage bridge.

viewer/src/lib/components/viewer/
  ViewerShell.svelte
  canvasInteractionController.ts
    Shared listener attach/detach helper. Route-specific callbacks remain in routes for now.
```

The plan should reduce route responsibility without pretending routes become tiny immediately. `debug/+page.svelte` will likely remain large after this slice because it is still the parity/proof workbench.

## Follow-Up Roadmap After This Plan

Do not start these inside this plan. Capture them so the next planning step is intentional:

1. **Static debt cleanup:** make `npm run check` meaningful again. The current known repo-wide `svelte-check` debt should become a dedicated plan with a filtered baseline and no behavior changes.
2. **Debug route decomposition:** split debug parity/collect/comparison panels and helpers out of `viewer/src/routes/debug/+page.svelte` after the selected-hour runtime contract is stable.
3. **Compute folder organization:** reorganize `viewer/src/lib/compute` into clearer subfolders with import-only moves after tests are green.
4. **Runes/local Svelte modernization:** convert smaller, stable components or `.svelte.ts` helpers where `$state`, `$derived`, and `$effect` clarify local state/effects.
5. **Performance pass:** optimize normalized-mode range work, tooltip/fallback CPU data, and then 0.5m cold-start/render-sync behavior with the new diagnostics.

## Future Compute Folder Shape

The current flat `viewer/src/lib/compute` folder mixes WebGPU plumbing, selected-hour orchestration, grid math, UTCI/domain math, diagnostics, and worker/client code. Do not reorganize it in this plan, but the likely target shape is:

```text
viewer/src/lib/compute/
  selected-hour/
    liveSelectedHourController.ts
    liveSelectedHourRouteHost.ts
    liveSelectedHourRouteProjection.ts
    liveUtciSelectedHourSession.ts
    selectedHourOutputHandle.ts

  webgpu/
    webgpuUtciPipeline.ts
    webgpuDeviceLimits.ts
    bvhGpuUpload.ts
    gpu-pipeline.ts
    shaders/

  grid/
    analysisGridFromBounds.ts
    canonicalGrid.ts
    grid-generator.ts

  utci-model/
    utci.ts
    solarcal.ts
    sunpath.ts
    tregenza.ts
    projectWeather.ts
    epw-parser.ts

  diagnostics/
    onDemandDiagnostics.ts
    telemetry.ts

  workers/
    mergeAndBvhWorkerClient.ts
    mergeAndBvh.worker.ts
```

Rules for that future reorg:

- Do it after this plan, not during it.
- Prefer import-only moves first, with no behavior changes.
- Keep selected-hour orchestration separate from raw WebGPU pipeline primitives.
- Keep UTCI/Ladybug/domain math separate from GPU transport and render concerns.
- Preserve parity/import paths with focused tests before and after each move.

## Quality Gates

Stop and report before continuing if any of these happen:

- `/` imports debug/parity/bin helpers or requests `.bin`.
- `/debug` normal f32 reports `selectedHourEngine: "shared-host"` while legacy debug dispatch or scrub counters still increment.
- `/debug` parity mode loses `legacy-debug` / Python `.bin` comparison behavior for August.
- Non-August parity implies Python baseline validity.
- `selectedHourReadbackCount=0` is used as a success claim while range/tooltip/comparison readbacks are not separately accounted for.
- A route publishes `strongVisibleGpuPath: true` without an explicit readback instrumentation state of `instrumented`.
- A selected-hour GPU output handle is disposed before the scene copy completes, fails, or is explicitly superseded.
- A Playwright probe times out without dumping the relevant diagnostics payload.
- A new Playwright wait relies only on the outer test timeout instead of using a short wait that dumps `window.__utciRenderDiagnostics__` or `window.__onDemandPrototypeDiagnostics__`.
- A refactor moves renderer/scene lifecycle code out of scene-facing modules without an explicit review checkpoint.

## File Structure

### Create

- `docs/superpowers/plans/2026-05-11-selected-hour-runtime-quality-baseline.md`
  - This plan.
- `viewer/src/lib/diagnostics/selectedHourRuntimeContract.ts`
  - Pure diagnostics contract helpers shared by main/debug route tests and route publishers. No Svelte imports. No browser access.
- `viewer/tests/diagnostics/selectedHourRuntimeContract.test.ts`
  - Unit tests for selected-hour runtime proof classification, CPU-readback reason accounting, and touched-file `svelte-check` parsing helpers if added here.
- `viewer/src/lib/compute/selectedHourOutputHandle.ts`
  - Explicit selected-hour GPU output handle type and lifecycle helpers. No WebGPU allocation; pure ownership/disposal contract.
- `viewer/tests/compute/selectedHourOutputHandle.test.ts`
  - Unit tests for output handle ownership, disposal idempotence, and request identity.
- `viewer/src/lib/components/scene/utciComputeBufferRenderBridge.ts`
  - Plain TypeScript helper for compute-buffer-to-render-storage sync. It may accept renderer/backend access as injected callbacks, but it must not import Svelte.
- `viewer/tests/scene/utciComputeBufferRenderBridge.test.ts`
  - Unit tests for storage-buffer wait, supersession, failure reasons, and timing output.

### Modify

- `docs/webgpu_strategy_analysis.md`
  - Refresh route names and add a dated note pointing to this quality baseline plan.
- `docs/superpowers/plans/2026-05-10-webgpu-debug-route-shared-viewer-rebuild.md`
  - Mark as superseded or route-name-stale for current execution, with a pointer to this plan.
- `docs/superpowers/plans/2026-05-10-webgpu-debug-shell-shared-cleanup.md`
  - Mark as superseded or route-name-stale for current execution, with a pointer to this plan.
- `viewer/package.json`
  - Add focused verification scripts so current proof commands are executable affordances, not only doc text.
- `viewer/src/lib/compute/gpu-pipeline.ts`
  - Replace or augment `gpuBuffer?: unknown` with the explicit selected-hour output handle type where safe.
- `viewer/src/lib/compute/webgpuUtciPipeline.ts`
  - Return the selected-hour snapshot buffer and byte length without owning route/controller request identity.
- `viewer/src/lib/compute/liveUtciSelectedHourSession.ts`
  - Wrap pipeline snapshot buffers in explicit request-scoped handles and separate visible selected-hour GPU transport from CPU readbacks used for range, tooltip, comparison, or fallback.
- `viewer/src/lib/compute/liveSelectedHourController.ts`
  - Publish request-scoped runtime contract fields, including accepted request id and CPU readback reasons.
- `viewer/src/lib/compute/liveSelectedHourRouteHost.ts`
  - Forward request-scoped contract fields to route state.
- `viewer/src/lib/compute/onDemandDiagnostics.ts`
  - Add or consolidate timing/readback lifecycle fields used by both routes.
- `viewer/src/lib/debug/debugWebgpuUtciDiagnostics.ts`
  - Consume shared selected-hour runtime contract helpers and keep debug-only parity validity separate.
- `viewer/src/routes/+page.svelte`
  - Keep as a composition root. Remove only duplicated selected-hour contract shaping after shared helpers exist.
- `viewer/src/routes/debug/+page.svelte`
  - Keep debug-only parity/collect/comparison tools. Remove or gate duplicated normal f32 selected-hour ownership after contract tests fail.
- `viewer/src/lib/components/scene/UTCIPointCloud.svelte`
  - Delegate compute-buffer render sync to `utciComputeBufferRenderBridge.ts` while preserving scene ownership and diagnostics callbacks.
- `viewer/tests/e2e/main-route-manual-diagnostics.spec.ts`
  - Assert request-scoped runtime contract fields.
- `viewer/tests/e2e/debug-route-shared-host-diagnostics.spec.ts`
  - Assert normal f32 shared-host ownership and zero legacy counters.
- `viewer/tests/e2e/debug-route-parity-runtime-diagnostics.spec.ts`
  - Assert debug-only parity validity still works.
- `viewer/tests/routes/main-route-debug-boundary-source-lock.test.ts`
  - Extend if new helper names could accidentally leak debug-only concepts into `/`.

### Inspect But Do Not Move Without Separate Approval

- `viewer/src/lib/components/scene/Scene.svelte`
- `viewer/src/lib/components/scene/ComparisonRenderer.svelte`
- `viewer/src/lib/components/scene/utciSurfaceSync.ts`
- `viewer/src/lib/components/viewer/ViewerShell.svelte`
- `viewer/src/lib/services/tooltipService.ts`

---

## Task 0: Baseline Record And Current Route Proof

**Files:**
- Inspect: `viewer/src/routes/+page.svelte`
- Inspect: `viewer/src/routes/debug/+page.svelte`
- Inspect: `viewer/tests/e2e/main-route-manual-diagnostics.spec.ts`
- Inspect: `viewer/tests/e2e/debug-route-baseline-diagnostics.spec.ts`
- Inspect: `viewer/tests/e2e/debug-route-shared-host-diagnostics.spec.ts`
- Inspect: `viewer/package.json`

- [ ] **Step 1: Record current git state**

Run from repo root:

```powershell
git status --short
git log --oneline -8
```

Expected:

- Record all dirty files in task notes.
- Do not revert or overwrite unrelated dirty files.
- Recent history should include the `/debug` route rename work.

- [ ] **Step 2: Confirm stale route path is not the active route**

Run from repo root:

```powershell
Test-Path viewer/src/routes/debug/+page.svelte
Test-Path viewer/src/routes/debug-webgpu-utci/+page.svelte
rg -n "debug-webgpu-utci|/debug\?" docs/webgpu_strategy_analysis.md docs/superpowers/plans viewer/tests viewer/src
```

Expected:

- `viewer/src/routes/debug/+page.svelte` exists.
- If `viewer/src/routes/debug-webgpu-utci/+page.svelte` exists as a compatibility or stale file, inspect before editing anything and report what owns the active route.
- Existing docs may still mention `/debug-webgpu-utci`; this is expected and fixed in Task 1.

- [ ] **Step 3: Run focused unit baseline**

Run:

```powershell
cd viewer
npx vitest run tests/compute/live-selected-hour-controller.test.ts tests/compute/live-selected-hour-route-host.test.ts tests/compute/live-selected-hour-route-projection.test.ts tests/compute/live-selected-hour-render-context.test.ts tests/scene/utci-surface-sync.test.ts tests/diagnostics/main-route-utci-diagnostics.test.ts tests/debug/debug-webgpu-utci-diagnostics.test.ts tests/routes/main-route-debug-boundary-source-lock.test.ts
```

Expected: PASS. If it fails, use `superpowers:systematic-debugging` before changing implementation.

- [ ] **Step 4: Run current required WebGPU route probes**

Run:

```powershell
cd viewer
$env:REQUIRE_WEBGPU_ON_DEMAND='1'
npx playwright test tests/e2e/main-route-manual-diagnostics.spec.ts tests/e2e/debug-route-baseline-diagnostics.spec.ts tests/e2e/debug-route-shared-host-diagnostics.spec.ts tests/e2e/debug-route-parity-runtime-diagnostics.spec.ts --project=chromium --workers=1 --reporter=list --timeout=30000
```

Expected:

- PASS.
- `/` proves `compute-buffer-selected-hour`, same-device compute/render, no `.bin` requests.
- `/debug` normal f32 proves shared-host ownership where expected.
- `/debug` parity proves legacy/Python `.bin` comparison where expected.

- [ ] **Step 5: Capture current static-check baseline**

Run:

```powershell
cd viewer
npm run check
```

Expected:

- May FAIL with inherited repo-wide `svelte-check` errors.
- Save the count and any errors that mention plan-touched files.
- Do not start fixing unrelated static debt.

### Task 0 Evidence Note

Captured during the 2026-05-11 current run:

- `git status --short`: only `?? docs/superpowers/plans/2026-05-11-selected-hour-runtime-quality-baseline.md`.
- Recent commit anchor: `60ca80c chore(debug): rename parity route to /debug`.
- Route proof: `viewer/src/routes/debug/+page.svelte` returned `True`; `viewer/src/routes/debug-webgpu-utci/+page.svelte` returned `False`. Old route mentions remain in historical docs/plans; current tests use `/debug`.
- Focused Vitest baseline: 8 files passed, 82 tests passed.
- WebGPU Playwright baseline with `REQUIRE_WEBGPU_ON_DEMAND=1`: 12 passed.
- `npm run check` baseline: exit 1; `svelte-check` found 147 errors and 4 warnings in 33 files. This remains inherited baseline debt, not a pass.
- Plan-relevant static mentions: existing errors in `viewer/src/routes/debug/+page.svelte`; unused exported `model` warning in `viewer/src/lib/components/scene/UTCIPointCloud.svelte`; optional `supportsMrtComponentDiagnostics` call errors in `viewer/tests/compute/webgpuUtciPipeline.behavior.test.ts`.

---

## Task 1: Refresh Canonical Docs And Verification Scripts

**Files:**
- Modify: `docs/webgpu_strategy_analysis.md`
- Modify: `docs/superpowers/plans/2026-05-10-webgpu-debug-route-shared-viewer-rebuild.md`
- Modify: `docs/superpowers/plans/2026-05-10-webgpu-debug-shell-shared-cleanup.md`
- Modify: `viewer/package.json`

- [ ] **Step 1: Add focused verification scripts**

Edit `viewer/package.json` scripts to add these entries:

```json
"test:quality:selected-hour": "vitest run tests/compute/live-selected-hour-controller.test.ts tests/compute/live-selected-hour-route-host.test.ts tests/compute/live-selected-hour-route-projection.test.ts tests/compute/live-selected-hour-render-context.test.ts tests/scene/utci-surface-sync.test.ts tests/diagnostics/main-route-utci-diagnostics.test.ts tests/debug/debug-webgpu-utci-diagnostics.test.ts tests/routes/main-route-debug-boundary-source-lock.test.ts",
"test:e2e:selected-hour": "cross-env REQUIRE_WEBGPU_ON_DEMAND=1 playwright test tests/e2e/main-route-manual-diagnostics.spec.ts tests/e2e/debug-route-baseline-diagnostics.spec.ts tests/e2e/debug-route-shared-host-diagnostics.spec.ts tests/e2e/debug-route-parity-runtime-diagnostics.spec.ts --project=chromium --workers=1 --reporter=list --timeout=30000"
```

If the surrounding JSON comma placement differs, preserve valid JSON and keep existing scripts unchanged.

- [ ] **Step 2: Verify scripts parse**

Run:

```powershell
cd viewer
node -e "JSON.parse(require('fs').readFileSync('package.json','utf8')); console.log('package.json ok')"
```

Expected: prints `package.json ok`.

- [ ] **Step 3: Add a current-route note to the strategy doc**

In `docs/webgpu_strategy_analysis.md`, add a short dated note near the top:

```markdown
> **2026-05-11 route-name note:** The active debug route in this checkout is now `/debug` at `viewer/src/routes/debug/+page.svelte`. Older references to `/debug-webgpu-utci` describe the same debug/parity role before the route rename and should not be copied into new execution plans.
```

Also update the `Recommended Next Step` section to point to:

```markdown
Current planning artifact: [2026-05-11 selected-hour runtime quality baseline](superpowers/plans/2026-05-11-selected-hour-runtime-quality-baseline.md).
```

Do not rewrite historical timing URLs or evidence tables in this task. They were captured before the route rename and should remain historical evidence with the new note above them.

- [ ] **Step 4: Mark stale 2026-05-10 plans as superseded for execution**

At the top of both old plan files, below their existing header block, add:

```markdown
> **Superseded for new execution as of 2026-05-11:** This plan uses the old `/debug-webgpu-utci` route name. For current quality/refactor execution, use `docs/superpowers/plans/2026-05-11-selected-hour-runtime-quality-baseline.md` and treat this file as historical context only.
```

Do not rewrite their task bodies.

- [ ] **Step 5: Run doc/script verification**

Run:

```powershell
cd viewer
npm run test:quality:selected-hour
```

Expected: PASS. If it fails, report the exact failure before touching production code.

---

## Task 2: Add A Shared Selected-Hour Runtime Contract

**Files:**
- Create: `viewer/src/lib/diagnostics/selectedHourRuntimeContract.ts`
- Create: `viewer/tests/diagnostics/selectedHourRuntimeContract.test.ts`
- Modify: `viewer/src/lib/diagnostics/mainRouteUtciDiagnostics.ts`
- Modify: `viewer/src/lib/debug/debugWebgpuUtciDiagnostics.ts`
- Modify: `viewer/tests/diagnostics/main-route-utci-diagnostics.test.ts`
- Modify: `viewer/tests/debug/debug-webgpu-utci-diagnostics.test.ts`

- [ ] **Step 1: Write contract helper tests**

Create `viewer/tests/diagnostics/selectedHourRuntimeContract.test.ts`:

```ts
import { describe, expect, it } from 'vitest';
import {
	buildSelectedHourRuntimeContract,
	type SelectedHourReadbackReason
} from '$lib/diagnostics/selectedHourRuntimeContract';

describe('selectedHourRuntimeContract', () => {
	it('classifies a strong compute-buffer visible path', () => {
		const contract = buildSelectedHourRuntimeContract({
			route: 'main',
			selectedHourEngine: 'shared-host',
			renderTransport: 'compute-buffer-selected-hour',
			utciSurfaceSource: 'compute-buffer-selected-hour',
			sameDeviceForComputeAndRender: true,
			dataTextureBuildCount: 0,
			visibleSelectedHourReadbackCount: 0,
			readbackInstrumentation: 'instrumented',
			legacySelectedHourDispatchCount: 0,
			legacyScrubScheduleCount: 0,
			requestId: 42,
			selectionKey: 'Ben-Gurion|7|12',
			sceneSelectionKey: 'Ben-Gurion|7|12',
			readbackReasons: []
		});

		expect(contract.strongVisibleGpuPath).toBe(true);
		expect(contract.visibleRenderPathAvoidsCpuReadback).toBe(true);
		expect(contract.hasLegacyDebugOverlap).toBe(false);
		expect(contract.acceptedRequestId).toBe(42);
	});

	it('keeps non-visible CPU readbacks separate from visible transport proof', () => {
		const reasons: SelectedHourReadbackReason[] = ['range', 'tooltip'];
		const contract = buildSelectedHourRuntimeContract({
			route: 'debug',
			selectedHourEngine: 'shared-host',
			renderTransport: 'compute-buffer-selected-hour',
			utciSurfaceSource: 'compute-buffer-selected-hour',
			sameDeviceForComputeAndRender: true,
			dataTextureBuildCount: 0,
			visibleSelectedHourReadbackCount: 0,
			readbackInstrumentation: 'instrumented',
			legacySelectedHourDispatchCount: 0,
			legacyScrubScheduleCount: 0,
			requestId: 7,
			selectionKey: 'Ben-Gurion|7|12',
			sceneSelectionKey: 'Ben-Gurion|7|12',
			readbackReasons: reasons
		});

		expect(contract.strongVisibleGpuPath).toBe(true);
		expect(contract.visibleRenderPathAvoidsCpuReadback).toBe(true);
		expect(contract.readbackReasons).toEqual(['range', 'tooltip']);
		expect(contract.totalSelectedHourReadbackReasonCount).toBe(2);
	});

	it('does not allow strong GPU proof when readback instrumentation is missing', () => {
		const contract = buildSelectedHourRuntimeContract({
			route: 'main',
			selectedHourEngine: 'shared-host',
			renderTransport: 'compute-buffer-selected-hour',
			utciSurfaceSource: 'compute-buffer-selected-hour',
			sameDeviceForComputeAndRender: true,
			dataTextureBuildCount: 0,
			visibleSelectedHourReadbackCount: 0,
			readbackInstrumentation: 'not-instrumented',
			legacySelectedHourDispatchCount: 0,
			legacyScrubScheduleCount: 0,
			requestId: 8,
			selectionKey: 'Ben-Gurion|7|12',
			sceneSelectionKey: 'Ben-Gurion|7|12',
			readbackReasons: []
		});

		expect(contract.visibleRenderPathAvoidsCpuReadback).toBe(false);
		expect(contract.strongVisibleGpuPath).toBe(false);
	});

	it('flags legacy debug overlap when shared-host claims coexist with legacy counters', () => {
		const contract = buildSelectedHourRuntimeContract({
			route: 'debug',
			selectedHourEngine: 'shared-host',
			renderTransport: 'compute-buffer-selected-hour',
			utciSurfaceSource: 'compute-buffer-selected-hour',
			sameDeviceForComputeAndRender: true,
			dataTextureBuildCount: 0,
			visibleSelectedHourReadbackCount: 0,
			readbackInstrumentation: 'instrumented',
			legacySelectedHourDispatchCount: 1,
			legacyScrubScheduleCount: 0,
			requestId: 9,
			selectionKey: 'Ben-Gurion|7|12',
			sceneSelectionKey: 'Ben-Gurion|7|12',
			readbackReasons: []
		});

		expect(contract.strongVisibleGpuPath).toBe(false);
		expect(contract.hasLegacyDebugOverlap).toBe(true);
	});
});
```

- [ ] **Step 2: Run the new test to verify it fails**

Run:

```powershell
cd viewer
npx vitest run tests/diagnostics/selectedHourRuntimeContract.test.ts
```

Expected: FAIL because `selectedHourRuntimeContract.ts` does not exist.

- [ ] **Step 3: Implement the contract helper**

Create `viewer/src/lib/diagnostics/selectedHourRuntimeContract.ts`:

```ts
export type SelectedHourRouteRole = 'main' | 'debug';
export type SelectedHourEngine = 'legacy-debug' | 'shared-host';
export type SelectedHourRenderTransport =
	| 'none'
	| 'cpu-uploaded-selected-hour'
	| 'compute-buffer-selected-hour';
export type SelectedHourReadbackReason = 'visible-fallback' | 'range' | 'tooltip' | 'comparison' | 'debug';
export type SelectedHourReadbackInstrumentation = 'instrumented' | 'not-instrumented';

export interface SelectedHourRuntimeContractInputs {
	route: SelectedHourRouteRole;
	selectedHourEngine: SelectedHourEngine;
	renderTransport?: SelectedHourRenderTransport;
	utciSurfaceSource?: SelectedHourRenderTransport;
	sameDeviceForComputeAndRender?: boolean;
	dataTextureBuildCount?: number;
	visibleSelectedHourReadbackCount?: number;
	readbackInstrumentation: SelectedHourReadbackInstrumentation;
	legacySelectedHourDispatchCount?: number;
	legacyScrubScheduleCount?: number;
	requestId?: number;
	selectionKey?: string;
	sceneSelectionKey?: string;
	readbackReasons?: readonly SelectedHourReadbackReason[];
}

export interface SelectedHourRuntimeContract {
	route: SelectedHourRouteRole;
	selectedHourEngine: SelectedHourEngine;
	renderTransport: SelectedHourRenderTransport;
	utciSurfaceSource: SelectedHourRenderTransport;
	sameDeviceForComputeAndRender: boolean;
	dataTextureBuildCount: number;
	visibleSelectedHourReadbackCount: number;
	readbackInstrumentation: SelectedHourReadbackInstrumentation;
	legacySelectedHourDispatchCount: number;
	legacyScrubScheduleCount: number;
	acceptedRequestId?: number;
	selectionKey?: string;
	sceneSelectionKey?: string;
	readbackReasons: SelectedHourReadbackReason[];
	totalSelectedHourReadbackReasonCount: number;
	hasLegacyDebugOverlap: boolean;
	selectionMatchesScene: boolean;
	strongVisibleGpuPath: boolean;
	visibleRenderPathAvoidsCpuReadback: boolean;
}

export function buildSelectedHourRuntimeContract(
	inputs: SelectedHourRuntimeContractInputs
): SelectedHourRuntimeContract {
	const renderTransport = inputs.renderTransport ?? 'none';
	const utciSurfaceSource = inputs.utciSurfaceSource ?? 'none';
	const dataTextureBuildCount = inputs.dataTextureBuildCount ?? 0;
	const visibleSelectedHourReadbackCount = inputs.visibleSelectedHourReadbackCount ?? 0;
	const legacySelectedHourDispatchCount = inputs.legacySelectedHourDispatchCount ?? 0;
	const legacyScrubScheduleCount = inputs.legacyScrubScheduleCount ?? 0;
	const hasLegacyDebugOverlap =
		inputs.selectedHourEngine === 'shared-host' &&
		(legacySelectedHourDispatchCount > 0 || legacyScrubScheduleCount > 0);
	const selectionMatchesScene =
		inputs.selectionKey !== undefined &&
		inputs.sceneSelectionKey !== undefined &&
		inputs.selectionKey === inputs.sceneSelectionKey;
	const visibleRenderPathAvoidsCpuReadback =
		inputs.readbackInstrumentation === 'instrumented' &&
		renderTransport === 'compute-buffer-selected-hour' &&
		utciSurfaceSource === 'compute-buffer-selected-hour' &&
		visibleSelectedHourReadbackCount === 0 &&
		dataTextureBuildCount === 0;
	const strongVisibleGpuPath =
		inputs.selectedHourEngine === 'shared-host' &&
		visibleRenderPathAvoidsCpuReadback &&
		inputs.sameDeviceForComputeAndRender === true &&
		selectionMatchesScene &&
		!hasLegacyDebugOverlap;

	return {
		route: inputs.route,
		selectedHourEngine: inputs.selectedHourEngine,
		renderTransport,
		utciSurfaceSource,
		sameDeviceForComputeAndRender: inputs.sameDeviceForComputeAndRender === true,
		dataTextureBuildCount,
		visibleSelectedHourReadbackCount,
		readbackInstrumentation: inputs.readbackInstrumentation,
		legacySelectedHourDispatchCount,
		legacyScrubScheduleCount,
		acceptedRequestId: inputs.requestId,
		selectionKey: inputs.selectionKey,
		sceneSelectionKey: inputs.sceneSelectionKey,
		readbackReasons: [...(inputs.readbackReasons ?? [])],
		totalSelectedHourReadbackReasonCount: inputs.readbackReasons?.length ?? 0,
		hasLegacyDebugOverlap,
		selectionMatchesScene,
		strongVisibleGpuPath,
		visibleRenderPathAvoidsCpuReadback
	};
}
```

- [ ] **Step 4: Run contract tests**

Run:

```powershell
cd viewer
npx vitest run tests/diagnostics/selectedHourRuntimeContract.test.ts
```

Expected: PASS.

- [ ] **Step 5: Wire the contract into existing diagnostics helpers**

In `viewer/src/lib/diagnostics/mainRouteUtciDiagnostics.ts` and `viewer/src/lib/debug/debugWebgpuUtciDiagnostics.ts`, import `buildSelectedHourRuntimeContract` and expose the returned contract object under a stable diagnostics key:

```ts
selectedHourRuntimeContract: buildSelectedHourRuntimeContract({
	route: 'main',
	selectedHourEngine: 'shared-host',
	renderTransport: inputs.baseRenderTransport,
	utciSurfaceSource: inputs.utciSurfaceSource,
	sameDeviceForComputeAndRender: inputs.baseSameDeviceForComputeAndRender,
	dataTextureBuildCount: inputs.dataTextureBuildCount,
	visibleSelectedHourReadbackCount: inputs.visibleSelectedHourReadbackCount,
	readbackInstrumentation: inputs.visibleSelectedHourReadbackInstrumentation,
	requestId: inputs.baseRequestId,
	selectionKey: inputs.baseSelectionKey,
	sceneSelectionKey: inputs.baseSceneSelectionKey,
	readbackReasons: inputs.selectedHourReadbackReasons ?? []
})
```

For debug diagnostics use:

```ts
route: 'debug',
selectedHourEngine: inputs.selectedHourEngine ?? 'legacy-debug',
legacySelectedHourDispatchCount: inputs.legacySelectedHourDispatchCount,
legacyScrubScheduleCount: inputs.legacyScrubScheduleCount
```

If exact input names differ, map from the existing diagnostic fields without changing their current public names.

- [ ] **Step 6: Add helper assertions to existing diagnostics tests**

In `viewer/tests/diagnostics/main-route-utci-diagnostics.test.ts`, add an assertion to the existing strong-path test:

```ts
expect(diagnostics.selectedHourRuntimeContract.strongVisibleGpuPath).toBe(true);
expect(diagnostics.selectedHourRuntimeContract.visibleRenderPathAvoidsCpuReadback).toBe(true);
```

If `MainRouteUtciDiagnosticsInputs` does not yet expose an explicit visible readback count, add `visibleSelectedHourReadbackInstrumentation: 'not-instrumented'` first and assert `strongVisibleGpuPath === false`. Do not default an unknown readback count to zero.

In `viewer/tests/debug/debug-webgpu-utci-diagnostics.test.ts`, add one shared-host assertion and one legacy assertion:

```ts
expect(state.selectedHourRuntimeContract.selectedHourEngine).toBe('shared-host');
expect(state.selectedHourRuntimeContract.hasLegacyDebugOverlap).toBe(false);
```

```ts
expect(state.selectedHourRuntimeContract.selectedHourEngine).toBe('legacy-debug');
expect(state.selectedHourRuntimeContract.strongVisibleGpuPath).toBe(false);
```

- [ ] **Step 7: Run diagnostics unit tests**

Run:

```powershell
cd viewer
npx vitest run tests/diagnostics/selectedHourRuntimeContract.test.ts tests/diagnostics/main-route-utci-diagnostics.test.ts tests/debug/debug-webgpu-utci-diagnostics.test.ts
```

Expected: PASS.

---

## Task 3: Formalize Selected-Hour GPU Output Ownership

**Files:**
- Create: `viewer/src/lib/compute/selectedHourOutputHandle.ts`
- Create: `viewer/tests/compute/selectedHourOutputHandle.test.ts`
- Modify: `viewer/src/lib/compute/gpu-pipeline.ts`
- Modify: `viewer/src/lib/compute/webgpuUtciPipeline.ts`
- Modify: `viewer/src/lib/compute/liveUtciSelectedHourSession.ts`
- Modify: `viewer/tests/compute/live-selected-hour-session.test.ts`

- [ ] **Step 1: Write output handle tests**

Create `viewer/tests/compute/selectedHourOutputHandle.test.ts`:

```ts
import { describe, expect, it, vi } from 'vitest';
import { createSelectedHourOutputHandle, disposeSelectedHourOutputHandle } from '$lib/compute/selectedHourOutputHandle';

describe('selectedHourOutputHandle', () => {
	it('disposes the owned GPU buffer once', () => {
		const destroy = vi.fn();
		const handle = createSelectedHourOutputHandle({
			buffer: { destroy } as unknown as GPUBuffer,
			byteLength: 16,
			requestId: 5,
			timeIndex: 12,
			source: 'webgpu-on-demand-snapshot'
		});

		disposeSelectedHourOutputHandle(handle);
		disposeSelectedHourOutputHandle(handle);

		expect(destroy).toHaveBeenCalledTimes(1);
		expect(handle.disposed).toBe(true);
	});

	it('keeps request identity with the buffer handle', () => {
		const handle = createSelectedHourOutputHandle({
			buffer: { destroy: vi.fn() } as unknown as GPUBuffer,
			byteLength: 32,
			requestId: 17,
			timeIndex: 90,
			source: 'webgpu-on-demand-snapshot'
		});

		expect(handle.requestId).toBe(17);
		expect(handle.timeIndex).toBe(90);
		expect(handle.byteLength).toBe(32);
		expect(handle.source).toBe('webgpu-on-demand-snapshot');
	});
});
```

- [ ] **Step 2: Run the new test to verify it fails**

Run:

```powershell
cd viewer
npx vitest run tests/compute/selectedHourOutputHandle.test.ts
```

Expected: FAIL because the module does not exist.

- [ ] **Step 3: Implement the output handle helper**

Create `viewer/src/lib/compute/selectedHourOutputHandle.ts`:

```ts
export type SelectedHourOutputSource = 'webgpu-on-demand-snapshot';

export interface SelectedHourOutputHandle {
	buffer: GPUBuffer;
	byteLength: number;
	requestId?: number;
	timeIndex?: number;
	source: SelectedHourOutputSource;
	disposed: boolean;
	dispose(): void;
}

export interface SelectedHourOutputHandleParams {
	buffer: GPUBuffer;
	byteLength: number;
	requestId?: number;
	timeIndex?: number;
	source: SelectedHourOutputSource;
}

export function createSelectedHourOutputHandle(
	params: SelectedHourOutputHandleParams
): SelectedHourOutputHandle {
	const handle: SelectedHourOutputHandle = {
		buffer: params.buffer,
		byteLength: params.byteLength,
		requestId: params.requestId,
		timeIndex: params.timeIndex,
		source: params.source,
		disposed: false,
		dispose() {
			if (handle.disposed) return;
			handle.buffer.destroy();
			handle.disposed = true;
		}
	};
	return handle;
}

export function disposeSelectedHourOutputHandle(handle: SelectedHourOutputHandle | null | undefined): void {
	handle?.dispose();
}
```

- [ ] **Step 4: Run output handle tests**

Run:

```powershell
cd viewer
npx vitest run tests/compute/selectedHourOutputHandle.test.ts
```

Expected: PASS.

- [ ] **Step 5: Replace unknown GPU buffer typing in pipeline contract**

In `viewer/src/lib/compute/gpu-pipeline.ts`, import and use the handle:

```ts
import type { SelectedHourOutputHandle } from './selectedHourOutputHandle';
```

Change selected-hour output result typing from:

```ts
gpuBuffer?: unknown;
```

to:

```ts
gpuOutputHandle?: SelectedHourOutputHandle;
```

If existing callers still need `gpuBuffer` during migration, keep it temporarily but mark the handle as canonical:

```ts
gpuBuffer?: unknown;
gpuOutputHandle?: SelectedHourOutputHandle;
```

Do not remove compatibility fields until all current tests pass.

- [ ] **Step 6: Wrap WebGPU snapshot output in the handle**

In `viewer/src/lib/compute/webgpuUtciPipeline.ts`, keep route/controller request identity out of the pipeline. If the pipeline creates a handle, create it without `requestId` or `timeIndex`:

```ts
import { createSelectedHourOutputHandle } from './selectedHourOutputHandle';
```

```ts
gpuOutputHandle: createSelectedHourOutputHandle({
	buffer: snapshotBuffer,
	byteLength: outputBytes,
	source: 'webgpu-on-demand-snapshot'
})
```

If the pipeline continues returning only `gpuBuffer`, that is acceptable for this task. The session wraps it in Step 7.

- [ ] **Step 7: Attach request identity in selected-hour session**

In `viewer/src/lib/compute/liveUtciSelectedHourSession.ts`, after a selected-hour request receives pipeline output, create or annotate the handle in session-owned code:

```ts
const gpuOutputHandle =
	output.gpuOutputHandle ??
	createSelectedHourOutputHandle({
		buffer: output.gpuBuffer as GPUBuffer,
		byteLength: output.outputBytes,
		source: 'webgpu-on-demand-snapshot'
	});

gpuOutputHandle.requestId = requestId;
gpuOutputHandle.timeIndex = timeIndex;
```

Do not push `requestId` down into `webgpuUtciPipeline.ts`.

- [ ] **Step 8: Add lifecycle tests before replacing direct disposal**

In `viewer/tests/compute/live-selected-hour-session.test.ts`, add or extend tests to prove an accepted handle remains alive until the scene copy is completed, failed, or superseded:

```ts
expect(acceptedOutput.gpuOutputHandle?.disposed).toBe(false);
```

Then prove a superseded output is disposed only after supersession is recorded:

```ts
expect(supersededOutput.gpuOutputHandle?.disposed).toBe(true);
```

If current tests cannot observe scene-copy completion directly, expose a narrow session-level callback or diagnostic event for `selected-hour-output-disposed` and assert the event order.

- [ ] **Step 9: Dispose through the handle in selected-hour session**

In `viewer/src/lib/compute/liveUtciSelectedHourSession.ts`, replace direct `GPUBuffer.destroy()` calls for accepted/superseded selected-hour outputs with:

```ts
disposeSelectedHourOutputHandle(output.gpuOutputHandle);
```

Allowed disposal moments:

- stale/superseded before scene copy starts
- scene copy completed
- scene copy failed
- session/controller disposed

Keep any old `gpuBuffer` disposal only as a compatibility fallback:

```ts
if (!output.gpuOutputHandle && output.gpuBuffer && typeof (output.gpuBuffer as GPUBuffer).destroy === 'function') {
	(output.gpuBuffer as GPUBuffer).destroy();
}
```

- [ ] **Step 10: Run focused compute tests**

Run:

```powershell
cd viewer
npx vitest run tests/compute/selectedHourOutputHandle.test.ts tests/compute/live-selected-hour-session.test.ts tests/compute/live-selected-hour-controller.test.ts tests/compute/live-selected-hour-route-host.test.ts
```

Expected: PASS.

---

## Task 4: Add Readback Reason Accounting Data Model

**Files:**
- Modify: `viewer/src/lib/compute/onDemandDiagnostics.ts`
- Modify: `viewer/tests/compute/onDemandDiagnostics.test.ts`

- [ ] **Step 1: Add diagnostics tests for readback reasons**

In `viewer/tests/compute/onDemandDiagnostics.test.ts`, add:

```ts
it('records selected-hour CPU readback reasons separately from visible readback count', () => {
	const diagnostics = createEmptyOnDemandDiagnostics();
	const next = recordSelectedHourReadbackReason(
		recordSelectedHourReadbackReason(diagnostics, 'range'),
		'tooltip'
	);

	expect(next.selectedHourReadbackReasons).toEqual(['range', 'tooltip']);
	expect(next.selectedHourReadbackReasonCounts).toEqual({
		range: 1,
		tooltip: 1
	});
});
```

- [ ] **Step 2: Run diagnostics test to verify it fails**

Run:

```powershell
cd viewer
npx vitest run tests/compute/onDemandDiagnostics.test.ts
```

Expected: FAIL because `recordSelectedHourReadbackReason` does not exist.

- [ ] **Step 3: Implement reason accounting**

In `viewer/src/lib/compute/onDemandDiagnostics.ts`, add:

```ts
import type { SelectedHourReadbackReason } from '$lib/diagnostics/selectedHourRuntimeContract';

export function recordSelectedHourReadbackReason(
	diagnostics: OnDemandRuntimeDiagnostics,
	reason: SelectedHourReadbackReason
): OnDemandRuntimeDiagnostics {
	const existingReasons = diagnostics.selectedHourReadbackReasons ?? [];
	const existingCounts = diagnostics.selectedHourReadbackReasonCounts ?? {};
	return {
		...diagnostics,
		selectedHourReadbackReasons: [...existingReasons, reason],
		selectedHourReadbackReasonCounts: {
			...existingCounts,
			[reason]: (existingCounts[reason] ?? 0) + 1
		}
	};
}
```

Add these fields to the diagnostics type:

```ts
selectedHourReadbackReasons?: SelectedHourReadbackReason[];
selectedHourReadbackReasonCounts?: Partial<Record<SelectedHourReadbackReason, number>>;
```

- [ ] **Step 4: Run diagnostics tests**

Run:

```powershell
cd viewer
npx vitest run tests/compute/onDemandDiagnostics.test.ts tests/diagnostics/selectedHourRuntimeContract.test.ts
```

Expected: PASS.

---

## Task 4A: Tag Session Readback Sites

**Files:**
- Modify: `viewer/src/lib/compute/liveUtciSelectedHourSession.ts`
- Modify: `viewer/tests/compute/live-selected-hour-session.test.ts`

- [ ] **Step 1: Add session tests for readback reasons**

In `viewer/tests/compute/live-selected-hour-session.test.ts`, add or extend focused tests to assert that selected-day range work records `range`, comparison work records `comparison`, and fallback visible CPU use records `visible-fallback`.

Use assertions shaped like:

```ts
expect(result.diagnostics.selectedHourReadbackReasons).toContain('range');
expect(result.diagnostics.selectedHourReadbackReasonCounts?.range).toBeGreaterThan(0);
```

- [ ] **Step 2: Run session tests to verify they fail**

Run:

```powershell
cd viewer
npx vitest run tests/compute/live-selected-hour-session.test.ts
```

Expected: FAIL because session readback sites are not tagged yet.

- [ ] **Step 3: Tag known CPU readback sites**

In `viewer/src/lib/compute/liveUtciSelectedHourSession.ts`, tag readbacks by reason:

- `resolveSelectedDayUtciRange(...)` -> `'range'`
- tooltip/fallback selected-hour CPU values -> `'tooltip'` or `'visible-fallback'`
- Python/bin/debug comparison readback paths -> `'comparison'`
- diagnostic-only explicit readback paths -> `'debug'`

Use the diagnostics helper instead of incrementing only a generic count.

- [ ] **Step 4: Run session and diagnostics tests**

Run:

```powershell
cd viewer
npx vitest run tests/compute/live-selected-hour-session.test.ts tests/compute/onDemandDiagnostics.test.ts
```

Expected: PASS.

---

## Task 4B: Propagate Accepted Visible Request Timing

**Files:**
- Modify: `viewer/src/lib/compute/liveUtciSelectedHourSession.ts`
- Modify: `viewer/src/lib/compute/liveSelectedHourController.ts`
- Modify: `viewer/src/lib/compute/liveSelectedHourRouteHost.ts`
- Modify: `viewer/tests/compute/live-selected-hour-session.test.ts`
- Modify: `viewer/tests/compute/live-selected-hour-controller.test.ts`
- Modify: `viewer/tests/compute/live-selected-hour-route-host.test.ts`

- [ ] **Step 1: Add accepted visible request tests**

Add focused assertions to the existing controller and route-host tests:

```ts
expect(state.acceptedRequestId).toBe(expectedRequestId);
expect(state.acceptedSelectionKey).toBe(expectedSelectionKey);
expect(state.acceptedVisibleAtMs).toEqual(expect.any(Number));
```

Also assert stale requests do not overwrite the accepted visible request id.

- [ ] **Step 2: Run focused tests to verify they fail**

Run:

```powershell
cd viewer
npx vitest run tests/compute/live-selected-hour-controller.test.ts tests/compute/live-selected-hour-route-host.test.ts
```

Expected: FAIL because accepted visible request timing is not propagated yet.

- [ ] **Step 3: Propagate accepted request id into timings**

In session/controller/route-host state, include:

```ts
acceptedRequestId: requestId,
acceptedSelectionKey: selectionKey,
acceptedVisibleAtMs: performance.now()
```

Only set these when the request is the accepted visible request, not for stale/superseded results.

- [ ] **Step 4: Run focused diagnostics and controller tests**

Run:

```powershell
cd viewer
npx vitest run tests/compute/onDemandDiagnostics.test.ts tests/compute/live-selected-hour-session.test.ts tests/compute/live-selected-hour-controller.test.ts tests/compute/live-selected-hour-route-host.test.ts tests/diagnostics/selectedHourRuntimeContract.test.ts
```

Expected: PASS.

---

## Task 5: Extract The Compute-Buffer Render Bridge From `UTCIPointCloud`

**Files:**
- Create: `viewer/src/lib/components/scene/utciComputeBufferRenderBridge.ts`
- Create: `viewer/tests/scene/utciComputeBufferRenderBridge.test.ts`
- Modify: `viewer/src/lib/components/scene/UTCIPointCloud.svelte`
- Modify: `viewer/tests/scene/utci-surface-sync.test.ts`

- [ ] **Step 1: Write render bridge tests**

Create `viewer/tests/scene/utciComputeBufferRenderBridge.test.ts`:

```ts
import { describe, expect, it, vi } from 'vitest';
import { waitForRenderStorageBuffer, copyComputeBufferToRenderStorage } from '$lib/components/scene/utciComputeBufferRenderBridge';

describe('utciComputeBufferRenderBridge', () => {
	it('waits for a render storage buffer and reports timing', async () => {
		const targetBuffer = { size: 64 } as GPUBuffer;
		const result = await waitForRenderStorageBuffer({
			deadlineMs: 100,
			now: (() => {
				let time = 0;
				return () => (time += 1);
			})(),
			waitForNextFrame: async () => undefined,
			isSuperseded: () => false,
			readStorageBuffer: () => ({ device: {} as GPUDevice, targetBuffer })
		});

		expect(result.targetBuffer).toBe(targetBuffer);
		expect(result.waitMs).toBeGreaterThanOrEqual(0);
	});

	it('fails when a copy is superseded before storage initializes', async () => {
		await expect(
			waitForRenderStorageBuffer({
				deadlineMs: 100,
				now: (() => {
					let time = 0;
					return () => (time += 1);
				})(),
				waitForNextFrame: async () => undefined,
				isSuperseded: () => true,
				readStorageBuffer: () => null
			})
		).rejects.toThrow('superseded');
	});

	it('copies compute output into render-owned storage', async () => {
		const copyBufferToBuffer = vi.fn();
		const finish = vi.fn(() => 'commands' as unknown as GPUCommandBuffer);
		const submit = vi.fn();
		const onSubmittedWorkDone = vi.fn(async () => undefined);

		await copyComputeBufferToRenderStorage({
			device: {
				createCommandEncoder: () => ({ copyBufferToBuffer, finish })
			} as unknown as GPUDevice,
			queue: { submit, onSubmittedWorkDone } as unknown as GPUQueue,
			sourceBuffer: {} as GPUBuffer,
			targetBuffer: { size: 64 } as GPUBuffer,
			byteLength: 32,
			now: performance.now.bind(performance)
		});

		expect(copyBufferToBuffer).toHaveBeenCalledWith(expect.anything(), 0, expect.anything(), 0, 32);
		expect(submit).toHaveBeenCalledWith(['commands']);
		expect(onSubmittedWorkDone).toHaveBeenCalled();
	});

	it('reports supersession after queue drain before publication', async () => {
		const copyBufferToBuffer = vi.fn();
		const finish = vi.fn(() => 'commands' as unknown as GPUCommandBuffer);
		const submit = vi.fn();
		const onSubmittedWorkDone = vi.fn(async () => undefined);

		await expect(
			copyComputeBufferToRenderStorage({
				device: {
					createCommandEncoder: () => ({ copyBufferToBuffer, finish })
				} as unknown as GPUDevice,
				queue: { submit, onSubmittedWorkDone } as unknown as GPUQueue,
				sourceBuffer: {} as GPUBuffer,
				targetBuffer: { size: 64 } as GPUBuffer,
				byteLength: 32,
				now: performance.now.bind(performance),
				isSuperseded: () => true
			})
		).rejects.toThrow('superseded');
	});
});
```

- [ ] **Step 2: Run the bridge test to verify it fails**

Run:

```powershell
cd viewer
npx vitest run tests/scene/utciComputeBufferRenderBridge.test.ts
```

Expected: FAIL because the module does not exist.

- [ ] **Step 3: Implement the bridge helper**

Create `viewer/src/lib/components/scene/utciComputeBufferRenderBridge.ts` with pure helpers:

```ts
export interface RenderStorageBufferRef {
	device: GPUDevice;
	targetBuffer: GPUBuffer;
}

export interface WaitForRenderStorageBufferParams {
	deadlineMs: number;
	now: () => number;
	waitForNextFrame: () => Promise<void>;
	isSuperseded: () => boolean;
	readStorageBuffer: () => RenderStorageBufferRef | null;
}

export async function waitForRenderStorageBuffer(
	params: WaitForRenderStorageBufferParams
): Promise<RenderStorageBufferRef & { waitMs: number }> {
	const startedAt = params.now();
	const deadline = startedAt + params.deadlineMs;
	while (params.now() < deadline) {
		if (params.isSuperseded()) {
			throw new Error('GPU-resident render copy was superseded before storage initialization.');
		}
		const storage = params.readStorageBuffer();
		if (storage) {
			return { ...storage, waitMs: params.now() - startedAt };
		}
		await params.waitForNextFrame();
	}
	throw new Error('Timed out waiting for render-owned UTCI storage buffer.');
}

export interface CopyComputeBufferToRenderStorageParams {
	device: GPUDevice;
	queue: GPUQueue;
	sourceBuffer: GPUBuffer;
	targetBuffer: GPUBuffer & { size?: number };
	byteLength: number;
	now: () => number;
	isSuperseded?: () => boolean;
}

export async function copyComputeBufferToRenderStorage(
	params: CopyComputeBufferToRenderStorageParams
): Promise<{ bufferCopyMs: number; queueDrainMs: number }> {
	if (params.targetBuffer.size !== undefined && params.targetBuffer.size < params.byteLength) {
		throw new Error('Three storage buffer is smaller than the accepted compute output buffer.');
	}
	const copyStartedAt = params.now();
	const encoder = params.device.createCommandEncoder();
	encoder.copyBufferToBuffer(params.sourceBuffer, 0, params.targetBuffer, 0, params.byteLength);
	params.queue.submit([encoder.finish()]);
	const bufferCopyMs = params.now() - copyStartedAt;
	const queueDrainStartedAt = params.now();
	await params.queue.onSubmittedWorkDone();
	if (params.isSuperseded?.()) {
		throw new Error('GPU-resident render copy was superseded after queue drain.');
	}
	return {
		bufferCopyMs,
		queueDrainMs: params.now() - queueDrainStartedAt
	};
}
```

- [ ] **Step 4: Delegate from `UTCIPointCloud.svelte`**

In `viewer/src/lib/components/scene/UTCIPointCloud.svelte`, import:

```ts
import {
	copyComputeBufferToRenderStorage,
	waitForRenderStorageBuffer
} from './utciComputeBufferRenderBridge';
```

Replace the local storage wait/copy implementation with calls to the helper. Preserve:

- request supersession checks
- `gpuResidentCopyRunToken`
- visible-state gating
- diagnostics callback payload shape
- scene-owned mesh/material creation and disposal

- [ ] **Step 5: Run scene tests**

Run:

```powershell
cd viewer
npx vitest run tests/scene/utciComputeBufferRenderBridge.test.ts tests/scene/utci-surface-sync.test.ts tests/compute/live-selected-hour-route-projection.test.ts
```

Expected: PASS.

- [ ] **Step 6: Run route smoke probes**

Run:

```powershell
cd viewer
npm run test:e2e:selected-hour
```

Expected: PASS. If it fails, dump diagnostics and do not proceed to route extraction until root cause is understood.

---

## Task 6: Extract Canvas Interaction Plumbing Only

**Files:**
- Modify: `viewer/src/routes/+page.svelte`
- Modify: `viewer/src/routes/debug/+page.svelte`
- Create: `viewer/src/lib/components/viewer/canvasInteractionController.ts`
- Create: `viewer/tests/components/canvasInteractionController.test.ts`
- Modify: `viewer/tests/e2e/main-route-manual-diagnostics.spec.ts`
- Modify: `viewer/tests/e2e/debug-route-shared-host-diagnostics.spec.ts`

- [ ] **Step 1: Record the current canvas listener inventory**

Before editing, record exact line ranges for route-owned canvas/window listeners:

```powershell
rg -n "mousemove|mouseleave|wheel|pointerdown|pointerup|pointercancel|click|copy|disableTooltip|hover|cameraInteraction" viewer/src/routes/+page.svelte viewer/src/routes/debug/+page.svelte
```

Expected:

- This task is canvas/listener extraction only.
- Do not extract `ViewerShell`, `Scene`, `Model`, `UTCIPointCloud`, renderer diagnostics, or selected-hour host wiring in this task.
- If the inventory shows route-specific copy-click or debug-only controls, keep the route-specific callback in the route and only centralize listener attach/detach.

- [ ] **Step 2: Write controller unit tests**

Create `viewer/tests/components/canvasInteractionController.test.ts`:

```ts
import { describe, expect, it, vi } from 'vitest';
import { createCanvasInteractionController } from '$lib/components/viewer/canvasInteractionController';

describe('createCanvasInteractionController', () => {
	it('attaches and detaches canvas and window interaction listeners', () => {
		const canvas = document.createElement('canvas');
		const onPointerMove = vi.fn();
		const onPointerLeave = vi.fn();
		const onWheel = vi.fn();
		const onPointerDown = vi.fn();
		const onWindowPointerUp = vi.fn();

		const controller = createCanvasInteractionController({
			canvas,
			windowTarget: window,
			onPointerMove,
			onPointerLeave,
			onWheel,
			onPointerDown,
			onWindowPointerUp,
			onWindowPointerCancel: onWindowPointerUp
		});

		canvas.dispatchEvent(new MouseEvent('mousemove'));
		canvas.dispatchEvent(new MouseEvent('mouseleave'));
		canvas.dispatchEvent(new WheelEvent('wheel'));
		canvas.dispatchEvent(new PointerEvent('pointerdown'));
		window.dispatchEvent(new PointerEvent('pointerup'));
		window.dispatchEvent(new PointerEvent('pointercancel'));

		expect(onPointerMove).toHaveBeenCalledTimes(1);
		expect(onPointerLeave).toHaveBeenCalledTimes(1);
		expect(onWheel).toHaveBeenCalledTimes(1);
		expect(onPointerDown).toHaveBeenCalledTimes(1);
		expect(onWindowPointerUp).toHaveBeenCalledTimes(2);

		controller.dispose();
		canvas.dispatchEvent(new MouseEvent('mousemove'));
		window.dispatchEvent(new PointerEvent('pointerup'));

		expect(onPointerMove).toHaveBeenCalledTimes(1);
		expect(onWindowPointerUp).toHaveBeenCalledTimes(2);
	});
});
```

- [ ] **Step 3: Run controller test to verify it fails**

Run:

```powershell
cd viewer
npx vitest run tests/components/canvasInteractionController.test.ts
```

Expected: FAIL because the controller does not exist.

- [ ] **Step 4: Write or extend route-level regression probe**

If extracting canvas interactions, add assertions to existing E2E tests that still pass through both routes:

```ts
expect(diagnostics.tooltipInteraction?.hoverSampleCount ?? 0).toBeGreaterThan(0);
expect(diagnostics.cameraInteraction?.wheelEventCount ?? 0).toBeGreaterThan(0);
```

Use existing diagnostic payloads rather than adding DOM-only assertions.

- [ ] **Step 5: Run probe to establish baseline**

Run:

```powershell
cd viewer
npx playwright test tests/e2e/main-route-manual-diagnostics.spec.ts tests/e2e/debug-route-shared-host-diagnostics.spec.ts --project=chromium --workers=1 --reporter=list --timeout=30000
```

Expected: PASS before extraction.

- [ ] **Step 6: Extract listener attach/detach only**

Create `viewer/src/lib/components/viewer/canvasInteractionController.ts` with this shape:

```ts
export interface CanvasInteractionController {
	dispose(): void;
}

export interface CanvasInteractionControllerParams {
	canvas: HTMLCanvasElement;
	windowTarget?: Window;
	onPointerMove: (event: MouseEvent) => void;
	onPointerLeave?: (event: MouseEvent) => void;
	onWheel?: (event: WheelEvent) => void;
	onPointerDown?: (event: PointerEvent) => void;
	onClick?: (event: MouseEvent) => void;
	onWindowPointerUp?: (event: PointerEvent) => void;
	onWindowPointerCancel?: (event: PointerEvent) => void;
}

export function createCanvasInteractionController(
	params: CanvasInteractionControllerParams
): CanvasInteractionController {
	const windowTarget = params.windowTarget;
	params.canvas.addEventListener('mousemove', params.onPointerMove);
	if (params.onPointerLeave) params.canvas.addEventListener('mouseleave', params.onPointerLeave);
	if (params.onWheel) params.canvas.addEventListener('wheel', params.onWheel, { passive: true });
	if (params.onPointerDown) params.canvas.addEventListener('pointerdown', params.onPointerDown);
	if (params.onClick) params.canvas.addEventListener('click', params.onClick);
	if (windowTarget && params.onWindowPointerUp) windowTarget.addEventListener('pointerup', params.onWindowPointerUp);
	if (windowTarget && params.onWindowPointerCancel) windowTarget.addEventListener('pointercancel', params.onWindowPointerCancel);

	return {
		dispose() {
			params.canvas.removeEventListener('mousemove', params.onPointerMove);
			if (params.onPointerLeave) params.canvas.removeEventListener('mouseleave', params.onPointerLeave);
			if (params.onWheel) params.canvas.removeEventListener('wheel', params.onWheel);
			if (params.onPointerDown) params.canvas.removeEventListener('pointerdown', params.onPointerDown);
			if (params.onClick) params.canvas.removeEventListener('click', params.onClick);
			if (windowTarget && params.onWindowPointerUp) windowTarget.removeEventListener('pointerup', params.onWindowPointerUp);
			if (windowTarget && params.onWindowPointerCancel) windowTarget.removeEventListener('pointercancel', params.onWindowPointerCancel);
		}
	};
}
```

Then replace duplicated route listener setup with the controller. Keep route-specific callbacks in the routes for now.

- [ ] **Step 7: Run unit/build and route probes**

Run:

```powershell
cd viewer
npx vitest run tests/components/canvasInteractionController.test.ts
npm run test:quality:selected-hour
npm run test:e2e:selected-hour
npm run build
```

Expected: PASS.

- [ ] **Step 8: Capture touched-file static-check status**

Run:

```powershell
cd viewer
npm run check
```

Expected:

- May still fail repo-wide.
- No new errors in files touched by this task. If touched files appear in output, classify each as new vs inherited before proceeding.

---

## Task 7: Final Verification And Review Stop

**Files:**
- Inspect: `docs/webgpu_strategy_analysis.md`
- Inspect: `viewer/package.json`
- Inspect: `viewer/src/routes/+page.svelte`
- Inspect: `viewer/src/routes/debug/+page.svelte`
- Inspect: `viewer/src/lib/compute/liveSelectedHourController.ts`
- Inspect: `viewer/src/lib/components/scene/UTCIPointCloud.svelte`
- Inspect: `viewer/tests/e2e/main-route-manual-diagnostics.spec.ts`
- Inspect: `viewer/tests/e2e/debug-route-shared-host-diagnostics.spec.ts`
- Inspect: `viewer/tests/e2e/debug-route-parity-runtime-diagnostics.spec.ts`

- [ ] **Step 1: Run final focused units**

Run:

```powershell
cd viewer
npm run test:quality:selected-hour
npx vitest run tests/diagnostics/selectedHourRuntimeContract.test.ts tests/compute/selectedHourOutputHandle.test.ts tests/scene/utciComputeBufferRenderBridge.test.ts tests/compute/onDemandDiagnostics.test.ts
```

Expected: PASS.

- [ ] **Step 2: Run final required WebGPU probes**

Run:

```powershell
cd viewer
npm run test:e2e:selected-hour
```

Expected: PASS.

- [ ] **Step 3: Run build and static baseline**

Run:

```powershell
cd viewer
npm run build
npm run check
```

Expected:

- `npm run build` PASS.
- `npm run check` may still FAIL with known inherited debt. Report exact current count and whether any plan-touched file has new errors.

- [ ] **Step 4: Run whitespace diff guard**

Run from repo root:

```powershell
git diff --check
```

Expected: PASS or only preexisting line-ending warnings that are not introduced by this plan. If it reports whitespace in touched files, fix those files.

- [ ] **Step 5: Request review before implementation is called complete**

Ask two review subagents:

1. Runtime contract reviewer:

```text
Review the selected-hour runtime quality implementation. Focus on whether / and /debug now publish an honest selected-hour runtime contract, whether visible GPU transport is separated from range/tooltip/comparison readbacks, and whether debug parity remains debug-only. Return findings first with file/line evidence. Do not edit files.
```

2. Svelte/maintainability reviewer:

```text
Review the selected-hour runtime quality implementation for Svelte maintainability. Focus on whether routes are smaller composition roots, whether imperative side effects moved into stable plain TypeScript helpers where appropriate, and whether scene/WebGPU lifecycle ownership stayed in scene-facing modules. Return findings first with file/line evidence. Do not edit files.
```

- [ ] **Step 6: Stop for human review**

Report:

- files changed
- verification commands and results
- remaining inherited `svelte-check` debt
- any reviewer findings
- recommended next plan candidate

Do not commit. Do not continue into broader route-shell or 0.5m optimization work without user approval.

---

## Final Verification Commands

Run these before claiming this plan is complete:

```powershell
cd viewer
npm run test:quality:selected-hour
npx vitest run tests/diagnostics/selectedHourRuntimeContract.test.ts tests/compute/selectedHourOutputHandle.test.ts tests/scene/utciComputeBufferRenderBridge.test.ts tests/compute/onDemandDiagnostics.test.ts
npx vitest run tests/components/canvasInteractionController.test.ts
npm run test:e2e:selected-hour
npm run build
npm run check
cd ..
git diff --check
```

Expected completion state:

- `/` still proves `compute-buffer-selected-hour`, same-device compute/render, no `.bin` requests, and no data texture build on the strong visible path.
- `/debug` normal f32 still proves shared-host ownership with zero legacy dispatch/scrub overlap.
- `/debug` parity still proves `legacy-debug` and August-only Python `.bin` comparison validity.
- Readback reasons distinguish visible fallback, range, tooltip, comparison, and debug readbacks.
- Selected-hour GPU output ownership is explicit and disposal is idempotent.
- `UTCIPointCloud.svelte` delegates compute-buffer copy mechanics to a testable helper but still owns scene lifecycle.
- `npm run check` status is honestly reported; known inherited debt is not confused with this plan’s work.

---

## Next-Agent Execution Handoff

Use this plan with `subagent-driven-development` and `verification-before-completion`.

Execution requirements:

- Work in `D:\Projects\Nur\Shade\fast-utci`.
- Do not create commits.
- Do not create git worktrees.
- Preserve unrelated dirty files.
- Read this full plan before dispatching implementers.
- Execute task-by-task, not as one giant edit.
- Use a fresh implementation subagent per task.
- After each implementation task, run a spec-compliance review subagent first, then a code-quality review subagent.
- Do not start code-quality review until spec compliance is clean.
- If blocked, report findings first with file/line evidence and stop before broad rewrites.
- Do not claim completion without fresh verification output from the relevant commands in this plan.
