# Ness Tziona Red Test And Compute Organization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Workflow overrides:** Do **not** create commits. Do **not** create git worktrees. Preserve unrelated dirty files. Treat any generated Playwright state such as `viewer/test-results/.last-run.json` as generated state; clean it before final status. Do not reorganize `viewer/src/lib/compute/` until the Ness Tziona selected-hour E2E is green in the current topology.

**Goal:** Restore the red Ness Tziona main-route selected-hour E2E, then perform a narrowly scoped compute-folder organization pass without weakening selected-hour, debug parity, or fallback proof boundaries.

**Architecture:** First debug the live product route as a behavior problem using `superpowers:systematic-debugging`: reproduce, trace route/host/session/range publication, add the smallest missing coverage, and fix root cause before any folder movement. Only after the selected-hour quality and E2E gates are green, reorganize `viewer/src/lib/compute/` by moving one already-proven ownership cluster at a time, with import/source-lock updates and no behavior edits.

**Tech Stack:** SvelteKit/Svelte 5 with legacy `$:` reactivity, TypeScript, Three/Threlte, WebGPU/WGSL, Vitest, Playwright Chromium with WebGPU, PowerShell on Windows.

---

## Current Evidence

The current clean-tree verification found:

- `cd viewer; npm run test:quality:selected-hour` passed: 18 files / 158 tests.
- `cd viewer; npm run test:e2e:selected-hour` failed: 12 passed, 1 failed.
- The reproducible failing test is `viewer/tests/e2e/main-route-manual-diagnostics.spec.ts`, case `uses live WebGPU UTCI range independent of .bin metadata for Ness Tziona`.
- The focused one-test rerun also failed.
- Last observed diagnostics during failure:

```json
{
  "baseRenderTransport": "idle",
  "comparisonRenderTransport": "idle",
  "baseLiveReady": false,
  "comparisonLiveReady": true,
  "baseSelectedMonthIndex": 7,
  "baseSelectedHourIndex": 0,
  "baseSelectedTimeIndex": 168,
  "selectedHourRuntimeContract": {
    "renderTransport": "none",
    "utciSurfaceSource": "none",
    "sameDeviceForComputeAndRender": false,
    "dataTextureBuildCount": 0,
    "strongVisibleGpuPath": false,
    "visibleRenderPathAvoidsCpuReadback": false
  }
}
```

This is a publication/readiness failure for the main route Ness Tziona base selected-hour path, not merely a range assertion failure.

## Hard Constraints

- Do not create commits.
- Do not create git worktrees.
- Do not remove or rename fallback/proof surfaces:
  - `runAll()`
  - `.bin`
  - Python comparison/reference paths
  - `readUtciBulk()`
  - `readUtcisSlice()`
  - `dataTexture`
  - debug parity, collect, and legacy selected-hour paths
- Do not make the main route pass by reading `.bin` metadata or Python reference output.
- Do not weaken `strongVisibleGpuPath` requirements.
- Do not claim GPU-resident memory release beyond what diagnostics prove.
- Do not move files under `viewer/src/lib/compute/` until Task 3 is green.
- If three different behavior fixes fail, stop and report findings before continuing.

## Perspective-Ensemble Review Summary

Panel A council:

- Treat this as a sequencing and proof-boundary problem.
- Fix the Ness Tziona E2E first.
- Preserve `compute-buffer-selected-hour`, same-device proof, zero visible selected-hour readback, and zero `dataTexture` rebuilds.
- Trace whether failure originates in selected-hour session, route host publication, route projection, scene acknowledgement, or legend/range projection.
- Reorganize compute only after the behavior surface is green.

Panel B red-cell:

- Do not treat compute reorganization as a harmless mechanical move.
- `viewer/src/lib/compute/` is imported by routes, scene components, debug helpers, services, parity tests, WGSL raw imports, and Vitest mocks.
- Moving files can silently affect bundling, mocks, source-lock coverage, circular import timing, and debug parity gates.
- The reorg must be a separate gated phase inside this plan, with no behavior edits and full selected-hour verification afterward.

## File Structure Target

### Behavior Fix Phase

Likely inspect/modify:

- `viewer/tests/e2e/main-route-manual-diagnostics.spec.ts`
  - Existing red browser proof.
  - Add diagnostic capture only if needed.

- `viewer/tests/compute/live-selected-hour-route-host.test.ts`
  - Route-host readiness/publication coverage if the root cause is host state.

- `viewer/tests/compute/live-selected-hour-route-projection.test.ts`
  - Projection coverage if the root cause is route projection from host state.

- `viewer/tests/compute/live-selected-hour-session.test.ts`
  - Session coverage if the root cause is selected-day/range/session result generation.

- `viewer/tests/compute/live-selected-hour-controller.test.ts`
  - Controller coverage if the root cause is accepted GPU output, fallback, release, or readiness transition.

- `viewer/src/routes/+page.svelte`
  - Only if route-level reactive input or diagnostics wiring is the root cause.

- `viewer/src/lib/compute/liveSelectedHourRouteHost.ts`
  - Only if route host does not start/publish the Ness Tziona base computation correctly.

- `viewer/src/lib/compute/liveSelectedHourRouteProjection.ts`
  - Only if projection misclassifies readiness or selected-hour surface state.

- `viewer/src/lib/compute/liveUtciSelectedHourSession.ts`
  - Only if session execution/range generation is wrong.

- `viewer/src/lib/compute/liveSelectedHourController.ts`
  - Only if controller readiness/accepted output state is wrong.

### Compute Organization Phase

Create folders only after the behavior gate is green. Preferred first-pass structure:

- `viewer/src/lib/compute/core/`
  - `analysisGridFromBounds.ts`
  - `canonicalGrid.ts`
  - `grid-generator.ts`
  - `mrtReference.ts`
  - `solarcal.ts`
  - `sunpath.ts`
  - `tregenza.ts`
  - `utci.ts`

- `viewer/src/lib/compute/gpu/`
  - `bvhGpuUpload.ts`
  - `gpu-pipeline.ts`
  - `mergeAndBvh.worker.ts`
  - `mergeAndBvhWorkerClient.ts`
  - `meshMerger.ts`
  - `webgpuDeviceLimits.ts`
  - `webgpuUtciPipeline.ts`
  - `shaders/`

- `viewer/src/lib/compute/selected-hour/`
  - `liveSelectedHourController.ts`
  - `liveSelectedHourRenderContext.ts`
  - `liveSelectedHourRouteHost.ts`
  - `liveSelectedHourRouteProjection.ts`
  - `liveSelectedHourSurfaceIdentity.ts`
  - `liveUtciAnalysis.ts`
  - `liveUtciSelectedHour.ts`
  - `liveUtciSelectedHourSession.ts`
  - `selectedHourOutputHandle.ts`

- `viewer/src/lib/compute/on-demand/`
  - `onDemandDiagnostics.ts`
  - `onDemandOutputFormat.ts`
  - `onDemandPrototypeStatus.ts`
  - `onDemandScrubState.ts`
  - `onDemandSizing.ts`

- `viewer/src/lib/compute/weather/`
  - `epw-parser.ts`
  - `projectWeather.ts`

- `viewer/src/lib/compute/telemetry.ts`
  - Keep at root unless imports clearly fit a later diagnostics folder.

- `viewer/src/lib/compute/compute-manager.ts`
  - Keep at root for this implementation plan as an intentional legacy/orchestration facade.
  - It imports across GPU, weather, core math, and on-demand modules and is used by debug and selected-hour paths.
  - Decide its final home in a later dedicated compute-manager plan after the selected-hour move lands.

Do not move all clusters at once. This implementation plan executes only the Ness Tziona fix and selected-hour cluster move. It also records the full compute organization roadmap so follow-up plans do not have to rediscover the same file buckets. GPU/shader, core math, on-demand, weather, `compute-manager.ts`, and root-leftover moves are roadmap-only here and must not be executed by the next implementation agent.

## Task 0: Baseline And Plan Gates

**Files:**
- Inspect only.

- [ ] **Step 1: Confirm clean status**

Run from repo root:

```powershell
git status --short
git log --oneline -8
```

Expected:

- `git status --short` has no output, or only unrelated user changes that are documented and preserved.
- Recent history includes `75bf6d9 refactor(viewer): thin main route product shell`.

- [ ] **Step 2: Reproduce the focused red test**

Run from `viewer/`:

```powershell
npx playwright test tests/e2e/main-route-manual-diagnostics.spec.ts --project=chromium --grep "uses live WebGPU UTCI range independent of .bin metadata for Ness Tziona" --workers=1 --reporter=list --timeout=30000
```

Expected before fix:

- FAIL with timeout waiting for selected-hour publication diagnostics.
- Last diagnostics show `baseLiveReady: false` and `baseRenderTransport: "idle"` or another concrete failure mode.

If the test unexpectedly passes twice in a row, stop and report that the previous failure may be environmental or order-dependent. Then run the full selected-hour E2E before proceeding.

- [ ] **Step 3: Record current generated state**

Run from repo root:

```powershell
git status --short
```

Expected:

- If Playwright modified `viewer/test-results/.last-run.json`, leave it unstaged and clean it at final verification.

## Task 1: Root-Cause Investigation For Ness Tziona Publication

**Files:**
- Inspect: `viewer/tests/e2e/main-route-manual-diagnostics.spec.ts`
- Inspect: `viewer/src/routes/+page.svelte`
- Inspect: `viewer/src/routes/main/MainRouteViewport.svelte`
- Inspect: `viewer/src/lib/compute/liveSelectedHourRouteHost.ts`
- Inspect: `viewer/src/lib/compute/liveSelectedHourRouteProjection.ts`
- Inspect: `viewer/src/lib/compute/liveUtciSelectedHourSession.ts`
- Inspect: `viewer/src/lib/compute/liveSelectedHourController.ts`
- Inspect: `viewer/src/lib/compute/projectWeather.ts`

- [ ] **Step 1: Read the failing test helper and target case**

Run from repo root:

```powershell
rg -n "waitForSelectedHourPublication|uses live WebGPU UTCI range independent|expectedSelectionKey|baseAcceptedUtciRange|readUtciLegendValues" viewer/tests/e2e/main-route-manual-diagnostics.spec.ts
```

Expected:

- Identify the helper gate requiring `baseLiveReady === true`, selected-hour selection match, `baseRenderTransport === "compute-buffer-selected-hour"`, same-device proof, zero visible readback, and `strongVisibleGpuPath === true`.

- [ ] **Step 2: Trace route-level inputs**

Run:

```powershell
rg -n "selectedMonthIndex|selectedHourIndex|selectedTimeIndex|setRouteInputs|projectMainRouteLiveSceneState|publishMainRouteUtciDiagnostics" viewer/src/routes/+page.svelte
```

Expected:

- Confirm `liveRouteHost.setRouteInputs(...)` receives Ness Tziona `analysisId`, base analysis/model, selected month/hour/time index, render mode, renderer backend/device, and surface backend.
- Confirm diagnostics are published from route-owned state, not hidden inside extracted viewport/tooltip components.

- [ ] **Step 3: Trace host/session start conditions**

Run:

```powershell
rg -n "setRouteInputs|base|comparison|analysisId|model|timeIndex|compute|start|request|selectedHour" viewer/src/lib/compute/liveSelectedHourRouteHost.ts
```

Expected:

- Identify the condition that should start the base selected-hour computation.
- Compare base and comparison paths because observed diagnostics showed `comparisonLiveReady: true` while `baseLiveReady: false`.

- [ ] **Step 4: Trace selected-hour session output and range ownership**

Run:

```powershell
rg -n "utciRange|tooltipUtciValues|computeSelectedHour|gpuResident|cpu-uploaded-selected-hour|compute-buffer-selected-hour|resolveLiveGpuResidentUtciRange" viewer/src/lib/compute/liveUtciSelectedHourSession.ts viewer/src/lib/compute/liveUtciSelectedHour.ts viewer/src/lib/compute/liveSelectedHourController.ts
```

Expected:

- Determine whether the Ness Tziona base request fails to start, starts but rejects, starts but never publishes, or publishes without scene acknowledgement.

- [ ] **Step 5: Form one written hypothesis**

Before editing code, add a short note to the implementation log section of this plan or the final results doc with exactly one hypothesis:

```markdown
### Ness Tziona Root-Cause Hypothesis

I think the root cause is: name the single failing boundary, such as route input, route host scheduling, session result generation, controller acceptance, route projection, scene acknowledgement, or diagnostics publication.

Evidence:
- Include the focused Playwright failure diagnostic field that proves the boundary is failing.
- Include the code path or unit-test observation that explains why that boundary fails for Ness Tziona.

The smallest test that should fail before the fix is: name one Vitest case or the existing focused Playwright case.
```

Expected:

- The hypothesis names one component boundary: route input, route host, session, controller, projection, scene acknowledgement, or diagnostics.
- Do not implement a fix until this is written.

## Task 2: Add Focused Regression Coverage

**Files:**
- Modify one or more, based on Task 1:
  - `viewer/tests/compute/live-selected-hour-route-host.test.ts`
  - `viewer/tests/compute/live-selected-hour-route-projection.test.ts`
  - `viewer/tests/compute/live-selected-hour-session.test.ts`
  - `viewer/tests/compute/live-selected-hour-controller.test.ts`
  - `viewer/tests/e2e/main-route-manual-diagnostics.spec.ts`

- [ ] **Step 1: Choose the smallest non-browser regression test**

Use this decision table:

```text
If route host does not start base computation:
  add/adjust viewer/tests/compute/live-selected-hour-route-host.test.ts

If host state is correct but projected state says base is not live:
  add/adjust viewer/tests/compute/live-selected-hour-route-projection.test.ts

If session returns wrong/missing selected-hour range:
  add/adjust viewer/tests/compute/live-selected-hour-session.test.ts

If controller discards accepted output or falls back incorrectly:
  add/adjust viewer/tests/compute/live-selected-hour-controller.test.ts

If behavior depends on Svelte/scene acknowledgement only:
  keep the existing Playwright case as the primary regression and add diagnostic detail there only if needed.
```

Expected:

- Do not add broad tests that only duplicate the existing E2E.
- The new/adjusted unit test should fail before implementation if the root cause is unit-testable.

- [ ] **Step 2: Run the chosen focused test and prove red**

Run the exact Vitest file selected in Step 1. Example commands:

```powershell
npx vitest run tests/compute/live-selected-hour-route-host.test.ts
npx vitest run tests/compute/live-selected-hour-route-projection.test.ts
npx vitest run tests/compute/live-selected-hour-session.test.ts
npx vitest run tests/compute/live-selected-hour-controller.test.ts
```

Expected:

- FAIL for the new regression if a unit-level reproduction exists.
- If no unit reproduction exists, document why the browser E2E remains the failing regression.

## Task 3: Implement The Minimal Ness Tziona Fix

**Files:**
- Modify only the root-cause file(s) identified by Task 1 and guarded by Task 2.
- Do not move files in this task.

- [ ] **Step 1: Make the smallest behavior change**

Allowed examples:

```text
Route input fix:
  ensure Ness Tziona base analysis/model/timeIndex reaches liveRouteHost.setRouteInputs(...)

Route host fix:
  ensure base computation is scheduled when analysis/model changes from Ben-Gurion to Ness Tziona

Session fix:
  ensure selected-hour output/range is produced from live WebGPU result, not .bin metadata/default range

Controller fix:
  ensure accepted GPU output remains publishable until scene acknowledgement/replacement

Projection fix:
  ensure baseLiveReady reflects the current base compute-buffer-selected-hour surface only when selection/request match
```

Expected:

- No debug parity changes unless Task 1 proves shared code requires them.
- No `.bin`, Python, or metadata shortcut on the main route.
- No changes to compute folder layout.

- [ ] **Step 2: Run the focused unit regression**

Run the chosen command from Task 2.

Expected:

- PASS.

- [ ] **Step 3: Run the focused browser regression**

Run from `viewer/`:

```powershell
npx playwright test tests/e2e/main-route-manual-diagnostics.spec.ts --project=chromium --grep "uses live WebGPU UTCI range independent of .bin metadata for Ness Tziona" --workers=1 --reporter=list --timeout=30000
```

Expected:

- PASS.
- Diagnostics satisfy `baseRenderTransport === "compute-buffer-selected-hour"`, `baseLiveReady === true`, and `strongVisibleGpuPath === true`.
- The range is finite and not equal to `{ min: 23.165335456154907, max: 37.661211206353144 }` or `{ min: -20, max: 60 }`.

- [ ] **Step 4: Run selected-hour quality and E2E gates**

Run from `viewer/`:

```powershell
npm run test:quality:selected-hour
npm run test:e2e:selected-hour
```

Expected:

- `test:quality:selected-hour`: PASS, currently 18 files / 158 tests.
- `test:e2e:selected-hour`: PASS, 13 Chromium tests.

- [ ] **Step 5: Request subagent verification before reorg**

Dispatch two read-only reviewers:

```text
Reviewer 1: Selected-hour behavior/proof reviewer.
Scope: Review changes from Task 3 only. Confirm Ness Tziona fix does not weaken strongVisibleGpuPath, same-device proof, zero visible readback, dataTexture fallback, or debug parity boundaries.

Reviewer 2: Static/scope reviewer.
Scope: Review changes from Task 3 only. Confirm no compute folder move, no unrelated cleanup, no generated state, and no new touched-file check debt.
```

Expected:

- Both reviewers report no blockers.
- If either reports a blocker, fix it before Task 4.

## Task 4: Compute Organization Map And Stop Gate

**Files:**
- Create: `docs/superpowers/plans/2026-05-12-compute-folder-organization-map.md`
- Inspect: `viewer/src/lib/compute/**`
- Inspect: `viewer/tests/**`
- Inspect: `docs/webgpu_strategy_analysis.md`

- [ ] **Step 1: Generate a before/after import map**

Run from repo root:

```powershell
rg -n "\$lib/compute|src/lib/compute|compute/" viewer/src viewer/tests docs/webgpu_strategy_analysis.md > .compute-import-map.before.txt
```

Expected:

- `.compute-import-map.before.txt` exists as a temporary planning artifact.
- Do not commit it.

- [ ] **Step 2: Write the organization map**

Create `docs/superpowers/plans/2026-05-12-compute-folder-organization-map.md` with:

```markdown
# Compute Folder Organization Map

Date: 2026-05-12

## Current Problem

`viewer/src/lib/compute/` currently mixes domain math, WebGPU pipeline code, selected-hour orchestration, on-demand diagnostics, weather parsing, workers, WGSL shaders, and route-facing contracts.

## Proposed Buckets

| Bucket | New path | Files |
| --- | --- | --- |
| Core/domain math | `viewer/src/lib/compute/core/` | `analysisGridFromBounds.ts`, `canonicalGrid.ts`, `grid-generator.ts`, `mrtReference.ts`, `solarcal.ts`, `sunpath.ts`, `tregenza.ts`, `utci.ts` |
| WebGPU pipeline and workers | `viewer/src/lib/compute/gpu/` | `bvhGpuUpload.ts`, `gpu-pipeline.ts`, `mergeAndBvh.worker.ts`, `mergeAndBvhWorkerClient.ts`, `meshMerger.ts`, `webgpuDeviceLimits.ts`, `webgpuUtciPipeline.ts`, `shaders/` |
| Selected-hour orchestration | `viewer/src/lib/compute/selected-hour/` | `liveSelectedHourController.ts`, `liveSelectedHourRenderContext.ts`, `liveSelectedHourRouteHost.ts`, `liveSelectedHourRouteProjection.ts`, `liveSelectedHourSurfaceIdentity.ts`, `liveUtciAnalysis.ts`, `liveUtciSelectedHour.ts`, `liveUtciSelectedHourSession.ts`, `selectedHourOutputHandle.ts` |
| On-demand diagnostics/state | `viewer/src/lib/compute/on-demand/` | `onDemandDiagnostics.ts`, `onDemandOutputFormat.ts`, `onDemandPrototypeStatus.ts`, `onDemandScrubState.ts`, `onDemandSizing.ts` |
| Weather | `viewer/src/lib/compute/weather/` | `epw-parser.ts`, `projectWeather.ts` |
| Root for now | `viewer/src/lib/compute/` | `compute-manager.ts`, `telemetry.ts` |

## First Move Slice

Move only the selected-hour orchestration bucket first. This slice has the clearest current boundary and is already guarded by selected-hour tests and source locks.

## Reorg Non-Goals

- No behavior edits.
- No Python/data/script movement.
- No debug route beautification.
- No fallback removal.
- No `webgpuUtciPipeline.ts` splitting in this pass.
```

Expected:

- The map documents the intended full shape, but the implementation only moves the first slice unless reviewers approve more.

- [ ] **Step 3: Request subagent review of the map**

Dispatch three read-only reviewers:

```text
Reviewer 1: Import-boundary reviewer.
Check whether the selected-hour first slice is the safest first move and identify source-lock/import updates.

Reviewer 2: Selected-hour behavior reviewer.
Check that moving selected-hour files will not blur route, controller, session, scene, or diagnostics proof boundaries.

Reviewer 3: Debug parity/fallback reviewer.
Check that the move plan preserves debug `.bin`, Python comparison, August-only validity, collect, runAll, dataTexture, readUtciBulk, and readUtcisSlice.
```

Expected:

- If any reviewer rejects the selected-hour first slice, stop and revise the map before moving files.

## Task 5: Mechanical Compute Reorg, Slice 1 - Selected-Hour

**Files:**
- Move selected-hour files only:
  - `viewer/src/lib/compute/liveSelectedHourController.ts`
  - `viewer/src/lib/compute/liveSelectedHourRenderContext.ts`
  - `viewer/src/lib/compute/liveSelectedHourRouteHost.ts`
  - `viewer/src/lib/compute/liveSelectedHourRouteProjection.ts`
  - `viewer/src/lib/compute/liveSelectedHourSurfaceIdentity.ts`
  - `viewer/src/lib/compute/liveUtciAnalysis.ts`
  - `viewer/src/lib/compute/liveUtciSelectedHour.ts`
  - `viewer/src/lib/compute/liveUtciSelectedHourSession.ts`
  - `viewer/src/lib/compute/selectedHourOutputHandle.ts`
- Update imports in:
  - `viewer/src/**`
  - `viewer/tests/**`
- Update source-lock paths in:
  - `viewer/tests/debug/debug-route-decomposition-source-lock.test.ts`
  - `viewer/tests/routes/main-route-debug-boundary-source-lock.test.ts`
  - Any other test that explicitly reads `src/lib/compute/liveSelectedHour*.ts`

- [ ] **Step 1: Move only the selected-hour cluster**

Use native PowerShell move commands or editor move operations. Target:

```text
viewer/src/lib/compute/selected-hour/liveSelectedHourController.ts
viewer/src/lib/compute/selected-hour/liveSelectedHourRenderContext.ts
viewer/src/lib/compute/selected-hour/liveSelectedHourRouteHost.ts
viewer/src/lib/compute/selected-hour/liveSelectedHourRouteProjection.ts
viewer/src/lib/compute/selected-hour/liveSelectedHourSurfaceIdentity.ts
viewer/src/lib/compute/selected-hour/liveUtciAnalysis.ts
viewer/src/lib/compute/selected-hour/liveUtciSelectedHour.ts
viewer/src/lib/compute/selected-hour/liveUtciSelectedHourSession.ts
viewer/src/lib/compute/selected-hour/selectedHourOutputHandle.ts
```

Expected:

- No other compute files move in this step.

- [ ] **Step 2: Update imports mechanically**

Replace exact import roots:

```text
$lib/compute/liveSelectedHourController -> $lib/compute/selected-hour/liveSelectedHourController
$lib/compute/liveSelectedHourRenderContext -> $lib/compute/selected-hour/liveSelectedHourRenderContext
$lib/compute/liveSelectedHourRouteHost -> $lib/compute/selected-hour/liveSelectedHourRouteHost
$lib/compute/liveSelectedHourRouteProjection -> $lib/compute/selected-hour/liveSelectedHourRouteProjection
$lib/compute/liveSelectedHourSurfaceIdentity -> $lib/compute/selected-hour/liveSelectedHourSurfaceIdentity
$lib/compute/liveUtciAnalysis -> $lib/compute/selected-hour/liveUtciAnalysis
$lib/compute/liveUtciSelectedHour -> $lib/compute/selected-hour/liveUtciSelectedHour
$lib/compute/liveUtciSelectedHourSession -> $lib/compute/selected-hour/liveUtciSelectedHourSession
$lib/compute/selectedHourOutputHandle -> $lib/compute/selected-hour/selectedHourOutputHandle
```

Expected:

- `rg "\$lib/compute/(liveSelectedHour|liveUtci|selectedHourOutputHandle)" viewer/src viewer/tests` returns no old selected-hour imports.

- [ ] **Step 3: Update source-lock paths**

Run:

```powershell
rg -n "src/lib/compute/liveSelectedHour|src/lib/compute/liveUtci|src/lib/compute/selectedHourOutputHandle" viewer/tests
```

Update those tests to the new selected-hour paths.

Expected:

- Source locks still scan the selected-hour host/projection files for debug-only leakage.
- Main-route source locks still forbid `.bin`, parity, Python, `loadReferenceFromFs`, `__onDemandPrototypeDiagnostics__`, and legacy debug constants.

- [ ] **Step 4: Run mechanical import verification**

Run from `viewer/`:

```powershell
npx vitest run tests/compute/live-selected-hour-controller.test.ts tests/compute/live-selected-hour-route-host.test.ts tests/compute/live-selected-hour-route-projection.test.ts tests/compute/live-selected-hour-render-context.test.ts tests/compute/live-selected-hour-session.test.ts tests/compute/live-selected-hour-time-index.test.ts tests/compute/live-utci-analysis.test.ts tests/compute/selectedHourOutputHandle.test.ts tests/routes/main-route-debug-boundary-source-lock.test.ts tests/debug/debug-route-decomposition-source-lock.test.ts
```

Expected:

- PASS.

- [ ] **Step 5: Run selected-hour quality and browser gates**

Run from `viewer/`:

```powershell
npm run test:quality:selected-hour
npm run test:e2e:selected-hour
npm run build
```

Expected:

- `test:quality:selected-hour`: PASS.
- `test:e2e:selected-hour`: PASS, 13 Chromium tests.
- `build`: PASS, allowing only existing documented warnings.

- [ ] **Step 6: Run check and diff hygiene**

Run:

```powershell
cd viewer
npm run check
cd ..
git diff --check
git status --short
```

Expected:

- `npm run check` may still FAIL with inherited static debt. Any touched-file errors introduced by the reorg must be fixed.
- `git diff --check`: PASS.
- `git status --short` contains only intended source/doc changes. Clean generated Playwright state if present.

- [ ] **Step 7: Request post-reorg subagent verification**

Dispatch three read-only reviewers:

```text
Reviewer 1: Import-boundary reviewer.
Confirm old selected-hour import paths are gone, source locks point at new files, and no unrelated compute clusters moved.

Reviewer 2: Selected-hour behavior reviewer.
Confirm strongVisibleGpuPath proof boundaries are preserved and no route reactivity was hidden or weakened.

Reviewer 3: Debug parity/fallback reviewer.
Confirm debug `.bin`, Python comparison, August-only validity, collect, runAll, dataTexture, readUtciBulk, and readUtcisSlice paths remain available.
```

Expected:

- No blockers.
- If blockers exist, fix them before writing results.

## Task 6: Future Roadmap Only - GPU Pipeline And Shaders

Do not execute this task in the current implementation pass. Convert it into a separate future implementation plan after Task 11 lands and the selected-hour result note is reviewed.

**Files:**
- Move GPU files only:
  - `viewer/src/lib/compute/bvhGpuUpload.ts`
  - `viewer/src/lib/compute/gpu-pipeline.ts`
  - `viewer/src/lib/compute/mergeAndBvh.worker.ts`
  - `viewer/src/lib/compute/mergeAndBvhWorkerClient.ts`
  - `viewer/src/lib/compute/meshMerger.ts`
  - `viewer/src/lib/compute/webgpuDeviceLimits.ts`
  - `viewer/src/lib/compute/webgpuUtciPipeline.ts`
  - `viewer/src/lib/compute/shaders/`
- Update imports in:
  - `viewer/src/**`
  - `viewer/tests/**`
- Update source-lock paths in:
  - `viewer/tests/compute/webgpu-on-demand-source-locks.test.ts`
  - `viewer/tests/compute/webgpu-pipeline-implementation.test.ts`
  - Any test that explicitly reads `src/lib/compute/shaders/*` or `src/lib/compute/webgpuUtciPipeline.ts`

- [ ] **Step 1: Future plan gate - confirm selected-hour slice is green**

Run from `viewer/`:

```powershell
npm run test:quality:selected-hour
npm run test:e2e:selected-hour
```

Expected for the future plan:

- Both PASS before moving GPU files.

- [ ] **Step 2: Future plan action - move only the GPU cluster**

Target:

```text
viewer/src/lib/compute/gpu/bvhGpuUpload.ts
viewer/src/lib/compute/gpu/gpu-pipeline.ts
viewer/src/lib/compute/gpu/mergeAndBvh.worker.ts
viewer/src/lib/compute/gpu/mergeAndBvhWorkerClient.ts
viewer/src/lib/compute/gpu/meshMerger.ts
viewer/src/lib/compute/gpu/webgpuDeviceLimits.ts
viewer/src/lib/compute/gpu/webgpuUtciPipeline.ts
viewer/src/lib/compute/gpu/shaders/
```

Expected for the future plan:

- No core math, on-demand, weather, or debug files move in this step.
- WGSL raw imports still use valid Vite `?raw` paths after the move.
- Exact `vi.mock(...)` specifiers that point at moved GPU modules are updated.

- [ ] **Step 3: Future plan action - update imports mechanically**

Replace exact import roots:

```text
$lib/compute/bvhGpuUpload -> $lib/compute/gpu/bvhGpuUpload
$lib/compute/gpu-pipeline -> $lib/compute/gpu/gpu-pipeline
$lib/compute/mergeAndBvh.worker -> $lib/compute/gpu/mergeAndBvh.worker
$lib/compute/mergeAndBvhWorkerClient -> $lib/compute/gpu/mergeAndBvhWorkerClient
$lib/compute/meshMerger -> $lib/compute/gpu/meshMerger
$lib/compute/webgpuDeviceLimits -> $lib/compute/gpu/webgpuDeviceLimits
$lib/compute/webgpuUtciPipeline -> $lib/compute/gpu/webgpuUtciPipeline
$lib/compute/shaders -> $lib/compute/gpu/shaders
```

Expected for the future plan:

- `rg "\$lib/compute/(bvhGpuUpload|gpu-pipeline|mergeAndBvh|meshMerger|webgpuDeviceLimits|webgpuUtciPipeline|shaders)" viewer/src viewer/tests` returns no old GPU imports.
- Hard-coded filesystem reads such as `src/lib/compute/webgpuUtciPipeline.ts` and `src/lib/compute/shaders/*.wgsl` are updated.
- `docs/webgpu_strategy_analysis.md` path references are updated if this slice moves files mentioned there.

- [ ] **Step 4: Future plan gate - run GPU-focused tests**

Run from `viewer/`:

```powershell
npx vitest run tests/compute/bvhGpuUpload.test.ts tests/compute/exposure-pipeline.test.ts tests/compute/gpu-pipeline.test.ts tests/compute/live-utci-analysis.test.ts tests/compute/mergeAndBvhWorkerClient.test.ts tests/compute/meshMerger.test.ts tests/compute/mrt-reference-vs-shader.test.ts tests/compute/mrt-utci-gpu-solarcal.test.ts tests/compute/scene-renderer-sizing.test.ts tests/compute/webgpu-on-demand-source-locks.test.ts tests/compute/webgpu-pipeline-implementation.test.ts tests/compute/webgpuUtciPipeline.behavior.test.ts
```

Expected:

- PASS.

- [ ] **Step 5: Future plan gate - run selected-hour and build gates**

Run from `viewer/`:

```powershell
npm run test:quality:selected-hour
npm run test:e2e:selected-hour
npm run build
```

Expected for the future plan:

- PASS. Stop if a browser-only WGSL/raw-import failure appears.

## Task 7: Future Roadmap Only - Core Math

Do not execute this task in the current implementation pass. Convert it into a separate future implementation plan after the GPU/shader slice is complete and verified.

**Files:**
- Move core math files only:
  - `viewer/src/lib/compute/analysisGridFromBounds.ts`
  - `viewer/src/lib/compute/canonicalGrid.ts`
  - `viewer/src/lib/compute/grid-generator.ts`
  - `viewer/src/lib/compute/mrtReference.ts`
  - `viewer/src/lib/compute/solarcal.ts`
  - `viewer/src/lib/compute/sunpath.ts`
  - `viewer/src/lib/compute/tregenza.ts`
  - `viewer/src/lib/compute/utci.ts`
- Move colocated source test only if it still makes sense:
  - `viewer/src/lib/compute/analysisGridFromBounds.test.ts`
- Update imports and hard-coded source references in `viewer/src/**`, `viewer/tests/**`, WGSL comments, and docs.

- [ ] **Step 1: Future plan gate - confirm GPU slice is green**

Run from `viewer/`:

```powershell
npm run test:quality:selected-hour
npm run test:e2e:selected-hour
npm run build
```

Expected for the future plan:

- PASS before moving core math.

- [ ] **Step 2: Future plan action - move only the core math cluster**

Target:

```text
viewer/src/lib/compute/core/analysisGridFromBounds.ts
viewer/src/lib/compute/core/canonicalGrid.ts
viewer/src/lib/compute/core/grid-generator.ts
viewer/src/lib/compute/core/mrtReference.ts
viewer/src/lib/compute/core/solarcal.ts
viewer/src/lib/compute/core/sunpath.ts
viewer/src/lib/compute/core/tregenza.ts
viewer/src/lib/compute/core/utci.ts
```

Expected for the future plan:

- No behavior edits.
- Preserve Ladybug/Python parity-sensitive formulas exactly.
- Update comments such as `src/lib/compute/mrtReference.ts` references inside WGSL files.

- [ ] **Step 3: Future plan action - update imports mechanically**

Replace exact import roots:

```text
$lib/compute/analysisGridFromBounds -> $lib/compute/core/analysisGridFromBounds
$lib/compute/canonicalGrid -> $lib/compute/core/canonicalGrid
$lib/compute/grid-generator -> $lib/compute/core/grid-generator
$lib/compute/mrtReference -> $lib/compute/core/mrtReference
$lib/compute/solarcal -> $lib/compute/core/solarcal
$lib/compute/sunpath -> $lib/compute/core/sunpath
$lib/compute/tregenza -> $lib/compute/core/tregenza
$lib/compute/utci -> $lib/compute/core/utci
```

Expected for the future plan:

- No old core math imports remain in `viewer/src` or `viewer/tests`.

- [ ] **Step 4: Future plan gate - run core/parity-sensitive tests**

Run from `viewer/`:

```powershell
npx vitest run tests/compute/compute-manager.test.ts tests/compute/epw-parser.test.ts tests/compute/grid-generator.test.ts tests/compute/mrt-reference-vs-shader.test.ts tests/compute/mrt-utci-gpu-solarcal.test.ts tests/compute/solar-altitude-packing.test.ts tests/compute/solarcal.test.ts tests/compute/sun-vector-alignment.test.ts tests/compute/sunpath.test.ts tests/compute/tregenza.test.ts tests/compute/utci.test.ts tests/compute/utci-boundary-averaging.test.ts tests/compute/utci-domain-parity.test.ts tests/compute/webgpu-python-parity.test.ts
```

Expected for the future plan:

- PASS.
- Add app-visible parity or normal collector verification if any import movement touches parity-sensitive runtime wiring rather than only imports.

- [ ] **Step 5: Future plan gate - run selected-hour and build gates**

Run from `viewer/`:

```powershell
npm run test:quality:selected-hour
npm run test:e2e:selected-hour
npm run build
```

Expected for the future plan:

- PASS.

## Task 8: Future Roadmap Only - On-Demand State And Diagnostics

Do not execute this task in the current implementation pass. Convert it into a separate future implementation plan after selected-hour and GPU/core slices are stable.

**Files:**
- Move on-demand files only:
  - `viewer/src/lib/compute/onDemandDiagnostics.ts`
  - `viewer/src/lib/compute/onDemandOutputFormat.ts`
  - `viewer/src/lib/compute/onDemandPrototypeStatus.ts`
  - `viewer/src/lib/compute/onDemandScrubState.ts`
  - `viewer/src/lib/compute/onDemandSizing.ts`
- Update imports and exact `vi.mock(...)` specifiers in `viewer/src/**` and `viewer/tests/**`.

- [ ] **Step 1: Future plan gate - confirm core slice is green**

Run from `viewer/`:

```powershell
npm run test:quality:selected-hour
npm run test:e2e:selected-hour
npm run build
```

Expected for the future plan:

- PASS.

- [ ] **Step 2: Future plan action - move only the on-demand cluster**

Target:

```text
viewer/src/lib/compute/on-demand/onDemandDiagnostics.ts
viewer/src/lib/compute/on-demand/onDemandOutputFormat.ts
viewer/src/lib/compute/on-demand/onDemandPrototypeStatus.ts
viewer/src/lib/compute/on-demand/onDemandScrubState.ts
viewer/src/lib/compute/on-demand/onDemandSizing.ts
```

Expected for the future plan:

- No behavior edits.
- Diagnostics field names and proof-boundary semantics remain unchanged.

- [ ] **Step 3: Future plan action - update imports mechanically**

Replace exact import roots:

```text
$lib/compute/onDemandDiagnostics -> $lib/compute/on-demand/onDemandDiagnostics
$lib/compute/onDemandOutputFormat -> $lib/compute/on-demand/onDemandOutputFormat
$lib/compute/onDemandPrototypeStatus -> $lib/compute/on-demand/onDemandPrototypeStatus
$lib/compute/onDemandScrubState -> $lib/compute/on-demand/onDemandScrubState
$lib/compute/onDemandSizing -> $lib/compute/on-demand/onDemandSizing
```

Expected for the future plan:

- No old on-demand imports remain in `viewer/src` or `viewer/tests`.

- [ ] **Step 4: Future plan gate - run on-demand tests**

Run from `viewer/`:

```powershell
npx vitest run tests/compute/compute-manager-on-demand.test.ts tests/compute/onDemandDiagnostics.test.ts tests/compute/onDemandOutputFormat.test.ts tests/compute/onDemandScrubState.test.ts tests/compute/onDemandSizing.test.ts tests/lib/onDemandPrototypeStatus.test.ts tests/debug/debug-on-demand-prototype-diagnostics.test.ts
```

Expected for the future plan:

- PASS.

- [ ] **Step 5: Future plan gate - run selected-hour and debug gates**

Run from `viewer/`:

```powershell
npm run test:quality:selected-hour
npm run test:e2e:selected-hour
```

Expected for the future plan:

- PASS.

## Task 9: Future Roadmap Only - Weather

Do not execute this task in the current implementation pass. Convert it into a separate future implementation plan after the earlier slices are complete and verified.

**Files:**
- Move weather files only:
  - `viewer/src/lib/compute/epw-parser.ts`
  - `viewer/src/lib/compute/projectWeather.ts`
- Update imports in `viewer/src/**`, `viewer/tests/**`, and docs that mention weather path assumptions.

- [ ] **Step 1: Future plan gate - confirm on-demand slice is green**

Run from `viewer/`:

```powershell
npm run test:quality:selected-hour
npm run test:e2e:selected-hour
npm run build
```

Expected for the future plan:

- PASS.

- [ ] **Step 2: Future plan action - move only the weather cluster**

Target:

```text
viewer/src/lib/compute/weather/epw-parser.ts
viewer/src/lib/compute/weather/projectWeather.ts
```

Expected for the future plan:

- No behavior edits.
- Project/weather mapping remains unchanged for Ben-Gurion and Ness Tziona.

- [ ] **Step 3: Future plan action - update imports mechanically**

Replace exact import roots:

```text
$lib/compute/epw-parser -> $lib/compute/weather/epw-parser
$lib/compute/projectWeather -> $lib/compute/weather/projectWeather
```

Expected for the future plan:

- No old weather imports remain in `viewer/src` or `viewer/tests`.

- [ ] **Step 4: Future plan gate - run weather and selected-hour tests**

Run from `viewer/`:

```powershell
npx vitest run tests/compute/compute-manager.test.ts tests/compute/epw-parser.test.ts tests/compute/weather-index-alignment.test.ts tests/compute/live-selected-hour-route-host.test.ts
npm run test:quality:selected-hour
npm run test:e2e:selected-hour
npm run build
```

Expected for the future plan:

- PASS.

## Task 10: Future Roadmap Only - Root Leftovers And Final Compute Import Audit

Do not execute this task in the current implementation pass. Convert it into a separate future implementation plan after the previous compute organization slices land.

**Files:**
- Inspect:
  - `viewer/src/lib/compute/telemetry.ts`
  - `viewer/src/lib/compute/compute-manager.ts`
  - `viewer/src/lib/compute/**`
  - `viewer/src/**`
  - `viewer/tests/**`
- Modify only if the previous slices leave clear stale paths.

- [ ] **Step 1: Audit remaining compute root files**

Run from repo root:

```powershell
Get-ChildItem viewer\src\lib\compute -File | Select-Object -ExpandProperty Name
Get-ChildItem viewer\src\lib\compute -Directory | Select-Object -ExpandProperty Name
```

Expected for the future plan:

- Root files should be limited to intentional leftovers, currently likely `telemetry.ts`, `compute-manager.ts`, plus any colocated file that reviewers explicitly kept.

- [ ] **Step 2: Future plan action - decide whether to move `telemetry.ts` and `compute-manager.ts`**

If telemetry is only used by compute/gpu/selected-hour internals, either:

```text
Keep `viewer/src/lib/compute/telemetry.ts` at root as a shared compute utility.
```

or:

```text
Move to `viewer/src/lib/compute/diagnostics/telemetry.ts` only if that folder exists and all imports remain simple.
```

Expected:

- Do not create a diagnostics folder for a single file unless reviewers agree it improves clarity.
- Keep `compute-manager.ts` at root unless a dedicated plan proves a better facade/orchestration home and updates debug/selected-hour tests.

- [ ] **Step 3: Future plan action - run final import map**

Run:

```powershell
rg -n "\$lib/compute|src/lib/compute|compute/" viewer/src viewer/tests docs/webgpu_strategy_analysis.md > .compute-import-map.after.txt
```

Expected for the future plan:

- New imports reflect the planned buckets.
- `.compute-import-map.after.txt` is a temporary planning artifact and must not be committed.

- [ ] **Step 4: Future plan action - remove temporary import-map artifacts**

Run from repo root:

```powershell
Remove-Item .compute-import-map.before.txt, .compute-import-map.after.txt -ErrorAction SilentlyContinue
```

Expected for the future plan:

- Temporary import maps are gone.

- [ ] **Step 5: Future plan gate - run full organization verification**

Run from `viewer/`:

```powershell
npm run test:quality:selected-hour
npm run test:e2e:selected-hour
npm run build
npm run check
```

Then from repo root:

```powershell
git diff --check
git status --short
```

Expected for the future plan:

- Selected-hour quality: PASS.
- Selected-hour E2E: PASS.
- Build: PASS, allowing only existing documented warnings.
- Check: PASS or inherited static debt only; touched files must be clean.
- `git diff --check`: PASS.
- No generated Playwright state or temp import maps remain.

## Task 11: Results Documentation

**Files:**
- Create: `docs/superpowers/plans/2026-05-12-ness-tziona-red-test-and-compute-organization-results.md`
- Modify only if needed: `docs/webgpu_strategy_analysis.md`

- [ ] **Step 1: Write the results note**

Create `docs/superpowers/plans/2026-05-12-ness-tziona-red-test-and-compute-organization-results.md` with:

```markdown
# Ness Tziona Red Test And Compute Organization Results

Date: 2026-05-12

## Scope

This pass fixed the Ness Tziona main-route selected-hour E2E before reorganizing the selected-hour compute cluster. No commits or worktrees were created. Later compute slices remained roadmap-only.

## Root Cause

Write one paragraph naming the failing boundary and the evidence that proved it. Include the specific diagnostic fields, test name, and file/function where the bad state originated.

## Behavior Fix

Write one paragraph describing the minimal behavior change. State explicitly that the main route still does not read `.bin`, Python reference output, or debug parity state.

## Compute Organization

List each completed slice:

- Selected-hour orchestration: moved/not moved, final path, verification.
- GPU pipeline and shaders: roadmap-only, not moved in this pass.
- Core math: roadmap-only, not moved in this pass.
- On-demand diagnostics/state: roadmap-only, not moved in this pass.
- Weather: roadmap-only, not moved in this pass.
- Root leftovers: list intentional root files, including `compute-manager.ts` and `telemetry.ts`.

## Verification

- Focused Ness Tziona Playwright: record PASS/FAIL, test count, and runtime.
- `npm run test:quality:selected-hour`: record PASS/FAIL and test count.
- `npm run test:e2e:selected-hour`: record PASS/FAIL and Chromium test count.
- `npm run build`: record PASS/FAIL and any warnings.
- `npm run check`: record PASS/FAIL, error/warning counts, and whether touched files are clean.
- `git diff --check`: record PASS/FAIL.

## Subagent Reviews

- Selected-hour behavior review: record reviewer status and blockers, if any.
- Static/scope review before reorg: record reviewer status and blockers, if any.
- Import-boundary review after reorg: record reviewer status and blockers, if any.
- Debug parity/fallback review after reorg: record reviewer status and blockers, if any.

## Remaining Work

- Convert the roadmap-only GPU/shader, core math, on-demand, weather, and root-leftover sections into later separate implementation plans.
- Keep broader Python/scripts/data organization as a separate migration.
```

Do not leave generic filler in the final results note. Every verification bullet must contain the command outcome from the current run.

- [ ] **Step 2: Final verification**

Run from repo root:

```powershell
git status --short
git diff --stat
```

Expected:

- No generated Playwright state.
- Diff matches the behavior fix, selected-hour compute move, import/source-lock updates, roadmap-only plan text, and docs only.

## Final Completion Criteria

This plan is complete only when:

- The focused Ness Tziona Playwright test passes before any compute move.
- `npm run test:quality:selected-hour` passes.
- `npm run test:e2e:selected-hour` passes.
- `npm run build` passes.
- `git diff --check` passes.
- `npm run check` is either green or fails only with inherited static debt and no touched-file errors.
- Subagent reviewers report no blockers after the behavior fix and after the selected-hour reorg slice.
- Roadmap-only slices are reviewed for future-plan completeness but are not executed in this pass.
- Results are documented in the results markdown.

## Explicit Stop Conditions

Stop and report findings before continuing if:

- The Ness Tziona test becomes flaky instead of consistently passing.
- A proposed fix weakens `strongVisibleGpuPath` or hides a CPU readback.
- The main route needs `.bin`, Python, parity, or debug globals to pass.
- Any debug parity E2E regresses.
- A mechanical move touches more than the currently active slice.
- Three attempted behavior fixes fail.
- `npm run check` gains errors in touched files that are not clearly inherited.
