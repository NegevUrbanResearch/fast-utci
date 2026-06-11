# Main Route Init Smoothness Attribution Implementation Plan

## Copy-Ready Prompt For The Next AI Agent

You are working in `D:\Projects\Nur\Shade\fast-utci`. Execute `docs/superpowers/plans/2026-06-01-main-route-init-smoothness-attribution.md` task by task.

Hard constraints:

- Do not create git worktrees.
- Do not commit.
- Preserve unrelated dirty/staged files.
- Use `/` as the proof route, not `/debug`.
- This is a diagnostics/attribution pass first. Do not implement a runtime optimization unless the plan is explicitly updated after evidence review.
- Preserve the GPU-native proof boundary: `rendererBackend=webgpu`, `compute-buffer-selected-hour`, same compute/render device, `visibleSelectedHourReadbackCount=0`, and `dataTextureBuildCount=0`.
- Do not tune smaller exposure chunks, implement lazy/background exposure fill, or reintroduce prepared-layout runtime plumbing.

Goal:

Build a complete init smoothness map for Ness Tziona `0.5m` on the main route. Separate total latency from visible freeze. Attribute the early startup/pre-exposure rAF/long-task gap, summarize the rAF/interval/long-task distribution instead of relying only on the maximum, and preserve render-publication and exposure overlap evidence.

Required execution shape:

1. Read this whole plan.
2. Run the existing artifact parser commands in Task 1 to verify the current evidence.
3. Add/extend diagnostics so the collector preserves all rAF gaps or enough histogram/distribution data for rAF, interval, and long-task evidence, plus phase marks before exposure starts.
4. Re-run the focused headed collector for `/` Ness Tziona `0.5m`.
5. Write `docs/performance/main-route-init-smoothness-attribution.md` with the ranked owner taxonomy and falsifiers.
6. Update `docs/webgpu_strategy_analysis.md` only after the artifact is collected and parsed.
7. Run the verification commands in the plan.
8. Dispatch a spec-compliance review subagent first. Do not start code-quality review until spec compliance is clean. Then dispatch code-quality/maintainability review.

If evidence is inconclusive, stop with an inconclusive note. Do not invent a fix.

---

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Attribute NZ `0.5m` main-route init smoothness pain across early startup, exposure breathing, and render publication before choosing an optimization.

**Architecture:** This plan extends diagnostics and docs, not product behavior. It treats rAF as a timestamped freeze detector, separates wall-clock latency from visible freeze, and ranks owners from evidence collected on `/`.

**Tech Stack:** Svelte 5, Playwright headed collectors, Three.js WebGPU renderer, WebGPU selected-hour diagnostics, Node artifact parsers, Markdown docs.

---

## Non-Negotiables

- Do not create git worktrees.
- Do not commit.
- Preserve unrelated dirty/staged files.
- Proof surface is `/`, not `/debug`.
- Preserve `rendererBackend=webgpu`, `compute-buffer-selected-hour`, same-device compute/render proof, `visibleSelectedHourReadbackCount=0`, and `dataTextureBuildCount=0`.
- Do not implement performance fixes in this pass.
- Do not tune smaller exposure chunks.
- Do not implement lazy/background exposure fill.
- Do not reintroduce prepared-layout runtime plumbing.
- Do not claim render publication is the main freeze until the early startup/pre-exposure gap is attributed.

## Current Evidence

- `data/performance-results/main-route-exposure-and-raf-diagnostics.json` has `rafGapCount=76` for `ness-tziona-0_5m`, but only stores top rAF gaps.
- Top overall rAF gap: `4314.5 -> 5663.2 ms`, `1348.7 ms`, no exposure or render-publication overlap.
- Top overall interval gap: `4284.3 -> 5669.5 ms`, `1385.2 ms`, no exposure or render-publication overlap.
- Top overall long task: `4310.3 -> 5666.3 ms`, `1356 ms`, no exposure or render-publication overlap.
- Largest render-publication rAF gap: `23613.8 -> 24927.9 ms`, `1314.1 ms`, overlapping publication/pre-storage/storage wait.
- Render-publication tail rAF gap: `24934.9 -> 25261.5 ms`, `326.6 ms`, overlapping queue drain.
- Exposure-overlapped rAF gaps are smaller, about `326-355 ms` around slices `27-32`.
- Exposure remains the total-latency owner at about `17.2 s`, but it is not proven to be the largest page-local rAF freeze owner.

## File Structure

- Modify: `viewer/tests/e2e/main-route-visual-freeze-map.spec.ts`
  - Preserve full rAF/interval/long-task lists or add compact distribution bins and phase marks needed for attribution.
- Modify if diagnostics helpers are split out: `viewer/tests/e2e/*visual-freeze*` or existing collector helper files found by `rg`.
  - Keep collector logic local to the existing collector surface; do not create a new runtime app path.
- Create: `docs/performance/main-route-init-smoothness-attribution.md`
  - Evidence note with proof boundary, rAF distribution, phase attribution, ranked owners, and next-step recommendation.
- Modify: `docs/webgpu_strategy_analysis.md`
  - Refresh only after the new artifact exists.
- Optional create: `viewer/scripts/summarize-main-route-init-smoothness.js`
  - Only if inline `node -e` parsing becomes too long to maintain. Keep it read-only and artifact-focused.

---

### Task 1: Verify Current Artifact And Baseline Assumptions

**Files:**
- Read: `data/performance-results/main-route-exposure-and-raf-diagnostics.json`
- Read: `docs/performance/main-route-exposure-and-raf-diagnostics.md`
- Read: `docs/webgpu_strategy_analysis.md`

- [ ] **Step 1: Parse the current NZ 0.5m gap evidence**

Run:

```powershell
@'
const fs=require('fs');
const p='data/performance-results/main-route-exposure-and-raf-diagnostics.json';
const a=JSON.parse(fs.readFileSync(p,'utf8'));
const nz=a.cases.find(c=>c.caseId==='ness-tziona-0_5m');
if(!nz) throw new Error('missing ness-tziona-0_5m');
const s=nz.summary;
const gs=s.gapOverlapSummary || {};
console.log(JSON.stringify({
  sourceRoute:a.sourceRoute,
  caseId:nz.caseId,
  firstVisible:s.firstSelectedHourVisibleMs,
  pipelineFirstVisible:s.finalTimingBuckets?.pipelineFirstSelectedHourVisibleMs,
  rafGapCount:s.rafGapCount,
  storedTopRafCount:(gs.topRafGaps||[]).length,
  intervalGapCount:s.intervalGapCount,
  storedTopIntervalCount:(gs.topIntervalGaps||[]).length,
  longTaskCount:s.longTaskCount,
  storedLongTaskCount:(gs.longTasks||[]).length,
  topRaf:gs.topRafGaps?.[0],
  topInterval:gs.topIntervalGaps?.[0],
  topLongTask:gs.longTasks?.[0],
  firstRenderRaf:(gs.topRafGaps||[]).find(g=>(g.overlapRenderPublicationWindowLabels||[]).length>0),
  firstExposureRaf:(gs.topRafGaps||[]).find(g=>(g.overlapExposureSliceCount||0)>0)
}, null, 2));
'@ | node -
```

Expected:

- `sourceRoute` is `/`.
- `rafGapCount` is greater than `storedTopRafCount`.
- The top overall rAF/interval/long-task windows have no exposure or render-publication overlap.
- The first render-overlapped rAF gap is about `1314 ms`.

- [ ] **Step 2: Confirm the current proof boundary**

Run:

```powershell
@'
const fs=require('fs');
const p='data/performance-results/main-route-exposure-and-raf-diagnostics.json';
const a=JSON.parse(fs.readFileSync(p,'utf8'));
const nz=a.cases.find(c=>c.caseId==='ness-tziona-0_5m');
const d=nz.raw.finalDiagnostics;
const proof=d.selectedHourRuntimeContract || {};
console.log(JSON.stringify({
  rendererBackend:d.rendererBackend,
  surface:d.utciSurfaceSource,
  transport:d.baseRenderTransport,
  sameDevice:d.baseSameDeviceForComputeAndRender,
  visibleReadback:proof.visibleSelectedHourReadbackCount,
  dataTextureBuildCount:d.dataTextureBuildCount ?? d.timings?.dataTextureBuildCount ?? null,
  pageErrors:nz.summary.pageErrorCount,
  requestFailures:nz.summary.requestFailureCount,
  crashes:nz.summary.crashCount
}, null, 2));
'@ | node -
```

Expected:

- `rendererBackend` is `webgpu`.
- `surface` and `transport` are `compute-buffer-selected-hour`.
- `sameDevice` is `true`.
- `visibleReadback` is `0`.
- `dataTextureBuildCount` is `0` or another explicit zero-valued field used by the current diagnostics contract.
- page errors, request failures, and crashes are `0`.

---

### Task 2: Extend The Collector To Preserve Attribution Data

**Files:**
- Modify: `viewer/tests/e2e/main-route-visual-freeze-map.spec.ts`
- Modify if needed: collector helper files found by `rg -n "topRafGaps|rafGapCount|gapOverlapSummary|requestAnimationFrame" viewer/tests viewer/src`

- [ ] **Step 1: Locate the collector gap recording code**

Run:

```powershell
rg -n "topRafGaps|rafGapCount|gapOverlapSummary|requestAnimationFrame|longTasks|PerformanceObserver" viewer/tests viewer/src
```

Expected:

- The output points to the headed collector and any helper that truncates or summarizes rAF gaps.

- [ ] **Step 2: Preserve frame-gap and task distribution fields**

Change the collector so the artifact can answer all of these without re-running browser traces:

- total rAF gap count
- stored rAF gap count
- all rAF gaps above `50 ms`, or all rAF gaps if the list is small enough
- rAF gap histogram buckets: `>50 ms`, `>100 ms`, `>250 ms`, `>500 ms`, `>1000 ms`
- total duration in each bucket
- top rAF gaps with `startMs`, `endMs`, `durationMs`, exposure overlaps, render-publication overlaps, and nearest phase labels
- interval gap histogram buckets: `>50 ms`, `>100 ms`, `>250 ms`, `>500 ms`, `>1000 ms`
- long-task histogram buckets: `>50 ms`, `>100 ms`, `>250 ms`, `>500 ms`, `>1000 ms`
- top interval gaps and long tasks with the same overlap labels where the collector can compute them

Use these stable artifact paths under `summary.gapOverlapSummary`:

```ts
{
  rafGapStoredCount: number,
  intervalGapStoredCount: number,
  longTaskStoredCount: number,
  rafGapDistribution: {
    over50Ms: { count: number, totalDurationMs: number },
    over100Ms: { count: number, totalDurationMs: number },
    over250Ms: { count: number, totalDurationMs: number },
    over500Ms: { count: number, totalDurationMs: number },
    over1000Ms: { count: number, totalDurationMs: number }
  },
  intervalGapDistribution: {
    over50Ms: { count: number, totalDurationMs: number },
    over100Ms: { count: number, totalDurationMs: number },
    over250Ms: { count: number, totalDurationMs: number },
    over500Ms: { count: number, totalDurationMs: number },
    over1000Ms: { count: number, totalDurationMs: number }
  },
  longTaskDistribution: {
    over50Ms: { count: number, totalDurationMs: number },
    over100Ms: { count: number, totalDurationMs: number },
    over250Ms: { count: number, totalDurationMs: number },
    over500Ms: { count: number, totalDurationMs: number },
    over1000Ms: { count: number, totalDurationMs: number }
  },
  allRafGaps?: Array<{ startMs: number, endMs: number, durationMs: number }>
}
```

If storing full rAF gaps is reasonable, add `allRafGaps` and keep existing top fields for compatibility. If full storage is too noisy, the distribution fields are required and `allRafGaps` can be omitted.

- [ ] **Step 3: Preserve early phase marks before exposure**

Add or preserve phase marks that bracket the current unattributed early freeze:

- route navigation started
- analysis selection started/resolved
- model or payload load started/resolved, if already observable
- payload preparation started/resolved, if already observable
- worker/BVH request started/resolved, if already observable
- static upload started/resolved
- controller session run started
- first exposure slice started

Use existing diagnostics fields when available. Do not add product-runtime behavior solely for this collector. If a phase cannot be observed from existing runtime diagnostics, record it as `null` and document the gap.

Use this stable artifact path under `summary.gapOverlapSummary`:

```ts
{
  startupPhaseMarks: {
    routeNavigationStartedMs: number | null,
    analysisSelectionStartedMs: number | null,
    analysisSelectionResolvedMs: number | null,
    modelOrPayloadLoadStartedMs: number | null,
    modelOrPayloadLoadResolvedMs: number | null,
    payloadPreparationStartedMs: number | null,
    payloadPreparationResolvedMs: number | null,
    workerBvhStartedMs: number | null,
    workerBvhResolvedMs: number | null,
    staticUploadStartedMs: number | null,
    staticUploadResolvedMs: number | null,
    controllerSessionRunStartedMs: number | null,
    firstExposureSliceStartedMs: number | null
  }
}
```

- [ ] **Step 4: Add assertions/source-locks for the new artifact shape**

Update the existing collector source-lock test or add focused assertions in the collector test so stale artifacts cannot silently drop:

- `rafGapCount`
- `gapOverlapSummary.rafGapStoredCount`
- `rafGapDistribution`
- `intervalGapDistribution`
- `longTaskDistribution`
- `startupPhaseMarks`
- early startup/pre-exposure phase marks
- exposure slice windows
- render-publication windows

Do not require exact timing values in source-lock tests.

---

### Task 3: Recollect The Focused Main-Route Artifact

**Files:**
- Modify via collector output: `data/performance-results/main-route-visual-freeze-map.json`
- Modify via collector output if configured: `data/performance-results/main-route-exposure-and-raf-diagnostics.json`

- [ ] **Step 1: Run fast static checks before the headed collector**

Run:

```powershell
cd viewer
npm test -- --run tests/e2e/main-route-exposure-scheduler-collectors-source-lock.test.ts
npm run check
```

Expected:

- Vitest source-lock tests pass.
- `svelte-check found 0 errors and 0 warnings`.

- [ ] **Step 2: Run the headed collector**

Run:

```powershell
cd viewer
npx playwright test --config=playwright.collect.config.ts tests/e2e/main-route-visual-freeze-map.spec.ts --project=chromium --workers=1 --reporter=list --timeout=900000
```

Expected:

- The headed Chromium collector passes.
- The artifact includes `ness-tziona-0_5m`.
- The artifact keeps the GPU-native proof boundary.

- [ ] **Step 3: Parse the refreshed artifact**

Run the Task 1 parser again against the refreshed artifact path. Confirm:

- all required proof fields remain valid
- the early startup/pre-exposure gap has either a named owner or a documented unknown window
- the rAF distribution is available, not just the max
- render-publication and exposure overlap labels are still present

---

### Task 4: Write The Attribution Evidence Note

**Files:**
- Create: `docs/performance/main-route-init-smoothness-attribution.md`

- [ ] **Step 1: Create the evidence note**

Write the file with this structure:

```markdown
# Main Route Init Smoothness Attribution

Updated: 2026-06-01

## Scope

- Route: `/`
- Decision case: `ness-tziona-0_5m`
- Goal: separate total latency from visible freeze during initial load.

## Proof Boundary

[renderer, surface, transport, same-device, visible readback, dataTexture, page errors]

## What rAF Means Here

rAF gaps are page-local frame starvation windows. They are useful for freeze attribution, but they are not the same as total first-visible latency.

## Gap Distribution

[histogram and top gaps]

## Phase Attribution

[early startup/pre-exposure, exposure, render publication, queue-drain tail]

## Ranked Owner Taxonomy

[rank total-latency owners separately from visible-freeze owners]

## What Is App-Owned vs Workload/HW-Limited

[classify with confidence and falsifiers]

## Recommendation

[diagnostics/implementation next step, or stop as inconclusive]

## Verification

[commands and results]
```

- [ ] **Step 2: Keep the taxonomy honest**

Use these labels exactly:

- `total-latency owner`
- `visible-freeze owner`
- `app-owned`
- `workload/HW-limited`
- `inconclusive`

Do not describe a phase as a root cause if the artifact only proves overlap.

---

### Task 5: Refresh The Strategy Doc

**Files:**
- Modify: `docs/webgpu_strategy_analysis.md`

- [ ] **Step 1: Update the init smoothness section**

Refresh the `2026-06-01 Init Smoothness And rAF Attribution Correction` section with:

- link to `docs/performance/main-route-init-smoothness-attribution.md`
- refreshed top gaps and distribution summary
- updated owner ranking
- clear next-step recommendation

- [ ] **Step 2: Preserve older conclusions as bounded**

Keep these boundaries:

- render publication is a proven app-owned freeze, not necessarily the only or largest freeze
- exposure remains the largest total-latency owner
- selected-hour UTCI dispatch is not the current init bottleneck
- visible selected-hour readback and `DataTexture` rebuilds remain ruled out by proof

---

### Task 6: Review And Verify

**Files:**
- Review: `docs/performance/main-route-init-smoothness-attribution.md`
- Review: `docs/webgpu_strategy_analysis.md`
- Review: collector/test changes

- [ ] **Step 1: Run final verification commands**

Run:

```powershell
cd viewer
npm test -- --run tests/e2e/main-route-exposure-scheduler-collectors-source-lock.test.ts
npm run check
```

Then from repo root:

```powershell
git diff --check
```

Expected:

- Vitest passes.
- Svelte check passes.
- `git diff --check` has no errors. LF/CRLF warnings are acceptable only if they are pre-existing line-ending warnings.

- [ ] **Step 2: Run perspective-ensemble on the result**

Use both panels:

```markdown
## Panel A - Council
- UX smoothness: concern -> flag -> counter-move
- Performance attribution: concern -> flag -> counter-move
- App/HW boundary: concern -> flag -> counter-move
- Maintainability: concern -> flag -> counter-move

## Panel B - Adversarial
- Attack target: the attribution note and next-step recommendation
- Hidden assumption: [specific assumption]
- Failure scenario: [how the note can still mislead the next optimization]
- Mitigation/probe: [specific metric or collector change]
```

Patch the docs/plan if the ensemble finds a blocker.

- [ ] **Step 3: Dispatch review subagents**

Dispatch a fresh spec-compliance review subagent first. Ask it:

```text
Review docs/superpowers/plans/2026-06-01-main-route-init-smoothness-attribution.md, docs/performance/main-route-init-smoothness-attribution.md, and docs/webgpu_strategy_analysis.md for compliance with the requested scope: no commits, no worktrees, diagnostics/attribution first, main route proof, no optimization until evidence, and a copy-ready next-agent prompt at the top of the plan. Return blockers first.
```

Only after spec compliance is clean, dispatch a fresh code-quality/maintainability review subagent. Ask it:

```text
Review the collector/test/doc changes for maintainability and risk. Focus on whether the artifact schema is clear, whether the attribution labels avoid overclaiming causality, and whether the plan is executable by a fresh agent without chat context. Return blockers first.
```

- [ ] **Step 4: Stop without committing**

Do not commit. Leave the working tree ready for the user to inspect.

## Self-Review

- Spec coverage: The plan captures rAF distribution, early startup attribution, exposure breathing, render publication, total latency vs visible freeze, app-owned vs workload/HW-limited classification, no commits, no worktrees, subagent review, and perspective review.
- Placeholder scan: The plan contains no `TBD`, `TODO`, or "write tests later" placeholders.
- Type consistency: Artifact terms are consistent: `rafGapDistribution`, `allRafGaps`, `gapOverlapSummary`, `visibleSelectedHourReadbackCount`, and `dataTextureBuildCount`.
- Execution mode: This is a diagnostics/attribution plan. Runtime optimization is explicitly out of scope until evidence is reviewed.
