# Main Route Cold Publication Next Steps Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Choose the next cold initial render-publication optimization from existing `/` artifacts without keeping unused runtime code or repeating the prepared-layout mistake.

**Architecture:** This is a decision-and-planning pass, not a behavior-changing implementation. It converts the failed prepared-layout evidence into a ranked candidate note, applies an upper-bound falsifier before any code is written, and produces one follow-up implementation plan for the single candidate that can plausibly move the render-publication rAF/interval gate.

**Tech Stack:** Svelte 5, Three.js WebGPU renderer, WebGPU compute-buffer selected-hour transport, Vitest, Playwright collector artifacts.

---

## Non-Negotiables

- Do not create git worktrees.
- Do not commit.
- Preserve unrelated dirty/staged files.
- Proof surface is `/`, not `/debug`.
- Preserve `rendererBackend=webgpu`, `compute-buffer-selected-hour`, same-device compute/render proof, and `visibleSelectedHourReadbackCount=0`.
- Do not move load cost onto scrub.
- Do not implement lazy/background exposure fill.
- Do not tune smaller exposure chunks as the solution.
- Do not implement cooperative render-publication chunking/yielding.
- Do not keep unused runtime code. This plan should not edit production runtime files.
- If evidence is inconclusive, stop with an inconclusive decision note rather than writing an implementation plan.

## Current Evidence

The discarded prepared-layout attempt proved a useful negative result:

- `renderPublicationPreStorageMs` improved to about `515-517 ms`, meeting the `<600 ms` bucket target.
- The largest render-publication-overlapped rAF gap stayed around `1342 ms`, and the interval gap stayed around `1395 ms`.
- The largest gap began while the prepared layout was still building, so moving work into selected-hour result assembly did not make it non-blocking.
- Scene-side mesh creation is mostly typed-array work: position fill about `78 ms`, index fill about `127 ms`, and cell-to-point fill about `54 ms`.
- Render-owned storage wait still costs about `275 ms`, and queue drain still creates a later gap around `361 ms`.

The key falsifier: a candidate must plausibly remove about `550 ms` from the largest render-publication-overlapped rAF window to reach the `<800 ms` gate from a `1342 ms` starting point. A candidate smaller than that can be useful later, but it should not become the next performance-fix implementation by itself.

## File Structure

- Create: `docs/performance/main-route-cold-publication-next-steps.md`
  - Evidence note with parsed proof, upper-bound table, candidate decision, and stop/next-plan outcome.
- Modify: `docs/webgpu_strategy_analysis.md`
  - Keep the prepared-layout failure conclusion and link the current decision note/plan.
- Create only if Task 2 produces a clear winner: `docs/superpowers/plans/YYYY-MM-DD-main-route-<candidate>.md`
  - Follow-up implementation plan for the chosen candidate.

---

### Task 1: Parse Existing Artifacts and Compute Upper Bounds

**Files:**
- Create: `docs/performance/main-route-cold-publication-next-steps.md`
- Modify: `docs/webgpu_strategy_analysis.md`

- [ ] **Step 1: Parse the existing freeze-map artifact**

Run:

```powershell
node -e "const fs=require('fs'); const p='data/performance-results/main-route-exposure-and-raf-diagnostics.json'; const a=JSON.parse(fs.readFileSync(p,'utf8')); const nz=a.cases.find(c=>c.caseId.includes('ness-tziona')&&c.gridResolutionMeters===0.5); if(!nz) throw new Error('missing NZ 0.5m case'); const d=nz.raw.finalDiagnostics; const t=d.timings?.renderPublication?.renderPublicationTimeline||{}; const mesh=t.renderSurfaceMeshTrace||{}; const topRaf=nz.summary.gapOverlapSummary?.topRafGaps?.find(x=>(x.overlapRenderPublicationWindowLabels||[]).length>0); const topInterval=nz.summary.gapOverlapSummary?.topIntervalGaps?.find(x=>(x.overlapRenderPublicationWindowLabels||[]).length>0); console.log(JSON.stringify({sourceRoute:a.sourceRoute, rendererBackend:d.rendererBackend, surface:d.utciSurfaceSource, transport:d.baseRenderTransport, sameDevice:d.baseSameDeviceForComputeAndRender, visibleReadback:d.selectedHourRuntimeContract?.visibleSelectedHourReadbackCount, firstVisible:nz.summary.firstSelectedHourVisibleMs, pipelineFirstVisible:nz.summary.finalTimingBuckets?.pipelineFirstSelectedHourVisibleMs, topRenderRafDurationMs:topRaf?.durationMs, topRenderIntervalDurationMs:topInterval?.durationMs, preStorageMs:t.renderPublicationPreStorageMs, renderSurfaceMeshMs:t.renderSurfaceMeshMs, renderStorageInitWaitMs:t.renderStorageInitWaitMs, renderQueueDrainMs:t.renderQueueDrainMs, meshPositionFillMs:mesh.createIndexedGridSurfaceGeometryPositionFillMs, meshIndexFillMs:mesh.createIndexedGridSurfaceGeometryIndexFillMs, meshCellToPointMs:mesh.createComputeBufferSurfaceCellToPointAllocFillMs}, null, 2));"
```

Expected: output includes `/`, `webgpu`, `compute-buffer-selected-hour`, same-device `true`, visible readback `0`, and non-null render-publication gap details.

- [ ] **Step 2: Compute candidate upper bounds**

In the evidence note, compute a table like this using the parsed values:

```markdown
| Candidate | Measured bucket | Best-case remaining rAF gap if fully removed | Can clear <800 alone? |
| --- | ---: | ---: | --- |
| Static mesh typed-array construction | 268 ms | 1074 ms | No |
| Render-owned storage wait | 275 ms | 1067 ms | No |
| Copy queue drain | 379 ms | 963 ms | No |
| Static mesh + storage wait | 543 ms | 799 ms | Barely, but combines two mechanisms |
| Static mesh + queue drain | 647 ms | 695 ms | Plausible, but combines two mechanisms |
| Truly non-blocking static render asset preparation that removes adjacent layout/prep + mesh work | 800+ ms if proven | <800 possible | Candidate for follow-up plan |
```

Use the actual parsed values. Do not claim a candidate can pass if the arithmetic does not support it.

- [ ] **Step 3: Write the evidence note**

Create `docs/performance/main-route-cold-publication-next-steps.md` with these sections:

```markdown
# Main Route Cold Publication Next Steps

Updated: 2026-06-01

## Proof Boundary

- route: `/`
- renderer: `webgpu`
- surface/transport: `compute-buffer-selected-hour`
- same compute/render device: `true`
- visible selected-hour readback: `0`

## Prepared-Layout Attempt Outcome

Prepared layout inside selected-hour result assembly is ruled out as a performance fix. It improved `renderPublicationPreStorageMs` but did not reduce the largest rAF/interval gap because the synchronous layout build still ran immediately before scene publication.

## Upper-Bound Check

[candidate table from Step 2]

## Decision

[one of:]
- Proceed to a follow-up plan for true non-blocking static render asset preparation, because it is the smallest candidate that can plausibly remove enough adjacent work from the largest rAF gap.
- Stop as inconclusive, because no candidate has enough measured upside without combining unrelated mechanisms.

## Boundaries For The Follow-Up Plan

- Do not reintroduce prepared-layout runtime plumbing.
- Do not implement static mesh reuse alone as a claimed freeze fix unless the success gate is explicitly revised.
- Do not mix storage lifecycle and queue-drain behavior in the same task unless the implementation plan treats them as one coherent mechanism with one proof gate.
```

- [ ] **Step 4: Update the strategy doc**

Ensure `docs/webgpu_strategy_analysis.md` links to the evidence note and says static mesh reuse alone is too small to clear the current rAF gate. The next implementation plan should target a candidate whose measured upper bound can plausibly clear the gate.

- [ ] **Step 5: Dispatch reviews**

Dispatch a fresh spec-compliance review subagent for Task 1. Do not start code-quality review until spec compliance is clean. Then dispatch a fresh code-quality/maintainability review subagent for the documentation.

---

### Task 2: Write One Follow-Up Implementation Plan or Stop

**Files:**
- Create only if Task 1 has a clear winner: `docs/superpowers/plans/YYYY-MM-DD-main-route-<candidate>.md`
- Modify: `docs/performance/main-route-cold-publication-next-steps.md`

- [ ] **Step 1: Decide from Task 1**

If Task 1 says "Stop as inconclusive", update the evidence note with:

```markdown
## Outcome

Stopped before implementation planning. The existing artifacts do not identify a single scoped candidate with enough measured upside to plausibly clear the `<800 ms` rAF/interval gate.
```

Then stop the workflow.

- [ ] **Step 2: If there is a winner, write the follow-up implementation plan**

Use `superpowers:writing-plans` and create exactly one follow-up plan. If Task 1 selects true non-blocking static render asset preparation, name it:

```text
docs/superpowers/plans/2026-06-01-main-route-nonblocking-static-render-assets.md
```

The follow-up plan must include:

- tests before behavior changes
- no commits/worktrees
- no prepared-layout runtime path
- no static mesh reuse as a standalone claimed freeze fix
- exact proof boundary on `/`
- exact collector commands
- a remove-on-fail instruction if rAF/interval gates do not move

- [ ] **Step 3: Run perspective-ensemble on the follow-up plan**

Run both panels:

```markdown
## Panel A - Council
- Performance proof: concern -> flag -> counter-move
- Correctness/lifetime: concern -> flag -> counter-move
- Memory: concern -> flag -> counter-move
- Maintainability: concern -> flag -> counter-move

## Panel B - Adversarial
- Attack target: [chosen candidate]
- Hidden assumption: [specific assumption]
- Failure scenario: [how the plan can still produce cleaner buckets but same rAF gap]
- Mitigation/probe: [specific gate]
```

Patch the plan if the ensemble finds blockers.

- [ ] **Step 4: Dispatch reviews**

Dispatch a fresh spec-compliance review subagent for the follow-up plan. Do not start code-quality review until spec compliance is clean. Then dispatch a fresh code-quality/maintainability review subagent.

## Self-Review

- Spec coverage: The plan removes unused prepared-layout runtime/code artifacts, keeps the failed-attempt conclusion, and prevents a new unused implementation from being planned before an upper-bound falsifier.
- Placeholder scan: No `TBD`, `TODO`, or "write tests later" placeholders are intentionally left.
- Type consistency: This decision pass does not introduce runtime types.
- Execution mode: Use SDD with fresh subagents per task and two-stage review. Despite the writing-plans skill's generic template, this plan explicitly forbids commits/worktrees because the user requested that.

