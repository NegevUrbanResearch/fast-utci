# Main Route Selected-Hour Hot Publication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce Ness Tziona 0.5m post-visible scrub/month publication latency by fixing selected-hour render publication hot-path reuse and scene-sync start delay without changing scheduler defaults.

**Architecture:** Keep the compute path and cooperative exposure scheduler unchanged. First make render layout reuse deterministic for the first post-visible scrub, then identify and reduce the month-change scene-sync start delay. Use `/` main-route GPU-native diagnostics and the existing transition scrub collector as the before/after proof.

**Tech Stack:** Svelte 5, SvelteKit, Three.js WebGPU renderer, Playwright, Vitest, WebGPU selected-hour compute-buffer render path.

---

## Constraints

- Do not commit.
- Do not create git worktrees.
- Preserve unrelated dirty files.
- Chunked exposure scheduling remains query-gated; do not promote it to default.
- Proof surface is `/`, not `/debug`.
- Preserve GPU-native proof: `rendererBackend=webgpu`, `utciSurfaceSource=compute-buffer-selected-hour`, `baseRenderTransport=compute-buffer-selected-hour`, same compute/render device, no visible selected-hour readback fallback.
- Use `svelte-code-writer` before editing or analyzing `.svelte` files.

## Current Evidence

Baseline artifact: `data/performance-results/main-route-transition-scrub-diagnostics.json`.

Observed hot-path symptoms:

- Direct NZ single-submit hour 1: `firstSelectedHourVisibleMs=651`, `renderLayoutReuseAction=build-required`, `renderLayoutReuseReason=canonical-mismatch`.
- Direct NZ single-submit hour 2/3: `firstSelectedHourVisibleMs=117/118`, `renderLayoutReuseAction=reused`, `renderLayoutReuseReason=reuse-safe`.
- BG -> NZ single-submit hour 1: `firstSelectedHourVisibleMs=864`, `renderLayoutReuseReason=canonical-mismatch`.
- Month 8 in all cases: `firstSelectedHourVisibleMs=1907-2308`, dominated by `renderSceneSyncStartDelayMs=1860-2258`, while `renderSceneSyncTotalMs` is about `33`.
- `oneHourDispatchMs` stays small for hot interactions, about `11-21 ms`.

## Known Dirty Baseline To Preserve

These files are intentional uncommitted inputs from the diagnostic side quest. Do not delete, revert, or treat them as unrelated generated junk:

- `viewer/tests/e2e/main-route-transition-scrub-diagnostics.spec.ts`
- `viewer/tests/e2e/main-route-transition-scrub-diagnostics-source-lock.test.ts`
- `data/performance-results/main-route-transition-scrub-diagnostics.json`
- `data/performance-results/main-route-transition-scrub-diagnostics-progress.json`
- `docs/webgpu_strategy_analysis.md`

## File Map

- Modify: `viewer/src/lib/services/pointCloudService.ts`
  - Owns layout reuse keys, compatibility proof, and publication planning.
- Modify: `viewer/tests/services/pointCloudService.surface.test.ts`
  - Unit tests for layout reuse planning and proof safety.
- Modify: `viewer/src/lib/components/scene/UTCIPointCloud.svelte`
  - Owns accepted GPU-resident surface sync, scene-sync start timing, and render publication diagnostics.
- Modify: `viewer/tests/scene/utci-surface-sync.test.ts`
  - Existing scene-sync timing/reset tests.
- Modify: `viewer/tests/e2e/main-route-transition-scrub-diagnostics.spec.ts`
  - Add/adjust only narrow before/after assertions or extra fields if needed.
- Modify: `docs/webgpu_strategy_analysis.md`
  - Record final before/after result.

## Task 1: Prove And Fix First Post-Visible Layout Reuse

**Files:**
- Modify: `viewer/src/lib/services/pointCloudService.ts`
- Modify: `viewer/tests/services/pointCloudService.surface.test.ts`

- [ ] **Step 1: Add a failing unit test for safe reuse after initial publication**

Add a test near the existing `planUtciLayoutPublication` tests in `viewer/tests/services/pointCloudService.surface.test.ts`.

```ts
it('reuses the previous layout after initial publication when runtime compatibility is proven safe', () => {
	const analysis = createAnalysis({ sourceAnalysisId: 'Ness-Tziona/hot-path' });
	const previousLayout = buildUtciGridLayout(analysis);
	const currentKey = createReuseKey(analysis);
	const safeProof = buildUtciGridLayoutReuseProofDiagnostics({
		previousLayout,
		nextLayout: previousLayout,
		canonicalRuntimeCompatibilityWouldReuse: true,
		canonicalPointCompatibility: {
			compatible: true,
			cellToPointMappingMatch: true,
			requiredExpensiveMappingComparison: false,
			performedExpensiveMappingComparison: false
		}
	});

	expect(
		planUtciLayoutPublication({
			previousLayout,
			previousProof: safeProof,
			previousKey: currentKey,
			currentKey,
			currentSurfaceSource: 'compute-buffer-selected-hour',
			currentRendererBackend: 'webgpu',
			publicationPhase: 'scrub'
		})
	).toEqual({
		action: 'reuse-existing',
		layout: previousLayout,
		reason: 'reuse-safe',
		keyMatch: true
	});
});
```

- [ ] **Step 2: Run the focused test red/green check**

Run:

```bash
cd viewer
npx vitest run tests/services/pointCloudService.surface.test.ts
```

Expected before implementation: the new test should fail only if the current proof cannot be made safe for identical layouts. If it already passes, add the narrower failing test from Step 3 instead.

- [ ] **Step 3: Add a failing regression test for stale proof on first scrub**

If Step 1 already passes, add this test to prove the current `canonical-mismatch` class: when the previous proof was built before runtime compatibility was known, the planner should be able to refresh a full safe proof against the runtime-compatible existing mesh rather than forcing a rebuild. The refreshed proof must satisfy `isUtciLayoutReuseProofSafe()`; do not allow reuse from only a partial width/height/count check.

```ts
it('allows a runtime-compatible existing mesh to refresh stale initial proof for the first scrub', () => {
	const analysis = createAnalysis({ sourceAnalysisId: 'Ness-Tziona/hot-path' });
	const previousLayout = buildUtciGridLayout(analysis);
	const currentKey = createReuseKey(analysis);
	const staleInitialProof = buildUtciGridLayoutReuseProofDiagnostics({
		previousLayout: null,
		nextLayout: previousLayout,
		canonicalRuntimeCompatibilityWouldReuse: null
	});

	expect(
		planUtciLayoutPublication({
			previousLayout,
			previousProof: staleInitialProof,
			previousKey: currentKey,
			currentKey,
			currentSurfaceSource: 'compute-buffer-selected-hour',
			currentRendererBackend: 'webgpu',
			publicationPhase: 'scrub',
			refreshedProof: buildUtciGridLayoutReuseProofDiagnostics({
				previousLayout,
				nextLayout: previousLayout,
				canonicalRuntimeCompatibilityWouldReuse: true,
				canonicalPointCompatibility: {
					compatible: true,
					cellToPointMappingMatch: true,
					requiredExpensiveMappingComparison: false,
					performedExpensiveMappingComparison: false
				}
			})
		})
	).toEqual({
		action: 'reuse-existing',
		layout: previousLayout,
		reason: 'refreshed-proof-safe',
		keyMatch: true
	});
});
```

Also add negative tests:

```ts
it('does not reuse from refreshed proof when hover/cell safety is inconclusive', () => {
	const analysis = createAnalysis({ sourceAnalysisId: 'Ness-Tziona/hot-path' });
	const previousLayout = buildUtciGridLayout(analysis);
	const currentKey = createReuseKey(analysis);
	const refreshedProof = {
		...buildUtciGridLayoutReuseProofDiagnostics({
			previousLayout,
			nextLayout: previousLayout,
			canonicalRuntimeCompatibilityWouldReuse: true,
			canonicalPointCompatibility: {
				compatible: true,
				cellToPointMappingMatch: true,
				requiredExpensiveMappingComparison: false,
				performedExpensiveMappingComparison: false
			}
		}),
		hoverCellLookupProofStatus: 'proof-inconclusive' as const
	};

	expect(
		planUtciLayoutPublication({
			previousLayout,
			previousProof: null,
			previousKey: currentKey,
			currentKey,
			currentSurfaceSource: 'compute-buffer-selected-hour',
			currentRendererBackend: 'webgpu',
			publicationPhase: 'scrub',
			refreshedProof
		})
	).toEqual({ action: 'build-new', reason: 'proof-not-safe', keyMatch: true });
});
```

- [ ] **Step 4: Implement minimal planner support for full safe-proof refresh**

Update `planUtciLayoutPublication` in `viewer/src/lib/services/pointCloudService.ts` to accept an optional `refreshedProof` field and allow reuse when:

- publication phase is `scrub`
- current source/backend are `compute-buffer-selected-hour` / `webgpu`
- previous layout exists
- previous and current keys match
- `isUtciLayoutReuseProofSafe(refreshedProof) === true`

Do not relax safety for backend/source mismatch, layout-key mismatch, or missing previous layout.

Implementation shape:

```ts
export function planUtciLayoutPublication(params: {
	previousLayout: UtciGridLayout | null;
	previousProof: UtciGridLayoutReuseProofDiagnostics | null;
	previousKey: UtciLayoutReuseKey | null;
	currentKey: UtciLayoutReuseKey;
	currentSurfaceSource: string | null;
	currentRendererBackend: string | null;
	publicationPhase: 'initial' | 'scrub';
	refreshedProof?: UtciGridLayoutReuseProofDiagnostics | null;
}): UtciLayoutPublicationPlan {
	// keep existing initial and backend/source guards
	const candidate = planUtciLayoutReuseCandidate({
		previousLayout: params.previousLayout,
		proof: params.previousProof,
		previousKey: params.previousKey,
		currentKey: params.currentKey
	});
	if (candidate.action === 'reuse-candidate' && params.previousLayout) {
		return { action: 'reuse-existing', layout: params.previousLayout, reason: 'reuse-safe', keyMatch: true };
	}
	if (
		params.previousLayout &&
		params.previousKey &&
		candidate.keyMatch &&
		isUtciLayoutReuseProofSafe(params.refreshedProof)
	) {
		return {
			action: 'reuse-existing',
			layout: params.previousLayout,
			reason: 'refreshed-proof-safe',
			keyMatch: true
		};
	}
	return { action: 'build-new', reason: candidate.reason, keyMatch: candidate.keyMatch };
}
```

- [ ] **Step 5: Wire runtime compatibility into the Svelte publication path**

Use `svelte-code-writer` before editing.

In `viewer/src/lib/components/scene/UTCIPointCloud.svelte`, compute runtime layout compatibility before finalizing `layoutPublicationPlan` when the candidate needs proof refresh. Keep layout building bounded:

1. Build `currentKey`.
2. Call `planUtciLayoutPublication` without runtime compatibility.
3. If the plan is `build-new` with `reason === 'canonical-mismatch'` and `previousLayout` exists, build the next layout with diagnostics, evaluate `evaluateComputeBufferUtciSurfaceLayoutCompatibility`, build a refreshed proof from that compatibility, and call `planUtciLayoutPublication` again with `refreshedProof`.
4. If reuse is accepted, use `previousLayout` and do not create a new surface.

Do not add async waits or scheduler changes. Do not count this task as successful merely because the label changes to reuse: the recollected artifact must show that hour-1 `renderLayoutBuildTrace.totalMs`, `renderLayoutCompatibilityMs`, and `renderLayoutReuseDecisionMs` are materially reduced or that hour-1 visible timing moves toward the hour 2/3 band.

- [ ] **Step 6: Verify Task 1**

Run:

```bash
cd viewer
npx vitest run tests/services/pointCloudService.surface.test.ts
npm run check
```

Expected: all selected tests pass and `svelte-check` reports `0 errors and 0 warnings`.

## Task 2: Explain Month-Change Scene-Sync Start Delay

**Files:**
- Modify: `viewer/src/lib/components/scene/UTCIPointCloud.svelte`
- Modify: `viewer/tests/scene/utci-surface-sync.test.ts`
- Modify only if needed: `viewer/src/lib/compute/selected-hour/liveSelectedHourController.ts`

- [ ] **Step 1: Add failing diagnostics preservation coverage**

In `viewer/tests/scene/utci-surface-sync.test.ts` and, if needed, `viewer/tests/diagnostics/main-route-utci-diagnostics.test.ts`, first extend the relevant fixture/assertion to expect these fields to survive clone/merge/publish paths:

```ts
expect(result.renderPublicationTimeline?.sceneReactiveToSyncQueuedMs).toBe(12);
expect(result.renderPublicationTimeline?.sceneSyncQueuedToStartMs).toBe(34);
```

Run:

```bash
cd viewer
npx vitest run tests/scene/utci-surface-sync.test.ts tests/diagnostics/main-route-utci-diagnostics.test.ts
```

Expected: FAIL until the diagnostics fields are wired.

- [ ] **Step 2: Add diagnostics that split the pre-sync delay**

In `UTCIPointCloud.svelte`, preserve existing timing fields and add these timeline fields under `renderPublicationTimeline`:

```ts
sceneReactiveToSyncQueuedMs:
	sceneSyncInvocationQueuedAtMs != null && sceneReactiveBlockEnteredAtMs != null
		? Math.max(0, sceneSyncInvocationQueuedAtMs - sceneReactiveBlockEnteredAtMs)
		: undefined,
sceneSyncQueuedToStartMs:
	reactiveTiming?.sceneSyncInvocationQueuedAtMs != null
		? Math.max(0, sceneSyncAttemptStartedAtMs - reactiveTiming.sceneSyncInvocationQueuedAtMs)
		: undefined
```

These fields identify whether the month-8 delay happens before the reactive block queues sync, or between queueing and `syncAcceptedGpuResidentSurface` starting.

- [ ] **Step 3: Run focused tests**

Run:

```bash
cd viewer
npx vitest run tests/scene/utci-surface-sync.test.ts tests/diagnostics/main-route-utci-diagnostics.test.ts
```

Expected: tests pass after adding the fields to the relevant diagnostics type/clone helpers. If TypeScript identifies a missing type field, add it to the smallest existing diagnostics type that owns `renderPublicationTimeline`.

- [ ] **Step 4: Collect split-field evidence before choosing a fix**

Run the existing collector after the split fields are wired:

```bash
cd viewer
npx playwright test --config=playwright.collect.config.ts tests/e2e/main-route-transition-scrub-diagnostics.spec.ts --project=chromium --workers=1 --reporter=list --timeout=900000
cd ..
node -e "const fs=require('fs');const d=JSON.parse(fs.readFileSync('data/performance-results/main-route-transition-scrub-diagnostics.json','utf8'));for(const c of d.cases){const m8=c.samples.find(s=>s.actionLabel==='month-8');const tl=m8?.renderPublication?.timeline??{};console.log(c.caseId,{month8Visible:m8?.timings?.firstSelectedHourVisibleMs,startDelay:m8?.timings?.renderSceneSyncStartDelayMs,reactiveToQueued:tl.sceneReactiveToSyncQueuedMs,queuedToStart:tl.sceneSyncQueuedToStartMs});}"
```

Expected: the output identifies whether the month-8 delay is mostly `sceneReactiveToSyncQueuedMs` or `sceneSyncQueuedToStartMs`. Do not implement Step 5 or Step 6 without this evidence.

- [ ] **Step 5: If the split shows queued-to-start delay, avoid deferred duplicate sync calls**

Only implement this step if the diagnostic output shows `sceneSyncQueuedToStartMs` is the large bucket. In `UTCIPointCloud.svelte`, ensure the reactive block does not queue a duplicate sync while one for the same `acceptedSyncRunKey` is active. Use the existing `acceptedGpuResidentSurfaceSync.getActiveSyncRunKey()` and `lastObservedPendingSurface` state; do not introduce a separate scheduler.

- [ ] **Step 6: If the split shows reactive-to-queued delay, trace upstream publication**

Only implement this step if `sceneReactiveToSyncQueuedMs` is the large bucket. Add one narrowly scoped timestamp from route/controller publication to the render publication timeline, using existing `controllerStatePublishedAtMs`, `routePendingSurfaceExposedAtMs`, and `routeProjectedAtMs` fields first. Do not add a new global store.

- [ ] **Step 7: Verify Task 2**

Run:

```bash
cd viewer
npx vitest run tests/scene/utci-surface-sync.test.ts tests/diagnostics/main-route-utci-diagnostics.test.ts
npm run check
```

Expected: tests pass and `svelte-check` reports `0 errors and 0 warnings`.

## Task 3: Recollect Before/After Evidence And Update Docs

**Files:**
- Modify: `docs/webgpu_strategy_analysis.md`
- Existing artifact refreshed: `data/performance-results/main-route-transition-scrub-diagnostics.json`
- Existing progress artifact refreshed: `data/performance-results/main-route-transition-scrub-diagnostics-progress.json`

- [ ] **Step 1: Run the transition scrub collector**

Run:

```bash
cd viewer
npx playwright test --config=playwright.collect.config.ts tests/e2e/main-route-transition-scrub-diagnostics.spec.ts --project=chromium --workers=1 --reporter=list --timeout=900000
```

Expected: `1 passed`; artifact has four cases and seven samples per case.

- [ ] **Step 2: Parse the artifact and compare the target fields**

Run from the repository root:

```bash
node -e "const fs=require('fs');const d=JSON.parse(fs.readFileSync('data/performance-results/main-route-transition-scrub-diagnostics.json','utf8'));for(const c of d.cases){const h1=c.samples.find(s=>s.actionLabel==='hour-1');const m8=c.samples.find(s=>s.actionLabel==='month-8');console.log(c.caseId,{hour1Visible:h1?.timings?.firstSelectedHourVisibleMs,hour1Layout:h1?.layout?.renderLayoutReuseAction+':'+h1?.layout?.renderLayoutReuseReason,month8Visible:m8?.timings?.firstSelectedHourVisibleMs,month8Delay:m8?.timings?.renderSceneSyncStartDelayMs,month8Sync:m8?.timings?.renderSceneSyncTotalMs});}"
```

Expected target movement:

- Hour 1 should move from `build-required:canonical-mismatch` toward `reused:refreshed-proof-safe` or `reused:reuse-safe`.
- Hour 1 visible time should move toward the hour 2/3 band, ideally below `250 ms` on the tested machine.
- Month 8 should either improve or have a clear split showing the remaining delay source.

- [ ] **Step 3: Repeat the narrow target actions if the result is borderline**

If hour 1 or month 8 is within `25%` of the old baseline or contradicts manual feel, run the collector a second time and compare medians for the target samples rather than trusting one run. Do not require a second run if the first run shows a large clear movement and proof remains clean.

- [ ] **Step 4: Update `docs/webgpu_strategy_analysis.md`**

Add a short dated subsection under `2026-05-31 Transition Scrub Diagnostic`:

```md
#### Hot Publication Fix Follow-Up

After the selected-hour publication hot-path change, `main-route-transition-scrub-diagnostics.json` shows:

- Hour 1 layout result: ...
- Hour 1 visible timing: ...
- Month 8 scene-sync start delay: ...
- Remaining bottleneck: ...

Conclusion: ...
```

- [ ] **Step 5: Final verification**

Run:

```bash
cd viewer
npm run check
npx vitest run tests/services/pointCloudService.surface.test.ts tests/scene/utci-surface-sync.test.ts tests/diagnostics/main-route-utci-diagnostics.test.ts tests/e2e/main-route-transition-scrub-diagnostics-source-lock.test.ts
cd ..
git diff --check
git status --short --branch
```

Expected:

- `npm run check`: `0 errors and 0 warnings`
- Vitest: all listed tests pass
- `git diff --check`: no whitespace errors
- Working tree remains uncommitted, with only intended files changed plus generated evidence artifacts.
