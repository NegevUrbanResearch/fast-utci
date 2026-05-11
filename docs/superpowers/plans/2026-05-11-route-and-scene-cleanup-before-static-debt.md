# Route And Scene Cleanup Before Static Debt Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Workflow overrides:** Do **not** create commits. Do **not** create git worktrees. Preserve unrelated dirty files. This plan intentionally defers broad inherited `npm run check` cleanup until after route decomposition, scene sync extraction, and folder/module organization. Fix only local type/static issues that directly block a planned extraction.

**Goal:** Decompose the main and debug routes, then extract selected-hour scene sync safely, before doing broader folder/module organization or inherited repo-wide static-debt cleanup.

**Architecture:** Treat the selected-hour lifecycle closure as the stable contract layer. First split route responsibilities in place, preserving current route behavior and diagnostics. Then extract the scene sync state machine with a mounted late-old-instance async-copy regression test. Only after those boundaries exist should folder moves happen; repo-wide `npm run check` debt remains a separate later cleanup lane.

**Tech Stack:** SvelteKit, Svelte components/reactivity, TypeScript, Three/Threlte WebGPU renderer, WebGPU UTCI compute, Vitest, Playwright Chromium with `--enable-unsafe-webgpu`, PowerShell on Windows.

---

## Discussion Capture

This plan captures the sequencing discussion after `docs/superpowers/plans/2026-05-11-selected-hour-lifecycle-closure-before-broader-cleanup-results.md`.

What we clarified:

- The selected-hour lifecycle closure is complete enough to unlock broader cleanup.
- The next architectural target is **route decomposition**, not broad static-debt cleanup.
- Route decomposition means **both** routes:
  - main product route: `viewer/src/routes/+page.svelte`
  - debug/proof route: `viewer/src/routes/debug/+page.svelte`
- The debug route is huge and must be decomposed too, but conservatively because it carries parity, `.bin`, collect, legacy debug selected-hour, shared-host selected-hour, and diagnostic proof behavior.
- The residual lifecycle risk belongs to the later scene sync extraction phase: there is still no mounted Svelte behavior test for a late old-instance async copy completion/failure after props swap.
- Folder reorganization should not start first. Moving files before extracting real route/scene boundaries creates noisy import churn and makes behavior regressions harder to review.
- `npm run check` still fails from inherited repo-wide static debt. The latest observed failure was 163 errors and 4 warnings in 34 files. That debt is real, but it should not lead this cleanup sequence.
- Local static/type issues may be fixed if they directly block a planned extraction, but the broad 163-error cleanup should be a separate later plan.

## Current State Snapshot

- `viewer/src/routes/+page.svelte` is about 899 lines.
- `viewer/src/routes/debug/+page.svelte` is about 4593 lines.
- `docs/superpowers/plans/2026-05-11-selected-hour-lifecycle-closure-before-broader-cleanup-results.md` reports:
  - `npm run test:quality:selected-hour` PASS: 18 files / 158 tests.
  - `npm run test:e2e:selected-hour` PASS: 13 Chromium tests.
  - `npm run build` PASS.
  - `npm run check` FAIL from inherited static debt.
  - `git diff --check` PASS.
- Fresh review after that note also reran:
  - `cd viewer; npm run test:quality:selected-hour` PASS: 18 files / 158 tests.
  - `cd viewer; npm run test:e2e:selected-hour` PASS: 13 Chromium tests.
  - `git diff --check` PASS.
- Review agents found no blocking lifecycle issues.
- Review agents agreed on the remaining non-blocking residual risk: no mounted scene behavior test for late old-instance async copy completion/failure.

## Non-Goals

- Do not start a broad `npm run check` cleanup.
- Do not try to make all 163 inherited static errors green in this plan.
- Do not optimize 0.5m performance.
- Do not remove `dataTexture`, `.bin`, Python comparison, parity, collect, debug fallback, or legacy debug selected-hour paths.
- Do not loosen `strongVisibleGpuPath`.
- Do not change selected-hour lifecycle ownership semantics except where tests prove an extraction preserved them.
- Do not create commits.
- Do not create git worktrees.

## Quality Gates

Stop and report findings before continuing if any of these happen:

- `test:quality:selected-hour` fails.
- `test:e2e:selected-hour` fails.
- Main route starts importing or emitting debug-only `.bin`, parity, or Python comparison behavior.
- Debug route loses default August Python `.bin` parity behavior.
- Debug route claims non-August Python `.bin` validity.
- Debug route shared-host selected-hour path regresses to legacy dispatch in normal non-parity f32 mode.
- Debug route normal collect mode stops publishing `__normalUtciResults__`, claims Python `.bin` validity, or stops using the legacy debug path.
- `strongVisibleGpuPath` becomes true without compute-buffer transport/source, same-device proof, selected request/selection matching, and zero visible-readback proof.
- Route decomposition requires broad static-debt cleanup outside touched route/support files.
- Scene sync extraction begins before the mounted late-old-instance async-copy behavior test exists.
- Folder moves start before route decomposition and scene sync extraction establish stable module boundaries.

## File Structure Target

### Phase 1: Main Route Decomposition In Place

- `viewer/src/routes/+page.svelte`
  - Remains the page composition shell.
  - Should mostly wire stores, route props, and components.
- `viewer/src/routes/mainRouteLiveSelectedHour.ts`
  - Owns main-route selected-hour host wiring, release forwarding, diagnostics callbacks, and route-to-scene selected-hour state projection.
- `viewer/src/routes/mainRouteModelSelection.ts`
  - Owns initial analysis id, URL-driven analysis changes, project selection, and model load bookkeeping.
- `viewer/src/routes/mainRouteTooltip.ts`
  - Owns tooltip state, hover sampling, interaction suppression, and comparison-curtain hover routing.
- Existing support files remain:
  - `viewer/src/routes/mainRouteOverlayGating.ts`
  - `viewer/src/lib/diagnostics/mainRouteUtciDiagnostics.ts`
  - `viewer/src/lib/compute/liveSelectedHourRouteHost.ts`
  - `viewer/src/lib/compute/liveSelectedHourRouteProjection.ts`

### Phase 2: Debug Route Decomposition In Place

- `viewer/src/routes/debug/+page.svelte`
  - Remains the debug page composition shell.
  - Should keep visual markup and high-level wiring.
- `viewer/src/routes/debug/debugRouteQueryState.ts`
  - If existing query parsing is not already fully extracted, owns debug query state derivation and default policy wiring.
  - Must preserve existing `parseDebugWebgpuUtciQuery` behavior.
- `viewer/src/routes/debug/debugRouteSelectedHourMode.ts`
  - Owns shared-host vs legacy selected-hour mode decision wiring.
  - Must preserve parity/collect/strict-exposure/one-hour comparison on legacy debug paths.
- `viewer/src/routes/debug/debugRouteSharedHostWiring.ts`
  - Owns debug shared-host route-host setup, release forwarding, and selected-hour diagnostics.
- `viewer/src/routes/debug/debugRouteLegacySelectedHourWiring.ts`
  - Owns legacy selected-hour dispatch, lifecycle release forwarding, deferred CPU fallback, and legacy counters.
- `viewer/src/routes/debug/debugRouteParityRuntime.ts`
  - Owns Python `.bin` parity metadata validity, parity intermediates publication, and debug-only parity diagnostics.
- Existing support files remain authoritative:
  - `viewer/src/lib/debug/debugWebgpuUtciQuery.ts`
  - `viewer/src/lib/debug/debugSelectedHourMode.ts`
  - `viewer/src/lib/debug/debugSelectedHourLegacyHost.ts`
  - `viewer/src/lib/debug/debugWebgpuUtciDiagnostics.ts`
  - `viewer/src/lib/debug/debugOnDemandPrototypeDiagnostics.ts`

### Phase 3: Scene Sync Extraction

- `viewer/src/lib/components/scene/UTCIPointCloud.svelte`
  - Keeps scene-specific mesh composition.
- `viewer/src/lib/components/scene/ComparisonRenderer.svelte`
  - Keeps comparison scene-specific mesh composition.
- `viewer/src/lib/components/scene/acceptedGpuResidentOutputRelease.ts`
  - Remains the exactly-once release notifier.
- `viewer/src/lib/components/scene/acceptedGpuResidentSurfaceSync.ts`
  - New extracted sync state machine/helper, created only after mounted behavior tests exist.
- `viewer/tests/scene/acceptedGpuResidentSurfaceSync.behavior.test.ts`
  - New mounted or integration-style behavior test for late old-instance async copy completion/failure after props swap.

### Phase 4: Folder/Module Organization

Move files only after the above boundaries are proven. Candidate organization:

- `viewer/src/routes/main/`
  - main route helpers currently beside `+page.svelte`.
- `viewer/src/routes/debug/`
  - debug route helpers beside `debug/+page.svelte` or under `debug/selectedHour/` if the helper count justifies it.
- `viewer/src/lib/components/scene/selectedHour/`
  - selected-hour scene sync helpers if extraction produces multiple related files.

### Phase 5: Broad Static-Debt Cleanup

Separate future plan. Likely families from the latest `npm run check` failure:

- `ArrayBufferLike` vs `ArrayBuffer` in worker/parity loaders.
- Reference type narrowing for `SolarReference | SkyReference | MrtReference`.
- Three `Object3D` narrowing for `isMesh`, `isLine`, and `isLineSegments`.
- Service/test fixture shape drift for UTCI data and grid layout.
- `SunPath.svelte` store shape drift.
- `Model.svelte` Svelte warning / stale `@ts-expect-error`.
- Debug parity intermediate array/null typing.

## Task 0: Baseline And Scope Lock

**Files:**
- Inspect only.

- [ ] **Step 1: Record dirty state**

Run from repo root:

```powershell
git status --short
git log --oneline -6
```

Expected:

- Preserve unrelated dirty files.
- Do not create commits.
- Do not create git worktrees.

- [ ] **Step 2: Run selected-hour quality baseline**

Run:

```powershell
cd viewer
npm run test:quality:selected-hour
```

Expected:

- PASS.
- Record file/test counts in the eventual result note.
- If this fails, stop and use `superpowers:systematic-debugging` before editing.

- [ ] **Step 3: Run selected-hour browser baseline**

Run:

```powershell
cd viewer
npm run test:e2e:selected-hour
```

Expected:

- PASS with 13 Chromium tests.
- Main route proves selected-hour diagnostics, hour/month changes, strong visible GPU path, comparison readbacks accounted separately, and Ness Tziona live range.
- Debug route proves shared-host behavior and parity `.bin` scoping.

- [ ] **Step 4: Confirm inherited static debt remains out of scope**

Run:

```powershell
cd viewer
npm run check
```

Expected:

- Likely FAIL from inherited repo-wide static debt.
- Record the current count.
- Do not fix broad static-debt families in this plan.
- If new errors appear in files touched by this plan during later tasks, fix those local errors only.

## Task 1: Main Route Characterization Before Extraction

**Files:**
- Modify: `viewer/tests/routes/main-route-debug-boundary-source-lock.test.ts`
- Modify or create: `viewer/tests/routes/main-route-live-selected-hour-source-lock.test.ts`
- Inspect: `viewer/src/routes/+page.svelte`

- [ ] **Step 1: Add source-level guard for main-route helper boundaries**

Create or update `viewer/tests/routes/main-route-live-selected-hour-source-lock.test.ts` with checks that the future main-route helper files stay free of debug-only behavior.

Use this exact structure:

```ts
import { describe, expect, it } from 'vitest';
import { readFileSync, existsSync } from 'node:fs';
import { resolve } from 'node:path';

const repoRoot = resolve(__dirname, '../..');
const optionalCandidatePaths = [
	'src/routes/+page.svelte',
	'src/routes/mainRouteLiveSelectedHour.ts',
	'src/routes/mainRouteModelSelection.ts',
	'src/routes/mainRouteTooltip.ts'
];

const debugOnlyPatterns = [
	/\.bin/i,
	/\bparity\b/i,
	/Python/i,
	/loadReferenceFromFs/i,
	/__onDemandPrototypeDiagnostics__/i,
	/LEGACY_DEBUG_SELECTED_HOUR_CONTROLLER_ID/i
];

function readIfPresent(relativePath: string): string | null {
	const absolutePath = resolve(repoRoot, relativePath);
	if (!existsSync(absolutePath)) return null;
	return readFileSync(absolutePath, 'utf8');
}

describe('main route decomposition source locks', () => {
	for (const relativePath of optionalCandidatePaths) {
		it(`${relativePath} remains free of debug-only selected-hour behavior when present`, () => {
			const source = readIfPresent(relativePath);
			if (source == null) return;
			for (const pattern of debugOnlyPatterns) {
				expect(source, `${relativePath} matched ${pattern}`).not.toMatch(pattern);
			}
		});
	}
});
```

- [ ] **Step 2: Run the source lock test**

Run:

```powershell
cd viewer
npx vitest run tests/routes/main-route-debug-boundary-source-lock.test.ts tests/routes/main-route-live-selected-hour-source-lock.test.ts
```

Expected:

- PASS.
- This locks the main route away from debug-only parity/bin behavior before extraction begins.
- In Task 2 and Task 3, convert any helper that has been created from optional to required by adding an existence assertion for that path. By the end of Task 3, `mainRouteLiveSelectedHour.ts`, `mainRouteModelSelection.ts`, and `mainRouteTooltip.ts` must be required by this source-lock test.

## Task 2: Extract Main Route Selected-Hour Wiring

**Files:**
- Create: `viewer/src/routes/mainRouteLiveSelectedHour.ts`
- Modify: `viewer/src/routes/+page.svelte`
- Modify: `viewer/tests/diagnostics/main-route-utci-diagnostics.test.ts`
- Modify: `viewer/tests/routes/main-route-live-selected-hour-source-lock.test.ts`

- [ ] **Step 1: Write a helper-level test for release forwarding and diagnostics shape**

Create focused tests only after inspecting the current route wiring. The test should prove:

- base release forwards `controllerInstanceId`.
- comparison release forwards `controllerInstanceId`.
- diagnostics payloads do not contain `.bin`, parity, Python, or debug global fields.

Recommended file:

```text
viewer/tests/routes/main-route-live-selected-hour.test.ts
```

Expected test names:

```ts
it('forwards base and comparison accepted GPU releases with controller instance ids', () => {});
it('builds main route diagnostics without debug-only parity fields', () => {});
```

- [ ] **Step 2: Run the failing helper test**

Run:

```powershell
cd viewer
npx vitest run tests/routes/main-route-live-selected-hour.test.ts
```

Expected:

- FAIL because `mainRouteLiveSelectedHour.ts` does not exist yet.

- [ ] **Step 3: Extract selected-hour route-host wiring**

Move the main-route responsibilities currently clustered near the top of `viewer/src/routes/+page.svelte` into `viewer/src/routes/mainRouteLiveSelectedHour.ts`:

- `createLiveSelectedHourRouteHost(...)` setup wrapper.
- base/comparison accepted GPU release handlers.
- renderer diagnostics callback shape.
- UTCI surface diagnostics callback shape.
- route-host state projection into scene props.
- main-route diagnostics update function inputs.

Do not move:

- markup.
- model loading.
- tooltip/raycast code.
- comparison curtain markup.
- debug-only behavior.

Keep the page import surface explicit and boring. The route should read like:

```ts
const liveSelectedHour = createMainRouteLiveSelectedHourController({
	getRendererDevice: () => rendererDeviceForMain,
	getRendererBackend: () => rendererBackend,
	updateDiagnostics: updateUtciRenderDiagnostics
});
```

Use the actual local names once the extraction is performed; avoid new architecture beyond this wrapper.

- [ ] **Step 4: Re-run focused tests**

Run:

```powershell
cd viewer
npx vitest run tests/routes/main-route-live-selected-hour.test.ts tests/diagnostics/main-route-utci-diagnostics.test.ts tests/routes/main-route-debug-boundary-source-lock.test.ts tests/routes/main-route-live-selected-hour-source-lock.test.ts
```

Expected:

- PASS.
- No debug-only parity/bin behavior enters the main route or extracted main-route helper.
- `viewer/tests/routes/main-route-live-selected-hour-source-lock.test.ts` now requires `viewer/src/routes/mainRouteLiveSelectedHour.ts` to exist.

- [ ] **Step 5: Run selected-hour quality**

Run:

```powershell
cd viewer
npm run test:quality:selected-hour
```

Expected:

- PASS.

## Task 3: Extract Main Route Model And Tooltip Helpers

**Files:**
- Create: `viewer/src/routes/mainRouteModelSelection.ts`
- Create: `viewer/src/routes/mainRouteTooltip.ts`
- Modify: `viewer/src/routes/+page.svelte`
- Modify or create: `viewer/tests/routes/main-route-model-selection.test.ts`
- Modify or create: `viewer/tests/routes/main-route-tooltip.test.ts`

- [ ] **Step 1: Extract model/URL selection as pure helpers first**

Move pure or mostly pure logic out of `+page.svelte`:

- default analysis id resolution wrapper.
- URL analysis id update policy.
- project selection URL mutation policy.
- model-file reload guard.

Write tests for:

```ts
it('keeps the current analysis when the URL has not changed after mount', () => {});
it('updates the URL analysis parameter during project selection', () => {});
it('detects model-file changes that require model reload bookkeeping', () => {});
```

- [ ] **Step 2: Extract tooltip/raycast state helpers**

Move non-markup helper logic out of `+page.svelte`:

- tooltip visibility state transitions.
- pointer-down/pointer-up/wheel suppression policy.
- comparison curtain hover side selection.
- selected live vs fallback analysis selection for tooltip samples.

Write tests for:

```ts
it('suppresses tooltip updates during canvas pointer interaction', () => {});
it('routes comparison-side hover samples to the comparison mesh past the curtain', () => {});
it('uses selected-hour live analysis when the live route is active', () => {});
```

- [ ] **Step 3: Re-run route/helper tests**

Run:

```powershell
cd viewer
npx vitest run tests/routes/main-route-model-selection.test.ts tests/routes/main-route-tooltip.test.ts tests/components/canvasInteractionController.test.ts
```

Expected:

- PASS.
- `viewer/tests/routes/main-route-live-selected-hour-source-lock.test.ts` now requires `viewer/src/routes/mainRouteModelSelection.ts` and `viewer/src/routes/mainRouteTooltip.ts` to exist.

- [ ] **Step 4: Re-run selected-hour browser proof**

Run:

```powershell
cd viewer
npm run test:e2e:selected-hour
```

Expected:

- PASS with 13 Chromium tests.

## Task 4: Debug Route Characterization Before Extraction

**Files:**
- Modify or create: `viewer/tests/debug/debug-route-decomposition-source-lock.test.ts`
- Inspect: `viewer/src/routes/debug/+page.svelte`

- [ ] **Step 1: Add debug route source-lock tests**

Create `viewer/tests/debug/debug-route-decomposition-source-lock.test.ts`.

The tests should assert:

- debug route helper files may contain debug-only behavior.
- main route helper files may not import debug route helper files.
- legacy debug selected-hour constants remain confined to debug route/helper files.
- parity/bin/Python references remain confined to debug route/helper/parity files.

Use explicit allowlists so a future file move is intentional:

```ts
const debugAllowedPaths = [
	'src/routes/debug/+page.svelte',
	'src/routes/debug/debugRouteQueryState.ts',
	'src/routes/debug/debugRouteSelectedHourMode.ts',
	'src/routes/debug/debugRouteSharedHostWiring.ts',
	'src/routes/debug/debugRouteLegacySelectedHourWiring.ts',
	'src/routes/debug/debugRouteParityRuntime.ts',
	'src/lib/debug/debugWebgpuUtciQuery.ts',
	'src/lib/debug/debugSelectedHourMode.ts',
	'src/lib/debug/debugSelectedHourLegacyHost.ts',
	'src/lib/debug/debugWebgpuUtciDiagnostics.ts',
	'src/lib/debug/debugOnDemandPrototypeDiagnostics.ts'
];
```

- [ ] **Step 2: Run debug characterization tests**

Run:

```powershell
cd viewer
npx vitest run tests/debug/debug-webgpu-utci-query.test.ts tests/debug/debug-selected-hour-mode.test.ts tests/debug/debug-selected-hour-legacy-host.test.ts tests/debug/debug-webgpu-utci-diagnostics.test.ts tests/debug/debug-on-demand-prototype-diagnostics.test.ts tests/debug/debug-route-decomposition-source-lock.test.ts
```

Expected:

- PASS.
- This establishes debug proof-lab behavior before moving logic.

- [ ] **Step 3: Add lightweight normal-collect browser characterization**

Create or update a focused Playwright diagnostic spec for debug normal collect mode. Prefer a new lightweight spec:

```text
viewer/tests/e2e/debug-route-normal-collect-diagnostics.spec.ts
```

It must visit a normal collect URL such as:

```text
/debug?analysis=Ben-Gurion%2F20250815_grid_2m_fullday&utciRender=gpu&parity=0&collect=normal
```

The test must assert:

- `window.__normalUtciResults__` appears with collected UTCI results.
- debug diagnostics do not claim Python `.bin` validity.
- parity mode is not active.
- the selected-hour engine remains legacy/debug path for collect mode, not shared-host.

Run:

```powershell
cd viewer
npx playwright test tests/e2e/debug-route-normal-collect-diagnostics.spec.ts --project=chromium --workers=1 --reporter=list --timeout=30000
```

Expected:

- PASS.
- The collect proof is intentionally narrower than the full parity collection suite; it guards route decomposition against losing the app-visible normal collect export.

## Task 5: Extract Debug Route Query, Mode, And Shared-Host Wiring

**Files:**
- Create or modify: `viewer/src/routes/debug/debugRouteQueryState.ts`
- Create: `viewer/src/routes/debug/debugRouteSelectedHourMode.ts`
- Create: `viewer/src/routes/debug/debugRouteSharedHostWiring.ts`
- Modify: `viewer/src/routes/debug/+page.svelte`
- Modify or create: route-helper tests under `viewer/tests/debug/`

- [ ] **Step 1: Extract debug query and mode policy without behavior changes**

Move glue around:

- `parseDebugWebgpuUtciQuery($page.url.searchParams)`.
- parity mode defaults.
- normal collect mode.
- shared-host vs legacy selected-hour decision.

Do not change the underlying policy functions unless a failing test proves current behavior is wrong.

- [ ] **Step 2: Extract shared-host wiring**

Move normal non-parity f32 shared-host selected-hour wiring out of `debug/+page.svelte`:

- debug shared route-host setup.
- shared-host release forwarding.
- shared-host diagnostics merge.
- shared-host visible GPU proof fields.

Keep legacy debug selected-hour path untouched in this task.

- [ ] **Step 3: Run focused debug shared-host tests**

Run:

```powershell
cd viewer
npx vitest run tests/debug/debug-selected-hour-mode.test.ts tests/debug/debug-webgpu-utci-diagnostics.test.ts tests/debug/debug-on-demand-prototype-diagnostics.test.ts
npx playwright test tests/e2e/debug-route-shared-host-diagnostics.spec.ts --project=chromium --workers=1 --reporter=list --timeout=30000
npx playwright test tests/e2e/debug-route-normal-collect-diagnostics.spec.ts --project=chromium --workers=1 --reporter=list --timeout=30000
```

Expected:

- Vitest PASS.
- Playwright PASS with 2 Chromium tests.
- Normal collect diagnostic PASS.
- Diagnostics still show `selectedHourEngine: 'shared-host'` and `strongVisibleGpuPath: true` only when the proof gates are met.
- Normal collect mode still publishes `__normalUtciResults__`, does not claim Python `.bin` validity, and stays on the legacy/debug path.

## Task 6: Extract Debug Legacy Selected-Hour And Parity Runtime Wiring

**Files:**
- Create: `viewer/src/routes/debug/debugRouteLegacySelectedHourWiring.ts`
- Create: `viewer/src/routes/debug/debugRouteParityRuntime.ts`
- Modify: `viewer/src/routes/debug/+page.svelte`
- Modify or create: route-helper tests under `viewer/tests/debug/`

- [ ] **Step 1: Extract legacy selected-hour wiring**

Move legacy debug-only selected-hour behavior out of `debug/+page.svelte`:

- `LEGACY_DEBUG_SELECTED_HOUR_CONTROLLER_ID` branch handling.
- legacy accepted GPU release forwarding.
- legacy dispatch counters.
- deferred CPU fallback activation.
- legacy scrub scheduling/invalidation counters.

Keep existing `createDebugSelectedHourLegacyHost` tests green.

- [ ] **Step 2: Extract parity runtime wiring**

Move debug-only parity runtime behavior:

- Python `.bin` validity metadata.
- August-only `.bin` comparison validity.
- parity intermediate publication.
- debug-only runtime diagnostics fields.

Do not move any parity/bin/Python behavior into the main route or shared selected-hour host.

- [ ] **Step 3: Run focused debug parity and legacy tests**

Run:

```powershell
cd viewer
npx vitest run tests/debug/debug-selected-hour-legacy-host.test.ts tests/debug/debug-webgpu-utci-query.test.ts tests/debug/debug-webgpu-utci-diagnostics.test.ts tests/debug/debug-on-demand-prototype-diagnostics.test.ts tests/debug/debug-route-decomposition-source-lock.test.ts
npx playwright test tests/e2e/debug-route-baseline-diagnostics.spec.ts tests/e2e/debug-route-parity-runtime-diagnostics.spec.ts --project=chromium --workers=1 --reporter=list --timeout=30000
npx playwright test tests/e2e/debug-route-normal-collect-diagnostics.spec.ts --project=chromium --workers=1 --reporter=list --timeout=30000
```

Expected:

- Vitest PASS.
- Playwright PASS with 5 Chromium tests.
- Normal collect diagnostic PASS.
- August Python `.bin` comparison remains valid only for debug/parity August paths.
- Non-August parity does not claim Python `.bin` validity.
- Normal collect mode still publishes `__normalUtciResults__` without claiming parity/Python `.bin` validity.

## Task 7: Full Route Decomposition Verification

**Files:**
- Inspect changed route/helper files.

- [ ] **Step 1: Run selected-hour quality**

Run:

```powershell
cd viewer
npm run test:quality:selected-hour
```

Expected:

- PASS.

- [ ] **Step 2: Run selected-hour browser proof**

Run:

```powershell
cd viewer
npm run test:e2e:selected-hour
```

Expected:

- PASS with 13 Chromium tests.

- [ ] **Step 3: Run all new route decomposition tests explicitly**

Run:

```powershell
cd viewer
npx vitest run tests/routes/main-route-live-selected-hour-source-lock.test.ts tests/routes/main-route-live-selected-hour.test.ts tests/routes/main-route-model-selection.test.ts tests/routes/main-route-tooltip.test.ts tests/debug/debug-route-decomposition-source-lock.test.ts
npx playwright test tests/e2e/debug-route-normal-collect-diagnostics.spec.ts --project=chromium --workers=1 --reporter=list --timeout=30000
```

Expected:

- PASS.
- This command is required even if `test:quality:selected-hour` does not include the new decomposition tests yet.

- [ ] **Step 4: Run build**

Run:

```powershell
cd viewer
npm run build
```

Expected:

- PASS.

- [ ] **Step 5: Run check and classify only touched-file errors**

Run:

```powershell
cd viewer
npm run check
```

Expected:

- May still FAIL from inherited static debt.
- Any new errors in files created or materially touched by Tasks 1-6 must be fixed before continuing.
- Do not chase unrelated inherited families.

- [ ] **Step 6: Run whitespace guard**

Run from repo root:

```powershell
git diff --check
```

Expected:

- PASS or only preexisting/generated warnings outside plan-touched files.

## Task 8: Scene Sync Extraction Guardrail Test

**Files:**
- Create: `viewer/tests/scene/acceptedGpuResidentSurfaceSync.behavior.test.ts`
- Modify only as needed for testability:
  - `viewer/src/lib/components/scene/UTCIPointCloud.svelte`
  - `viewer/src/lib/components/scene/ComparisonRenderer.svelte`
  - `viewer/src/lib/components/scene/acceptedGpuResidentOutputRelease.ts`

- [ ] **Step 1: Write mounted late-old-instance async-copy behavior test**

Create a mounted/integration test that simulates this behavior:

1. Mount the selected-hour scene sync path with controller identity `same-content` and instance id `1`.
2. Begin an async GPU-resident copy.
3. Update props to the same content identity but instance id `2`.
4. Resolve the old instance `1` async copy late.
5. Assert no release is emitted for instance `1` into the current instance `2` route.
6. Assert the notifier still emits exactly once for the current instance `2` accepted output.
7. Repeat the same scenario with a late rejection/failure from old instance `1`.
8. Assert the failure path also cannot release into the current instance `2`, and current instance `2` still releases exactly once.

First create the mounted/test-harness behavior test without production scene extraction. If direct WebGPU/Three mounting is too heavy, create a small test-only harness component around the planned sync dependency seams before extracting the production helper. If production seams are required for a meaningful test, stop and report the blocker before modifying scene components. Do not accept another source-string-only test for this residual risk.

- [ ] **Step 2: Run the failing behavior test**

Run:

```powershell
cd viewer
npx vitest run tests/scene/acceptedGpuResidentSurfaceSync.behavior.test.ts
```

Expected:

- FAIL until the test harness or extraction seam exists.
- If it cannot be made meaningful without extraction, stop and report the exact blocker before touching production scene sync.

## Task 9: Extract Shared Scene Sync Helper

**Files:**
- Create: `viewer/src/lib/components/scene/acceptedGpuResidentSurfaceSync.ts`
- Modify: `viewer/src/lib/components/scene/UTCIPointCloud.svelte`
- Modify: `viewer/src/lib/components/scene/ComparisonRenderer.svelte`
- Modify: `viewer/tests/scene/acceptedGpuResidentSurfaceSync.behavior.test.ts`
- Keep: `viewer/src/lib/components/scene/acceptedGpuResidentOutputRelease.ts`

- [ ] **Step 1: Extract only the shared state-machine logic**

Move duplicated logic from both scene components:

- sync key capture.
- active copy run token.
- controller identity and instance id snapshot.
- supersession check.
- release notification reason mapping.
- copy-failed/copy-complete/superseded state transition policy.

Do not move:

- mesh creation specifics.
- comparison-specific mesh visibility.
- base-specific point cloud/geometry behavior.
- LUT/material construction.

- [ ] **Step 2: Keep scene components as thin adapters**

`UTCIPointCloud.svelte` and `ComparisonRenderer.svelte` should provide:

- component name.
- mesh accessors.
- accepted output.
- controller identity/instance id.
- callbacks for mesh-specific visibility and diagnostics.

- [ ] **Step 3: Run scene tests**

Run:

```powershell
cd viewer
npx vitest run tests/scene/acceptedGpuResidentOutputRelease.test.ts tests/scene/acceptedGpuResidentOutputReleaseCallsites.test.ts tests/scene/acceptedGpuResidentSurfaceSync.behavior.test.ts tests/scene/utci-surface-sync.test.ts tests/scene/utciComputeBufferRenderBridge.test.ts
```

Expected:

- PASS.
- The late-old-instance mounted/integration behavior is covered.

- [ ] **Step 4: Run selected-hour quality and browser proof**

Run:

```powershell
cd viewer
npm run test:quality:selected-hour
npm run test:e2e:selected-hour
```

Expected:

- Both PASS.

- [ ] **Step 5: Run check and classify only touched-file scene errors**

Run:

```powershell
cd viewer
npm run check
```

Expected:

- May still FAIL from inherited static debt.
- Any new errors in `UTCIPointCloud.svelte`, `ComparisonRenderer.svelte`, `acceptedGpuResidentOutputRelease.ts`, `acceptedGpuResidentSurfaceSync.ts`, or the new scene behavior test must be fixed before folder moves.
- Do not chase unrelated inherited families.

## Task 10: Folder/Module Organization Pass

**Files:**
- Move only files created/extracted by this plan.
- Do not move legacy data files or unrelated services.

- [ ] **Step 1: Move extracted main-route helpers only after tests pass**

Candidate moves:

```text
viewer/src/routes/mainRouteLiveSelectedHour.ts -> viewer/src/routes/main/liveSelectedHour.ts
viewer/src/routes/mainRouteModelSelection.ts -> viewer/src/routes/main/modelSelection.ts
viewer/src/routes/mainRouteTooltip.ts -> viewer/src/routes/main/tooltip.ts
```

Update imports and source-lock tests in the same task.

Do not move preexisting support files such as `viewer/src/routes/mainRouteOverlayGating.ts` in this plan unless they were materially changed by the route decomposition and the move is needed to keep imports coherent. Otherwise leave them for the later broad folder-cleanup plan.

- [ ] **Step 2: Move extracted debug-route helpers only after tests pass**

Candidate moves:

```text
viewer/src/routes/debug/debugRouteQueryState.ts -> viewer/src/routes/debug/queryState.ts
viewer/src/routes/debug/debugRouteSelectedHourMode.ts -> viewer/src/routes/debug/selectedHourMode.ts
viewer/src/routes/debug/debugRouteSharedHostWiring.ts -> viewer/src/routes/debug/sharedHostWiring.ts
viewer/src/routes/debug/debugRouteLegacySelectedHourWiring.ts -> viewer/src/routes/debug/legacySelectedHourWiring.ts
viewer/src/routes/debug/debugRouteParityRuntime.ts -> viewer/src/routes/debug/parityRuntime.ts
```

Keep debug-only behavior under `viewer/src/routes/debug/` or `viewer/src/lib/debug/`.

Update debug source-lock allowlists and import-boundary tests in the same step. The explicit allowlists added in Task 4 must not silently go stale after the move.

- [ ] **Step 3: Move scene sync helpers only if there is more than one helper**

Candidate moves:

```text
viewer/src/lib/components/scene/acceptedGpuResidentOutputRelease.ts -> viewer/src/lib/components/scene/selectedHour/acceptedGpuResidentOutputRelease.ts
viewer/src/lib/components/scene/acceptedGpuResidentSurfaceSync.ts -> viewer/src/lib/components/scene/selectedHour/acceptedGpuResidentSurfaceSync.ts
```

Do this only if imports remain readable. If this creates churn without clarity, leave scene helpers beside the scene components.

- [ ] **Step 4: Run full route/scene verification**

Run:

```powershell
cd viewer
npm run test:quality:selected-hour
npm run test:e2e:selected-hour
npx vitest run tests/routes/main-route-live-selected-hour-source-lock.test.ts tests/routes/main-route-live-selected-hour.test.ts tests/routes/main-route-model-selection.test.ts tests/routes/main-route-tooltip.test.ts tests/debug/debug-route-decomposition-source-lock.test.ts
npx playwright test tests/e2e/debug-route-normal-collect-diagnostics.spec.ts --project=chromium --workers=1 --reporter=list --timeout=30000
npm run build
npm run check
cd ..
git diff --check
```

Expected:

- Selected-hour quality PASS.
- Selected-hour browser proof PASS.
- New route decomposition tests PASS.
- Normal collect diagnostic PASS.
- Build PASS.
- `npm run check` may still FAIL from inherited static debt; touched-file errors from route/scene/folder work must be fixed or explicitly proven preexisting and unchanged.
- Whitespace guard PASS.

## Task 11: Final Status Note And Review Agents

**Files:**
- Create: `docs/superpowers/plans/2026-05-11-route-and-scene-cleanup-before-static-debt-results.md`
- Inspect changed files.

- [ ] **Step 1: Request read-only review agents**

Use at least two review agents. They must not edit files.

Route decomposition reviewer prompt:

```text
Review the route decomposition implementation in D:\Projects\Nur\Shade\fast-utci. Do not edit files. Focus on whether viewer/src/routes/+page.svelte and viewer/src/routes/debug/+page.svelte were decomposed without changing selected-hour behavior, debug parity/bin behavior, shared-host diagnostics, or legacy debug selected-hour fallback paths. Return findings first with file/line evidence and missing tests.
```

Scene lifecycle reviewer prompt:

```text
Review the scene sync extraction implementation in D:\Projects\Nur\Shade\fast-utci. Do not edit files. Focus on whether the mounted late-old-instance async copy completion/failure risk is behaviorally tested, whether UTCIPointCloud and ComparisonRenderer preserve exactly-once release semantics, and whether strong visible GPU diagnostics remain honest. Return findings first with file/line evidence and missing tests.
```

Static-debt boundary reviewer prompt:

```text
Review the cleanup implementation in D:\Projects\Nur\Shade\fast-utci. Do not edit files. Focus on whether broad inherited npm run check debt was kept out of scope, whether touched-file static errors were addressed, and whether folder moves happened only after route and scene boundaries were proven. Return findings first with file/line evidence and missing tests.
```

- [ ] **Step 2: Run final verification**

Run:

```powershell
cd viewer
npm run test:quality:selected-hour
npm run test:e2e:selected-hour
npx vitest run tests/routes/main-route-live-selected-hour-source-lock.test.ts tests/routes/main-route-live-selected-hour.test.ts tests/routes/main-route-model-selection.test.ts tests/routes/main-route-tooltip.test.ts tests/debug/debug-route-decomposition-source-lock.test.ts tests/scene/acceptedGpuResidentSurfaceSync.behavior.test.ts
npx playwright test tests/e2e/debug-route-normal-collect-diagnostics.spec.ts --project=chromium --workers=1 --reporter=list --timeout=30000
npm run build
npm run check
cd ..
git diff --check
```

Expected:

- Selected-hour quality PASS.
- Selected-hour browser proof PASS.
- New route decomposition and scene sync behavior tests PASS.
- Normal collect diagnostic PASS.
- Build PASS.
- `npm run check` may still FAIL from inherited static debt.
- Any touched-file check errors must be fixed or explicitly documented if preexisting and unchanged.
- Whitespace guard PASS.

- [ ] **Step 3: Write result note**

Create `docs/superpowers/plans/2026-05-11-route-and-scene-cleanup-before-static-debt-results.md` with:

```md
# Route And Scene Cleanup Before Static Debt Results

Date: 2026-05-11

## Scope

Summarize route decomposition, scene sync extraction, folder organization, and static-debt boundary.

## Route Decomposition

Record what moved out of the main route and debug route.

## Scene Sync Extraction

Record the mounted late-old-instance async-copy behavior test and the extracted helper boundary.

## Folder Organization

Record any file moves. If a candidate move was skipped, explain why.

## Static Debt Boundary

Record current `npm run check` result. Separate touched-file errors from inherited repo-wide debt.

## Verification

Record actual command output summaries with PASS/FAIL and file/test/error counts.

## Review Agents

Record each review agent finding. Blocking findings must include file/line evidence.

## Remaining Work

List broad inherited static-debt cleanup as a separate future plan.
```

- [ ] **Step 4: Stop for human review**

Report:

- changed files
- verification results
- review-agent findings
- remaining `npm run check` debt
- whether route/scene/folder cleanup is ready to close

Do not commit.

## Final Verification Commands

Run before claiming this plan is complete:

```powershell
cd viewer
npm run test:quality:selected-hour
npm run test:e2e:selected-hour
npx vitest run tests/routes/main-route-live-selected-hour-source-lock.test.ts tests/routes/main-route-live-selected-hour.test.ts tests/routes/main-route-model-selection.test.ts tests/routes/main-route-tooltip.test.ts tests/debug/debug-route-decomposition-source-lock.test.ts tests/scene/acceptedGpuResidentSurfaceSync.behavior.test.ts
npx playwright test tests/e2e/debug-route-normal-collect-diagnostics.spec.ts --project=chromium --workers=1 --reporter=list --timeout=30000
npm run build
npm run check
cd ..
git diff --check
```

Completion can be claimed only if:

- selected-hour quality passes.
- selected-hour browser proof passes.
- route decomposition tests pass.
- scene sync late-old-instance behavior test passes.
- normal collect diagnostic passes.
- build passes.
- whitespace guard passes.
- any touched-file static errors are fixed or explicitly proven preexisting and unchanged.
- broad inherited `npm run check` debt remains documented as a separate lane if still failing.
- debug route parity/bin/legacy/shared-host behavior is preserved.
- main route remains free of debug-only parity/bin behavior.
- scene sync extraction includes the mounted late-old-instance async-copy behavior test.

## Next-Agent Execution Handoff

Use this plan with `superpowers:subagent-driven-development` and `superpowers:verification-before-completion`.

Execution requirements:

- Work in `D:\Projects\Nur\Shade\fast-utci`.
- Do not create commits.
- Do not create git worktrees.
- Preserve unrelated dirty files.
- Execute task-by-task.
- Use failing tests before implementation changes.
- Run fresh verification after each task.
- Use review agents before claiming completion.
- If browser verification hangs, use shorter targeted waits and `superpowers:systematic-debugging`; do not patch by guesswork.
- If `npm run check` fails, classify touched-file errors separately from inherited repo-wide debt instead of expanding the plan into broad static cleanup.
