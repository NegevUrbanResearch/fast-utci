# WebGPU Live Selected-Hour Route Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Workflow overrides for this plan:** Do not create git worktrees. Do not create commits. Keep all verification local, preserve unrelated dirty files, and report verification evidence explicitly before advancing between tasks.

**Goal:** Rebuild the main route and debug route around one shared live selected-hour WebGPU host so `/` matches the debug route's selected-hour behavior exactly, while the debug route keeps the legacy `.bin` vs live comparison and diagnostics surface.

**Architecture:** Keep the existing per-analysis selected-hour compute/session layer, harden the existing controller's transport/publication contract, then add a new shared route-facing host that owns selection inputs, request invalidation, source-analysis switching, render-surface publication truth, and comparison-safe orchestration. Rebuild both routes as thin coordinators around that host, and move debug-only parity/diagnostic logic into a debug-only module instead of leaving it embedded in the route.

**Tech Stack:** SvelteKit, Svelte stores/reactivity, Threlte/Three.js, WebGPU UTCI compute pipeline, Playwright, Vitest.

---

## Root Cause Summary

The current regression is not a single bug. It is the result of promoting an incomplete seam:

1. The shared extraction currently stops at the compute/session/controller layer. Too much route-facing selection and publication orchestration still lives in [viewer/src/routes/+page.svelte](d:/Projects/Nur/Shade/fast-utci/viewer/src/routes/+page.svelte) and [viewer/src/routes/debug-webgpu-utci/+page.svelte](d:/Projects/Nur/Shade/fast-utci/viewer/src/routes/debug-webgpu-utci/+page.svelte).
2. The current main route therefore does not inherit the exact selected-hour behavior from the debug route. It re-coordinates enough state locally to diverge on initialization, month/hour scrubbing, and comparison transitions.
3. The lower-level live controller still needs stronger proof boundaries:
   - `compute-buffer-selected-hour` must require `sameDeviceForComputeAndRender === true`
   - comparison live source ownership must reset atomically when comparison analysis eligibility changes
   - CPU publication readiness must be scoped to the current selection instead of any non-zero historical publish counter
4. Both routes are still too large to be trustworthy coordinators:
   - `viewer/src/routes/+page.svelte`: about 1124 lines
   - `viewer/src/routes/debug-webgpu-utci/+page.svelte`: about 4309 lines

The plan below fixes the architecture first, then lets the route bug regressions fall out of the stronger shared contract.

## Target File Structure

### Keep and Harden

- `viewer/src/lib/compute/liveUtciSelectedHour.ts`
  - Keep as the selected-hour live-analysis builder and range helper.
- `viewer/src/lib/compute/liveUtciSelectedHourSession.ts`
  - Keep as the per-analysis live compute/session layer.
  - Harden the selected-hour result contract so GPU transport claims remain honest and CPU fallback publication remains request-scoped.
- `viewer/src/lib/compute/liveSelectedHourController.ts`
  - Keep as the low-level request/session controller.
  - Harden it so render transport and ready/publication state cannot go green on stale or insufficient evidence.

### Create

- `viewer/src/lib/compute/liveSelectedHourSurfaceIdentity.ts`
  - Shared selected-hour surface identity contract used by the controller, route host, and scene components.
- `viewer/src/lib/compute/liveSelectedHourRouteHost.ts`
  - New shared route-facing host for base and comparison flows.
  - Owns selection inputs, controller lifecycle, render-surface diagnostics intake, comparison source-analysis switching, trigger invalidation, and route-facing derived state.
- `viewer/src/lib/debug/debugWebgpuUtciRouteLayer.ts`
  - New debug-only module that composes the shared host with parity/prototype/telemetry state.
  - Owns `.bin` comparison glue, debug window diagnostics shaping, and prototype/debug-only derivations.
- `viewer/tests/compute/live-selected-hour-route-host.test.ts`
  - New focused unit tests for route-facing selected-hour host state transitions.

### Modify

- `viewer/src/lib/components/scene/UTCIPointCloud.svelte`
  - Publish request-scoped CPU and GPU surface diagnostics so the shared host can prove current-surface publication honestly.
- `viewer/src/lib/components/scene/ComparisonRenderer.svelte`
  - Accept the same live selected-hour surface identity contract as the base path, including optional GPU-resident comparison publication.
  - Publish the same scoped surface diagnostics contract for comparison rendering.
- `viewer/src/routes/+page.svelte`
  - Shrink into a coordinator that wires stores/UI/model loading into one or two shared route hosts.
- `viewer/src/routes/debug-webgpu-utci/+page.svelte`
  - Replace duplicated selected-hour orchestration with the shared route host and move debug-only logic into the debug module where safe.
- `viewer/tests/e2e/webgpu-on-demand-prototype.spec.ts`
  - Tighten route-level app-visible assertions around `/` and the debug route baseline.

## Behavioral Baseline

This plan treats the current debug route's selected-hour WebGPU behavior as the non-negotiable baseline for `/`:

- Initial selected-hour UTCI surface must appear with correct colors.
- Month/hour scrubbing on `/` must follow the same live selected-hour path and responsiveness expectations as debug.
- Comparison mode on `/` must keep month/hour changes wired through the same shared live host without freezing on stale comparison source state.
- The main route must not use `.bin`, `runAll()`, or route-local fallback hacks to simulate success.
- The debug route must continue to expose the `.bin` vs live comparison surface and diagnostics honestly.
- The baseline must be locked explicitly before the debug route is rebuilt, so the refactor cannot silently redefine it.

## Task 1: Lock the Behavioral Baseline With Failing App-Visible Tests

**Files:**
- Modify: `viewer/tests/e2e/webgpu-on-demand-prototype.spec.ts`
- Modify: `viewer/tests/compute/live-selected-hour-controller.test.ts`
- Create: `viewer/tests/compute/live-selected-hour-route-host.test.ts`

- [ ] **Step 1: Add a focused failing main-route scrub/comparison regression spec**

Add or update a Playwright test so the main route proves all three current regressions in one place:

```ts
test('main route matches debug selected-hour behavior for init, scrubbing, and comparison month changes', async ({
	page
}) => {
	await page.goto('/?analysis=Ben-Gurion&utciRender=auto&utciRenderDiagnostics=1');
	await skipIfMainRouteLiveComputeUnavailable(page);
	await expectPublishedBaseSurface(page);
	await expectBaseSurfaceSource(page, /selected-hour/);

	const initial = await readMainRouteDiagnostics(page);
	expect(initial.baseLiveReady).toBe(true);

	await setMainRouteMonth(page, 8);
	await expectPublishedBaseSurface(page);

	await enableScenarioComparison(page);
	await selectComparisonScenario(page, 'Winter');
	await expectPublishedComparisonSurface(page);

	await setMainRouteMonth(page, 1);
	await expectPublishedBaseSurface(page);
	await expectPublishedComparisonSurface(page);
});
```

- [ ] **Step 2: Add an explicit route-to-route baseline assertion before rewriting the debug route**

Add one focused Playwright test that captures the same month/hour selection on both routes and compares the app-visible diagnostics that matter:

```ts
test('main route matches debug selected-hour baseline for the same selection', async ({
	page,
	browser
}) => {
	const debugPage = await browser.newPage();

	await page.goto('/?analysis=Ben-Gurion&utciRender=auto&utciRenderDiagnostics=1');
	await debugPage.goto('/debug-webgpu-utci?analysis=Ben-Gurion&utciRender=auto');

	await skipIfMainRouteLiveComputeUnavailable(page);
	await skipIfDebugRouteLiveComputeUnavailable(debugPage);

	await setMainRouteMonth(page, 7);
	await setMainRouteHour(page, 12);
	await setDebugRouteMonth(debugPage, 7);
	await setDebugRouteHour(debugPage, 12);

	const mainDiagnostics = await readMainRouteDiagnostics(page);
	const debugDiagnostics = await readDebugRouteDiagnostics(debugPage);

	expect(mainDiagnostics.baseLiveReady).toBe(true);
	expect(debugDiagnostics.utciSurfaceSource).toMatch(/selected-hour/);
	expect(mainDiagnostics.utciSurfaceSource).toMatch(/selected-hour/);
});
```

- [ ] **Step 3: Add controller-level failing proofs for the known contract holes**

Extend `viewer/tests/compute/live-selected-hour-controller.test.ts` with explicit failures for:

```ts
it('does not accept compute-buffer transport without same-device proof', async () => {
	const controller = createLiveSelectedHourController({ prepareSession: prepareSessionMock });
	await controller.requestSelection(makeGpuResidentRequest());

	await controller.handleRenderSurfaceDiagnostics({
		utciSurfaceSource: 'compute-buffer-selected-hour',
		gpuResidentCopyStatus: 'complete',
		gpuResidentCopyRequestId: 1
	});

	expect(hasPublishedLiveSelectedHourSurface(controller.getState())).toBe(false);
});
```

```ts
it('ignores stale cpu publication from a previous request', async () => {
	const controller = createLiveSelectedHourController({ prepareSession: prepareSessionMock });
	await controller.requestSelection(makeCpuFallbackRequest({ requestId: 1 }));
	await controller.requestSelection(makeCpuFallbackRequest({ requestId: 2 }));

	await controller.handleRenderSurfaceDiagnostics({
		selectedHourTransferCount: 1
	});

	expect(hasPublishedLiveSelectedHourSurface(controller.getState())).toBe(false);
});
```

- [ ] **Step 4: Add a new failing route-host test file**

Create `viewer/tests/compute/live-selected-hour-route-host.test.ts` with cases for:

```ts
it('clears comparison source ownership when the selected comparison analysis becomes ineligible', () => {
	const host = createLiveSelectedHourRouteHost(makeHostDeps());
	host.setRouteInputs(makeComparisonInputs({ comparisonAnalysisType: 'single_hour' }));
	expect(host.getState().comparisonSourceAnalysisId).toBeNull();
});
```

```ts
it('recomputes comparison state when month changes after comparison activation', async () => {
	const host = createLiveSelectedHourRouteHost(makeHostDeps());
	host.setRouteInputs(makeComparisonInputs({ currentMonth: 7 }));
	await host.flush();
	host.setRouteInputs(makeComparisonInputs({ currentMonth: 1 }));
	await host.flush();
	expect(host.getState().comparisonSurfaceIdentity?.selectionKey).toContain('|1|');
});
```

- [ ] **Step 5: Run the focused failing tests**

Run:

```bash
cd viewer
npx playwright test tests/e2e/webgpu-on-demand-prototype.spec.ts --grep "main route matches debug selected-hour behavior"
npx playwright test tests/e2e/webgpu-on-demand-prototype.spec.ts --grep "main route matches debug selected-hour baseline for the same selection"
npx vitest run tests/compute/live-selected-hour-controller.test.ts tests/compute/live-selected-hour-route-host.test.ts
```

Expected:
- Playwright fails on at least one of init/scrub/comparison assertions with the current implementation.
- The route-to-route baseline test fails if `/` and debug diverge on the selected-hour publication contract.
- Vitest fails on the new controller/route-host cases because the contract is not yet enforced.

## Task 2: Harden the Low-Level Selected-Hour Transport and Publication Contract

**Files:**
- Create: `viewer/src/lib/compute/liveSelectedHourSurfaceIdentity.ts`
- Modify: `viewer/src/lib/compute/liveUtciSelectedHourSession.ts`
- Modify: `viewer/src/lib/compute/liveSelectedHourController.ts`
- Modify: `viewer/src/lib/components/scene/UTCIPointCloud.svelte`
- Modify: `viewer/src/lib/components/scene/ComparisonRenderer.svelte`
- Test: `viewer/tests/compute/live-selected-hour-controller.test.ts`

- [ ] **Step 1: Add request-scoped surface publication metadata**

Create the shared surface-identity type first so Task 2 and Task 3 consume the same contract:

```ts
export type LiveSelectedHourSurfaceIdentity = {
	requestId: number;
	monthIndex: number;
	hourIndex: number;
	timeIndex: number;
	selectionKey: string;
	pendingRenderUpdateStartedAt: number | undefined;
	acceptedGpuResidentOutput: SelectedHourGpuResidentOutput | null;
};
```

Extend the render-surface diagnostics contract to carry the information the controller needs to prove current publication:

```ts
export type LiveSelectedHourControllerSurfaceDiagnostics = {
	utciSurfaceSource?: string;
	selectedHourTransferCount?: number;
	dataTextureBuildCount?: number;
	gpuResidentCopyStatus?: 'idle' | 'pending' | 'complete' | 'failed';
	gpuResidentCopyError?: string;
	gpuResidentCopyRequestId?: number;
	cpuPublishRequestId?: number;
	cpuPublishMonthIndex?: number;
	cpuPublishHourIndex?: number;
	cpuPublishTimeIndex?: number;
	cpuPublishSelectionKey?: string;
} & SelectedHourRenderTimingSubsteps;
```

- [ ] **Step 2: Reject GPU publication without same-device proof**

In `liveSelectedHourController.ts`, tighten the GPU publication predicate so this is the acceptance bar:

```ts
return (
	acceptedRequestId !== undefined &&
	state.sameDeviceForComputeAndRender === true &&
	state.renderSurfaceDiagnostics.gpuResidentCopyRequestId === acceptedRequestId &&
	state.renderSurfaceDiagnostics.gpuResidentCopyStatus === 'complete' &&
	state.renderSurfaceDiagnostics.utciSurfaceSource === 'compute-buffer-selected-hour'
);
```

- [ ] **Step 3: Track the current CPU publish token inside the controller**

Add controller state for the accepted CPU fallback selection:

```ts
type AcceptedCpuPublication = {
	requestId: number;
	monthIndex: number;
	hourIndex: number;
	timeIndex: number;
};
```

Update `deriveState()` and `hasPublishedLiveSelectedHourSurface(...)` so CPU readiness only becomes true when the render diagnostics match the accepted CPU publish token, not when any historical upload count is non-zero.

- [ ] **Step 4: Publish request-scoped CPU diagnostics from the scene components**

When `UTCIPointCloud.svelte` and `ComparisonRenderer.svelte` build or update a CPU-driven selected-hour surface, publish:

```ts
{
	selectedHourTransferCount,
	cpuPublishRequestId,
	cpuPublishMonthIndex,
	cpuPublishHourIndex,
	cpuPublishTimeIndex,
	cpuPublishSelectionKey
}
```

Source the request-scoped values from one explicit live-surface identity object owned by the controller/host and passed into the scene components. Do not recreate this bookkeeping in the routes.

- [ ] **Step 5: Rerun the focused unit tests**

Run:

```bash
cd viewer
npx vitest run tests/compute/live-selected-hour-controller.test.ts
```

Expected:
- New same-device and stale-CPU-publication tests pass.
- Existing controller tests stay green.

## Task 3: Build the Shared Route-Facing Selected-Hour Host

**Files:**
- Create: `viewer/src/lib/compute/liveSelectedHourRouteHost.ts`
- Create: `viewer/tests/compute/live-selected-hour-route-host.test.ts`
- Modify: `viewer/src/lib/compute/liveSelectedHourSurfaceIdentity.ts`
- Modify: `viewer/src/lib/compute/liveSelectedHourController.ts`
- Modify: `viewer/src/lib/compute/projectWeather.ts`

- [ ] **Step 1: Create the route-host input and state contracts**

Define one shared host for both base and comparison usage:

```ts
import type { LiveSelectedHourSurfaceIdentity } from '$lib/compute/liveSelectedHourSurfaceIdentity';

export type LiveSelectedHourRouteInputs = {
	enabled: boolean;
	analysisId: string | null;
	baseAnalysis: Analysis | null;
	baseModel: Group | null;
	selection: {
		monthIndex: number;
		hourIndex: number;
		timeIndex: number;
		selectionKey: string;
	};
	colorMode: 'normalized' | 'discrete';
	rendererBackend: 'unknown' | 'webgpu';
	rendererDevice?: GPUDevice;
	utciSurfaceBackend: 'dataTexture' | 'gpuNative';
	comparison: {
		active: boolean;
		analysisId: string | null;
		sourceAnalysis: Analysis | null;
		model: Group | null;
		rendererDevice?: GPUDevice;
	};
};
export type LiveSelectedHourRouteState = {
	base: LiveSelectedHourControllerState;
	comparison: LiveSelectedHourControllerState;
	baseDisplayAnalysis: Analysis | null;
	comparisonDisplayAnalysis: Analysis | null | undefined;
	baseSurfaceIdentity: LiveSelectedHourSurfaceIdentity | null;
	comparisonSurfaceIdentity: LiveSelectedHourSurfaceIdentity | null;
	baseReady: boolean;
	comparisonReady: boolean;
	comparisonSourceAnalysisId: string | null;
	liveUnifiedRange: { utciMin: number; utciMax: number } | null;
};
```

- [ ] **Step 2: Implement host-owned lifecycle and trigger logic**

`createLiveSelectedHourRouteHost(...)` should own:

```ts
setRouteInputs(inputs: LiveSelectedHourRouteInputs): void;
handleBaseSurfaceDiagnostics(diagnostics: LiveSelectedHourControllerSurfaceDiagnostics): void;
handleComparisonSurfaceDiagnostics(diagnostics: LiveSelectedHourControllerSurfaceDiagnostics): void;
getState(): LiveSelectedHourRouteState;
subscribe(listener: (state: LiveSelectedHourRouteState) => void): () => void;
dispose(): void;
```

Inside the host:
- replace controllers when analysis/model/session identity changes
- accept one route-resolved selection payload instead of recomputing selection identity separately in each route
- clear comparison source ownership immediately when comparison becomes inactive or ineligible
- trigger recompute for base/comparison from one internal scheduler
- expose `baseDisplayAnalysis`, `comparisonDisplayAnalysis`, `baseReady`, `comparisonReady`, surface identities, and unified range as derived host state
- own the comparison-side GPU-publication path as a first-class concern instead of leaving comparison stuck on a CPU-only rendering seam

- [ ] **Step 3: Add host tests before integrating routes**

Cover at least these cases:

```ts
it('keeps base and comparison controller orchestration out of the route', async () => {
	const host = createLiveSelectedHourRouteHost(makeHostDeps());
	host.setRouteInputs(makeBaseInputs());
	await host.flush();
	expect(host.getState().baseDisplayAnalysis).not.toBeNull();
});
```

```ts
it('drops stale comparison source state when comparison analysis identity changes', async () => {
	const host = createLiveSelectedHourRouteHost(makeHostDeps());
	host.setRouteInputs(makeComparisonInputs({ comparisonAnalysisId: 'winter' }));
	await host.flush();
	host.setRouteInputs(makeComparisonInputs({ comparisonAnalysisId: 'summer' }));
	await host.flush();
expect(host.getState().comparisonSourceAnalysisId).toBe('summer');
});
```

```ts
it('publishes comparison surface identity for the active selection', async () => {
	const host = createLiveSelectedHourRouteHost(makeComparisonHostDeps());
	host.setRouteInputs(makeComparisonInputs({ selectionKey: '7|12|180' }));
	await host.flush();
	expect(host.getState().comparisonSurfaceIdentity?.selectionKey).toBe('7|12|180');
});
```

- [ ] **Step 4: Run the new host tests**

Run:

```bash
cd viewer
npx vitest run tests/compute/live-selected-hour-route-host.test.ts
```

Expected:
- All host tests pass and the host becomes the new single place for route-facing selected-hour orchestration.

The host test helper API must explicitly include:

```ts
flush(): Promise<void>;
```

so test code does not invent ad hoc async hooks during implementation.

## Task 4: Rebuild the Main Route Around the Shared Host

**Files:**
- Modify: `viewer/src/routes/+page.svelte`
- Modify: `viewer/src/lib/components/scene/UTCIPointCloud.svelte`
- Modify: `viewer/src/lib/components/scene/ComparisonRenderer.svelte`
- Test: `viewer/tests/e2e/webgpu-on-demand-prototype.spec.ts`

- [ ] **Step 1: Remove route-local base/comparison trigger orchestration**

Delete the route-owned selected-hour state that the host should now own:

```ts
let baseLiveSelectionTriggerKey: string | null = null;
let comparisonLiveSelectionTriggerKey: string | null = null;
let comparisonSourceAnalysis: Analysis | null = null;
let comparisonSourceAnalysisId: string | null = null;
```

Replace them with:

```ts
const liveRouteHost = createLiveSelectedHourRouteHost();
let liveRouteState = liveRouteHost.getState();
const unsubscribeLiveRouteHost = liveRouteHost.subscribe((state) => {
	liveRouteState = state;
});
```

- [ ] **Step 2: Route all live inputs through the host**

Replace the route's manual recompute/reactive key blocks with one host input update:

```ts
$: liveRouteHost.setRouteInputs({
	enabled: useLiveUtciOnMainRoute,
	analysisId,
	baseAnalysis: $analysisStore,
	baseModel: model,
	selection: {
		monthIndex: $viewerStore.currentMonth ?? 7,
		hourIndex: $viewerStore.currentHour,
		timeIndex: getEffectiveHourIndex(
			$analysisStore,
			$viewerStore.currentHour,
			$viewerStore.currentMonth ?? 7
		),
		selectionKey: [
			analysisId,
			$viewerStore.currentMonth ?? 7,
			$viewerStore.currentHour
		].join('|')
	},
	colorMode: $viewerStore.colorMode,
	rendererBackend,
	rendererDevice: rendererDeviceForMain,
	utciSurfaceBackend: resolvedUtciSurfaceBackend,
	comparison: {
		active: $comparisonStore.isComparing,
		analysisId: $comparisonStore.comparisonAnalysisId,
		sourceAnalysis: $comparisonAnalysis,
		model: comparisonModelForLiveCompute,
		rendererDevice: rendererDeviceForMain
	}
});
```

- [ ] **Step 3: Bind the view to host-derived state**

Replace route-local derivations with host state:

```ts
$: baseDisplayedAnalysis = useLiveUtciOnMainRoute
	? liveRouteState.baseDisplayAnalysis
	: $analysisStore;

$: comparisonRendererDisplayAnalysis = !isComparing
	? undefined
	: useLiveUtciOnMainRoute
		? liveRouteState.comparisonDisplayAnalysis
		: undefined;

$: baseLiveReady = useLiveUtciOnMainRoute
	? liveRouteState.baseReady
	: $analysisStore != null;
```

- [ ] **Step 4: Promote comparison rendering to the same shared live-surface contract**

Extend `ComparisonRenderer.svelte` so the comparison side can accept the same live-surface identity as the base path:

```ts
export let acceptedGpuResidentOutput: SelectedHourGpuResidentOutput | null = null;
export let pendingRenderUpdateStartedAt: number | undefined = undefined;
export let liveSurfaceIdentity: LiveSelectedHourSurfaceIdentity | null = null;
```

Make sure the comparison path can either:
- publish a true GPU-resident comparison surface when the shared host provides one, or
- fall back honestly with request-scoped CPU publication metadata for the active selection.

- [ ] **Step 5: Keep diagnostics and readiness honest**

Update the route diagnostics export to source all live state from the host, not from scattered local fields, and make sure `onUtciSurfaceDiagnostics` handlers forward straight into the host.

- [ ] **Step 6: Run focused main-route verification**

Run:

```bash
cd viewer
npx playwright test tests/e2e/webgpu-on-demand-prototype.spec.ts --grep "main route"
```

Expected:
- The main-route suite passes.
- The route no longer owns selected-hour trigger glue or comparison-source state.
- Comparison rendering follows the same live-surface identity contract instead of a separate CPU-only seam.

## Task 5: Rebuild the Debug Route Around the Shared Host and Extract the Debug Layer

**Files:**
- Create: `viewer/src/lib/debug/debugWebgpuUtciRouteLayer.ts`
- Modify: `viewer/src/routes/debug-webgpu-utci/+page.svelte`
- Modify: `viewer/tests/e2e/webgpu-on-demand-prototype.spec.ts`

- [ ] **Step 1: Create a debug-only composition module**

Move debug-only logic that should not live in the shared host into a dedicated module:

```ts
export type DebugWebgpuUtciRouteLayer = {
	updateDebugInputs(inputs: DebugWebgpuUtciRouteInputs): void;
	consumeHostState(state: LiveSelectedHourRouteState): void;
	getState(): DebugWebgpuUtciRouteLayerState;
	subscribe(listener: (state: DebugWebgpuUtciRouteLayerState) => void): () => void;
	dispose(): void;
};
```

This module should own:
- Python `.bin` comparison state
- prototype status assembly
- debug window diagnostics shaping
- parity-only hour/month reference mapping
- any debug-only telemetry rollups that currently clutter the route

It must not re-implement the shared live selected-hour host.
It must not accept raw live-route inputs or become a second controller for selection/publication state.

- [ ] **Step 2: Replace route-local selected-hour orchestration with the shared host**

In the debug route:
- keep model loading, project switching, and debug UI concerns in the route
- remove route-local selected-hour recompute state where the host now owns it
- wire the route through the shared host first, then let the debug layer consume the host outputs plus debug-only extras

- [ ] **Step 3: Preserve debug route parity surface**

Keep these debug-only behaviors intact:
- `.bin` vs live comparison surface
- explicit `utciOnDemand=off` opt-out behavior
- prototype diagnostics window exports
- parity/debug-only timing and telemetry

Do not route any of these through the main-route host.

- [ ] **Step 4: Run focused debug-route verification**

Run:

```bash
cd viewer
npx playwright test tests/e2e/webgpu-on-demand-prototype.spec.ts --grep "debug route selected-hour scrub"
npx playwright test tests/e2e/webgpu-on-demand-prototype.spec.ts --grep "plain debug route defaults|debug route honors utciOnDemand=off explicit opt-out"
```

Expected:
- The debug route still behaves as the legacy `.bin` vs live surface.
- The route gets smaller because the shared live selected-hour orchestration moved out.
- The debug selected-hour scrub/query behavior still works after the extraction.

## Task 6: Full Focused Verification and Honest Debt Reporting

**Files:**
- Modify: `viewer/tests/e2e/webgpu-on-demand-prototype.spec.ts` if final test wording needs tightening
- No new production files required unless verification exposes a real gap

- [ ] **Step 1: Manual/browser verification**

Confirm in a real browser session:

1. `http://localhost:5173/` visibly shows the UTCI layer on initial load.
2. Month/hour scrubbing on `/` follows the debug-route baseline and does not regress into slow route-local recompute behavior.
3. Scenario comparison on `/` still works, including month dial changes after comparison activation.
4. `http://localhost:5173/debug-webgpu-utci` still behaves as the legacy `.bin` vs live debug surface.

- [ ] **Step 2: Run the required Playwright commands**

Run:

```bash
cd viewer
npx playwright test tests/e2e/webgpu-on-demand-prototype.spec.ts --grep "main route"
npx playwright test tests/e2e/webgpu-on-demand-prototype.spec.ts --grep "main route matches debug selected-hour baseline for the same selection"
npx playwright test tests/e2e/webgpu-on-demand-prototype.spec.ts --grep "debug route selected-hour scrub"
npx playwright test tests/e2e/webgpu-on-demand-prototype.spec.ts --grep "plain debug route defaults|debug route honors utciOnDemand=off explicit opt-out"
```

Expected:
- All commands pass.

- [ ] **Step 3: Run focused static/type sanity**

Run:

```bash
cd viewer
npm run check
```

Expected:
- Capture the actual output.
- Separate new failures from preexisting repo debt explicitly.
- Do not claim repo-wide green unless the command truly returns clean.

- [ ] **Step 4: Re-read the plan requirements before reporting completion**

Before claiming success, verify that the finished implementation actually satisfies:
- main route behavior matches debug selected-hour baseline
- shared selected-hour route-facing host exists and is used by both routes
- debug-only parity/diagnostic logic remains outside the shared host
- both routes are smaller and more coordinator-like
- no `.bin` or `runAll()` fallback was introduced on `/`

## Execution Notes

- Use one fresh implementer subagent per task.
- After each task:
  1. spec/compliance review subagent
  2. fix any findings
  3. code-quality review subagent
  4. fix any findings
  5. only then continue
- Do not begin code-quality review before spec compliance is clean.
- Do not revert unrelated dirty files.
- Do not treat a passing helper test as sufficient proof for route behavior.
- Keep route shrinkage intentional: only extract modules that clarify ownership and directly support the shared-host architecture.
