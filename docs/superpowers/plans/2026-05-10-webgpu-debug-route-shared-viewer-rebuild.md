# WebGPU Debug Route Shared Viewer Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Workflow override:** The generic writing-plans template suggests commits. Do **not** commit in this repo slice. Do **not** create git worktrees. Preserve unrelated dirty files and report blockers before broad rewrites.

**Goal:** Rebuild `/debug-webgpu-utci` so normal non-parity f32 rendering uses the same selected-hour host/projection/render spine as `/`, while Python `.bin`, parity, collect, strict exposure, and visual proof tools remain debug-only.

**Architecture:** Add an explicit debug shared-host mode for normal f32 only, then make all legacy debug selected-hour dispatch paths opt out when that mode is active. Route scene props and diagnostics through the shared main-route host/projection family, and only report `selectedHourEngine: "shared-host"` after legacy normal f32 dispatch is disabled and shared host owns compute/render. Keep Threlte/WebGPU scene lifecycle orchestration inside scene components.

**Tech Stack:** SvelteKit, Svelte reactive statements/stores, Threlte/Three.js, WebGPU UTCI compute, Vitest, short targeted Playwright probes, PowerShell on Windows.

---

## Current State

`/` is the trusted canonical selected-hour WebGPU viewer path. It loads metadata without `.bin`, builds live selected-hour compute through `createLiveSelectedHourRouteHost`, projects route state through `projectMainRouteLiveSceneState`, and publishes `window.__utciRenderDiagnostics__` through `buildMainRouteUtciDiagnostics`.

`/debug-webgpu-utci` still owns legacy f32 selected-hour scheduling through `runDebugOnDemandSelectedHour`, `scheduleDebugOnDemandScrubRecompute`, `acceptedGpuResidentUtciOutput`, and `liveUtciSurfaceDiagnostics`. It now publishes `selectedHourEngine: "legacy-debug"` honestly. This rebuild must not leave both legacy debug dispatch and shared-host dispatch active for normal f32.

## Legacy Debug Owner Inventory

These names are the duplication-risk surface. Do not remove or rename them as drive-by cleanup; gate them deliberately in the tasks below.

| Owner | Current responsibility | Shared-host rule |
| --- | --- | --- |
| `debugOnDemandSelection` | Route-local selected month/hour/time for debug f32 | Still supplies selected-hour input for normal f32 shared host |
| `debugOnDemandSelectionKey` | Legacy dedupe key for scrub scheduling | Must not trigger legacy scrub when shared host is active |
| `lastDebugOnDemandScrubTriggerKey` | Last legacy scrub trigger | Reset while shared host is active |
| `scheduleDebugOnDemandScrubRecompute` | Legacy debounced selected-hour recompute | Must not schedule in normal f32 shared-host mode |
| `runDebugOnDemandSelectedHour` | Legacy debug f32 compute/readback dispatch | Must not run in normal f32 shared-host mode |
| `acceptedGpuResidentUtciOutput` | Legacy accepted GPU output for scene props | Replaced by shared projection output in normal f32 shared-host mode |
| `deferredCpuFallbackSelectedHour` | Legacy CPU fallback handoff | Remains legacy-only unless an explicit later plan migrates fallback |
| `liveUtciSurfaceDiagnostics` | Legacy scene/surface proof feed | Replaced by shared route/projection diagnostics in normal f32 shared-host mode |
| `updateOnDemandPrototypeDiagnostics` | Debug route window diagnostics publisher | Publishes shared-host fields only after legacy normal f32 dispatch is gated off |
| `shouldReadbackForComparison` | Python `.bin` comparison readback toggle | Remains debug-only and August-valid only |
| `pythonBinComparisonActive` | Python `.bin` proof flag | Must never appear on `/`; non-August must report invalid, not active |

## Diagnostics Contract

The final implementation must prove these exact fields.

Main route `/`:

```ts
expect(value.utciRenderResolved).toBe('gpuNative');
expect(value.baseRenderTransport).toBe('compute-buffer-selected-hour');
expect(value.utciSurfaceSource).toBe('compute-buffer-selected-hour');
expect(value.dataTextureBuildCount).toBe(0);
expect(value.baseSameDeviceForComputeAndRender).toBe(true);
expect(value.baseSelectionKey).toBe(value.baseSceneSelectionKey);
expect(value.baseSelectedTimeIndex).toBe(value.baseRenderContextTimeIndex);
```

Debug normal f32 shared-host path:

```ts
expect(value.selectedHourEngine).toBe('shared-host');
expect(value.renderTransport).toBe('compute-buffer-selected-hour');
expect(value.utciSurfaceSource).toBe('compute-buffer-selected-hour');
expect(value.sameDeviceForComputeAndRender).toBe(true);
expect(value.selectedHourReadbackCount).toBe(0);
expect(value.dataTextureBuildCount).toBe(0);
expect(value.legacySelectedHourDispatchCount).toBe(0);
expect(value.legacyScrubScheduleCount).toBe(0);
expect(value.selectionKey).toBe(value.sceneSelectionKey);
expect(value.selectedTimeIndex).toBe(value.renderContextTimeIndex);
```

Debug parity path:

```ts
expect(value.selectedHourEngine).toBe('legacy-debug');
expect(value.binComparisonEnabled).toBe(true);
expect(value.binComparisonValid).toBe(true); // August only
expect(value.pythonBinComparisonActive).toBe(true);
expect(value.debugComparisonReference).toBe('python-bin');
```

Non-August parity path:

```ts
expect(value.binComparisonEnabled).toBe(true);
expect(value.binComparisonValid).toBe(false);
expect(value.pythonBinComparisonActive).not.toBe(true);
expect(value.debugComparisonReference).not.toBe('python-bin');
expect(value.pythonBaselineStatus).toBe('unavailable-non-august');
```

## Playwright Probe Rules

Every Playwright spec added or changed in this plan must use:

```ts
test.afterEach(async ({ page }) => {
	await page.goto('about:blank').catch(() => undefined);
});
```

If a probe times out, it must throw with the relevant diagnostic payload:

```ts
const lastDiagnostics = await page.evaluate(() => {
	return (window as any).__onDemandPrototypeDiagnostics__ ?? null;
});
throw new Error(JSON.stringify(lastDiagnostics, null, 2));
```

For `/`, use `window.__utciRenderDiagnostics__`. Do not silently skip WebGPU failures; if WebGPU is unavailable in Chromium, treat it as a harness/runtime failure and report diagnostics.

## File Structure

### Create

- `viewer/src/lib/debug/debugSelectedHourMode.ts`
  - Pure mode predicate and diagnostics counter helpers for debug normal shared-host routing.
  - No Svelte imports, no browser/window access, no Python `.bin` access.
- `viewer/tests/debug/debug-selected-hour-mode.test.ts`
  - Unit tests for mode predicate and counter patch behavior.
- `viewer/tests/e2e/debug-route-shared-host-diagnostics.spec.ts`
  - Short Playwright proof for normal f32 debug shared-host ownership and rapid month/hour convergence.
- `viewer/tests/e2e/debug-route-parity-runtime-diagnostics.spec.ts`
  - Short Playwright proof for August Python `.bin` positive case and non-August negative validity case.

### Modify

- `viewer/src/routes/debug-webgpu-utci/+page.svelte`
  - Add shared-host state/wiring for normal f32 mode.
  - Gate legacy f32 schedulers/dispatchers off when shared-host mode is active.
  - Publish shared-host diagnostics and legacy dispatch counters.
  - Preserve parity/collect/strict exposure behavior.
- `viewer/src/lib/debug/debugWebgpuUtciDiagnostics.ts`
  - Add diagnostics fields used by shared-host proof.
- `viewer/tests/debug/debug-webgpu-utci-diagnostics.test.ts`
  - Cover new diagnostics fields and honest engine transition.
- `viewer/tests/e2e/debug-route-baseline-diagnostics.spec.ts`
  - Keep proving legacy baseline route still reports `legacy-debug` before migration-specific query is used.
- `viewer/tests/routes/main-route-debug-boundary-source-lock.test.ts`
  - Extend main-route source lock if any new debug helper names or Python/bin diagnostic fields appear.

### Inspect But Do Not Move

- `viewer/src/lib/components/scene/UTCIPointCloud.svelte`
- `viewer/src/lib/components/scene/ComparisonRenderer.svelte`
- `viewer/src/lib/components/scene/utciSurfaceSync.ts`
- `viewer/src/lib/compute/liveUtciSelectedHourSession.ts`

Leave `scene.add`, storage-buffer wait, `invalidate`, GPU buffer disposal, and render lifecycle orchestration inside scene/session components.

## Stop Conditions

- Stop if `/` requests `.bin`, imports debug/parity helpers, or loses `compute-buffer-selected-hour` proof.
- Stop if normal debug f32 has both shared-host dispatch and legacy `runDebugOnDemandSelectedHour` dispatch active.
- Stop if `selectedHourEngine: "shared-host"` appears before shared host owns compute/render.
- Stop if `legacySelectedHourDispatchCount` or `legacyScrubScheduleCount` is nonzero in shared-host normal f32 diagnostics.
- Stop if parity mode loses Python `.bin` comparison or implies non-August Python baseline validity.
- Stop if implementation requires moving Threlte/WebGPU scene lifecycle orchestration out of scene components.
- Stop if a Playwright probe hangs. Capture `window.__utciRenderDiagnostics__` for `/` or `window.__onDemandPrototypeDiagnostics__` for debug, then tighten the probe instead of rerunning unchanged.

---

### Task 0: Baseline Verification And Dirty-Tree Record

**Files:**
- Inspect: `docs/superpowers/plans/2026-05-10-webgpu-debug-route-shared-viewer-rebuild.md`
- Inspect: `viewer/src/routes/+page.svelte`
- Inspect: `viewer/src/routes/debug-webgpu-utci/+page.svelte`
- Inspect: `viewer/src/lib/debug/debugWebgpuUtciDiagnostics.ts`
- Inspect: `viewer/tests/e2e/main-route-manual-diagnostics.spec.ts`
- Inspect: `viewer/tests/e2e/debug-route-baseline-diagnostics.spec.ts`

- [ ] **Step 1: Record current dirty state**

Run:

```powershell
git status --short
```

Expected: dirty files may exist from prior reviewed work. Record them in the task notes and preserve them. Do not revert unrelated files.

- [ ] **Step 2: Run focused Vitest baseline**

Run:

```powershell
cd viewer
npx vitest run tests/compute/live-selected-hour-controller.test.ts tests/compute/live-selected-hour-route-host.test.ts tests/compute/live-selected-hour-route-projection.test.ts tests/routes/main-route-overlay-gating-helper.test.ts tests/scene/utci-surface-sync.test.ts tests/debug/debug-webgpu-utci-diagnostics.test.ts tests/debug/debug-webgpu-utci-query.test.ts tests/diagnostics/main-route-utci-diagnostics.test.ts tests/routes/main-route-debug-boundary-source-lock.test.ts
```

Expected: PASS. If any test fails, use `superpowers:systematic-debugging` and report the failing test before changing implementation.

- [ ] **Step 3: Run main route baseline probe**

Run:

```powershell
cd viewer
npx playwright test tests/e2e/main-route-manual-diagnostics.spec.ts --project=chromium --workers=1 --timeout=30000
```

Expected: PASS with main route proving `gpuNative`, `compute-buffer-selected-hour`, same-device render/compute, no data texture build, and no `.bin` requests.

- [ ] **Step 4: Run debug route baseline probe**

Run:

```powershell
cd viewer
npx playwright test tests/e2e/debug-route-baseline-diagnostics.spec.ts --project=chromium --workers=1 --timeout=30000
```

Expected: PASS with current debug route still honest about `selectedHourEngine === "legacy-debug"` before this rebuild starts.

- [ ] **Step 5: Stop on hangs**

If either Playwright command hangs or times out, do not rerun unchanged. Capture:

```ts
await page.evaluate(() => (window as any).__utciRenderDiagnostics__ ?? null);
await page.evaluate(() => (window as any).__onDemandPrototypeDiagnostics__ ?? null);
```

Expected: diagnostics are included in the report, and the next probe is shortened or tightened before another run.

### Task 1: Add Pure Debug Shared-Host Mode Predicate And Counter Shape

**Files:**
- Create: `viewer/src/lib/debug/debugSelectedHourMode.ts`
- Create: `viewer/tests/debug/debug-selected-hour-mode.test.ts`

- [ ] **Step 1: Write the failing predicate and counter tests**

Create `viewer/tests/debug/debug-selected-hour-mode.test.ts`:

```ts
import { describe, expect, it } from 'vitest';
import {
	buildDebugSelectedHourDispatchCounters,
	shouldUseDebugSharedSelectedHourHost
} from '$lib/debug/debugSelectedHourMode';

describe('debug selected-hour mode', () => {
	it('uses the shared selected-hour host only for normal non-parity f32 mode', () => {
		expect(
			shouldUseDebugSharedSelectedHourHost({
				onDemandPrototypeEnabled: true,
				debugOnDemandMode: 'f32',
				parityMode: false,
				normalCollectMode: false,
				strictExposureOnlyEnabled: false,
				compareOneHourEnabled: false
			})
		).toBe(true);
	});

	it('keeps parity, collect, strict exposure, and one-hour comparison on legacy debug paths', () => {
		const base = {
			onDemandPrototypeEnabled: true,
			debugOnDemandMode: 'f32' as const,
			parityMode: false,
			normalCollectMode: false,
			strictExposureOnlyEnabled: false,
			compareOneHourEnabled: false
		};

		expect(shouldUseDebugSharedSelectedHourHost({ ...base, parityMode: true })).toBe(false);
		expect(shouldUseDebugSharedSelectedHourHost({ ...base, normalCollectMode: true })).toBe(false);
		expect(shouldUseDebugSharedSelectedHourHost({ ...base, strictExposureOnlyEnabled: true })).toBe(false);
		expect(shouldUseDebugSharedSelectedHourHost({ ...base, compareOneHourEnabled: true })).toBe(false);
		expect(shouldUseDebugSharedSelectedHourHost({ ...base, debugOnDemandMode: 'off' })).toBe(false);
		expect(shouldUseDebugSharedSelectedHourHost({ ...base, onDemandPrototypeEnabled: false })).toBe(false);
	});

	it('builds explicit legacy dispatch counters for diagnostics', () => {
		expect(
			buildDebugSelectedHourDispatchCounters({
				legacySelectedHourDispatchCount: 2,
				legacyScrubScheduleCount: 3
			})
		).toEqual({
			legacySelectedHourDispatchCount: 2,
			legacyScrubScheduleCount: 3
		});
	});
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```powershell
cd viewer
npx vitest run tests/debug/debug-selected-hour-mode.test.ts
```

Expected: FAIL because `$lib/debug/debugSelectedHourMode` does not exist.

- [ ] **Step 3: Implement the pure helper**

Create `viewer/src/lib/debug/debugSelectedHourMode.ts`:

```ts
export type DebugSelectedHourModeInput = {
	onDemandPrototypeEnabled: boolean;
	debugOnDemandMode: 'off' | 'f32';
	parityMode: boolean;
	normalCollectMode: boolean;
	strictExposureOnlyEnabled: boolean;
	compareOneHourEnabled: boolean;
};

export type DebugSelectedHourDispatchCounters = {
	legacySelectedHourDispatchCount: number;
	legacyScrubScheduleCount: number;
};

export function shouldUseDebugSharedSelectedHourHost(
	input: DebugSelectedHourModeInput
): boolean {
	return (
		input.onDemandPrototypeEnabled &&
		input.debugOnDemandMode === 'f32' &&
		!input.parityMode &&
		!input.normalCollectMode &&
		!input.strictExposureOnlyEnabled &&
		!input.compareOneHourEnabled
	);
}

export function buildDebugSelectedHourDispatchCounters(
	counters: DebugSelectedHourDispatchCounters
): DebugSelectedHourDispatchCounters {
	return {
		legacySelectedHourDispatchCount: counters.legacySelectedHourDispatchCount,
		legacyScrubScheduleCount: counters.legacyScrubScheduleCount
	};
}
```

- [ ] **Step 4: Run the helper tests**

Run:

```powershell
cd viewer
npx vitest run tests/debug/debug-selected-hour-mode.test.ts
```

Expected: PASS.

---

### Task 2: Extend Debug Diagnostics State Without Changing Route Behavior

**Files:**
- Modify: `viewer/src/lib/debug/debugWebgpuUtciDiagnostics.ts`
- Modify: `viewer/tests/debug/debug-webgpu-utci-diagnostics.test.ts`

- [ ] **Step 1: Write failing diagnostics tests**

Append these tests to `viewer/tests/debug/debug-webgpu-utci-diagnostics.test.ts`:

```ts
it('keeps shared-host diagnostics honest before migration is proven', () => {
	const state = deriveDebugWebgpuUtciDiagnosticsState({
		parityMode: false,
		collectMode: 'off',
		debugOnDemandMode: 'f32',
		utciRenderMode: 'auto',
		selectedMonthIndex: 7,
		selectedHourIndex: 0,
		selectedTimeIndex: 168,
		selectedHourEngine: 'legacy-debug',
		legacySelectedHourDispatchCount: 1,
		legacyScrubScheduleCount: 1
	});

	expect(state.selectedHourEngine).toBe('legacy-debug');
	expect(state.legacySelectedHourDispatchCount).toBe(1);
	expect(state.legacyScrubScheduleCount).toBe(1);
});

it('allows shared-host diagnostics only with zero legacy dispatch counters', () => {
	const state = deriveDebugWebgpuUtciDiagnosticsState({
		parityMode: false,
		collectMode: 'off',
		debugOnDemandMode: 'f32',
		utciRenderMode: 'auto',
		selectedMonthIndex: 7,
		selectedHourIndex: 0,
		selectedTimeIndex: 168,
		selectedHourEngine: 'shared-host',
		legacySelectedHourDispatchCount: 0,
		legacyScrubScheduleCount: 0
	});

	expect(state.selectedHourEngine).toBe('shared-host');
	expect(state.legacySelectedHourDispatchCount).toBe(0);
	expect(state.legacyScrubScheduleCount).toBe(0);
});
```

- [ ] **Step 2: Run the diagnostics tests to verify they fail**

Run:

```powershell
cd viewer
npx vitest run tests/debug/debug-webgpu-utci-diagnostics.test.ts
```

Expected: FAIL because `legacySelectedHourDispatchCount` and `legacyScrubScheduleCount` are not part of the diagnostics helper yet.

- [ ] **Step 3: Add diagnostics fields**

Modify `viewer/src/lib/debug/debugWebgpuUtciDiagnostics.ts`:

```ts
export type DebugWebgpuUtciDiagnosticsInputs = {
	parityMode: boolean;
	collectMode: 'off' | 'normal';
	debugOnDemandMode: 'off' | 'f32';
	utciRenderMode: UtciRenderMode;
	selectedMonthIndex: number;
	selectedHourIndex: number;
	selectedTimeIndex: number;
	selectedHourEngine?: DebugSelectedHourEngine;
	legacySelectedHourDispatchCount?: number;
	legacyScrubScheduleCount?: number;
};

export type DebugWebgpuUtciDiagnosticsState = {
	onDemandEnabled: boolean;
	binComparisonEnabled: boolean;
	collectNormalMode: boolean;
	windowDiagnosticsEnabled: boolean;
	renderMode: UtciRenderMode;
	selectedHourEngine: DebugSelectedHourEngine;
	legacySelectedHourDispatchCount: number;
	legacyScrubScheduleCount: number;
	selection: {
		monthIndex: number;
		hourIndex: number;
		timeIndex: number;
	};
};
```

Inside `deriveDebugWebgpuUtciDiagnosticsState`, include:

```ts
legacySelectedHourDispatchCount: inputs.legacySelectedHourDispatchCount ?? 0,
legacyScrubScheduleCount: inputs.legacyScrubScheduleCount ?? 0,
```

- [ ] **Step 4: Run diagnostics tests**

Run:

```powershell
cd viewer
npx vitest run tests/debug/debug-webgpu-utci-diagnostics.test.ts tests/debug/debug-selected-hour-mode.test.ts
```

Expected: PASS.

---

### Task 3: Add Route-Level Shared-Host Predicate And Legacy Dispatch Counters

**Files:**
- Modify: `viewer/src/routes/debug-webgpu-utci/+page.svelte`
- Modify: `viewer/tests/e2e/debug-route-baseline-diagnostics.spec.ts`

- [ ] **Step 1: Add a failing baseline diagnostics assertion**

In `viewer/tests/e2e/debug-route-baseline-diagnostics.spec.ts`, after the existing `selectedHourEngine` assertion, add:

```ts
expect(diagnostics.legacySelectedHourDispatchCount).toBeGreaterThanOrEqual(1);
expect(diagnostics.legacyScrubScheduleCount).toBeGreaterThanOrEqual(0);
```

- [ ] **Step 2: Run the baseline probe to verify it fails**

Run:

```powershell
cd viewer
npx playwright test tests/e2e/debug-route-baseline-diagnostics.spec.ts --project=chromium --workers=1 --timeout=30000
```

Expected: FAIL because the debug route does not publish legacy dispatch counters yet.

- [ ] **Step 3: Import the helper and add route counters**

In `viewer/src/routes/debug-webgpu-utci/+page.svelte`, import:

```ts
import {
	buildDebugSelectedHourDispatchCounters,
	shouldUseDebugSharedSelectedHourHost
} from '$lib/debug/debugSelectedHourMode';
```

Near the existing debug selected-hour state, add:

```ts
let legacySelectedHourDispatchCount = 0;
let legacyScrubScheduleCount = 0;
```

Add the shared-host predicate after `onDemandPrototypeEnabled`, `strictExposureOnlyEnabled`, and compare-mode reactive values are available:

```ts
$: useDebugSharedSelectedHourHost = shouldUseDebugSharedSelectedHourHost({
	onDemandPrototypeEnabled,
	debugOnDemandMode,
	parityMode,
	normalCollectMode,
	strictExposureOnlyEnabled,
	compareOneHourEnabled
});
```

Update the `deriveDebugWebgpuUtciDiagnosticsState(...)` call:

```ts
legacySelectedHourDispatchCount,
legacyScrubScheduleCount
```

Update `updateOnDemandPrototypeDiagnostics(...)` so the published payload includes:

```ts
...buildDebugSelectedHourDispatchCounters({
	legacySelectedHourDispatchCount,
	legacyScrubScheduleCount
}),
```

- [ ] **Step 4: Increment counters in legacy paths only**

Inside `scheduleDebugOnDemandScrubRecompute(...)`, before the `setTimeout`, add:

```ts
legacyScrubScheduleCount += 1;
```

Inside `runDebugOnDemandSelectedHour(...)`, before compute starts and only after checking the request is not aborted, add:

```ts
legacySelectedHourDispatchCount += 1;
```

- [ ] **Step 5: Run baseline diagnostics**

Run:

```powershell
cd viewer
npx playwright test tests/e2e/debug-route-baseline-diagnostics.spec.ts --project=chromium --workers=1 --timeout=30000
```

Expected: PASS. The route still reports `selectedHourEngine === "legacy-debug"`.

---

### Task 4: Wire Shared Host For Normal Debug f32 Without Enabling Shared Engine Diagnostics

**Files:**
- Modify: `viewer/src/routes/debug-webgpu-utci/+page.svelte`
- Modify: `viewer/tests/debug/debug-webgpu-utci-diagnostics.test.ts`

- [ ] **Step 1: Add a unit-level test that preserves impossible shared-host evidence**

In `viewer/tests/debug/debug-webgpu-utci-diagnostics.test.ts`, add:

```ts
it('preserves shared-host counter evidence when legacy counters are nonzero', () => {
	const state = deriveDebugWebgpuUtciDiagnosticsState({
		parityMode: false,
		collectMode: 'off',
		debugOnDemandMode: 'f32',
		utciRenderMode: 'auto',
		selectedMonthIndex: 7,
		selectedHourIndex: 0,
		selectedTimeIndex: 168,
		selectedHourEngine: 'shared-host',
		legacySelectedHourDispatchCount: 1,
		legacyScrubScheduleCount: 0
	});

	expect(state.selectedHourEngine).toBe('shared-host');
	expect(state.legacySelectedHourDispatchCount).toBe(1);
});
```

This test documents the diagnostic state. The route-level Playwright test in Task 5 will enforce that shared-host mode only passes with zero legacy counters.

- [ ] **Step 2: Run diagnostics tests**

Run:

```powershell
cd viewer
npx vitest run tests/debug/debug-webgpu-utci-diagnostics.test.ts
```

Expected: PASS.

- [ ] **Step 3: Add shared host state mirroring main route**

In `viewer/src/routes/debug-webgpu-utci/+page.svelte`, import:

```ts
import { createLiveSelectedHourRouteHost } from '$lib/compute/liveSelectedHourRouteHost';
import { projectMainRouteLiveSceneState } from '$lib/compute/liveSelectedHourRouteProjection';
import type { LiveSelectedHourControllerSurfaceDiagnostics } from '$lib/compute/liveSelectedHourController';
import type { LiveSelectedHourSurfaceIdentity } from '$lib/compute/liveSelectedHourSurfaceIdentity';
import type { LiveSelectedHourPublishedRenderContext } from '$lib/compute/liveSelectedHourRenderContext';
import type { SelectedHourGpuResidentOutput } from '$lib/compute/liveUtciSelectedHourSession';
```

Add state and diagnostics handoff:

```ts
const debugSharedRouteHost = createLiveSelectedHourRouteHost({
	dataBasePath: getDataBasePath()
});
let debugSharedRouteState = debugSharedRouteHost.getState();
const unsubscribeDebugSharedRouteHost = debugSharedRouteHost.subscribe((state) => {
	debugSharedRouteState = state;
});

let debugSharedBaseLiveReady = false;
let debugSharedBaseHasVisibleSurface = false;
let debugSharedBaseSceneAnalysis: Analysis | null = null;
let debugSharedBaseRenderContext: LiveSelectedHourPublishedRenderContext | null = null;
let debugSharedBaseSurfaceIdentity: LiveSelectedHourSurfaceIdentity | null = null;
let debugSharedBasePendingGpuResidentOutput: SelectedHourGpuResidentOutput | null = null;
let debugSharedBasePendingRenderUpdateStartedAt: number | undefined = undefined;

function handleDebugSharedBaseSurfaceDiagnostics(
	diagnostics: LiveSelectedHourControllerSurfaceDiagnostics
): void {
	debugSharedRouteHost.handleBaseSurfaceDiagnostics(diagnostics);
}
```

In `onDestroy`, call:

```ts
unsubscribeDebugSharedRouteHost();
debugSharedRouteHost.dispose();
```

- [ ] **Step 4: Feed shared host only in normal f32 mode**

Add:

```ts
$: debugSharedRouteHost.setRouteInputs({
	enabled: useDebugSharedSelectedHourHost,
	analysisId,
	baseAnalysis: $analysisStore,
	baseModel: modelFileForLoadedModel === $analysisStore?.metadata.model_file ? model : null,
	selection: {
		monthIndex: debugOnDemandSelection.monthIndex,
		hourIndex: debugOnDemandSelection.hourIndex,
		timeIndex: debugOnDemandSelection.timeIndex,
		selectionKey: [analysisId, debugOnDemandSelection.monthIndex, debugOnDemandSelection.hourIndex].join('|')
	},
	colorMode: $viewerStore.colorMode,
	utciRenderMode,
	rendererBackend,
	rendererDevice: rendererDeviceForDebug,
	utciSurfaceBackend: resolvedUtciSurfaceBackend,
	comparison: {
		active: false,
		analysisId: null,
		sourceAnalysis: null,
		model: null,
		rendererDevice: rendererDeviceForDebug
	}
});
```

`rendererDeviceForDebug` already exists in `viewer/src/routes/debug-webgpu-utci/+page.svelte` and is updated by `handleRendererDiagnostics(...)`. Do not introduce a second renderer-device variable.

- [ ] **Step 5: Project shared host scene state**

Add:

```ts
$: ({
	baseLiveReady: debugSharedBaseLiveReady,
	baseHasVisibleLiveSurface: debugSharedBaseHasVisibleSurface,
	baseSceneAnalysis: debugSharedBaseSceneAnalysis,
	baseSceneRenderContext: debugSharedBaseRenderContext,
	baseSceneSurfaceIdentity: debugSharedBaseSurfaceIdentity,
	basePendingGpuResidentOutput: debugSharedBasePendingGpuResidentOutput,
	basePendingRenderUpdateStartedAt: debugSharedBasePendingRenderUpdateStartedAt
} = projectMainRouteLiveSceneState({
	useLiveUtciOnMainRoute: useDebugSharedSelectedHourHost,
	isComparing: false,
	baseAnalysis: $analysisStore,
	comparisonAnalysis: null,
	liveRouteState: debugSharedRouteState
}));
```

- [ ] **Step 6: Keep diagnostics legacy until duplicate dispatch is disabled**

Do not set `selectedHourEngine: "shared-host"` yet. Keep:

```ts
selectedHourEngine: 'legacy-debug'
```

- [ ] **Step 7: Run focused unit checks**

Run:

```powershell
cd viewer
npx vitest run tests/debug/debug-webgpu-utci-diagnostics.test.ts tests/compute/live-selected-hour-route-host.test.ts tests/compute/live-selected-hour-route-projection.test.ts
```

Expected: PASS.

---

### Task 5: Disable Legacy Normal f32 Dispatch When Shared Host Is Active

**Files:**
- Modify: `viewer/src/routes/debug-webgpu-utci/+page.svelte`
- Create: `viewer/tests/e2e/debug-route-shared-host-diagnostics.spec.ts`

- [ ] **Step 1: Write the failing shared-host Playwright probe**

Create `viewer/tests/e2e/debug-route-shared-host-diagnostics.spec.ts`:

```ts
import { expect, test, type Page } from '@playwright/test';

async function readDebugDiagnostics(page: Page) {
	return page.evaluate(() => {
		return (window as any).__onDemandPrototypeDiagnostics__ ?? null;
	});
}

async function waitForSharedHostPublication(
	page: Page,
	options?: { previousRequestId?: number; expectedSelectionKey?: string }
) {
	const handle = await page
		.waitForFunction(
			(args) => {
				const value = (window as any).__onDemandPrototypeDiagnostics__;
				if (!value) return null;
				if (
					value.selectedHourEngine === 'shared-host' &&
					value.utciSurfaceSource === 'compute-buffer-selected-hour' &&
					value.renderTransport === 'compute-buffer-selected-hour' &&
					value.sameDeviceForComputeAndRender === true &&
					value.legacySelectedHourDispatchCount === 0 &&
					value.legacyScrubScheduleCount === 0 &&
					(typeof args.previousRequestId !== 'number' ||
						value.surfaceRequestId !== args.previousRequestId) &&
					(!args.expectedSelectionKey || value.selectionKey === args.expectedSelectionKey) &&
					(!args.expectedSelectionKey || value.sceneSelectionKey === args.expectedSelectionKey)
				) {
					return value;
				}
				return null;
			},
			{
				previousRequestId: options?.previousRequestId,
				expectedSelectionKey: options?.expectedSelectionKey
			},
			{ timeout: 15_000 }
		)
		.catch(async (error) => {
			const lastDiagnostics = await readDebugDiagnostics(page);
			throw new Error(
				[
					'Timed out waiting for debug shared-host diagnostics.',
					error instanceof Error ? error.message : String(error),
					'Last window.__onDemandPrototypeDiagnostics__:',
					JSON.stringify(lastDiagnostics, null, 2)
				].join('\n')
			);
		});
	return handle.jsonValue() as Promise<any>;
}

test.describe('debug route shared-host selected-hour diagnostics', () => {
	test.afterEach(async ({ page }) => {
		await page.goto('about:blank').catch(() => undefined);
	});

	test('uses shared host for normal f32 selected-hour rendering without legacy dispatch', async ({
		page
	}) => {
		test.setTimeout(30_000);
		const analysisId = 'Ben-Gurion/20250815_grid_2m_fullday';
		await page.goto(
			'/debug-webgpu-utci?analysis=Ben-Gurion%2F20250815_grid_2m_fullday&utciRender=gpu'
		);

		const initial = await waitForSharedHostPublication(page, {
			expectedSelectionKey: `${analysisId}|7|0`
		});

		expect(initial.surfaceRequestId).toBeGreaterThan(0);
		expect(initial.sceneSurfaceRequestId).toBe(initial.surfaceRequestId);
		expect(initial.selectedTimeIndex).toBe(initial.renderContextTimeIndex);
		expect(initial.acceptedUtciRange).toEqual({
			min: expect.any(Number),
			max: expect.any(Number)
		});
	});
});
```

- [ ] **Step 2: Run the shared-host probe to verify it fails**

Run:

```powershell
cd viewer
npx playwright test tests/e2e/debug-route-shared-host-diagnostics.spec.ts --project=chromium --workers=1 --timeout=30000
```

Expected: FAIL because the route still reports `selectedHourEngine: "legacy-debug"` and has legacy dispatch counters.

- [ ] **Step 3: Gate off legacy scheduling**

In the reactive block that calls `scheduleDebugOnDemandScrubRecompute(nextScrubTriggerKey)`, add `!useDebugSharedSelectedHourHost` to the condition:

```ts
browser &&
mounted &&
onDemandPrototypeEnabled &&
debugOnDemandMode === "f32" &&
!useDebugSharedSelectedHourHost &&
!strictExposureOnlyEnabled &&
```

In the branch that resets legacy scheduling state, include `useDebugSharedSelectedHourHost`:

```ts
useDebugSharedSelectedHourHost ||
!browser ||
!onDemandPrototypeEnabled ||
```

- [ ] **Step 4: Gate off legacy dispatch calls**

Guard every normal-path call to `runDebugOnDemandSelectedHour(...)` with:

```ts
if (useDebugSharedSelectedHourHost) {
	return;
}
```

Apply this guard at these exact search targets:

```ts
await runDebugOnDemandSelectedHour({
	monthIndex: debugOnDemandSelection.monthIndex,
```

```ts
void runDebugOnDemandSelectedHour({
	monthIndex: retryMonthIndex,
```

```ts
const result = await runDebugOnDemandSelectedHour({
	monthIndex: latestSelection.monthIndex,
```

```ts
scheduleDebugOnDemandScrubRecompute(nextScrubTriggerKey);
```

```ts
scheduleDebugOnDemandScrubRecompute(debugOnDemandSelectionKey);
```

Do not guard parity/strict/collect comparison calls that still intentionally use legacy debug instrumentation.

- [ ] **Step 5: Route normal f32 scene props from shared projection**

Find this live debug block:

```svelte
{#if liveAnalysis || acceptedGpuResidentUtciOutput}
	<UTCIPointCloud
		analysis={liveAnalysis ?? $analysisStore}
		{model}
		bind:utciSurface={liveUtciMesh}
		utciSurfaceBackend={resolvedUtciSurfaceBackend}
		acceptedGpuResidentOutput={acceptedGpuResidentUtciOutput}
		pendingRenderUpdateStartedAt={onDemandDebugPrepared?.pendingRenderUpdate?.startedAt}
		onUtciSurfaceDiagnostics={handleLiveUtciSurfaceDiagnostics}
	/>
{/if}
```

Replace it with:

```svelte
{#if liveAnalysis || acceptedGpuResidentUtciOutput || debugSharedBaseSceneAnalysis || debugSharedBasePendingGpuResidentOutput}
	<UTCIPointCloud
		analysis={useDebugSharedSelectedHourHost ? debugSharedBaseSceneAnalysis : liveAnalysis ?? $analysisStore}
		{model}
		bind:utciSurface={liveUtciMesh}
		utciSurfaceBackend={resolvedUtciSurfaceBackend}
		acceptedGpuResidentOutput={useDebugSharedSelectedHourHost ? debugSharedBasePendingGpuResidentOutput : acceptedGpuResidentUtciOutput}
		selectedHourRenderContext={useDebugSharedSelectedHourHost ? debugSharedBaseRenderContext : null}
		liveSelectedHourSurfaceIdentity={useDebugSharedSelectedHourHost ? debugSharedBaseSurfaceIdentity : null}
		pendingRenderUpdateStartedAt={useDebugSharedSelectedHourHost ? debugSharedBasePendingRenderUpdateStartedAt : onDemandDebugPrepared?.pendingRenderUpdate?.startedAt}
		onUtciSurfaceDiagnostics={useDebugSharedSelectedHourHost ? handleDebugSharedBaseSurfaceDiagnostics : handleLiveUtciSurfaceDiagnostics}
	/>
{/if}
```

Keep all scene component lifecycle code inside `UTCIPointCloud.svelte`. Do not move GPU buffer attachment, `scene.add`, `invalidate`, or disposal logic into the route.

- [ ] **Step 6: Publish shared-host diagnostics**

When `useDebugSharedSelectedHourHost` is true, merge these fields into `updateOnDemandPrototypeDiagnostics(...)`:

```ts
selectedHourEngine: 'shared-host',
legacySelectedHourDispatchCount,
legacyScrubScheduleCount,
surfaceRequestId: debugSharedRouteState.baseSurfaceIdentity?.requestId,
selectionKey: debugSharedRouteState.baseSurfaceIdentity?.selectionKey,
sceneSurfaceRequestId: debugSharedBaseSurfaceIdentity?.requestId,
sceneSelectionKey: debugSharedBaseSurfaceIdentity?.selectionKey,
selectedMonthIndex: debugOnDemandSelection.monthIndex,
selectedHourIndex: debugOnDemandSelection.hourIndex,
selectedTimeIndex: debugOnDemandSelection.timeIndex,
renderContextTimeIndex: debugSharedBaseRenderContext?.timeIndex,
acceptedUtciRange: debugSharedBasePendingGpuResidentOutput?.utciRange,
renderTransport: debugSharedRouteState.base.renderTransport,
sameDeviceForComputeAndRender: debugSharedRouteState.base.sameDeviceForComputeAndRender,
utciSurfaceSource: debugSharedRouteState.base.renderSurfaceDiagnostics.utciSurfaceSource
```

Only report `selectedHourEngine: 'shared-host'` in this branch.

Do not reset, overwrite, or hardcode the legacy counters in this branch. The Playwright proof must fail if a missed legacy path increments either counter after shared-host mode becomes active.

- [ ] **Step 7: Run the shared-host Playwright probe**

Run:

```powershell
cd viewer
npx playwright test tests/e2e/debug-route-shared-host-diagnostics.spec.ts --project=chromium --workers=1 --timeout=30000
```

Expected: PASS.

- [ ] **Step 8: Run the legacy debug baseline probe**

Run:

```powershell
cd viewer
npx playwright test tests/e2e/debug-route-baseline-diagnostics.spec.ts --project=chromium --workers=1 --timeout=30000
```

Expected: PASS. The baseline spec must still prove debug-only legacy/parity branches are available. If the default normal debug route now reports `shared-host`, add a query in this spec that enables parity mode and asserts `selectedHourEngine === "legacy-debug"` instead of weakening the check.

---

### Task 6: Add Debug Rapid Month/Hour Convergence Proof

**Files:**
- Modify: `viewer/tests/e2e/debug-route-shared-host-diagnostics.spec.ts`

- [ ] **Step 1: Add a failing rapid selection test**

Append to `viewer/tests/e2e/debug-route-shared-host-diagnostics.spec.ts`:

```ts
async function setDebugHourSelection(page: Page, hourIndex: number) {
	const modeButton = page.getByRole('button', { name: /^day$/i });
	await expect(modeButton).toBeVisible();
	await modeButton.click();
	const slider = page.getByRole('slider', { name: /select analysis hour/i });
	await expect(slider).toBeVisible();
	await slider.click();
	await slider.focus();
	await slider.press('Home');
	for (let step = 0; step < hourIndex; step += 1) {
		await slider.press('ArrowRight');
	}
	await expect(slider).toHaveAttribute('aria-valuenow', String(hourIndex));
}

async function setDebugMonthSelection(page: Page, monthIndex: number) {
	const modeButton = page.getByRole('button', { name: /^month$/i });
	await expect(modeButton).toBeVisible();
	await modeButton.click();
	const slider = page.getByRole('slider', { name: /select month/i });
	await expect(slider).toBeVisible();
	await slider.click();
	await slider.focus();
	await slider.press('Home');
	for (let step = 0; step < monthIndex; step += 1) {
		await slider.press('ArrowRight');
	}
	await expect(slider).toHaveAttribute('aria-valuenow', String(monthIndex));
}

test('converges shared-host diagnostics after rapid month and hour changes', async ({ page }) => {
	test.setTimeout(30_000);
	const analysisId = 'Ben-Gurion/20250815_grid_2m_fullday';
	await page.goto(
		'/debug-webgpu-utci?analysis=Ben-Gurion%2F20250815_grid_2m_fullday&utciRender=gpu'
	);

	const initial = await waitForSharedHostPublication(page, {
		expectedSelectionKey: `${analysisId}|7|0`
	});

	await setDebugHourSelection(page, 2);
	await setDebugMonthSelection(page, 8);
	await setDebugHourSelection(page, 3);

	const finalValue = await waitForSharedHostPublication(page, {
		previousRequestId: initial.surfaceRequestId,
		expectedSelectionKey: `${analysisId}|8|3`
	});

	expect(finalValue.selectedTimeIndex).toBe(8 * 24 + 3);
	expect(finalValue.renderContextTimeIndex).toBe(8 * 24 + 3);
	expect(finalValue.selectionKey).toBe(finalValue.sceneSelectionKey);
	expect(finalValue.surfaceRequestId).toBe(finalValue.sceneSurfaceRequestId);
	expect(finalValue.acceptedUtciRange).toEqual({
		min: expect.any(Number),
		max: expect.any(Number)
	});
	expect(finalValue.legacySelectedHourDispatchCount).toBe(0);
	expect(finalValue.legacyScrubScheduleCount).toBe(0);
});
```

- [ ] **Step 2: Run the new convergence test**

Run:

```powershell
cd viewer
npx playwright test tests/e2e/debug-route-shared-host-diagnostics.spec.ts --project=chromium --workers=1 --timeout=30000
```

Expected: PASS.

---

### Task 7: Add Runtime Parity Validity Diagnostics

**Files:**
- Modify: `viewer/src/lib/debug/debugWebgpuUtciDiagnostics.ts`
- Modify: `viewer/src/routes/debug-webgpu-utci/+page.svelte`
- Modify: `viewer/tests/debug/debug-webgpu-utci-diagnostics.test.ts`
- Create: `viewer/tests/e2e/debug-route-parity-runtime-diagnostics.spec.ts`

- [ ] **Step 1: Write failing diagnostics unit tests**

Add tests to `viewer/tests/debug/debug-webgpu-utci-diagnostics.test.ts`:

```ts
it('carries bin comparison validity separately from parity enablement', () => {
	const state = deriveDebugWebgpuUtciDiagnosticsState({
		parityMode: true,
		collectMode: 'off',
		debugOnDemandMode: 'f32',
		utciRenderMode: 'auto',
		selectedMonthIndex: 3,
		selectedHourIndex: 9,
		selectedTimeIndex: 81,
		selectedHourEngine: 'legacy-debug',
		binComparisonValid: false
	});

	expect(state.binComparisonEnabled).toBe(true);
	expect(state.binComparisonValid).toBe(false);
});
```

- [ ] **Step 2: Run diagnostics tests to verify failure**

Run:

```powershell
cd viewer
npx vitest run tests/debug/debug-webgpu-utci-diagnostics.test.ts
```

Expected: FAIL because `binComparisonValid` is not part of diagnostics state.

- [ ] **Step 3: Add `binComparisonValid` to diagnostics state**

In `viewer/src/lib/debug/debugWebgpuUtciDiagnostics.ts`, add optional input and state field:

```ts
binComparisonValid?: boolean;
```

and:

```ts
binComparisonValid: inputs.binComparisonValid ?? false,
```

- [ ] **Step 4: Thread query validity into route diagnostics**

In `viewer/src/routes/debug-webgpu-utci/+page.svelte`, pass:

```ts
binComparisonValid: debugQueryState.binComparisonValid
```

to `deriveDebugWebgpuUtciDiagnosticsState(...)`, and publish:

```ts
binComparisonValid:
	diagnostics.binComparisonValid ??
	existing?.binComparisonValid ??
	debugDiagnosticsState.binComparisonValid,
```

inside `updateOnDemandPrototypeDiagnostics(...)`.

- [ ] **Step 5: Prevent non-August Python `.bin` comparison from becoming active**

Where `shouldReadbackForComparison` is computed, change:

```ts
const shouldReadbackForComparison = params.readbackForComparison && parityMode;
```

to:

```ts
const shouldReadbackForComparison =
	params.readbackForComparison && parityMode && debugDiagnosticsState.binComparisonValid;
```

When invalid, publish:

```ts
pythonBaselineStatus: parityMode && !debugDiagnosticsState.binComparisonValid
	? 'unavailable-non-august'
	: undefined,
debugComparisonReference: undefined,
pythonBinComparisonActive: false
```

Add `pythonBaselineStatus?: 'available-august' | 'unavailable-non-august'` to the route diagnostics type.

- [ ] **Step 6: Write parity runtime E2E**

Create `viewer/tests/e2e/debug-route-parity-runtime-diagnostics.spec.ts`:

```ts
import { expect, test } from '@playwright/test';

test.describe('debug route parity runtime diagnostics', () => {
	test.afterEach(async ({ page }) => {
		await page.goto('about:blank').catch(() => undefined);
	});

	test('allows August Python bin comparison only on debug route', async ({ page }) => {
		test.setTimeout(30_000);
		await page.goto(
			'/debug-webgpu-utci?analysis=Ben-Gurion%2F20250815_grid_2m_fullday&utciRender=gpu&parity=1&utciOnDemand=f32&monthIndex=7&timeIndex=168'
		);

		const handle = await page.waitForFunction(() => {
			const value = (window as any).__onDemandPrototypeDiagnostics__;
			if (
				value?.binComparisonEnabled === true &&
				value?.binComparisonValid === true &&
				value?.pythonBinComparisonActive === true &&
				value?.debugComparisonReference === 'python-bin'
			) {
				return value;
			}
			return null;
		}, undefined, { timeout: 20_000 });

		const diagnostics = await handle.jsonValue() as any;
		expect(diagnostics.selectedHourEngine).toBe('legacy-debug');
	});

	test('does not claim non-August Python bin validity', async ({ page }) => {
		test.setTimeout(30_000);
		await page.goto(
			'/debug-webgpu-utci?analysis=Ben-Gurion%2F20250815_grid_2m_fullday&utciRender=gpu&parity=1&utciOnDemand=f32&monthIndex=3&timeIndex=81'
		);

		const handle = await page.waitForFunction(() => {
			const value = (window as any).__onDemandPrototypeDiagnostics__;
			if (value?.binComparisonEnabled === true && value?.binComparisonValid === false) {
				return value;
			}
			return null;
		}, undefined, { timeout: 20_000 });

		const diagnostics = await handle.jsonValue() as any;
		expect(diagnostics.pythonBinComparisonActive).not.toBe(true);
		expect(diagnostics.debugComparisonReference).not.toBe('python-bin');
		expect(diagnostics.pythonBinSampleComparison).toBeUndefined();
		expect(diagnostics.pythonBaselineStatus).toBe('unavailable-non-august');
	});
});
```

- [ ] **Step 7: Run parity runtime tests**

Run:

```powershell
cd viewer
npx vitest run tests/debug/debug-webgpu-utci-diagnostics.test.ts tests/debug/debug-webgpu-utci-query.test.ts
npx playwright test tests/e2e/debug-route-parity-runtime-diagnostics.spec.ts --project=chromium --workers=1 --timeout=30000
```

Expected: PASS.

---

### Task 8: Strengthen Main Route Debug Boundary Proof

**Files:**
- Modify: `viewer/tests/routes/main-route-debug-boundary-source-lock.test.ts`
- Modify: `viewer/tests/e2e/main-route-manual-diagnostics.spec.ts`

- [ ] **Step 1: Extend source-lock forbidden patterns**

In `viewer/tests/routes/main-route-debug-boundary-source-lock.test.ts`, add to strict forbidden patterns:

```ts
/pythonBin/i,
/binComparison/i,
/__onDemandPrototypeDiagnostics__/,
/parityMode/i
```

Add `src/lib/diagnostics/mainRouteUtciDiagnostics.ts` to the strict protected list if it is not already present.

- [ ] **Step 2: Run source-lock test**

Run:

```powershell
cd viewer
npx vitest run tests/routes/main-route-debug-boundary-source-lock.test.ts
```

Expected: PASS.

- [ ] **Step 3: Add suspicious-query main-route request guard test**

In `viewer/tests/e2e/main-route-manual-diagnostics.spec.ts`, add:

```ts
test('ignores debug parity query params on the main route without bin requests', async ({ page }) => {
	test.setTimeout(30_000);
	await page.goto(
		'/?analysis=Ben-Gurion%2F20250815_grid_2m_fullday&utciRender=auto&utciRenderDiagnostics=1&parity=1&utciOnDemand=f32&monthIndex=7'
	);

	const value = await waitForSelectedHourPublication(page, {
		expectedSelectionKey: 'Ben-Gurion/20250815_grid_2m_fullday|7|0'
	});

	expect(value.utciRenderResolved).toBe('gpuNative');
	expect(JSON.stringify(value)).not.toMatch(/pythonBin|binComparison|__onDemandPrototypeDiagnostics__|parityMode/i);
});
```

The existing `afterEach` request guard must remain in place.

- [ ] **Step 4: Run main-route probe**

Run:

```powershell
cd viewer
npx playwright test tests/e2e/main-route-manual-diagnostics.spec.ts --project=chromium --workers=1 --timeout=30000
```

Expected: existing tests plus the new suspicious-query test pass.

---

### Task 9: Final Verification And Review Stop

**Files:**
- Inspect: `viewer/src/routes/+page.svelte`
- Inspect: `viewer/src/routes/debug-webgpu-utci/+page.svelte`
- Inspect: `viewer/src/lib/debug/debugSelectedHourMode.ts`
- Inspect: `viewer/src/lib/debug/debugWebgpuUtciDiagnostics.ts`
- Inspect: `viewer/tests/e2e/debug-route-shared-host-diagnostics.spec.ts`
- Inspect: `viewer/tests/e2e/debug-route-parity-runtime-diagnostics.spec.ts`

- [ ] **Step 1: Run focused unit verification**

Run:

```powershell
cd viewer
npx vitest run tests/diagnostics/main-route-utci-diagnostics.test.ts tests/routes/main-route-debug-boundary-source-lock.test.ts tests/debug/debug-selected-hour-mode.test.ts tests/debug/debug-webgpu-utci-query.test.ts tests/debug/debug-webgpu-utci-diagnostics.test.ts tests/compute/live-selected-hour-controller.test.ts tests/compute/live-selected-hour-route-host.test.ts tests/compute/live-selected-hour-route-projection.test.ts tests/routes/main-route-overlay-gating-helper.test.ts tests/scene/utci-surface-sync.test.ts
```

Expected: all listed tests pass.

- [ ] **Step 2: Run short route probes**

Run:

```powershell
cd viewer
npx playwright test tests/e2e/main-route-manual-diagnostics.spec.ts --project=chromium --workers=1 --timeout=30000
npx playwright test tests/e2e/debug-route-baseline-diagnostics.spec.ts --project=chromium --workers=1 --timeout=30000
npx playwright test tests/e2e/debug-route-shared-host-diagnostics.spec.ts --project=chromium --workers=1 --timeout=30000
npx playwright test tests/e2e/debug-route-parity-runtime-diagnostics.spec.ts --project=chromium --workers=1 --timeout=30000
```

Expected:

- Main route remains WebGPU-first and `.bin`-free.
- Debug baseline either remains `legacy-debug` where intended or has a separate legacy/parity proof.
- Debug normal f32 shared-host probe reports `selectedHourEngine === "shared-host"` with zero legacy counters.
- Debug parity August positive and non-August negative probes pass.

- [ ] **Step 3: Run source grep boundary check**

Run:

```powershell
rg -n "\.bin|\$lib/debug|debugWebgpuUtci|loadReferenceFromFs|readbackForComparison|\bparity\b|Python|\brunAll\b" viewer/src/routes/+page.svelte viewer/src/lib/compute/liveSelectedHourRouteHost.ts viewer/src/lib/compute/liveSelectedHourRouteProjection.ts viewer/src/lib/components/scene/utciSurfaceSync.ts viewer/src/lib/diagnostics/mainRouteUtciDiagnostics.ts
```

Expected: no output.

- [ ] **Step 4: Confirm workflow constraints**

Run:

```powershell
git status --short
git worktree list
```

Expected: working tree shows only intentional changes; worktree list shows only the existing checkout. Do not commit.

- [ ] **Step 5: Stop for user review**

Report:

- What changed.
- Exact verification results.
- Whether any legacy dispatch path remains for normal f32.
- Whether parity/collect/strict exposure remain debug-only.
- Confirmation that no commits and no worktrees were created.

Do not implement additional cleanup beyond this plan without a new reviewed plan.
