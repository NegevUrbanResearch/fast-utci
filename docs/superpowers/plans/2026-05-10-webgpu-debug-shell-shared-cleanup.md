# WebGPU Main Viewer And Debug Shell Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Hard workflow constraints:** No commits. No git worktrees. Do not rewrite history. Preserve unrelated dirty files. Do not run long or hanging E2E suites. Stop and report findings if `/` regresses, if debug parity behavior becomes unclear, if tests hang, or if any task depends on `.bin` behavior in the main route.

**Goal:** Make `/` the high-quality canonical WebGPU selected-hour viewer composition, then rebuild `/debug-webgpu-utci` as the same viewer plus debug-only visual parity/proof tools.

**Architecture:** First protect and lightly thin the main route into a clean Svelte composition root: route/query decisions, live selected-hour host wiring, route-to-scene projection, diagnostics shaping, and loading overlay decisions should have explicit reusable seams. Then rebuild the debug route on that shared spine, layering Python `.bin` comparison, parity collection, timing/proof diagnostics, and strict-exposure tools on top as debug-only instruments. Do not move Threlte/WebGPU scene lifecycle orchestration out of scene components in this slice.

**Tech Stack:** SvelteKit, Svelte reactive statements/stores, Threlte/Three.js, WebGPU UTCI compute, Vitest, short targeted Playwright probes, PowerShell on Windows.

---

## Current Understanding

`/` is now the trusted rendering proof for normal selected-hour WebGPU UTCI. It is working better than `/debug-webgpu-utci` and should be treated as the canonical product viewer path.

`/debug-webgpu-utci` is still essential, but its role is different: it should be a visual proof and parity lab for showing WebGPU alignment with Python/Ladybug results and for future regression investigation. It should reuse the same selected-hour engine and viewer composition as `/`, while keeping `.bin`, Python parity, timing comparison, and strict-exposure tooling debug-only.

This plan intentionally does not “overclean” `/`. It brings the main route to a high-quality composition-root standard so the debug route has a sane pattern to reuse. The finish line for the main-route pass is “stable reusable spine,” not “perfectly tiny file.”

## Verified Starting State

Fresh verification from the prior planning pass:

- `git status --short`
  - Observed before writing this plan: clean tree except this untracked plan artifact.
- `cd viewer; npx vitest run tests/compute/live-selected-hour-controller.test.ts tests/compute/live-selected-hour-route-host.test.ts tests/compute/live-selected-hour-route-projection.test.ts tests/routes/main-route-overlay-gating-helper.test.ts tests/scene/utci-surface-sync.test.ts tests/debug/debug-webgpu-utci-diagnostics.test.ts`
  - Observed: 6 files passed, 71 tests passed.
- `cd viewer; npx playwright test tests/e2e/main-route-manual-diagnostics.spec.ts --project=chromium --workers=1`
  - Observed: 4 passed.
- `cd viewer; npx playwright test tests/e2e/debug-route-baseline-diagnostics.spec.ts --project=chromium --workers=1`
  - Observed: 1 passed.

Before implementing this plan, rerun the focused commands in Task 0 because claims from earlier sessions are hypotheses until refreshed.

Known caveat: `npm run check` currently has broader preexisting static debt. Do not claim repo-wide type cleanliness unless that command is actually fixed and rerun. If a task touches a file involved in static errors, separate new failures from inherited debt.

## Architectural Position

Main route `/`:

- Owns product viewer composition.
- Owns normal WebGPU-first selected-hour UTCI rendering.
- May expose timing and diagnostics for its own WebGPU path.
- Must not load Python `.bin` baselines.
- Must not contain Python parity or debug comparison branches.

Main route phase is done when:

- Diagnostics shaping and source/runtime proof boundaries are extracted and tested.
- `/` still proves `gpuNative`, `compute-buffer-selected-hour`, same-device rendering, selected-hour freshness, and no `.bin` requests in the targeted route probe.
- Route-to-scene projection and overlay gating remain shared, pure, and covered by focused tests.
- A Svelte architecture review finds no additional low-risk extraction needed before rebuilding debug.
- Threlte/WebGPU lifecycle orchestration stays in scene components unless later debug reuse proves a cleaner boundary.

Debug route `/debug-webgpu-utci`:

- Reuses the main viewer selected-hour engine and route-to-scene composition where possible.
- Adds visual proof/debug tools on top.
- Owns Python `.bin` comparison, parity collection, strict exposure checks, timing comparison tools, and experimental toggles.
- Should become thin enough that future selected-hour engine changes are shared instead of duplicated.

## File Structure Target

### Create

- `viewer/src/lib/diagnostics/mainRouteUtciDiagnostics.ts`
  - Pure builder for `window.__utciRenderDiagnostics__` payloads from route state.
  - No direct `window` access.
  - No `.bin`, Python, parity, or debug route references.
- `viewer/tests/diagnostics/main-route-utci-diagnostics.test.ts`
  - Unit tests for main route diagnostics shaping.
- `viewer/tests/routes/main-route-debug-boundary-source-lock.test.ts`
  - Static lock that prevents `.bin`/Python/parity/debug helpers from entering `/` and shared main-route selected-hour files.
- `viewer/src/lib/debug/debugWebgpuUtciQuery.ts`
  - Pure parser for debug route query state that exactly ports current debug semantics for `parityMode`, normal collect mode, and debug on-demand mode.
- `viewer/tests/debug/debug-webgpu-utci-query.test.ts`
  - Focused unit tests for debug query parsing.

### Modify

- `viewer/src/routes/+page.svelte`
  - Keep as canonical Svelte composition root.
  - Replace inline main-route diagnostics payload shaping with `mainRouteUtciDiagnostics.ts`.
  - Keep live host wiring visible unless a later task extracts a proven pure builder.
  - Do not add `.bin`, Python parity, or debug imports.
- `viewer/src/lib/compute/liveSelectedHourRouteProjection.ts`
  - Keep as the shared route-to-scene projection surface.
  - Add narrow fields only if needed by diagnostics or debug reuse.
- `viewer/src/routes/mainRouteOverlayGating.ts`
  - Keep as the pure loading overlay state helper.
- `viewer/src/lib/debug/debugWebgpuUtciDiagnostics.ts`
  - Expand as a pure debug diagnostics state shaper only.
  - Add honest `selectedHourEngine` fields before the debug rebuild so diagnostics can distinguish legacy debug execution from shared-host execution.
- `viewer/src/routes/debug-webgpu-utci/+page.svelte`
  - Phase 1: only behavior-preserving query/diagnostics extraction.
  - Phase 2: rebuild normal selected-hour visual path on the shared main-route spine.
  - Keep `.bin`/Python comparison explicitly debug-only.
- `viewer/tests/e2e/main-route-manual-diagnostics.spec.ts`
  - Keep short proof that `/` is WebGPU-first, `.bin`-independent, same-device, and selected-hour-fresh.
  - Any touched WebGPU Playwright spec must keep `test.afterEach(async ({ page }) => { await page.goto('about:blank').catch(() => undefined); })`, use short assertion waits, and dump the relevant window diagnostics before timeout.
- `viewer/tests/e2e/debug-route-baseline-diagnostics.spec.ts`
  - Keep short proof that debug route still publishes visual proof diagnostics.
  - Any touched WebGPU Playwright spec must keep `test.afterEach(async ({ page }) => { await page.goto('about:blank').catch(() => undefined); })`, use short assertion waits, and dump the relevant window diagnostics before timeout.

### Inspect But Do Not Extract Yet

- `viewer/src/lib/components/scene/UTCIPointCloud.svelte`
- `viewer/src/lib/components/scene/ComparisonRenderer.svelte`
- `viewer/src/lib/components/scene/utciSurfaceSync.ts`
- `viewer/src/lib/compute/liveUtciSelectedHourSession.ts`

Leave Threlte `scene.add`, storage-buffer wait, `invalidate`, GPU buffer disposal, and render lifecycle orchestration near these components unless a later, separate proof plan establishes a clean boundary.

## Review Gates

- **Gate A - Before implementation:** Have two subagents review this plan. One checks Svelte architecture and extraction seams. One red-teams behavior/proof risks.
- **Gate B - After Task 2:** Confirm `/` still passes focused main-route Vitest and Playwright probes. Do not touch debug if `/` regresses.
- **Gate C - Before debug rebuild:** Have subagents inspect whether the proposed debug rebuild will leave duplicate selected-hour state or duplicate compute dispatch.
- **Gate D - Before completion:** Rerun the focused unit and route probes listed in Final Verification. Do not claim completion from code inspection alone.

## Stop Conditions

- Stop if `/` no longer reports `utciRenderResolved: "gpuNative"`, `baseRenderTransport: "compute-buffer-selected-hour"`, `utciSurfaceSource: "compute-buffer-selected-hour"`, `dataTextureBuildCount: 0`, `baseSameDeviceForComputeAndRender: true`, `baseSelectionKey === baseSceneSelectionKey`, and `baseSelectedTimeIndex === baseRenderContextTimeIndex` in the targeted main-route probe.
- Stop if `.bin`, `loadReferenceFromFs`, `readbackForComparison`, `runAll`, Python, or parity references appear in `viewer/src/routes/+page.svelte`, `viewer/src/lib/compute/liveSelectedHourRouteHost.ts`, `viewer/src/lib/compute/liveSelectedHourRouteProjection.ts`, `viewer/src/lib/components/scene/utciSurfaceSync.ts`, or `viewer/src/lib/diagnostics/mainRouteUtciDiagnostics.ts`.
- Stop if `?parity=1`, `?collect=normal`, `.bin vs live compute`, or Python comparison behavior on `/debug-webgpu-utci` becomes ambiguous.
- Stop if a Playwright probe hangs or exceeds the short targeted timeout. Capture last diagnostics and shorten the probe; do not rerun the same hung probe unchanged.
- Stop if a task requires moving Threlte/WebGPU storage-buffer lifecycle out of `UTCIPointCloud.svelte` or `ComparisonRenderer.svelte` without a separate proof plan.

---

## Task 0: Refresh Baseline Verification

**Files:**
- Read: `viewer/src/routes/+page.svelte`
- Read: `viewer/src/routes/debug-webgpu-utci/+page.svelte`
- Read: `viewer/tests/e2e/main-route-manual-diagnostics.spec.ts`
- Read: `viewer/tests/e2e/debug-route-baseline-diagnostics.spec.ts`

- [ ] **Step 1: Confirm repo status**

Run:

```powershell
git status --short
```

Expected: only this plan file is untracked or modified. If unrelated dirty files exist, preserve them and record them before proceeding.

- [ ] **Step 2: Run focused unit baseline**

Run:

```powershell
cd viewer
npx vitest run tests/compute/live-selected-hour-controller.test.ts tests/compute/live-selected-hour-route-host.test.ts tests/compute/live-selected-hour-route-projection.test.ts tests/routes/main-route-overlay-gating-helper.test.ts tests/scene/utci-surface-sync.test.ts tests/debug/debug-webgpu-utci-diagnostics.test.ts
```

Expected: all listed tests pass.

- [ ] **Step 3: Run short main-route proof**

Run:

```powershell
cd viewer
npx playwright test tests/e2e/main-route-manual-diagnostics.spec.ts --project=chromium --workers=1 --timeout=30000
```

Expected: 4 passed. If this hangs, capture the last `window.__utciRenderDiagnostics__` output from the failing test and stop.

- [ ] **Step 4: Run short debug baseline proof**

Run:

```powershell
cd viewer
npx playwright test tests/e2e/debug-route-baseline-diagnostics.spec.ts --project=chromium --workers=1 --timeout=30000
```

Expected: 1 passed. If this hangs, capture `window.__onDemandPrototypeDiagnostics__` from the failing test and stop.

## Task 1: Lock The Main Route Debug Boundary

**Files:**
- Create: `viewer/tests/routes/main-route-debug-boundary-source-lock.test.ts`
- Inspect: `viewer/src/routes/+page.svelte`
- Inspect: `viewer/src/lib/compute/liveSelectedHourRouteHost.ts`
- Inspect: `viewer/src/lib/compute/liveSelectedHourRouteProjection.ts`
- Inspect: `viewer/src/lib/components/scene/utciSurfaceSync.ts`

- [ ] **Step 1: Write the source-lock test**

Create `viewer/tests/routes/main-route-debug-boundary-source-lock.test.ts`:

```ts
import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const viewerRoot = resolve(__dirname, '../..');

const strictProtectedFiles = ['src/routes/+page.svelte'];

const sharedProtectedFiles = [
	'src/lib/compute/liveSelectedHourRouteHost.ts',
	'src/lib/compute/liveSelectedHourRouteProjection.ts',
	'src/lib/components/scene/utciSurfaceSync.ts'
];

const strictForbiddenPatterns = [
	/\\.bin\\b/i,
	/['"]\\$lib\\/debug/,
	/debugWebgpuUtci/i,
	/loadReferenceFromFs/,
	/readbackForComparison/,
	/\\bparity\\b/i,
	/Python/i,
	/\\brunAll\\b/
];

const sharedForbiddenPatterns = [
	/\\.bin\\b/i,
	/['"]\\$lib\\/debug/,
	/debugWebgpuUtci/i,
	/loadReferenceFromFs/,
	/readbackForComparison/,
	/\\brunAll\\b/
];

describe('main route debug boundary source lock', () => {
	for (const relativePath of strictProtectedFiles) {
		it(`${relativePath} stays free of debug-only bin, parity, and Python behavior`, () => {
			const source = readFileSync(resolve(viewerRoot, relativePath), 'utf8');
			for (const pattern of strictForbiddenPatterns) {
				expect(source, `${relativePath} should not match ${pattern}`).not.toMatch(pattern);
			}
		});
	}

	for (const relativePath of sharedProtectedFiles) {
		it(`${relativePath} stays free of debug-only imports and runtime baseline hooks`, () => {
			const source = readFileSync(resolve(viewerRoot, relativePath), 'utf8');
			for (const pattern of sharedForbiddenPatterns) {
				expect(source, `${relativePath} should not match ${pattern}`).not.toMatch(pattern);
			}
		});
	}
});
```

- [ ] **Step 2: Run the source-lock test**

Run:

```powershell
cd viewer
npx vitest run tests/routes/main-route-debug-boundary-source-lock.test.ts
```

Expected: PASS.

- [ ] **Step 3: Record the post-Task-2 source-lock update**

Do not add `viewer/src/lib/diagnostics/mainRouteUtciDiagnostics.ts` to `strictProtectedFiles` yet because Task 2 has not created it. Task 2 Step 5 will add:

```ts
'src/lib/diagnostics/mainRouteUtciDiagnostics.ts'
```

Expected: the source-lock continues to pass after Task 2.

## Task 2: Extract Main Route Diagnostics Shaping

**Files:**
- Create: `viewer/src/lib/diagnostics/mainRouteUtciDiagnostics.ts`
- Create: `viewer/tests/diagnostics/main-route-utci-diagnostics.test.ts`
- Modify: `viewer/src/routes/+page.svelte`
- Modify: `viewer/tests/routes/main-route-debug-boundary-source-lock.test.ts`

- [ ] **Step 1: Write diagnostics tests**

Create `viewer/tests/diagnostics/main-route-utci-diagnostics.test.ts`:

```ts
import { describe, expect, it } from 'vitest';
import { buildMainRouteUtciDiagnostics } from '$lib/diagnostics/mainRouteUtciDiagnostics';

describe('buildMainRouteUtciDiagnostics', () => {
	it('returns undefined when diagnostics are disabled', () => {
		expect(
			buildMainRouteUtciDiagnostics({
				enabled: false,
				utciOnDemand: 'f32',
				utciRenderRequested: 'auto',
				utciRenderResolved: 'gpuNative',
				rendererBackend: 'webgpu',
				baseSurfaceDiagnostics: {},
				comparisonSurfaceDiagnostics: {},
				baseRenderTransport: 'idle',
				comparisonRenderTransport: 'idle',
				baseLiveReady: false,
				comparisonLiveReady: true,
				baseSameDeviceForComputeAndRender: null,
				comparisonSameDeviceForComputeAndRender: null,
				baseSelectedMonthIndex: 7,
				baseSelectedHourIndex: 12,
				baseSelectedTimeIndex: 180
			})
		).toBeUndefined();
	});

	it('builds a gpu-native selected-hour payload without debug parity fields', () => {
		const diagnostics = buildMainRouteUtciDiagnostics({
			enabled: true,
			utciOnDemand: 'f32',
			utciRenderRequested: 'auto',
			utciRenderResolved: 'gpuNative',
			rendererBackend: 'webgpu',
			rendererRequiredLimits: { maxStorageBufferBindingSize: 1, maxBufferSize: 1 },
			rendererDeviceLimits: { maxStorageBufferBindingSize: 1, maxBufferSize: 1 },
			baseSurfaceDiagnostics: {
				utciSurfaceSource: 'compute-buffer-selected-hour',
				selectedHourTransferCount: 0,
				dataTextureBuildCount: 0,
				gpuResidentCopyStatus: 'complete',
				gpuResidentCopyRequestId: 3
			},
			comparisonSurfaceDiagnostics: {},
			baseRenderTransport: 'compute-buffer-selected-hour',
			comparisonRenderTransport: 'idle',
			baseLiveReady: true,
			comparisonLiveReady: true,
			baseSurfaceRequestId: 3,
			baseSelectionKey: 'analysis|7|12',
			baseSceneSurfaceRequestId: 3,
			baseSceneSelectionKey: 'analysis|7|12',
			baseSameDeviceForComputeAndRender: true,
			baseSelectedMonthIndex: 7,
			baseSelectedHourIndex: 12,
			baseSelectedTimeIndex: 180,
			baseRenderContextTimeIndex: 180,
			baseAcceptedUtciRange: { min: 20, max: 41 },
			comparisonSameDeviceForComputeAndRender: null
		});

		expect(diagnostics).toMatchObject({
			utciOnDemand: 'f32',
			utciRenderRequested: 'auto',
			utciRenderResolved: 'gpuNative',
			rendererBackend: 'webgpu',
			utciSurfaceSource: 'compute-buffer-selected-hour',
			baseRenderTransport: 'compute-buffer-selected-hour',
			baseLiveReady: true,
			baseSurfaceRequestId: 3,
			baseSelectionKey: 'analysis|7|12',
			baseSelectedMonthIndex: 7,
			baseSelectedHourIndex: 12,
			baseSelectedTimeIndex: 180,
			baseAcceptedUtciRange: { min: 20, max: 41 }
		});
		expect(JSON.stringify(diagnostics)).not.toMatch(/\\.bin|parity|Python|loadReferenceFromFs/i);
	});
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
cd viewer
npx vitest run tests/diagnostics/main-route-utci-diagnostics.test.ts
```

Expected: FAIL because `mainRouteUtciDiagnostics.ts` does not exist yet.

- [ ] **Step 3: Create the pure diagnostics builder**

Create `viewer/src/lib/diagnostics/mainRouteUtciDiagnostics.ts`:

```ts
import type {
	LiveSelectedHourControllerSurfaceDiagnostics,
	LiveSelectedHourRenderTransport
} from '$lib/compute/liveSelectedHourController';
import type {
	WebgpuLargeBufferDeviceLimits,
	WebgpuLargeBufferRequiredLimits
} from '$lib/compute/webgpuDeviceLimits';
import type { UtciRendererBackend, UtciRenderMode } from '$lib/utciRenderMode';

export type MainRouteUtciDiagnosticsPayload = {
	utciOnDemand: 'f32';
	utciRenderRequested: UtciRenderMode;
	utciRenderResolved: 'dataTexture' | 'gpuNative';
	rendererBackend: UtciRendererBackend;
	rendererRequiredLimits?: WebgpuLargeBufferRequiredLimits;
	rendererDeviceLimits?: WebgpuLargeBufferDeviceLimits;
	utciSurfaceSource?: string;
	selectedHourTransferCount?: number;
	dataTextureBuildCount?: number;
	gpuResidentCopyStatus?: 'idle' | 'pending' | 'complete' | 'failed';
	gpuResidentCopyError?: string;
	gpuResidentCopyRequestId?: number;
	lastGpuResidentCopyFailureError?: string;
	lastGpuResidentCopyFailureRequestId?: number;
	baseRenderTransport: LiveSelectedHourRenderTransport;
	comparisonRenderTransport: LiveSelectedHourRenderTransport;
	baseLiveReady: boolean;
	comparisonLiveReady: boolean;
	baseSurfaceRequestId?: number;
	baseSelectionKey?: string;
	baseSceneSurfaceRequestId?: number;
	baseSceneSelectionKey?: string;
	baseSameDeviceForComputeAndRender: boolean | null;
	baseSelectedMonthIndex: number;
	baseSelectedHourIndex: number;
	baseSelectedTimeIndex: number;
	baseRenderContextTimeIndex?: number;
	baseAcceptedUtciRange?: { min: number; max: number };
	comparisonSurfaceRequestId?: number;
	comparisonSelectionKey?: string;
	comparisonSameDeviceForComputeAndRender: boolean | null;
	comparisonUtciSurfaceSource?: string;
	comparisonSelectedHourTransferCount?: number;
	comparisonDataTextureBuildCount?: number;
	comparisonGpuResidentCopyStatus?: 'idle' | 'pending' | 'complete' | 'failed';
	comparisonGpuResidentCopyError?: string;
	comparisonGpuResidentCopyRequestId?: number;
};

export type MainRouteUtciDiagnosticsInput = {
	enabled: boolean;
	utciOnDemand: 'f32';
	utciRenderRequested: UtciRenderMode;
	utciRenderResolved: 'dataTexture' | 'gpuNative';
	rendererBackend: UtciRendererBackend;
	rendererRequiredLimits?: WebgpuLargeBufferRequiredLimits;
	rendererDeviceLimits?: WebgpuLargeBufferDeviceLimits;
	baseSurfaceDiagnostics: LiveSelectedHourControllerSurfaceDiagnostics;
	comparisonSurfaceDiagnostics: LiveSelectedHourControllerSurfaceDiagnostics;
	lastGpuResidentCopyFailure?: { error?: string; requestId?: number };
	baseRenderTransport: LiveSelectedHourRenderTransport;
	comparisonRenderTransport: LiveSelectedHourRenderTransport;
	baseLiveReady: boolean;
	comparisonLiveReady: boolean;
	baseSurfaceRequestId?: number;
	baseSelectionKey?: string;
	baseSceneSurfaceRequestId?: number;
	baseSceneSelectionKey?: string;
	baseSameDeviceForComputeAndRender: boolean | null;
	baseSelectedMonthIndex: number;
	baseSelectedHourIndex: number;
	baseSelectedTimeIndex: number;
	baseRenderContextTimeIndex?: number;
	baseAcceptedUtciRange?: { min: number; max: number };
	comparisonSurfaceRequestId?: number;
	comparisonSelectionKey?: string;
	comparisonSameDeviceForComputeAndRender: boolean | null;
};

export function buildMainRouteUtciDiagnostics(
	input: MainRouteUtciDiagnosticsInput
): MainRouteUtciDiagnosticsPayload | undefined {
	if (!input.enabled) return undefined;

	return {
		utciOnDemand: input.utciOnDemand,
		utciRenderRequested: input.utciRenderRequested,
		utciRenderResolved: input.utciRenderResolved,
		rendererBackend: input.rendererBackend,
		rendererRequiredLimits: input.rendererRequiredLimits,
		rendererDeviceLimits: input.rendererDeviceLimits,
		utciSurfaceSource: input.baseSurfaceDiagnostics.utciSurfaceSource,
		selectedHourTransferCount: input.baseSurfaceDiagnostics.selectedHourTransferCount,
		dataTextureBuildCount: input.baseSurfaceDiagnostics.dataTextureBuildCount,
		gpuResidentCopyStatus: input.baseSurfaceDiagnostics.gpuResidentCopyStatus,
		gpuResidentCopyError: input.baseSurfaceDiagnostics.gpuResidentCopyError,
		gpuResidentCopyRequestId: input.baseSurfaceDiagnostics.gpuResidentCopyRequestId,
		lastGpuResidentCopyFailureError: input.lastGpuResidentCopyFailure?.error,
		lastGpuResidentCopyFailureRequestId: input.lastGpuResidentCopyFailure?.requestId,
		baseRenderTransport: input.baseRenderTransport,
		comparisonRenderTransport: input.comparisonRenderTransport,
		baseLiveReady: input.baseLiveReady,
		comparisonLiveReady: input.comparisonLiveReady,
		baseSurfaceRequestId: input.baseSurfaceRequestId,
		baseSelectionKey: input.baseSelectionKey,
		baseSceneSurfaceRequestId: input.baseSceneSurfaceRequestId,
		baseSceneSelectionKey: input.baseSceneSelectionKey,
		baseSameDeviceForComputeAndRender: input.baseSameDeviceForComputeAndRender,
		baseSelectedMonthIndex: input.baseSelectedMonthIndex,
		baseSelectedHourIndex: input.baseSelectedHourIndex,
		baseSelectedTimeIndex: input.baseSelectedTimeIndex,
		baseRenderContextTimeIndex: input.baseRenderContextTimeIndex,
		baseAcceptedUtciRange: input.baseAcceptedUtciRange,
		comparisonSurfaceRequestId: input.comparisonSurfaceRequestId,
		comparisonSelectionKey: input.comparisonSelectionKey,
		comparisonSameDeviceForComputeAndRender: input.comparisonSameDeviceForComputeAndRender,
		comparisonUtciSurfaceSource: input.comparisonSurfaceDiagnostics.utciSurfaceSource,
		comparisonSelectedHourTransferCount:
			input.comparisonSurfaceDiagnostics.selectedHourTransferCount,
		comparisonDataTextureBuildCount:
			input.comparisonSurfaceDiagnostics.dataTextureBuildCount,
		comparisonGpuResidentCopyStatus:
			input.comparisonSurfaceDiagnostics.gpuResidentCopyStatus,
		comparisonGpuResidentCopyError:
			input.comparisonSurfaceDiagnostics.gpuResidentCopyError,
		comparisonGpuResidentCopyRequestId:
			input.comparisonSurfaceDiagnostics.gpuResidentCopyRequestId
	};
}
```

- [ ] **Step 4: Wire `/` to the builder**

In `viewer/src/routes/+page.svelte`, import:

```ts
import {
	buildMainRouteUtciDiagnostics,
	type MainRouteUtciDiagnosticsPayload
} from '$lib/diagnostics/mainRouteUtciDiagnostics';
```

Replace the local `MainRouteUtciRenderDiagnostics` type with `MainRouteUtciDiagnosticsPayload`:

```ts
type MainRouteWindow = Window & {
	__utciRenderDiagnostics__?: MainRouteUtciDiagnosticsPayload;
};
```

Replace `updateUtciRenderDiagnostics(...)` internals so it calls `buildMainRouteUtciDiagnostics(...)` and assigns the result:

```ts
const payload = buildMainRouteUtciDiagnostics({
	enabled: diagnostics.utciRenderDiagnosticsEnabled,
	utciOnDemand: diagnostics.utciOnDemandMode,
	utciRenderRequested: diagnostics.utciRenderMode,
	utciRenderResolved: diagnostics.resolvedUtciSurfaceBackend,
	rendererBackend: diagnostics.rendererBackend,
	rendererRequiredLimits,
	rendererDeviceLimits,
	baseSurfaceDiagnostics: diagnostics.baseUtciSurfaceDiagnostics,
	comparisonSurfaceDiagnostics: diagnostics.comparisonUtciSurfaceDiagnostics,
	lastGpuResidentCopyFailure: lastBaseGpuResidentCopyFailure,
	baseRenderTransport: diagnostics.baseRenderTransport,
	comparisonRenderTransport: diagnostics.comparisonRenderTransport,
	baseLiveReady: diagnostics.baseLiveReady,
	comparisonLiveReady: diagnostics.comparisonLiveReady,
	baseSurfaceRequestId: diagnostics.baseSurfaceRequestId,
	baseSelectionKey: diagnostics.baseSelectionKey,
	baseSceneSurfaceRequestId: diagnostics.baseSceneSurfaceRequestId,
	baseSceneSelectionKey: diagnostics.baseSceneSelectionKey,
	baseSameDeviceForComputeAndRender: diagnostics.baseSameDeviceForComputeAndRender,
	baseSelectedMonthIndex: diagnostics.baseSelectedMonthIndex,
	baseSelectedHourIndex: diagnostics.baseSelectedHourIndex,
	baseSelectedTimeIndex: diagnostics.baseSelectedTimeIndex,
	baseRenderContextTimeIndex: diagnostics.baseRenderContextTimeIndex,
	baseAcceptedUtciRange: diagnostics.baseAcceptedUtciRange,
	comparisonSurfaceRequestId: diagnostics.comparisonSurfaceRequestId,
	comparisonSelectionKey: diagnostics.comparisonSelectionKey,
	comparisonSameDeviceForComputeAndRender:
		diagnostics.comparisonSameDeviceForComputeAndRender
});
win.__utciRenderDiagnostics__ = payload;
```

Expected behavior: when disabled, `payload` is `undefined`, preserving the previous `window.__utciRenderDiagnostics__ = undefined` behavior.

- [ ] **Step 5: Add diagnostics builder to the source lock**

Modify `viewer/tests/routes/main-route-debug-boundary-source-lock.test.ts`:

```ts
const protectedFiles = [
	'src/routes/+page.svelte',
	'src/lib/compute/liveSelectedHourRouteHost.ts',
	'src/lib/compute/liveSelectedHourRouteProjection.ts',
	'src/lib/components/scene/utciSurfaceSync.ts',
	'src/lib/diagnostics/mainRouteUtciDiagnostics.ts'
];
```

- [ ] **Step 6: Run focused tests**

Run:

```powershell
cd viewer
npx vitest run tests/diagnostics/main-route-utci-diagnostics.test.ts tests/routes/main-route-debug-boundary-source-lock.test.ts tests/compute/live-selected-hour-route-projection.test.ts tests/routes/main-route-overlay-gating-helper.test.ts
```

Expected: PASS.

- [ ] **Step 7: Run main-route proof**

Before running the proof, update or confirm `viewer/tests/e2e/main-route-manual-diagnostics.spec.ts` has:

```ts
test.afterEach(async ({ page }) => {
	await page.goto('about:blank').catch(() => undefined);
});
```

In the existing selected-hour diagnostics assertions, require the stronger proof surface:

```ts
expect(value.utciRenderResolved).toBe('gpuNative');
expect(value.baseRenderTransport).toBe('compute-buffer-selected-hour');
expect(value.utciSurfaceSource).toBe('compute-buffer-selected-hour');
expect(value.dataTextureBuildCount).toBe(0);
expect(value.baseSameDeviceForComputeAndRender).toBe(true);
expect(value.baseSelectionKey).toBe(value.baseSceneSelectionKey);
expect(value.baseSelectedTimeIndex).toBe(value.baseRenderContextTimeIndex);
expect(value.baseAcceptedUtciRange).toBeDefined();
```

Also add a request guard in the main-route probe setup:

```ts
const requestedUrls: string[] = [];
page.on('request', (request) => requestedUrls.push(request.url()));
```

After the page reaches diagnostics, assert:

```ts
expect(requestedUrls.filter((url) => /\\.bin(\\?|$)|loadReferenceFromFs|parity/i.test(url))).toEqual([]);
```

Run:

```powershell
cd viewer
npx playwright test tests/e2e/main-route-manual-diagnostics.spec.ts --project=chromium --workers=1 --timeout=30000
```

Expected: 4 passed.

- [ ] **Step 8: Add one main-route freshness probe if it is not already covered**

If `viewer/tests/e2e/main-route-manual-diagnostics.spec.ts` does not already assert this, add a narrow test that changes hour/month quickly and checks final diagnostics agree:

```ts
expect(finalValue.baseSelectedTimeIndex).toBe(finalValue.baseRenderContextTimeIndex);
expect(finalValue.baseSelectionKey).toBe(finalValue.baseSceneSelectionKey);
expect(finalValue.baseAcceptedUtciRange).toBeDefined();
expect(finalValue.utciSurfaceSource).toBe('compute-buffer-selected-hour');
```

Run the same main-route Playwright command after adding it. Expected: 4 passed if extending an existing test, or the updated expected count if adding a new test. Keep the run short and targeted.

## Task 3: Review Main Route Quality Boundary

**Files:**
- Inspect: `viewer/src/routes/+page.svelte`
- Inspect: `viewer/src/lib/diagnostics/mainRouteUtciDiagnostics.ts`
- Inspect: `viewer/src/lib/compute/liveSelectedHourRouteProjection.ts`
- Inspect: `viewer/src/routes/mainRouteOverlayGating.ts`

- [ ] **Step 1: Ask a Svelte architecture subagent to review the main route**

Prompt:

```text
Review the current main route after diagnostics extraction. Do not edit files. Is viewer/src/routes/+page.svelte now a reasonable Svelte composition root, or is there one more low-risk extraction needed before using it as the model for /debug-webgpu-utci? Focus on stable pure seams only. Do not recommend moving Threlte/WebGPU lifecycle code. Return exact file/line evidence and one recommendation: proceed to debug, or do one more main-route extraction first.
```

Expected: reviewer either says proceed, or identifies one concrete low-risk extraction.

- [ ] **Step 2: If and only if reviewer identifies one low-risk extraction, write a mini-plan before implementing it**

The mini-plan must specify:

```text
Exact files
Exact tests
Why this extraction is needed for debug reuse
Why it is not overcleaning
```

Do not do opportunistic cleanup. If the recommendation is broad or aesthetic, skip it and proceed to Task 4.

## Task 4: Extract Debug Query Semantics Without Behavior Change

**Files:**
- Create: `viewer/src/lib/debug/debugWebgpuUtciQuery.ts`
- Create: `viewer/tests/debug/debug-webgpu-utci-query.test.ts`
- Modify: `viewer/src/routes/debug-webgpu-utci/+page.svelte`

- [ ] **Step 1: Write query parser tests**

Create `viewer/tests/debug/debug-webgpu-utci-query.test.ts`:

```ts
import { describe, expect, it } from 'vitest';
import { parseDebugWebgpuUtciQuery } from '$lib/debug/debugWebgpuUtciQuery';

function params(query: string): URLSearchParams {
	return new URLSearchParams(query);
}

describe('parseDebugWebgpuUtciQuery', () => {
	it('defaults normal debug view to f32 on-demand mode', () => {
		const state = parseDebugWebgpuUtciQuery(params(''));
		expect(state.parityMode).toBe(false);
		expect(state.collectMode).toBe('off');
		expect(state.debugOnDemandMode).toBe('f32');
		expect(state.binComparisonEnabled).toBe(false);
		expect(state.binComparisonValid).toBe(false);
	});

	it('keeps parity defaulting to full-run mode unless f32 is explicit', () => {
		const state = parseDebugWebgpuUtciQuery(params('parity=1&timeIndex=23'));
		expect(state.parityMode).toBe(true);
		expect(state.collectMode).toBe('off');
		expect(state.debugOnDemandMode).toBe('off');
		expect(state.binComparisonEnabled).toBe(true);
		expect(state.binComparisonValid).toBe(false);
	});

	it('does not claim bin validity when parity f32 has no explicit August month', () => {
		const state = parseDebugWebgpuUtciQuery(params('parity=1&utciOnDemand=f32'));
		expect(state.debugOnDemandMode).toBe('f32');
		expect(state.binComparisonEnabled).toBe(true);
		expect(state.binComparisonValid).toBe(false);
	});

	it('allows explicit f32 on-demand in parity mode while keeping August validity separate', () => {
		const state = parseDebugWebgpuUtciQuery(params('parity=1&utciOnDemand=f32&monthIndex=7'));
		expect(state.debugOnDemandMode).toBe('f32');
		expect(state.binComparisonEnabled).toBe(true);
		expect(state.binComparisonValid).toBe(true);
	});

	it('does not claim bin validity for non-August parity months', () => {
		const state = parseDebugWebgpuUtciQuery(params('parity=1&utciOnDemand=f32&monthIndex=3'));
		expect(state.binComparisonEnabled).toBe(true);
		expect(state.binComparisonValid).toBe(false);
	});

	it('forces normal collect mode away from f32 on-demand by default', () => {
		const state = parseDebugWebgpuUtciQuery(params('collect=normal&monthIndex=3&hour=9'));
		expect(state.parityMode).toBe(false);
		expect(state.collectMode).toBe('normal');
		expect(state.debugOnDemandMode).toBe('off');
		expect(state.binComparisonEnabled).toBe(false);
	});

	it('preserves onDemandPrototype as an f32 opt-in', () => {
		const state = parseDebugWebgpuUtciQuery(params('parity=1&onDemandPrototype=1'));
		expect(state.debugOnDemandMode).toBe('f32');
	});
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
cd viewer
npx vitest run tests/debug/debug-webgpu-utci-query.test.ts
```

Expected: FAIL because `debugWebgpuUtciQuery.ts` does not exist yet.

- [ ] **Step 3: Implement the parser**

Create `viewer/src/lib/debug/debugWebgpuUtciQuery.ts`:

```ts
export type DebugWebgpuUtciCollectMode = 'off' | 'normal';
export type DebugWebgpuUtciOnDemandMode = 'off' | 'f32';

export type DebugWebgpuUtciQueryState = {
	parityMode: boolean;
	collectMode: DebugWebgpuUtciCollectMode;
	debugOnDemandMode: DebugWebgpuUtciOnDemandMode;
	binComparisonEnabled: boolean;
	binComparisonValid: boolean;
};

export function parseDebugWebgpuUtciQuery(
	searchParams: URLSearchParams
): DebugWebgpuUtciQueryState {
	const parityMode = searchParams.get('parity') === '1';
	const collectMode: DebugWebgpuUtciCollectMode =
		!parityMode && searchParams.get('collect') === 'normal' ? 'normal' : 'off';

	let debugOnDemandMode: DebugWebgpuUtciOnDemandMode;
	if (searchParams.has('utciOnDemand')) {
		debugOnDemandMode = searchParams.get('utciOnDemand') === 'f32' ? 'f32' : 'off';
	} else if (searchParams.get('onDemandPrototype') === '1') {
		debugOnDemandMode = 'f32';
	} else if (collectMode === 'normal') {
		debugOnDemandMode = 'off';
	} else {
		debugOnDemandMode = parityMode ? 'off' : 'f32';
	}

	const monthIndexParam = searchParams.get('monthIndex');
	const monthIndexRaw = Number(monthIndexParam ?? 'NaN');
	const monthIndex = Number.isInteger(monthIndexRaw)
		? Math.min(Math.max(monthIndexRaw, 0), 11)
		: null;

	return {
		parityMode,
		collectMode,
		debugOnDemandMode,
		binComparisonEnabled: parityMode,
		binComparisonValid: parityMode && monthIndexParam !== null && monthIndex === 7
	};
}
```

- [ ] **Step 4: Wire parser into debug route without touching selected-hour scheduling**

In `viewer/src/routes/debug-webgpu-utci/+page.svelte`, import:

```ts
import { parseDebugWebgpuUtciQuery } from '$lib/debug/debugWebgpuUtciQuery';
```

Replace only these route-local derivations:

```ts
$: debugQueryState = parseDebugWebgpuUtciQuery($page.url.searchParams);
$: parityMode = debugQueryState.parityMode;
$: normalCollectMode = debugQueryState.collectMode === 'normal';
$: debugOnDemandMode = debugQueryState.debugOnDemandMode;
```

Remove `resolveDebugOnDemandMode(...)` only after verifying no other code calls it.

Keep these route-local for now:

```text
debugOnDemandSelection
strictExposureOnlyEnabled
compareHours
compareHoursEnabled
compareMonthHoursEnabled
liveComputeModeKey
scheduleDebugOnDemandScrubRecompute
runDebugOnDemandSelectedHour
```

Reason: they depend on store/browser state, strict-exposure comparison semantics, and legacy debug scheduling. They will be addressed during the debug rebuild, not in this behavior-preserving query extraction.

- [ ] **Step 5: Run focused tests**

Run:

```powershell
cd viewer
npx vitest run tests/debug/debug-webgpu-utci-query.test.ts tests/debug/debug-webgpu-utci-diagnostics.test.ts
```

Expected: PASS.

- [ ] **Step 6: Run debug baseline proof**

Run:

```powershell
cd viewer
npx playwright test tests/e2e/debug-route-baseline-diagnostics.spec.ts --project=chromium --workers=1 --timeout=30000
```

Expected: 1 passed.

## Task 5: Add Honest Debug Engine Diagnostics

**Files:**
- Modify: `viewer/src/lib/debug/debugWebgpuUtciDiagnostics.ts`
- Modify: `viewer/tests/debug/debug-webgpu-utci-diagnostics.test.ts`
- Modify: `viewer/src/routes/debug-webgpu-utci/+page.svelte`
- Modify: `viewer/tests/e2e/debug-route-baseline-diagnostics.spec.ts`

- [ ] **Step 1: Add diagnostics tests**

In `viewer/tests/debug/debug-webgpu-utci-diagnostics.test.ts`, add:

```ts
it('reports legacy debug selected-hour execution until shared host migration is proven', () => {
	const state = deriveDebugWebgpuUtciDiagnosticsState({
		parityMode: false,
		collectMode: 'off',
		debugOnDemandMode: 'f32',
		utciRenderMode: 'auto',
		selectedMonthIndex: 7,
		selectedHourIndex: 12,
		selectedTimeIndex: 180,
		selectedHourEngine: 'legacy-debug'
	});

	expect(state.onDemandEnabled).toBe(true);
	expect(state.binComparisonEnabled).toBe(false);
	expect(state.selectedHourEngine).toBe('legacy-debug');
});

it('keeps parity comparison explicitly debug-only', () => {
	const state = deriveDebugWebgpuUtciDiagnosticsState({
		parityMode: true,
		collectMode: 'off',
		debugOnDemandMode: 'off',
		utciRenderMode: 'auto',
		selectedMonthIndex: 7,
		selectedHourIndex: 12,
		selectedTimeIndex: 180,
		selectedHourEngine: 'legacy-debug'
	});

	expect(state.onDemandEnabled).toBe(false);
	expect(state.binComparisonEnabled).toBe(true);
	expect(state.selectedHourEngine).toBe('legacy-debug');
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
cd viewer
npx vitest run tests/debug/debug-webgpu-utci-diagnostics.test.ts
```

Expected: FAIL because `selectedHourEngine` is not part of the helper yet.

- [ ] **Step 3: Add the diagnostics field**

Modify `viewer/src/lib/debug/debugWebgpuUtciDiagnostics.ts`:

```ts
export type DebugSelectedHourEngine = 'legacy-debug' | 'shared-host';
```

Add to `DebugWebgpuUtciDiagnosticsInputs`:

```ts
selectedHourEngine?: DebugSelectedHourEngine;
```

Add to `DebugWebgpuUtciDiagnosticsState`:

```ts
selectedHourEngine: DebugSelectedHourEngine;
```

Add to the debug route's `OnDemandPrototypeDiagnostics` type in `viewer/src/routes/debug-webgpu-utci/+page.svelte`:

```ts
selectedHourEngine?: DebugSelectedHourEngine;
```

Inside `deriveDebugWebgpuUtciDiagnosticsState`, set:

```ts
selectedHourEngine: inputs.selectedHourEngine ?? 'legacy-debug',
```

- [ ] **Step 4: Thread honest legacy identity in debug route**

In `viewer/src/routes/debug-webgpu-utci/+page.svelte`, add to the `deriveDebugWebgpuUtciDiagnosticsState(...)` call:

```ts
selectedHourEngine: 'legacy-debug',
```

Do not set `"shared-host"` anywhere in this task.

In `updateOnDemandPrototypeDiagnostics(...)`, merge the field into `window.__onDemandPrototypeDiagnostics__`:

```ts
selectedHourEngine:
	diagnostics.selectedHourEngine ??
	existing?.selectedHourEngine ??
	debugDiagnosticsState.selectedHourEngine,
```

- [ ] **Step 5: Add a cheap debug E2E invariant**

In `viewer/tests/e2e/debug-route-baseline-diagnostics.spec.ts`, after the diagnostics object is available, assert:

```ts
expect(value.selectedHourEngine).toBe('legacy-debug');
expect(value.binComparisonEnabled ?? false).toBe(false);
```

Use the actual variable name in the existing test. Do not add new long waits.

- [ ] **Step 6: Run focused tests and debug proof**

Run:

```powershell
cd viewer
npx vitest run tests/debug/debug-webgpu-utci-diagnostics.test.ts tests/debug/debug-webgpu-utci-query.test.ts
npx playwright test tests/e2e/debug-route-baseline-diagnostics.spec.ts --project=chromium --workers=1 --timeout=30000
```

Expected:

- Vitest files pass.
- Debug Playwright: 1 passed.

## Task 6: Design The Debug Rebuild Over The Main Spine (Follow-Up Plan Only)

**Files:**
- Create or modify: `docs/superpowers/plans/2026-05-10-webgpu-debug-route-shared-viewer-rebuild.md`
- Inspect: `viewer/src/routes/+page.svelte`
- Inspect: `viewer/src/routes/debug-webgpu-utci/+page.svelte`
- Inspect: `viewer/src/lib/compute/liveSelectedHourRouteHost.ts`
- Inspect: `viewer/src/lib/compute/liveSelectedHourRouteProjection.ts`
- Inspect: `viewer/src/lib/diagnostics/mainRouteUtciDiagnostics.ts`
- Inspect: `viewer/src/lib/debug/debugWebgpuUtciDiagnostics.ts`

- [ ] **Step 1: Inventory legacy debug selected-hour owners**

Record exact line references for:

```text
debugOnDemandSelection
debugOnDemandSelectionKey
lastDebugOnDemandScrubTriggerKey
scheduleDebugOnDemandScrubRecompute
runDebugOnDemandSelectedHour
acceptedGpuResidentUtciOutput
deferredCpuFallbackSelectedHour
liveUtciSurfaceDiagnostics
updateOnDemandPrototypeDiagnostics
shouldReadbackForComparison
pythonBinComparisonActive
```

Expected: a short table in the follow-up plan with columns:

```text
Owner
Current responsibility
Future responsibility
Keep / move / delete
Verification needed
```

- [ ] **Step 2: Define the target debug architecture**

Write this invariant into the follow-up plan:

```text
For normal non-parity f32 debug rendering, /debug-webgpu-utci must use the same selected-hour engine and route-to-scene projection family as /. The debug page may add visual parity, timing, and proof overlays, but selected-hour compute ownership must not be duplicated.

For ?parity=1 and ?collect=normal, Python/.bin and collection behavior remains debug-only and may use separate debug instrumentation until explicitly migrated.
```

- [ ] **Step 3: Define required probes before deleting legacy debug machinery**

The follow-up plan must include these probes:

```text
1. Main route remains WebGPU-first and .bin-free.
2. Debug normal f32 route reports selectedHourEngine === "shared-host" only after the shared host actually drives compute.
3. Debug parity route still exposes Python/.bin comparison status only on /debug-webgpu-utci.
4. Rapid month/hour changes on debug converge to the final selectionKey/timeIndex/range.
5. Debug visual proof can show WebGPU-vs-Python alignment for the August Python baseline without implying non-August Python baseline validity.
6. Normal debug f32 exposes a single dispatch owner and one accepted request id, with no calls through `runDebugOnDemandSelectedHour` or `scheduleDebugOnDemandScrubRecompute` after `selectedHourEngine === "shared-host"`.
```

- [ ] **Step 4: Use subagents before finalizing the follow-up plan**

Ask two subagents:

```text
Reviewer A: Inspect the proposed debug shared-viewer rebuild. Will it leave duplicate selected-hour state or duplicate compute dispatch? Return exact variables/schedulers that must be removed or retained.

Reviewer B: Inspect the proposed tests. Would they catch main-route regressions, .bin leakage into /, stale selected-hour surfaces, parity-mode regressions, and false non-August Python baseline claims?
```

Expected: both reviewers return actionable findings, and the follow-up plan incorporates them before implementation.

- [ ] **Step 5: Stop for user review**

Do not implement the debug rebuild until the user approves the follow-up plan.

## Final Verification Commands

Run after Tasks 0-6 only. This plan ends with a follow-up rebuild plan and user review; it does not implement the debug rebuild.

```powershell
cd viewer
npx vitest run tests/diagnostics/main-route-utci-diagnostics.test.ts tests/routes/main-route-debug-boundary-source-lock.test.ts tests/debug/debug-webgpu-utci-query.test.ts tests/debug/debug-webgpu-utci-diagnostics.test.ts tests/compute/live-selected-hour-controller.test.ts tests/compute/live-selected-hour-route-host.test.ts tests/compute/live-selected-hour-route-projection.test.ts tests/routes/main-route-overlay-gating-helper.test.ts tests/scene/utci-surface-sync.test.ts
npx playwright test tests/e2e/main-route-manual-diagnostics.spec.ts --project=chromium --workers=1 --timeout=30000
npx playwright test tests/e2e/debug-route-baseline-diagnostics.spec.ts --project=chromium --workers=1 --timeout=30000
```

Expected:

- Vitest: all listed tests pass.
- Main route Playwright: 4 passed.
- Debug route Playwright: 1 passed.
- No `.bin` / Python parity source-lock failure in protected main/shared files.
- Debug diagnostics honestly report `selectedHourEngine: "legacy-debug"` until the follow-up rebuild proves shared-host ownership. The shared-host assertion belongs to the approved follow-up rebuild plan, not this plan.
- No commits created.
- No git worktrees created.

## Non-Goals

- Do not fix repo-wide `npm run check` debt in this slice.
- Do not remove `.bin`, parity, collect, timing comparison, or visual proof tooling from `/debug-webgpu-utci`.
- Do not add Python baseline behavior to `/`.
- Do not move WebGPU/Threlte scene add/remove, storage-buffer wait, invalidate, or dispose orchestration into a generic helper.
- Do not perform broad visual redesign or UI copy cleanup.
- Do not make `/` tiny for its own sake. The goal is a high-quality composition root with stable seams for debug reuse.
