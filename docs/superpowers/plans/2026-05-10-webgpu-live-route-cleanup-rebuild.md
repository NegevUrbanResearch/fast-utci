# WebGPU Live Route Cleanup Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Workflow overrides for this plan:** Do not create git worktrees. Do not create commits. Preserve unrelated dirty files. Keep the debug route behavior frozen until this plan explicitly reaches the debug rebuild task. Report verification evidence before advancing between tasks.

**Goal:** Make `/` use the selected-hour WebGPU path as its normal UTCI renderer with clean shared architecture, while keeping `/debug-webgpu-utci` as the frozen baseline until the main route is visually correct and maintainable.

**Architecture:** Stabilize the main-route render handshake first, then extract only stable shared selected-hour responsibilities: host orchestration, route-to-scene projection, GPU-resident UTCI surface synchronization, and diagnostics shaping. Rebuild the main route around those shared units before rebuilding the debug route last as a debug/parity composition layer over the same shared selected-hour engine.

**Tech Stack:** SvelteKit, Svelte stores/reactivity, Threlte/Three.js, WebGPU UTCI compute pipeline, Vitest, Playwright used only for short targeted probes, PowerShell on Windows.

---

## Why This Plan Replaces The Previous Attempt

The previous plan had the right target but the wrong execution pressure. It moved quickly from controller/host work into route rewrites and even touched the debug route before the main route had a proven visible UTCI layer. That created an interim state where focused unit tests can pass while the app-visible canvas still fails.

This plan changes the order:

1. Freeze `/debug-webgpu-utci` as the behavior baseline.
2. Prove and repair the main-route selected-hour scene publication handshake.
3. Capture a debug-route baseline snapshot before any shared scene component edits.
4. Extract duplicated rendering/sync behavior only after the handshake is understood and the debug baseline is captured.
5. Rebuild `/` as a thin product route.
6. Rebuild `/debug-webgpu-utci` last, preserving `.bin` comparison and debug diagnostics as debug-only concerns.

Do not continue to the previous Task 5. Treat this file as the current execution plan.

## Current State Snapshot

The current dirty tree already contains useful pieces:

- `viewer/src/lib/compute/liveSelectedHourController.ts`
- `viewer/src/lib/compute/liveSelectedHourSurfaceIdentity.ts`
- `viewer/src/lib/compute/liveSelectedHourRouteHost.ts`
- `viewer/src/lib/compute/liveSelectedHourRouteProjection.ts`
- `viewer/src/lib/compute/liveSelectedHourRenderContext.ts`
- `viewer/src/lib/compute/liveUtciSelectedHour.ts`
- `viewer/src/lib/compute/liveUtciSelectedHourSession.ts`
- `viewer/src/lib/compute/projectWeather.ts`
- `viewer/src/lib/components/viewer/ViewerShell.svelte`
- `viewer/src/routes/mainRouteOverlayGating.ts`

The current dirty tree also has risks:

- `viewer/src/routes/+page.svelte` is still large and is not visually proven.
- `viewer/src/routes/debug-webgpu-utci/+page.svelte` is still the working baseline but was already partially touched.
- `viewer/src/lib/components/scene/UTCIPointCloud.svelte` and `viewer/src/lib/components/scene/ComparisonRenderer.svelte` duplicate GPU-resident surface sync logic.
- Broad Playwright runs can hang and leave stale dev-server/browser processes. Do not use them as the primary investigation loop.

## Implementation Notes

### Task 1 Baseline Freeze

```text
Frozen baseline route: viewer/src/routes/debug-webgpu-utci/+page.svelte
Allowed before Task 6: read-only inspection only
Reason: debug route is the selected-hour WebGPU behavior baseline for rebuilding /
Baseline diff artifact during execution: tmp-debug-route-freeze.diff (local temp artifact, not committed)
Task 1 focused Vitest: 4 files passed, 53 tests passed
Generated temp artifacts removed: tmp-main-route-probe.*, tmp-main-route*.png, tmp-viewer-dev.*, viewer/tmp-main-route-probe.png, viewer/tmp-viewer-dev.*
Generated Playwright residue preserved as prior failure evidence, not source baseline: viewer/test-results/.last-run.json and viewer/test-results/webgpu-on-demand-prototype-*/
```

## File Structure Target

### Shared Compute / Host Layer

- `viewer/src/lib/compute/liveSelectedHourController.ts`
  - Owns request lifecycle, session calls, accepted output, same-device proof, request-scoped render readiness.
- `viewer/src/lib/compute/liveSelectedHourRouteHost.ts`
  - Owns base/comparison selected-hour route orchestration.
  - Does not know about `.bin`, parity exports, debug-only globals, or UI overlays.
- `viewer/src/lib/compute/liveSelectedHourRouteProjection.ts`
  - Pure function that turns host state plus route state into scene props.
  - Must handle bootstrap publication states where the scene needs render props before `baseHasVisibleLiveSurface` is true.
- `viewer/src/lib/compute/liveSelectedHourSurfaceIdentity.ts`
  - Shared request/selection/surface identity contract.
- `viewer/src/lib/compute/liveSelectedHourRenderContext.ts`
  - Shared render-state resolver for selected-hour UTCI surfaces.

### Shared Scene Rendering Layer

- Create `viewer/src/lib/components/scene/utciSurfaceSync.ts`
  - Shared pure helpers for GPU-resident UTCI surface keys, authoritative compute-buffer detection, and diagnostics shaping.
  - No Svelte store subscriptions.
  - No route awareness.
  - Does not own Threlte lifecycle, scene add/remove, or GPU queue-copy orchestration in the first pass.
  - Used by both `UTCIPointCloud.svelte` and `ComparisonRenderer.svelte`.
- Modify `viewer/src/lib/components/scene/UTCIPointCloud.svelte`
  - Base UTCI surface component.
  - Delegates GPU copy/sync and diagnostics shaping to `utciSurfaceSync.ts`.
- Modify `viewer/src/lib/components/scene/ComparisonRenderer.svelte`
  - Comparison scene component.
  - Reuses `utciSurfaceSync.ts` instead of carrying a separate copy of the GPU-resident path.

### Main Route Layer

- `viewer/src/routes/+page.svelte`
  - Product route coordinator only.
  - Owns project/query/store wiring, model loading, UI shell, tooltip/camera interactions, and scene composition.
  - Does not own selected-hour trigger keys, publication readiness, comparison source ownership, or GPU surface sync internals.
- `viewer/src/routes/mainRouteOverlayGating.ts`
  - Pure overlay state helper.
- Create or keep `viewer/src/lib/components/viewer/ViewerShell.svelte`
  - Dumb layout shell with slots only.

### Debug Route Layer

- Create `viewer/src/lib/debug/debugWebgpuUtciDiagnostics.ts`
  - Debug-only pure helpers used only after the main route is correct.
  - Owns debug/prototype diagnostics shaping and debug window-export object construction.
  - Does not become a stateful layer until repeated debug-only decisions emerge.
  - Does not reimplement selected-hour host orchestration.
- Modify `viewer/src/routes/debug-webgpu-utci/+page.svelte`
  - Rebuilt last.
  - Remains the debug/proof route and preserves `.bin` vs live behavior.

### Tests

- `viewer/tests/compute/live-selected-hour-route-projection.test.ts`
  - Main proof point for route-to-scene bootstrap props.
- `viewer/tests/routes/main-route-overlay-gating-helper.test.ts`
  - Overlay pure-function coverage.
- `viewer/tests/compute/live-selected-hour-route-host.test.ts`
  - Host orchestration coverage.
- Create `viewer/tests/scene/utci-surface-sync.test.ts`
  - Shared GPU surface sync helper coverage with fakes where possible.
- `viewer/tests/e2e/webgpu-on-demand-prototype.spec.ts`
  - Keep only short, focused probes for manual/diagnostic confirmation.

---

## Task 1: Freeze Baseline And Clean The Investigation Surface

**Purpose:** Stop the current churn from moving the debug baseline or hiding the main-route failure behind stale artifacts.

**Files:**
- Read: `viewer/src/routes/debug-webgpu-utci/+page.svelte`
- Read: `viewer/src/routes/+page.svelte`
- Read: `viewer/test-results/.last-run.json`
- Remove only if confirmed generated and unrelated to source review: repo-local `tmp-*.log`, `tmp-*.png`, `viewer/tmp-*.log`, `viewer/tmp-*.png`
- Do not edit production code in this task.

- [x] **Step 1: Record the dirty source surface**

Run:

```powershell
git status --short
git diff --stat
```

Expected:
- The plan file is untracked or modified.
- Existing dirty production/test files are listed.
- No source files are reverted.

- [x] **Step 2: Identify stale Playwright/dev-server processes without killing user Chrome or Cursor**

Run:

```powershell
Get-CimInstance Win32_Process |
  Where-Object {
    ($_.Name -eq 'node.exe' -and
      $_.CommandLine -like '*D:\Projects\Nur\Shade\fast-utci*' -and (
      $_.CommandLine -like '*playwright test*' -or
      $_.CommandLine -like '*vite.js* dev*' -or
      $_.CommandLine -like '*npm-cli.js run dev*'
    )) -or
    ($_.Name -eq 'chrome.exe' -and $_.CommandLine -like '*ms-playwright*')
  } |
  Select-Object ProcessId,Name,CommandLine |
  Format-List
```

Expected:
- Only repo dev-server, Playwright node, or Playwright Chromium processes appear.
- Regular Chrome and Cursor helper processes are not targeted.

- [x] **Step 3: Stop only stale repo/Playwright processes**

Run only for processes found in Step 2:

```powershell
$targets = Get-CimInstance Win32_Process |
  Where-Object {
    ($_.Name -eq 'node.exe' -and
      $_.CommandLine -like '*D:\Projects\Nur\Shade\fast-utci*' -and (
      $_.CommandLine -like '*playwright test*' -or
      $_.CommandLine -like '*vite.js* dev*' -or
      $_.CommandLine -like '*npm-cli.js run dev*'
    )) -or
    ($_.Name -eq 'chrome.exe' -and $_.CommandLine -like '*ms-playwright*')
  }
$targets | ForEach-Object {
  Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
}
```

Expected:
- Stale Playwright/dev-server processes are gone.
- User Chrome/Cursor remains running.

- [x] **Step 4: Mark debug route as frozen for this plan and capture the dirty baseline**

Do not edit the file yet. Capture the current dirty debug-route snapshot so later agents can distinguish inherited work from new changes:

```powershell
git diff -- viewer/src/routes/debug-webgpu-utci/+page.svelte > tmp-debug-route-freeze.diff
git diff --stat -- viewer/src/routes/debug-webgpu-utci/+page.svelte
```

In the implementation notes for this task, record:

```text
Frozen baseline route: viewer/src/routes/debug-webgpu-utci/+page.svelte
Allowed before Task 6: read-only inspection only
Reason: debug route is the selected-hour WebGPU behavior baseline for rebuilding /
Baseline diff artifact during execution: tmp-debug-route-freeze.diff (local temp artifact, not committed)
```

Expected:
- All later workers understand that debug-route production edits before Task 6 are plan violations.
- If a later task changes shared scene components used by the debug route, it must run the debug baseline probe before and after the shared edit.

- [x] **Step 4.5: Add the pre-implementation review gate**

Before any production code changes, dispatch two review agents against this plan and the Task 1 evidence:

1. Spec compliance review:
   - Are debug-freeze boundaries enforceable?
   - Are visual proof gates strong enough?
   - Are no-commit/no-worktree constraints preserved?
2. Code-fit review:
   - Are planned extractions narrow enough?
   - Are Svelte/Three/WebGPU lifecycle boundaries respected?
   - Are snippets consistent with current test helpers and types?

Expected:
- Findings are fixed in the plan before implementation continues.
- If unresolved findings remain, stop and ask the user.

- [x] **Step 5: Run focused non-browser verification only**

Run:

```powershell
cd viewer
npx vitest run tests/compute/live-selected-hour-controller.test.ts tests/compute/live-selected-hour-route-host.test.ts tests/compute/live-selected-hour-route-projection.test.ts tests/routes/main-route-overlay-gating-helper.test.ts
```

Expected:
- Capture pass/fail counts.
- If failures occur, report findings first. Do not fix in this task unless the failure is caused only by missing tests introduced later in this plan.

---

## Task 2: Repair The Main-Route Render Handshake With Pure Projection Tests

**Purpose:** Fix the likely bootstrap gap where the host needs scene diagnostics to publish a visible surface, but the route projection withholds the render props needed for the scene to produce those diagnostics.

**Files:**
- Modify: `viewer/tests/compute/live-selected-hour-route-projection.test.ts`
- Modify: `viewer/src/lib/compute/liveSelectedHourRouteProjection.ts`
- Read: `viewer/src/lib/compute/liveSelectedHourRouteHost.ts`
- Read: `viewer/src/routes/+page.svelte`

- [x] **Step 1: Replace the old bootstrap-withheld projection expectation**

`viewer/tests/compute/live-selected-hour-route-projection.test.ts` currently has a test named:

```ts
it('projects pending gpu output without exposing bootstrap render props', () => {
```

That expectation is now the suspected bug. Replace it with a test that proves the scene receives bootstrap props when the controller has an accepted GPU output but the host has not yet marked the surface visible.

Use the existing local helpers already in the file: `createFullDayAnalysis(...)` and `createState()`.

```ts
it('passes bootstrap GPU selected-hour props to the base scene before visible publication', () => {
	const baseAnalysis = createFullDayAnalysis('base');
	const bootstrapAnalysis = createFullDayAnalysis('bootstrap-selected');
	const acceptedGpuResidentOutput = {
		requestId: 7,
		monthIndex: 7,
		hourIndex: 12,
		timeIndex: 180,
		utciRange: { min: 10, max: 40 },
		output: {
			format: 'f32-utci',
			numPoints: 2,
			timeIndex: 180,
			gpuBuffer: { label: 'gpu-buffer' } as unknown as GPUBuffer
		}
	};
	const liveRouteState = createState();
	liveRouteState.baseDisplayAnalysis = bootstrapAnalysis;
	liveRouteState.baseHasVisibleLiveSurface = false;
	liveRouteState.baseSurfaceIdentity = null;
	liveRouteState.baseSceneSurfaceIdentity = {
		requestId: 7,
		monthIndex: 7,
		hourIndex: 12,
		timeIndex: 180,
		selectionKey: 'Ben-Gurion/20250815_grid_2m_fullday|7|12',
		pendingRenderUpdateStartedAt: 100,
		acceptedGpuResidentOutput
	};
	liveRouteState.baseRenderContext = {
		analysis: bootstrapAnalysis,
		monthIndex: 7,
		hourIndex: 12,
		timeIndex: 180,
		selectionKey: 'Ben-Gurion/20250815_grid_2m_fullday|7|12',
		colorMode: 'discrete',
		metricType: 'utci',
		rangeOverride: null
	};

	const projected = projectMainRouteLiveSceneState({
		useLiveUtciOnMainRoute: true,
		isComparing: false,
		baseAnalysis,
		comparisonAnalysis: null,
		liveRouteState
	});

	expect(projected.baseSceneAnalysis).toBe(bootstrapAnalysis);
	expect(projected.basePendingGpuResidentOutput).toBe(acceptedGpuResidentOutput);
	expect(projected.baseSceneSurfaceIdentity?.requestId).toBe(7);
	expect(projected.baseSceneRenderContext?.timeIndex).toBe(180);
});
```

Expected current failure:
- `baseSceneRenderContext` or `baseSceneSurfaceIdentity` is `null` because projection currently gates them on `baseHasVisibleLiveSurface`.

- [x] **Step 2: Add the comparison equivalent**

Add:

```ts
it('passes bootstrap GPU selected-hour props to the comparison scene before visible publication', () => {
	const baseAnalysis = createFullDayAnalysis('base');
	const comparisonAnalysis = createFullDayAnalysis('winter');
	const acceptedGpuResidentOutput = {
		requestId: 11,
		monthIndex: 1,
		hourIndex: 9,
		timeIndex: 42,
		utciRange: { min: 8, max: 35 },
		output: {
			format: 'f32-utci',
			numPoints: 2,
			timeIndex: 42,
			gpuBuffer: { label: 'comparison-gpu-buffer' } as unknown as GPUBuffer
		}
	};
	const liveRouteState = createState();
	liveRouteState.comparisonDisplayAnalysis = comparisonAnalysis;
	liveRouteState.comparisonHasVisibleLiveSurface = false;
	liveRouteState.comparisonSurfaceIdentity = null;
	liveRouteState.comparisonSceneSurfaceIdentity = {
		requestId: 11,
		monthIndex: 1,
		hourIndex: 9,
		timeIndex: 42,
		selectionKey: 'winter|1|9',
		pendingRenderUpdateStartedAt: 120,
		acceptedGpuResidentOutput
	};
	liveRouteState.comparisonRenderContext = {
		analysis: comparisonAnalysis,
		monthIndex: 1,
		hourIndex: 9,
		timeIndex: 42,
		selectionKey: 'winter|1|9',
		colorMode: 'discrete',
		metricType: 'utci',
		rangeOverride: null
	};

	const projected = projectMainRouteLiveSceneState({
		useLiveUtciOnMainRoute: true,
		isComparing: true,
		baseAnalysis,
		comparisonAnalysis,
		liveRouteState
	});

	expect(projected.comparisonSceneAnalysis).toBe(comparisonAnalysis);
	expect(projected.comparisonPendingGpuResidentOutput).toBe(acceptedGpuResidentOutput);
	expect(projected.comparisonSceneSurfaceIdentity?.requestId).toBe(11);
	expect(projected.comparisonSceneRenderContext?.timeIndex).toBe(42);
});
```

Expected current failure:
- Comparison projection has the same visible-surface gate.

- [x] **Step 3: Run projection tests and confirm the failure**

Run:

```powershell
cd viewer
npx vitest run tests/compute/live-selected-hour-route-projection.test.ts
```

Expected:
- New bootstrap tests fail for the gating reason.
- Existing projection tests reveal any compatibility expectations that must be preserved.

- [x] **Step 4: Update projection to pass bootstrap scene props**

Modify `viewer/src/lib/compute/liveSelectedHourRouteProjection.ts`.

The base rules should become:

```ts
const baseBootstrapSurfaceIdentity = useLiveUtciOnMainRoute
	? liveRouteState.baseSceneSurfaceIdentity
	: null;
const basePublishedSurfaceIdentity = useLiveUtciOnMainRoute
	? liveRouteState.baseSurfaceIdentity
	: null;
const baseSceneSurfaceIdentity =
	basePublishedSurfaceIdentity ?? baseBootstrapSurfaceIdentity;
const basePendingGpuResidentOutput =
	baseSceneSurfaceIdentity?.acceptedGpuResidentOutput ?? null;
const baseSceneRenderContext =
	useLiveUtciOnMainRoute && (baseHasVisibleLiveSurface || basePendingGpuResidentOutput != null)
		? liveRouteState.baseRenderContext
		: null;
const baseBootstrapAnalysis = baseSceneRenderContext?.analysis ?? baseDisplayedAnalysis;
const baseSceneAnalysis = !useLiveUtciOnMainRoute
	? baseAnalysis
	: baseHasVisibleLiveSurface
		? baseDisplayedAnalysis
		: basePendingGpuResidentOutput != null
			? baseBootstrapAnalysis
			: null;
```

The comparison rules should mirror the same logic while preserving `undefined` when comparison is inactive:

```ts
const comparisonBootstrapSurfaceIdentity =
	isComparing && useLiveUtciOnMainRoute
		? liveRouteState.comparisonSceneSurfaceIdentity
		: undefined;
const comparisonPublishedSurfaceIdentity =
	isComparing && useLiveUtciOnMainRoute
		? liveRouteState.comparisonSurfaceIdentity
		: undefined;
const comparisonSceneSurfaceIdentity = !isComparing
	? undefined
	: comparisonPublishedSurfaceIdentity ?? comparisonBootstrapSurfaceIdentity ?? null;
const comparisonPendingGpuResidentOutput =
	comparisonSceneSurfaceIdentity?.acceptedGpuResidentOutput ?? null;
const comparisonSceneRenderContext = !isComparing
	? undefined
	: useLiveUtciOnMainRoute &&
		  (comparisonHasVisibleLiveSurface || comparisonPendingGpuResidentOutput != null)
		? liveRouteState.comparisonRenderContext
		: null;
const comparisonBootstrapAnalysis =
	comparisonSceneRenderContext?.analysis ?? comparisonRendererDisplayAnalysis;
const comparisonSceneAnalysis = !isComparing
	? undefined
	: !useLiveUtciOnMainRoute
		? undefined
		: comparisonHasVisibleLiveSurface
			? comparisonRendererDisplayAnalysis
			: comparisonPendingGpuResidentOutput != null
				? comparisonBootstrapAnalysis
				: null;
```

Expected:
- The scene gets enough props to render/copy/publish a GPU-selected surface during bootstrap.
- Scene `analysis`, render context `analysis`, tooltip source analysis, and diagnostics all refer to the same bootstrap/published analysis when a live surface exists.
- `baseHasVisibleLiveSurface` still means “published current visible surface,” not “scene may start rendering.”

- [x] **Step 5: Run focused non-browser tests**

Run:

```powershell
cd viewer
npx vitest run tests/compute/live-selected-hour-route-projection.test.ts tests/routes/main-route-overlay-gating-helper.test.ts
```

Expected:
- Projection tests pass.
- Overlay gating tests remain unchanged.

---

## Task 3: Add A Short Manual Diagnostics Probe For The Main Route

**Purpose:** Confirm the main route is visually and diagnostically on the selected-hour path without re-entering broad hanging E2E loops.

**Files:**
- Create: `viewer/tests/e2e/main-route-manual-diagnostics.spec.ts`
- Read: `viewer/src/routes/+page.svelte`
- Read: `viewer/src/lib/compute/liveSelectedHourRouteProjection.ts`

- [x] **Step 1: Create a single short Playwright diagnostic spec**

Create `viewer/tests/e2e/main-route-manual-diagnostics.spec.ts`:

```ts
import { expect, test } from '@playwright/test';

test.describe('main route manual diagnostics probe', () => {
	test('publishes selected-hour diagnostics without waiting for the full e2e suite', async ({ page }) => {
		test.setTimeout(30_000);
		await page.goto('/?analysis=Ben-Gurion%2F20250815_grid_2m_fullday&utciRender=auto&utciRenderDiagnostics=1');

		await page.waitForFunction(() => {
			const diagnostics = (window as any).__utciRenderDiagnostics__;
			return diagnostics?.rendererBackend === 'webgpu';
		}, undefined, { timeout: 10_000 });

		const diagnostics = await page.waitForFunction(() => {
			const value = (window as any).__utciRenderDiagnostics__;
			if (!value) return null;
			if (
				value.baseLiveReady === true &&
				value.utciSurfaceSource === 'compute-buffer-selected-hour' &&
				value.baseRenderTransport === 'compute-buffer-selected-hour' &&
				value.baseSameDeviceForComputeAndRender === true
			) {
				return value;
			}
			return null;
		}, undefined, { timeout: 15_000 });

		const value = await diagnostics.jsonValue() as any;
		expect(value.utciRenderResolved).toBe('gpuNative');
		expect(value.baseRenderTransport).toBe('compute-buffer-selected-hour');
		expect(value.utciSurfaceSource).toBe('compute-buffer-selected-hour');
		expect(value.baseLiveReady).toBe(true);
		expect(value.baseSameDeviceForComputeAndRender).toBe(true);
	});
});
```

Expected:
- This is a probe, not a broad route suite.
- It has a 30-second test timeout and short internal waits.

- [x] **Step 2: Run only the short probe**

Run:

```powershell
cd viewer
npx playwright test tests/e2e/main-route-manual-diagnostics.spec.ts --project=chromium --workers=1
```

Expected:
- If it passes, record diagnostics values.
- If it fails, capture the last `window.__utciRenderDiagnostics__` value manually if the browser can reach the route.
- If it hangs, kill only Playwright/dev-server processes using the Task 1 commands and report this as harness/runtime debt, not route proof.
- A pass means selected-hour publication is ready, not merely bootstrap state.

Task 3 automated result:

```text
Focused unit slice: 5 files passed, 68 tests passed.
Strict probe: tests/e2e/main-route-manual-diagnostics.spec.ts passed, 1 test passed.
Verified selected-hour source: compute-buffer-selected-hour.
Verified transport: compute-buffer-selected-hour.
Verified same-device: true.
Main-route publication bug found and fixed before this pass: pending compute-buffer meshes were not render-traversable, so Three never initialized renderer-owned storage buffers.
```

- [x] **Step 3: Manual browser check**

Open:

```text
http://localhost:5173/?analysis=Ben-Gurion%2F20250815_grid_2m_fullday&utciRender=auto&utciRenderDiagnostics=1
```

Inspect:

```js
window.__utciRenderDiagnostics__
```

Expected visual result:
- Buildings/model are visible.
- UTCI layer is visibly colored according to the UTCI legend, not only the green shading/exposure layer.

Expected diagnostic result:

```ts
{
	utciRenderResolved: 'gpuNative',
	rendererBackend: 'webgpu',
	baseRenderTransport: 'compute-buffer-selected-hour',
	utciSurfaceSource: 'compute-buffer-selected-hour',
	baseSameDeviceForComputeAndRender: true
}
```

If the probe passes but the layer is not visually visible, stop and inspect scene material/layer visibility before continuing to Task 4. Do not let diagnostics-only proof substitute for visual proof.

Task 3 manual result:

```text
Manual check confirmed the main route displays the UTCI texture. A follow-up bug was found: hour/month changes updated tooltip values but did not update the visible WebGPU UTCI surface. That follow-up is tracked and handled in Task 3.1 before Task 3.5.
```

---

## Task 3.1: Prove And Fix Hour/Month Updates On The Main WebGPU Surface

**Purpose:** The main route now publishes an initial visible WebGPU UTCI surface, but manual testing showed the surface colors do not respond to hour/month changes even though tooltip values do. Do not proceed to Task 3.5 until the selected-hour WebGPU surface is proven reactive to selection changes.

**Files:**
- Read/edit: `viewer/tests/e2e/main-route-manual-diagnostics.spec.ts`
- Read/edit as needed: `viewer/src/routes/+page.svelte`
- Read/edit as needed: `viewer/src/lib/compute/liveSelectedHourRouteHost.ts`
- Read/edit as needed: `viewer/src/lib/compute/liveSelectedHourRouteProjection.ts`
- Read/edit as needed: `viewer/src/lib/components/scene/UTCIPointCloud.svelte`
- Avoid debug route production edits.

- [x] **Step 1: Add a short interaction proof**

Extend the main-route diagnostic spec or add a second test that:

```text
1. Opens /?analysis=Ben-Gurion%2F20250815_grid_2m_fullday&utciRender=auto&utciRenderDiagnostics=1.
2. Waits for baseRenderTransport=compute-buffer-selected-hour and utciSurfaceSource=compute-buffer-selected-hour.
3. Changes selected hour and, if practical, selected month through the real UI or route-visible stores.
4. Waits for a new selected-hour request/publication to complete.
5. Asserts the visible WebGPU surface publishes the new selection, not only tooltip data.
```

Expected:
- The first run may fail and should capture last diagnostics.
- Failure should identify whether the host request, projection props, accepted-output key, or scene copy/update is stale.

- [x] **Step 2: Fix the stale selection-change layer**

Apply the smallest fix at the failing layer. Preserve:
- Initial main-route `compute-buffer-selected-hour` proof.
- CPU/dataTexture fallback behavior.
- Debug route production source freeze.

- [x] **Step 3: Verify the focused slice**

Run:

```powershell
cd viewer
npx vitest run tests/compute/live-selected-hour-controller.test.ts tests/compute/live-selected-hour-route-host.test.ts tests/compute/live-selected-hour-route-projection.test.ts tests/routes/main-route-overlay-gating-helper.test.ts tests/services/pointCloudService.surface.test.ts
npx playwright test tests/e2e/main-route-manual-diagnostics.spec.ts --project=chromium --workers=1
```

Expected:
- Focused unit slice passes.
- Main-route diagnostic spec proves both initial publication and selection-change republication on `compute-buffer-selected-hour`.
- If Playwright harness fails to start/stop cleanly, report harness debt separately from route behavior.

Task 3.1 result:

```text
Root cause: while an old visible surface remained published, the host/projection could feed the scene the new pending surface identity with the old published render context. The scene could therefore fail to recopy the new selected-hour output or pair request identity and render context incoherently.
Fix: scene identities and render contexts now follow the current pending controller request when current/error-free, while published surface identity/readiness still represent the last visible published surface until replacement completes. Non-live comparison projection was also restored.
Red evidence: host blocker test failed with baseRenderContext on request 1 (Ben-Gurion/base|7|12|180) while scene identity expected request 2 (Ben-Gurion/base|1|9|33); projection blocker test failed because non-live comparison analysis was undefined.
Verification: host/projection focused run passed, 2 files / 37 tests. Focused unit slice passed, 5 files / 71 tests. Main-route diagnostic spec passed, 2 browser tests, including real UI hour change and new compute-buffer publication.
Note: manual follow-up found that the month dial still did not change the visible color field because the main route was still deriving the WebGPU selected `timeIndex` from single-month `.bin` metadata. That follow-up is tracked in Task 3.2 before Task 3.5.
```

---

## Task 3.2: Fix Full-Year WebGPU Time Index Selection On The Main Route

**Purpose:** The main route must ask the live WebGPU session for the full-year selected hour (`monthIndex * 24 + hourIndex`). The loaded product-route analysis is still the August `.bin` metadata, so `getEffectiveHourIndex($analysisStore, hour, month)` collapses every month to the same 0-23 slice and makes the month dial visually inert.

**Files:**
- Edit: `viewer/src/lib/compute/liveUtciSelectedHour.ts`
- Edit: `viewer/src/routes/+page.svelte`
- Edit: `viewer/tests/e2e/main-route-manual-diagnostics.spec.ts`
- Create: `viewer/tests/compute/live-selected-hour-time-index.test.ts`
- Avoid debug route production edits.

- [x] **Step 1: Add a shared live selected-hour time-index helper**

Add `resolveLiveSelectedHourTimeIndex({ monthIndex, hourIndex, numHours = 24 })` in the selected-hour live compute helper layer.

Expected:
- `0,0 -> 0`
- `7,12 -> 180`
- `10,18 -> 258`

- [x] **Step 2: Route the main WebGPU selection through the live helper**

In `viewer/src/routes/+page.svelte`, replace the product-route selected-time-index derivation:

```ts
getEffectiveHourIndex($analysisStore, selectedHourIndex, selectedMonthIndex)
```

with:

```ts
resolveLiveSelectedHourTimeIndex({
	monthIndex: selectedMonthIndex,
	hourIndex: selectedHourIndex
})
```

Expected:
- The main route no longer derives WebGPU compute time from `.bin` month metadata.
- Selection key remains the product route key shape: `analysisId|monthIndex|hourIndex`.

- [x] **Step 3: Align tooltip lookup with the live render context**

When the live WebGPU route is active, pass the projected live render context `timeIndex` into `getTooltipData(...)` instead of the 0-23 UI hour.

Expected:
- Tooltip values and the visible WebGPU surface refer to the same selected full-year slice.

- [x] **Step 4: Extend the short browser proof**

Extend `viewer/tests/e2e/main-route-manual-diagnostics.spec.ts` to assert:
- Initial August hour 0 publishes `baseSelectedTimeIndex = 7 * 24`.
- Hour 1 publishes `baseSelectedTimeIndex = 7 * 24 + 1`.
- Month 8 hour 0 publishes `baseSelectedTimeIndex = 8 * 24`.

Expected:
- The test drives the real radial month UI instead of only poking store state.

- [x] **Step 5: Verify the focused slice**

Run:

```powershell
cd viewer
npx vitest run tests/compute/live-selected-hour-time-index.test.ts tests/compute/live-selected-hour-route-host.test.ts tests/compute/live-selected-hour-route-projection.test.ts
npx vitest run tests/compute/live-selected-hour-controller.test.ts tests/services/pointCloudService.surface.test.ts
npx playwright test tests/e2e/main-route-manual-diagnostics.spec.ts --project=chromium --workers=1
```

Task 3.2 result:

```text
Root cause: `/` computed live WebGPU selectedTimeIndex through getEffectiveHourIndex($analysisStore, ...). The product route analysis metadata is still the August .bin/full-day artifact with no num_months, so month selection collapsed to 0-23 and the month dial could not affect WebGPU compute.
Fix: added resolveLiveSelectedHourTimeIndex() for live WebGPU year indexing, used it in `/`, and made live-route tooltips use the projected live render context timeIndex.
Verification: time-index/host/projection run passed, 3 files / 38 tests. Controller/surface run passed, 2 files / 30 tests. Main-route diagnostic spec passed, 3 browser tests, including real UI hour and month changes.
Limit: SDD subagent review could not be dispatched in-session because the agent thread limit was already reached.
```

---

## Task 3.3: Decouple Main-Route UTCI Color Range From .bin Metadata

**Purpose:** The main route should be a WebGPU-first product route. `.bin` ranges, `.bin` comparison, and `.bin` month assumptions are debug-route concerns only. Even when the loaded product-route analysis still carries August `.bin` metadata, `/` must not use that metadata to color the GPU-resident UTCI surface.

**Files:**
- Modify: `viewer/src/lib/compute/liveUtciSelectedHour.ts`
- Modify: `viewer/tests/compute/live-selected-hour-time-index.test.ts`
- Modify: `viewer/tests/e2e/main-route-manual-diagnostics.spec.ts`
- Do not edit: `viewer/src/routes/debug-webgpu-utci/+page.svelte`

- [x] **Step 1: Remove `.bin` metadata from live-route accepted range resolution**

`resolveAcceptedGpuResidentUtciRange(...)` now returns the route-level live UTCI display range for normalized GPU-resident rendering instead of consulting `base.metadata.utci_range`, `hour_statistics`, or source `.bin` date/month fields.

Expected:
- August and non-August selections use the same live-route range semantics on `/`.
- Debug route remains the only place that can preserve `.bin` comparison/range behavior.

- [x] **Step 2: Keep explicit selected-hour readback range only for callers that provide values**

`selectedHourUtci` is still accepted for explicit per-hour range calculation in discrete mode. The main GPU-resident path does not currently provide CPU selected-hour values, so it remains decoupled from `.bin` metadata.

Expected:
- This preserves the helper's non-main-route utility without reintroducing `.bin` metadata into `/`.

- [x] **Step 3: Update focused tests to enforce no `.bin` range on `/`**

`viewer/tests/compute/live-selected-hour-time-index.test.ts` now asserts that even an August analysis with `.bin` metadata resolves to the live-route display range for normalized coloring.

`viewer/tests/e2e/main-route-manual-diagnostics.spec.ts` now asserts that Ness Tziona starts with the live-route range and keeps that range after changing to January.

Expected:
- A future regression that reuses `.bin` ranges on `/` fails in both compute and app-visible diagnostics tests.

- [x] **Step 4: Verify the focused slice**

Run:

```powershell
cd viewer
npx vitest run tests/compute/live-selected-hour-time-index.test.ts tests/compute/live-selected-hour-controller.test.ts tests/compute/live-selected-hour-route-host.test.ts tests/compute/live-selected-hour-route-projection.test.ts
npx playwright test tests/e2e/main-route-manual-diagnostics.spec.ts --project=chromium --workers=1
```

Expected:
- Focused unit tests pass.
- Main-route diagnostic spec passes without using `.bin` ranges for Ness Tziona.

Task 3.3 result:

```text
Root cause: after fixing full-year time indexing, the main route still needed the old visual color semantics but could not keep deriving them from the loaded August .bin metadata. A temporary fixed -20..60 fallback made colors wrong and also changed a shared helper used by the frozen debug route.
Fix: split the policies. resolveAcceptedGpuResidentUtciRange() remains metadata-aware for debug-route callers. The main live GPU-resident session now uses resolveLiveGpuResidentUtciRange() with WebGPU selected-hour readback values, attaches those values to the selected-hour analysis for tooltip lookup, and keeps .bin metadata out of `/` range selection.
Verification: focused unit run passed, 5 files / 61 tests. Main-route diagnostic spec passed, 4 browser tests, including Ness Tziona August and January selections using finite live WebGPU ranges independent of .bin metadata and not the fixed -20..60 fallback.
Note: tests/compute/live-utci-analysis.test.ts still has two existing failures unrelated to this range split: telemetry readback stage expectation and unexpected-slice-length rejection. The selected-hour range test in that file passes again after restoring the metadata-aware helper.
```

---

## Task 3.4: Align Live WebGPU Full-Day Ranges And Legend Values

**Purpose:** After `/` began using WebGPU-derived selected-hour ranges, manual testing showed two related issues: full-day mode still behaved like per-hour mode, and the UTCI legend still showed values derived from the loaded `.bin` analysis instead of the live WebGPU surface. Do not proceed to Task 3.5 until the mesh and legend use the same live range.

**Files:**
- Modify: `viewer/src/lib/compute/liveUtciSelectedHour.ts`
- Modify: `viewer/src/lib/compute/liveUtciSelectedHourSession.ts`
- Modify: `viewer/src/lib/components/ui/ColorLegend.svelte`
- Modify: `viewer/src/routes/+page.svelte`
- Modify: `viewer/tests/compute/live-selected-hour-session.test.ts`
- Modify: `viewer/tests/e2e/main-route-manual-diagnostics.spec.ts`

- [x] **Step 1: Preserve distinct full-day and per-hour WebGPU range semantics**

For live GPU-resident selected-hour rendering:
- `discrete` / per-hour mode uses the selected-hour WebGPU UTCI readback range.
- `normalized` / full-day mode computes and caches the selected month/day WebGPU UTCI range across the 24 hourly slices.

Expected:
- Moving the hour dial in full-day mode keeps the same month/day range.
- Switching to per-hour mode can narrow the range to the selected-hour values.

- [x] **Step 2: Keep legend and mesh range in sync**

`ColorLegend.svelte` now accepts an explicit UTCI range override. The main route passes the accepted live GPU-resident range to the legend when live UTCI is active.

Expected:
- The legend labels match the WebGPU surface range instead of the loaded August `.bin` metadata.

- [x] **Step 3: Verify focused unit and browser coverage**

Run:

```powershell
cd viewer
npx vitest run tests/compute/live-selected-hour-time-index.test.ts tests/compute/live-selected-hour-session.test.ts tests/compute/live-selected-hour-controller.test.ts tests/compute/live-selected-hour-route-host.test.ts tests/compute/live-selected-hour-route-projection.test.ts
npx playwright test tests/e2e/main-route-manual-diagnostics.spec.ts --project=chromium --workers=1
```

Task 3.4 result:

```text
Root cause: the live GPU session only attached selected-hour readback values, so both full-day and per-hour modes used a per-hour range. The legend also computed from analysisStore, which is still the loaded .bin analysis on the main route.
Fix: normalized live WebGPU mode now computes/caches a selected-month 24-hour range; discrete mode keeps the selected-hour range. The main route passes the accepted live range into ColorLegend.
Verification: focused unit run passed, 5 files / 62 tests. Main-route diagnostic spec passed, 4 browser tests, including full-day range stability across hour change, per-hour narrowing, and legend labels matching the accepted live range for Ness Tziona.
```

---

## Task 3.5: Capture The Debug Baseline Before Shared Scene Edits

**Purpose:** Shared scene components are used by both `/` and `/debug-webgpu-utci`. Before changing them, capture a short debug baseline so “frozen debug route” is enforceable even when shared code changes.

**Files:**
- Create: `viewer/tests/e2e/debug-route-baseline-diagnostics.spec.ts`
- Read only: `viewer/src/routes/debug-webgpu-utci/+page.svelte`

- [x] **Step 1: Create a short debug baseline probe**

Create `viewer/tests/e2e/debug-route-baseline-diagnostics.spec.ts`:

```ts
import { expect, test } from '@playwright/test';

test.describe('debug route baseline diagnostics probe', () => {
	test('publishes selected-hour GPU diagnostics on the frozen debug baseline', async ({ page }) => {
		test.setTimeout(30_000);
		await page.goto('/debug-webgpu-utci?analysis=Ben-Gurion%2F20250815_grid_2m_fullday&utciRender=gpu');

		const diagnosticsHandle = await page.waitForFunction(() => {
			const value = (window as any).__onDemandPrototypeDiagnostics__;
			if (
				value?.utciSurfaceSource === 'compute-buffer-selected-hour' &&
				value?.renderTransport === 'compute-buffer-selected-hour' &&
				value?.sameDeviceForComputeAndRender === true
			) {
				return value;
			}
			return null;
		}, undefined, { timeout: 20_000 });

		const diagnostics = await diagnosticsHandle.jsonValue() as any;
		expect(diagnostics.utciSurfaceSource).toBe('compute-buffer-selected-hour');
		expect(diagnostics.renderTransport).toBe('compute-buffer-selected-hour');
		expect(diagnostics.sameDeviceForComputeAndRender).toBe(true);
	});
});
```

- [x] **Step 2: Run the debug baseline probe**

Run:

```powershell
cd viewer
npx playwright test tests/e2e/debug-route-baseline-diagnostics.spec.ts --project=chromium --workers=1
```

Expected:
- Passes within 30 seconds, or fails with a concrete diagnostics object.
- If it hangs, kill only repo-local Playwright/dev-server processes using the Task 1 commands and report harness debt.

- [x] **Step 3: Store the baseline result in task notes**

Record:

```text
Debug baseline route: /debug-webgpu-utci?analysis=Ben-Gurion%2F20250815_grid_2m_fullday&utciRender=gpu
Expected selected-hour source: compute-buffer-selected-hour
Expected transport: compute-buffer-selected-hour
Expected same-device: true
Result: <actual pass/fail and diagnostics>
```

Expected:
- Task 4 shared component edits must run this probe again after changes.

Task 3.5 result:

```text
Debug baseline route: /debug-webgpu-utci?analysis=Ben-Gurion%2F20250815_grid_2m_fullday&utciRender=gpu
Expected selected-hour source: compute-buffer-selected-hour
Expected transport: compute-buffer-selected-hour
Expected same-device: true
Result: PASS. npx playwright test tests/e2e/debug-route-baseline-diagnostics.spec.ts --project=chromium --workers=1 passed, 1 browser test, diagnostics matched the expected compute-buffer-selected-hour baseline.
```

---

## Task 4: Extract Shared GPU-Resident UTCI Surface Sync

**Purpose:** Remove duplicated pure GPU-resident surface key/detection/diagnostics logic from `UTCIPointCloud.svelte` and `ComparisonRenderer.svelte` without moving Threlte lifecycle or GPU queue-copy orchestration into a vague callback helper.

**Files:**
- Create: `viewer/src/lib/components/scene/utciSurfaceSync.ts`
- Create: `viewer/tests/scene/utci-surface-sync.test.ts`
- Modify: `viewer/src/lib/components/scene/UTCIPointCloud.svelte`
- Modify: `viewer/src/lib/components/scene/ComparisonRenderer.svelte`
- Read: `viewer/src/lib/services/gpuUtciRenderBridge.ts`
- Read: `viewer/src/lib/services/pointCloudService.ts`

- [x] **Step 1: Create helper types and pure key helpers**

Create `viewer/src/lib/components/scene/utciSurfaceSync.ts` with:

```ts
import type { Mesh } from 'three';
import type { SelectedHourGpuResidentOutput } from '$lib/compute/liveUtciSelectedHourSession';
import type { SelectedHourRenderTimingSubsteps } from '$lib/compute/onDemandDiagnostics';
import { getGpuNativeUtciSurfaceSource } from '$lib/services/gpuUtciRenderBridge';

export type GpuResidentCopyStatus = 'idle' | 'pending' | 'complete' | 'failed';

export type UtciSurfaceDiagnostics = {
	utciSurfaceSource?: string;
	selectedHourTransferCount?: number;
	dataTextureBuildCount?: number;
	cpuPublishRequestId?: number;
	cpuPublishMonthIndex?: number;
	cpuPublishHourIndex?: number;
	cpuPublishTimeIndex?: number;
	cpuPublishSelectionKey?: string;
	gpuResidentCopyStatus?: GpuResidentCopyStatus;
	gpuResidentCopyError?: string;
	gpuResidentCopyRequestId?: number;
} & SelectedHourRenderTimingSubsteps;

export function getAcceptedGpuResidentKey(
	value: SelectedHourGpuResidentOutput | null
): string | null {
	if (!value) return null;
	return `${value.requestId}:${value.monthIndex}:${value.timeIndex}:${value.utciRange.min}:${value.utciRange.max}`;
}

export function isComputeBufferUtciSurface(mesh: Mesh | null): boolean {
	return mesh != null && getGpuNativeUtciSurfaceSource(mesh) === 'compute-buffer-selected-hour';
}
```

Expected:
- No Svelte imports.
- No route imports.

- [x] **Step 2: Add helper tests for stable key and surface detection**

Create `viewer/tests/scene/utci-surface-sync.test.ts`:

```ts
import { describe, expect, it } from 'vitest';
import { Mesh } from 'three';
import {
	getAcceptedGpuResidentKey,
	isComputeBufferUtciSurface
} from '../../src/lib/components/scene/utciSurfaceSync';

describe('utciSurfaceSync', () => {
	it('builds a stable accepted GPU resident key from request and range', () => {
		const key = getAcceptedGpuResidentKey({
			requestId: 5,
			monthIndex: 7,
			hourIndex: 12,
			timeIndex: 180,
			utciRange: { min: 18.5, max: 41.25 },
			output: {
				format: 'f32-utci',
				numPoints: 2,
				timeIndex: 180,
				gpuBuffer: {} as GPUBuffer
			}
		});

		expect(key).toBe('5:7:180:18.5:41.25');
	});

	it('does not treat userData alone as authoritative compute-buffer proof', () => {
		const mesh = new Mesh();
		mesh.userData.utciSurfaceSource = 'compute-buffer-selected-hour';
		expect(isComputeBufferUtciSurface(mesh)).toBe(false);
	});
});
```

Expected:
- Test compiles against the actual `SelectedHourGpuResidentOutput` type. If the type requires additional fields, add explicit fake values in the test instead of weakening production types.

- [x] **Step 3: Move CPU diagnostics shaping into the helper**

Add:

```ts
import type { LiveSelectedHourSurfaceIdentity } from '$lib/compute/liveSelectedHourSurfaceIdentity';

export function buildCpuPublicationDiagnostics(params: {
	mesh: Mesh | null;
	liveSelectedHourSurfaceIdentity: LiveSelectedHourSurfaceIdentity | null;
	isComputeBufferSurface: boolean;
}): Partial<UtciSurfaceDiagnostics> {
	if (
		params.mesh == null ||
		params.isComputeBufferSurface ||
		params.liveSelectedHourSurfaceIdentity == null
	) {
		return {};
	}
	return {
		utciSurfaceSource: 'cpu-uploaded-selected-hour',
		cpuPublishRequestId: params.liveSelectedHourSurfaceIdentity.requestId,
		cpuPublishMonthIndex: params.liveSelectedHourSurfaceIdentity.monthIndex,
		cpuPublishHourIndex: params.liveSelectedHourSurfaceIdentity.hourIndex,
		cpuPublishTimeIndex: params.liveSelectedHourSurfaceIdentity.timeIndex,
		cpuPublishSelectionKey: params.liveSelectedHourSurfaceIdentity.selectionKey
	};
}
```

Add a test:

```ts
it('builds request-scoped CPU publication diagnostics for non-compute surfaces', () => {
	const mesh = new Mesh();
	const diagnostics = buildCpuPublicationDiagnostics({
		mesh,
		isComputeBufferSurface: false,
		liveSelectedHourSurfaceIdentity: {
			requestId: 9,
			monthIndex: 1,
			hourIndex: 8,
			timeIndex: 32,
			selectionKey: 'selection',
			pendingRenderUpdateStartedAt: undefined,
			acceptedGpuResidentOutput: null
		}
	});

	expect(diagnostics).toMatchObject({
		utciSurfaceSource: 'cpu-uploaded-selected-hour',
		cpuPublishRequestId: 9,
		cpuPublishSelectionKey: 'selection'
	});
});
```

- [x] **Step 4: Do not move GPU queue-copy orchestration in this pass**

Implementation rule:
- Keep active sync key, run token, last analysis/backend, scene add/remove, visibility, storage-buffer waits, and queue-copy logic inside the Svelte components for now.
- Extract only pure helpers: stable accepted-output key, authoritative compute-buffer detection, CPU publication diagnostics, and common diagnostics object assembly.
- Do not introduce a callback-heavy `syncGpuResidentSurface(...)` mega-helper in this plan.
- Do not change the visual behavior in this task.

- [x] **Step 5: Update `UTCIPointCloud.svelte` to use the helper**

Replace local duplicate functions:

```ts
function getAcceptedGpuResidentKey(...)
function isComputeBufferSurface(...)
function publishUtciSurfaceDiagnostics(...)
```

with imports from `utciSurfaceSync.ts` where behavior is identical.

Expected:
- `UTCIPointCloud.svelte` keeps Svelte lifecycle and Threlte access.
- Request-scoped diagnostics remain identical.

- [x] **Step 6: Update `ComparisonRenderer.svelte` to use the same helper**

Apply the same imports and helper calls.

Expected:
- Base and comparison now share diagnostics/key/surface-source semantics.
- Comparison-specific clipping/scissor/curtain logic remains in `ComparisonRenderer.svelte`.

- [x] **Step 7: Run focused scene/helper tests**

Run:

```powershell
cd viewer
npx vitest run tests/scene/utci-surface-sync.test.ts tests/compute/live-selected-hour-route-projection.test.ts tests/compute/live-selected-hour-controller.test.ts
```

Expected:
- Helper tests pass.
- Controller/projection tests still pass.

- [x] **Step 8: Rerun the debug baseline probe after shared scene edits**

Run:

```powershell
cd viewer
npx playwright test tests/e2e/debug-route-baseline-diagnostics.spec.ts --project=chromium --workers=1
```

Expected:
- The same debug selected-hour diagnostics still pass.
- If this fails after Task 4, stop and fix the shared scene regression before rebuilding the main route further.

Task 4 result:

```text
Implementation: created utciSurfaceSync.ts with shared stable accepted-output keying, authoritative compute-buffer surface detection, CPU publication diagnostics, common surface diagnostics assembly, and shared copy-status typing. Updated UTCIPointCloud.svelte and ComparisonRenderer.svelte to use those helpers while keeping active sync keys, run tokens, storage-buffer waits, GPU queue-copy, scene lifecycle, and visibility orchestration local to the Svelte components.
Spec review: initial review found common diagnostics assembly was still duplicated. Fixed by adding buildUtciSurfaceDiagnostics() and using it in both scene components.
Code-quality review: initial review found caller-supplied compute-buffer classification and stale getEffectiveHourIndex imports. Fixed by deriving compute-buffer classification inside buildCpuPublicationDiagnostics() and removing stale imports. Re-review approved Task 4 code quality with no findings.
Verification: npx vitest run tests/scene/utci-surface-sync.test.ts tests/compute/live-selected-hour-route-projection.test.ts tests/compute/live-selected-hour-controller.test.ts passed, 3 files / 32 tests. npx playwright test tests/e2e/debug-route-baseline-diagnostics.spec.ts --project=chromium --workers=1 passed, 1 browser test.
Deferred intentionally: deeper storage-wait/copy lifecycle extraction remains duplicated by design because Task 4 Step 4 explicitly prohibited moving GPU queue-copy orchestration in this pass.
```

---

## Task 5: Rebuild The Main Route As A Thin Coordinator

**Purpose:** Finish the main-route cleanup without touching the debug route. The route should wire stores, model loading, UI, host inputs, and scene props. It should not own selected-hour state-machine internals.

**Files:**
- Modify: `viewer/src/routes/+page.svelte`
- Modify or keep: `viewer/src/lib/components/viewer/ViewerShell.svelte`
- Modify or keep: `viewer/src/routes/mainRouteOverlayGating.ts`
- Read: `viewer/src/lib/compute/liveSelectedHourRouteHost.ts`
- Read: `viewer/src/lib/compute/liveSelectedHourRouteProjection.ts`

- [x] **Step 1: List route-owned selected-hour internals and remove any leftovers**

Search:

```powershell
rg -n "SelectionTriggerKey|comparisonSourceAnalysis|requestSelection|prepareSession|acceptedGpuResident|renderTransport|sameDeviceForComputeAndRender" viewer/src/routes/+page.svelte
```

Allowed after cleanup:
- Diagnostic field names inside `updateUtciRenderDiagnostics`.
- Props passed to scene components.
- `liveRouteHost.setRouteInputs`.

Not allowed:
- Route-owned trigger keys.
- Route-owned selected-hour sessions/controllers.
- Route-owned comparison source-analysis ownership.

- [x] **Step 2: Keep one host-input reactive block**

The main route should have one live-host input block in this shape:

```ts
$: liveRouteHost.setRouteInputs({
	enabled: useLiveUtciOnMainRoute,
	analysisId,
	baseAnalysis: $analysisStore,
	baseModel: modelLoading ? null : model,
	selection: {
		monthIndex: selectedMonthIndex,
		hourIndex: selectedHourIndex,
		timeIndex: selectedTimeIndex,
		selectionKey: [analysisId, selectedMonthIndex, selectedHourIndex].join('|')
	},
	colorMode: $viewerStore.colorMode,
	utciRenderMode,
	rendererBackend,
	rendererDevice: rendererDeviceForMain,
	utciSurfaceBackend: resolvedUtciSurfaceBackend,
	comparison: {
		active: isComparing,
		analysisId: $comparisonStore.comparisonAnalysisId,
		sourceAnalysis: $comparisonStore.isLoading ? null : $comparisonAnalysis,
		model: comparisonModelForLiveCompute,
		rendererDevice: rendererDeviceForMain
	}
});
```

Expected:
- If this exact block already exists, keep it and do not churn formatting.
- If additional route-owned selected-hour reactive blocks exist, remove them.

- [x] **Step 3: Keep one host-to-scene projection block**

The route should project host state through `projectMainRouteLiveSceneState`:

```ts
$: ({
	baseDisplayedAnalysis,
	comparisonRendererDisplayAnalysis,
	baseLiveReady,
	comparisonLiveReady,
	baseHasVisibleLiveSurface,
	comparisonHasVisibleLiveSurface,
	baseSceneAnalysis,
	comparisonSceneAnalysis,
	baseSceneRenderContext,
	comparisonSceneRenderContext,
	baseSceneSurfaceIdentity,
	comparisonSceneSurfaceIdentity,
	basePendingGpuResidentOutput,
	comparisonPendingGpuResidentOutput,
	basePendingRenderUpdateStartedAt,
	comparisonPendingRenderUpdateStartedAt
} = projectMainRouteLiveSceneState({
	useLiveUtciOnMainRoute,
	isComparing,
	baseAnalysis: $analysisStore,
	comparisonAnalysis: $comparisonAnalysis,
	liveRouteState
}));
```

Expected:
- Route template consumes these projected values.
- The route does not recreate projection logic inline.

- [x] **Step 4: Ensure the scene receives bootstrap props**

The base UTCI component call should include:

```svelte
<UTCIPointCloud
	analysis={baseSceneAnalysis}
	{model}
	bind:utciSurface={utciMesh}
	acceptedGpuResidentOutput={basePendingGpuResidentOutput}
	selectedHourRenderContext={baseSceneRenderContext}
	liveSelectedHourSurfaceIdentity={baseSceneSurfaceIdentity}
	onUtciSurfaceDiagnostics={handleUtciSurfaceDiagnostics}
	pendingRenderUpdateStartedAt={basePendingRenderUpdateStartedAt}
	utciSurfaceBackend={resolvedUtciSurfaceBackend}
/>
```

Expected:
- `analysis`, `acceptedGpuResidentOutput`, `selectedHourRenderContext`, and `liveSelectedHourSurfaceIdentity` all come from the projection.

- [x] **Step 5: Ensure comparison receives the same contract**

The comparison renderer call should include:

```svelte
<ComparisonRenderer
	bind:this={comparisonRenderer}
	acceptedGpuResidentOutput={comparisonPendingGpuResidentOutput}
	baseCamera={cameraRef}
	displayAnalysis={comparisonSceneAnalysis}
	selectedHourRenderContext={comparisonSceneRenderContext}
	liveSelectedHourSurfaceIdentity={comparisonSceneSurfaceIdentity}
	onUtciSurfaceDiagnostics={handleComparisonUtciSurfaceDiagnostics}
	pendingRenderUpdateStartedAt={comparisonPendingRenderUpdateStartedAt}
	utciSurfaceBackend={resolvedUtciSurfaceBackend}
/>
```

Expected:
- Comparison does not have a separate CPU-only selected-hour seam.

- [x] **Step 6: Keep diagnostics route-level but host-sourced**

`updateUtciRenderDiagnostics` should source selected-hour state from:

```ts
liveRouteState.base
liveRouteState.comparison
liveRouteState.baseSurfaceIdentity
liveRouteState.baseSceneSurfaceIdentity
liveRouteState.comparisonSurfaceIdentity
liveRouteState.comparisonSceneSurfaceIdentity
```

Expected:
- Diagnostics are still route-specific.
- Diagnostics do not drive the selected-hour state machine.

- [x] **Step 7: Run focused main-route non-browser tests**

Run:

```powershell
cd viewer
npx vitest run tests/compute/live-selected-hour-route-projection.test.ts tests/routes/main-route-overlay-gating-helper.test.ts tests/compute/live-selected-hour-route-host.test.ts
```

Expected:
- All pass.

- [x] **Step 8: Run the short manual diagnostics probe only**

Run:

```powershell
cd viewer
npx playwright test tests/e2e/main-route-manual-diagnostics.spec.ts --project=chromium --workers=1
```

Expected:
- Pass, or fail with a concrete diagnostics object.
- Do not run broad `--grep "main route"` suites in this task.

Task 5 result:

```text
Implementation: kept the existing single liveRouteHost.setRouteInputs() block and single projectMainRouteLiveSceneState() projection block, removed stale published-render-context variables, and collapsed baseLiveControllerState/comparisonLiveControllerState mirrors so diagnostics, overlay gating, and error UI read liveRouteState.base / liveRouteState.comparison directly.
Spec review: initial review found local host-state mirrors. Fixed; re-review found no Task 5 compliance findings in the working tree. It also noted inherited dirty/staged debug-route production edits and stale staged +page.svelte content as workflow caveats, not current working-tree Task 5 code findings.
Code-quality review: approved Task 5 main-route cleanup with no blocking or non-blocking findings. Reviewer noted the route still reads as a coordinator: host inputs go into liveRouteHost, scene values come through projection, and overlay policy stays in mainRouteOverlayGating.
Verification: npx vitest run tests/compute/live-selected-hour-route-projection.test.ts tests/routes/main-route-overlay-gating-helper.test.ts tests/compute/live-selected-hour-route-host.test.ts passed, 3 files / 41 tests. npx playwright test tests/e2e/main-route-manual-diagnostics.spec.ts --project=chromium --workers=1 passed, 4 browser tests. git diff --check for +page.svelte and this plan passed with only LF-to-CRLF warnings.
Workflow caveat: viewer/src/routes/+page.svelte is MM; the staged blob may not include the latest mirror-removal fix. Refresh the index before any future commit. Inherited debug-route production edits remain dirty/staged from earlier work and were not touched in Task 5.
```

---

## Task 6: Extract Debug-Only Pure Diagnostics Helpers Without Changing Baseline Behavior

**Purpose:** Start rebuilding `/debug-webgpu-utci` last, after `/` is visible and cleaner. Move only pure debug diagnostics/window-shaping helpers first, preserving the route as the proof/baseline surface.

**Files:**
- Create: `viewer/src/lib/debug/debugWebgpuUtciDiagnostics.ts`
- Create: `viewer/tests/debug/debug-webgpu-utci-diagnostics.test.ts`
- Modify: `viewer/src/routes/debug-webgpu-utci/+page.svelte`
- Read: `viewer/src/lib/compute/onDemandPrototypeStatus.ts`

- [x] **Step 1: Define debug diagnostics helper responsibilities**

Create `viewer/src/lib/debug/debugWebgpuUtciDiagnostics.ts` with:

```ts
import type { UtciRenderMode } from '$lib/utciRenderMode';

export type DebugWebgpuUtciDiagnosticsInputs = {
	parityMode: boolean;
	collectMode: 'off' | 'normal';
	debugOnDemandMode: 'off' | 'f32';
	utciRenderMode: UtciRenderMode;
	selectedMonthIndex: number;
	selectedHourIndex: number;
	selectedTimeIndex: number;
};

export type DebugWebgpuUtciDiagnosticsState = {
	onDemandEnabled: boolean;
	binComparisonEnabled: boolean;
	collectNormalMode: boolean;
	windowDiagnosticsEnabled: boolean;
};
```

- [x] **Step 2: Implement pure debug diagnostics derivation**

Add:

```ts
export function deriveDebugWebgpuUtciDiagnosticsState(
	inputs: DebugWebgpuUtciDiagnosticsInputs
): DebugWebgpuUtciDiagnosticsState {
	return {
		onDemandEnabled: inputs.debugOnDemandMode === 'f32',
		binComparisonEnabled: inputs.parityMode,
		collectNormalMode: inputs.collectMode === 'normal',
		windowDiagnosticsEnabled: true
	};
}
```

Expected:
- This first debug helper is intentionally small.
- Do not add subscriptions or stateful lifecycle machinery yet.

- [x] **Step 3: Add tests for debug-only decisions**

Create `viewer/tests/debug/debug-webgpu-utci-diagnostics.test.ts`:

```ts
import { describe, expect, it } from 'vitest';
import { deriveDebugWebgpuUtciDiagnosticsState } from '../../src/lib/debug/debugWebgpuUtciDiagnostics';

describe('debugWebgpuUtciDiagnostics', () => {
	it('keeps on-demand enabled only for f32 debug mode', () => {
		const state = deriveDebugWebgpuUtciDiagnosticsState({
			parityMode: false,
			collectMode: 'off',
			debugOnDemandMode: 'f32',
			utciRenderMode: 'auto',
			selectedMonthIndex: 7,
			selectedHourIndex: 12,
			selectedTimeIndex: 180
		});
		expect(state.onDemandEnabled).toBe(true);
	});

	it('keeps normal collect mode distinct from on-demand diagnostics', () => {
		const state = deriveDebugWebgpuUtciDiagnosticsState({
			parityMode: false,
			collectMode: 'normal',
			debugOnDemandMode: 'off',
			utciRenderMode: 'data',
			selectedMonthIndex: 7,
			selectedHourIndex: 12,
			selectedTimeIndex: 180
		});
		expect(state).toMatchObject({
			onDemandEnabled: false,
			collectNormalMode: true
		});
	});

	it('allows bin comparison only as debug/parity behavior', () => {
		const state = deriveDebugWebgpuUtciDiagnosticsState({
			parityMode: true,
			collectMode: 'off',
			debugOnDemandMode: 'f32',
			utciRenderMode: 'gpu',
			selectedMonthIndex: 7,
			selectedHourIndex: 12,
			selectedTimeIndex: 180
		});
		expect(state.binComparisonEnabled).toBe(true);
	});
});
```

Expected:
- Tests clarify that `.bin` comparison remains debug-only.

- [x] **Step 4: Integrate the debug diagnostics helper incrementally**

In `viewer/src/routes/debug-webgpu-utci/+page.svelte`, add:

```ts
import { deriveDebugWebgpuUtciDiagnosticsState } from '$lib/debug/debugWebgpuUtciDiagnostics';
```

Reactive input:

```ts
$: debugDiagnosticsState = deriveDebugWebgpuUtciDiagnosticsState({
	parityMode,
	collectMode: normalCollectMode ? 'normal' : 'off',
	debugOnDemandMode,
	utciRenderMode,
	selectedMonthIndex,
	selectedHourIndex,
	selectedTimeIndex
});
```

Expected:
- This is a first debug-helper foothold, not a wholesale rewrite.
- Existing debug behavior remains intact.

- [x] **Step 5: Move only debug diagnostics shaping that is proven pure**

Move pure object-shaping helpers from the debug route into `debugWebgpuUtciDiagnostics.ts` only when they:
- do not touch DOM,
- do not touch Threlte,
- do not run WebGPU commands,
- do not depend on local Svelte component lifecycle.

Example acceptable function:

```ts
export function shouldExposeDebugWindowDiagnostics(state: DebugWebgpuUtciDiagnosticsState): boolean {
	return state.windowDiagnosticsEnabled;
}
```

Expected:
- Debug route shrinks gradually.
- Debug-only behavior does not leak into the shared host.

- [x] **Step 6: Run debug diagnostics tests and baseline probe**

Run:

```powershell
cd viewer
npx vitest run tests/debug/debug-webgpu-utci-diagnostics.test.ts
npx playwright test tests/e2e/debug-route-baseline-diagnostics.spec.ts --project=chromium --workers=1
```

Expected:
- Tests pass.
- The short debug baseline still passes.
- No broad debug Playwright run yet.

Task 6 result:

```text
Implementation: added debugWebgpuUtciDiagnostics.ts as a small pure debug-only helper. It derives on-demand/debug parity/normal-collect/window-diagnostics decisions, includes render and selection context in the derived debug state, and exposes shouldExposeDebugWindowDiagnostics() as a fail-open window diagnostics gate. Integrated it incrementally in /debug-webgpu-utci without moving DOM, Threlte, WebGPU, or lifecycle code.
Bug found and fixed: the first route integration could call shouldExposeDebugWindowDiagnostics() before Svelte reactive state initialized during hydration. Fixed by allowing null/undefined state to preserve existing diagnostics exposure.
Spec review: approved Task 6; noted broad debug-route staged/dirty edits as inherited context, not this incremental helper delta.
Code-quality review: initial low finding said carried render/selection inputs looked unused. Fixed first with comments/void markers, then final Task 7 review found the helper still looked like a second source of truth because the route barely consumed its derived state. Fixed by storing render/selection context in the derived state and using the derived state for the debug route on-demand gate, selected-hour key, route update trigger, and live compute mode key.
Verification: npx vitest run tests/debug/debug-webgpu-utci-diagnostics.test.ts passed, 1 file / 6 tests. npx playwright test tests/e2e/debug-route-baseline-diagnostics.spec.ts --project=chromium --workers=1 passed, 1 browser test. Re-review also reran equivalent helper and baseline checks successfully.
```

---

## Task 7: Final Verification And Review Gates

**Purpose:** Verify behavior honestly without overstating proof. Capture remaining debt and only then decide whether the implementation is ready.

**Files:**
- Read: all files modified by this plan
- Do not create commits.

- [x] **Step 1: Run focused unit tests**

Run:

```powershell
cd viewer
npx vitest run tests/compute/live-selected-hour-controller.test.ts tests/compute/live-selected-hour-route-host.test.ts tests/compute/live-selected-hour-route-projection.test.ts tests/routes/main-route-overlay-gating-helper.test.ts tests/scene/utci-surface-sync.test.ts tests/debug/debug-webgpu-utci-diagnostics.test.ts
```

Expected:
- All focused unit tests pass.
- If failures appear, classify whether they are new plan failures or preexisting dirty-tree debt.

- [x] **Step 2: Run the short main-route diagnostic probe**

Run:

```powershell
cd viewer
npx playwright test tests/e2e/main-route-manual-diagnostics.spec.ts --project=chromium --workers=1
```

Expected:
- Passes within 30 seconds, or fails with captured diagnostics.
- If it hangs, kill stale processes using Task 1 and report harness debt.

- [x] **Step 3: Manual browser proof**

Start dev server:

```powershell
cd viewer
npm run dev -- --host 127.0.0.1
```

Open:

```text
http://127.0.0.1:5173/?analysis=Ben-Gurion%2F20250815_grid_2m_fullday&utciRender=auto&utciRenderDiagnostics=1
```

Confirm:
- UTCI layer is visibly rendered.
- The visible UTCI layer corresponds to the legend, not only the green shading layer.
- `window.__utciRenderDiagnostics__` shows `gpuNative` and selected-hour transport.

Then open:

```text
http://127.0.0.1:5173/debug-webgpu-utci?analysis=Ben-Gurion%2F20250815_grid_2m_fullday&utciRender=gpu
```

Confirm:
- Debug route still exposes selected-hour WebGPU behavior.
- Debug `.bin` / parity behavior is not moved into `/`.

- [x] **Step 4: Run type/static check**

Run:

```powershell
cd viewer
npm run check
```

Expected:
- Capture exact output.
- Separate new failures from preexisting repo debt.
- Do not claim repo-wide green unless this command returns clean.

- [x] **Step 5: Run git hygiene checks**

Run:

```powershell
git status --short
git diff --check
```

Expected:
- No whitespace errors from `git diff --check`.
- Dirty files are explainable and scoped to this plan plus preexisting known artifacts.
- No commits were made.
- No git worktrees were created.

- [x] **Step 6: Review gates before implementation is called complete**

Dispatch review subagents:

1. Spec compliance review:
   - Does the result satisfy this plan?
   - Did any task edit debug before the allowed debug step?
   - Is `/` WebGPU-first without `.bin` fallback?
2. Code quality review:
   - Are responsibilities cleaner?
   - Did helpers reduce duplication without becoming vague mega-modules?
   - Are route files thinner and easier to reason about?

Expected:
- Findings first.
- Fix findings or explicitly defer with rationale before completion is reported.

Task 7 result:

```text
Verification:
- npx vitest run tests/compute/live-selected-hour-controller.test.ts tests/compute/live-selected-hour-route-host.test.ts tests/compute/live-selected-hour-route-projection.test.ts tests/routes/main-route-overlay-gating-helper.test.ts tests/scene/utci-surface-sync.test.ts tests/debug/debug-webgpu-utci-diagnostics.test.ts passed, 6 files / 71 tests.
- npx playwright test tests/e2e/main-route-manual-diagnostics.spec.ts --project=chromium --workers=1 passed, 4 browser tests.
- npx playwright test tests/e2e/debug-route-baseline-diagnostics.spec.ts --project=chromium --workers=1 passed, 1 browser test.
- User manually verified the main route UTCI texture, hour/month color updates, full-day vs per-hour range behavior, and legend values before Task 7.
- npm run check still fails repo-wide: svelte-check found 143 errors and 4 warnings in 32 files. The plan-owned liveSelectedHourRouteHost errors found in the first check run were fixed with narrowed render-context locals and are absent from the rerun. Remaining static errors are broader repo/inherited debug-route debt, including ArrayBufferLike transfer types, parity union narrowing, Three Object3D type guards, SunPath/viewerStore typing, inherited debug-webgpu route typing, scripts, and older tests.
- git diff --check on the plan-touched Task 6/7 files passed.

Implementation note: liveSelectedHourRouteHost.ts received a narrow type-only cleanup during Task 7 so svelte-check no longer flags the current/requested render-context publication code.
Review gate:
- Spec review approved with one evidence caveat: tmp-debug-route-freeze.diff was generated as UTF-16LE/binary, so it was weaker as a reviewable baseline artifact. It was removed from the staged commit as a local temp artifact; the plan keeps the caveat for provenance.
- Code-quality review found the debug diagnostics helper looked broader than its route usage. Fixed by carrying render/selection context in state and consuming derived state in the debug route. Follow-up verification passed: focused unit slice passed, 6 files / 71 tests; main-route diagnostic probe passed, 4 browser tests; debug-route baseline probe passed, 1 browser test; git diff --check on Task 6/7 touched files passed. A final npm run check rerun still failed with the known repo-wide 143 errors / 4 warnings, with no new debugDiagnosticsState/liveSelectedHourRouteHost diagnostics in the filtered output.
```

## Completion Criteria

This plan is complete only when:

- `/` visibly shows the UTCI layer through the selected-hour WebGPU path.
- `/` does not use `.bin`, `runAll()`, or route-local fallback hacks for normal UTCI rendering.
- Debug `.bin` comparison remains debug-only.
- The debug route was not rebuilt before the main route was visually proven.
- Shared selected-hour behavior is owned by host/projection/render-sync helpers, not duplicated in routes.
- Focused unit tests pass.
- Browser proof is either passing or honestly reported with diagnostics and harness boundaries.
- No commits or git worktrees were created.
