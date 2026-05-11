# Main Route Product Shell Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Workflow overrides:** Do **not** create commits. Do **not** create git worktrees. Preserve unrelated dirty files. This plan intentionally treats `viewer/src/routes/debug/+page.svelte` as temporary/proof scaffolding; do not spend effort beautifying it unless a main-route boundary test requires a debug source-lock update.

**Goal:** Make the product route `viewer/src/routes/+page.svelte` visibly thinner and more maintainable while preserving selected-hour WebGPU behavior and keeping debug-only behavior contained.

**Architecture:** Extract a few chunky, Svelte-facing product-route boundaries instead of creating many tiny helpers. The main route should remain the coordinator for route/query state, selected-hour host inputs, diagnostics publication, and `ViewerShell` composition; viewport rendering, model-load side effects, and tooltip interaction should move into focused main-route modules. Debug route and compute folder reorganization are explicitly deferred.

**Tech Stack:** SvelteKit/Svelte 5 with legacy `$:` reactivity, TypeScript, Three/Threlte scene components, Vitest, Playwright Chromium with WebGPU, PowerShell on Windows.

---

## Context

The previous cleanup created these product-route helpers:

- `viewer/src/routes/main/liveSelectedHour.ts`
- `viewer/src/routes/main/modelSelection.ts`
- `viewer/src/routes/main/tooltip.ts`

It also extracted scene GPU-resident surface sync into:

- `viewer/src/lib/components/scene/acceptedGpuResidentSurfaceSync.ts`

Current size snapshot:

- `viewer/src/routes/+page.svelte`: about 866 lines.
- `viewer/src/routes/debug/+page.svelte`: about 4539 lines.

The debug route is temporary/proof infrastructure until the WebGPU path matures. Product-route maintainability is the priority.

## Non-Goals

- Do not reorganize `viewer/src/lib/compute/` in this plan.
- Do not further decompose `viewer/src/routes/debug/+page.svelte` except source-lock/test maintenance.
- Do not start broad `npm run check` cleanup.
- Do not convert the route to Svelte runes.
- Do not introduce a pile of one-line helper files.
- Do not remove `dataTexture`, `.bin`, Python comparison, parity, collect, debug fallback, or legacy debug selected-hour paths.
- Do not weaken `strongVisibleGpuPath` diagnostics.
- Do not create commits.
- Do not create git worktrees.

## Target End State

`viewer/src/routes/+page.svelte` should read as a product-route coordinator:

- route/query and project selection state.
- selected-hour route-host input wiring.
- projected scene state.
- diagnostics publication.
- `ViewerShell` slot composition.

Expected route size target: roughly 450-600 lines. Do not chase a smaller number if doing so creates indirection.

## File Structure Target

Create:

- `viewer/src/routes/main/MainRouteViewport.svelte`
  - Owns the main route viewport render tree: `Scene`, `Camera`, `Model`, `GridHelper`, `UTCIPointCloud`, `ComparisonRenderer`.
  - Emits model-loaded and discovered-layers events to the route.
  - Keeps scene markup and scene component bindings out of `+page.svelte`.

- `viewer/src/routes/main/modelLifecycle.ts`
  - Owns pure policy for model-loaded side effects: model bounds, scene-config bounds, first camera-fit position/target, and next `hasFitOnce`.
  - Does not mutate Svelte stores directly.

- `viewer/src/routes/main/MainRouteTooltipLayer.svelte`
  - Owns tooltip state, hover throttling, tooltip motion suppression, canvas listener attachment/detachment, and `<MetricTooltip>` rendering.
  - Exposes only telemetry counters to the route: `tooltipHoverSampleCount` and `cameraWheelEventCount`.

Modify:

- `viewer/src/routes/+page.svelte`
  - Replace inline viewport tree with `MainRouteViewport`.
  - Replace inline tooltip state/listeners/slot content with `MainRouteTooltipLayer`.
  - Keep selected-hour route-host reactive inputs explicit.
  - Keep diagnostic publication in the route, passing all dependencies explicitly.

- `viewer/src/routes/main/tooltip.ts`
  - Reuse from `MainRouteTooltipLayer`; do not duplicate hover policy.

- `viewer/tests/routes/main-route-live-selected-hour-source-lock.test.ts`
  - Add new main route files to the source-lock list and keep debug-only markers forbidden.

Add tests:

- `viewer/tests/routes/main-route-model-lifecycle.test.ts`
- `viewer/tests/routes/main-route-viewport-source-lock.test.ts`
- Optional only if needed: `viewer/tests/routes/main-route-tooltip-layer-source-lock.test.ts`

## Svelte Reactivity Rules

Follow these rules during execution:

- Do not hide reactive dependencies inside no-argument helper calls.
- Keep `$:` blocks in `+page.svelte` explicit when they drive selected-hour host inputs or diagnostics.
- Pure helpers must receive every dependency as an argument.
- Svelte component extractions can receive props and dispatch events, but event-time reads such as comparison mesh lookup should use function props only when the value must be read at event time.
- Run `svelte-autofixer` on every edited `.svelte` file, but do not migrate legacy `$:` code to runes in this plan.

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
- No commits.
- No worktrees.

- [ ] **Step 2: Record current route sizes**

Run:

```powershell
(Get-Content viewer/src/routes/+page.svelte).Count
(Get-Content viewer/src/routes/debug/+page.svelte).Count
```

Expected:

- Main route is around 866 lines.
- Debug route is around 4539 lines.
- Record the numbers in the final result note.

- [ ] **Step 3: Run product/selected-hour baseline**

Run:

```powershell
cd viewer
npx vitest run tests/routes/main-route-live-selected-hour-source-lock.test.ts tests/routes/main-route-model-selection.test.ts tests/routes/main-route-tooltip.test.ts tests/routes/main-route-overlay-gating-helper.test.ts tests/routes/main-route-live-selected-hour.test.ts tests/diagnostics/main-route-utci-diagnostics.test.ts
npm run test:quality:selected-hour
npm run test:e2e:selected-hour
```

Expected:

- Focused Vitest passes.
- `test:quality:selected-hour` passes.
- `test:e2e:selected-hour` passes with 13 Chromium tests.
- If any browser route behavior fails, stop and use `superpowers:systematic-debugging`.

- [ ] **Step 4: Run check to preserve static-debt boundary**

Run:

```powershell
cd viewer
npm run check
```

Expected:

- Likely FAIL with inherited static debt, recently 163 errors / 4 warnings in 34 files.
- Do not fix broad inherited families.
- Later touched-file errors introduced by this plan must be fixed or explicitly proven preexisting.

## Task 1: Characterize Main Route Product Shell Boundaries

**Files:**
- Modify: `viewer/tests/routes/main-route-live-selected-hour-source-lock.test.ts`
- Create: `viewer/tests/routes/main-route-viewport-source-lock.test.ts`
- Inspect: `viewer/src/routes/+page.svelte`

- [ ] **Step 1: Add optional source-lock coverage for planned main route components**

Update `viewer/tests/routes/main-route-live-selected-hour-source-lock.test.ts` so new planned files are optional until their task creates them. Do not add not-yet-created files to a required existence list.

Use this structure:

```ts
const requiredMainRoutePaths = [
	'src/routes/+page.svelte',
	'src/routes/main/liveSelectedHour.ts',
	'src/routes/main/modelSelection.ts',
	'src/routes/main/tooltip.ts'
];

const optionalMainRoutePaths = [
	'src/routes/main/MainRouteViewport.svelte',
	'src/routes/main/MainRouteTooltipLayer.svelte',
	'src/routes/main/modelLifecycle.ts'
];
```

If the current test uses a single `candidatePaths` array, split it into required and optional arrays. Required paths must assert existence. Optional paths should only be scanned when present.

Keep the existing forbidden debug-only patterns for both required and optional paths:

```ts
const debugOnlyPatterns = [
	/\.bin/i,
	/\bparity\b/i,
	/Python/i,
	/loadReferenceFromFs/i,
	/__onDemandPrototypeDiagnostics__/i,
	/LEGACY_DEBUG_SELECTED_HOUR_CONTROLLER_ID/i
];
```

- [ ] **Step 2: Add viewport source-lock test**

Create `viewer/tests/routes/main-route-viewport-source-lock.test.ts`:

```ts
import { describe, expect, it } from 'vitest';
import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const repoRoot = resolve(__dirname, '../..');
const viewportPath = 'src/routes/main/MainRouteViewport.svelte';

function readIfPresent(relativePath: string): string | null {
	const absolutePath = resolve(repoRoot, relativePath);
	if (!existsSync(absolutePath)) return null;
	return readFileSync(absolutePath, 'utf8');
}

describe('main route viewport source lock', () => {
	it('keeps the viewport component product-only when it exists', () => {
		const source = readIfPresent(viewportPath);
		if (source == null) return;

		expect(source).toMatch(/<Scene/);
		expect(source).toMatch(/<Model/);
		expect(source).toMatch(/<UTCIPointCloud/);
		expect(source).toMatch(/<ComparisonRenderer/);
		expect(source).not.toMatch(/\.bin/i);
		expect(source).not.toMatch(/\bparity\b/i);
		expect(source).not.toMatch(/Python/i);
		expect(source).not.toMatch(/__onDemandPrototypeDiagnostics__/i);
	});
});
```

- [ ] **Step 3: Run source-lock tests**

Run:

```powershell
cd viewer
npx vitest run tests/routes/main-route-live-selected-hour-source-lock.test.ts tests/routes/main-route-viewport-source-lock.test.ts tests/debug/debug-route-decomposition-source-lock.test.ts
```

Expected:

- PASS.
- The viewport source-lock test is optional until the component exists.
- Future tasks must move each newly created file from optional to required after creation.

## Task 2: Extract Main Route Model Lifecycle Policy

**Files:**
- Create: `viewer/src/routes/main/modelLifecycle.ts`
- Create: `viewer/tests/routes/main-route-model-lifecycle.test.ts`
- Modify later: `viewer/src/routes/+page.svelte`

- [ ] **Step 1: Write model lifecycle tests**

Create `viewer/tests/routes/main-route-model-lifecycle.test.ts`:

```ts
import { describe, expect, it } from 'vitest';
import * as THREE from 'three';

import { getMainRouteModelLoadedEffects } from '../../src/routes/main/modelLifecycle';

function createBounds() {
	return new THREE.Box3(
		new THREE.Vector3(-1, 0, -2),
		new THREE.Vector3(3, 4, 2)
	);
}

describe('main route model lifecycle policy', () => {
	it('computes first camera fit and marks fit as complete', () => {
		const center = new THREE.Vector3(1, 2, 0);
		const size = new THREE.Vector3(4, 4, 4);

		const result = getMainRouteModelLoadedEffects({
			bounds: createBounds(),
			center,
			size,
			hasFitOnce: false
		});

		expect(result.nextHasFitOnce).toBe(true);
		expect(result.sceneBounds).toBeInstanceOf(THREE.Box3);
		expect(result.cameraFit).toBeDefined();
		expect(result.cameraFit?.target.equals(center)).toBe(true);
		expect(result.cameraFit?.position.y).toBeCloseTo(center.y + 4.2);
	});

	it('does not request another first camera fit after one has already happened', () => {
		const result = getMainRouteModelLoadedEffects({
			bounds: createBounds(),
			center: new THREE.Vector3(1, 2, 0),
			size: new THREE.Vector3(4, 4, 4),
			hasFitOnce: true
		});

		expect(result.nextHasFitOnce).toBe(true);
		expect(result.cameraFit).toBeUndefined();
	});
});
```

- [ ] **Step 2: Run the failing test**

Run:

```powershell
cd viewer
npx vitest run tests/routes/main-route-model-lifecycle.test.ts
```

Expected:

- FAIL because `modelLifecycle.ts` does not exist yet.

- [ ] **Step 3: Implement pure model lifecycle helper**

Create `viewer/src/routes/main/modelLifecycle.ts`:

```ts
import * as THREE from 'three';

export type MainRouteModelLoadedEffects = {
	sceneBounds: THREE.Box3;
	cameraFit?: {
		position: THREE.Vector3;
		target: THREE.Vector3;
	};
	nextHasFitOnce: boolean;
};

export function getMainRouteModelLoadedEffects(params: {
	bounds: THREE.Box3;
	center: THREE.Vector3;
	size: THREE.Vector3;
	hasFitOnce: boolean;
}): MainRouteModelLoadedEffects {
	if (params.hasFitOnce) {
		return {
			sceneBounds: params.bounds,
			nextHasFitOnce: true
		};
	}

	const maxDim = Math.max(params.size.x, params.size.y, params.size.z);
	const distance = maxDim * 1.05;
	const position = params.center
		.clone()
		.add(new THREE.Vector3(0, distance, 0.01));

	return {
		sceneBounds: params.bounds,
		cameraFit: {
			position,
			target: params.center.clone()
		},
		nextHasFitOnce: true
	};
}
```

- [ ] **Step 4: Run model lifecycle test**

Run:

```powershell
cd viewer
npx vitest run tests/routes/main-route-model-lifecycle.test.ts
```

Expected:

- PASS, 2 tests.

## Task 3: Extract Main Route Viewport Component

**Files:**
- Create: `viewer/src/routes/main/MainRouteViewport.svelte`
- Modify: `viewer/src/routes/+page.svelte`
- Modify: `viewer/tests/routes/main-route-viewport-source-lock.test.ts`
- Modify: `viewer/tests/routes/main-route-live-selected-hour-source-lock.test.ts`

- [ ] **Step 1: Create viewport component with explicit props and events**

Create `viewer/src/routes/main/MainRouteViewport.svelte`.

The component must:

- Render the existing `Scene`/`Camera`/`Model`/`GridHelper`/`UTCIPointCloud`/`ComparisonRenderer` tree.
- Bind `canvasElement`, `cameraRef`, `utciMesh`, and `comparisonRenderer` back to the route.
- Dispatch `modelLoaded` with the loaded `Group`.
- Dispatch `layersDiscovered` with discovered layers.
- Forward renderer diagnostics, UTCI surface diagnostics, and accepted GPU-resident release callbacks.

Use this shape as the implementation target, adapting imported type names as needed from the current route:

```svelte
<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import Scene from '$lib/components/scene/Scene.svelte';
	import Camera from '$lib/components/scene/Camera.svelte';
	import Lights from '$lib/components/scene/Lights.svelte';
	import GridHelper from '$lib/components/scene/GridHelper.svelte';
	import Model from '$lib/components/scene/Model.svelte';
	import UTCIPointCloud from '$lib/components/scene/UTCIPointCloud.svelte';
	import ComparisonRenderer from '$lib/components/scene/ComparisonRenderer.svelte';
	import { resolveAnalysisModelPath } from '$lib/utils/analysisPaths';
	import type { Analysis } from '$lib/types/analysis';
	import type { LiveSelectedHourControllerSurfaceDiagnostics } from '$lib/compute/liveSelectedHourController';
	import type { LiveSelectedHourPublishedRenderContext } from '$lib/compute/liveSelectedHourRenderContext';
	import type { LiveSelectedHourSurfaceIdentity } from '$lib/compute/liveSelectedHourSurfaceIdentity';
	import type { SelectedHourGpuResidentOutput } from '$lib/compute/liveUtciSelectedHourSession';
	import type {
		WebgpuLargeBufferDeviceLimits,
		WebgpuLargeBufferRequiredLimits
	} from '$lib/compute/webgpuDeviceLimits';
	import type { UtciRendererBackend } from '$lib/utciRenderMode';
	import type { UtciSurfaceBackendType } from '$lib/services/pointCloudService';
	import type { Group, Mesh, PerspectiveCamera } from 'three';
	import type { MainRouteAcceptedGpuResidentOutputReleaseParams } from './liveSelectedHour';

	type RendererDiagnostics = {
		rendererBackend: UtciRendererBackend;
		rendererDevice?: GPUDevice;
		rendererRequiredLimits?: WebgpuLargeBufferRequiredLimits;
		rendererDeviceLimits?: WebgpuLargeBufferDeviceLimits;
		error?: string;
	};

	const dispatch = createEventDispatcher<{
		modelLoaded: Group;
		layersDiscovered: unknown;
	}>();

	export let analysis: Analysis | null;
	export let analysisId: string;
	export let dataBasePath: string;
	export let theme: 'light' | 'dark';
	export let requestLargeWebgpuLimits: boolean;
	export let cameraNear: number;
	export let cameraFar: number;
	export let gridVisible: boolean;
	export let model: Group | null;
	export let isComparing: boolean;
	export let baseSceneAnalysis: Analysis | null;
	export let comparisonSceneAnalysis: Analysis | null | undefined;
	export let basePendingGpuResidentOutput: SelectedHourGpuResidentOutput | null;
	export let comparisonPendingGpuResidentOutput: SelectedHourGpuResidentOutput | null;
	export let baseSceneRenderContext: LiveSelectedHourPublishedRenderContext | null;
	export let comparisonSceneRenderContext: LiveSelectedHourPublishedRenderContext | null | undefined;
	export let baseSceneSurfaceIdentity: LiveSelectedHourSurfaceIdentity | null;
	export let comparisonSceneSurfaceIdentity: LiveSelectedHourSurfaceIdentity | null | undefined;
	export let basePendingRenderUpdateStartedAt: number | undefined;
	export let comparisonPendingRenderUpdateStartedAt: number | undefined;
	export let resolvedUtciSurfaceBackend: UtciSurfaceBackendType;
	export let onRendererDiagnostics: (diagnostics: RendererDiagnostics) => void;
	export let onBaseUtciSurfaceDiagnostics: (
		diagnostics: LiveSelectedHourControllerSurfaceDiagnostics
	) => void;
	export let onComparisonUtciSurfaceDiagnostics: (
		diagnostics: LiveSelectedHourControllerSurfaceDiagnostics
	) => void;
	export let onBaseAcceptedGpuResidentOutputRelease: (
		params: MainRouteAcceptedGpuResidentOutputReleaseParams
	) => void;
	export let onComparisonAcceptedGpuResidentOutputRelease: (
		params: MainRouteAcceptedGpuResidentOutputReleaseParams
	) => void;

	export let canvasElement: HTMLCanvasElement | null = null;
	export let cameraRef: PerspectiveCamera | undefined = undefined;
	export let utciMesh: Mesh | null = null;
	export let comparisonRenderer: ComparisonRenderer;

	$: modelPath =
		analysis == null
			? null
			: resolveAnalysisModelPath(analysis.metadata, analysisId).replace(
					'data/',
					`${dataBasePath}/data/`
				);
</script>

{#key requestLargeWebgpuLimits}
	<Scene
		backgroundColor={theme === 'light' ? 0x4b5563 : 0x111827}
		bind:canvasElement
		onRendererDiagnostics={onRendererDiagnostics}
		{requestLargeWebgpuLimits}
	>
		<Camera bind:cameraRef near={cameraNear} far={cameraFar} />
		<Lights />

		{#if analysis && modelPath}
			{#key analysis.metadata.model_file}
				<Model
					{modelPath}
					coordinateSystem={analysis.metadata.coordinate_system || 'xy_ground'}
					metadata={analysis.metadata}
					on:modelLoaded={(event) => dispatch('modelLoaded', event.detail)}
					on:layersDiscovered={(event) => dispatch('layersDiscovered', event.detail)}
				/>
			{/key}

			{#if model}
				<GridHelper {model} visible={gridVisible} />
				<UTCIPointCloud
					analysis={baseSceneAnalysis}
					{model}
					bind:utciSurface={utciMesh}
					acceptedGpuResidentOutput={basePendingGpuResidentOutput}
					selectedHourRenderContext={baseSceneRenderContext}
					liveSelectedHourSurfaceIdentity={baseSceneSurfaceIdentity}
					onUtciSurfaceDiagnostics={onBaseUtciSurfaceDiagnostics}
					onAcceptedGpuResidentOutputRelease={onBaseAcceptedGpuResidentOutputRelease}
					pendingRenderUpdateStartedAt={basePendingRenderUpdateStartedAt}
					utciSurfaceBackend={resolvedUtciSurfaceBackend}
				/>
			{/if}

			{#if isComparing}
				<ComparisonRenderer
					bind:this={comparisonRenderer}
					acceptedGpuResidentOutput={comparisonPendingGpuResidentOutput}
					baseCamera={cameraRef}
					displayAnalysis={comparisonSceneAnalysis}
					selectedHourRenderContext={comparisonSceneRenderContext}
					liveSelectedHourSurfaceIdentity={comparisonSceneSurfaceIdentity}
					onUtciSurfaceDiagnostics={onComparisonUtciSurfaceDiagnostics}
					onAcceptedGpuResidentOutputRelease={onComparisonAcceptedGpuResidentOutputRelease}
					pendingRenderUpdateStartedAt={comparisonPendingRenderUpdateStartedAt}
					utciSurfaceBackend={resolvedUtciSurfaceBackend}
				/>
			{/if}
		{/if}
	</Scene>
{/key}
```

If the exact `Camera` prop syntax differs, follow the current route markup. Do not change scene behavior.

- [ ] **Step 2: Wire viewport component in the route**

In `viewer/src/routes/+page.svelte`:

- Remove direct imports for scene components moved into `MainRouteViewport.svelte`.
- Import `MainRouteViewport` and `getMainRouteModelLoadedEffects`.
- Replace the viewport slot body with:

```svelte
<MainRouteViewport
	bind:canvasElement
	bind:cameraRef
	bind:utciMesh
	bind:comparisonRenderer
	analysis={$analysisStore}
	{analysisId}
	dataBasePath={getDataBasePath()}
	theme={$viewerStore.theme}
	{requestLargeWebgpuLimits}
	cameraNear={$sceneConfigStore.cameraNear}
	cameraFar={$sceneConfigStore.cameraFar}
	{gridVisible}
	{model}
	{isComparing}
	{baseSceneAnalysis}
	{comparisonSceneAnalysis}
	{basePendingGpuResidentOutput}
	{comparisonPendingGpuResidentOutput}
	{baseSceneRenderContext}
	{comparisonSceneRenderContext}
	{baseSceneSurfaceIdentity}
	{comparisonSceneSurfaceIdentity}
	{basePendingRenderUpdateStartedAt}
	{comparisonPendingRenderUpdateStartedAt}
	{resolvedUtciSurfaceBackend}
	onRendererDiagnostics={handleRendererDiagnostics}
	onBaseUtciSurfaceDiagnostics={handleUtciSurfaceDiagnostics}
	onComparisonUtciSurfaceDiagnostics={handleComparisonUtciSurfaceDiagnostics}
	onBaseAcceptedGpuResidentOutputRelease={handleBaseAcceptedGpuResidentOutputRelease}
	onComparisonAcceptedGpuResidentOutputRelease={handleComparisonAcceptedGpuResidentOutputRelease}
	on:modelLoaded={handleMainRouteModelLoaded}
	on:layersDiscovered={(event) => setDiscoveredLayers(event.detail)}
/>
```

Add this route handler:

```ts
function handleMainRouteModelLoaded(event: CustomEvent<Group>): void {
	model = event.detail;
	modelLoading = false;
	if (!model) return;

	const bounds = calculateModelBounds(model);
	const center = calculateModelCenter(model);
	const size = calculateModelSize(model);
	const effects = getMainRouteModelLoadedEffects({
		bounds,
		center,
		size,
		hasFitOnce,
	});

	updateSceneConfigFromBounds(effects.sceneBounds);
	if (effects.cameraFit) {
		cameraStore.update((state) => ({
			...state,
			position: effects.cameraFit!.position,
			target: effects.cameraFit!.target,
		}));
	}
	hasFitOnce = effects.nextHasFitOnce;
}
```

After `MainRouteViewport.svelte` and `modelLifecycle.ts` exist, move both paths from `optionalMainRoutePaths` to `requiredMainRoutePaths` in `viewer/tests/routes/main-route-live-selected-hour-source-lock.test.ts`.

- [ ] **Step 3: Run Svelte autofixer on edited Svelte files**

Run:

```powershell
cd viewer
npx @sveltejs/mcp svelte-autofixer src/routes/+page.svelte --svelte-version 5
npx @sveltejs/mcp svelte-autofixer src/routes/main/MainRouteViewport.svelte --svelte-version 5
```

Expected:

- No new blocking issues.
- Do not migrate to runes.

- [ ] **Step 4: Run focused route tests**

Run:

```powershell
cd viewer
npx vitest run tests/routes/main-route-model-lifecycle.test.ts tests/routes/main-route-viewport-source-lock.test.ts tests/routes/main-route-live-selected-hour-source-lock.test.ts tests/routes/main-route-live-selected-hour.test.ts tests/diagnostics/main-route-utci-diagnostics.test.ts
```

Expected:

- PASS.
- Main route source-lock still forbids debug-only behavior.

- [ ] **Step 5: Run browser route proof**

Run:

```powershell
cd viewer
npm run test:e2e:selected-hour
```

Expected:

- PASS with 13 Chromium tests.
- If it fails or hangs, stop and use `superpowers:systematic-debugging`.

## Task 4: Extract Main Route Tooltip Layer

**Files:**
- Create: `viewer/src/routes/main/MainRouteTooltipLayer.svelte`
- Modify: `viewer/src/routes/+page.svelte`
- Modify: `viewer/src/routes/main/tooltip.ts` only if required
- Modify: `viewer/tests/routes/main-route-tooltip.test.ts`
- Modify: `viewer/tests/routes/main-route-live-selected-hour-source-lock.test.ts`

- [ ] **Step 1: Add helper and source tests for event-time comparison mesh lookup**

Extend `viewer/tests/routes/main-route-tooltip.test.ts` with:

```ts
it('uses event-time comparison mesh when the cursor is past the curtain', () => {
	const baseMesh = { id: 'base' };
	const comparisonMesh = { id: 'comparison' };
	const baseAnalysis = { id: 'base-analysis' };
	const comparisonAnalysis = { id: 'comparison-analysis' };

	const target = resolveMainRouteTooltipTarget({
		baseMesh,
		baseAnalysis,
		baseSceneTimeIndex: 180,
		comparisonMesh,
		comparisonAnalysis,
		comparisonSceneTimeIndex: 181,
		useLiveUtciOnMainRoute: true,
		isComparing: true,
		mouseClientX: 75,
		mainViewportRect: { left: 0, width: 100 },
		curtainPosition: 0.5,
		viewerCurrentHour: 12
	});

	expect(target.meshToRaycast).toBe(comparisonMesh);
	expect(target.analysisToUse).toBe(comparisonAnalysis);
expect(target.tooltipHourIndex).toBe(181);
});
```

Create `viewer/tests/routes/main-route-tooltip-layer-source-lock.test.ts` so the component cannot accidentally capture comparison mesh state outside the pointer handler:

```ts
import { describe, expect, it } from 'vitest';
import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const repoRoot = resolve(__dirname, '../..');
const tooltipLayerPath = resolve(
	repoRoot,
	'src/routes/main/MainRouteTooltipLayer.svelte'
);

describe('main route tooltip layer source lock', () => {
	it('reads the comparison mesh through the event-time getter inside pointer handling', () => {
		if (!existsSync(tooltipLayerPath)) return;

		const source = readFileSync(tooltipLayerPath, 'utf8');
		expect(source).toMatch(/export let getComparisonUtciMesh/);
		expect(source).toMatch(/function handleMouseMove/);
		const handlerSource = source.slice(source.indexOf('function handleMouseMove'));
		expect(handlerSource).toMatch(/getComparisonUtciMesh\(\)/);
	});
});
```

- [ ] **Step 2: Create tooltip layer component**

Create `viewer/src/routes/main/MainRouteTooltipLayer.svelte`.

The component must own:

- `tooltipVisible`, `tooltipX`, `tooltipY`, `tooltipValue`, `tooltipPosition`.
- `lastTooltipUpdate`.
- `tooltipMotionSuppression`.
- hover listener and motion listener attach/detach.
- `<MetricTooltip>` rendering.

It must receive explicit props:

```ts
export let canvasElement: HTMLCanvasElement | null;
export let cameraRef: PerspectiveCamera | undefined;
export let baseMesh: Mesh | null;
export let baseDisplayedAnalysis: Analysis | null;
export let baseSceneTimeIndex: number | undefined;
export let comparisonDisplayedAnalysis: Analysis | null | undefined;
export let comparisonSceneTimeIndex: number | undefined;
export let getComparisonUtciMesh: () => Mesh | null;
export let useLiveUtciOnMainRoute: boolean;
export let isComparing: boolean;
export let mainViewportElement: HTMLElement | null;
export let curtainPosition: number;
export let viewerCurrentHour: number;
export let metricType: string;
export let utciVisible: boolean;
export let tooltipHoverSampleCount = 0;
export let cameraWheelEventCount = 0;
```

Keep the `getTooltipData(...)` call in the component so tooltip state no longer lives in the route.

Inside `handleMouseMove`, pass `comparisonMesh: getComparisonUtciMesh()` directly into `resolveMainRouteTooltipTarget(...)`. Do not assign the getter result to a reactive variable outside the handler.

- [ ] **Step 3: Replace route tooltip state and listeners**

In `viewer/src/routes/+page.svelte`:

- Remove tooltip local state variables and listener attach/detach functions.
- Remove direct `MetricTooltip`, `getTooltipData`, and tooltip motion suppression imports.
- Keep the telemetry variables only as bindable props:

```ts
let tooltipHoverSampleCount = 0;
let cameraWheelEventCount = 0;
```

In the `tooltip` slot, render:

```svelte
<MainRouteTooltipLayer
	bind:tooltipHoverSampleCount
	bind:cameraWheelEventCount
	{canvasElement}
	{cameraRef}
	baseMesh={utciMesh}
	{baseDisplayedAnalysis}
	baseSceneTimeIndex={baseSceneRenderContext?.timeIndex}
	comparisonDisplayedAnalysis={useLiveUtciOnMainRoute
		? comparisonRendererDisplayAnalysis
		: $comparisonAnalysis}
	comparisonSceneTimeIndex={comparisonSceneRenderContext?.timeIndex}
	getComparisonUtciMesh={() => comparisonRenderer?.getComparisonUtciMesh() ?? null}
	{useLiveUtciOnMainRoute}
	{isComparing}
	{mainViewportElement}
	curtainPosition={$comparisonStore.curtainPosition}
	viewerCurrentHour={$viewerStore.currentHour}
	metricType={$viewerStore.metricType}
	utciVisible={$viewerStore.utciVisible}
/>
```

- [ ] **Step 4: Run Svelte autofixer**

Run:

```powershell
cd viewer
npx @sveltejs/mcp svelte-autofixer src/routes/+page.svelte --svelte-version 5
npx @sveltejs/mcp svelte-autofixer src/routes/main/MainRouteTooltipLayer.svelte --svelte-version 5
```

Expected:

- No new blocking issues.
- Do not migrate to runes.

- [ ] **Step 5: Run focused tooltip and route tests**

Run:

```powershell
cd viewer
	npx vitest run tests/routes/main-route-tooltip.test.ts tests/components/canvasInteractionController.test.ts tests/services/tooltipService.test.ts tests/services/tooltipMotionSuppression.test.ts tests/routes/main-route-live-selected-hour-source-lock.test.ts
```

Expected:

- PASS.

Also run the tooltip layer source lock after the component exists:

```powershell
cd viewer
npx vitest run tests/routes/main-route-tooltip-layer-source-lock.test.ts
```

Expected:

- PASS.

After `MainRouteTooltipLayer.svelte` exists, move it from `optionalMainRoutePaths` to `requiredMainRoutePaths` in `viewer/tests/routes/main-route-live-selected-hour-source-lock.test.ts`.

- [ ] **Step 6: Run main route E2E**

Run:

```powershell
cd viewer
npx playwright test tests/e2e/main-route-manual-diagnostics.spec.ts --project=chromium --workers=1 --reporter=list --timeout=30000
```

Expected:

- PASS.
- Main route still publishes diagnostics, ignores debug parity query params, updates selected hour/month surfaces, keeps strong visible GPU path honest, and uses live WebGPU range for Ness Tziona.

## Task 5: Optional Overlay Component If Route Is Still Too Bulky

**Files:**
- Create only if needed: `viewer/src/routes/main/MainRouteOverlays.svelte`
- Modify only if needed: `viewer/src/routes/+page.svelte`
- Existing test: `viewer/tests/routes/main-route-overlay-gating-helper.test.ts`

- [ ] **Step 1: Recount main route after Tasks 3 and 4**

Run:

```powershell
(Get-Content viewer/src/routes/+page.svelte).Count
```

Expected:

- If the route is under 600 lines and reads clearly, skip this task.
- If overlays still obscure the `ViewerShell` structure, continue.

- [ ] **Step 2: Extract dumb overlay markup**

Create `viewer/src/routes/main/MainRouteOverlays.svelte` with props only:

```ts
export let loading: boolean;
export let error: string | null;
export let baseLiveError: string | undefined;
export let comparisonLiveError: string | undefined;
export let showMainRouteOverlay: boolean;
export let showMainRouteComparisonOverlay: boolean;
export let curtainPosition: number;
export let modelLoading: boolean;
export let comparisonModelLoading: boolean;
export let useLiveUtciOnMainRoute: boolean;
export let isComparing: boolean;
export let mainViewportElement: HTMLElement | null;
export let comparisonScenarioName: string;
```

Move only overlay markup and `ComparisonCurtain` rendering. Keep `getMainRouteOverlayGating(...)` in `+page.svelte`.

- [ ] **Step 3: Run overlay tests**

Run:

```powershell
cd viewer
npx vitest run tests/routes/main-route-overlay-gating-helper.test.ts tests/routes/main-route-live-selected-hour-source-lock.test.ts
```

Expected:

- PASS.

## Task 6: Full Main Route Verification And Review Agents

**Files:**
- Inspect changed files.
- Create: `docs/superpowers/plans/2026-05-12-main-route-product-shell-cleanup-results.md`

- [ ] **Step 1: Run final route size check**

Run:

```powershell
(Get-Content viewer/src/routes/+page.svelte).Count
(Get-Content viewer/src/routes/debug/+page.svelte).Count
```

Expected:

- Main route should be visibly thinner, target 450-600 lines.
- Debug route size is not a success metric for this plan.

- [ ] **Step 2: Run full verification**

Run:

```powershell
cd viewer
npx vitest run tests/routes/main-route-model-lifecycle.test.ts tests/routes/main-route-viewport-source-lock.test.ts tests/routes/main-route-model-selection.test.ts tests/routes/main-route-tooltip.test.ts tests/routes/main-route-overlay-gating-helper.test.ts tests/routes/main-route-live-selected-hour.test.ts tests/routes/main-route-live-selected-hour-source-lock.test.ts tests/routes/main-route-debug-boundary-source-lock.test.ts tests/debug/debug-route-decomposition-source-lock.test.ts tests/diagnostics/main-route-utci-diagnostics.test.ts
npx vitest run tests/routes/main-route-tooltip-layer-source-lock.test.ts
npm run test:quality:selected-hour
npm run test:e2e:selected-hour
npm run build
npm run check
cd ..
git diff --check
```

Expected:

- Focused Vitest passes.
- Selected-hour quality passes.
- Selected-hour E2E passes with 13 Chromium tests.
- Build passes.
- `npm run check` may still fail from inherited repo-wide debt; classify touched-file errors separately.
- `git diff --check` passes or prints only LF/CRLF warnings.

- [ ] **Step 3: Request read-only review agents**

Use at least three read-only review agents. They must not edit files.

Main route Svelte reviewer prompt:

```text
Review the main product route cleanup in D:\Projects\Nur\Shade\fast-utci. Do not edit files. Focus on viewer/src/routes/+page.svelte and viewer/src/routes/main/*. Check whether the route is visibly thinner without hiding Svelte reactive dependencies, whether extracted Svelte components have clear prop/event boundaries, and whether product route readability improved without creating tiny indirection. Return findings first with file/line evidence.
```

Selected-hour behavior reviewer prompt:

```text
Review selected-hour behavior preservation after the main route cleanup in D:\Projects\Nur\Shade\fast-utci. Do not edit files. Focus on selected-hour diagnostics, route-host inputs, scene props, accepted GPU-resident output release callbacks, strongVisibleGpuPath honesty, and main-route debug-only source locks. Return findings first with file/line evidence.
```

Static-debt boundary reviewer prompt:

```text
Review static-debt and scope boundaries for the main route cleanup in D:\Projects\Nur\Shade\fast-utci. Do not edit files. Confirm the plan did not broaden into debug beautification, compute folder reorganization, or repo-wide npm check cleanup. Classify npm run check touched-file diagnostics separately from inherited debt. Return findings first with file/line evidence.
```

- [ ] **Step 4: Fix review blockers only**

If review agents find blockers:

- Reproduce with a focused test or line-level inspection.
- Fix only the blocker.
- Rerun the focused verification and any affected final gates.

Do not expand into debug route cleanup, compute reorganization, or broad static debt.

- [ ] **Step 5: Write results note**

Create `docs/superpowers/plans/2026-05-12-main-route-product-shell-cleanup-results.md`:

```md
# Main Route Product Shell Cleanup Results

Date: 2026-05-12

## Scope

Summarize product-route-focused cleanup and explicit deferrals.

## Main Route Size

Record before/after line counts for `viewer/src/routes/+page.svelte`.

## Extracted Boundaries

Summarize `MainRouteViewport.svelte`, `modelLifecycle.ts`, `MainRouteTooltipLayer.svelte`, and optional overlays if used.

## Svelte Reactivity Notes

Record how selected-hour host inputs and diagnostics kept explicit reactive dependencies.

## Verification

Record command results with PASS/FAIL and counts.

## Review Agents

Record findings and fixes.

## Remaining Work

List deferred compute folder reorganization, debug-route temporary cleanup, and inherited npm check debt.
```

## Final Completion Criteria

Completion can be claimed only when:

- Main route is visibly thinner and still readable.
- Debug route was not beautified beyond source-lock/test maintenance.
- No compute folder reorganization occurred.
- Main route source locks pass and still forbid debug-only behavior.
- Selected-hour quality tests pass.
- Selected-hour E2E passes.
- Build passes.
- `git diff --check` passes.
- `npm run check` touched-file errors are fixed or explicitly documented as preexisting/inherited.
- Read-only review agents report no unresolved blockers.

## Execution Handoff

Use this plan with:

- `superpowers:subagent-driven-development`
- `superpowers:verification-before-completion`
- `superpowers:systematic-debugging` for any test/browser/route failure
- `svelte-code-writer` for every `.svelte` edit

Execution constraints:

- Work in `D:\Projects\Nur\Shade\fast-utci`.
- Do not create commits.
- Do not create git worktrees.
- Preserve unrelated dirty files.
- Use focused worker subagents task-by-task and review agents before completion.
- Stop and report findings first if route behavior, selected-hour diagnostics, or browser verification fails.
