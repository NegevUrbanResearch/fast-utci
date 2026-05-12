# NPM Check Debt And Selected-Hour Boundary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Workflow overrides:** Do **not** create commits. Do **not** create git worktrees. Preserve unrelated dirty files. Treat generated Playwright state such as `viewer/test-results/.last-run.json` as generated state; clean it before final status. Use subagent reviews at the required gates before moving to the next task group.

**Goal:** Make `cd viewer; npm run check` pass while removing the selected-hour output-handle boundary inversion introduced by the compute-folder organization.

**Architecture:** First move the generic selected-hour GPU output handle out of `compute/selected-hour/` so `compute/gpu/` no longer imports upward into a feature bucket. Then reduce `svelte-check` debt by error family: recent selected-hour/compute contract drift, ArrayBuffer transfer typing, parity reference narrowing, UTCI data union narrowing, Three object guards, Svelte component typing, and stale test fixtures. Keep behavior changes out of scope unless an error exposes a real type-contract bug that must be fixed at the source.

**Tech Stack:** SvelteKit/Svelte 5, TypeScript 5.9, Vitest, Playwright Chromium/WebGPU, Three/Threlte, WebGPU/WGSL, PowerShell on Windows.

---

## Current Evidence

- Current branch had no source/test changes before this planning pass; during review, this plan file itself may appear as an intentional untracked/modified doc change.
- Recent history includes `44bf534 refactor(webgpu): organize compute modules by responsibility`.
- `cd viewer; npm run build` passes with existing warning profile.
- `cd viewer; npm run check` fails with `160 errors and 4 warnings in 33 files`.
- The strongest current debt clusters are:
  - `viewer/src/lib/compute/gpu/mergeAndBvh.worker.ts`: `ArrayBufferLike` transfer-list typing.
  - `viewer/src/lib/parity/buildParityReport.ts`, `viewer/scripts/diagnose-solar-flips.ts`, `viewer/tests/parity/loadReferenceIntermediatesFromFs.test.ts`: `SolarReference | SkyReference | MrtReference` union narrowing.
  - `viewer/src/lib/services/validationService.ts`, `viewer/src/lib/components/ui/AnalyticsPanel.svelte`, `viewer/tests/services/dataLoader.test.ts`: `UTCIData` union narrowing.
  - `viewer/src/lib/services/modelLoaderService.ts`, `viewer/src/lib/services/layerManagerService.ts`, `viewer/src/lib/services/modelCacheService.ts`, related tests: Three `Object3D` type guard debt.
  - `viewer/src/lib/components/scene/Model.svelte`, `viewer/src/lib/components/scene/SunPath.svelte`, `viewer/src/lib/components/scene/UTCIPointCloud.svelte`, `viewer/src/lib/components/ui/ThemeToggle.svelte`: Svelte/Threlte strictness debt.
- `viewer/tests/services/pointCloudService.surface.test.ts`: stale `UtciGridLayout` fixture shape.
- Smaller current check errors also need explicit coverage:
  - `viewer/src/lib/parity/loadReferenceFromFs.ts`: `Buffer.buffer.slice(...)` returns `ArrayBuffer | SharedArrayBuffer`.
  - `viewer/src/lib/services/lruCache.ts`: `Map.keys().next().value` can be `undefined`.
  - `viewer/src/lib/stores/layerStore.ts`: `newVisible` can be read before assignment.
  - `viewer/tests/parity/loadWebgpuCollectedFromFs.test.ts`: optional value used as the left side of `instanceof`.
- `selectedHourOutputHandle.ts` is a generic GPU output buffer handle, but it currently lives in `viewer/src/lib/compute/selected-hour/` and is imported by `viewer/src/lib/compute/gpu/gpu-pipeline.ts` and `viewer/src/lib/compute/gpu/webgpuUtciPipeline.ts`. This creates a feature-bucket back-edge from GPU code into selected-hour orchestration.

## Hard Constraints

- Do not create commits.
- Do not create git worktrees.
- Do not weaken or delete these proof/fallback surfaces:
  - `runAll()`
  - `.bin`
  - Python comparison/reference paths
  - `readUtciBulk()`
  - `readUtcisSlice()`
  - `dataTexture`
  - debug parity, collect, and legacy selected-hour paths
- Do not make main route or debug route behavior depend on `.bin` metadata unless that path already explicitly does so for parity/reference comparison.
- Do not weaken `strongVisibleGpuPath`, same-device, zero visible readback, or selected-hour runtime contract semantics.
- Do not broaden scope into WebGPU performance, Ness Tziona flake hunting, feature work, bundle-size cleanup, eslint setup, or repo-wide architectural reorganization.
- Do not move `viewer/src/lib/compute/compute-manager.ts` or `viewer/src/lib/compute/telemetry.ts` in this plan.
- If a check error requires nontrivial behavior change, stop and report the root-cause finding before applying the behavior change.

## File Structure Target

### Create

- `viewer/src/lib/compute/gpu/selectedHourOutputHandle.ts`
  - Owns the generic GPU output handle type and disposal helper.
  - This is a neutral GPU contract because the handle wraps a `GPUBuffer` and is used by `gpu-pipeline.ts`/`webgpuUtciPipeline.ts`.

### Delete / Move

- Move from `viewer/src/lib/compute/selected-hour/selectedHourOutputHandle.ts`
  to `viewer/src/lib/compute/gpu/selectedHourOutputHandle.ts`.

### Modify

- `viewer/src/lib/compute/gpu/gpu-pipeline.ts`
  - Import `SelectedHourOutputHandle` from `compute/gpu/selectedHourOutputHandle`.
- `viewer/src/lib/compute/gpu/webgpuUtciPipeline.ts`
  - Import `createSelectedHourOutputHandle` from `compute/gpu/selectedHourOutputHandle`.
- `viewer/src/lib/compute/selected-hour/liveUtciSelectedHourSession.ts`
  - Import output handle helpers/types from `compute/gpu/selectedHourOutputHandle`.
- Any tests or source-lock files that reference the old selected-hour output-handle path.
- `viewer/src/lib/parity/loadReferenceIntermediatesFromFs.ts`
  - Add stage-specific overloads so callers get narrowed return types.
- `viewer/src/lib/parity/buildParityReport.ts`
  - Use narrowed return types and preserve existing report math.
- `viewer/scripts/compare-parity.ts`
  - Use narrowed return types if `npm run check` reports parity union errors here.
- `viewer/scripts/diagnose-mrt-worst-cell.ts`
  - Use narrowed return types if `npm run check` reports parity union errors here.
- `viewer/scripts/diagnose-solar-flips.ts`
  - Use narrowed return types and explicit optional-component checks.
- `viewer/tests/parity/loadReferenceIntermediatesFromFs.test.ts`
  - Use narrowed return types through overloads; no `as` cast should be needed for stage-specific calls.
- `viewer/src/lib/types/analysis.ts`
  - Add small exported type guards for `SingleHourData` and full-day storage shape.
- `viewer/src/lib/services/dataLoader.ts`
  - Replace unsafe non-null full-day access with the new type guards or explicit branch checks.
- `viewer/src/lib/services/validationService.ts`
  - Narrow `UTCIData` before accessing `utciValues` or `utciByHour`.
- `viewer/src/lib/components/ui/AnalyticsPanel.svelte`
  - Narrow `analysis.data` before accessing `utciStorage` or `utciByHour`.
- `viewer/tests/services/dataLoader.test.ts`
  - Narrow parsed data before single-hour/full-day assertions.
- `viewer/src/lib/services/modelLoaderService.ts`
  - Replace `child.isMesh`, `child.isLine`, and `child.isLineSegments` direct reads with `instanceof THREE.Mesh`, `instanceof THREE.Line`, and `instanceof THREE.LineSegments`.
- `viewer/src/lib/services/layerManagerService.ts`
  - Replace `child.isMesh` / `child.isLineSegments` direct reads with `instanceof` checks.
- `viewer/src/lib/services/modelCacheService.ts`
  - Use a typed material texture disposal helper rather than property-in-object checks that narrow to `{}`.
- `viewer/tests/services/modelLoaderService.unknownLayers.test.ts`
  - Replace test-side `child.isMesh` direct reads with `child instanceof THREE.Mesh`.
- `viewer/src/lib/components/scene/Model.svelte`
  - Remove unused `@ts-expect-error` and make `oncreate` callback return `void`.
- `viewer/src/lib/components/scene/SunPath.svelte`
  - Fix only non-behavioral Threlte typing issues. If the missing `sunPathVisible` store field requires behavior/store decisions, stop and report instead of adding state or removing the component inside this plan.
  - Convert `Vector3` positions to tuple positions for Threlte.
- `viewer/src/lib/components/scene/UTCIPointCloud.svelte`
  - Fix or remove the unused exported `model` prop only if `svelte-check` still reports it after other errors.
- `viewer/src/lib/components/ui/ThemeToggle.svelte`
  - Replace self-closing non-void `<span />` with `<span></span>`.
- `viewer/src/routes/debug/+page.svelte`
  - Convert parity intermediate `Float32Array | null` assignments to the declared `number[] | undefined` shape.
- `viewer/tests/services/pointCloudService.surface.test.ts`
  - Update `UtciGridLayout` fixtures with `coordinateSystem`, `minY`, and `maxY`.
- `viewer/src/lib/parity/loadReferenceFromFs.ts`
  - Convert Node `Buffer` data to an owned `ArrayBuffer` before parsing.
- `viewer/src/lib/services/lruCache.ts`
  - Guard empty iterator results before eviction.
- `viewer/src/lib/stores/layerStore.ts`
  - Initialize `newVisible` safely before it is used after store update.
- `viewer/tests/parity/loadWebgpuCollectedFromFs.test.ts`
  - Narrow optional collected values before `instanceof`.
- Any selected-hour test files still reported by `npm run check`, especially:
  - `viewer/tests/compute/live-selected-hour-route-host.test.ts`
  - `viewer/tests/compute/live-selected-hour-route-projection.test.ts`
  - `viewer/tests/compute/live-selected-hour-controller.test.ts`
  - `viewer/tests/compute/live-selected-hour-time-index.test.ts`
  - `viewer/tests/compute/live-utci-analysis.test.ts`
  - `viewer/tests/compute/webgpuUtciPipeline.behavior.test.ts`

## Verification Commands

Run from repo root unless command starts with `cd viewer`.

```powershell
git status --short
```

Expected final output: no source changes except the intended plan/execution diff. Generated `viewer/test-results/.last-run.json` must not remain modified.

```powershell
cd viewer; npm run check
```

Expected final output: `svelte-check found 0 errors and 0 warnings`.

```powershell
cd viewer; npm run test:quality:selected-hour
```

Expected final output: PASS, preserving the selected-hour quality suite.

```powershell
cd viewer; npm run test:e2e:selected-hour
```

Expected final output: PASS, preserving main/debug selected-hour runtime proof.

```powershell
cd viewer; npm run build
```

Expected final output: PASS. Existing large-chunk warnings are acceptable unless they become errors.

```powershell
git diff --check
```

Expected final output: no whitespace errors.

```powershell
cd viewer; npx vitest run tests/parity/loadReferenceFromFs.test.ts tests/parity/loadReferenceIntermediatesFromFs.test.ts tests/parity/loadWebgpuCollectedFromFs.test.ts tests/parity/buildParityReport.test.ts --reporter=dot
```

Expected final output: PASS, preserving `.bin`, Python/reference, and parity artifact loading/report behavior.

```powershell
cd viewer; npx vitest run tests/compute/webgpu-on-demand-source-locks.test.ts tests/debug/debug-webgpu-utci-diagnostics.test.ts tests/debug/debug-selected-hour-mode.test.ts tests/services/pointCloudService.surface.test.ts tests/diagnostics/selectedHourRuntimeContract.test.ts --reporter=dot
```

Expected final output: PASS, preserving `readUtcisSlice()`, `readUtciBulk()`, debug parity mode boundaries, `dataTexture` fallback, and selected-hour runtime contract semantics.

---

## Task 0: Baseline, Error Inventory, And Scope Lock

**Files:**
- Inspect only.

- [ ] **Step 1: Confirm current branch and clean status**

Run:

```powershell
git status --short
git log --oneline -8
```

Expected:

- `git status --short` has no output except this plan file if it is not yet staged, or unrelated user changes that are documented and preserved.
- Recent history includes `44bf534 refactor(webgpu): organize compute modules by responsibility`.

- [ ] **Step 2: Capture the current check failure inventory**

Run:

```powershell
cd viewer
npm run check
```

Expected before this plan is implemented:

- FAIL.
- Approximately `160 errors and 4 warnings in 33 files`.
- First error cluster begins in `src/lib/compute/gpu/mergeAndBvh.worker.ts`.

- [ ] **Step 3: Save a temporary local check log for implementation use**

Run:

```powershell
cd viewer
npm run check *> ..\.npm-check-before.txt
```

Expected:

- Command exits nonzero.
- Repo root contains `.npm-check-before.txt`.
- This file is temporary and must be removed before final status.

- [ ] **Step 4: Scope-lock what this plan will not fix**

Before editing, write this note in the execution status update:

```text
This pass fixes npm-check static debt and the selected-hour output-handle boundary only. It does not investigate the historical one-off Ness Tziona timeout, does not move compute-manager.ts or telemetry.ts, and does not change selected-hour proof semantics.
```

---

## Task 1: Move The Selected-Hour Output Handle To The GPU Bucket

**Files:**
- Move: `viewer/src/lib/compute/selected-hour/selectedHourOutputHandle.ts` -> `viewer/src/lib/compute/gpu/selectedHourOutputHandle.ts`
- Modify: `viewer/src/lib/compute/gpu/gpu-pipeline.ts`
- Modify: `viewer/src/lib/compute/gpu/webgpuUtciPipeline.ts`
- Modify: `viewer/src/lib/compute/selected-hour/liveUtciSelectedHourSession.ts`
- Modify: tests/source-lock files reported by `rg selectedHourOutputHandle viewer`

- [ ] **Step 1: Find all old output-handle imports**

Run:

```powershell
rg -n "selectedHourOutputHandle" viewer/src viewer/tests
```

Expected:

- At minimum, references in `gpu-pipeline.ts`, `webgpuUtciPipeline.ts`, and selected-hour session/controller tests.

- [ ] **Step 2: Move the file without changing its contents**

Use a normal file move or `git mv` if available:

```powershell
Move-Item -LiteralPath "viewer\src\lib\compute\selected-hour\selectedHourOutputHandle.ts" -Destination "viewer\src\lib\compute\gpu\selectedHourOutputHandle.ts"
```

Expected:

- Old path no longer exists.
- New path exists.
- File content is byte-for-byte identical except line endings if Git normalizes them.
- Do not rename `SelectedHourOutputHandle`, `SelectedHourOutputSource`, `createSelectedHourOutputHandle`, or `disposeSelectedHourOutputHandle` in this plan. This is a path-only boundary neutralization.

- [ ] **Step 3: Update GPU pipeline import**

In `viewer/src/lib/compute/gpu/gpu-pipeline.ts`, change:

```ts
import type { SelectedHourOutputHandle } from '$lib/compute/selected-hour/selectedHourOutputHandle';
```

to:

```ts
import type { SelectedHourOutputHandle } from '$lib/compute/gpu/selectedHourOutputHandle';
```

- [ ] **Step 4: Update WebGPU implementation import**

In `viewer/src/lib/compute/gpu/webgpuUtciPipeline.ts`, change any import from:

```ts
} from '$lib/compute/selected-hour/selectedHourOutputHandle';
```

to:

```ts
} from '$lib/compute/gpu/selectedHourOutputHandle';
```

- [ ] **Step 5: Update selected-hour session imports**

In `viewer/src/lib/compute/selected-hour/liveUtciSelectedHourSession.ts`, change any import from:

```ts
} from '$lib/compute/selected-hour/selectedHourOutputHandle';
```

to:

```ts
} from '$lib/compute/gpu/selectedHourOutputHandle';
```

- [ ] **Step 6: Update tests and source-lock references**

Run:

```powershell
rg -n "compute/selected-hour/selectedHourOutputHandle|selected-hour\\selectedHourOutputHandle|selectedHourOutputHandle" viewer/src viewer/tests
```

Expected:

- No imports from `$lib/compute/selected-hour/selectedHourOutputHandle`.
- Remaining imports use `$lib/compute/gpu/selectedHourOutputHandle`.

- [ ] **Step 7: Run the narrow compute tests affected by the move**

Run:

```powershell
cd viewer
npx vitest run tests/compute/gpu-pipeline.test.ts tests/compute/webgpuUtciPipeline.behavior.test.ts tests/compute/selectedHourOutputHandle.test.ts tests/compute/live-selected-hour-session.test.ts --reporter=dot
```

Expected:

- PASS.
- If `tests/compute/selectedHourOutputHandle.test.ts` does not exist, replace it with the actual test path returned by `rg -n "createSelectedHourOutputHandle|disposeSelectedHourOutputHandle" viewer/tests`.

- [ ] **Step 8: Subagent review gate for boundary move**

In Codex, dispatch one review subagent with `spawn_agent`, wait for it with `wait_agent`, and paste the review result into the implementation status before proceeding. Use this exact prompt:

```text
Review the selected-hour output-handle move in D:\Projects\Nur\Shade\fast-utci. Do not edit files. Verify that gpu code no longer imports from compute/selected-hour/selectedHourOutputHandle, the moved handle remains behavior-identical, selected-hour proof/fallback semantics were not changed, and compute-manager.ts / telemetry.ts were not moved. Report blockers first.
```

Expected:

- No blockers.
- If blockers are found, fix only this task's boundary move before continuing.

---

## Task 2: Fix Recently Touched Selected-Hour And Compute-Adjacent Check Errors

**Files:**
- Modify: selected-hour/compute test files reported by `npm run check`
- Modify: `viewer/src/lib/compute/gpu/mergeAndBvh.worker.ts`
- Modify: `viewer/src/routes/debug/+page.svelte`
- Modify: `viewer/tests/compute/webgpuUtciPipeline.behavior.test.ts` if reported

- [ ] **Step 1: Filter current check output to selected-hour/compute-adjacent files**

Run:

```powershell
rg -n "src\\lib\\compute|tests\\compute|routes\\debug\\+page\.svelte" ..\.npm-check-before.txt
```

Expected:

- A concrete list of current selected-hour/compute-adjacent static errors.

- [ ] **Step 2: Fix stale selected-hour controller mocks**

In any test fake controller object missing `releaseAcceptedGpuResidentOutput`, add the current no-op method:

```ts
releaseAcceptedGpuResidentOutput() {
	return undefined;
}
```

If the real interface returns a specific release result, match the exact return type in `viewer/src/lib/compute/selected-hour/liveSelectedHourController.ts`.

- [ ] **Step 3: Fix stale selected-hour surface identity fixtures**

In selected-hour route/controller/projection tests, update fake `LiveSelectedHourSurfaceIdentity` objects so they include the current identity fields from `viewer/src/lib/compute/selected-hour/liveSelectedHourSurfaceIdentity.ts`.

Use the same stable fixture values across a given test:

```ts
const surfaceIdentity = {
	analysisId: 'test-analysis',
	side: 'base',
	requestId: 1,
	timeIndex: 168,
	controllerIdentity: 'test-analysis:base',
	controllerInstanceId: 1
} satisfies LiveSelectedHourSurfaceIdentity;
```

If the current type uses different field names, use the exact names from `liveSelectedHourSurfaceIdentity.ts`; do not add optional `any` escape hatches.

- [ ] **Step 4: Fix stale accepted GPU output fixtures**

In selected-hour tests, update fake `SelectedHourGpuResidentOutput` values to satisfy the current type from `viewer/src/lib/compute/selected-hour/liveUtciSelectedHourSession.ts`.

Use `satisfies SelectedHourGpuResidentOutput` where possible:

```ts
const output = {
	format: 'gpu-buffer',
	numPoints: 4,
	timeIndex: 168,
	outputBytes: 16,
	gpuOutputHandle: fakeHandle
} satisfies SelectedHourGpuResidentOutput;
```

If the actual type has different required fields, copy the required field list from `liveUtciSelectedHourSession.ts` and keep fake values inert.

- [ ] **Step 5: Fix `ArrayBufferLike` transfer-list errors in `mergeAndBvh.worker.ts`**

In `viewer/src/lib/compute/gpu/mergeAndBvh.worker.ts`, avoid passing typed-array `.buffer` directly when TypeScript sees it as `ArrayBufferLike`.

Use a helper local to the worker:

```ts
function transferableArrayBuffer(view: ArrayBufferView): ArrayBuffer {
	const { buffer, byteOffset, byteLength } = view;
	if (buffer instanceof ArrayBuffer && byteOffset === 0 && byteLength === buffer.byteLength) {
		return buffer;
	}
	return buffer.slice(byteOffset, byteOffset + byteLength);
}
```

Then update the transfer list so typed-array buffers use the helper:

```ts
self.postMessage(
	{ type: 'complete', result: serialized },
	[
		serialized.bvhNodeBuffer,
		serialized.bvhIndexBuffer,
		transferableArrayBuffer(serialized.vertexBuffer),
		transferableArrayBuffer(serialized.indexBuffer)
	]
);
```

Preserve any existing message shape and worker result fields.

- [ ] **Step 6: Fix debug-route parity intermediate assignment types**

In `viewer/src/routes/debug/+page.svelte`, find the assignment to `win.__parityIntermediates__`.

Preserve the existing keys, slicing windows, parity/export semantics, and collection timing. This step only coerces container types from `Float32Array | null` to `number[] | undefined` for the already-declared browser contract.

Introduce a small conversion helper near the assignment:

```ts
function toNumberArrayOrUndefined(values: Float32Array | number[] | null | undefined): number[] | undefined {
	if (!values) return undefined;
	return Array.from(values);
}
```

Then assign properties using the helper:

```ts
win.__parityIntermediates__ = {
	solarExposure: toNumberArrayOrUndefined(
		results[0]?.slice(parityMode ? 0 : augustStart, parityMode ? results[0].length : augustEnd)
	),
	skyExposure: toNumberArrayOrUndefined(results[1] ? results[1].map((v) => v / TOME_WEIGHT) : undefined),
	mrt: toNumberArrayOrUndefined(
		results[2]?.slice(parityMode ? 0 : augustStart, parityMode ? results[2].length : augustEnd)
	),
	shortErf: toNumberArrayOrUndefined(
		mrtComponents?.shortErf?.slice(parityMode ? 0 : augustStart, parityMode ? mrtComponents.shortErf.length : augustEnd)
	),
	longErf: toNumberArrayOrUndefined(
		mrtComponents?.longErf?.slice(parityMode ? 0 : augustStart, parityMode ? mrtComponents.longErf.length : augustEnd)
	),
	shortDmrt: toNumberArrayOrUndefined(
		mrtComponents?.shortDmrt?.slice(parityMode ? 0 : augustStart, parityMode ? mrtComponents.shortDmrt.length : augustEnd)
	),
	longDmrt: toNumberArrayOrUndefined(
		mrtComponents?.longDmrt?.slice(parityMode ? 0 : augustStart, parityMode ? mrtComponents.longDmrt.length : augustEnd)
	)
};
```

If the existing object has additional properties, keep them and apply the same helper only where `Float32Array | null` conflicts with `number[] | undefined`.

- [ ] **Step 7: Run check after compute-adjacent cleanup**

Run:

```powershell
cd viewer
npm run check
```

Expected:

- FAIL until later tasks finish.
- Selected-hour/compute-adjacent errors from Task 2 are gone.
- Remaining errors are parity, UTCIData, Three/Svelte, or stale test-fixture families.

- [ ] **Step 8: Run selected-hour quality gate**

Run:

```powershell
cd viewer
npm run test:quality:selected-hour
```

Expected:

- PASS.

- [ ] **Step 9: Subagent review gate for compute-adjacent cleanup**

In Codex, dispatch one review subagent with `spawn_agent`, wait for it with `wait_agent`, and paste the review result into the implementation status before proceeding. Use this exact prompt:

```text
Review the compute-adjacent npm-check fixes in D:\Projects\Nur\Shade\fast-utci. Do not edit files. Focus on selected-hour runtime contract integrity, debug parity/fallback preservation, ArrayBuffer transfer correctness, and whether any `any`/type assertion was used to hide a real contract drift. Report blockers first.
```

Expected:

- No blockers before moving to inherited debt families.

---

## Task 3: Fix Parity Reference Union Narrowing

**Files:**
- Modify: `viewer/src/lib/parity/loadReferenceIntermediatesFromFs.ts`
- Modify: `viewer/src/lib/parity/buildParityReport.ts`
- Modify: `viewer/scripts/compare-parity.ts`
- Modify: `viewer/scripts/diagnose-mrt-worst-cell.ts`
- Modify: `viewer/scripts/diagnose-solar-flips.ts`
- Modify: `viewer/tests/parity/loadReferenceIntermediatesFromFs.test.ts`

- [ ] **Step 1: Add stage-specific overloads**

In `viewer/src/lib/parity/loadReferenceIntermediatesFromFs.ts`, replace the function signature with overloads followed by the existing implementation signature:

```ts
export function loadReferenceIntermediatesFromFs(basePath: string, stage: 'solar'): Promise<SolarReference>;
export function loadReferenceIntermediatesFromFs(basePath: string, stage: 'sky'): Promise<SkyReference>;
export function loadReferenceIntermediatesFromFs(basePath: string, stage: 'mrt'): Promise<MrtReference>;
export function loadReferenceIntermediatesFromFs(
	basePath: string,
	stage: 'solar' | 'sky' | 'mrt'
): Promise<SolarReference | SkyReference | MrtReference>;
```

Keep the implementation body unchanged except for necessary formatting.

- [ ] **Step 2: Update parity report call sites**

In `viewer/src/lib/parity/buildParityReport.ts`, keep stage arguments as string literals:

```ts
const ref = await loadReferenceIntermediatesFromFs(basePath, 'solar');
const refArr = refToArr(ref.solarExposure);
```

Do not store the stage in a widened `string` variable unless it is explicitly typed as `'solar'`, `'sky'`, or `'mrt'`.

- [ ] **Step 3: Update parity scripts**

In `viewer/scripts/diagnose-solar-flips.ts`, make sure parallel loads preserve literal-stage narrowing:

```ts
const [refSolar, refMrt, refSky] = await Promise.all([
	loadReferenceIntermediatesFromFs(basePath, 'solar'),
	loadReferenceIntermediatesFromFs(basePath, 'mrt'),
	loadReferenceIntermediatesFromFs(basePath, 'sky').catch(() => null)
]);
```

After this, `refSolar.solarExposure`, `refMrt.mrt`, `refMrt.short_erf`, and `refSky?.skyExposure` should type-check without broad casts.

- [ ] **Step 4: Guard optional MRT component arrays before use**

In parity scripts, before reading optional MRT component arrays, add explicit guards:

```ts
if (!refMrt.short_erf || !refMrt.short_dmrt || !refMrt.long_erf || !refMrt.long_dmrt) {
	throw new Error('MRT reference is missing component arrays required for this diagnostic.');
}
```

Only add this guard in scripts that actually require all component arrays.

- [ ] **Step 5: Run parity-related tests and check**

Run:

```powershell
cd viewer
npx vitest run tests/parity/loadReferenceIntermediatesFromFs.test.ts --reporter=dot
npm run check
```

Expected:

- Parity test PASS.
- `npm run check` may still fail, but `SolarReference | SkyReference | MrtReference` property errors are gone.

---

## Task 4: Fix UTCIData Union Narrowing

**Files:**
- Modify: `viewer/src/lib/types/analysis.ts`
- Modify: `viewer/src/lib/services/dataLoader.ts`
- Modify: `viewer/src/lib/services/validationService.ts`
- Modify: `viewer/src/lib/components/ui/AnalyticsPanel.svelte`
- Modify: `viewer/tests/services/dataLoader.test.ts`
- Modify: any other file still reported for `utciValues`, `utciByHour`, or `utciStorage` on `UTCIData`

- [ ] **Step 1: Add exported data-shape guards**

In `viewer/src/lib/types/analysis.ts`, after `export type UTCIData = SingleHourData | FullDayData;`, add:

```ts
export function isSingleHourData(data: UTCIData): data is SingleHourData {
	return data.numHours === 1 && 'utciValues' in data;
}

export function isFullDayData(data: UTCIData): data is FullDayData {
	return data.numHours !== 1;
}

export function hasDecodedUtciByHour(data: UTCIData): data is FullDayData & { utciByHour: Float32Array[] } {
	return isFullDayData(data) && Array.isArray(data.utciByHour);
}

export function hasCompactUtciStorage(data: UTCIData): data is FullDayData & { utciStorage: UtciStorage } {
	return isFullDayData(data) && data.utciStorage !== undefined;
}
```

- [ ] **Step 2: Replace unsafe full-day access in `dataLoader.ts`**

In `viewer/src/lib/services/dataLoader.ts`, import the guards:

```ts
import {
	hasCompactUtciStorage,
	hasDecodedUtciByHour,
	isSingleHourData,
	type UTCIData
} from '$lib/types/analysis';
```

Update `getUTCIValue` and `getUTCIForHour` branches so they never use `full.utciByHour!` without checking:

```ts
if (isSingleHourData(data)) {
	return data.utciValues[positionIndex];
}
if (hasCompactUtciStorage(data)) {
	// keep existing compact-storage decode path
}
if (hasDecodedUtciByHour(data)) {
	return data.utciByHour[hourIndex]?.[positionIndex];
}
return undefined;
```

For `getUTCIForHour`, preserve current return semantics but guard the decoded array:

```ts
if (isSingleHourData(data)) {
	return data.utciValues;
}
if (hasCompactUtciStorage(data)) {
	// keep existing compact-storage decode path
}
if (hasDecodedUtciByHour(data)) {
	return data.utciByHour[hourIndex];
}
return undefined;
```

If the existing function currently throws for missing hour data, keep the existing throw behavior instead of returning `undefined`.

- [ ] **Step 3: Narrow analysis data in `validationService.ts`**

In `viewer/src/lib/services/validationService.ts`, import `isSingleHourData` and `hasDecodedUtciByHour`.

Replace direct union access with:

```ts
if (isSingleHourData(analysis.data)) {
	analysisValues = analysis.data.utciValues;
	validationHourIndex = Number(analysis.metadata.hours[0] ?? 0);
} else if (hasDecodedUtciByHour(analysis.data)) {
	analysisValues = analysis.data.utciByHour[hourIndex];
} else {
	throw new Error('Validation requires decoded UTCI arrays for the selected analysis.');
}
```

Use the actual local variable names from the file.

- [ ] **Step 4: Narrow analysis data in `AnalyticsPanel.svelte`**

In `viewer/src/lib/components/ui/AnalyticsPanel.svelte`, import `hasCompactUtciStorage` and `hasDecodedUtciByHour`.

Replace direct checks:

```ts
analysis.data.utciStorage != null || (analysis.data.utciByHour?.length ?? 0) > 0
```

with:

```ts
hasCompactUtciStorage(analysis.data) || hasDecodedUtciByHour(analysis.data)
```

- [ ] **Step 5: Update data-loader tests with guards**

In `viewer/tests/services/dataLoader.test.ts`, import `isSingleHourData` and `hasDecodedUtciByHour`.

Before single-hour assertions, add:

```ts
expect(isSingleHourData(result)).toBe(true);
if (!isSingleHourData(result)) throw new Error('Expected single-hour result');
```

Before full-day decoded assertions, add:

```ts
expect(hasDecodedUtciByHour(result)).toBe(true);
if (!hasDecodedUtciByHour(result)) throw new Error('Expected decoded full-day result');
```

Then keep existing assertions against `result.utciValues` or `result.utciByHour`.

- [ ] **Step 6: Run data tests and check**

Run:

```powershell
cd viewer
npx vitest run tests/services/dataLoader.test.ts --reporter=dot
npm run check
```

Expected:

- Data-loader tests PASS.
- `UTCIData` union property errors are gone.

---

## Task 5: Fix Three Object And Material Typing Debt

**Files:**
- Modify: `viewer/src/lib/services/modelLoaderService.ts`
- Modify: `viewer/src/lib/services/layerManagerService.ts`
- Modify: `viewer/src/lib/services/modelCacheService.ts`
- Modify: `viewer/tests/services/modelLoaderService.unknownLayers.test.ts`

- [ ] **Step 1: Replace `child.isMesh` direct checks in runtime services**

In `viewer/src/lib/services/modelLoaderService.ts`, replace patterns like:

```ts
if (child.isMesh) {
	const mesh = child as THREE.Mesh;
}
```

with:

```ts
if (child instanceof THREE.Mesh) {
	const mesh = child;
}
```

Replace line checks:

```ts
if (child.isLine || child.isLineSegments) {
```

with:

```ts
if (child instanceof THREE.Line || child instanceof THREE.LineSegments) {
```

- [ ] **Step 2: Replace layer manager object flags**

In `viewer/src/lib/services/layerManagerService.ts`, replace:

```ts
if (child.isMesh && child.userData.layerType) {
	const mesh = child as THREE.Mesh;
}
```

with:

```ts
if (child instanceof THREE.Mesh && child.userData.layerType) {
	const mesh = child;
}
```

Replace:

```ts
if (child.isLineSegments && child.name.includes('_edges')) {
```

with:

```ts
if (child instanceof THREE.LineSegments && child.name.includes('_edges')) {
```

- [ ] **Step 3: Add typed texture disposal helper**

In `viewer/src/lib/services/modelCacheService.ts`, add:

```ts
function disposeTexture(value: unknown): void {
	if (value instanceof THREE.Texture) {
		value.dispose();
	}
}
```

Then replace map disposal blocks with:

```ts
disposeTexture(material.map);
disposeTexture(material.lightMap);
disposeTexture(material.bumpMap);
disposeTexture(material.normalMap);
disposeTexture(material.specularMap);
disposeTexture(material.envMap);
```

Only access properties that exist on the narrowed material type. If the material is typed as `THREE.Material`, first narrow:

```ts
const materialWithTextures = material as THREE.Material & {
	map?: unknown;
	lightMap?: unknown;
	bumpMap?: unknown;
	normalMap?: unknown;
	specularMap?: unknown;
	envMap?: unknown;
};
```

Then dispose through `materialWithTextures`.

- [ ] **Step 4: Update tests to use `instanceof THREE.Mesh`**

In `viewer/tests/services/modelLoaderService.unknownLayers.test.ts`, replace test traversal checks:

```ts
if (child.isMesh) {
	const meshChild = child as THREE.Mesh;
}
```

with:

```ts
if (child instanceof THREE.Mesh) {
	const meshChild = child;
}
```

- [ ] **Step 5: Run service tests and check**

Run:

```powershell
cd viewer
npx vitest run tests/services/modelLoaderService.unknownLayers.test.ts --reporter=dot
npm run check
```

Expected:

- Service tests PASS.
- Three `Object3D` property errors are gone.

---

## Task 6: Fix Svelte/Threlte Strictness Debt

**Files:**
- Modify: `viewer/src/lib/components/scene/Model.svelte`
- Modify: `viewer/src/lib/components/scene/SunPath.svelte`
- Modify: `viewer/src/lib/components/scene/UTCIPointCloud.svelte`
- Modify: `viewer/src/lib/components/ui/ThemeToggle.svelte`
- Inspect: `viewer/src/lib/stores/viewerStore.ts` or exact store file that defines `ViewerState`

- [ ] **Step 1: Fix `Model.svelte` unused expect-error and callback return**

In `viewer/src/lib/components/scene/Model.svelte`, remove:

```ts
// @ts-expect-error - Svelte warns about module var mutation but it's intentional here
```

Do not remove the following `globalModelVersion++` unless a later check error requires a different state pattern.

Change:

```svelte
<T is={THREE.Group} oncreate={(ref) => ref.add(gltfGroup!)}>
```

to:

```svelte
<T is={THREE.Group} oncreate={(ref) => {
	ref.add(gltfGroup!);
}}>
```

- [ ] **Step 2: Investigate `sunPathVisible` without changing route/store behavior**

Inspect store definition:

```powershell
rg -n "interface ViewerState|type ViewerState|sunPathVisible|currentHour" viewer/src/lib/stores viewer/src/lib
```

Expected:

- If `sunPathVisible` already exists under a different store/type name, update `SunPath.svelte` to use the existing field.
- If `sunPathVisible` does not exist and `SunPath.svelte` is active, stop and report this as a behavior/store contract issue before adding new state.
- If `SunPath.svelte` is dead/unmounted, stop and report that dead-code removal is a separate cleanup decision before removing it.

Do not add a new store field or remove the component inside this plan unless the user explicitly approves that behavior decision during execution.

- [ ] **Step 3: Fix Threlte position tuple in `SunPath.svelte`**

Where marker positions are passed as `THREE.Vector3`, convert to a tuple:

```svelte
position={[markerPositions[hour].x, markerPositions[hour].y, markerPositions[hour].z]}
```

If `markerPositions[hour]` can be undefined, guard the render block or create a helper:

```ts
function vectorToTuple(vector: THREE.Vector3): [number, number, number] {
	return [vector.x, vector.y, vector.z];
}
```

Then use:

```svelte
position={vectorToTuple(markerPositions[hour])}
```

- [ ] **Step 4: Fix `ThemeToggle.svelte` self-closing span**

In `viewer/src/lib/components/ui/ThemeToggle.svelte`, replace:

```svelte
<span class:knob-dark={$viewerStore.theme === 'dark'} class="knob" />
```

with:

```svelte
<span class:knob-dark={$viewerStore.theme === 'dark'} class="knob"></span>
```

- [ ] **Step 5: Fix or intentionally quiet `UTCIPointCloud.svelte` unused export**

Inspect whether `model` is passed into `UTCIPointCloud.svelte`:

```powershell
rg -n "<UTCIPointCloud|model=" viewer/src
```

If no consumer passes `model`, remove the prop:

```ts
export let model: Group | null = null;
```

and remove the unused `Group` import only if it becomes unused.

If consumers pass it for future external reference, change it to:

```ts
export const model: Group | null = null;
```

Only choose this second option if Svelte's warning text applies and the component API really needs an exported constant.

- [ ] **Step 6: Run check**

Run:

```powershell
cd viewer
npm run check
```

Expected:

- Svelte component warnings/errors from this task are gone.
- Remaining errors, if any, are stale tests or isolated type mismatches.

---

## Task 7: Fix Stale Test Fixtures And Remaining Check Errors

**Files:**
- Modify: `viewer/tests/services/pointCloudService.surface.test.ts`
- Modify: any remaining files reported by `npm run check`

- [ ] **Step 1: Update `UtciGridLayout` test fixtures**

In `viewer/tests/services/pointCloudService.surface.test.ts`, add the required layout fields to each inline layout object missing them:

```ts
coordinateSystem: 'xy_ground',
minY: 0,
maxY: 0,
```

For fixtures with `baseY` set to a nonzero value, use:

```ts
minY: baseY,
maxY: baseY,
```

or inline the same numeric value.

- [ ] **Step 2: Run the point-cloud surface tests**

Run:

```powershell
cd viewer
npx vitest run tests/services/pointCloudService.surface.test.ts --reporter=dot
```

Expected:

- PASS.

- [ ] **Step 3: Run full check and inspect remaining errors**

Run:

```powershell
cd viewer
npm run check
```

Expected:

- Ideally PASS.
- If it still fails, every remaining error must be triaged into one of:
  - missed family already covered by Tasks 2-7
  - real behavior contract mismatch
  - out-of-scope unrelated user change

- [ ] **Step 4: Fix only remaining static errors that match this plan's families**

For each remaining check error, follow this rule:

```text
Prefer source-level narrowing, overloads, or accurate fixture updates. Do not use `as any`, `@ts-ignore`, broad `unknown as`, or deleted tests unless the test is proven obsolete and a reviewer agrees.
```

Expected:

- `npm run check` reaches 0 errors and 0 warnings.

- [ ] **Step 5: Subagent review gate for check cleanup**

In Codex, dispatch two review subagents in parallel with `spawn_agent`, wait for both with `wait_agent`, and paste both review results into the implementation status before proceeding. Use these exact prompts:

```text
Review the npm-check cleanup in D:\Projects\Nur\Shade\fast-utci. Do not edit files. Focus on type-safety quality: no `as any`/`@ts-ignore`/deleted coverage hiding real issues, no behavior changes outside static correctness, and `npm run check` is legitimately clean. Report blockers first.
```

```text
Review the npm-check cleanup in D:\Projects\Nur\Shade\fast-utci. Do not edit files. Focus on WebGPU/selected-hour/parity proof surfaces: runAll, readUtciBulk, readUtcisSlice, dataTexture fallback, Python/.bin comparison, debug parity, and strongVisibleGpuPath semantics must remain intact. Report blockers first.
```

Expected:

- Both subagents report no blockers.
- If a blocker is found, fix it and re-dispatch the relevant review once.

---

## Task 8: Fix Remaining Known Current Check Debt

**Files:**
- Modify: `viewer/src/lib/parity/loadReferenceFromFs.ts`
- Modify: `viewer/src/lib/services/lruCache.ts`
- Modify: `viewer/src/lib/stores/layerStore.ts`
- Modify: `viewer/tests/parity/loadWebgpuCollectedFromFs.test.ts`
- Modify: any remaining file reported by `npm run check` only if it matches this task's small-narrowing category

- [ ] **Step 1: Fix `loadReferenceFromFs.ts` ArrayBuffer ownership**

In `viewer/src/lib/parity/loadReferenceFromFs.ts`, replace:

```ts
const data = parseFullDayBinaryNode(
	buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength)
);
```

with:

```ts
const ownedBuffer = new Uint8Array(buffer.byteLength);
ownedBuffer.set(buffer);
const data = parseFullDayBinaryNode(ownedBuffer.buffer);
```

Expected:

- `parseFullDayBinaryNode` still receives an `ArrayBuffer`.
- `.bin` parsing semantics are unchanged.

- [ ] **Step 2: Fix `lruCache.ts` iterator narrowing**

In `viewer/src/lib/services/lruCache.ts`, replace:

```ts
const firstKey = this.cache.keys().next().value;
const evictedValue = this.cache.get(firstKey)!;
this.cache.delete(firstKey);
```

with:

```ts
const firstKey = this.cache.keys().next().value;
if (firstKey !== undefined) {
	const evictedValue = this.cache.get(firstKey)!;
	this.cache.delete(firstKey);

	if (this.onEvict) {
		this.onEvict(firstKey, evictedValue);
	}
}
```

Then remove the old duplicate `if (this.onEvict)` block for the same eviction so the callback is not called twice.

- [ ] **Step 3: Fix `layerStore.ts` assigned-before-use error**

In `viewer/src/lib/stores/layerStore.ts`, replace:

```ts
let newVisible: boolean;
layerStore.update((state) => {
	newVisible = !state[layerId];
	return {
		...state,
		[layerId]: newVisible
	};
});
toggleLayerVisibility(layerId, newVisible);
```

with:

```ts
let newVisible = false;
layerStore.update((state) => {
	newVisible = !state[layerId];
	return {
		...state,
		[layerId]: newVisible
	};
});
toggleLayerVisibility(layerId, newVisible);
```

Do not change visibility semantics.

- [ ] **Step 4: Fix optional `instanceof` in `loadWebgpuCollectedFromFs.test.ts`**

In `viewer/tests/parity/loadWebgpuCollectedFromFs.test.ts`, replace optional `instanceof` checks like:

```ts
expect(Array.isArray(out.solar?.solarExposure) || out.solar?.solarExposure instanceof Float32Array).toBe(true);
```

with a narrowed local:

```ts
const solarExposure = out.solar?.solarExposure;
expect(
	Array.isArray(solarExposure) || solarExposure instanceof Float32Array
).toBe(true);
```

Apply the same pattern for any optional collected array in the file.

- [ ] **Step 5: Run targeted tests**

Run:

```powershell
cd viewer
npx vitest run tests/parity/loadReferenceFromFs.test.ts tests/parity/loadWebgpuCollectedFromFs.test.ts --reporter=dot
```

Expected:

- PASS.

- [ ] **Step 6: Run check and apply the stop rule**

Run:

```powershell
cd viewer
npm run check
```

Expected:

- PASS, or only a small number of remaining static errors.

Stop and report before fixing any remaining error that touches:

- route behavior,
- parity payload schema,
- `.bin` or Python reference semantics,
- selected-hour runtime contract semantics,
- `dataTexture` fallback behavior,
- `readUtciBulk()` / `readUtcisSlice()` behavior.

Only fix remaining errors inline if they are mechanical type narrowing, overloads, or stale test fixture shape within this plan's already listed families.

## Task 9: Full Verification And Cleanup

**Files:**
- Inspect only unless generated state needs cleanup.

- [ ] **Step 1: Run final check**

Run:

```powershell
cd viewer
npm run check
```

Expected:

- PASS with 0 errors and 0 warnings.

- [ ] **Step 2: Run selected-hour quality suite**

Run:

```powershell
cd viewer
npm run test:quality:selected-hour
```

Expected:

- PASS.

- [ ] **Step 3: Run selected-hour E2E suite**

Run:

```powershell
cd viewer
npm run test:e2e:selected-hour
```

Expected:

- PASS.
- If this modifies `viewer/test-results/.last-run.json`, restore it before final status.

- [ ] **Step 4: Run parity/reference preservation tests**

Run:

```powershell
cd viewer
npx vitest run tests/parity/loadReferenceFromFs.test.ts tests/parity/loadReferenceIntermediatesFromFs.test.ts tests/parity/loadWebgpuCollectedFromFs.test.ts tests/parity/buildParityReport.test.ts --reporter=dot
```

Expected:

- PASS.
- `.bin`, Python/reference, and parity artifact loading/report behavior remain covered.

- [ ] **Step 5: Run readback/fallback/source-lock preservation tests**

Run:

```powershell
cd viewer
npx vitest run tests/compute/webgpu-on-demand-source-locks.test.ts tests/debug/debug-webgpu-utci-diagnostics.test.ts tests/debug/debug-selected-hour-mode.test.ts tests/services/pointCloudService.surface.test.ts tests/diagnostics/selectedHourRuntimeContract.test.ts --reporter=dot
```

Expected:

- PASS.
- `readUtcisSlice()`, `readUtciBulk()`, debug parity boundaries, `dataTexture` fallback, and selected-hour runtime contract semantics remain covered.

- [ ] **Step 6: Run production build**

Run:

```powershell
cd viewer
npm run build
```

Expected:

- PASS.
- Existing large-chunk warnings are acceptable.

- [ ] **Step 7: Remove temporary check log**

Run:

```powershell
Remove-Item -LiteralPath ".npm-check-before.txt" -ErrorAction SilentlyContinue
```

Expected:

- `.npm-check-before.txt` is absent.

- [ ] **Step 8: Clean generated Playwright state if needed**

Run:

```powershell
git status --short
```

If it shows only `viewer/test-results/.last-run.json` as generated state, restore it:

```powershell
git restore -- "viewer/test-results/.last-run.json"
```

Then run:

```powershell
git status --short
```

Expected:

- Only intentional source/test/doc changes remain.

- [ ] **Step 9: Run diff whitespace check**

Run:

```powershell
git diff --check
```

Expected:

- PASS with no output.

- [ ] **Step 10: Final subagent verification**

In Codex, dispatch one final review subagent with `spawn_agent`, wait for it with `wait_agent`, and paste the review result into the final implementation status. Use this exact prompt:

```text
Final review for D:\Projects\Nur\Shade\fast-utci npm-check cleanup and selected-hour boundary move. Do not edit files. Verify final command evidence, generated-state cleanup, no commits/worktrees, selected-hour proof/fallback preservation, and whether code quality is now sufficient to proceed with feature/improvement work. Report blockers first, then recommended next step.
```

Expected:

- No blockers.
- Recommendation should explicitly say whether more refactor is needed before feature work.

## Completion Criteria

This plan is complete only when all are true:

- `cd viewer; npm run check` passes with 0 errors and 0 warnings.
- `cd viewer; npm run test:quality:selected-hour` passes.
- `cd viewer; npm run test:e2e:selected-hour` passes.
- `cd viewer; npm run build` passes.
- `git diff --check` passes.
- `viewer/test-results/.last-run.json` is not left modified.
- `viewer/src/lib/compute/gpu/` no longer imports from `viewer/src/lib/compute/selected-hour/selectedHourOutputHandle`.
- `compute-manager.ts` and `telemetry.ts` remain at `viewer/src/lib/compute/`.
- Subagent reviews report no blockers.

## Next Best Steps After This Plan

If this plan completes cleanly, code quality should be high enough to resume feature and performance improvements without another broad refactor first.

The next best step is to pick product/performance work based on user value, not housekeeping:

- If the priority is UX/product: improve the main route controls and debug-to-product polish now that selected-hour proof gates and static typing are clean.
- If the priority is performance: return to cold-start, memory, and tiling work for dense grids, using the current selected-hour GPU path as the stable baseline.
- If the priority is maintainability: write a dedicated compute-manager split plan only when a concrete feature requires it. Do not split `compute-manager.ts` just because it is central; it is currently an honest facade.

Do not schedule another general cleanup pass immediately after this one unless final verification or subagent review finds a real blocker.
