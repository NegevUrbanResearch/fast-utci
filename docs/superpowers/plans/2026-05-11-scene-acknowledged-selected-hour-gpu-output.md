# Scene-Acknowledged Selected-Hour GPU Output Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Workflow override:** Do **not** create commits. Do **not** create git worktrees. Preserve unrelated dirty files. Every task must report fresh verification before claiming completion.

**Goal:** Make selected-hour GPU output ownership scene-acknowledged for the shared-host `/` and `/debug` paths, with only the smallest verification-script and browser-proof additions needed to keep this ownership fix reviewable.

**Architecture:** Keep `liveSelectedHourController` / `liveSelectedHourRouteHost` as the canonical selected-hour runtime spine, but move disposal of accepted GPU-resident outputs behind an explicit scene release handshake. `UTCIPointCloud` and `ComparisonRenderer` remain scene-facing owners of render-copy completion knowledge; routes forward release callbacks to the route host, and the route host forwards them to the correct controller. Tests must prove that old accepted GPU buffers survive until scene release and that visible-path diagnostics remain honest while comparison/range/tooltip readbacks are separately accounted.

**Tech Stack:** SvelteKit, Svelte components, Three/Threlte WebGPU renderer, WebGPU UTCI compute, Vitest, Playwright Chromium with `--enable-unsafe-webgpu`, PowerShell on Windows.

---

## Current State Snapshot

This plan starts after:

- `2c363bc feat(debug): baseline selected-hour runtime quality`
- `3aade04 feat(debug): prove selected-hour visible GPU path`

Relevant current behavior:

- `UTCIPointCloud.svelte` already has `onAcceptedGpuResidentOutputRelease`, and it calls the callback for `copy-complete`, `copy-failed`, and `superseded`.
- That callback is only wired for the legacy debug path. The main route does not pass a release callback to `UTCIPointCloud`, and `ComparisonRenderer` does not expose a release callback.
- `liveSelectedHourController` currently disposes the previous accepted GPU output immediately when it is replaced by another accepted result.
- `liveSelectedHourRouteHost` already tracks `controllerIdentity` and `baseControllerGeneration` / `comparisonControllerGeneration` for diagnostics safety. Scene release forwarding must use the same identity boundary; request ids alone are not globally unique because each controller starts at request id `1`.
- `test:quality:selected-hour` is green but does not include all helper tests introduced by the recent quality baseline.
- Main route comparison readback accounting is unit-tested, but not route-visible Playwright-proven.

## Non-Goals

- Do not decompose `viewer/src/routes/debug/+page.svelte` in this plan.
- Do not extract the full `UTCIPointCloud` / `ComparisonRenderer` scene sync state machine in this plan.
- Do not optimize 0.5m performance.
- Do not remove `dataTexture`, `.bin`, Python comparison, parity, collect, or debug fallback paths.
- Do not fix repo-wide `npm run check` static debt except for new errors created by this plan.
- Do not reintroduce `/debug-webgpu-utci` compatibility unless a later plan explicitly chooses that.

## Scope Boundary

Tasks 1-4 are the core ownership fix and must land as one coherent slice. Tasks 5-6 are intentionally small proof-surface follow-ups in the same plan because they were direct review findings from the completed runtime quality baseline:

- Task 5 makes the existing selected-hour quality command cover the helper tests this ownership fix depends on.
- Task 6 adds one missing route-visible comparison readback proof.

If Tasks 1-4 expose a larger lifecycle flaw, stop before Tasks 5-6 and report the ownership findings first. Do not let the proof follow-ups expand into broader route decomposition or scene sync extraction.

## Quality Gates

Stop and report findings before continuing if any of these happen:

- A shared-host accepted GPU output is disposed before `copy-complete`, `copy-failed`, `superseded`, controller disposal, or route-host disposal.
- A scene release from an old controller generation can release a current controller's accepted or retired output.
- A stale output can never be released after being superseded.
- `strongVisibleGpuPath` becomes true without scene request id / selection key matching the accepted request.
- Main route comparison readbacks are counted as visible selected-hour readbacks.
- `/debug` parity mode loses legacy debug `.bin` behavior.
- `test:quality:selected-hour` can pass while helper tests for ownership and bridge behavior fail.
- A new Playwright wait relies on the outer test timeout instead of a short wait that dumps diagnostics.

## File Structure

### Modify

- `viewer/src/lib/compute/liveSelectedHourController.ts`
  - Add an explicit `releaseAcceptedGpuResidentOutput(...)` API.
  - Retire superseded accepted GPU outputs until the scene releases them.
  - Keep stale/unaccepted outputs disposed immediately.
- `viewer/src/lib/compute/liveSelectedHourRouteHost.ts`
  - Add base/comparison release methods.
  - Attach `controllerIdentity` to published scene surface identities.
  - Forward scene releases only when the release identity matches the current slot controller identity.
- `viewer/src/lib/compute/liveSelectedHourSurfaceIdentity.ts`
  - Add `controllerIdentity` so scene release callbacks can identify the controller instance that produced an accepted GPU output.
- `viewer/src/lib/components/scene/ComparisonRenderer.svelte`
  - Add `onAcceptedGpuResidentOutputRelease`.
  - Emit release reasons using the same payload shape as `UTCIPointCloud`.
- `viewer/src/lib/components/scene/UTCIPointCloud.svelte`
  - Include `controllerIdentity` from `liveSelectedHourSurfaceIdentity` in release callback payloads.
- `viewer/src/routes/+page.svelte`
  - Pass base and comparison release callbacks into scene components.
- `viewer/src/routes/debug/+page.svelte`
  - Wire shared-host base release callback to the debug shared route host while preserving the existing legacy debug release callback.
- `viewer/package.json`
  - Expand `test:quality:selected-hour` or add a companion quality script so helper ownership tests are included in the required quality command.
- `viewer/tests/compute/live-selected-hour-controller.test.ts`
  - Replace eager-disposal expectations with scene-release expectations.
- `viewer/tests/compute/live-selected-hour-route-host.test.ts`
  - Assert route host release forwarding for base and comparison controllers.
- `viewer/tests/scene/utciComputeBufferRenderBridge.test.ts`
  - Keep existing low-level bridge coverage; extend only if needed for release failure cases.
- `viewer/tests/e2e/main-route-manual-diagnostics.spec.ts`
  - Add route-visible comparison readback separation proof.
- `viewer/tests/e2e/debug-route-shared-host-diagnostics.spec.ts`
  - Keep existing shared-host route proof green. Do not claim release-event proof from browser diagnostics unless explicit release diagnostics are added in a later plan.

### Inspect Only

- `viewer/src/lib/components/scene/UTCIPointCloud.svelte`
  - Existing release callback behavior should be preserved, with only `controllerIdentity` added to release payloads.
- `viewer/src/lib/debug/debugSelectedHourLegacyHost.ts`
  - Use as the proven reference pattern for retire-then-release ownership.
- `viewer/src/lib/diagnostics/selectedHourRuntimeContract.ts`
  - Do not loosen `strongVisibleGpuPath` conditions.

---

## Task 0: Baseline And Dirty-State Record

**Files:**
- Inspect only: repo state and current proof commands.

- [ ] **Step 1: Record current git state**

Run from repo root:

```powershell
git status --short
git log --oneline -6
```

Expected:

- Preserve all unrelated dirty files.
- Recent history includes `3aade04` and `2c363bc`.
- Do not create commits.
- Do not create git worktrees.

- [ ] **Step 2: Run focused baseline tests**

Run:

```powershell
cd viewer
npm run test:quality:selected-hour
npx vitest run tests/debug/debug-selected-hour-legacy-host.test.ts tests/debug/debug-on-demand-prototype-diagnostics.test.ts tests/compute/selectedHourOutputHandle.test.ts tests/scene/utciComputeBufferRenderBridge.test.ts tests/compute/onDemandDiagnostics.test.ts tests/components/canvasInteractionController.test.ts
```

Expected:

- PASS.
- If either command fails, use `superpowers:systematic-debugging` before editing.

- [ ] **Step 3: Run current browser proof**

Run:

```powershell
cd viewer
npm run test:e2e:selected-hour
```

Expected:

- PASS.
- `/` proves `compute-buffer-selected-hour`, zero visible selected-hour readbacks, no `.bin` requests.
- `/debug` shared-host and parity tests still pass.

---

## Task 1: Add Controller-Side Retired Output Release

**Files:**
- Modify: `viewer/src/lib/compute/liveSelectedHourController.ts`
- Modify: `viewer/tests/compute/live-selected-hour-controller.test.ts`

- [ ] **Step 1: Write the failing controller ownership test**

In `viewer/tests/compute/live-selected-hour-controller.test.ts`, replace the current eager-disposal test named:

```ts
it('disposes the previously accepted GPU buffer when a new accepted result supersedes it', async () => {
```

with this behavior:

```ts
it('retires a superseded accepted GPU output until the scene releases it', async () => {
	const firstGpu = createGpuResidentOutput(7, 7);
	const secondGpu = createGpuResidentOutput(8, 8);
	const sessionMock = createSessionMock([
		async () =>
			createLiveResult({
				requestId: 7,
				timeIndex: 7,
				analysis: createSelectionAnalysis('gpu-first', [19, 21]),
				gpuResidentOutput: firstGpu.accepted,
				renderTransport: 'compute-buffer-selected-hour',
				sameDeviceForComputeAndRender: true,
				pendingRenderUpdateStartedAt: 700
			}),
		async () =>
			createLiveResult({
				requestId: 8,
				timeIndex: 8,
				analysis: createSelectionAnalysis('gpu-second', [25, 27]),
				gpuResidentOutput: secondGpu.accepted,
				renderTransport: 'compute-buffer-selected-hour',
				sameDeviceForComputeAndRender: true,
				pendingRenderUpdateStartedAt: 800
			})
	]);
	const controller = createLiveSelectedHourController({
		prepareSession: vi.fn(async () => sessionMock.session)
	});

	await controller.requestSelection(createRequestParams(7));
	await controller.requestSelection(createRequestParams(8));

	expect(firstGpu.destroy).not.toHaveBeenCalled();
	expect(secondGpu.destroy).not.toHaveBeenCalled();

	controller.releaseAcceptedGpuResidentOutput({
		controllerIdentity: 'controller',
		requestId: 7,
		monthIndex: 0,
		timeIndex: 7,
		reason: 'superseded'
	});

	expect(firstGpu.destroy).toHaveBeenCalledTimes(1);
	expect(secondGpu.destroy).not.toHaveBeenCalled();
	expect(controller.getState().acceptedGpuResidentOutput?.requestId).toBe(8);
});
```

Add a second test:

```ts
it('marks the current accepted GPU output releasable and disposes it after replacement', async () => {
	const firstGpu = createGpuResidentOutput(11, 11);
	const secondGpu = createGpuResidentOutput(12, 12);
	const sessionMock = createSessionMock([
		async () =>
			createLiveResult({
				requestId: 11,
				timeIndex: 11,
				analysis: createSelectionAnalysis('gpu-first', [19, 21]),
				gpuResidentOutput: firstGpu.accepted,
				renderTransport: 'compute-buffer-selected-hour',
				sameDeviceForComputeAndRender: true
			}),
		async () =>
			createLiveResult({
				requestId: 12,
				timeIndex: 12,
				analysis: createSelectionAnalysis('gpu-second', [25, 27]),
				gpuResidentOutput: secondGpu.accepted,
				renderTransport: 'compute-buffer-selected-hour',
				sameDeviceForComputeAndRender: true
			})
	]);
	const controller = createLiveSelectedHourController({
		prepareSession: vi.fn(async () => sessionMock.session)
	});

	await controller.requestSelection(createRequestParams(11));
	controller.releaseAcceptedGpuResidentOutput({
		controllerIdentity: 'controller',
		requestId: 11,
		monthIndex: 0,
		timeIndex: 11,
		reason: 'copy-complete'
	});
	expect(firstGpu.destroy).not.toHaveBeenCalled();

	await controller.requestSelection(createRequestParams(12));

	expect(firstGpu.destroy).toHaveBeenCalledTimes(1);
	expect(secondGpu.destroy).not.toHaveBeenCalled();
});
```

Add a third test:

```ts
it('disposes retired GPU outputs on controller disposal even without scene release', async () => {
	const firstGpu = createGpuResidentOutput(13, 13);
	const secondGpu = createGpuResidentOutput(14, 14);
	const sessionMock = createSessionMock([
		async () =>
			createLiveResult({
				requestId: 13,
				timeIndex: 13,
				analysis: createSelectionAnalysis('gpu-first', [19, 21]),
				gpuResidentOutput: firstGpu.accepted,
				renderTransport: 'compute-buffer-selected-hour',
				sameDeviceForComputeAndRender: true
			}),
		async () =>
			createLiveResult({
				requestId: 14,
				timeIndex: 14,
				analysis: createSelectionAnalysis('gpu-second', [25, 27]),
				gpuResidentOutput: secondGpu.accepted,
				renderTransport: 'compute-buffer-selected-hour',
				sameDeviceForComputeAndRender: true
			})
	]);
	const controller = createLiveSelectedHourController({
		prepareSession: vi.fn(async () => sessionMock.session)
	});

	await controller.requestSelection(createRequestParams(13));
	await controller.requestSelection(createRequestParams(14));
	controller.dispose();

	expect(firstGpu.destroy).toHaveBeenCalledTimes(1);
	expect(secondGpu.destroy).toHaveBeenCalledTimes(1);
});
```

- [ ] **Step 2: Run the failing controller tests**

Run:

```powershell
cd viewer
npx vitest run tests/compute/live-selected-hour-controller.test.ts
```

Expected:

- FAIL because `releaseAcceptedGpuResidentOutput` does not exist and previous accepted GPU output is still disposed eagerly.

- [ ] **Step 3: Implement retired output ownership in the controller**

In `viewer/src/lib/compute/liveSelectedHourController.ts`, add exported types near the existing controller types:

```ts
export type LiveSelectedHourGpuResidentReleaseReason =
	| 'copy-complete'
	| 'copy-failed'
	| 'superseded';

export interface LiveSelectedHourGpuResidentRelease {
	controllerIdentity: string;
	requestId: number;
	monthIndex: number;
	timeIndex: number;
	reason: LiveSelectedHourGpuResidentReleaseReason;
}
```

Extend `LiveSelectedHourController`:

```ts
releaseAcceptedGpuResidentOutput(release: LiveSelectedHourGpuResidentRelease): void;
```

Add a managed-output helper shape near the mutable state helpers:

```ts
type ManagedGpuResidentOutput = {
	value: SelectedHourGpuResidentOutput;
	releasable: boolean;
};

function getGpuResidentOutputKey(output: Pick<SelectedHourGpuResidentOutput, 'requestId' | 'monthIndex' | 'timeIndex'>): string {
	return `${output.requestId}:${output.monthIndex}:${output.timeIndex}`;
}
```

Inside `createLiveSelectedHourController`, add:

```ts
const retiredGpuResidentOutputs = new Map<string, ManagedGpuResidentOutput>();
let acceptedGpuResidentOutputReleaseState: ManagedGpuResidentOutput | null = null;

function disposeManagedGpuResidentOutput(entry: ManagedGpuResidentOutput | null | undefined): void {
	if (!entry) return;
	disposeSelectedHourGpuResidentOutput(entry.value);
}

function retireAcceptedGpuResidentOutput(output: SelectedHourGpuResidentOutput | null): void {
	if (!output) return;
	const existingState =
		acceptedGpuResidentOutputReleaseState?.value === output
			? acceptedGpuResidentOutputReleaseState
			: { value: output, releasable: false };
	if (existingState.releasable) {
		disposeManagedGpuResidentOutput(existingState);
		return;
	}
	retiredGpuResidentOutputs.set(getGpuResidentOutputKey(output), existingState);
}

function setAcceptedGpuResidentOutputReleaseState(output: SelectedHourGpuResidentOutput | null): void {
	if (!output) {
		acceptedGpuResidentOutputReleaseState = null;
		return;
	}
	if (acceptedGpuResidentOutputReleaseState?.value === output) {
		return;
	}
	acceptedGpuResidentOutputReleaseState = { value: output, releasable: false };
}
```

Update `replaceAcceptedGpuResidentOutput` so it retires, not eagerly disposes:

```ts
function replaceAcceptedGpuResidentOutput(
	next: SelectedHourGpuResidentOutput | null,
	patch: Partial<LiveSelectedHourControllerMutableState>
): void {
	const previous = state.acceptedGpuResidentOutput;
	setState({
		...patch,
		acceptedGpuResidentOutput: next
	});
	if (previous && previous.output !== next?.output) {
		retireAcceptedGpuResidentOutput(previous);
	}
	setAcceptedGpuResidentOutputReleaseState(next);
}
```

Update `resetControllerState` and `dispose` to dispose both current and retired outputs:

```ts
function disposeAllGpuResidentOutputs(): void {
	for (const retired of retiredGpuResidentOutputs.values()) {
		disposeManagedGpuResidentOutput(retired);
	}
	retiredGpuResidentOutputs.clear();
	disposeManagedGpuResidentOutput(acceptedGpuResidentOutputReleaseState);
	acceptedGpuResidentOutputReleaseState = null;
}
```

Use `disposeAllGpuResidentOutputs()` when resetting or disposing the controller, while preserving the existing session disposal.

Add the public method:

```ts
releaseAcceptedGpuResidentOutput(release) {
	const releasedKey = getGpuResidentOutputKey(release);
	const current = acceptedGpuResidentOutputReleaseState;
	if (current && getGpuResidentOutputKey(current.value) === releasedKey) {
		acceptedGpuResidentOutputReleaseState = {
			...current,
			releasable: true
		};
		return;
	}

	const retired = retiredGpuResidentOutputs.get(releasedKey);
	if (!retired) return;
	retiredGpuResidentOutputs.delete(releasedKey);
	disposeManagedGpuResidentOutput(retired);
}
```

Keep stale/unaccepted result disposal unchanged:

```ts
disposeSelectedHourGpuResidentOutput(result.gpuResidentOutput);
```

- [ ] **Step 4: Run controller tests**

Run:

```powershell
cd viewer
npx vitest run tests/compute/live-selected-hour-controller.test.ts
```

Expected:

- PASS.
- Tests prove stale unaccepted GPU outputs still dispose immediately.
- Tests prove accepted outputs dispose only after release or controller disposal.

---

## Task 2: Forward Scene Release Through The Route Host

**Files:**
- Modify: `viewer/src/lib/compute/liveSelectedHourRouteHost.ts`
- Modify: `viewer/src/lib/compute/liveSelectedHourSurfaceIdentity.ts`
- Modify: `viewer/tests/compute/live-selected-hour-route-host.test.ts`

- [ ] **Step 1: Add controller identity to surface identity**

In `viewer/src/lib/compute/liveSelectedHourSurfaceIdentity.ts`, add `controllerIdentity`:

```ts
export type LiveSelectedHourSurfaceIdentity = {
	controllerIdentity: string;
	requestId: number;
	monthIndex: number;
	hourIndex: number;
	timeIndex: number;
	selectionKey: string;
	pendingRenderUpdateStartedAt: number | undefined;
	acceptedGpuResidentOutput: SelectedHourGpuResidentOutput | null;
};
```

In `viewer/src/lib/compute/liveSelectedHourRouteHost.ts`, when creating published surface snapshots, copy the existing route-host `controllerIdentity` into the scene-facing surface identity:

```ts
surfaceIdentity: {
	...params.surfaceIdentity,
	controllerIdentity: params.controllerIdentity
},
```

Do this for both `createPublishedSurfaceSnapshot` and any exposed scene surface identity that is built directly from controller state.

- [ ] **Step 2: Add failing route-host tests**

In `viewer/tests/compute/live-selected-hour-route-host.test.ts`, extend the controller mock type with:

```ts
releaseAcceptedGpuResidentOutput: Mock<(release: {
	controllerIdentity: string;
	requestId: number;
	monthIndex: number;
	timeIndex: number;
	reason: 'copy-complete' | 'copy-failed' | 'superseded';
}) => void>;
```

When constructing fake controllers, include:

```ts
releaseAcceptedGpuResidentOutput: vi.fn()
```

Also extend the test harness `ControllerRecord` shape so each `factory.records[index]` exposes the same mock:

```ts
type ControllerRecord = {
	requests: LiveSelectedHourControllerRequest[];
	diagnostics: LiveSelectedHourControllerSurfaceDiagnostics[];
	dispose: Mock<() => void>;
	releaseAcceptedGpuResidentOutput: Mock<(release: {
		controllerIdentity: string;
		requestId: number;
		monthIndex: number;
		timeIndex: number;
		reason: 'copy-complete' | 'copy-failed' | 'superseded';
	}) => void>;
};
```

When `createControllerFactory()` pushes a record, include the exact `releaseAcceptedGpuResidentOutput` mock stored on the fake controller.

Add this test after arranging the host with the existing route-host test helper pattern so a base scene surface has been published:

```ts
it('forwards base scene GPU output release to the base controller', async () => {
	const factory = createControllerFactory();
	const host = createLiveSelectedHourRouteHost({ createController: factory.create });

	await publishBaseGpuSurface(host);
	const controllerIdentity = host.getState().baseSceneSurfaceIdentity?.controllerIdentity;
	expect(controllerIdentity).toEqual(expect.any(String));

	host.releaseBaseAcceptedGpuResidentOutput({
		controllerIdentity: controllerIdentity!,
		requestId: 101,
		monthIndex: 7,
		timeIndex: 175,
		reason: 'copy-complete'
	});

	expect(factory.records[0].releaseAcceptedGpuResidentOutput).toHaveBeenCalledWith({
		controllerIdentity: controllerIdentity!,
		requestId: 101,
		monthIndex: 7,
		timeIndex: 175,
		reason: 'copy-complete'
	});
	expect(factory.records[1].releaseAcceptedGpuResidentOutput).not.toHaveBeenCalled();
});
```

Add this test after arranging the host with the existing route-host test helper pattern so a comparison scene surface has been published:

```ts
it('forwards comparison scene GPU output release to the comparison controller', async () => {
	const factory = createControllerFactory();
	const host = createLiveSelectedHourRouteHost({ createController: factory.create });

	await publishComparisonGpuSurface(host);
	const controllerIdentity = host.getState().comparisonSceneSurfaceIdentity?.controllerIdentity;
	expect(controllerIdentity).toEqual(expect.any(String));

	host.releaseComparisonAcceptedGpuResidentOutput({
		controllerIdentity: controllerIdentity!,
		requestId: 202,
		monthIndex: 8,
		timeIndex: 200,
		reason: 'superseded'
	});

	expect(factory.records[1].releaseAcceptedGpuResidentOutput).toHaveBeenCalledWith({
		controllerIdentity: controllerIdentity!,
		requestId: 202,
		monthIndex: 8,
		timeIndex: 200,
		reason: 'superseded'
	});
	expect(factory.records[0].releaseAcceptedGpuResidentOutput).not.toHaveBeenCalled();
});
```

Add this replacement-safety test:

```ts
it('ignores late base releases from a replaced controller identity', async () => {
	const factory = createControllerFactory();
	const host = createLiveSelectedHourRouteHost({ createController: factory.create });

	const staleIdentity = 'stale-controller-identity';
	host.releaseBaseAcceptedGpuResidentOutput({
		controllerIdentity: staleIdentity,
		requestId: 1,
		monthIndex: 0,
		timeIndex: 0,
		reason: 'superseded'
	});

	expect(factory.records[0].releaseAcceptedGpuResidentOutput).not.toHaveBeenCalled();
});
```

If the existing harness makes it cheap to force actual controller replacement, prefer a stronger test: capture `baseSceneSurfaceIdentity.controllerIdentity`, trigger an input change that replaces the base controller, then assert a release with the old identity is ignored and a release with the new identity forwards.

`publishBaseGpuSurface(host)` and `publishComparisonGpuSurface(host)` are not new production helpers. In the test file, implement them as local test helpers using the same `setRouteInputs`, controller `requestSelection`, and surface diagnostics patterns already used by nearby route-host tests. They must leave `host.getState().baseSceneSurfaceIdentity` or `comparisonSceneSurfaceIdentity` populated.

- [ ] **Step 3: Run the failing route-host test**

Run:

```powershell
cd viewer
npx vitest run tests/compute/live-selected-hour-route-host.test.ts
```

Expected:

- FAIL because route-host release methods and controller-identity checks do not exist yet.

- [ ] **Step 4: Add release methods to the route host**

In `viewer/src/lib/compute/liveSelectedHourRouteHost.ts`, import the release type:

```ts
import type { LiveSelectedHourGpuResidentRelease } from '$lib/compute/liveSelectedHourController';
```

Extend `LiveSelectedHourRouteHost`:

```ts
releaseBaseAcceptedGpuResidentOutput(release: LiveSelectedHourGpuResidentRelease): void;
releaseComparisonAcceptedGpuResidentOutput(release: LiveSelectedHourGpuResidentRelease): void;
```

Implement methods in the returned host object:

```ts
releaseBaseAcceptedGpuResidentOutput(release) {
	if (disposed) return;
	if (release.controllerIdentity !== baseControllerIdentity) return;
	baseController.releaseAcceptedGpuResidentOutput(release);
},

releaseComparisonAcceptedGpuResidentOutput(release) {
	if (disposed) return;
	if (release.controllerIdentity !== comparisonControllerIdentity) return;
	comparisonController.releaseAcceptedGpuResidentOutput(release);
},
```

- [ ] **Step 5: Run route-host tests**

Run:

```powershell
cd viewer
npx vitest run tests/compute/live-selected-hour-route-host.test.ts tests/compute/live-selected-hour-controller.test.ts
```

Expected:

- PASS.

---

## Task 3: Emit Scene Release From ComparisonRenderer

**Files:**
- Modify: `viewer/src/lib/components/scene/UTCIPointCloud.svelte`
- Modify: `viewer/src/lib/components/scene/ComparisonRenderer.svelte`
- Modify: `viewer/tests/scene/utciComputeBufferRenderBridge.test.ts` only if the implementation needs a helper-level edge-case test.

- [ ] **Step 1: Add controller identity to UTCIPointCloud release payloads**

In `viewer/src/lib/components/scene/UTCIPointCloud.svelte`, extend the release callback payload:

```ts
export let onAcceptedGpuResidentOutputRelease:
	| ((params: {
			controllerIdentity: string;
			requestId: number;
			monthIndex: number;
			timeIndex: number;
			reason: 'copy-complete' | 'copy-failed' | 'superseded';
	  }) => void | Promise<void>)
	| undefined = undefined;
```

When invoking the release callback, read the identity from `liveSelectedHourSurfaceIdentity?.controllerIdentity`. If it is missing, do not emit a normal release; mark the in-flight copy as superseded and return so a missing identity cannot release the wrong controller.

- [ ] **Step 2: Add the ComparisonRenderer release prop and helper**

In `viewer/src/lib/components/scene/ComparisonRenderer.svelte`, add a prop next to `onUtciSurfaceDiagnostics`:

```ts
export let onAcceptedGpuResidentOutputRelease:
	| ((params: {
			controllerIdentity: string;
			requestId: number;
			monthIndex: number;
			timeIndex: number;
			reason: 'copy-complete' | 'copy-failed' | 'superseded';
	  }) => void | Promise<void>)
	| undefined = undefined;
```

Add a helper near the existing diagnostics callback helper:

```ts
function invokeAcceptedGpuResidentOutputRelease(params: {
	controllerIdentity: string;
	requestId: number;
	monthIndex: number;
	timeIndex: number;
	reason: 'copy-complete' | 'copy-failed' | 'superseded';
}): void {
	invokeDiagnosticsCallbackSafely(
		onAcceptedGpuResidentOutputRelease,
		params,
		'ComparisonRenderer onAcceptedGpuResidentOutputRelease'
	);
}
```

- [ ] **Step 3: Notify release exactly once per accepted comparison output**

In `syncAcceptedGpuResidentSurface`, create the release notifier and supersede helper before calling `copyComputeBufferIntoRenderOwnedStorage`. Pass both into `copyComputeBufferIntoRenderOwnedStorage` so the copy helper can notify before superseded returns and `copy-complete`, while the outer `catch` can notify `superseded` or `copy-failed`. Do not define the notifier only inside `copyComputeBufferIntoRenderOwnedStorage`; the catch block also needs access to it.

```ts
const controllerIdentity = liveSelectedHourSurfaceIdentity?.controllerIdentity;
if (!controllerIdentity) {
	return;
}

let releaseNotified = false;
const notifyAcceptedOutputRelease = (
	reason: 'copy-complete' | 'copy-failed' | 'superseded'
): void => {
	if (releaseNotified) return;
	releaseNotified = true;
	invokeAcceptedGpuResidentOutputRelease({
		controllerIdentity,
		requestId: acceptedOutput.requestId,
		monthIndex: acceptedOutput.monthIndex,
		timeIndex: acceptedOutput.timeIndex,
		reason
	});
};

const isSuperseded = () =>
	copyRunToken !== gpuResidentCopyRunToken ||
	activeGpuResidentSyncKey !== syncKey ||
	acceptedGpuResidentOutput?.requestId !== acceptedOutput.requestId;
```

If `liveSelectedHourSurfaceIdentity?.controllerIdentity` is missing, return before starting the copy. Do not emit a release with an empty controller identity.

Extend the `copyComputeBufferIntoRenderOwnedStorage` params:

```ts
notifyAcceptedOutputRelease: (reason: 'copy-complete' | 'copy-failed' | 'superseded') => void;
isSuperseded: () => boolean;
```

Call it:

```ts
notifyAcceptedOutputRelease('superseded');
```

immediately before each `return` in superseded-before-publication branches, including the existing branch after `waitForRenderStorageBuffer`. Pass `isSuperseded` into `waitForRenderStorageBuffer` and `copyComputeBufferToRenderStorage`.

Call:

```ts
notifyAcceptedOutputRelease('copy-complete');
```

immediately before publishing `gpuResidentCopyStatus: 'complete'`.

In the `catch` block in `syncAcceptedGpuResidentSurface`, use the same `isSuperseded()` guard and call:

```ts
notifyAcceptedOutputRelease('superseded');
```

for superseded errors and:

```ts
notifyAcceptedOutputRelease('copy-failed');
```

for active copy failures.

- [ ] **Step 4: Run scene and build checks**

Run:

```powershell
cd viewer
npx vitest run tests/scene/utciComputeBufferRenderBridge.test.ts tests/scene/utci-surface-sync.test.ts
npm run build
```

Expected:

- PASS.
- Existing Svelte warnings may remain; no new build failure.

---

## Task 4: Wire Release Callbacks In Main And Debug Routes

**Files:**
- Modify: `viewer/src/routes/+page.svelte`
- Modify: `viewer/src/routes/debug/+page.svelte`
- Modify: `viewer/tests/e2e/main-route-manual-diagnostics.spec.ts`
- Modify: `viewer/tests/e2e/debug-route-shared-host-diagnostics.spec.ts`

- [ ] **Step 1: Add route release handlers**

In `viewer/src/routes/+page.svelte`, add:

```ts
function handleBaseAcceptedGpuResidentOutputRelease(params: {
	controllerIdentity: string;
	requestId: number;
	monthIndex: number;
	timeIndex: number;
	reason: 'copy-complete' | 'copy-failed' | 'superseded';
}): void {
	liveRouteHost.releaseBaseAcceptedGpuResidentOutput(params);
}

function handleComparisonAcceptedGpuResidentOutputRelease(params: {
	controllerIdentity: string;
	requestId: number;
	monthIndex: number;
	timeIndex: number;
	reason: 'copy-complete' | 'copy-failed' | 'superseded';
}): void {
	liveRouteHost.releaseComparisonAcceptedGpuResidentOutput(params);
}
```

Wire them:

```svelte
<UTCIPointCloud
	...
	onAcceptedGpuResidentOutputRelease={handleBaseAcceptedGpuResidentOutputRelease}
/>
```

and:

```svelte
<ComparisonRenderer
	...
	onAcceptedGpuResidentOutputRelease={handleComparisonAcceptedGpuResidentOutputRelease}
/>
```

- [ ] **Step 2: Wire debug shared-host base release**

In `viewer/src/routes/debug/+page.svelte`, add:

```ts
function handleDebugSharedBaseAcceptedGpuResidentOutputRelease(params: {
	controllerIdentity: string;
	requestId: number;
	monthIndex: number;
	timeIndex: number;
	reason: 'copy-complete' | 'copy-failed' | 'superseded';
}): void {
	debugSharedRouteHost.releaseBaseAcceptedGpuResidentOutput(params);
}
```

Update the `UTCIPointCloud` prop:

```svelte
onAcceptedGpuResidentOutputRelease={useDebugSharedSelectedHourHost
	? handleDebugSharedBaseAcceptedGpuResidentOutputRelease
	: handleAcceptedGpuResidentOutputRelease}
```

Preserve the existing legacy debug `handleAcceptedGpuResidentOutputRelease` behavior.

- [ ] **Step 3: Run route build checks**

Run:

```powershell
cd viewer
npm run build
npx vitest run tests/compute/live-selected-hour-route-host.test.ts tests/compute/live-selected-hour-controller.test.ts
```

Expected:

- PASS.

- [ ] **Step 4: Run browser proof after wiring**

Run:

```powershell
cd viewer
npm run test:e2e:selected-hour
```

Expected:

- PASS.
- `/` and `/debug` still prove `compute-buffer-selected-hour` visible path where expected.
- Debug parity tests still prove legacy `.bin` behavior only on debug parity path.

---

## Task 5: Expand The Required Quality Script

**Files:**
- Modify: `viewer/package.json`

- [ ] **Step 1: Update `test:quality:selected-hour`**

In `viewer/package.json`, extend `test:quality:selected-hour` to include the helper tests that are now part of the required selected-hour proof surface:

```json
"test:quality:selected-hour": "vitest run tests/compute/live-selected-hour-controller.test.ts tests/compute/live-selected-hour-route-host.test.ts tests/compute/live-selected-hour-route-projection.test.ts tests/compute/live-selected-hour-render-context.test.ts tests/compute/live-selected-hour-session.test.ts tests/compute/onDemandDiagnostics.test.ts tests/compute/selectedHourOutputHandle.test.ts tests/scene/utci-surface-sync.test.ts tests/scene/utciComputeBufferRenderBridge.test.ts tests/components/canvasInteractionController.test.ts tests/diagnostics/selectedHourRuntimeContract.test.ts tests/diagnostics/main-route-utci-diagnostics.test.ts tests/debug/debug-webgpu-utci-diagnostics.test.ts tests/debug/debug-on-demand-prototype-diagnostics.test.ts tests/debug/debug-selected-hour-legacy-host.test.ts tests/routes/main-route-debug-boundary-source-lock.test.ts"
```

- [ ] **Step 2: Run the expanded script**

Run:

```powershell
cd viewer
npm run test:quality:selected-hour
```

Expected:

- PASS.
- This command now covers controller ownership, route-host forwarding, diagnostics contract helpers, debug legacy host, and scene buffer bridge helpers.

---

## Task 6: Add Main Route Comparison Readback Browser Proof

**Files:**
- Modify: `viewer/tests/e2e/main-route-manual-diagnostics.spec.ts`

- [ ] **Step 1: Add a comparison-mode Playwright test**

Add this test to `viewer/tests/e2e/main-route-manual-diagnostics.spec.ts`:

```ts
test('keeps visible GPU path strong while comparison readbacks are accounted separately', async ({ page }) => {
	await page.goto(
		'/?analysis=Ben-Gurion%2F20250815_grid_2m_fullday&utciRender=auto&utciRenderDiagnostics=1'
	);
	await waitForSelectedHourPublication(page, {
		expectedSelectionKey: 'Ben-Gurion/20250815_grid_2m_fullday|7|0'
	});

	await page.getByRole('button', { name: /browse variants/i }).click();
	await page.getByRole('button', { name: /existing tree cover/i }).click();

	await expect
		.poll(
			async () => {
				const value = await page.evaluate(() => window.__utciRenderDiagnostics__ ?? null);
				if (
					value?.selectedHourRuntimeContract?.strongVisibleGpuPath === true &&
					value?.selectedHourRuntimeContract?.readbackReasonCounts?.comparison >= 1
				) {
					return value;
				}
				return null;
			},
			{
				timeout: 15000,
				message:
					'Expected main route comparison mode to keep strong visible GPU path and account comparison readbacks separately'
			}
		)
		.not.toBeNull();

	const value = await page.evaluate(() => window.__utciRenderDiagnostics__ ?? null);
	expect(value?.selectedHourRuntimeContract).toMatchObject({
		route: 'main',
		selectedHourEngine: 'shared-host',
		readbackInstrumentation: 'instrumented',
		visibleSelectedHourReadbackCount: 0,
		strongVisibleGpuPath: true
	});
	expect(value?.selectedHourRuntimeContract?.readbackReasons).toContain('comparison');
	expect(value?.selectedHourRuntimeContract?.readbackReasonCounts?.comparison).toBeGreaterThan(0);
	expect(value?.selectedHourRuntimeContract?.visibleRenderPathAvoidsCpuReadback).toBe(true);
});
```

Use the same button-driven comparison activation that is already proven in `viewer/tests/e2e/webgpu-on-demand-prototype.spec.ts`: click `Browse variants`, then `Existing Tree Cover`. Do not add a long outer timeout; keep the short poll and use the file's existing `readUtciRenderDiagnostics` pattern in a `.catch(...)` wrapper if the poll needs diagnostic dumping.

- [ ] **Step 2: Run the new browser test**

Run:

```powershell
cd viewer
npx playwright test tests/e2e/main-route-manual-diagnostics.spec.ts --project=chromium --workers=1 --reporter=list --timeout=30000
```

Expected:

- PASS.
- The test uses the real main-route comparison UI trigger and does not require product behavior changes.

- [ ] **Step 3: Run all selected-hour browser proof**

Run:

```powershell
cd viewer
npm run test:e2e:selected-hour
```

Expected:

- PASS.

---

## Task 7: Final Verification And Review Stop

**Files:**
- Inspect all changed files.

- [ ] **Step 1: Run final focused units**

Run:

```powershell
cd viewer
npm run test:quality:selected-hour
```

Expected:

- PASS.

- [ ] **Step 2: Run final WebGPU route probes**

Run:

```powershell
cd viewer
npm run test:e2e:selected-hour
```

Expected:

- PASS.

- [ ] **Step 3: Run build and static baseline**

Run:

```powershell
cd viewer
npm run build
npm run check
```

Expected:

- `npm run build` PASS.
- `npm run check` may still FAIL with inherited repo-wide static debt.
- Report exact current error/warning count and any errors in files touched by this plan.

- [ ] **Step 4: Run whitespace guard**

Run from repo root:

```powershell
git diff --check
```

Expected:

- PASS or only preexisting/generated line-ending warnings outside source/docs.

- [ ] **Step 5: Request review agents before calling the implementation complete**

Runtime ownership reviewer prompt:

```text
Review the scene-acknowledged selected-hour GPU output implementation in D:\Projects\Nur\Shade\fast-utci. Do not edit files. Focus on whether shared-host accepted GPU outputs are retired until scene release, whether stale/unaccepted outputs still dispose immediately, whether route host forwards base/comparison releases to the right controller, and whether / and /debug visible GPU path diagnostics remain honest. Return findings first with file/line evidence.
```

Svelte/scene reviewer prompt:

```text
Review the Svelte scene integration for selected-hour GPU output release in D:\Projects\Nur\Shade\fast-utci. Do not edit files. Focus on UTCIPointCloud and ComparisonRenderer release callback behavior, exactly-once release notification for copy-complete/copy-failed/superseded, preservation of scene/WebGPU lifecycle ownership, and whether the implementation avoids broad route decomposition. Return findings first with file/line evidence.
```

- [ ] **Step 6: Stop for human review**

Report:

- changed files
- verification commands and results
- whether `npm run check` still fails and why
- review-agent findings
- remaining next-step candidates

Do not commit. Do not continue into debug-route decomposition, scene sync extraction, static cleanup, or 0.5m performance work without human approval.

---

## Final Verification Commands

Run these before claiming implementation complete:

```powershell
cd viewer
npm run test:quality:selected-hour
npx playwright test tests/e2e/main-route-manual-diagnostics.spec.ts --project=chromium --workers=1 --reporter=list --timeout=30000
npm run test:e2e:selected-hour
npm run build
npm run check
cd ..
git diff --check
```

Expected completion state:

- Shared-host accepted GPU outputs are not destroyed before scene release.
- Retired superseded outputs are destroyed once the scene releases them.
- Current accepted outputs released with `copy-complete` are destroyed when superseded or when the controller/host is disposed.
- Main route wires release callbacks for base and comparison scene outputs.
- Debug shared-host base route wires release callbacks without breaking legacy debug parity release behavior.
- Main route comparison readback reasons are browser-proven separate from visible selected-hour readbacks.
- `test:quality:selected-hour` covers the helper tests introduced by the selected-hour quality baseline.
- `npm run check` status is honestly reported and not confused with focused selected-hour correctness.

---

## Next-Agent Execution Handoff

Use this plan with `superpowers:subagent-driven-development` and `superpowers:verification-before-completion`.

Execution requirements:

- Work in `D:\Projects\Nur\Shade\fast-utci`.
- Do not create commits.
- Do not create git worktrees.
- Preserve unrelated dirty files.
- Execute task-by-task.
- Use fresh verification after every task.
- Use review agents before claiming completion.
- If a Playwright probe fails, dump diagnostics and use `superpowers:systematic-debugging`; do not patch by guesswork.
