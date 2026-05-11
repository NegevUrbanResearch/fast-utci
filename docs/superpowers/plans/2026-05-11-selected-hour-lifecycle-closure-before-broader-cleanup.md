# Selected-Hour Lifecycle Closure Before Broader Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Workflow override:** Do **not** create commits. Do **not** create git worktrees. Preserve unrelated dirty files. This is a closure pass before broader cleanup, not the broader cleanup itself.

**Goal:** Close the selected-hour GPU lifecycle contract gaps that would make route decomposition, scene sync extraction, and folder reorganization risky.

**Architecture:** Keep the current `liveSelectedHourController` / `liveSelectedHourRouteHost` / scene component architecture intact. Add only the missing ownership-contract strength: per-controller-instance release identity, exactly-once scene release notification, stale/reused GPU-handle protection, and focused proof artifacts. Broader route decomposition and scene sync extraction are deferred until this plan declares the lifecycle layer stable.

**Tech Stack:** SvelteKit, Svelte components, Three/Threlte WebGPU renderer, WebGPU UTCI compute, Vitest, Playwright Chromium with `--enable-unsafe-webgpu`, PowerShell on Windows.

---

## Perspective Ensemble Result

### Panel A - Council

- **Runtime ownership lens:** The main risk is not normal route behavior; it is stale lifecycle events crossing a controller replacement boundary. Counter-move: make release identity instance-safe, not only content-derived.
- **Scene lifecycle lens:** The scene is now the authority on render-copy completion, but that authority needs one release event per accepted output. Counter-move: use one shared release notifier in both scene components and test the notifier directly.
- **Test confidence lens:** Current browser tests prove visible GPU path honesty, but not release-event semantics. Counter-move: add unit-level lifecycle tests and keep browser proof focused on route-visible diagnostics.
- **Cleanup sequencing lens:** Route decomposition before closing these contracts would move unresolved lifecycle ambiguity into more files. Counter-move: finish this closure pass, document the stable boundary, then start broader cleanup in a separate plan.

### Tensions

- **Small patch vs stronger abstraction:** A tiny inline patch is faster, but a shared release notifier removes duplicated scene lifecycle logic that already diverged.
- **Unit proof vs browser proof:** Browser proof is closer to user behavior, but release races are easier and cheaper to prove with unit/helper tests.
- **Closure pass vs cleanup appetite:** This plan should not solve route size or folder organization; it should make those future changes safe.

### Panel B - Adversarial

- **Attack target:** The assumption that the previous scene-acknowledged implementation is stable enough to build broader cleanup on top of.
- **Identity collision vector:** A replaced controller can reuse the same content-derived `controllerIdentity`, and request ids restart at `1`; late releases could hit the wrong controller. Probe: same-content replacement test must prove old instance release is ignored.
- **Duplicate release vector:** `UTCIPointCloud` can notify through the copy helper and again from the outer `catch`. Probe: shared notifier test plus scene code migration must prove exactly-once notification.
- **Handle reuse vector:** Stale rejected outputs are destroyed immediately even if their GPU handle is shared with a still-owned accepted output. Probe: stale shared-handle test must fail before the guard and pass after it.

### Recommendation

Do a four-task closure pass: instance-safe release identity, shared exactly-once scene release notifier, stale/reused handle guard, final verification/status update. Do not start route decomposition, scene sync extraction, or folder reorganization in this plan.

---

## Current State Snapshot

This plan starts after:

- `7434a2a chore(webgpu): add scene-acknowledged selected-hour GPU release`
- Focused verification from review:
  - `cd viewer; npm run test:quality:selected-hour` passed with 16 files / 147 tests.
  - `cd viewer; npm run test:e2e:selected-hour` passed with 13 Chromium tests.

Known closure findings from review:

- Route-host release forwarding checks `controllerIdentity`, but that value is content-derived from analysis/model/device and can be reused by a new controller instance.
- `UTCIPointCloud.svelte` guards release notification inside the copy helper, but its outer catch can emit a second release event. `ComparisonRenderer.svelte` already has the safer single-notifier pattern.
- Stale unaccepted GPU outputs are destroyed immediately. That is correct for fresh stale outputs, but unsafe if a stale result ever returns a GPU handle that is also owned by the current accepted output.
- There is no durable status note saying the selected-hour lifecycle layer is closed and broader cleanup may begin.

## Non-Goals

- Do not decompose `viewer/src/routes/+page.svelte`.
- Do not decompose `viewer/src/routes/debug/+page.svelte`.
- Do not extract the full scene sync state machine.
- Do not reorganize folders.
- Do not optimize 0.5m performance.
- Do not remove `dataTexture`, `.bin`, Python comparison, parity, collect, or debug fallback paths.
- Do not loosen `strongVisibleGpuPath`.
- Do not create commits.
- Do not create git worktrees.

## Quality Gates

Stop and report findings before continuing if any of these happen:

- A stale release from an old controller instance can release a current controller output.
- A scene release payload lacks an instance-safe controller id.
- `UTCIPointCloud` or `ComparisonRenderer` can emit more than one release event for the same accepted output.
- A stale unaccepted result can destroy a GPU buffer still owned by the current accepted output.
- `strongVisibleGpuPath` becomes true without request id, selection key, same-device, and zero visible-readback proof.
- `/debug` parity mode loses legacy `.bin` behavior.
- `test:quality:selected-hour` stops covering the ownership helpers.
- A browser wait relies on the outer Playwright timeout instead of a short diagnostic wait.

## File Structure

### Modify

- `viewer/src/lib/compute/liveSelectedHourSurfaceIdentity.ts`
  - Add `controllerInstanceId: number`.
- `viewer/src/lib/compute/liveSelectedHourController.ts`
  - Add `controllerInstanceId` to release type.
  - Guard stale GPU disposal when the stale output shares the current accepted GPU buffer.
- `viewer/src/lib/compute/liveSelectedHourRouteHost.ts`
  - Track per-slot controller instance ids.
  - Attach instance ids to scene-facing surface identities.
  - Reject base/comparison releases unless both `controllerIdentity` and `controllerInstanceId` match.
- `viewer/src/lib/components/scene/UTCIPointCloud.svelte`
  - Use a shared exactly-once release notifier.
- `viewer/src/lib/components/scene/ComparisonRenderer.svelte`
  - Use the same shared release notifier.
- `viewer/src/routes/+page.svelte`
  - Accept and forward `controllerInstanceId` in base/comparison release handlers.
- `viewer/src/routes/debug/+page.svelte`
  - Accept and forward `controllerInstanceId` for shared-host releases while preserving the legacy debug release path.
- `viewer/package.json`
  - Include the new scene release notifier test in `test:quality:selected-hour`.

### Create

- `viewer/src/lib/components/scene/acceptedGpuResidentOutputRelease.ts`
  - Shared release payload type and exactly-once notifier helper.
- `viewer/tests/scene/acceptedGpuResidentOutputRelease.test.ts`
  - Direct tests for exactly-once release behavior and missing identity suppression.
- `viewer/tests/scene/acceptedGpuResidentOutputReleaseCallsites.test.ts`
  - Source-level regression test that scene components use the shared notifier instead of direct duplicate-prone callback helpers.
- `docs/superpowers/plans/2026-05-11-selected-hour-lifecycle-closure-before-broader-cleanup-results.md`
  - Final verification/status note created only after Tasks 1-4 pass.

### Test

- `viewer/tests/compute/live-selected-hour-route-host.test.ts`
- `viewer/tests/compute/live-selected-hour-controller.test.ts`
- `viewer/tests/scene/acceptedGpuResidentOutputRelease.test.ts`
- `viewer/tests/scene/acceptedGpuResidentOutputReleaseCallsites.test.ts`
- `viewer/tests/e2e/main-route-manual-diagnostics.spec.ts`
- Existing selected-hour quality and browser scripts.

---

## Task 0: Baseline And Dirty-State Record

**Files:**
- Inspect only.

- [ ] **Step 1: Record current state**

Run from repo root:

```powershell
git status --short
git log --oneline -6
```

Expected:

- Do not create commits.
- Do not create git worktrees.
- Preserve unrelated dirty files.
- Recent history includes `7434a2a chore(webgpu): add scene-acknowledged selected-hour GPU release`.

- [ ] **Step 2: Run focused baseline units**

Run:

```powershell
cd viewer
npm run test:quality:selected-hour
```

Expected:

- PASS with the current selected-hour quality suite.
- If this fails, use `superpowers:systematic-debugging` before editing.

---

## Task 1: Make Scene Releases Controller-Instance Safe

**Files:**
- Modify: `viewer/src/lib/compute/liveSelectedHourSurfaceIdentity.ts`
- Modify: `viewer/src/lib/compute/liveSelectedHourController.ts`
- Modify: `viewer/src/lib/compute/liveSelectedHourRouteHost.ts`
- Modify: `viewer/src/routes/+page.svelte`
- Modify: `viewer/src/routes/debug/+page.svelte`
- Modify: `viewer/tests/compute/live-selected-hour-route-host.test.ts`

- [ ] **Step 1: Write the failing same-identity replacement test**

In `viewer/tests/compute/live-selected-hour-route-host.test.ts`, add this test near the existing release-forwarding tests:

```ts
it('ignores stale base releases from an old controller instance with the same content identity', async () => {
	const factory = createControllerFactory();
	const host = createLiveSelectedHourRouteHost(makeHostDeps(factory));
	const sharedBaseModel = {} as Group;

	host.setRouteInputs(
		makeComparisonInputs({
			utciSurfaceBackend: 'gpuNative',
			baseModel: sharedBaseModel
		})
	);
	await host.flush();

	const staleBaseIdentity = host.getState().baseSceneSurfaceIdentity?.controllerIdentity;
	const staleBaseInstanceId = host.getState().baseSceneSurfaceIdentity?.controllerInstanceId;
	expect(staleBaseIdentity).toBe(factory.records[0].requests[0]?.sessionKey);
	expect(staleBaseInstanceId).toEqual(expect.any(Number));

	host.setRouteInputs(
		makeComparisonInputs({
			enabled: false,
			utciSurfaceBackend: 'gpuNative',
			baseModel: sharedBaseModel
		})
	);
	await host.flush();

	host.setRouteInputs(
		makeComparisonInputs({
			utciSurfaceBackend: 'gpuNative',
			baseModel: sharedBaseModel
		})
	);
	await host.flush();

	const currentBaseIdentity = host.getState().baseSceneSurfaceIdentity?.controllerIdentity;
	const currentBaseInstanceId = host.getState().baseSceneSurfaceIdentity?.controllerInstanceId;
	expect(currentBaseIdentity).toBe(staleBaseIdentity);
	expect(currentBaseInstanceId).not.toBe(staleBaseInstanceId);

	host.releaseBaseAcceptedGpuResidentOutput({
		controllerIdentity: staleBaseIdentity ?? 'missing-stale-base-controller',
		controllerInstanceId: staleBaseInstanceId ?? -1,
		requestId: 1,
		monthIndex: 7,
		timeIndex: 180,
		reason: 'superseded'
	});

	const currentBaseRecord = factory.records[factory.records.length - 1];
	expect(currentBaseRecord?.releases).toEqual([]);

	const currentRelease = {
		controllerIdentity: currentBaseIdentity ?? 'missing-current-base-controller',
		controllerInstanceId: currentBaseInstanceId ?? -1,
		requestId: 1,
		monthIndex: 7,
		timeIndex: 180,
		reason: 'copy-complete' as const
	};
	host.releaseBaseAcceptedGpuResidentOutput(currentRelease);

	expect(currentBaseRecord?.releases).toEqual([currentRelease]);
});
```

Also add the comparison-slot equivalent:

```ts
it('ignores stale comparison releases from an old controller instance with the same content identity', async () => {
	const factory = createControllerFactory();
	const host = createLiveSelectedHourRouteHost(makeHostDeps(factory));
	const sharedComparisonModel = {} as Group;

	host.setRouteInputs(
		makeComparisonInputs({
			utciSurfaceBackend: 'gpuNative',
			comparison: {
				model: sharedComparisonModel
			}
		})
	);
	await host.flush();

	const staleComparisonIdentity =
		host.getState().comparisonSceneSurfaceIdentity?.controllerIdentity;
	const staleComparisonInstanceId =
		host.getState().comparisonSceneSurfaceIdentity?.controllerInstanceId;
	expect(staleComparisonIdentity).toBe(factory.records[1].requests[0]?.sessionKey);
	expect(staleComparisonInstanceId).toEqual(expect.any(Number));

	host.setRouteInputs(
		makeComparisonInputs({
			enabled: false,
			utciSurfaceBackend: 'gpuNative',
			comparison: {
				model: sharedComparisonModel
			}
		})
	);
	await host.flush();

	host.setRouteInputs(
		makeComparisonInputs({
			utciSurfaceBackend: 'gpuNative',
			comparison: {
				model: sharedComparisonModel
			}
		})
	);
	await host.flush();

	const currentComparisonIdentity =
		host.getState().comparisonSceneSurfaceIdentity?.controllerIdentity;
	const currentComparisonInstanceId =
		host.getState().comparisonSceneSurfaceIdentity?.controllerInstanceId;
	expect(currentComparisonIdentity).toBe(staleComparisonIdentity);
	expect(currentComparisonInstanceId).not.toBe(staleComparisonInstanceId);

	host.releaseComparisonAcceptedGpuResidentOutput({
		controllerIdentity: staleComparisonIdentity ?? 'missing-stale-comparison-controller',
		controllerInstanceId: staleComparisonInstanceId ?? -1,
		requestId: 1,
		monthIndex: 7,
		timeIndex: 180,
		reason: 'superseded'
	});

	const currentComparisonRecord = factory.records[factory.records.length - 1];
	expect(currentComparisonRecord?.releases).toEqual([]);

	const currentRelease = {
		controllerIdentity: currentComparisonIdentity ?? 'missing-current-comparison-controller',
		controllerInstanceId: currentComparisonInstanceId ?? -1,
		requestId: 1,
		monthIndex: 7,
		timeIndex: 180,
		reason: 'copy-complete' as const
	};
	host.releaseComparisonAcceptedGpuResidentOutput(currentRelease);

	expect(currentComparisonRecord?.releases).toEqual([currentRelease]);
});
```

Add the same assertion style to the existing test named `publishes controller identity on base and comparison scene surface identities`:

```ts
expect(host.getState().baseSceneSurfaceIdentity?.controllerInstanceId).toEqual(expect.any(Number));
expect(host.getState().comparisonSceneSurfaceIdentity?.controllerInstanceId).toEqual(expect.any(Number));
expect(host.getState().baseSceneSurfaceIdentity?.controllerInstanceId).not.toBe(
	host.getState().comparisonSceneSurfaceIdentity?.controllerInstanceId
);
```

- [ ] **Step 2: Run the failing route-host test**

Run:

```powershell
cd viewer
npx vitest run tests/compute/live-selected-hour-route-host.test.ts
```

Expected:

- FAIL because `controllerInstanceId` does not exist on surface identities or release payloads.

- [ ] **Step 3: Add instance id to surface and release types**

In `viewer/src/lib/compute/liveSelectedHourSurfaceIdentity.ts`, change the type to:

```ts
export type LiveSelectedHourSurfaceIdentity = {
	controllerIdentity: string;
	controllerInstanceId: number;
	requestId: number;
	monthIndex: number;
	hourIndex: number;
	timeIndex: number;
	selectionKey: string;
	pendingRenderUpdateStartedAt: number | undefined;
	acceptedGpuResidentOutput: SelectedHourGpuResidentOutput | null;
};
```

In `viewer/src/lib/compute/liveSelectedHourController.ts`, extend the release interface:

```ts
export interface LiveSelectedHourGpuResidentRelease {
	controllerIdentity: string;
	controllerInstanceId: number;
	requestId: number;
	monthIndex: number;
	timeIndex: number;
	reason: LiveSelectedHourGpuResidentReleaseReason;
}
```

In the same file, update the local `createSurfaceIdentity` helper so the controller-owned default compiles:

```ts
return {
	controllerIdentity: 'controller',
	controllerInstanceId: 0,
	requestId: params.requestId,
	monthIndex: params.monthIndex,
	hourIndex: params.hourIndex,
	timeIndex: params.timeIndex,
	selectionKey: params.selectionKey,
	pendingRenderUpdateStartedAt: params.pendingRenderUpdateStartedAt,
	acceptedGpuResidentOutput: params.acceptedGpuResidentOutput
};
```

Update existing controller tests that call `releaseAcceptedGpuResidentOutput` directly by adding:

```ts
controllerInstanceId: 0,
```

to each release payload.

Update every `LiveSelectedHourSurfaceIdentity` literal in tests so the new required field is explicit. At minimum, inspect and update:

```powershell
cd viewer
rg -n "LiveSelectedHourSurfaceIdentity|liveSelectedHourSurfaceIdentity: \\{|controllerIdentity:" tests src/lib src/routes
```

Known fixtures to update include `viewer/tests/compute/live-selected-hour-route-projection.test.ts` and `viewer/tests/scene/utci-surface-sync.test.ts`. Use stable fixture values such as:

```ts
controllerInstanceId: 0,
```

for controller-local fixtures and unique positive numbers for base/comparison route-host fixtures.

- [ ] **Step 4: Track route-host controller instances**

In `viewer/src/lib/compute/liveSelectedHourRouteHost.ts`, update `attachControllerIdentityToSurfaceIdentity`:

```ts
function attachControllerIdentityToSurfaceIdentity(params: {
	surfaceIdentity: LiveSelectedHourSurfaceIdentity;
	controllerIdentity: string;
	controllerInstanceId: number;
}): LiveSelectedHourSurfaceIdentity {
	return {
		...params.surfaceIdentity,
		controllerIdentity: params.controllerIdentity,
		controllerInstanceId: params.controllerInstanceId
	};
}
```

Near the existing controller generation state, add stable per-slot instance ids:

```ts
let nextControllerInstanceId = 1;
let baseControllerInstanceId = nextControllerInstanceId++;
let comparisonControllerInstanceId = nextControllerInstanceId++;
```

In `replaceController('base')`, after `baseControllerGeneration += 1`, assign:

```ts
baseControllerInstanceId = nextControllerInstanceId++;
```

In the comparison branch, after `comparisonControllerGeneration += 1`, assign:

```ts
comparisonControllerInstanceId = nextControllerInstanceId++;
```

Where base and comparison published scene surface identities are created, pass the instance id:

```ts
surfaceIdentity: attachControllerIdentityToSurfaceIdentity({
	surfaceIdentity: controllerState.surfaceIdentity,
	controllerIdentity: baseControllerIdentity,
	controllerInstanceId: baseControllerInstanceId
})
```

and:

```ts
surfaceIdentity: attachControllerIdentityToSurfaceIdentity({
	surfaceIdentity: controllerState.surfaceIdentity,
	controllerIdentity: comparisonControllerIdentity,
	controllerInstanceId: comparisonControllerInstanceId
})
```

Use the local variable names already present at those call sites; do not introduce broad rewrites.

- [ ] **Step 5: Gate release forwarding by identity and instance**

In `viewer/src/lib/compute/liveSelectedHourRouteHost.ts`, update the release methods:

```ts
releaseBaseAcceptedGpuResidentOutput(release) {
	if (disposed) {
		return;
	}
	if (
		release.controllerIdentity !== baseControllerIdentity ||
		release.controllerInstanceId !== baseControllerInstanceId
	) {
		return;
	}
	baseController.releaseAcceptedGpuResidentOutput(release);
},

releaseComparisonAcceptedGpuResidentOutput(release) {
	if (disposed) {
		return;
	}
	if (
		release.controllerIdentity !== comparisonControllerIdentity ||
		release.controllerInstanceId !== comparisonControllerInstanceId
	) {
		return;
	}
	comparisonController.releaseAcceptedGpuResidentOutput(release);
},
```

- [ ] **Step 6: Forward instance ids from routes**

In `viewer/src/routes/+page.svelte`, add `controllerInstanceId: number` to both release handler parameter types:

```ts
function handleBaseAcceptedGpuResidentOutputRelease(params: {
	controllerIdentity: string;
	controllerInstanceId: number;
	requestId: number;
	monthIndex: number;
	timeIndex: number;
	reason: 'copy-complete' | 'copy-failed' | 'superseded';
}): void {
	liveRouteHost.releaseBaseAcceptedGpuResidentOutput(params);
}
```

and:

```ts
function handleComparisonAcceptedGpuResidentOutputRelease(params: {
	controllerIdentity: string;
	controllerInstanceId: number;
	requestId: number;
	monthIndex: number;
	timeIndex: number;
	reason: 'copy-complete' | 'copy-failed' | 'superseded';
}): void {
	liveRouteHost.releaseComparisonAcceptedGpuResidentOutput(params);
}
```

In `viewer/src/routes/debug/+page.svelte`, add `controllerInstanceId: number` to the shared-host release handler payloads. For the legacy handler, keep the existing legacy controller identity check and ignore `controllerInstanceId`; do not change `releaseAcceptedGpuResidentUtciOutput(...)`.

- [ ] **Step 7: Run route-host and controller tests**

Run:

```powershell
cd viewer
npx vitest run tests/compute/live-selected-hour-route-host.test.ts tests/compute/live-selected-hour-controller.test.ts
```

Expected:

- PASS.
- The same-content controller replacement test proves stale instance releases are ignored.
- Existing controller ownership tests still pass with `controllerInstanceId: 0`.

---

## Task 2: Use A Shared Exactly-Once Scene Release Notifier

**Files:**
- Create: `viewer/src/lib/components/scene/acceptedGpuResidentOutputRelease.ts`
- Create: `viewer/tests/scene/acceptedGpuResidentOutputRelease.test.ts`
- Create: `viewer/tests/scene/acceptedGpuResidentOutputReleaseCallsites.test.ts`
- Modify: `viewer/src/lib/components/scene/UTCIPointCloud.svelte`
- Modify: `viewer/src/lib/components/scene/ComparisonRenderer.svelte`
- Modify: `viewer/package.json`

- [ ] **Step 1: Add failing helper tests**

Create `viewer/tests/scene/acceptedGpuResidentOutputRelease.test.ts`:

```ts
import { describe, expect, it, vi } from 'vitest';

import {
	createAcceptedGpuResidentOutputReleaseNotifier,
	type AcceptedGpuResidentOutputReleaseCallback
} from '$lib/components/scene/acceptedGpuResidentOutputRelease';

describe('acceptedGpuResidentOutputRelease', () => {
	it('notifies release exactly once with controller identity and instance id', () => {
		const callback = vi.fn<AcceptedGpuResidentOutputReleaseCallback>();
		const notify = createAcceptedGpuResidentOutputReleaseNotifier({
			callback,
			componentName: 'test',
			controllerIdentity: 'controller-a',
			controllerInstanceId: 7,
			requestId: 11,
			monthIndex: 2,
			timeIndex: 51
		});

		expect(notify('copy-complete')).toBe(true);
		expect(notify('copy-failed')).toBe(false);
		expect(notify('superseded')).toBe(false);

		expect(callback).toHaveBeenCalledTimes(1);
		expect(callback).toHaveBeenCalledWith({
			controllerIdentity: 'controller-a',
			controllerInstanceId: 7,
			requestId: 11,
			monthIndex: 2,
			timeIndex: 51,
			reason: 'copy-complete'
		});
	});

	it('suppresses release when controller identity is missing', () => {
		const callback = vi.fn<AcceptedGpuResidentOutputReleaseCallback>();
		const notify = createAcceptedGpuResidentOutputReleaseNotifier({
			callback,
			componentName: 'test',
			controllerIdentity: undefined,
			controllerInstanceId: 7,
			requestId: 12,
			monthIndex: 3,
			timeIndex: 60
		});

		expect(notify('superseded')).toBe(false);
		expect(callback).not.toHaveBeenCalled();
	});

	it('suppresses release when controller instance id is missing', () => {
		const callback = vi.fn<AcceptedGpuResidentOutputReleaseCallback>();
		const notify = createAcceptedGpuResidentOutputReleaseNotifier({
			callback,
			componentName: 'test',
			controllerIdentity: 'controller-b',
			controllerInstanceId: undefined,
			requestId: 13,
			monthIndex: 4,
			timeIndex: 72
		});

		expect(notify('copy-failed')).toBe(false);
		expect(callback).not.toHaveBeenCalled();
	});
});
```

- [ ] **Step 2: Run the failing helper test**

Run:

```powershell
cd viewer
npx vitest run tests/scene/acceptedGpuResidentOutputRelease.test.ts
```

Expected:

- FAIL because the helper file does not exist.

- [ ] **Step 3: Add failing scene call-site contract test**

Create `viewer/tests/scene/acceptedGpuResidentOutputReleaseCallsites.test.ts`:

```ts
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

const sceneRoot = resolve(__dirname, '../../src/lib/components/scene');

function readSceneComponent(fileName: string): string {
	return readFileSync(resolve(sceneRoot, fileName), 'utf8');
}

describe('accepted GPU resident output release call sites', () => {
	it.each(['UTCIPointCloud.svelte', 'ComparisonRenderer.svelte'])(
		'%s uses the shared exactly-once release notifier',
		(fileName) => {
			const source = readSceneComponent(fileName);

			expect(source).toContain(
				"createAcceptedGpuResidentOutputReleaseNotifier"
			);
			expect(source).toContain(
				"from '$lib/components/scene/acceptedGpuResidentOutputRelease'"
			);
			expect(source).not.toContain(
				'function invokeAcceptedGpuResidentOutputRelease'
			);
		}
	);
});
```

Run:

```powershell
cd viewer
npx vitest run tests/scene/acceptedGpuResidentOutputReleaseCallsites.test.ts
```

Expected:

- FAIL because both scene components still use local release helpers.
- This test is intentionally source-level. It exists to catch the exact regression found in review: scene call sites bypassing the shared exactly-once notifier.

- [ ] **Step 4: Implement the shared helper**

Create `viewer/src/lib/components/scene/acceptedGpuResidentOutputRelease.ts`:

```ts
import { invokeDiagnosticsCallbackSafely } from '$lib/compute/onDemandDiagnostics';

export type AcceptedGpuResidentOutputReleaseReason =
	| 'copy-complete'
	| 'copy-failed'
	| 'superseded';

export type AcceptedGpuResidentOutputReleasePayload = {
	controllerIdentity: string;
	controllerInstanceId: number;
	requestId: number;
	monthIndex: number;
	timeIndex: number;
	reason: AcceptedGpuResidentOutputReleaseReason;
};

export type AcceptedGpuResidentOutputReleaseCallback = (
	params: AcceptedGpuResidentOutputReleasePayload
) => void | Promise<void>;

export function createAcceptedGpuResidentOutputReleaseNotifier(params: {
	callback: AcceptedGpuResidentOutputReleaseCallback | undefined;
	componentName: string;
	controllerIdentity: string | null | undefined;
	controllerInstanceId: number | null | undefined;
	requestId: number;
	monthIndex: number;
	timeIndex: number;
}): (reason: AcceptedGpuResidentOutputReleaseReason) => boolean {
	let releaseNotified = false;
	return (reason) => {
		if (releaseNotified) {
			return false;
		}
		if (!params.controllerIdentity || params.controllerInstanceId == null) {
			return false;
		}
		releaseNotified = true;
		invokeDiagnosticsCallbackSafely(
			params.callback,
			{
				controllerIdentity: params.controllerIdentity,
				controllerInstanceId: params.controllerInstanceId,
				requestId: params.requestId,
				monthIndex: params.monthIndex,
				timeIndex: params.timeIndex,
				reason
			},
			`${params.componentName} onAcceptedGpuResidentOutputRelease`
		);
		return true;
	};
}
```

- [ ] **Step 5: Migrate `UTCIPointCloud` to the shared notifier**

In `viewer/src/lib/components/scene/UTCIPointCloud.svelte`, import the helper and types:

```ts
import {
	createAcceptedGpuResidentOutputReleaseNotifier,
	type AcceptedGpuResidentOutputReleaseCallback
} from '$lib/components/scene/acceptedGpuResidentOutputRelease';
```

Replace the release prop type with:

```ts
export let onAcceptedGpuResidentOutputRelease:
	| AcceptedGpuResidentOutputReleaseCallback
	| undefined = undefined;
```

Remove the local `invokeAcceptedGpuResidentOutputRelease(...)` helper.

In `syncAcceptedGpuResidentSurface(...)`, create one notifier before the `try` block:

```ts
const notifyAcceptedOutputRelease = createAcceptedGpuResidentOutputReleaseNotifier({
	callback: onAcceptedGpuResidentOutputRelease,
	componentName: 'UTCIPointCloud',
	controllerIdentity: liveSelectedHourSurfaceIdentity?.controllerIdentity,
	controllerInstanceId: liveSelectedHourSurfaceIdentity?.controllerInstanceId,
	requestId: acceptedOutput.requestId,
	monthIndex: acceptedOutput.monthIndex,
	timeIndex: acceptedOutput.timeIndex
});
```

If the identity or instance id is missing, keep the current behavior of resetting sync and returning:

```ts
if (
	!liveSelectedHourSurfaceIdentity?.controllerIdentity ||
	liveSelectedHourSurfaceIdentity.controllerInstanceId == null
) {
	resetAcceptedGpuResidentCopySync({ invalidateActiveRun: true });
	return;
}
```

Pass `notifyAcceptedOutputRelease` into `copyComputeBufferIntoRenderOwnedStorage(...)` instead of building a second local notifier inside the copy helper. Update the copy helper params:

```ts
notifyAcceptedOutputRelease: (
	reason: 'copy-complete' | 'copy-failed' | 'superseded'
) => boolean;
```

Inside the `catch`, replace direct callback emission with:

```ts
if (isSuperseded() || errorMessage.includes('superseded')) {
	notifyAcceptedOutputRelease('superseded');
	return;
}

if (utciSurface) {
	setComputeBufferSurfacePublicationVisibility(utciSurface, false);
}
notifyAcceptedOutputRelease('copy-failed');
setGpuResidentCopyDiagnostics('failed', {
	error: errorMessage,
	requestId: acceptedOutput.requestId
});
```

The success path should continue to call:

```ts
if (!notifyAcceptedOutputRelease('copy-complete')) {
	return;
}
```

- [ ] **Step 6: Migrate `ComparisonRenderer` to the shared notifier**

In `viewer/src/lib/components/scene/ComparisonRenderer.svelte`, import:

```ts
import {
	createAcceptedGpuResidentOutputReleaseNotifier,
	type AcceptedGpuResidentOutputReleaseCallback
} from '$lib/components/scene/acceptedGpuResidentOutputRelease';
```

Replace the release prop type with:

```ts
export let onAcceptedGpuResidentOutputRelease:
	| AcceptedGpuResidentOutputReleaseCallback
	| undefined = undefined;
```

Remove the local `invokeAcceptedGpuResidentOutputRelease(...)` helper and local `releaseNotified` closure. In `syncAcceptedGpuResidentSurface(...)`, create:

```ts
const notifyAcceptedOutputRelease = createAcceptedGpuResidentOutputReleaseNotifier({
	callback: onAcceptedGpuResidentOutputRelease,
	componentName: 'ComparisonRenderer',
	controllerIdentity: liveSelectedHourSurfaceIdentity?.controllerIdentity,
	controllerInstanceId: liveSelectedHourSurfaceIdentity?.controllerInstanceId,
	requestId: acceptedOutput.requestId,
	monthIndex: acceptedOutput.monthIndex,
	timeIndex: acceptedOutput.timeIndex
});
```

Use the same missing-identity guard:

```ts
if (
	!liveSelectedHourSurfaceIdentity?.controllerIdentity ||
	liveSelectedHourSurfaceIdentity.controllerInstanceId == null
) {
	resetAcceptedGpuResidentCopySync({ invalidateActiveRun: true });
	return;
}
```

Keep the existing catch behavior, but call the shared notifier:

```ts
if (isSuperseded() || errorMessage.includes('superseded')) {
	notifyAcceptedOutputRelease('superseded');
	return;
}

if (comparisonUtciMesh) {
	setComputeBufferSurfacePublicationVisibility(comparisonUtciMesh, false);
}
notifyAcceptedOutputRelease('copy-failed');
setGpuResidentCopyDiagnostics('failed', {
	error: errorMessage,
	requestId: acceptedOutput.requestId
});
```

- [ ] **Step 7: Add the helper and call-site tests to the quality script**

In `viewer/package.json`, add `tests/scene/acceptedGpuResidentOutputRelease.test.ts` and `tests/scene/acceptedGpuResidentOutputReleaseCallsites.test.ts` to `test:quality:selected-hour` near the other scene tests:

```json
"test:quality:selected-hour": "vitest run tests/compute/live-selected-hour-controller.test.ts tests/compute/live-selected-hour-route-host.test.ts tests/compute/live-selected-hour-route-projection.test.ts tests/compute/live-selected-hour-render-context.test.ts tests/compute/live-selected-hour-session.test.ts tests/compute/onDemandDiagnostics.test.ts tests/compute/selectedHourOutputHandle.test.ts tests/scene/acceptedGpuResidentOutputRelease.test.ts tests/scene/acceptedGpuResidentOutputReleaseCallsites.test.ts tests/scene/utci-surface-sync.test.ts tests/scene/utciComputeBufferRenderBridge.test.ts tests/components/canvasInteractionController.test.ts tests/diagnostics/selectedHourRuntimeContract.test.ts tests/diagnostics/main-route-utci-diagnostics.test.ts tests/debug/debug-webgpu-utci-diagnostics.test.ts tests/debug/debug-on-demand-prototype-diagnostics.test.ts tests/debug/debug-selected-hour-legacy-host.test.ts tests/routes/main-route-debug-boundary-source-lock.test.ts"
```

- [ ] **Step 8: Run scene and quality tests**

Run:

```powershell
cd viewer
npx vitest run tests/scene/acceptedGpuResidentOutputRelease.test.ts tests/scene/acceptedGpuResidentOutputReleaseCallsites.test.ts tests/scene/utci-surface-sync.test.ts tests/scene/utciComputeBufferRenderBridge.test.ts
npm run test:quality:selected-hour
```

Expected:

- PASS.
- The quality script now includes the exactly-once release helper and scene call-site contract.

---

## Task 3: Guard Stale Reused GPU Handles

**Files:**
- Modify: `viewer/src/lib/compute/liveSelectedHourController.ts`
- Modify: `viewer/tests/compute/live-selected-hour-controller.test.ts`

- [ ] **Step 1: Write the failing stale shared-handle test**

In `viewer/tests/compute/live-selected-hour-controller.test.ts`, add this test after `does not let a reused GPU output object inherit releasable state across controller requests`:

```ts
it('does not destroy a stale rejected GPU output when it shares the current accepted output handle', async () => {
	const destroy = vi.fn();
	const sharedHandle = {
		buffer: { destroy } as unknown as GPUBuffer,
		byteLength: 4,
		requestId: 31,
		timeIndex: 31,
		source: 'webgpu-on-demand-snapshot',
		disposed: false,
		dispose() {
			if (sharedHandle.disposed) return;
			sharedHandle.disposed = true;
			destroy();
		}
	};
	const currentOutput = {
		gpuOutputHandle: sharedHandle
	} as unknown as SelectedHourGpuResidentOutput['output'];
	const staleOutput = {
		gpuOutputHandle: sharedHandle
	} as unknown as SelectedHourGpuResidentOutput['output'];
	const first = deferred<SelectedHourLiveResult>();
	const second = deferred<SelectedHourLiveResult>();
	const sessionMock = createSessionMock([
		async () => first.promise,
		async () => second.promise
	]);
	const controller = createLiveSelectedHourController({
		prepareSession: vi.fn(async () => sessionMock.session)
	});

	const firstRequest = controller.requestSelection(createRequestParams(30));
	await vi.waitFor(() => {
		expect(sessionMock.runSelectedHour).toHaveBeenCalledTimes(1);
	});
	const secondRequest = controller.requestSelection(createRequestParams(31));
	await vi.waitFor(() => {
		expect(sessionMock.runSelectedHour).toHaveBeenCalledTimes(2);
	});

	second.resolve(
		createLiveResult({
			requestId: 31,
			timeIndex: 31,
			analysis: createSelectionAnalysis('current-shared-output', [21, 23]),
			gpuResidentOutput: {
				...createGpuResidentOutput(31, 31).accepted,
				output: currentOutput,
				gpuOutputHandle: sharedHandle
			},
			renderTransport: 'compute-buffer-selected-hour',
			sameDeviceForComputeAndRender: true
		})
	);
	await expect(secondRequest).resolves.toMatchObject({ accepted: true });
	const currentRequestId = getCurrentSurfaceRequestId(controller);

	first.resolve(
		createLiveResult({
			requestId: 30,
			timeIndex: 30,
			analysis: createSelectionAnalysis('stale-shared-output', [17, 19]),
			gpuResidentOutput: {
				...createGpuResidentOutput(30, 30).accepted,
				output: staleOutput,
				gpuOutputHandle: sharedHandle
			},
			renderTransport: 'compute-buffer-selected-hour',
			sameDeviceForComputeAndRender: true
		})
	);
	await expect(firstRequest).resolves.toMatchObject({ accepted: false, reason: 'stale' });

	expect(destroy).not.toHaveBeenCalled();

	controller.releaseAcceptedGpuResidentOutput({
		controllerIdentity: 'controller',
		controllerInstanceId: 0,
		requestId: currentRequestId,
		monthIndex: 0,
		timeIndex: 31,
		reason: 'copy-complete'
	});
	await controller.requestSelection(createRequestParams(32));

	expect(destroy).toHaveBeenCalledTimes(1);
});
```

- [ ] **Step 2: Run the failing controller test**

Run:

```powershell
cd viewer
npx vitest run tests/compute/live-selected-hour-controller.test.ts
```

Expected:

- FAIL because the stale rejected result destroys the shared output immediately.

- [ ] **Step 3: Add a safe stale disposal helper**

In `viewer/src/lib/compute/liveSelectedHourController.ts`, add these helpers inside `createLiveSelectedHourController(...)` near the existing GPU output helpers:

```ts
function getGpuResidentOwnershipHandle(
	output: SelectedHourGpuResidentOutput | null
): unknown {
	return (
		output?.gpuOutputHandle ??
		output?.output.gpuOutputHandle ??
		output?.output.gpuBuffer ??
		output?.output ??
		null
	);
}

function gpuResidentOutputsShareOwnership(
	left: SelectedHourGpuResidentOutput | null,
	right: SelectedHourGpuResidentOutput | null
): boolean {
	const leftHandle = getGpuResidentOwnershipHandle(left);
	const rightHandle = getGpuResidentOwnershipHandle(right);
	return leftHandle != null && leftHandle === rightHandle;
}

function disposeStaleGpuResidentOutput(output: SelectedHourGpuResidentOutput | null): void {
	if (!output) return;
	if (gpuResidentOutputsShareOwnership(acceptedGpuResidentOutputEntry?.value ?? null, output)) {
		return;
	}
	for (const retired of retiredGpuResidentOutputs.values()) {
		if (gpuResidentOutputsShareOwnership(retired.value, output)) {
			return;
		}
	}
	disposeSelectedHourGpuResidentOutput(output);
}
```

Replace the stale-result disposal in `requestSelection(...)`:

```ts
disposeSelectedHourGpuResidentOutput(result.gpuResidentOutput);
```

with:

```ts
disposeStaleGpuResidentOutput(result.gpuResidentOutput);
```

Do not change disposal for controller reset/dispose. Those paths must still destroy current and retired outputs.

- [ ] **Step 4: Run controller tests**

Run:

```powershell
cd viewer
npx vitest run tests/compute/live-selected-hour-controller.test.ts
```

Expected:

- PASS.
- Existing test `reuses the session for repeated requests and rejects stale results when a newer request wins` still proves fresh stale outputs are destroyed immediately.
- New test proves shared stale handles are not destroyed while current ownership remains active.

---

## Task 4: Final Verification, Review Agents, And Lifecycle Closure Note

**Files:**
- Create: `docs/superpowers/plans/2026-05-11-selected-hour-lifecycle-closure-before-broader-cleanup-results.md`
- Inspect changed files.

- [ ] **Step 1: Run final selected-hour quality**

Run:

```powershell
cd viewer
npm run test:quality:selected-hour
```

Expected:

- PASS.
- Report total files/tests.

- [ ] **Step 2: Run final selected-hour browser proof**

Run:

```powershell
cd viewer
npm run test:e2e:selected-hour
```

Expected:

- PASS.
- Main route still proves `compute-buffer-selected-hour`, `visibleSelectedHourReadbackCount: 0`, `strongVisibleGpuPath: true`, and comparison readbacks counted separately.
- Debug route still proves shared-host selected-hour behavior and debug parity `.bin` scoping.

- [ ] **Step 3: Run build and static baseline**

Run:

```powershell
cd viewer
npm run build
npm run check
```

Expected:

- `npm run build` PASS.
- `npm run check` may still fail from inherited static debt. Report exact touched-file errors separately from inherited errors.

- [ ] **Step 4: Run whitespace guard**

Run from repo root:

```powershell
git diff --check
```

Expected:

- PASS, or only preexisting/generated warnings outside files touched by this plan.

- [ ] **Step 5: Request review subagents before calling the closure pass complete**

Use two review agents. They must not edit files.

Runtime ownership reviewer prompt:

```text
Review the selected-hour lifecycle closure implementation in D:\Projects\Nur\Shade\fast-utci. Do not edit files. Focus on whether scene release forwarding is now controller-instance safe, whether stale releases from old same-content controller instances are ignored, and whether stale/reused GPU handles cannot destroy still-owned accepted outputs. Return findings first with file/line evidence and list any missing tests.
```

Scene lifecycle reviewer prompt:

```text
Review the scene release lifecycle implementation in D:\Projects\Nur\Shade\fast-utci. Do not edit files. Focus on whether UTCIPointCloud and ComparisonRenderer now use one exactly-once release notifier, whether missing controller identity/instance id cannot emit a release, and whether selected-hour visible GPU diagnostics remain honest. Return findings first with file/line evidence and list any missing tests.
```

Expected:

- If either reviewer finds a correctness issue, stop and fix that issue with a failing test first.
- If reviewers only find broader route decomposition or scene sync extraction suggestions, document them as next-stage cleanup candidates, not blockers for this plan.

- [ ] **Step 6: Write the lifecycle closure result note**

Create `docs/superpowers/plans/2026-05-11-selected-hour-lifecycle-closure-before-broader-cleanup-results.md`:

```md
# Selected-Hour Lifecycle Closure Results

Date: 2026-05-11

## Scope

This note closes the selected-hour GPU lifecycle blockers that had to be resolved before broader route decomposition, scene sync extraction, or folder reorganization.

## Closed Contracts

- Scene releases are controller-instance safe: route-host release forwarding checks both `controllerIdentity` and `controllerInstanceId`.
- `UTCIPointCloud` and `ComparisonRenderer` use the same exactly-once release notifier.
- Missing controller identity or instance id suppresses release emission.
- Stale rejected GPU outputs do not destroy a GPU handle still owned by the current accepted output.
- Fresh stale GPU outputs still dispose immediately.

## Verification

Record the actual command output summaries here. Each line must include the command, PASS or FAIL, and the runner's file/test/error counts when the runner prints them.

## Review Agents

Record the actual review-agent findings here. Each line must state whether the reviewer found blocking issues, and any blocking issue must include file/line evidence.

## Broader Cleanup Readiness

The selected-hour lifecycle layer is stable enough to begin a separate broader cleanup plan for route decomposition, scene sync extraction, and folder/module organization.

## Explicitly Deferred

- Route decomposition.
- Full scene sync state-machine extraction.
- Folder reorganization.
- 0.5m performance work.
- Removal of legacy `.bin`, `dataTexture`, parity, collect, or debug fallback paths.
```

Do not finish while this section contains generic summaries; it must contain the observed command results and review findings from this execution.

- [ ] **Step 7: Stop for human review**

Report:

- changed files
- verification commands and results
- review-agent findings
- whether `npm run check` still fails and why
- whether this closure pass is ready to unlock the next broader cleanup plan

Do not commit. Do not start route decomposition or scene sync extraction without human approval.

---

## Final Verification Commands

Run these before claiming the closure pass is complete:

```powershell
cd viewer
npm run test:quality:selected-hour
npm run test:e2e:selected-hour
npm run build
npm run check
cd ..
git diff --check
```

Expected completion state:

- Same-content controller replacement cannot let stale scene releases reach a new controller instance.
- Scene release payloads include `controllerIdentity` and `controllerInstanceId`.
- `UTCIPointCloud` and `ComparisonRenderer` use one shared exactly-once notifier.
- Missing scene release identity suppresses release emission instead of releasing the wrong controller.
- Stale shared GPU handles cannot destroy still-owned accepted output buffers.
- Fresh stale GPU outputs still dispose immediately.
- Main and debug selected-hour browser proofs remain green.
- The result note clearly states that broader route/scene/folder cleanup is now unblocked, without starting it.

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
- If a Playwright probe fails, dump diagnostics and use `superpowers:systematic-debugging`; do not patch by guesswork.
