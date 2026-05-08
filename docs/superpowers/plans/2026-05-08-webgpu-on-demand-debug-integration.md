# WebGPU On-Demand Debug Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `f32` WebGPU on-demand the app-visible debug WebGPU path for selected hour/month scrubbing while preserving Python `.bin` comparison, the existing full `runAll()` path, and explicit `dataTexture` fallback.

**Architecture:** Treat the current May 8 vertical slice as the proof base, not as code to replace wholesale. Promote the strict exposure-only path into a reusable selected-hour controller that precomputes exposure once, computes only the requested hour/month on scrub, discards stale results, and publishes honest diagnostics. Track WebGPU buffer allocation bytes by path so the on-demand route proves it avoids the old all-hours VRAM shape; label this as tracked allocation, not total browser/OS VRAM. Keep rendering initially as `cpu-uploaded-selected-hour` unless a later reviewed task proves direct compute-buffer rendering; do not call it zero-copy.

**Tech Stack:** SvelteKit, Svelte 5 stores/components, Three.js r175 `WebGPURenderer`, WGSL/WebGPU compute, existing Python `.bin` reference loading/parity helpers, Vitest, Playwright.

**User Constraints:** No commits. No git worktrees. The worktree is expected to be dirty. Preserve the current `.bin`, `runAll()`, `readUtciBulk()`, and `dataTexture` paths until a later reviewed production-switch plan.

---

## Current State

Directly verified in this session:

- Focused Vitest command passed in `viewer`: `7` files, `36` tests.
- Current dirty tree already contains the May 8 vertical-slice implementation files and tests.

Proven by current docs and tests:

- Strict debug route `?onDemandPrototype=1&strictExposureOnly=1` reaches `path=exposure-only-f32`.
- Runtime evidence shows no all-hours UTCI/MRT allocation on the strict path.
- Existing diagnostics already expose the most important memory proof fields: `allHoursUtciBytesAllocated`, `allHoursMrtBytesAllocated`, and `oneHourOutputBytes`.
- Multi-hour `f32` comparison against a separate `runAll()` baseline covers hours `12`, `23`, `16`, `17`.
- Known point `31079` is clean for hours `16` and `17`.
- Main-route diagnostics can report `utciSurfaceSource=cpu-uploaded-selected-hour`, `selectedHourTransferCount`, and `dataTextureBuildCount=0`.
- `utciRender=data` fallback remains observable.

Not yet proven:

- Normal debug WebGPU comparison view is not yet driven by on-demand; it still reaches full `runAll()` / live-analysis behavior unless strict flags are used.
- Smooth scrubbing is not yet proven. We need repeated hour/month interaction evidence, stale-result rejection, and final-selection diagnostics.
- Direct compute-buffer-to-render zero-copy is not proven. Current honest transport is `cpu-uploaded-selected-hour`.
- Total browser/OS VRAM is not directly proven. This plan will track known WebGPU buffer allocations owned by the UTCI pipeline and use those counters as the verified memory gate.
- Python `.bin` comparison must remain active as the reference side while WebGPU on-demand becomes the debug WebGPU side.
- Packed MRT/UTCI output remains deferred.

## File Structure

Modify:

- `viewer/src/lib/compute/onDemandDiagnostics.ts`  
  Extend diagnostics with selected month/hour request state, completed state, sequence counters, stale-result counters, render update timing, scrub samples, and tracked WebGPU buffer allocation summaries.

- `viewer/src/lib/compute/compute-manager.ts`  
  Keep the existing on-demand wrappers. Add only thin diagnostics merge helpers if needed; do not weaken `runAll()` or `readUtciBulk()`.

- `viewer/src/lib/compute/webgpuUtciPipeline.ts`  
  Record tracked allocation bytes for persistent exposure buffers, all-hours full-run buffers, and selected-hour output buffers. These counters are the VRAM proxy used by this plan.

- `viewer/src/routes/debug-webgpu-utci/+page.svelte`  
  Promote strict exposure-only logic into a reusable debug on-demand controller. Wire debug WebGPU selected-hour updates to `runUtciForTimeIndex()` for UI hour/month changes. Keep Python `.bin` comparison on the other side.

- `viewer/src/routes/+page.svelte`  
  Keep diagnostics aligned with any shared selected-hour diagnostics shape. Do not make on-demand the default production route.

- `viewer/src/lib/components/scene/UTCIPointCloud.svelte`  
  Preserve existing `Analysis` rendering. Thread selected-hour diagnostics without requiring a full all-hours analysis.

- `viewer/src/lib/services/pointCloudService.ts`  
  Preserve `dataTexture` and `gpuNative` metadata; keep `cpu-uploaded-selected-hour` honest.

- `viewer/tests/e2e/webgpu-f32-on-demand-vertical-slice.spec.ts`  
  Add repeated scrub coverage and app-visible debug comparison assertions.

- `viewer/tests/e2e/webgpu-on-demand-prototype.spec.ts`  
  Keep existing prototype guards passing; add only regression assertions that do not duplicate the vertical-slice suite.

- `docs/superpowers/plans/2026-05-07-webgpu-compute-on-demand-prototype-results.md`  
  Append results from the new debug integration only after fresh browser verification.

Create:

- `viewer/src/lib/compute/onDemandScrubState.ts`  
  Small pure helper for request sequencing and stale-result discard accounting.

- `viewer/tests/compute/onDemandScrubState.test.ts`

## Hard Gates

1. **Fallback gate:** `utciRender=data` still resolves to `dataTexture`, and the previous `.bin` / `runAll()` paths remain callable.
2. **Debug WebGPU on-demand gate:** the debug WebGPU side uses `runExposurePrecompute()` once and `runUtciForTimeIndex()` per selected hour/month, not `runAll()` for each visible scrub.
3. **Python comparison gate:** Python `.bin` reference comparison remains visible/active on the debug path and selected-hour values are numerically compared at sampled points.
4. **No all-hours hot-path gate:** repeated scrubbing under on-demand does not allocate all-hours UTCI/MRT, does not call bulk UTCI readback, and does not rebuild `DataTexture`. A one-hour CPU transfer is allowed only when diagnostics label it `cpu-uploaded-selected-hour`.
5. **Scrub correctness gate:** after rapid hour/month changes with forced overlap, the visible diagnostics report the final selected hour/month, stale earlier results are discarded, and no stale render wins.
6. **Tracked VRAM gate:** on-demand diagnostics report tracked WebGPU allocation bytes and prove the selected-hour path does not allocate all-hours UTCI/MRT buffers; repeated scrubbing must keep selected-hour output bytes bounded instead of accumulating per hour.
7. **Parity gate:** selected on-demand outputs still compare against separate baseline hours `12`, `23`, `16`, `17`, including point `31079` when present.
8. **Honesty gate:** diagnostics say `cpu-uploaded-selected-hour` unless direct compute-buffer rendering is actually implemented and verified.

## Panel A - Council

Task classification: this is a staged architecture promotion, where the main risk is over-promoting a valid prototype before the interactive path is proven.

- **Performance path:** concern -> scrubbing must avoid the old all-hours wall; flag -> current proof is static selected-hour, not repeated interaction; counter-move -> add sequence/timing diagnostics and a scrub E2E before claiming smoothness.
- **Memory path:** concern -> the whole point of on-demand is avoiding the multi-hundred-MB/all-hours VRAM shape; flag -> browser APIs do not expose portable total VRAM; counter-move -> track UTCI-owned WebGPU buffer bytes and compare full-run vs selected-hour allocation shapes.
- **Correctness path:** concern -> Python `.bin` must stay authoritative; flag -> same-pipeline WebGPU baseline is not enough; counter-move -> keep separate baseline and `.bin` collection tests active.
- **UX/debuggability:** concern -> user needs to test this like the old full-run debug path; flag -> strict route currently publishes diagnostics without becoming the normal debug experience; counter-move -> wire on-demand behind an explicit debug flag with status and fallback toggles.
- **Maintainability:** concern -> duplicated shader equations can drift; flag -> on-demand shader mirrors `mrt_utci.wgsl`; counter-move -> keep source-lock/parity tests and document mirrored sections.
- **Architecture ambition:** concern -> zero-copy is the real 0.5m unlock; flag -> current path is CPU-uploaded selected-hour; counter-move -> use this plan to prove smooth selected-hour behavior first, then write a separate direct-buffer plan.

## Tensions

- Smooth app-visible debug path vs zero-copy purity: selected-hour CPU upload may be acceptable for the next proof, but it must not be mislabeled as the final architecture.
- Fast iteration vs proof quality: we can expose a flag quickly, but completion requires repeated scrub E2E and captured diagnostics.
- Keeping `.bin` comparison vs removing CPU memory pressure: comparison stays for trust, but it must be kept off the hot on-demand render path.

## Panel B - Adversarial

- **Attack target:** promoting the current strict prototype into the debug WebGPU path.
- **Stale async results:** vulnerability -> hour/month scrubs can complete out of order; failure scenario -> UI says hour 17 but colors are hour 16; mitigation -> monotonically increasing request id, completed id, stale discard count, E2E final-selection assertion.
- **Hidden full-run contamination:** vulnerability -> a shared manager that previously ran `runAll()` can make strict diagnostics look cleaner than the actual path; failure scenario -> on-demand appears memory-light because baseline work happened elsewhere or earlier; mitigation -> isolate strict manager state and assert all-hours byte counters remain zero during scrub.
- **Comparison comfort trap:** vulnerability -> WebGPU-vs-WebGPU parity can pass while Python `.bin` comparison regresses; failure scenario -> user sees matching internal numbers but old debug baseline differs; mitigation -> keep `.bin` reference visible and add an app-visible selected-hour comparison assertion.
- **Transport mislabeling:** vulnerability -> `gpuNative` sounds like GPU-resident zero-copy; failure scenario -> plan claims 0.5m readiness while still uploading CPU selected-hour arrays; mitigation -> require `utciSurfaceSource` in every runtime result and block zero-copy wording unless it says `compute-buffer-selected-hour`.

Strongest attack: the biggest way this plan fails is by making the strict route feel like a success while the normal debug workflow still uses old full-run/live-analysis machinery or silently renders stale selected-hour data after scrubbing. The antidote is not another static parity test; it is an interaction test that proves the final UI selection owns the final rendered diagnostics while `.bin` comparison and fallback remain intact.

Falsifiers / early warnings:

- `dataTextureBuildCount > 0` on the on-demand GPU scrub path.
- `usedRunAllForSelectedHour === true` during selected-hour debug scrubbing.
- tracked allocation diagnostics show nonzero `allHoursUtciBytesAllocated` or `allHoursMrtBytesAllocated` during the on-demand selected-hour path.
- `selectedHourOutputBytesHighWatermark` grows with every scrub instead of staying near one selected-hour buffer.
- `completedTimeIndex !== selectedTimeIndex` after rapid scrub.
- `staleResultDiscardCount === 0` in a test intentionally causing overlapping requests.
- Results docs contain "zero-copy", "production-ready", or "0.5m ready" while transport is `cpu-uploaded-selected-hour`.

## Recommendation

Proceed with a debug-path promotion plan for `f32` on-demand, not a production default switch and not packed output. The next useful milestone is the one the user can feel: debug WebGPU route loads the Python `.bin` comparison as before, but the WebGPU side computes selected hour/month on demand and remains responsive during scrubbing.

## Milestone 1: Add Scrub-State Accounting

### Task 1: Create Pure Scrub Sequencing Helper

**Files:**
- Create: `viewer/src/lib/compute/onDemandScrubState.ts`
- Test: `viewer/tests/compute/onDemandScrubState.test.ts`

- [ ] **Step 1: Write the failing test**

Create `viewer/tests/compute/onDemandScrubState.test.ts`:

```ts
import { describe, expect, it } from 'vitest';
import {
	createOnDemandScrubState,
	markOnDemandRequestCompleted,
	startOnDemandRequest
} from '$lib/compute/onDemandScrubState';

describe('on-demand scrub state', () => {
	it('accepts the newest request and discards stale completions', () => {
		let state = createOnDemandScrubState();

		const first = startOnDemandRequest(state, { monthIndex: 0, timeIndex: 12 });
		state = first.state;
		const second = startOnDemandRequest(state, { monthIndex: 0, timeIndex: 17 });
		state = second.state;

		const stale = markOnDemandRequestCompleted(state, first.request);
		expect(stale.accepted).toBe(false);
		expect(stale.state.staleResultDiscardCount).toBe(1);
		expect(stale.state.completedTimeIndex).toBeNull();

		const fresh = markOnDemandRequestCompleted(stale.state, second.request);
		expect(fresh.accepted).toBe(true);
		expect(fresh.state.completedTimeIndex).toBe(17);
		expect(fresh.state.completedMonthIndex).toBe(0);
	});
});
```

- [ ] **Step 2: Run the failing test**

Run:

```powershell
cd viewer
npm test -- tests/compute/onDemandScrubState.test.ts
```

Expected: fails because `onDemandScrubState.ts` does not exist.

- [ ] **Step 3: Implement the helper**

Create `viewer/src/lib/compute/onDemandScrubState.ts`:

```ts
export interface OnDemandSelection {
	monthIndex: number;
	timeIndex: number;
}

export interface OnDemandScrubRequest extends OnDemandSelection {
	requestId: number;
}

export interface OnDemandScrubState {
	nextRequestId: number;
	activeRequestId: number | null;
	completedRequestId: number | null;
	selectedMonthIndex: number | null;
	selectedTimeIndex: number | null;
	completedMonthIndex: number | null;
	completedTimeIndex: number | null;
	staleResultDiscardCount: number;
	inFlightCount: number;
}

export function createOnDemandScrubState(): OnDemandScrubState {
	return {
		nextRequestId: 1,
		activeRequestId: null,
		completedRequestId: null,
		selectedMonthIndex: null,
		selectedTimeIndex: null,
		completedMonthIndex: null,
		completedTimeIndex: null,
		staleResultDiscardCount: 0,
		inFlightCount: 0
	};
}

export function startOnDemandRequest(
	state: OnDemandScrubState,
	selection: OnDemandSelection
): { state: OnDemandScrubState; request: OnDemandScrubRequest } {
	const request: OnDemandScrubRequest = {
		requestId: state.nextRequestId,
		...selection
	};
	return {
		request,
		state: {
			...state,
			nextRequestId: state.nextRequestId + 1,
			activeRequestId: request.requestId,
			selectedMonthIndex: selection.monthIndex,
			selectedTimeIndex: selection.timeIndex,
			inFlightCount: state.inFlightCount + 1
		}
	};
}

export function markOnDemandRequestCompleted(
	state: OnDemandScrubState,
	request: OnDemandScrubRequest
): { state: OnDemandScrubState; accepted: boolean } {
	const inFlightCount = Math.max(0, state.inFlightCount - 1);
	if (request.requestId !== state.activeRequestId) {
		return {
			accepted: false,
			state: {
				...state,
				inFlightCount,
				staleResultDiscardCount: state.staleResultDiscardCount + 1
			}
		};
	}
	return {
		accepted: true,
		state: {
			...state,
			inFlightCount,
			completedRequestId: request.requestId,
			completedMonthIndex: request.monthIndex,
			completedTimeIndex: request.timeIndex
		}
	};
}
```

- [ ] **Step 4: Run the test**

Run:

```powershell
cd viewer
npm test -- tests/compute/onDemandScrubState.test.ts
```

Expected: passes.

### Task 2: Extend Diagnostics With Scrub Fields

**Files:**
- Modify: `viewer/src/lib/compute/onDemandDiagnostics.ts`
- Test: `viewer/tests/compute/onDemandDiagnostics.test.ts`

- [ ] **Step 1: Add failing assertions**

In `viewer/tests/compute/onDemandDiagnostics.test.ts`, extend the conservative-default test:

```ts
expect(diagnostics.selectedMonthIndex).toBeNull();
expect(diagnostics.selectedTimeIndex).toBeNull();
expect(diagnostics.completedMonthIndex).toBeNull();
expect(diagnostics.completedTimeIndex).toBeNull();
expect(diagnostics.activeRequestId).toBeNull();
expect(diagnostics.completedRequestId).toBeNull();
expect(diagnostics.staleResultDiscardCount).toBe(0);
expect(diagnostics.inFlightCount).toBe(0);
expect(diagnostics.trackedGpuAllocationBytes.persistentExposureBytes).toBe(0);
expect(diagnostics.trackedGpuAllocationBytes.allHoursOutputBytes).toBe(0);
expect(diagnostics.trackedGpuAllocationBytes.selectedHourOutputBytes).toBe(0);
expect(diagnostics.trackedGpuAllocationBytes.selectedHourOutputBytesHighWatermark).toBe(0);
expect(diagnostics.trackedGpuAllocationBytes.trackingScope).toBe('utci-owned-webgpu-buffers');
```

- [ ] **Step 2: Run the failing test**

Run:

```powershell
cd viewer
npm test -- tests/compute/onDemandDiagnostics.test.ts
```

Expected: fails because the fields are missing.

- [ ] **Step 3: Add fields to the diagnostics type and defaults**

In `viewer/src/lib/compute/onDemandDiagnostics.ts`, add to `OnDemandRuntimeDiagnostics`:

```ts
selectedMonthIndex: number | null;
selectedTimeIndex: number | null;
completedMonthIndex: number | null;
completedTimeIndex: number | null;
activeRequestId: number | null;
completedRequestId: number | null;
staleResultDiscardCount: number;
inFlightCount: number;
scrubSampleCount: number;
```

Add to `createEmptyOnDemandDiagnostics()`:

```ts
selectedMonthIndex: null,
selectedTimeIndex: null,
completedMonthIndex: null,
completedTimeIndex: null,
activeRequestId: null,
completedRequestId: null,
staleResultDiscardCount: 0,
inFlightCount: 0,
scrubSampleCount: 0,
trackedGpuAllocationBytes: {
	persistentExposureBytes: 0,
	allHoursOutputBytes: 0,
	selectedHourOutputBytes: 0,
	selectedHourOutputBytesHighWatermark: 0,
	trackingScope: 'utci-owned-webgpu-buffers'
},
```

- [ ] **Step 4: Run diagnostics tests**

Run:

```powershell
cd viewer
npm test -- tests/compute/onDemandDiagnostics.test.ts tests/compute/onDemandScrubState.test.ts
```

Expected: both files pass.

### Task 3: Track UTCI-Owned WebGPU Buffer Allocation Bytes

**Files:**
- Modify: `viewer/src/lib/compute/onDemandDiagnostics.ts`
- Modify: `viewer/src/lib/compute/webgpuUtciPipeline.ts`
- Test: `viewer/tests/compute/onDemandDiagnostics.test.ts`
- Test: `viewer/tests/compute/webgpu-on-demand-source-locks.test.ts`

- [ ] **Step 1: Add diagnostics test for allocation merging**

In `viewer/tests/compute/onDemandDiagnostics.test.ts`, add:

```ts
import { mergeTrackedGpuAllocationBytes } from '$lib/compute/onDemandDiagnostics';

it('tracks selected-hour output high-watermark without pretending to know total browser VRAM', () => {
	const diagnostics = createEmptyOnDemandDiagnostics();

	const first = mergeTrackedGpuAllocationBytes(diagnostics, {
		persistentExposureBytes: 128,
		selectedHourOutputBytes: 64
	});
	const second = mergeTrackedGpuAllocationBytes(first, {
		selectedHourOutputBytes: 32
	});

	expect(second.trackedGpuAllocationBytes.trackingScope).toBe('utci-owned-webgpu-buffers');
	expect(second.trackedGpuAllocationBytes.persistentExposureBytes).toBe(128);
	expect(second.trackedGpuAllocationBytes.allHoursOutputBytes).toBe(0);
	expect(second.trackedGpuAllocationBytes.selectedHourOutputBytes).toBe(32);
	expect(second.trackedGpuAllocationBytes.selectedHourOutputBytesHighWatermark).toBe(64);
});
```

- [ ] **Step 2: Run the failing diagnostics test**

Run:

```powershell
cd viewer
npm test -- tests/compute/onDemandDiagnostics.test.ts
```

Expected: fails because `mergeTrackedGpuAllocationBytes()` does not exist.

- [ ] **Step 3: Add tracked allocation types and helper**

In `viewer/src/lib/compute/onDemandDiagnostics.ts`, add:

```ts
export interface TrackedGpuAllocationBytes {
	persistentExposureBytes: number;
	allHoursOutputBytes: number;
	selectedHourOutputBytes: number;
	selectedHourOutputBytesHighWatermark: number;
	trackingScope: 'utci-owned-webgpu-buffers';
}

export type TrackedGpuAllocationBytesPatch = Partial<
	Omit<TrackedGpuAllocationBytes, 'trackingScope' | 'selectedHourOutputBytesHighWatermark'>
>;
```

Add to `OnDemandRuntimeDiagnostics`:

```ts
trackedGpuAllocationBytes: TrackedGpuAllocationBytes;
```

Add to `createEmptyOnDemandDiagnostics()`:

```ts
trackedGpuAllocationBytes: {
	persistentExposureBytes: 0,
	allHoursOutputBytes: 0,
	selectedHourOutputBytes: 0,
	selectedHourOutputBytesHighWatermark: 0,
	trackingScope: 'utci-owned-webgpu-buffers'
},
```

Add:

```ts
export function mergeTrackedGpuAllocationBytes(
	diagnostics: OnDemandRuntimeDiagnostics,
	patch: TrackedGpuAllocationBytesPatch
): OnDemandRuntimeDiagnostics {
	const selectedHourOutputBytes =
		patch.selectedHourOutputBytes ?? diagnostics.trackedGpuAllocationBytes.selectedHourOutputBytes;
	return {
		...diagnostics,
		trackedGpuAllocationBytes: {
			...diagnostics.trackedGpuAllocationBytes,
			...patch,
			selectedHourOutputBytes,
			selectedHourOutputBytesHighWatermark: Math.max(
				diagnostics.trackedGpuAllocationBytes.selectedHourOutputBytesHighWatermark,
				selectedHourOutputBytes
			),
			trackingScope: 'utci-owned-webgpu-buffers'
		}
	};
}
```

- [ ] **Step 4: Record all-hours output bytes in `runAll()`**

In `viewer/src/lib/compute/webgpuUtciPipeline.ts`, import the helper:

```ts
import { mergeTrackedGpuAllocationBytes } from '$lib/compute/onDemandDiagnostics';
```

Where `runAll()` already records `utciBytes` and `mrtBytes`, merge:

```ts
this.onDemandDiagnostics = mergeTrackedGpuAllocationBytes(this.onDemandDiagnostics, {
	allHoursOutputBytes: utciBytes + mrtBytes,
	selectedHourOutputBytes: 0
});
```

Keep the existing `allHoursUtciBytesAllocated` and `allHoursMrtBytesAllocated` fields too; they are still useful explicit counters.

- [ ] **Step 5: Record persistent exposure bytes after exposure precompute**

After `runExposurePrecompute()` has created or reused the solar and sky exposure buffers, compute tracked persistent exposure bytes:

```ts
const persistentExposureBytes = solarExposureBytes + skyExposureBytes;
this.onDemandDiagnostics = mergeTrackedGpuAllocationBytes(this.onDemandDiagnostics, {
	persistentExposureBytes,
	allHoursOutputBytes: 0
});
```

Use the existing local byte variables if they already exist. If they do not, calculate:

```ts
const solarExposureBytes = Math.ceil((numPoints * numHours * numMonths) / 32) * 4;
const skyExposureBytes = numPoints * 4;
```

- [ ] **Step 6: Record selected-hour output bytes after each selected-hour dispatch**

In `runUtciForTimeIndex()`, after `outputBytes` is known, merge:

```ts
this.onDemandDiagnostics = mergeTrackedGpuAllocationBytes(this.onDemandDiagnostics, {
	selectedHourOutputBytes: outputBytes,
	allHoursOutputBytes: 0
});
```

This high-watermark must stay bounded during repeated scrubbing. It should not become `outputBytes * numberOfScrubbedHours`.

- [ ] **Step 7: Add a source-lock guard**

In `viewer/tests/compute/webgpu-on-demand-source-locks.test.ts`, add:

```ts
it('records tracked GPU allocation bytes without using browser total VRAM APIs', () => {
	expect(source.includes('mergeTrackedGpuAllocationBytes')).toBe(true);
	expect(source.includes('trackedGpuAllocationBytes')).toBe(true);
	expect(source.includes('performance.memory')).toBe(false);
	expect(source.includes('measureUserAgentSpecificMemory')).toBe(false);
});
```

- [ ] **Step 8: Run focused allocation tests**

Run:

```powershell
cd viewer
npm test -- tests/compute/onDemandDiagnostics.test.ts tests/compute/webgpu-on-demand-source-locks.test.ts
```

Expected: both files pass.

## Milestone 2: Promote On-Demand Into The Debug WebGPU Path

### Task 4: Add Debug Route On-Demand Controller State

**Files:**
- Modify: `viewer/src/routes/debug-webgpu-utci/+page.svelte`
- Test: `viewer/tests/e2e/webgpu-f32-on-demand-vertical-slice.spec.ts`

- [ ] **Step 1: Add failing E2E for normal debug on-demand mode**

Append to `viewer/tests/e2e/webgpu-f32-on-demand-vertical-slice.spec.ts`:

```ts
test('debug route can use f32 on-demand as the visible WebGPU side while keeping comparison active', async ({ page }) => {
	test.setTimeout(180_000);
	await page.goto('/debug-webgpu-utci?parity=1&onDemandPrototype=1&utciOnDemand=f32&utciRender=gpu&timeIndex=12');

	const hasWebGpu = await page.evaluate(() => Boolean(navigator.gpu));
	test.skip(!hasWebGpu && process.env.REQUIRE_WEBGPU_ON_DEMAND !== '1', 'WebGPU unavailable.');

	await page.waitForFunction(() => {
		const diagnostics = (window as Window & { __onDemandPrototypeDiagnostics__?: any })
			.__onDemandPrototypeDiagnostics__;
		return (
			diagnostics?.path === 'exposure-only-f32' &&
			diagnostics?.usedExposureOnlyPrecompute === true &&
			diagnostics?.usedRunAllForSelectedHour === false &&
			diagnostics?.completedTimeIndex === 12 &&
			diagnostics?.dataTextureBuildCount === 0 &&
			diagnostics?.appVisibleSelectedHour === true
		);
	}, undefined, { timeout: 180_000 });

	const diagnostics = await page.evaluate(() => {
		return (window as Window & { __onDemandPrototypeDiagnostics__?: any })
			.__onDemandPrototypeDiagnostics__;
	});

	expect(diagnostics.renderTransport).toMatch(/selected-hour|none/);
	expect(diagnostics.debugComparisonReference).toBe('python-bin');
	expect(diagnostics.pythonBinComparisonActive).toBe(true);
	expect(diagnostics.selectedHourReadbackCount).toBeLessThanOrEqual(1);
});
```

- [ ] **Step 2: Run the failing E2E**

Run:

```powershell
cd viewer
$env:REQUIRE_WEBGPU_ON_DEMAND='1'
npm run test:e2e -- tests/e2e/webgpu-f32-on-demand-vertical-slice.spec.ts --project=chromium
Remove-Item Env:\REQUIRE_WEBGPU_ON_DEMAND
```

Expected: fails because normal debug route mode does not yet drive WebGPU selected-hour rendering from on-demand.

- [ ] **Step 3: Reuse strict setup for flagged debug mode**

In `viewer/src/routes/debug-webgpu-utci/+page.svelte`, parse:

```ts
const onDemandDebugModeEnabled =
	$page.url.searchParams.get('utciOnDemand') === 'f32';
const debugOnDemandMode = onDemandDebugModeEnabled ? 'f32' : 'off';
```

Update the existing `onDemandPrototypeEnabled` reactive statement so diagnostics are also active for the debug on-demand path:

```ts
$: onDemandPrototypeEnabled =
	$page.url.searchParams.get('onDemandPrototype') === '1' || onDemandDebugModeEnabled;
```

Create a route-local controller state near the existing on-demand diagnostics state:

```ts
let onDemandDebugPrepared:
	| {
			computeManager: ComputeManager;
			pipeline: UTCIComputePipeline;
			numPoints: number;
			numHours: number;
			numMonths: number;
			base: Analysis;
			signal: AbortSignal;
			runId: number;
			zHeight: number;
			exposureReady: boolean;
	  }
	| undefined;
let onDemandScrubState = createOnDemandScrubState();
```

Add imports:

```ts
import {
	createOnDemandScrubState,
	markOnDemandRequestCompleted,
	startOnDemandRequest
} from '$lib/compute/onDemandScrubState';
```

- [ ] **Step 4: Add selected-hour runner**

Add a local helper in `debug-webgpu-utci/+page.svelte`:

```ts
async function runDebugOnDemandSelectedHour(params: {
	monthIndex: number;
	timeIndex: number;
	readbackForComparison: boolean;
}) {
	if (!onDemandDebugPrepared) {
		const zHeight = base.metadata.bounds?.z ?? 0.9;
		const signal = liveAbortController?.signal ?? new AbortController().signal;
		const prepared = await prepareWebgpuDebugInputsForCurrentSelection({
			base,
			signal,
			runId: liveRunCounter,
			zHeight
		});
		onDemandDebugPrepared = {
			...prepared,
			base,
			signal,
			runId: liveRunCounter,
			zHeight,
			exposureReady: false
		};
	}
	if (!onDemandDebugPrepared.exposureReady) {
		await onDemandDebugPrepared.computeManager.runExposurePrecompute({
			numPoints: onDemandDebugPrepared.numPoints,
			numHours: onDemandDebugPrepared.numHours,
			numMonths: onDemandDebugPrepared.numMonths
		});
		onDemandDebugPrepared.exposureReady = true;
	}

	const started = startOnDemandRequest(onDemandScrubState, {
		monthIndex: params.monthIndex,
		timeIndex: params.timeIndex
	});
	onDemandScrubState = started.state;

	const output = await onDemandDebugPrepared.computeManager.runUtciForTimeIndex({
		timeIndex: params.timeIndex,
		numPoints: onDemandDebugPrepared.numPoints,
		numHours: onDemandDebugPrepared.numHours,
		numMonths: onDemandDebugPrepared.numMonths,
		format: 'f32-utci'
	});

	const completed = markOnDemandRequestCompleted(onDemandScrubState, started.request);
	onDemandScrubState = completed.state;
	if (!completed.accepted) return undefined;

	const selectedHourUtci = params.readbackForComparison
		? await onDemandDebugPrepared.pipeline.readOnDemandUtciForDebug?.({
				numPoints: onDemandDebugPrepared.numPoints
			})
		: undefined;

	if (selectedHourUtci) {
		liveAnalysis = buildSelectedHourLiveAnalysis({
			base: onDemandDebugPrepared.base,
			utciValues: selectedHourUtci,
			monthIndex: params.monthIndex,
			timeIndex: params.timeIndex
		});
		comparisonStore.update((state) => ({
			...state,
			isComparing: true,
			comparisonAnalysis: liveAnalysis
		}));
	}

	const pipelineDiagnostics = onDemandDebugPrepared.computeManager.getOnDemandDiagnostics();
	updateOnDemandPrototypeDiagnostics({
		...pipelineDiagnostics,
		selectedMonthIndex: onDemandScrubState.selectedMonthIndex,
		selectedTimeIndex: onDemandScrubState.selectedTimeIndex,
		completedMonthIndex: onDemandScrubState.completedMonthIndex,
		completedTimeIndex: onDemandScrubState.completedTimeIndex,
		activeRequestId: onDemandScrubState.activeRequestId,
		completedRequestId: onDemandScrubState.completedRequestId,
		staleResultDiscardCount: onDemandScrubState.staleResultDiscardCount,
		inFlightCount: onDemandScrubState.inFlightCount,
		scrubSampleCount: onDemandScrubState.completedRequestId ?? 0,
		debugComparisonReference: 'python-bin',
		pythonBinComparisonActive: true,
		appVisibleSelectedHour: Boolean(selectedHourUtci),
		selectedHourReadbackCount: selectedHourUtci ? 1 : 0,
		renderTransport: selectedHourUtci ? 'cpu-uploaded-selected-hour' : 'none',
		dataTextureBuildCount: 0
	});

	return output;
}
```

Use the route's existing `updateOnDemandPrototypeDiagnostics()` helper; do not create a second window writer.

Add the selected-hour analysis builder locally, using the existing `Analysis` and `FullDayData` shapes already imported by the route:

```ts
function buildSelectedHourLiveAnalysis(params: {
	base: Analysis;
	utciValues: Float32Array;
	monthIndex: number;
	timeIndex: number;
}): Analysis {
	return {
		...params.base,
		id: `${params.base.id ?? 'webgpu'}-on-demand-${params.monthIndex}-${params.timeIndex}`,
		source: 'webgpu',
		data: {
			...params.base.data,
			type: 'full-day',
			values: params.utciValues,
			utci: params.utciValues,
			numHours: 1,
			numMonths: 1,
			numPositions: params.utciValues.length,
			selectedMonthIndex: params.monthIndex,
			selectedTimeIndex: params.timeIndex
		}
	} as Analysis;
}
```

If the local `Analysis` type uses different field names, map only the fields consumed by `UTCIPointCloud.svelte` and `pointCloudService.ts`, then add a focused test for that mapping before using it in the route.

- [ ] **Step 5: Route flagged debug WebGPU selection through the helper**

Before the existing `createLiveUtciAnalysisFromCompute()` branch in the debug route, add:

```ts
if (debugOnDemandMode === 'f32') {
	const monthIndex = currentMonth;
	const timeIndex = getEffectiveHourIndex(currentMonth, currentHour);
	const output = await runDebugOnDemandSelectedHour({
		monthIndex,
		timeIndex,
		readbackForComparison: true
	});
	if (!output) return;
	return;
}
```

Keep the previous strict branch available for existing tests. Do not delete the full-run branch.

- [ ] **Step 6: Run the E2E**

Run:

```powershell
cd viewer
$env:REQUIRE_WEBGPU_ON_DEMAND='1'
npm run test:e2e -- tests/e2e/webgpu-f32-on-demand-vertical-slice.spec.ts --project=chromium
Remove-Item Env:\REQUIRE_WEBGPU_ON_DEMAND
```

Expected: new debug on-demand test passes and existing vertical-slice tests still pass.

## Milestone 3: Prove Smooth Scrubbing Behavior

### Task 5: Add Repeated Hour/Month Scrub E2E

**Files:**
- Modify: `viewer/tests/e2e/webgpu-f32-on-demand-vertical-slice.spec.ts`
- Modify: `viewer/src/routes/debug-webgpu-utci/+page.svelte`

- [ ] **Step 1: Add failing scrub test**

Append:

```ts
test('debug on-demand discards stale scrub results and ends on the final selected hour', async ({ page }) => {
	test.setTimeout(180_000);
	await page.goto('/debug-webgpu-utci?parity=1&onDemandPrototype=1&utciOnDemand=f32&utciRender=gpu&forceOnDemandOverlapMs=50&timeIndex=12');

	const hasWebGpu = await page.evaluate(() => Boolean(navigator.gpu));
	test.skip(!hasWebGpu && process.env.REQUIRE_WEBGPU_ON_DEMAND !== '1', 'WebGPU unavailable.');

	await page.waitForFunction(() => {
		const diagnostics = (window as Window & { __onDemandPrototypeDiagnostics__?: any })
			.__onDemandPrototypeDiagnostics__;
		return diagnostics?.completedTimeIndex === 12;
	}, undefined, { timeout: 180_000 });

	for (const timeIndex of [13, 16, 17, 23]) {
		await page.evaluate((nextTimeIndex) => {
			const url = new URL(window.location.href);
			url.searchParams.set('timeIndex', String(nextTimeIndex));
			window.history.pushState({}, '', url);
			window.dispatchEvent(new PopStateEvent('popstate'));
		}, timeIndex);
	}

	await page.waitForFunction(() => {
		const diagnostics = (window as Window & { __onDemandPrototypeDiagnostics__?: any })
			.__onDemandPrototypeDiagnostics__;
		return (
			diagnostics?.selectedTimeIndex === 23 &&
			diagnostics?.completedTimeIndex === 23 &&
			diagnostics?.inFlightCount === 0
		);
	}, undefined, { timeout: 180_000 });

	const diagnostics = await page.evaluate(() => {
		return (window as Window & { __onDemandPrototypeDiagnostics__?: any })
			.__onDemandPrototypeDiagnostics__;
	});

	expect(diagnostics.usedRunAllForSelectedHour).toBe(false);
	expect(diagnostics.usedExposureOnlyPrecompute).toBe(true);
expect(diagnostics.allHoursUtciBytesAllocated).toBe(0);
expect(diagnostics.allHoursMrtBytesAllocated).toBe(0);
expect(diagnostics.trackedGpuAllocationBytes.allHoursOutputBytes).toBe(0);
expect(diagnostics.trackedGpuAllocationBytes.selectedHourOutputBytes).toBeGreaterThan(0);
expect(diagnostics.trackedGpuAllocationBytes.selectedHourOutputBytesHighWatermark).toBe(
	diagnostics.trackedGpuAllocationBytes.selectedHourOutputBytes
);
expect(diagnostics.dataTextureBuildCount).toBe(0);
expect(diagnostics.selectedTimeIndex).toBe(23);
expect(diagnostics.completedTimeIndex).toBe(23);
expect(diagnostics.scrubSampleCount).toBeGreaterThanOrEqual(2);
expect(diagnostics.staleResultDiscardCount).toBeGreaterThan(0);
});
```

- [ ] **Step 2: Run the failing scrub test**

Run:

```powershell
cd viewer
$env:REQUIRE_WEBGPU_ON_DEMAND='1'
npm run test:e2e -- tests/e2e/webgpu-f32-on-demand-vertical-slice.spec.ts --project=chromium
Remove-Item Env:\REQUIRE_WEBGPU_ON_DEMAND
```

Expected: fails if route changes do not trigger selected-hour recompute or diagnostics do not track final selection.

- [ ] **Step 3: Trigger recompute on selected hour/month changes**

In `debug-webgpu-utci/+page.svelte`, make the existing route/query handling call:

```ts
if (debugOnDemandMode === 'f32' && browser) {
	void runDebugOnDemandSelectedHour({
		monthIndex: currentMonth,
		timeIndex: getEffectiveHourIndex(currentMonth, currentHour),
		readbackForComparison: true
	});
}
```

Guard this with the same lifecycle protection used by existing route effects so it does not run during SSR or before data is loaded.

When `forceOnDemandOverlapMs` is present, delay completion before `markOnDemandRequestCompleted()`:

```ts
const forcedOverlapMs = Number($page.url.searchParams.get('forceOnDemandOverlapMs') ?? '0');
if (forcedOverlapMs > 0) {
	await new Promise((resolve) => setTimeout(resolve, forcedOverlapMs));
}
```

This query parameter is only for E2E race coverage and must not affect normal runtime.

- [ ] **Step 4: Add render timing when the selected-hour surface updates**

When selected-hour render data is handed to the UTCI surface, record:

```ts
const renderStarted = performance.now();
// existing selected-hour surface update
const renderUpdateMs = performance.now() - renderStarted;
updateOnDemandPrototypeDiagnostics({
	...getParityWindow().__onDemandPrototypeDiagnostics__,
	timings: {
		...getParityWindow().__onDemandPrototypeDiagnostics__?.timings,
		renderUpdateMs
	}
});
```

- [ ] **Step 5: Run scrub E2E**

Run:

```powershell
cd viewer
$env:REQUIRE_WEBGPU_ON_DEMAND='1'
npm run test:e2e -- tests/e2e/webgpu-f32-on-demand-vertical-slice.spec.ts --project=chromium
Remove-Item Env:\REQUIRE_WEBGPU_ON_DEMAND
```

Expected: passes, with final diagnostics matching the final selected hour.

## Milestone 4: Preserve Python `.bin` Comparison

### Task 6: Add App-Visible Python Reference Assertion

**Files:**
- Modify: `viewer/tests/e2e/webgpu-f32-on-demand-vertical-slice.spec.ts`
- Modify: `viewer/src/routes/debug-webgpu-utci/+page.svelte`

- [ ] **Step 1: Add failing assertion for Python comparison metadata**

Append:

```ts
test('debug on-demand keeps python bin comparison metadata active', async ({ page }) => {
	test.setTimeout(180_000);
	await page.goto('/debug-webgpu-utci?parity=1&onDemandPrototype=1&utciOnDemand=f32&utciRender=gpu&timeIndex=17');

	const hasWebGpu = await page.evaluate(() => Boolean(navigator.gpu));
	test.skip(!hasWebGpu && process.env.REQUIRE_WEBGPU_ON_DEMAND !== '1', 'WebGPU unavailable.');

	await page.waitForFunction(() => {
		const diagnostics = (window as Window & { __onDemandPrototypeDiagnostics__?: any })
			.__onDemandPrototypeDiagnostics__;
		return diagnostics?.completedTimeIndex === 17 && diagnostics?.pythonBinComparisonActive === true;
	}, undefined, { timeout: 180_000 });

	const diagnostics = await page.evaluate(() => {
		return (window as Window & { __onDemandPrototypeDiagnostics__?: any })
			.__onDemandPrototypeDiagnostics__;
	});

expect(diagnostics.debugComparisonReference).toBe('python-bin');
expect(diagnostics.usedRunAllForSelectedHour).toBe(false);
expect(diagnostics.pythonBinSampleComparison?.numCompared ?? 0).toBeGreaterThan(0);
expect(diagnostics.pythonBinSampleComparison?.maxAbsDiff ?? Number.POSITIVE_INFINITY).toBeLessThanOrEqual(1e-5);
});
```

- [ ] **Step 2: Run the failing assertion**

Run:

```powershell
cd viewer
$env:REQUIRE_WEBGPU_ON_DEMAND='1'
npm run test:e2e -- tests/e2e/webgpu-f32-on-demand-vertical-slice.spec.ts --project=chromium
Remove-Item Env:\REQUIRE_WEBGPU_ON_DEMAND
```

Expected: fails if Python reference state is not exposed.

- [ ] **Step 3: Publish reference metadata from the existing parity loader**

Where the debug route currently loads the Python `.bin` reference or enters normal parity mode, publish:

```ts
const pythonBinSampleComparison = selectedHourUtci
	? compareSelectedHourSamplesAgainstPythonBin({
			pythonAnalysis: $analysisStore,
			webgpuUtci: selectedHourUtci,
			monthIndex: currentMonth,
			timeIndex: getEffectiveHourIndex(currentMonth, currentHour),
			samplePointIndices: [0, 31079, Math.max(0, selectedHourUtci.length - 1)]
		})
	: undefined;

updateOnDemandPrototypeDiagnostics({
	...getParityWindow().__onDemandPrototypeDiagnostics__,
	debugComparisonReference: 'python-bin',
	pythonBinComparisonActive: Boolean($analysisStore && comparisonStore.isComparing),
	pythonBinSampleComparison
});
```

Use the actual local store names already present in `debug-webgpu-utci/+page.svelte`; do not introduce a second source of truth for the reference data.

Add the sampled comparison helper near the existing float comparison helper:

```ts
function compareSelectedHourSamplesAgainstPythonBin(params: {
	pythonAnalysis: Analysis | null;
	webgpuUtci: Float32Array;
	monthIndex: number;
	timeIndex: number;
	samplePointIndices: number[];
}) {
	if (!params.pythonAnalysis) {
		return { numCompared: 0, maxAbsDiff: null, samples: [] };
	}
	const samples = params.samplePointIndices
		.filter((pointIndex) => pointIndex >= 0 && pointIndex < params.webgpuUtci.length)
		.map((pointIndex) => {
			const python = readPythonReferenceUtciAt({
				analysis: params.pythonAnalysis,
				pointIndex,
				monthIndex: params.monthIndex,
				timeIndex: params.timeIndex
			});
			const webgpu = params.webgpuUtci[pointIndex];
			return { pointIndex, python, webgpu, diff: webgpu - python };
		});
	return {
		numCompared: samples.length,
		maxAbsDiff: Math.max(...samples.map((sample) => Math.abs(sample.diff))),
		samples
	};
}
```

If the route already has a helper for reading Python reference values, use that helper in place of `readPythonReferenceUtciAt`. If it does not, add a route-local helper that follows the same indexing rules used by the existing parity collector.

- [ ] **Step 4: Run the Python reference assertion**

Run:

```powershell
cd viewer
$env:REQUIRE_WEBGPU_ON_DEMAND='1'
npm run test:e2e -- tests/e2e/webgpu-f32-on-demand-vertical-slice.spec.ts --project=chromium
Remove-Item Env:\REQUIRE_WEBGPU_ON_DEMAND
```

Expected: passes.

### Task 7: Add Explicit Fallback Regression E2E

**Files:**
- Modify: `viewer/tests/e2e/webgpu-f32-on-demand-vertical-slice.spec.ts`

- [ ] **Step 1: Add fallback test**

Append:

```ts
test('debug on-demand still allows dataTexture fallback', async ({ page }) => {
	test.setTimeout(180_000);
	await page.goto('/debug-webgpu-utci?parity=1&onDemandPrototype=1&utciOnDemand=f32&utciRender=data&timeIndex=12');

	await page.waitForFunction(() => {
		const diagnostics = (window as Window & { __onDemandPrototypeDiagnostics__?: any })
			.__onDemandPrototypeDiagnostics__;
		return diagnostics?.utciRenderResolved === 'dataTexture';
	}, undefined, { timeout: 180_000 });

	const diagnostics = await page.evaluate(() => {
		return (window as Window & { __onDemandPrototypeDiagnostics__?: any })
			.__onDemandPrototypeDiagnostics__;
	});

	expect(diagnostics.utciRenderResolved).toBe('dataTexture');
	expect(diagnostics.dataTextureBuildCount).toBeGreaterThanOrEqual(1);
});
```

- [ ] **Step 2: Run fallback test with the vertical-slice suite**

Run:

```powershell
cd viewer
$env:REQUIRE_WEBGPU_ON_DEMAND='1'
npm run test:e2e -- tests/e2e/webgpu-f32-on-demand-vertical-slice.spec.ts --project=chromium
Remove-Item Env:\REQUIRE_WEBGPU_ON_DEMAND
```

Expected: passes and proves the fallback switch still works.

## Milestone 5: Full Verification And Results Capture

### Task 8: Run Focused Verification

**Files:**
- Modify only if tests reveal defects in files named above.

- [ ] **Step 1: Run focused Vitest**

Run:

```powershell
cd viewer
npm test -- tests/compute/onDemandDiagnostics.test.ts tests/compute/onDemandScrubState.test.ts tests/compute/onDemandSizing.test.ts tests/compute/onDemandOutputFormat.test.ts tests/compute/compute-manager-on-demand.test.ts tests/compute/webgpu-on-demand-source-locks.test.ts tests/compute/gpu-pipeline.test.ts tests/services/pointCloudService.surface.test.ts
```

Expected: all focused files pass.

- [ ] **Step 2: Run prototype E2E**

Run:

```powershell
cd viewer
$env:REQUIRE_WEBGPU_ON_DEMAND='1'
npm run test:e2e -- tests/e2e/webgpu-on-demand-prototype.spec.ts --project=chromium
Remove-Item Env:\REQUIRE_WEBGPU_ON_DEMAND
```

Expected: all prototype E2E tests pass. If WebGPU cannot run, do not claim runtime gates.

- [ ] **Step 3: Run vertical-slice and scrub E2E**

Run:

```powershell
cd viewer
$env:REQUIRE_WEBGPU_ON_DEMAND='1'
npm run test:e2e -- tests/e2e/webgpu-f32-on-demand-vertical-slice.spec.ts --project=chromium
Remove-Item Env:\REQUIRE_WEBGPU_ON_DEMAND
```

Expected: all vertical-slice tests pass, including repeated scrub and Python comparison metadata.

### Task 9: Update Results Without Overclaiming

**Files:**
- Modify: `docs/superpowers/plans/2026-05-07-webgpu-compute-on-demand-prototype-results.md`
- Modify: `docs/webgpu_strategy_analysis.md`

- [ ] **Step 1: Capture runtime objects from the passing browser run**

From the passing debug route, capture:

```ts
({
	onDemand: window.__onDemandPrototypeDiagnostics__,
	multiHour: window.__onDemandMultiHourComparison__,
	render: window.__utciRenderDiagnostics__
})
```

- [ ] **Step 2: Append a dated debug integration section**

Append to `docs/superpowers/plans/2026-05-07-webgpu-compute-on-demand-prototype-results.md`:

```ts
function passFail(value: boolean): 'pass' | 'fail' {
	return value ? 'pass' : 'fail';
}

function valueOrUnavailable(value: unknown): string {
	return value === undefined || value === null || value === ''
		? 'not exposed by runtime'
		: String(value);
}

function buildDebugIntegrationResults(params: {
	onDemand: any;
	fallback: any;
}): string {
	const { onDemand, fallback } = params;
	const finalSelectionMatches =
		onDemand.selectedTimeIndex === onDemand.completedTimeIndex &&
		onDemand.selectedMonthIndex === onDemand.completedMonthIndex;
	const noAllHours =
		onDemand.allHoursUtciBytesAllocated === 0 &&
		onDemand.allHoursMrtBytesAllocated === 0;
	const trackedAllocation =
		onDemand.trackedGpuAllocationBytes ?? {
			persistentExposureBytes: null,
			allHoursOutputBytes: null,
			selectedHourOutputBytes: null,
			selectedHourOutputBytesHighWatermark: null,
			trackingScope: null
		};
	const trackedVramPass =
		trackedAllocation.trackingScope === 'utci-owned-webgpu-buffers' &&
		trackedAllocation.allHoursOutputBytes === 0 &&
		trackedAllocation.selectedHourOutputBytes > 0 &&
		trackedAllocation.selectedHourOutputBytesHighWatermark ===
			trackedAllocation.selectedHourOutputBytes;
	const transportHonest =
		onDemand.renderTransport !== 'compute-buffer-selected-hour' ||
		onDemand.utciSurfaceSource === 'compute-buffer-selected-hour';

	return `## 2026-05-08 Debug On-Demand Integration Follow-Up

### Gate Results

| Gate | Result | Evidence |
| --- | --- | --- |
| Debug WebGPU on-demand path | ${passFail(onDemand.path === 'exposure-only-f32' && onDemand.usedRunAllForSelectedHour === false)} | completedTimeIndex=${valueOrUnavailable(onDemand.completedTimeIndex)}, usedRunAllForSelectedHour=${valueOrUnavailable(onDemand.usedRunAllForSelectedHour)} |
| Python \`.bin\` comparison preserved | ${passFail(onDemand.debugComparisonReference === 'python-bin' && onDemand.pythonBinComparisonActive === true)} | debugComparisonReference=${valueOrUnavailable(onDemand.debugComparisonReference)}, pythonBinComparisonActive=${valueOrUnavailable(onDemand.pythonBinComparisonActive)} |
| Repeated scrub final selection | ${passFail(finalSelectionMatches)} | selectedTimeIndex=${valueOrUnavailable(onDemand.selectedTimeIndex)}, completedTimeIndex=${valueOrUnavailable(onDemand.completedTimeIndex)}, staleResultDiscardCount=${valueOrUnavailable(onDemand.staleResultDiscardCount)} |
| No all-hours hot-path allocation | ${passFail(noAllHours)} | allHoursUtciBytesAllocated=${valueOrUnavailable(onDemand.allHoursUtciBytesAllocated)}, allHoursMrtBytesAllocated=${valueOrUnavailable(onDemand.allHoursMrtBytesAllocated)} |
| Tracked VRAM allocation shape | ${passFail(trackedVramPass)} | scope=${valueOrUnavailable(trackedAllocation.trackingScope)}, persistentExposureBytes=${valueOrUnavailable(trackedAllocation.persistentExposureBytes)}, allHoursOutputBytes=${valueOrUnavailable(trackedAllocation.allHoursOutputBytes)}, selectedHourOutputBytes=${valueOrUnavailable(trackedAllocation.selectedHourOutputBytes)}, selectedHourOutputBytesHighWatermark=${valueOrUnavailable(trackedAllocation.selectedHourOutputBytesHighWatermark)} |
| No hot-path \`DataTexture\` rebuild | ${passFail((onDemand.dataTextureBuildCount ?? 0) === 0)} | dataTextureBuildCount=${valueOrUnavailable(onDemand.dataTextureBuildCount)} |
| Render transport honesty | ${passFail(transportHonest)} | renderTransport=${valueOrUnavailable(onDemand.renderTransport)}, utciSurfaceSource=${valueOrUnavailable(onDemand.utciSurfaceSource)} |
| Fallback preserved | ${passFail(fallback?.utciRenderResolved === 'dataTexture')} | utciRenderResolved=${valueOrUnavailable(fallback?.utciRenderResolved)} |

### Notes

- This remains \`f32\` selected-hour on-demand.
- Tracked VRAM is measured as UTCI-owned WebGPU buffer allocation bytes, not total browser/OS VRAM.
- \`cpu-uploaded-selected-hour\` is not zero-copy.
- Packed output and 0.5m production readiness remain deferred.
`;
}
```

Generate the section from captured runtime values. Do not save the section if it contains `not exposed by runtime` for any gate-critical field.

- [ ] **Step 3: Link this plan from strategy analysis**

Add near the existing prototype links in `docs/webgpu_strategy_analysis.md`:

```md
Debug on-demand integration plan: [docs/superpowers/plans/2026-05-08-webgpu-on-demand-debug-integration.md](superpowers/plans/2026-05-08-webgpu-on-demand-debug-integration.md).
```

## Self-Review Checklist

- [ ] No commit steps.
- [ ] No git worktree steps.
- [ ] Default production route is not switched by this plan.
- [ ] `.bin`, `runAll()`, `readUtciBulk()`, and `dataTexture` fallback remain available.
- [ ] Python `.bin` comparison remains active in debug mode.
- [ ] Repeated scrub test proves final selected hour/month owns the final diagnostics.
- [ ] Tracked WebGPU allocation diagnostics prove the on-demand path avoids all-hours UTCI/MRT output buffers and keeps selected-hour output bounded across scrubs.
- [ ] No result says zero-copy unless `renderTransport === 'compute-buffer-selected-hour'`.
- [ ] Packed output remains deferred.
- [ ] Results doc requires captured runtime values, not aspirational text.

## Execution Options

Plan complete and saved to `docs/superpowers/plans/2026-05-08-webgpu-on-demand-debug-integration.md`. Two execution options:

1. **Subagent-Driven (recommended)** - Dispatch a fresh subagent per milestone or task, review between tasks, fast iteration. Every subagent prompt must repeat: no commits and no git worktrees.
2. **Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints.
