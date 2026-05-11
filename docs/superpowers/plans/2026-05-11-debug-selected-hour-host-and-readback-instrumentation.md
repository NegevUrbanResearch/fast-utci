# Debug Selected-Hour Host And Readback Instrumentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Workflow override:** Do **not** create commits. Do **not** create git worktrees. Preserve unrelated dirty files. Every task must report fresh verification before claiming completion.

**Goal:** Make `/debug` a cleaner selected-hour composition root while adding enough visible-path readback instrumentation to honestly promote `strongVisibleGpuPath` when the route proves it.

**Architecture:** Extract legacy-debug selected-hour state/disposal/scheduling into a plain TypeScript host, move debug diagnostics payload normalization into a tested helper, then wire explicit visible-readback instrumentation through main and debug contracts. Keep parity/Python `.bin` behavior debug-only and keep renderer/device/storage-buffer lifecycle in scene-facing modules.

**Tech Stack:** SvelteKit, Svelte 5-compatible Svelte components and stores, TypeScript, WebGPU UTCI compute, Threlte/Three.js, Vitest, Playwright Chromium with `--enable-unsafe-webgpu`, PowerShell on Windows.

---

## Current Baseline

- Active debug route: `viewer/src/routes/debug/+page.svelte`
- Main selected-hour route: `viewer/src/routes/+page.svelte`
- Current selected-hour quality commit: `2c363bc feat(debug): baseline selected-hour runtime quality`
- Current focused verification from 2026-05-11:
  - `npm run test:quality:selected-hour`: PASS, 92 tests.
  - new helper tests: PASS, 22 tests.
  - `npm run test:e2e:selected-hour`: PASS, 12 browser WebGPU tests.
  - `npm run build`: PASS.
  - `git diff --check`: PASS.
  - `npm run check`: FAIL with inherited baseline debt, 129 errors and 4 warnings in 33 files.

## Non-Goals

- Do not remove parity, collect, strict-exposure, `.bin`, Python comparison, or `dataTexture` fallback behavior.
- Do not move renderer/device/storage-buffer lifecycle out of `UTCIPointCloud.svelte`, `ComparisonRenderer.svelte`, `utciSurfaceSync.ts`, or render-bridge/service modules.
- Do not optimize 0.5m performance in this plan.
- Do not do broad Svelte runes migration.
- Do not fix unrelated `npm run check` debt inside this plan.
- Do not claim `strongVisibleGpuPath: true` unless visible-readback instrumentation is explicit and route probes prove it.

## Quality Gates

Stop and report before continuing if any of these happen:

- `/` imports debug/parity/bin helpers or requests `.bin`.
- `/debug` parity mode loses `legacy-debug` or August-only Python `.bin` comparison validity.
- `/debug` normal f32 reports shared-host while legacy selected-hour dispatch or scrub counters increment.
- `strongVisibleGpuPath: true` appears with `readbackInstrumentation: "not-instrumented"`.
- `visibleSelectedHourReadbackCount` is inferred from total selected-hour readbacks instead of explicit visible-path instrumentation.
- A route publishes `strongVisibleGpuPath: true` while `dataTextureBuildCount > 0`, `sameDeviceForComputeAndRender !== true`, request ids mismatch, scene selection mismatches, or `visibleSelectedHourReadbackCount` is absent.
- A selected-hour GPU output buffer can be disposed before render-copy completion, fallback activation, explicit supersession, or route teardown.
- Playwright waits time out without dumping the relevant `window.__utciRenderDiagnostics__` or `window.__onDemandPrototypeDiagnostics__` payload.

## File Structure

### Create

- `viewer/src/lib/debug/debugSelectedHourLegacyHost.ts`
  - Plain TypeScript typed host for legacy-debug selected-hour ownership helpers: accepted output disposal, deferred CPU fallback decisions, scrub scheduling guards, stale-work invalidation, and counter state.
- `viewer/tests/debug/debug-selected-hour-legacy-host.test.ts`
  - Unit tests for disposal, deferred fallback activation decisions, scrub scheduling guards, and counters.
- `viewer/src/lib/debug/debugOnDemandPrototypeDiagnostics.ts`
  - Pure helper that normalizes the debug window diagnostics payload and builds the selected-hour runtime contract.
- `viewer/tests/debug/debug-on-demand-prototype-diagnostics.test.ts`
  - Unit tests for payload merge/replace behavior, debug-only parity fields, shared-host contract fields, and instrumentation promotion.

### Modify

- `viewer/src/routes/debug/+page.svelte`
  - Delegate legacy selected-hour ownership and diagnostics shaping to helpers. Keep URL/store wiring, route mode selection, and scene prop publication in the route.
- `viewer/src/lib/diagnostics/selectedHourRuntimeContract.ts`
  - Tighten explicit readback instrumentation semantics so missing visible-readback counts do not look like proven zero.
- `viewer/tests/diagnostics/selectedHourRuntimeContract.test.ts`
  - Add regression tests for instrumented vs not-instrumented strong-path claims.
- `viewer/src/lib/diagnostics/mainRouteUtciDiagnostics.ts`
  - Accept explicit visible readback instrumentation fields and pass them into the shared contract.
- `viewer/tests/diagnostics/main-route-utci-diagnostics.test.ts`
  - Assert the main route can publish an instrumented strong visible path only when all proof fields are explicit.
- `viewer/src/lib/compute/liveSelectedHourController.ts`
  - Track visible selected-hour readback count separately from range/tooltip/comparison readback reasons.
- `viewer/src/lib/compute/liveSelectedHourRouteHost.ts`
  - Forward visible-readback instrumentation inside the existing per-slot state shape: `state.base` and `state.comparison`.
- `viewer/src/lib/compute/onDemandDiagnostics.ts`
  - Add a distinct `visibleSelectedHourReadbackCount?: number` field. Do not reuse `selectedHourReadbackCount` as visible-path proof.
- `viewer/tests/compute/live-selected-hour-controller.test.ts`
  - Add visible-readback instrumentation tests.
- `viewer/tests/compute/live-selected-hour-route-host.test.ts`
  - Add forwarding tests for visible-readback instrumentation.
- `viewer/tests/e2e/main-route-manual-diagnostics.spec.ts`
  - Promote expectations from honest non-claim to proven strong path when instrumentation is wired.
- `viewer/tests/e2e/debug-route-shared-host-diagnostics.spec.ts`
  - Promote debug shared-host expectations when instrumentation is wired, while preserving zero legacy counters.
- `viewer/tests/e2e/debug-route-parity-runtime-diagnostics.spec.ts`
  - Keep parity behavior locked as `legacy-debug`.
- `viewer/package.json`
  - Optionally add a focused script for the new debug-helper tests if command ergonomics justify it.

### Inspect But Do Not Move Without Separate Approval

- `viewer/src/lib/components/scene/UTCIPointCloud.svelte`
- `viewer/src/lib/components/scene/ComparisonRenderer.svelte`
- `viewer/src/lib/components/scene/utciComputeBufferRenderBridge.ts`
- `viewer/src/lib/services/gpuUtciRenderBridge.ts`
- `viewer/src/lib/components/scene/utciSurfaceSync.ts`

---

## Task 0: Baseline Confirmation

**Files:**
- Inspect: `viewer/src/routes/debug/+page.svelte`
- Inspect: `viewer/src/lib/debug/debugWebgpuUtciDiagnostics.ts`
- Inspect: `viewer/src/lib/diagnostics/selectedHourRuntimeContract.ts`
- Inspect: `viewer/package.json`

- [ ] **Step 1: Record current git state**

Run from repo root:

```powershell
git status --short
git log --oneline -8
```

Expected:

- Working tree state is recorded in task notes.
- No unrelated dirty files are reverted.
- Recent history includes `2c363bc feat(debug): baseline selected-hour runtime quality`.

- [ ] **Step 2: Run current selected-hour verification**

Run:

```powershell
cd viewer
npm run test:quality:selected-hour
npx vitest run tests/compute/selectedHourOutputHandle.test.ts tests/scene/utciComputeBufferRenderBridge.test.ts tests/compute/onDemandDiagnostics.test.ts tests/components/canvasInteractionController.test.ts
npm run test:e2e:selected-hour
npm run build
cd ..
git diff --check
```

Expected:

- All commands pass.
- If any fail, use `superpowers:systematic-debugging` before editing implementation.

- [ ] **Step 3: Capture static baseline without fixing it**

Run:

```powershell
cd viewer
npm run check
```

Expected:

- May fail with inherited repo-wide static debt.
- Record current count and any errors in files this plan will touch.
- Do not fix unrelated static debt in this plan.

---

## Task 1: Extract Legacy Debug Selected-Hour Host

**Files:**
- Create: `viewer/src/lib/debug/debugSelectedHourLegacyHost.ts`
- Create: `viewer/tests/debug/debug-selected-hour-legacy-host.test.ts`
- Modify: `viewer/src/routes/debug/+page.svelte`

- [ ] **Step 1: Write host unit tests**

Create `viewer/tests/debug/debug-selected-hour-legacy-host.test.ts`:

```ts
import { describe, expect, it, vi } from 'vitest';
import {
	createDebugSelectedHourLegacyHost,
	type DebugLegacyAcceptedOutput,
	type DebugLegacyDeferredFallback
} from '$lib/debug/debugSelectedHourLegacyHost';

function createAcceptedOutput(id: number): DebugLegacyAcceptedOutput {
	return {
		requestId: id,
		monthIndex: 7,
		timeIndex: 12,
		output: { gpuBuffer: { destroy: vi.fn() } },
		payload: { kind: 'accepted', id }
	};
}

describe('createDebugSelectedHourLegacyHost', () => {
	it('disposes the previous accepted GPU buffer when superseded', () => {
		const host = createDebugSelectedHourLegacyHost();
		const first = createAcceptedOutput(1);
		const second = createAcceptedOutput(2);

		host.setAcceptedOutput(first);
		host.setAcceptedOutput(second);

		expect(first.output.gpuBuffer.destroy).toHaveBeenCalledTimes(1);
		expect(first.output.gpuBuffer).toBeUndefined();
		expect(second.output.gpuBuffer.destroy).not.toHaveBeenCalled();
	});

	it('does not dispose the same accepted output twice', () => {
		const host = createDebugSelectedHourLegacyHost();
		const output = createAcceptedOutput(1);

		host.setAcceptedOutput(output);
		host.setAcceptedOutput(output);
		host.clearAcceptedOutput();
		host.clearAcceptedOutput();

		expect(output.output.gpuBuffer.destroy).toHaveBeenCalledTimes(1);
	});

	it('activates only the matching deferred CPU fallback', () => {
		const host = createDebugSelectedHourLegacyHost();
		const fallback: DebugLegacyDeferredFallback = {
			requestId: 3,
			monthIndex: 7,
			timeIndex: 12,
			payload: { kind: 'fallback', id: 3 }
		};

		host.setDeferredCpuFallback(fallback);

		expect(host.takeDeferredCpuFallback({ requestId: 4, monthIndex: 7, timeIndex: 12 })).toBeNull();
		expect(host.takeDeferredCpuFallback({ requestId: 3, monthIndex: 7, timeIndex: 12 })).toBe(fallback);
		expect(host.takeDeferredCpuFallback({ requestId: 3, monthIndex: 7, timeIndex: 12 })).toBeNull();
	});

	it('tracks legacy dispatch and scrub scheduling counters', () => {
		const host = createDebugSelectedHourLegacyHost();

		expect(host.getCounters()).toEqual({
			legacySelectedHourDispatchCount: 0,
			legacyScrubScheduleCount: 0
		});

		host.recordDispatch();
		const runId = host.recordScrubSchedule();

		expect(runId).toBe(1);
		expect(host.getCounters()).toEqual({
			legacySelectedHourDispatchCount: 1,
			legacyScrubScheduleCount: 1
		});
	});

	it('invalidates stale scrub work without incrementing scrub schedule counters', () => {
		const host = createDebugSelectedHourLegacyHost();

		const invalidationRunId = host.invalidateScrubSchedule();

		expect(invalidationRunId).toBe(1);
		expect(host.getScrubScheduleRunId()).toBe(1);
		expect(host.getCounters()).toEqual({
			legacySelectedHourDispatchCount: 0,
			legacyScrubScheduleCount: 0
		});
	});
});
```

- [ ] **Step 2: Run the host test to verify it fails**

Run:

```powershell
cd viewer
npx vitest run tests/debug/debug-selected-hour-legacy-host.test.ts
```

Expected: FAIL because `debugSelectedHourLegacyHost.ts` does not exist.

- [ ] **Step 3: Implement the plain TypeScript host**

Create `viewer/src/lib/debug/debugSelectedHourLegacyHost.ts`:

```ts
export interface DebugLegacyAcceptedOutput<TPayload = unknown> {
	requestId: number;
	monthIndex: number;
	timeIndex: number;
	output: { gpuBuffer?: { destroy?: () => void } };
	payload: TPayload;
}

export interface DebugLegacyDeferredFallback<TPayload = unknown> {
	requestId: number;
	monthIndex: number;
	timeIndex: number;
	payload: TPayload;
}

export interface DebugSelectedHourLegacyCounters {
	legacySelectedHourDispatchCount: number;
	legacyScrubScheduleCount: number;
}

export interface DeferredFallbackKey {
	requestId: number;
	monthIndex: number;
	timeIndex: number;
}

export interface DebugSelectedHourLegacyHost<TAcceptedPayload = unknown, TFallbackPayload = unknown> {
	getAcceptedOutput(): DebugLegacyAcceptedOutput<TAcceptedPayload> | null;
	setAcceptedOutput(next: DebugLegacyAcceptedOutput<TAcceptedPayload> | null): void;
	clearAcceptedOutput(): void;
	setDeferredCpuFallback(next: DebugLegacyDeferredFallback<TFallbackPayload> | null): void;
	takeDeferredCpuFallback(key: DeferredFallbackKey): DebugLegacyDeferredFallback<TFallbackPayload> | null;
	recordDispatch(): number;
	recordScrubSchedule(): number;
	invalidateScrubSchedule(): number;
	getScrubScheduleRunId(): number;
	getCounters(): DebugSelectedHourLegacyCounters;
	resetCounters(): void;
	dispose(): void;
}

function destroyAcceptedOutput(output: DebugLegacyAcceptedOutput | null): void {
	const buffer = output?.output.gpuBuffer;
	buffer?.destroy?.();
	if (output?.output && 'gpuBuffer' in output.output) {
		output.output.gpuBuffer = undefined;
	}
}

export function createDebugSelectedHourLegacyHost<
	TAcceptedPayload = unknown,
	TFallbackPayload = unknown
>(): DebugSelectedHourLegacyHost<TAcceptedPayload, TFallbackPayload> {
	let acceptedOutput: DebugLegacyAcceptedOutput<TAcceptedPayload> | null = null;
	let deferredCpuFallback: DebugLegacyDeferredFallback<TFallbackPayload> | null = null;
	let legacySelectedHourDispatchCount = 0;
	let legacyScrubScheduleCount = 0;
	let scrubScheduleRunId = 0;

	return {
		getAcceptedOutput() {
			return acceptedOutput;
		},
		setAcceptedOutput(next) {
			const previous = acceptedOutput;
			if (previous && previous !== next) {
				destroyAcceptedOutput(previous);
			}
			acceptedOutput = next;
		},
		clearAcceptedOutput() {
			this.setAcceptedOutput(null);
		},
		setDeferredCpuFallback(next) {
			deferredCpuFallback = next;
		},
		takeDeferredCpuFallback(key) {
			const fallback = deferredCpuFallback;
			if (
				fallback &&
				fallback.requestId === key.requestId &&
				fallback.monthIndex === key.monthIndex &&
				fallback.timeIndex === key.timeIndex
			) {
				deferredCpuFallback = null;
				return fallback;
			}
			return null;
		},
		recordDispatch() {
			legacySelectedHourDispatchCount += 1;
			return legacySelectedHourDispatchCount;
		},
		recordScrubSchedule() {
			legacyScrubScheduleCount += 1;
			scrubScheduleRunId += 1;
			return scrubScheduleRunId;
		},
		invalidateScrubSchedule() {
			scrubScheduleRunId += 1;
			return scrubScheduleRunId;
		},
		getScrubScheduleRunId() {
			return scrubScheduleRunId;
		},
		getCounters() {
			return {
				legacySelectedHourDispatchCount,
				legacyScrubScheduleCount
			};
		},
		resetCounters() {
			legacySelectedHourDispatchCount = 0;
			legacyScrubScheduleCount = 0;
			scrubScheduleRunId = 0;
		},
		dispose() {
			destroyAcceptedOutput(acceptedOutput);
			acceptedOutput = null;
			deferredCpuFallback = null;
		}
	};
}
```

- [ ] **Step 4: Run host tests**

Run:

```powershell
cd viewer
npx vitest run tests/debug/debug-selected-hour-legacy-host.test.ts
```

Expected: PASS.

- [ ] **Step 5: Delegate route-owned legacy state to the host**

In `viewer/src/routes/debug/+page.svelte`:

- Import `createDebugSelectedHourLegacyHost`.
- Replace these route-owned primitives with host accessors where behavior is equivalent:
  - `legacySelectedHourDispatchCount`
  - `legacyScrubScheduleCount`
  - `debugOnDemandScrubScheduleRunId`
  - `acceptedGpuResidentUtciOutput`
  - `deferredCpuFallbackSelectedHour`
  - `destroyOnDemandGpuBuffer`
  - `setAcceptedGpuResidentUtciOutput`
- Keep route-specific construction of `Analysis`, `comparisonStore`, and Svelte reactive assignments in the route.
- Use host counters in `buildDebugSelectedHourDispatchCounters(...)`.
- Use `host.recordScrubSchedule()` inside `scheduleDebugOnDemandScrubRecompute(...)`.
- Use `host.takeDeferredCpuFallback(...)` inside `activateDeferredCpuFallbackIfAvailable(...)`.

Expected route-level shape:

```ts
const debugSelectedHourLegacyHost = createDebugSelectedHourLegacyHost<
	AcceptedGpuResidentUtciOutput,
	DeferredCpuFallbackSelectedHour
>();

function getAcceptedGpuResidentUtciOutput(): AcceptedGpuResidentUtciOutput | null {
	return debugSelectedHourLegacyHost.getAcceptedOutput()?.payload ?? null;
}

function setAcceptedGpuResidentUtciOutput(next: AcceptedGpuResidentUtciOutput | null): void {
	debugSelectedHourLegacyHost.setAcceptedOutput(
		next
			? {
					requestId: next.requestId,
					monthIndex: next.monthIndex,
					timeIndex: next.timeIndex,
					output: next.output,
					payload: next
				}
			: null
	);
}
```

When a Svelte reactive statement needs the accepted output, assign a local variable from `getAcceptedGpuResidentUtciOutput()` so the route remains readable.

Use `debugSelectedHourLegacyHost.invalidateScrubSchedule()` for stale-work invalidations that currently only bump `debugOnDemandScrubScheduleRunId`. Use `recordScrubSchedule()` only when the route actually schedules a legacy scrub and should increment `legacyScrubScheduleCount`.

- [ ] **Step 6: Run focused route and debug tests**

Run:

```powershell
cd viewer
npx vitest run tests/debug/debug-selected-hour-legacy-host.test.ts tests/debug/debug-webgpu-utci-diagnostics.test.ts tests/debug/debug-selected-hour-mode.test.ts
npm run test:e2e:selected-hour
```

Expected:

- PASS.
- `/debug` parity remains `legacy-debug`.
- `/debug` normal f32 shared-host still has zero legacy dispatch/scrub overlap.

---

## Task 2: Extract Debug Diagnostics Payload Normalization

**Files:**
- Create: `viewer/src/lib/debug/debugOnDemandPrototypeDiagnostics.ts`
- Create: `viewer/tests/debug/debug-on-demand-prototype-diagnostics.test.ts`
- Modify: `viewer/src/routes/debug/+page.svelte`

- [ ] **Step 1: Write diagnostics helper tests**

Create `viewer/tests/debug/debug-on-demand-prototype-diagnostics.test.ts`:

```ts
import { describe, expect, it } from 'vitest';
import { createEmptyOnDemandDiagnostics } from '$lib/compute/onDemandDiagnostics';
import { buildDebugOnDemandPrototypeDiagnostics } from '$lib/debug/debugOnDemandPrototypeDiagnostics';

describe('buildDebugOnDemandPrototypeDiagnostics', () => {
	it('merges existing diagnostics and preserves debug-only parity validity', () => {
		const existing = {
			...createEmptyOnDemandDiagnostics(),
			navigatorGpu: true,
			selectedHourEngine: 'legacy-debug' as const,
			binComparisonEnabled: true,
			binComparisonValid: true
		};

		const result = buildDebugOnDemandPrototypeDiagnostics({
			existing,
			patch: { renderTransport: 'compute-buffer-selected-hour' },
			defaults: {
				navigatorGpu: true,
				rendererBackend: 'webgpu',
				utciRenderRequested: 'utci',
				utciRenderResolved: 'gpuNative',
				selectedHourEngine: 'legacy-debug',
				binComparisonEnabled: true,
				binComparisonValid: true,
				legacySelectedHourDispatchCount: 1,
				legacyScrubScheduleCount: 1,
				tooltipInteraction: { hoverSampleCount: 0, disabled: false },
				cameraInteraction: { wheelEventCount: 0 },
				readbackInstrumentation: 'not-instrumented'
			}
		});

		expect(result.binComparisonEnabled).toBe(true);
		expect(result.binComparisonValid).toBe(true);
		expect(result.selectedHourRuntimeContract.selectedHourEngine).toBe('legacy-debug');
		expect(result.selectedHourRuntimeContract.strongVisibleGpuPath).toBe(false);
	});

	it('keeps shared-host diagnostics conservative before visible-readback instrumentation exists', () => {
		const result = buildDebugOnDemandPrototypeDiagnostics({
			replace: true,
			patch: {
				selectedHourEngine: 'shared-host',
				renderTransport: 'compute-buffer-selected-hour',
				utciSurfaceSource: 'compute-buffer-selected-hour',
				sameDeviceForComputeAndRender: true,
				dataTextureBuildCount: 0,
				surfaceRequestId: 4,
				sceneSurfaceRequestId: 4,
				selectionKey: 'analysis|7|12',
				sceneSelectionKey: 'analysis|7|12',
				selectedHourReadbackReasons: ['range'],
				selectedHourReadbackReasonCounts: { range: 1 }
			},
			defaults: {
				navigatorGpu: true,
				rendererBackend: 'webgpu',
				utciRenderRequested: 'utci',
				utciRenderResolved: 'gpuNative',
				selectedHourEngine: 'shared-host',
				binComparisonEnabled: false,
				binComparisonValid: false,
				legacySelectedHourDispatchCount: 0,
				legacyScrubScheduleCount: 0,
				tooltipInteraction: { hoverSampleCount: 1, disabled: false },
				cameraInteraction: { wheelEventCount: 1 },
				readbackInstrumentation: 'not-instrumented'
			}
		});

		expect(result.selectedHourRuntimeContract).toMatchObject({
			selectedHourEngine: 'shared-host',
			readbackInstrumentation: 'not-instrumented',
			strongVisibleGpuPath: false,
			readbackReasons: ['range']
		});
	});

	it('does not make a strong claim when instrumentation is missing', () => {
		const result = buildDebugOnDemandPrototypeDiagnostics({
			replace: true,
			patch: {
				selectedHourEngine: 'shared-host',
				renderTransport: 'compute-buffer-selected-hour',
				utciSurfaceSource: 'compute-buffer-selected-hour',
				sameDeviceForComputeAndRender: true,
				dataTextureBuildCount: 0,
				visibleSelectedHourReadbackCount: 0,
				surfaceRequestId: 4,
				sceneSurfaceRequestId: 4,
				selectionKey: 'analysis|7|12',
				sceneSelectionKey: 'analysis|7|12'
			},
			defaults: {
				navigatorGpu: true,
				rendererBackend: 'webgpu',
				utciRenderRequested: 'utci',
				utciRenderResolved: 'gpuNative',
				selectedHourEngine: 'shared-host',
				binComparisonEnabled: false,
				binComparisonValid: false,
				legacySelectedHourDispatchCount: 0,
				legacyScrubScheduleCount: 0,
				tooltipInteraction: { hoverSampleCount: 0, disabled: false },
				cameraInteraction: { wheelEventCount: 0 },
				readbackInstrumentation: 'not-instrumented'
			}
		});

		expect(result.selectedHourRuntimeContract.strongVisibleGpuPath).toBe(false);
	});
});
```

- [ ] **Step 2: Run diagnostics helper tests to verify they fail**

Run:

```powershell
cd viewer
npx vitest run tests/debug/debug-on-demand-prototype-diagnostics.test.ts
```

Expected: FAIL because the helper does not exist.

- [ ] **Step 3: Implement diagnostics helper**

Create `viewer/src/lib/debug/debugOnDemandPrototypeDiagnostics.ts` with this API:

```ts
import {
	createEmptyOnDemandDiagnostics,
	type OnDemandPrototypeDiagnostics
} from '$lib/compute/onDemandDiagnostics';
import {
	buildSelectedHourRuntimeContract,
	type SelectedHourReadbackInstrumentation
} from '$lib/diagnostics/selectedHourRuntimeContract';

export interface DebugOnDemandPrototypeDiagnosticsDefaults {
	navigatorGpu: boolean;
	rendererBackend: OnDemandPrototypeDiagnostics['rendererBackend'];
	utciRenderRequested: OnDemandPrototypeDiagnostics['utciRenderRequested'];
	utciRenderResolved: OnDemandPrototypeDiagnostics['utciRenderResolved'];
	selectedHourEngine: NonNullable<OnDemandPrototypeDiagnostics['selectedHourEngine']>;
	binComparisonEnabled: boolean;
	binComparisonValid: boolean;
	legacySelectedHourDispatchCount: number;
	legacyScrubScheduleCount: number;
	tooltipInteraction: NonNullable<OnDemandPrototypeDiagnostics['tooltipInteraction']>;
	cameraInteraction: NonNullable<OnDemandPrototypeDiagnostics['cameraInteraction']>;
	readbackInstrumentation: SelectedHourReadbackInstrumentation;
}

export interface BuildDebugOnDemandPrototypeDiagnosticsParams {
	existing?: OnDemandPrototypeDiagnostics;
	patch: Partial<OnDemandPrototypeDiagnostics>;
	defaults: DebugOnDemandPrototypeDiagnosticsDefaults;
	replace?: boolean;
}

function normalizeTransport(
	value: OnDemandPrototypeDiagnostics['renderTransport'] | OnDemandPrototypeDiagnostics['utciSurfaceSource']
) {
	return value === 'compute-buffer-selected-hour' || value === 'cpu-uploaded-selected-hour'
		? value
		: 'none';
}

export function buildDebugOnDemandPrototypeDiagnostics(
	params: BuildDebugOnDemandPrototypeDiagnosticsParams
): OnDemandPrototypeDiagnostics {
	const existing = params.replace ? undefined : params.existing;
	const patch = params.patch;
	const defaults = params.defaults;
	const nextDiagnostics: OnDemandPrototypeDiagnostics = {
		...createEmptyOnDemandDiagnostics(),
		...existing,
		...patch,
		navigatorGpu: patch.navigatorGpu ?? existing?.navigatorGpu ?? defaults.navigatorGpu,
		rendererBackend: patch.rendererBackend ?? existing?.rendererBackend ?? defaults.rendererBackend,
		utciRenderRequested:
			patch.utciRenderRequested ?? existing?.utciRenderRequested ?? defaults.utciRenderRequested,
		utciRenderResolved:
			patch.utciRenderResolved ?? existing?.utciRenderResolved ?? defaults.utciRenderResolved,
		utciSurfaceSource:
			'utciSurfaceSource' in patch ? patch.utciSurfaceSource : existing?.utciSurfaceSource,
		selectedHourEngine:
			patch.selectedHourEngine ?? existing?.selectedHourEngine ?? defaults.selectedHourEngine,
		binComparisonEnabled:
			patch.binComparisonEnabled ??
			existing?.binComparisonEnabled ??
			defaults.binComparisonEnabled,
		binComparisonValid:
			patch.binComparisonValid ?? existing?.binComparisonValid ?? defaults.binComparisonValid,
		legacySelectedHourDispatchCount:
			patch.legacySelectedHourDispatchCount ?? defaults.legacySelectedHourDispatchCount,
		legacyScrubScheduleCount:
			patch.legacyScrubScheduleCount ?? defaults.legacyScrubScheduleCount,
		tooltipInteraction:
			patch.tooltipInteraction ?? existing?.tooltipInteraction ?? defaults.tooltipInteraction,
		cameraInteraction:
			patch.cameraInteraction ?? existing?.cameraInteraction ?? defaults.cameraInteraction
	};

	nextDiagnostics.selectedHourRuntimeContract =
		patch.selectedHourRuntimeContract ??
		buildSelectedHourRuntimeContract({
			route: 'debug',
			selectedHourEngine: nextDiagnostics.selectedHourEngine ?? defaults.selectedHourEngine,
			renderTransport: normalizeTransport(nextDiagnostics.renderTransport),
			utciSurfaceSource: normalizeTransport(nextDiagnostics.utciSurfaceSource),
			sameDeviceForComputeAndRender: nextDiagnostics.sameDeviceForComputeAndRender === true,
			dataTextureBuildCount: nextDiagnostics.dataTextureBuildCount,
			visibleSelectedHourReadbackCount:
				'visibleSelectedHourReadbackCount' in nextDiagnostics
					? nextDiagnostics.visibleSelectedHourReadbackCount
					: undefined,
			readbackInstrumentation: defaults.readbackInstrumentation,
			legacySelectedHourDispatchCount: nextDiagnostics.legacySelectedHourDispatchCount,
			legacyScrubScheduleCount: nextDiagnostics.legacyScrubScheduleCount,
			requestId: nextDiagnostics.surfaceRequestId,
			sceneRequestId: nextDiagnostics.sceneSurfaceRequestId,
			selectionKey: nextDiagnostics.selectionKey,
			sceneSelectionKey: nextDiagnostics.sceneSelectionKey,
			readbackReasons: nextDiagnostics.selectedHourReadbackReasons,
			readbackReasonCounts: nextDiagnostics.selectedHourReadbackReasonCounts
		});

	return nextDiagnostics;
}
```

If current type names differ, adapt only the import/type annotations while preserving the behavior and tests.

- [ ] **Step 4: Replace route diagnostics merge with helper**

In `viewer/src/routes/debug/+page.svelte`, update `updateOnDemandPrototypeDiagnostics(...)` so it:

- still guards on `browser`, `onDemandPrototypeEnabled`, and `shouldExposeDebugWindowDiagnostics(debugDiagnosticsState)`;
- still computes feasibility diagnostics using `buildGpuResidentRenderDiagnosticsPatch(...)`;
- calls `buildDebugOnDemandPrototypeDiagnostics(...)` with `existing`, merged `patch`, and route defaults;
- writes the returned payload to `win.__onDemandPrototypeDiagnostics__`;
- keeps status derivation in the route for now.

Expected route call shape:

```ts
const nextDiagnostics = buildDebugOnDemandPrototypeDiagnostics({
	existing,
	replace: options?.replace,
	patch: {
		...feasibilityDiagnostics,
		...diagnostics
	},
	defaults: {
		navigatorGpu: Boolean(navigator.gpu),
		rendererBackend: 'unknown',
		utciRenderRequested: utciRenderMode,
		utciRenderResolved: resolvedUtciSurfaceBackend,
		selectedHourEngine: useDebugSharedSelectedHourHost
			? existing?.selectedHourEngine ?? debugDiagnosticsState.selectedHourEngine
			: debugDiagnosticsState.selectedHourEngine,
		binComparisonEnabled: debugDiagnosticsState.binComparisonEnabled,
		binComparisonValid: debugDiagnosticsState.binComparisonValid,
		...debugSelectedHourLegacyHost.getCounters(),
		tooltipInteraction: createEmptyTooltipInteractionDiagnostics(debugTooltipHoverDisabled),
		cameraInteraction: cameraInteractionTelemetry.diagnostics,
		readbackInstrumentation: debugVisibleReadbackInstrumentation
	}
});
```

In this task, `debugVisibleReadbackInstrumentation` may remain `'not-instrumented'`. Task 3 promotes it.

Before Task 3, `visibleSelectedHourReadbackCount` is not yet part of the diagnostics type. If TypeScript rejects the `'visibleSelectedHourReadbackCount' in nextDiagnostics` guard in this task, leave the helper passing `undefined` for `visibleSelectedHourReadbackCount` and add the guarded field in Task 3. The key rule is unchanged: absence is intentional and must keep `strongVisibleGpuPath` false. Do not backfill it from `selectedHourReadbackCount`.

- [ ] **Step 5: Run focused diagnostics and route probes**

Run:

```powershell
cd viewer
npx vitest run tests/debug/debug-on-demand-prototype-diagnostics.test.ts tests/debug/debug-webgpu-utci-diagnostics.test.ts tests/diagnostics/selectedHourRuntimeContract.test.ts
npm run test:e2e:selected-hour
```

Expected:

- PASS.
- Contract fields are unchanged except for helper ownership.
- `/debug` parity still uses `legacy-debug`.

---

## Task 3: Add Explicit Visible Readback Instrumentation

**Files:**
- Modify: `viewer/src/lib/diagnostics/selectedHourRuntimeContract.ts`
- Modify: `viewer/tests/diagnostics/selectedHourRuntimeContract.test.ts`
- Modify: `viewer/src/lib/diagnostics/mainRouteUtciDiagnostics.ts`
- Modify: `viewer/src/lib/compute/onDemandDiagnostics.ts`
- Modify: `viewer/tests/diagnostics/main-route-utci-diagnostics.test.ts`
- Modify: `viewer/src/lib/compute/liveSelectedHourController.ts`
- Modify: `viewer/tests/compute/live-selected-hour-controller.test.ts`
- Modify: `viewer/src/lib/compute/liveSelectedHourRouteHost.ts`
- Modify: `viewer/tests/compute/live-selected-hour-route-host.test.ts`
- Modify: `viewer/src/routes/+page.svelte`
- Modify: `viewer/src/routes/debug/+page.svelte`
- Modify: `viewer/tests/e2e/main-route-manual-diagnostics.spec.ts`
- Modify: `viewer/tests/e2e/debug-route-shared-host-diagnostics.spec.ts`
- Modify: `viewer/tests/e2e/debug-route-parity-runtime-diagnostics.spec.ts`

- [ ] **Step 1: Strengthen contract tests before implementation**

Extend `viewer/tests/diagnostics/selectedHourRuntimeContract.test.ts` with:

```ts
it('does not treat missing visible-readback count as proven zero', () => {
	const contract = buildSelectedHourRuntimeContract({
		route: 'main',
		selectedHourEngine: 'shared-host',
		renderTransport: 'compute-buffer-selected-hour',
		utciSurfaceSource: 'compute-buffer-selected-hour',
		sameDeviceForComputeAndRender: true,
		dataTextureBuildCount: 0,
		readbackInstrumentation: 'instrumented',
		requestId: 1,
		sceneRequestId: 1,
		selectionKey: 'a|7|12',
		sceneSelectionKey: 'a|7|12'
	});

	expect(contract.visibleRenderPathAvoidsCpuReadback).toBe(false);
	expect(contract.strongVisibleGpuPath).toBe(false);
});

it('allows a strong visible path with explicit zero visible readbacks and auxiliary readback reasons', () => {
	const contract = buildSelectedHourRuntimeContract({
		route: 'main',
		selectedHourEngine: 'shared-host',
		renderTransport: 'compute-buffer-selected-hour',
		utciSurfaceSource: 'compute-buffer-selected-hour',
		sameDeviceForComputeAndRender: true,
		dataTextureBuildCount: 0,
		visibleSelectedHourReadbackCount: 0,
		readbackInstrumentation: 'instrumented',
		requestId: 1,
		sceneRequestId: 1,
		selectionKey: 'a|7|12',
		sceneSelectionKey: 'a|7|12',
		readbackReasons: ['range', 'tooltip'],
		readbackReasonCounts: { range: 1, tooltip: 1 }
	});

	expect(contract.visibleRenderPathAvoidsCpuReadback).toBe(true);
	expect(contract.strongVisibleGpuPath).toBe(true);
	expect(contract.totalSelectedHourReadbackReasonCount).toBe(2);
});
```

- [ ] **Step 2: Run contract test to verify the missing-count test fails if needed**

Run:

```powershell
cd viewer
npx vitest run tests/diagnostics/selectedHourRuntimeContract.test.ts
```

Expected:

- If the first new test already passes, keep it as a regression.
- If it fails, continue to Step 3 before broad changes.

- [ ] **Step 3: Make missing visible-readback counts explicit in the contract**

In `viewer/src/lib/diagnostics/selectedHourRuntimeContract.ts`:

- Add `visibleSelectedHourReadbackCountInstrumented: boolean` to `SelectedHourRuntimeContract`.
- Keep `visibleSelectedHourReadbackCount` numeric for compatibility.
- Compute `hasExplicitVisibleReadbackCount` before defaulting is interpreted.
- The current contract already uses the right strong-path predicate. Do not churn that logic unless the new regression tests expose a failure.
- Keep `visibleRenderPathAvoidsCpuReadback` requiring:
  - `readbackInstrumentation === 'instrumented'`
  - explicit visible-readback count
  - explicit data texture count
  - `renderTransport === 'compute-buffer-selected-hour'`
  - `utciSurfaceSource === 'compute-buffer-selected-hour'`
  - `visibleSelectedHourReadbackCount === 0`
  - `dataTextureBuildCount === 0`

Expected code shape:

```ts
const hasExplicitVisibleReadbackCount = typeof inputs.visibleSelectedHourReadbackCount === 'number';
const visibleSelectedHourReadbackCount = inputs.visibleSelectedHourReadbackCount ?? 0;

return {
	...
	visibleSelectedHourReadbackCount,
	visibleSelectedHourReadbackCountInstrumented: hasExplicitVisibleReadbackCount,
	...
};
```

- [ ] **Step 4: Add controller/host tests for visible instrumentation**

In `viewer/tests/compute/live-selected-hour-controller.test.ts`, add a test near existing accepted visible request tests:

```ts
it('tracks explicit zero visible readbacks for compute-buffer visible surfaces', async () => {
	const controller = createTestController();

	await controller.runSelectedHourRequest(createCompleteComputeBufferRequest({ requestId: 1 }));

	expect(controller.getState()).toMatchObject({
		visibleSelectedHourReadbackCount: undefined,
		readbackInstrumentation: 'not-instrumented'
	});

	controller.handleRenderSurfaceDiagnostics({
		status: 'complete',
		requestId: 1,
		renderTransport: 'compute-buffer-selected-hour',
		utciSurfaceSource: 'compute-buffer-selected-hour',
		sceneSurfaceRequestId: 1,
		sceneSelectionKey: 'analysis|7|12',
		sameDeviceForComputeAndRender: true,
		dataTextureBuildCount: 0
	});

	expect(controller.getState()).toMatchObject({
		visibleSelectedHourReadbackCount: 0,
		readbackInstrumentation: 'instrumented'
	});
});
```

In `viewer/tests/compute/live-selected-hour-route-host.test.ts`, add:

```ts
it('forwards visible-readback instrumentation to route state', async () => {
	const host = createTestRouteHost();

	await runHostToComputeBufferVisibleState(host);

	expect(host.getState().base).toMatchObject({
		visibleSelectedHourReadbackCount: 0,
		readbackInstrumentation: 'instrumented'
	});
});
```

Use the existing test helper names in each file. If the exact helper names differ, reuse the local helper that already drives the controller/host to a `compute-buffer-selected-hour` visible state. The route host state is per-slot: assert on `host.getState().base` for the main/base slot and `host.getState().comparison` for comparison-slot cases, not on top-level `host.getState()`. The test must drive render-surface diagnostics to a copy-complete state before expecting explicit zero visible readbacks; request completion alone is still pending/awaiting scene copy.

- [ ] **Step 5: Run controller/host tests and confirm failure**

Run:

```powershell
cd viewer
npx vitest run tests/compute/live-selected-hour-controller.test.ts tests/compute/live-selected-hour-route-host.test.ts
```

Expected: FAIL until visible instrumentation fields are implemented.

- [ ] **Step 6: Implement visible instrumentation in controller and host**

In `viewer/src/lib/compute/liveSelectedHourController.ts`:

- Add state fields:

```ts
visibleSelectedHourReadbackCount: number | undefined;
readbackInstrumentation: 'instrumented' | 'not-instrumented';
```

- Initialize as:

```ts
visibleSelectedHourReadbackCount: undefined,
readbackInstrumentation: 'not-instrumented',
```

- When render-surface diagnostics report a copy-complete accepted visible surface with `compute-buffer-selected-hour`, set:

```ts
visibleSelectedHourReadbackCount: 0,
readbackInstrumentation: 'instrumented',
```

- When the accepted visible surface falls back through CPU upload, set:

```ts
visibleSelectedHourReadbackCount: 1,
readbackInstrumentation: 'instrumented',
```

In `viewer/src/lib/compute/liveSelectedHourRouteHost.ts`:

- Do not add top-level route-host fields.
- Preserve the current `LiveSelectedHourRouteState` shape: `base` and `comparison` are `LiveSelectedHourControllerState`.
- Forward the new fields by keeping them on each controller state.
- Reset `state.base.visibleSelectedHourReadbackCount` / `state.comparison.visibleSelectedHourReadbackCount` to `undefined` and `readbackInstrumentation` to `'not-instrumented'` when the relevant slot is disabled or reset.

- [ ] **Step 7: Wire diagnostics builders**

In `viewer/src/lib/compute/onDemandDiagnostics.ts`, add a distinct optional field to `OnDemandRuntimeDiagnostics`:

```ts
visibleSelectedHourReadbackCount?: number;
```

Do not change the meaning of `selectedHourReadbackCount` if it exists elsewhere in route diagnostics. It is not the visible-path proof field.

If Task 2 left the debug diagnostics helper passing `undefined` because this field did not exist yet, update that helper now to read `nextDiagnostics.visibleSelectedHourReadbackCount`.

In `viewer/src/lib/diagnostics/mainRouteUtciDiagnostics.ts`:

- Add inputs:

```ts
visibleSelectedHourReadbackCount?: number;
readbackInstrumentation?: SelectedHourReadbackInstrumentation;
```

- Pass them to `buildSelectedHourRuntimeContract(...)`:

```ts
visibleSelectedHourReadbackCount: inputs.visibleSelectedHourReadbackCount,
readbackInstrumentation: inputs.readbackInstrumentation ?? 'not-instrumented',
```

In `viewer/src/routes/+page.svelte`:

- Pass the route-host visible instrumentation fields into `buildMainRouteUtciDiagnostics(...)`.

In `viewer/src/routes/debug/+page.svelte`:

- For shared-host mode, pass the shared-host visible instrumentation fields to `buildDebugOnDemandPrototypeDiagnostics(...)`.
- For legacy-debug/parity mode, keep `readbackInstrumentation: 'not-instrumented'` unless the legacy path has explicit visible-readback instrumentation in the same shape.
- Populate `visibleSelectedHourReadbackCount` only from the shared-host controller/route-host visible instrumentation field. Do not derive it from `selectedHourReadbackCount`, `debugReadbackCount`, readback reasons, or missing diagnostics.

- [ ] **Step 8: Update route diagnostics tests**

In `viewer/tests/diagnostics/main-route-utci-diagnostics.test.ts`, add:

```ts
it('publishes a strong visible GPU path when visible readbacks are explicitly instrumented', () => {
	const diagnostics = buildMainRouteUtciDiagnostics({
		enabled: true,
		utciOnDemand: 'f32',
		utciRenderRequested: 'utci',
		utciRenderResolved: 'gpuNative',
		navigatorGpu: true,
		rendererBackend: 'webgpu',
		sameDeviceForComputeAndRender: true,
		dataTextureBuildCount: 0,
		visibleSelectedHourReadbackCount: 0,
		readbackInstrumentation: 'instrumented',
		baseSurfaceRequestId: 3,
		baseSceneSurfaceRequestId: 3,
		baseSelectionKey: 'analysis|7|12',
		baseSceneSelectionKey: 'analysis|7|12',
		baseRenderTransport: 'compute-buffer-selected-hour',
		baseUtciSurfaceDiagnostics: {
			utciSurfaceSource: 'compute-buffer-selected-hour'
		}
	});

	expect(diagnostics?.selectedHourRuntimeContract).toMatchObject({
		readbackInstrumentation: 'instrumented',
		visibleSelectedHourReadbackCount: 0,
		visibleSelectedHourReadbackCountInstrumented: true,
		strongVisibleGpuPath: true
	});
});
```

If required fields differ, use the existing local diagnostics fixture and override only the instrumentation/proof fields.

- [ ] **Step 9: Promote E2E expectations**

In `viewer/tests/e2e/main-route-manual-diagnostics.spec.ts`, update both the wait predicate near the top of the file and the main proof assertions to expect instrumented strong-path diagnostics. The wait predicate must not keep requiring `readbackInstrumentation === 'not-instrumented'`.

Expected wait predicate condition:

```ts
value.selectedHourRuntimeContract?.route === 'main' &&
value.selectedHourRuntimeContract?.readbackInstrumentation === 'instrumented' &&
value.selectedHourRuntimeContract?.visibleSelectedHourReadbackCount === 0 &&
value.selectedHourRuntimeContract?.strongVisibleGpuPath === true
```

Expected proof assertion:

```ts
expect(value.selectedHourRuntimeContract).toMatchObject({
	route: 'main',
	selectedHourEngine: 'shared-host',
	readbackInstrumentation: 'instrumented',
	visibleSelectedHourReadbackCount: 0,
	visibleSelectedHourReadbackCountInstrumented: true,
	strongVisibleGpuPath: true
});
```

In `viewer/tests/e2e/debug-route-shared-host-diagnostics.spec.ts`, update both the shared-host wait predicate and normal f32 shared-host assertions. The wait predicate must not keep requiring `readbackInstrumentation === 'not-instrumented'`.

Expected wait predicate condition:

```ts
value.selectedHourRuntimeContract?.route === 'debug' &&
value.selectedHourRuntimeContract?.selectedHourEngine === 'shared-host' &&
value.selectedHourRuntimeContract?.readbackInstrumentation === 'instrumented' &&
value.selectedHourRuntimeContract?.visibleSelectedHourReadbackCount === 0 &&
value.selectedHourRuntimeContract?.strongVisibleGpuPath === true
```

Expected proof assertion:

```ts
expect(initial.selectedHourRuntimeContract).toMatchObject({
	route: 'debug',
	selectedHourEngine: 'shared-host',
	readbackInstrumentation: 'instrumented',
	visibleSelectedHourReadbackCount: 0,
	visibleSelectedHourReadbackCountInstrumented: true,
	strongVisibleGpuPath: true,
	legacySelectedHourDispatchCount: 0,
	legacyScrubScheduleCount: 0
});
```

In `viewer/tests/e2e/debug-route-parity-runtime-diagnostics.spec.ts`, keep parity assertions conservative:

```ts
expect(diagnostics.selectedHourRuntimeContract).toMatchObject({
	route: 'debug',
	selectedHourEngine: 'legacy-debug',
	readbackInstrumentation: 'not-instrumented',
	strongVisibleGpuPath: false
});
```

- [ ] **Step 10: Run focused instrumentation tests**

Run:

```powershell
cd viewer
npx vitest run tests/diagnostics/selectedHourRuntimeContract.test.ts tests/diagnostics/main-route-utci-diagnostics.test.ts tests/compute/live-selected-hour-controller.test.ts tests/compute/live-selected-hour-route-host.test.ts tests/debug/debug-on-demand-prototype-diagnostics.test.ts
```

Expected: PASS.

- [ ] **Step 11: Run browser route proof**

Run:

```powershell
cd viewer
npm run test:e2e:selected-hour
```

Expected:

- PASS.
- `/` publishes `strongVisibleGpuPath: true`.
- `/debug` normal f32 publishes `strongVisibleGpuPath: true` with zero legacy overlap.
- `/debug` parity remains `legacy-debug`, Python `.bin` remains August-only, and strong path remains false there unless explicitly instrumented later.

---

## Task 4: Final Verification And Review Stop

**Files:**
- Inspect: all files changed by Tasks 1-3.

- [ ] **Step 1: Run final focused units**

Run:

```powershell
cd viewer
npm run test:quality:selected-hour
npx vitest run tests/debug/debug-selected-hour-legacy-host.test.ts tests/debug/debug-on-demand-prototype-diagnostics.test.ts tests/compute/selectedHourOutputHandle.test.ts tests/scene/utciComputeBufferRenderBridge.test.ts tests/compute/onDemandDiagnostics.test.ts tests/components/canvasInteractionController.test.ts
```

Expected: PASS.

- [ ] **Step 2: Run final browser proof**

Run:

```powershell
cd viewer
npm run test:e2e:selected-hour
```

Expected: PASS.

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
- Report exact current count and whether any newly touched file has new errors.

- [ ] **Step 4: Run whitespace diff guard**

Run from repo root:

```powershell
git diff --check
```

Expected: PASS.

- [ ] **Step 5: Request implementation review subagents**

Ask two review subagents:

```text
Review the debug selected-hour host and diagnostics extraction. Focus on whether /debug legacy-debug behavior stayed parity-only, whether normal f32 shared-host has zero legacy overlap, whether accepted GPU buffers cannot be disposed early, and whether the route is now a cleaner composition root. Return findings first with file/line evidence. Do not edit files.
```

```text
Review visible readback instrumentation and selected-hour runtime contracts. Focus on whether strongVisibleGpuPath can only become true with explicit visible-readback instrumentation, same-device render/compute, zero dataTexture builds, matching request/scene ids, matching selection keys, and no legacy debug overlap. Return findings first with file/line evidence. Do not edit files.
```

- [ ] **Step 6: Stop for human review**

Report:

- files changed;
- verification commands and results;
- `npm run check` current count;
- any touched-file static errors and whether they are new or inherited;
- reviewer findings;
- whether `strongVisibleGpuPath` is now truly proven for `/` and `/debug` normal f32.

Do not commit. Do not continue into static debt cleanup, debug UI decomposition, or 0.5m performance work without user approval.

---

## Final Verification Commands

Run these before claiming this plan is complete:

```powershell
cd viewer
npm run test:quality:selected-hour
npx vitest run tests/debug/debug-selected-hour-legacy-host.test.ts tests/debug/debug-on-demand-prototype-diagnostics.test.ts tests/compute/selectedHourOutputHandle.test.ts tests/scene/utciComputeBufferRenderBridge.test.ts tests/compute/onDemandDiagnostics.test.ts tests/components/canvasInteractionController.test.ts
npm run test:e2e:selected-hour
npm run build
npm run check
cd ..
git diff --check
```

Expected completion state:

- `/debug` is a mode composition root for selected-hour behavior, not the owner of all legacy selected-hour state machinery.
- Debug diagnostics payload normalization is tested outside the route.
- `/` and `/debug` normal f32 can honestly publish `strongVisibleGpuPath: true`.
- `/debug` parity remains `legacy-debug` and keeps August-only Python `.bin` validity.
- Visible readbacks are accounted separately from range, tooltip, comparison, and debug readback reasons.
- Scene/WebGPU lifecycle remains in scene-facing modules.
- `npm run check` status is honestly reported and not confused with this plan's runtime quality result.

---

## Next Steps Tracker

Use this section to keep orientation across projects. It is not part of this plan's execution scope unless the user explicitly approves a follow-up.

1. **Current plan: debug selected-hour host + visible readback proof**
   - Status: planned, not implemented in this document.
   - Purpose: make `/debug` easier to maintain and make `strongVisibleGpuPath` an honest proven claim instead of a conservative non-claim.
   - Stop condition: final verification passes, review agents find no blocking issues, and `/debug` parity remains `legacy-debug`.

2. **Next likely plan: static debt cleanup**
   - Purpose: make `npm run check` useful again by fixing or quarantining inherited TypeScript/Svelte errors without changing runtime behavior.
   - Why next: the project is in cleanup/organization mode, and a meaningful static gate makes later refactors safer.
   - Starting point: compare the latest `npm run check` count against the recorded baseline and fix touched areas in small batches.

3. **Route thinning and shared composition cleanup**
   - Purpose: make both `viewer/src/routes/+page.svelte` and `viewer/src/routes/debug/+page.svelte` thinner composition roots after selected-hour ownership is cleaner.
   - Main route target: keep `/` focused on route/query/store wiring, project/model selection, selected-hour host inputs, scene props, and main diagnostics publication.
   - Debug route target: keep `/debug` focused on mode selection, debug controls, parity/collect flags, and debug diagnostics publication.
   - Constraint: keep Python `.bin`, collect, strict-exposure, and parity validity behavior intact.

4. **Compute folder organization and selective Svelte modernization**
   - Purpose: make `viewer/src/lib/compute` easier to navigate and use Svelte 5 runes only where they clarify local state/effects.
   - Constraint: import-only moves first, behavior changes separately, no broad mechanical runes migration.

5. **Last: performance / 0.5m work**
   - Purpose: optimize normalized-mode range work, tooltip/fallback CPU data, cold start, and 0.5m behavior after the codebase is cleaner and easier to reason about.
   - Constraint: do not start from timing hunches; start from the instrumented diagnostics this plan creates.

---

## Static Debt, Plainly

`npm run check` runs `svelte-check`, which is stricter than `npm run build` and focused route tests. It looks across the whole Svelte/TypeScript workspace for type errors, Svelte warnings, invalid component props, possibly undefined values, stale tests, and old helper scripts.

In this repo it currently fails even after the selected-hour runtime tests and browser probes pass. That is what "static debt" means here: there are known type/static-analysis problems already present across the wider viewer codebase, so `npm run check` is not yet a clean gate.

It came from accumulated repo evolution:

- data shapes changed, but old tests/scripts still reference older fields such as `utciByHour`, `solarExposure`, or `numHours` without type narrowing;
- Three.js object typing is stricter than runtime usage, so checks complain about fields like `isMesh` on generic `Object3D`;
- Svelte reports old component warnings such as unused exported props and module-level reactive warnings;
- `viewer/src/routes/debug/+page.svelte` still contains inherited type issues from its large parity/debug workbench role.

This plan records `npm run check` as a baseline and protects touched files from new static errors, but it does not pay down all static debt. A later static-debt cleanup plan should make `npm run check` meaningful again by fixing or quarantining those wider errors without changing runtime behavior.
