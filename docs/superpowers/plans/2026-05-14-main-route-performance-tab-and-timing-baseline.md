# Main Route Performance Tab And Timing Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Workflow overrides:** Do **not** create commits. Do **not** create git worktrees. Preserve unrelated dirty files. Use subagent review gates before moving between telemetry, UI, debug comparison, and timing-report tasks. Treat generated Playwright state such as `viewer/test-results/.last-run.json` as generated state; restore it before final status.

**Goal:** Replace the old validation-oriented Analytics tab with an end-user Performance tab, collect low-overhead real-route timing/memory telemetry from the main route, add a debug-only `.bin` vs WebGPU comparison table, and produce a fresh BG base / Ness Tziona main-route timing baseline for the next 0.5m optimization decision.

**Architecture:** Use passive runtime telemetry from the real route lifecycle: `performance.now()` at existing milestones and app-owned WebGPU allocation counters already tracked by the compute pipeline. The main route Performance tab must be end-user first and must not compute full-field UTCI/shading statistics, load validation data, poll memory, force GPU readbacks, or drain queues purely for display. Debug comparison remains validation-only and may use existing parity/debug diagnostics, but the main route must stay independent of `.bin`, Python reference output, and debug globals.

**Tech Stack:** SvelteKit/Svelte 5, TypeScript 5.9, Vitest, Playwright Chromium/WebGPU, Three/Threlte, WebGPU/WGSL, PowerShell on Windows.

---

## Current Evidence

- `viewer/src/lib/components/ui/AnalyticsPanel.svelte` is still an old analysis/validation component. It imports `loadValidationData`, `compareWithValidation`, `calculateAvgMeanDiffAllHours`, `getShadingIndex`, and `getUTCIForHour`, computes min/max/mean over UTCI/shading arrays, and shows "Validation vs Grasshopper".
- The main route renders this component under a sidebar section labeled `Analytics` in `viewer/src/routes/+page.svelte`.
- Main route diagnostics are published through `window.__utciRenderDiagnostics__` by `viewer/src/routes/main/liveSelectedHour.ts` and `viewer/src/lib/diagnostics/mainRouteUtciDiagnostics.ts`.
- Existing main-route E2E proof uses `viewer/tests/e2e/main-route-manual-diagnostics.spec.ts` and asserts the strong selected-hour GPU path with `utciSurfaceSource=compute-buffer-selected-hour`, `baseRenderTransport=compute-buffer-selected-hour`, same-device, zero visible selected-hour readback, and zero `dataTextureBuildCount`.
- Existing debug route diagnostics live on `window.__onDemandPrototypeDiagnostics__` and include `timings`, `trackedGpuAllocationBytes`, `pythonBinComparisonActive`, `binComparisonEnabled`, `binComparisonValid`, and `pythonBinSampleComparison` when parity comparison is active.
- `docs/webgpu_strategy_analysis.md` says the next strategic question is cold-start/render-path performance and 0.5m feasibility, not basic compute-buffer bridge feasibility.

## Hard Constraints

- Do not create commits.
- Do not create git worktrees.
- Do not update the older May 12 plan file as a tracking task; the user explicitly said plan checkbox state does not matter.
- Do not run timing collection across every Ben-Gurion variant. Collect only:
  - `Ben-Gurion/20250815_grid_2m_fullday`
  - `Ness-Tziona/exploded/nes_tziona_unblock_2`
- Main route timing must come from `/`, not `/debug`.
- Main route Performance tab must be end-user first. Do not show internal labels such as "GPU active", `compute-buffer-selected-hour`, `sameDevice`, or `strongVisibleGpuPath` in the visible panel.
- Main route Performance tab must not load `.bin`, Python validation data, or Grasshopper validation data.
- Main route Performance tab must not scan large UTCI/shading arrays to compute min/max/mean.
- Do not use `performance.memory` or browser process memory APIs for the app-facing memory number.
- Do not force GPU readback, extra `queue.onSubmittedWorkDone()`, or any new `mapAsync()` solely for the Performance tab.
- Memory shown to users must be scoped honestly as app-owned/UTCI-owned WebGPU buffers, not total GPU memory.
- Preserve `runAll()`, `.bin`, Python comparison/reference paths, `readUtciBulk()`, `readUtcisSlice()`, `dataTexture`, debug parity, collect, and legacy selected-hour paths.
- Preserve main-route proof semantics: zero visible selected-hour readback, same-device proof, `dataTexture` fallback, and selected-hour runtime contract behavior.
- If adding telemetry changes timings materially, stop and report before optimizing.

## File Structure Target

### Create

- `viewer/src/lib/stores/performanceStore.ts`
  - Owns the latest route performance snapshot shown by the Performance tab.
  - Stores only small scalar values and labels. No arrays, GPU buffers, or large objects.
- `viewer/src/lib/components/ui/PerformancePanel.svelte`
  - Replaces `AnalyticsPanel.svelte` for the main route.
  - Displays end-user runtime facts from `performanceStore`.
- `viewer/src/lib/components/ui/DebugPerformancePanel.svelte`
  - Debug-route-only compact comparison table for `.bin` vs WebGPU.
  - Reads a lightweight prop object derived from `__onDemandPrototypeDiagnostics__` state already maintained by the debug route.
- `viewer/src/lib/performance/mainRoutePerformanceTelemetry.ts`
  - Pure helpers for building and formatting performance snapshots from diagnostics and metadata.
  - Keeps Svelte components small and testable.
- `viewer/src/lib/performance/debugPerformanceComparison.ts`
  - Pure helpers for building the `.bin` vs WebGPU table model from debug diagnostics.
- `viewer/tests/performance/mainRoutePerformanceTelemetry.test.ts`
  - Unit tests for main-route snapshot building and formatting.
- `viewer/tests/performance/debugPerformanceComparison.test.ts`
  - Unit tests for debug comparison model.
- `viewer/tests/e2e/main-route-performance-baseline.spec.ts`
  - Playwright collection/proof for BG base and Ness Tziona on `/`.
- `viewer/scripts/summarize-main-route-performance.ts`
  - Reads Playwright-emitted JSON artifacts and writes a reviewable markdown summary.
- `docs/performance/main-route-selected-hour-current-head.md`
  - Fresh current-HEAD timing/memory analysis artifact for BG base and Ness Tziona.

### Delete

- `viewer/src/lib/components/ui/AnalyticsPanel.svelte`
  - Delete after `PerformancePanel.svelte` is wired and no imports remain.

### Modify

- `viewer/src/routes/+page.svelte`
  - Rename the sidebar section label from `Analytics` to `Performance`.
  - Import and render `PerformancePanel`.
  - Publish passive main-route performance snapshots into `performanceStore`.
  - Keep `window.__utciRenderDiagnostics__` publication for tests and diagnostics.
- `viewer/src/routes/debug/+page.svelte`
  - Add a Performance sidebar section using `DebugPerformancePanel`.
  - Keep debug comparison explicitly debug-only and `.bin`/August-validity scoped.
- `viewer/src/routes/main/liveSelectedHour.ts`
  - Extend `MainRouteLiveSelectedHourDiagnosticsParams` only if needed to pass timing/allocation fields into the diagnostics builder.
- `viewer/src/lib/diagnostics/mainRouteUtciDiagnostics.ts`
  - Add optional `timings` and `trackedGpuAllocationBytes` fields to the main-route diagnostics payload if not already available from the route state.
- `viewer/src/lib/compute/selected-hour/liveSelectedHourRouteHost.ts`
  - Expose existing selected-hour timing and allocation data through route state only if this is the narrowest way to get it into main-route diagnostics.
- `viewer/src/lib/compute/selected-hour/liveUtciSelectedHourSession.ts`
  - Prefer passing through existing `OnDemandRuntimeDiagnostics.timings` and `trackedGpuAllocationBytes`; do not add extra work to collect them.
- `viewer/src/lib/compute/selected-hour/liveSelectedHourController.ts`
  - If the controller is the correct owner for first-visible timing, expose already-known `visibleAtMs` / request timing as scalar diagnostics.
- `viewer/src/lib/components/viewer/ViewerShell.svelte`
  - Rename CSS class hooks from `analytics-section` to `performance-section` only if the slot is renamed. Preserve visual density.
- `viewer/tests/e2e/main-route-manual-diagnostics.spec.ts`
  - Add assertions that main-route diagnostics include cheap timing/memory fields and do not include `.bin` / Python comparison data.
- `viewer/tests/routes/main-route-debug-boundary-source-lock.test.ts`
  - Extend source-lock coverage so the main route Performance tab cannot import validation/parity/debug `.bin` helpers.
- `docs/webgpu_strategy_analysis.md`
  - Update the current-state timing section with the new main-route BG base / Ness Tziona baseline and next optimization inference.

## Performance Snapshot Shape

Use this data shape in `viewer/src/lib/stores/performanceStore.ts`:

```ts
import { writable } from 'svelte/store';

export type PerformanceStatus = 'idle' | 'loading' | 'ready' | 'fallback' | 'error';

export interface UserFacingPerformanceSnapshot {
	status: PerformanceStatus;
	analysisId: string | null;
	projectLabel: string | null;
	pointCount: number | null;
	gridSizeMeters: number | null;
	selectedMonthIndex: number | null;
	selectedHourIndex: number | null;
	totalToVisibleMs: number | null;
	utciComputeMs: number | null;
	ownedGpuMemoryBytes: number | null;
	memoryScope: 'utci-owned-webgpu-buffers' | null;
	measuredAt: number | null;
	error: string | null;
}

export const EMPTY_PERFORMANCE_SNAPSHOT: UserFacingPerformanceSnapshot = {
	status: 'idle',
	analysisId: null,
	projectLabel: null,
	pointCount: null,
	gridSizeMeters: null,
	selectedMonthIndex: null,
	selectedHourIndex: null,
	totalToVisibleMs: null,
	utciComputeMs: null,
	ownedGpuMemoryBytes: null,
	memoryScope: null,
	measuredAt: null,
	error: null
};

export const performanceStore = writable<UserFacingPerformanceSnapshot>(
	EMPTY_PERFORMANCE_SNAPSHOT
);
```

Interpretation rules:

- `totalToVisibleMs`: first route init / analysis selection start to first visible selected-hour surface for the current selection.
- `utciComputeMs`: selected-hour WebGPU dispatch time (`oneHourDispatchMs`) when available; fallback to first selected-hour ready timing only if the exact dispatch split is unavailable and label it only in code as fallback.
- `renderSceneSyncTotalMs`: remains available in route diagnostics/artifacts, but is not part of `UserFacingPerformanceSnapshot` because the main Performance tab no longer shows a render-prep row.
- `ownedGpuMemoryBytes`: current tracked app-owned GPU memory from `trackedGpuAllocationBytes`: `persistentExposureBytes + allHoursOutputBytes + selectedHourOutputBytes + renderOwnedSelectedHourBytes` when the render-owned scalar is available. This is not total browser/OS/device VRAM.
- `memoryScope`: always `utci-owned-webgpu-buffers` for the first version.

## Task 0: Baseline And Scope Lock

**Files:**
- Inspect only.

- [ ] **Step 1: Confirm clean branch and latest gate state**

Run:

```powershell
git status --short
git log --oneline -5
cd viewer
npm run check
```

Expected:

- `git status --short` has no output before edits.
- Latest history includes `43571c3 fix(viewer): svelte-check debt + SelectedHourOutputHandle under compute/gpu`.
- `npm run check` passes with `svelte-check found 0 errors and 0 warnings`.

- [ ] **Step 2: Confirm old Analytics dependencies before deleting them**

Run:

```powershell
rg -n "AnalyticsPanel|Validation vs Grasshopper|loadValidationData|compareWithValidation|calculateAvgMeanDiffAllHours|getShadingIndex|getUTCIForHour" viewer/src viewer/tests
```

Expected before implementation:

- Matches in `viewer/src/lib/components/ui/AnalyticsPanel.svelte`.
- `viewer/src/routes/+page.svelte` imports `AnalyticsPanel`.

- [ ] **Step 3: Scope-lock the execution**

Write this exact status note before editing:

```text
This pass replaces the sidebar Analytics artifact with a low-overhead Performance tab, adds debug-only .bin vs WebGPU comparison display, and collects fresh main-route timing for BG base and Ness Tziona only. It does not optimize 0.5m yet, does not run all BG variants, and does not make the main route depend on .bin/Python/debug parity paths.
```

## Task 1: Add Pure Performance Telemetry Models

**Files:**
- Create: `viewer/src/lib/stores/performanceStore.ts`
- Create: `viewer/src/lib/performance/mainRoutePerformanceTelemetry.ts`
- Create: `viewer/src/lib/performance/debugPerformanceComparison.ts`
- Test: `viewer/tests/performance/mainRoutePerformanceTelemetry.test.ts`
- Test: `viewer/tests/performance/debugPerformanceComparison.test.ts`

- [ ] **Step 1: Write main-route performance helper tests**

Create `viewer/tests/performance/mainRoutePerformanceTelemetry.test.ts` with tests that prove:

```ts
import { describe, expect, it } from 'vitest';
import {
	buildMainRoutePerformanceSnapshot,
	formatDuration,
	formatMemory
} from '$lib/performance/mainRoutePerformanceTelemetry';

describe('mainRoutePerformanceTelemetry', () => {
	it('builds an end-user snapshot from cheap route diagnostics', () => {
		const snapshot = buildMainRoutePerformanceSnapshot({
			analysisId: 'Ben-Gurion/20250815_grid_2m_fullday',
			projectLabel: 'Ben-Gurion',
			pointCount: 32123,
			gridSizeMeters: 2,
			selectedMonthIndex: 7,
			selectedHourIndex: 0,
			diagnostics: {
				baseLiveReady: true,
				timings: {
					firstSelectedHourVisibleMs: 4200,
					oneHourDispatchMs: 86.5,
					renderSceneSyncTotalMs: 154.7
				},
				trackedGpuAllocationBytes: {
					persistentExposureBytes: 4_194_304,
					allHoursOutputBytes: 1_048_576,
					selectedHourOutputBytes: 131_072,
					selectedHourOutputBytesHighWatermark: 262_144,
					renderOwnedSelectedHourBytes: 524_288,
					renderOwnedSelectedHourBytesHighWatermark: 524_288,
					trackingScope: 'utci-owned-webgpu-buffers'
				}
			},
			now: 10000
		});

		expect(snapshot).toMatchObject({
			status: 'ready',
			analysisId: 'Ben-Gurion/20250815_grid_2m_fullday',
			projectLabel: 'Ben-Gurion',
			pointCount: 32123,
			gridSizeMeters: 2,
			selectedMonthIndex: 7,
			selectedHourIndex: 0,
			totalToVisibleMs: 4200,
			utciComputeMs: 86.5,
			ownedGpuMemoryBytes: 5_898_240,
			memoryScope: 'utci-owned-webgpu-buffers',
			measuredAt: 10000,
			error: null
		});
	});

	it('marks fallback when the visible result is not the live ready path', () => {
		const snapshot = buildMainRoutePerformanceSnapshot({
			analysisId: 'Ben-Gurion/20250815_grid_2m_fullday',
			projectLabel: 'Ben-Gurion',
			pointCount: 32123,
			gridSizeMeters: 2,
			selectedMonthIndex: 7,
			selectedHourIndex: 0,
			diagnostics: {
				baseLiveReady: false,
				timings: {},
				trackedGpuAllocationBytes: undefined
			},
			now: 10000
		});

		expect(snapshot.status).toBe('fallback');
		expect(snapshot.totalToVisibleMs).toBeNull();
		expect(snapshot.ownedGpuMemoryBytes).toBeNull();
	});

	it('formats user-facing durations and memory without jargon', () => {
		expect(formatDuration(86.49)).toBe('86 ms');
		expect(formatDuration(4200)).toBe('4.2 s');
		expect(formatDuration(null)).toBe('Measuring');
		expect(formatMemory(4_456_448)).toBe('4.3 MB');
		expect(formatMemory(null)).toBe('Measuring');
	});
});
```

- [ ] **Step 2: Run main-route helper tests to verify they fail**

Run:

```powershell
cd viewer
npx vitest run tests/performance/mainRoutePerformanceTelemetry.test.ts --reporter=dot
```

Expected:

- FAIL because `mainRoutePerformanceTelemetry.ts` does not exist.

- [ ] **Step 3: Implement `performanceStore.ts`**

Create `viewer/src/lib/stores/performanceStore.ts` using the exact `UserFacingPerformanceSnapshot` shape in the "Performance Snapshot Shape" section.

- [ ] **Step 4: Implement `mainRoutePerformanceTelemetry.ts`**

Create `viewer/src/lib/performance/mainRoutePerformanceTelemetry.ts`:

```ts
import {
	EMPTY_PERFORMANCE_SNAPSHOT,
	type UserFacingPerformanceSnapshot
} from '$lib/stores/performanceStore';
import type { OnDemandTimings, TrackedGpuAllocationBytes } from '$lib/compute/on-demand/onDemandDiagnostics';

export interface MainRoutePerformanceDiagnosticsLike {
	baseLiveReady?: boolean;
	timings?: OnDemandTimings;
	trackedGpuAllocationBytes?: TrackedGpuAllocationBytes;
	error?: string;
}

export interface BuildMainRoutePerformanceSnapshotParams {
	analysisId: string | null;
	projectLabel: string | null;
	pointCount: number | null;
	gridSizeMeters: number | null;
	selectedMonthIndex: number | null;
	selectedHourIndex: number | null;
	diagnostics: MainRoutePerformanceDiagnosticsLike | null | undefined;
	now: number;
}

export function getOwnedGpuMemoryBytes(
	tracked: TrackedGpuAllocationBytes | undefined
): number | null {
	if (!tracked) return null;
	return (
		tracked.persistentExposureBytes +
		tracked.allHoursOutputBytes +
		tracked.selectedHourOutputBytes +
		(tracked.renderOwnedSelectedHourBytes ?? 0)
	);
}

export function buildMainRoutePerformanceSnapshot(
	params: BuildMainRoutePerformanceSnapshotParams
): UserFacingPerformanceSnapshot {
	const diagnostics = params.diagnostics;
	if (!diagnostics) {
		return {
			...EMPTY_PERFORMANCE_SNAPSHOT,
			status: 'loading',
			analysisId: params.analysisId,
			projectLabel: params.projectLabel,
			pointCount: params.pointCount,
			gridSizeMeters: params.gridSizeMeters,
			selectedMonthIndex: params.selectedMonthIndex,
			selectedHourIndex: params.selectedHourIndex,
			measuredAt: params.now
		};
	}

	const status = diagnostics.error
		? 'error'
		: diagnostics.baseLiveReady
			? 'ready'
			: 'fallback';

	return {
		status,
		analysisId: params.analysisId,
		projectLabel: params.projectLabel,
		pointCount: params.pointCount,
		gridSizeMeters: params.gridSizeMeters,
		selectedMonthIndex: params.selectedMonthIndex,
		selectedHourIndex: params.selectedHourIndex,
		totalToVisibleMs: diagnostics.timings?.firstSelectedHourVisibleMs ?? null,
		utciComputeMs: diagnostics.timings?.oneHourDispatchMs ?? null,
		ownedGpuMemoryBytes: getOwnedGpuMemoryBytes(diagnostics.trackedGpuAllocationBytes),
		memoryScope: diagnostics.trackedGpuAllocationBytes?.trackingScope ?? null,
		measuredAt: params.now,
		error: diagnostics.error ?? null
	};
}

export function formatDuration(valueMs: number | null): string {
	if (valueMs === null || !Number.isFinite(valueMs)) return 'Measuring';
	if (valueMs < 1000) return `${Math.round(valueMs)} ms`;
	return `${(valueMs / 1000).toFixed(1)} s`;
}

export function formatMemory(valueBytes: number | null): string {
	if (valueBytes === null || !Number.isFinite(valueBytes)) return 'Measuring';
	const mib = valueBytes / (1024 * 1024);
	return `${mib.toFixed(1)} MB`;
}
```

- [ ] **Step 5: Write debug comparison helper tests**

Create `viewer/tests/performance/debugPerformanceComparison.test.ts` with tests that prove `.bin` comparison is debug-only and validity-scoped:

```ts
import { describe, expect, it } from 'vitest';
import { buildDebugPerformanceComparisonRows } from '$lib/performance/debugPerformanceComparison';

describe('debugPerformanceComparison', () => {
	it('builds a valid .bin vs WebGPU comparison table from debug diagnostics', () => {
		const rows = buildDebugPerformanceComparisonRows({
			binComparisonEnabled: true,
			binComparisonValid: true,
			pythonBaselineStatus: 'valid',
			pythonBinSampleComparison: {
				sampleCount: 4,
				maxAbsDiff: 0.08,
				meanAbsDiff: 0.03
			},
			timings: {
				firstSelectedHourVisibleMs: 4200,
				oneHourDispatchMs: 86.5,
				debugReadbackMs: 12.25
			}
		});

		expect(rows).toEqual([
			{ metric: 'Mean UTCI', python: '28.12 C', webgpu: '28.15 C', diff: '+0.03 C' },
			{ metric: 'Visible time', python: '8.1 s', webgpu: '4.2 s', diff: '-3.9 s' }
		]);
	});

	it('reports unavailable comparison when the .bin baseline is invalid', () => {
		const rows = buildDebugPerformanceComparisonRows({
			binComparisonEnabled: true,
			binComparisonValid: false,
			pythonBaselineStatus: 'invalid-month',
			timings: {}
		});

		expect(rows[0]).toEqual({
			metric: 'Mean UTCI',
			python: 'Unavailable for this selection',
			webgpu: '-',
			diff: '-'
		});
	});
});
```

- [ ] **Step 6: Implement `debugPerformanceComparison.ts`**

Create `viewer/src/lib/performance/debugPerformanceComparison.ts`:

```ts
import type { OnDemandTimings } from '$lib/compute/on-demand/onDemandDiagnostics';
import { formatDuration } from '$lib/performance/mainRoutePerformanceTelemetry';

export interface PythonBinSampleComparisonLike {
	sampleCount?: number;
	maxAbsDiff?: number;
	meanAbsDiff?: number;
}

export interface DebugPerformanceComparisonDiagnosticsLike {
	binComparisonEnabled?: boolean;
	binComparisonValid?: boolean;
	pythonBaselineStatus?: string;
	pythonBinSampleComparison?: PythonBinSampleComparisonLike;
	timings?: OnDemandTimings;
}

export interface DebugPerformanceComparisonRow {
	metric: string;
	bin: string;
	webgpu: string;
}

function formatDifference(value: number | undefined): string {
	if (typeof value !== 'number' || !Number.isFinite(value)) return '-';
	return `${value.toFixed(2)} C`;
}

export function buildDebugPerformanceComparisonRows(
	diagnostics: DebugPerformanceComparisonDiagnosticsLike | null | undefined
): DebugPerformanceComparisonRow[] {
	if (!diagnostics?.binComparisonEnabled || !diagnostics.binComparisonValid) {
		return [
			{
				metric: 'Mean UTCI',
				python: 'Unavailable for this selection',
				webgpu: '-',
				diff: '-'
			},
			{
				metric: 'Visible time',
				python: '-',
				webgpu: formatDuration(diagnostics?.timings?.firstSelectedHourVisibleMs ?? null),
				diff: '-'
			}
		];
	}

	const comparison = diagnostics.pythonBinSampleComparison;
	return [
		{
			metric: 'Mean UTCI',
			python: formatUtci(diagnostics.pythonSelectedHourMeanUtci),
			webgpu: formatUtci(diagnostics.webgpuSelectedHourMeanUtci),
			diff: formatSignedUtciDiff(meanDiff)
		},
		{
			metric: 'Visible time',
			python: formatDuration(diagnostics.pythonDerivedOneHourMs ?? null),
			webgpu: formatDuration(diagnostics.timings?.firstSelectedHourVisibleMs ?? null),
			diff: formatSignedDurationDiff(visibleDiff)
		}
	];
}
```

- [ ] **Step 7: Run helper tests**

Run:

```powershell
cd viewer
npx vitest run tests/performance/mainRoutePerformanceTelemetry.test.ts tests/performance/debugPerformanceComparison.test.ts --reporter=dot
```

Expected:

- PASS.

## Task 2: Expose Cheap Main-Route Runtime Telemetry

**Files:**
- Modify: `viewer/src/lib/diagnostics/mainRouteUtciDiagnostics.ts`
- Modify: `viewer/src/routes/main/liveSelectedHour.ts`
- Modify: `viewer/src/routes/+page.svelte`
- Modify if required: `viewer/src/lib/compute/selected-hour/liveSelectedHourRouteHost.ts`
- Modify if required: `viewer/src/lib/compute/selected-hour/liveUtciSelectedHourSession.ts`
- Test: `viewer/tests/e2e/main-route-manual-diagnostics.spec.ts`
- Test: `viewer/tests/routes/main-route-debug-boundary-source-lock.test.ts`

- [ ] **Step 1: Add source-lock expectations for main-route Performance isolation**

In `viewer/tests/routes/main-route-debug-boundary-source-lock.test.ts`, add assertions that `viewer/src/routes/+page.svelte` and `viewer/src/lib/components/ui/PerformancePanel.svelte` do not contain:

```ts
[
	'loadValidationData',
	'compareWithValidation',
	'calculateAvgMeanDiffAllHours',
	'loadReferenceFromFs',
	'pythonBin',
	'__onDemandPrototypeDiagnostics__',
	'performance.memory'
]
```

Expected after implementation:

- Main route stays free of validation/debug `.bin` dependencies.

- [ ] **Step 2: Extend diagnostics types**

In `viewer/src/lib/diagnostics/mainRouteUtciDiagnostics.ts`, add optional fields:

```ts
import type {
	OnDemandTimings,
	TrackedGpuAllocationBytes
} from '$lib/compute/on-demand/onDemandDiagnostics';
```

Add to `MainRouteUtciDiagnosticsPayload` and `MainRouteUtciDiagnosticsInputs`:

```ts
timings?: OnDemandTimings;
trackedGpuAllocationBytes?: TrackedGpuAllocationBytes;
```

Return them from `buildMainRouteUtciDiagnostics`:

```ts
timings: inputs.timings,
trackedGpuAllocationBytes: inputs.trackedGpuAllocationBytes,
```

- [ ] **Step 3: Pass telemetry through `liveSelectedHour.ts`**

In `viewer/src/routes/main/liveSelectedHour.ts`, import the same types and extend `MainRouteLiveSelectedHourDiagnosticsParams`:

```ts
timings?: OnDemandTimings;
trackedGpuAllocationBytes?: TrackedGpuAllocationBytes;
```

Pass them through `buildMainRouteLiveSelectedHourDiagnosticsInputs`:

```ts
timings: params.timings,
trackedGpuAllocationBytes: params.trackedGpuAllocationBytes,
```

- [ ] **Step 4: Source the timings and allocation bytes from live route state**

Inspect `liveRouteState.base` and the selected-hour session result types. Use already-collected fields only. The intended source is:

```ts
const basePerformanceDiagnostics = liveRouteState.base.runtimeDiagnostics;
```

If the exact field does not exist, add the narrowest scalar fields to the route host state:

```ts
runtimeDiagnostics?: Pick<
	OnDemandRuntimeDiagnostics,
	'timings' | 'trackedGpuAllocationBytes'
>;
```

Populate it from the existing selected-hour session diagnostics that already contain `timings` and `trackedGpuAllocationBytes`.

Stop and report before adding any new GPU readback, forced queue drain, polling loop, or array scan.

- [ ] **Step 5: Publish performance snapshots to the store from the main route**

In `viewer/src/routes/+page.svelte`, import:

```ts
import { performanceStore, EMPTY_PERFORMANCE_SNAPSHOT } from '$lib/stores/performanceStore';
import { buildMainRoutePerformanceSnapshot } from '$lib/performance/mainRoutePerformanceTelemetry';
```

In the existing reactive diagnostics publication block, after `updateUtciRenderDiagnostics(...)`, set the store:

```ts
$: if (typeof window !== 'undefined' && mounted) {
	performanceStore.set(
		buildMainRoutePerformanceSnapshot({
			analysisId,
			projectLabel: currentProjectId,
			pointCount: $analysisStore?.metadata.num_positions ?? null,
			gridSizeMeters: $analysisStore?.metadata.grid_size ?? null,
			selectedMonthIndex,
			selectedHourIndex,
			diagnostics: {
				baseLiveReady,
				timings: liveRouteState.base.runtimeDiagnostics?.timings,
				trackedGpuAllocationBytes:
					liveRouteState.base.runtimeDiagnostics?.trackedGpuAllocationBytes,
				error: liveRouteState.base.error
			},
			now: performance.now()
		})
	);
}
```

On destroy, clear it:

```ts
performanceStore.set(EMPTY_PERFORMANCE_SNAPSHOT);
```

Use the actual state field names from Step 4. Do not add a frequent timer.

- [ ] **Step 6: Extend E2E diagnostics proof**

In `viewer/tests/e2e/main-route-manual-diagnostics.spec.ts`, extend the first test after the existing `value` assertions:

```ts
expect(value.timings).toEqual(expect.objectContaining({
	oneHourDispatchMs: expect.any(Number)
}));
expect(value.timings.firstSelectedHourVisibleMs).toEqual(expect.any(Number));
expect(value.trackedGpuAllocationBytes).toEqual(expect.objectContaining({
	trackingScope: 'utci-owned-webgpu-buffers',
	persistentExposureBytes: expect.any(Number),
	selectedHourOutputBytesHighWatermark: expect.any(Number)
}));
expect(JSON.stringify(value)).not.toMatch(/pythonBin|binComparison|__onDemandPrototypeDiagnostics__|performance\.memory/i);
```

If `firstSelectedHourVisibleMs` is legitimately not available for an already-warm selected-hour change, assert it in the initial publication only and keep subsequent scrub assertions focused on `oneHourDispatchMs` / render sync splits.

- [ ] **Step 7: Run targeted diagnostics tests**

Run:

```powershell
cd viewer
npx vitest run tests/routes/main-route-debug-boundary-source-lock.test.ts tests/compute/live-selected-hour-route-host.test.ts tests/compute/live-selected-hour-session.test.ts --reporter=dot
npx playwright test tests/e2e/main-route-manual-diagnostics.spec.ts --project=chromium --grep "publishes selected-hour diagnostics" --reporter=list --timeout=30000
```

Expected:

- Vitest PASS.
- Focused Playwright PASS.
- Main-route diagnostics include timing/allocation fields without debug `.bin` fields.

- [ ] **Step 8: Subagent review gate for telemetry**

Dispatch one review subagent:

```text
Review the main-route performance telemetry changes in D:\Projects\Nur\Shade\fast-utci. Do not edit files. Verify that telemetry is passive, main-route sourced, has no .bin/Python/debug dependency, does not use performance.memory, does not force GPU readback/queue drain for display, and preserves selected-hour proof semantics. Report blockers first.
```

Expected:

- No blockers before moving to UI work.

## Task 3: Replace Analytics With End-User Performance Panel

**Files:**
- Create: `viewer/src/lib/components/ui/PerformancePanel.svelte`
- Delete: `viewer/src/lib/components/ui/AnalyticsPanel.svelte`
- Modify: `viewer/src/routes/+page.svelte`
- Modify if needed: `viewer/src/lib/components/viewer/ViewerShell.svelte`
- Test: `viewer/tests/e2e/main-route-manual-diagnostics.spec.ts`

- [ ] **Step 1: Create `PerformancePanel.svelte`**

Create a compact panel that imports only:

```ts
import { performanceStore } from '$lib/stores/performanceStore';
import { formatDuration, formatMemory } from '$lib/performance/mainRoutePerformanceTelemetry';
```

Render these rows:

```svelte
<script lang="ts">
	import { performanceStore } from '$lib/stores/performanceStore';
	import {
		formatDuration,
		formatMemory
	} from '$lib/performance/mainRoutePerformanceTelemetry';

	$: snapshot = $performanceStore;
	$: statusLabel =
		snapshot.status === 'ready'
			? 'Ready'
			: snapshot.status === 'loading'
				? 'Preparing'
				: snapshot.status === 'fallback'
					? 'Preparing live result'
					: snapshot.status === 'error'
						? 'Needs attention'
						: 'Waiting';
</script>

<div class="performance-panel" data-testid="performance-panel">
	<div class="metric-row metric-row-primary">
		<span>Total calculation time</span>
		<strong>{formatDuration(snapshot.totalToVisibleMs)}</strong>
	</div>
	<div class="metric-row">
		<span>UTCI calculation</span>
		<strong>{formatDuration(snapshot.utciComputeMs)}</strong>
	</div>
	<div class="metric-row">
		<span>GPU VRAM</span>
		<strong>{formatMemory(snapshot.ownedGpuMemoryBytes)}</strong>
	</div>
	<div class="metric-row">
		<span>Grid size</span>
		<strong>
			{snapshot.gridSizeMeters === null
				? 'Loading'
				: `${snapshot.gridSizeMeters} m${
						snapshot.pointCount === null ? '' : ` (${snapshot.pointCount.toLocaleString()} pts)`
					}`}
		</strong>
	</div>
	<div class="performance-status">{statusLabel}</div>
</div>
```

Style it with the existing sidebar variables. Keep the panel dense and do not add explanatory paragraphs.

- [ ] **Step 2: Wire the main route to Performance**

In `viewer/src/routes/+page.svelte`:

- Replace `AnalyticsPanel` import with `PerformancePanel`.
- Rename `analyticsOpen` to `performanceOpen`.
- In the sidebar slot, change the label from `Analytics` to `Performance`.
- Render `<PerformancePanel />`.

Use this visible structure:

```svelte
<svelte:fragment slot="analytics">
	<button
		type="button"
		class="section-header section-header-toggle"
		on:click={() => (performanceOpen = !performanceOpen)}
	>
		<span>Performance</span>
		<span class:open={performanceOpen} class="chevron">v</span>
	</button>
	{#if performanceOpen}
		<PerformancePanel />
	{/if}
</svelte:fragment>
```

Keep the slot name `analytics` for this task unless renaming the slot is trivial and covered by tests. The visible product label must be `Performance`.

- [ ] **Step 3: Delete the old Analytics panel**

Delete `viewer/src/lib/components/ui/AnalyticsPanel.svelte`.

Run:

```powershell
rg -n "AnalyticsPanel|Validation vs Grasshopper|loadValidationData|compareWithValidation|calculateAvgMeanDiffAllHours|getShadingIndex|getUTCIForHour" viewer/src viewer/tests
```

Expected:

- No `AnalyticsPanel` matches.
- No main-route component imports validation/Grasshopper helpers.
- `validationService.ts` may still exist if used by parity/debug code; do not delete it unless `rg` proves it is unused everywhere.

- [ ] **Step 4: Add UI E2E assertion**

In `viewer/tests/e2e/main-route-manual-diagnostics.spec.ts`, add a focused assertion in the first test:

```ts
await page.getByRole('button', { name: /performance/i }).click();
await expect(page.getByTestId('performance-panel')).toBeVisible();
await expect(page.getByText(/Total calculation time/i)).toBeVisible();
await expect(page.getByText(/UTCI calculation/i)).toBeVisible();
await expect(page.getByText(/Render prep/i)).toHaveCount(0);
await expect(page.getByText(/GPU VRAM/i)).toBeVisible();
await expect(page.getByText(/Grid size/i)).toBeVisible();
await expect(page.getByText(/Validation vs Grasshopper/i)).toHaveCount(0);
```

- [ ] **Step 5: Run UI checks**

Run:

```powershell
cd viewer
npm run check
npx playwright test tests/e2e/main-route-manual-diagnostics.spec.ts --project=chromium --grep "publishes selected-hour diagnostics" --reporter=list --timeout=30000
```

Expected:

- `npm run check` PASS.
- Focused Playwright PASS.

- [ ] **Step 6: Subagent review gate for main Performance UI**

Dispatch one review subagent:

```text
Review the main-route Performance tab replacement in D:\Projects\Nur\Shade\fast-utci. Do not edit files. Verify the old Analytics/Grasshopper/validation UI code is removed from the main route, the visible panel is end-user-first, it does not scan large arrays or load .bin/Python data, and the wording does not expose internal GPU/debug jargon. Report blockers first.
```

Expected:

- No blockers before debug comparison work.

## Task 4: Add Debug-Only `.bin` vs WebGPU Performance Comparison

**Files:**
- Create: `viewer/src/lib/components/ui/DebugPerformancePanel.svelte`
- Modify: `viewer/src/routes/debug/+page.svelte`
- Test: `viewer/tests/e2e/debug-route-parity-runtime-diagnostics.spec.ts`
- Test: `viewer/tests/debug/debug-on-demand-prototype-diagnostics.test.ts` if diagnostics shape changes

- [ ] **Step 1: Create `DebugPerformancePanel.svelte`**

Create a component that accepts:

```ts
import { buildDebugPerformanceComparisonRows } from '$lib/performance/debugPerformanceComparison';

export let diagnostics: unknown = null;
```

Inside the component:

```svelte
<script lang="ts">
	import {
		buildDebugPerformanceComparisonRows,
		type DebugPerformanceComparisonDiagnosticsLike
	} from '$lib/performance/debugPerformanceComparison';

	export let diagnostics: DebugPerformanceComparisonDiagnosticsLike | null = null;
	$: rows = buildDebugPerformanceComparisonRows(diagnostics);
</script>

<div class="debug-performance-panel" data-testid="debug-performance-panel">
	<table>
		<thead>
			<tr>
				<th>Metric</th>
				<th>Python</th>
				<th>WebGPU</th>
				<th>Diff</th>
			</tr>
		</thead>
		<tbody>
			{#each rows as row}
				<tr>
					<td>{row.metric}</td>
					<td>{row.python}</td>
					<td>{row.webgpu}</td>
					<td>{row.diff}</td>
				</tr>
			{/each}
		</tbody>
	</table>
	<div class="comparison-note">Debug comparison only</div>
</div>
```

Keep styling compact and consistent with the sidebar. Do not add long explanatory copy.

- [ ] **Step 2: Wire debug route sidebar Performance section**

In `viewer/src/routes/debug/+page.svelte`:

- Import `DebugPerformancePanel`.
- Add a local reactive value:

```ts
$: debugPerformanceDiagnostics =
	typeof window === 'undefined'
		? null
		: getParityWindow().__onDemandPrototypeDiagnostics__ ?? null;
```

If Svelte reactivity does not update from the window global, update the existing `publishOnDemandDiagnostics` / diagnostics state path so the component receives the same `nextDiagnostics` object that is assigned to `win.__onDemandPrototypeDiagnostics__`.

Add a sidebar slot before `time` or after `scenario`:

```svelte
<svelte:fragment slot="analytics">
	<div class="section-header">Performance</div>
	<DebugPerformancePanel diagnostics={debugPerformanceDiagnostics} />
</svelte:fragment>
```

Keep the debug header label `.bin vs live compute`; this panel is explicitly debug-route validation.

- [ ] **Step 3: Add debug route UI proof**

In `viewer/tests/e2e/debug-route-parity-runtime-diagnostics.spec.ts`, add or extend an August parity test:

```ts
await page.goto('/debug?parity=1&utciOnDemand=f32&monthIndex=7');
await expect(page.getByTestId('debug-performance-panel')).toBeVisible();
await expect(page.getByRole('columnheader', { name: 'Python' })).toBeVisible();
await expect(page.getByRole('columnheader', { name: 'WebGPU' })).toBeVisible();
await expect(page.getByRole('columnheader', { name: 'Diff' })).toBeVisible();
await expect(page.getByText(/Mean UTCI/i)).toBeVisible();
await expect(page.getByText(/Visible time/i)).toBeVisible();
```

For a non-August/debug invalid baseline test, assert:

```ts
await expect(page.getByText(/Unavailable for this selection/i)).toBeVisible();
```

- [ ] **Step 4: Run debug tests**

Run:

```powershell
cd viewer
npx vitest run tests/performance/debugPerformanceComparison.test.ts tests/debug/debug-on-demand-prototype-diagnostics.test.ts --reporter=dot
npx playwright test tests/e2e/debug-route-parity-runtime-diagnostics.spec.ts --project=chromium --reporter=list --timeout=30000
```

Expected:

- PASS.
- Debug route comparison remains valid only when `.bin` comparison is valid.

- [ ] **Step 5: Subagent review gate for debug comparison**

Dispatch one review subagent:

```text
Review the debug Performance comparison in D:\Projects\Nur\Shade\fast-utci. Do not edit files. Verify .bin/Python comparison remains debug-only, August validity is explicit, the main route did not gain .bin dependencies, and debug parity/collect/legacy selected-hour paths are preserved. Report blockers first.
```

Expected:

- No blockers before timing-baseline collection.

## Task 5: Collect Fresh Main-Route BG Base And Ness Tziona Timing Baseline

**Files:**
- Create: `viewer/tests/e2e/main-route-performance-baseline.spec.ts`
- Create: `viewer/scripts/summarize-main-route-performance.ts`
- Create/Update generated artifact during execution: `data/performance-results/main-route-selected-hour-current-head.json`
- Create/Update doc: `docs/performance/main-route-selected-hour-current-head.md`
- Modify: `docs/webgpu_strategy_analysis.md`

- [ ] **Step 1: Create Playwright timing collector**

Create `viewer/tests/e2e/main-route-performance-baseline.spec.ts`.

The test should visit only:

```ts
const CASES = [
	{
		label: 'Ben-Gurion base 2m',
		analysisId: 'Ben-Gurion/20250815_grid_2m_fullday',
		path: '/?analysis=Ben-Gurion%2F20250815_grid_2m_fullday&utciRender=auto&utciRenderDiagnostics=1'
	},
	{
		label: 'Ness Tziona 2m',
		analysisId: 'Ness-Tziona/exploded/nes_tziona_unblock_2',
		path: '/?analysis=Ness-Tziona%2Fexploded%2Fnes_tziona_unblock_2&utciRender=auto&utciRenderDiagnostics=1'
	}
];
```

For each case:

- Wait for the same strong main-route selected-hour publication condition used by `main-route-manual-diagnostics.spec.ts`.
- Assert no `.bin` / Python / debug comparison fields in the main-route diagnostics.
- Extract:
  - analysis id
  - point count
  - grid size
  - selected month/hour/time index
  - `timings.firstSelectedHourVisibleMs`
  - `timings.oneHourDispatchMs`
  - `timings.exposurePrecomputeMs`
  - `timings.renderSceneSyncStartDelayMs`
  - `timings.renderSceneSyncTotalMs`
  - `timings.renderLayoutBuildMs`
  - `timings.renderSurfaceMeshMs`
  - `timings.renderStorageInitWaitMs`
  - `timings.renderBufferCopyMs`
  - `timings.renderQueueDrainMs`
  - `trackedGpuAllocationBytes`
  - proof fields: `utciSurfaceSource`, `baseRenderTransport`, `dataTextureBuildCount`, `selectedHourRuntimeContract.strongVisibleGpuPath`

Write one JSON artifact to:

```text
../data/performance-results/main-route-selected-hour-current-head.json
```

Use `test.info().attach(...)` as well if convenient, but the durable repo artifact is the JSON file.

- [ ] **Step 2: Run collector**

Run:

```powershell
cd viewer
npx playwright test tests/e2e/main-route-performance-baseline.spec.ts --project=chromium --workers=1 --reporter=list --timeout=45000
```

Expected:

- PASS for BG base and Ness Tziona.
- No BG variants are visited.
- JSON artifact exists at `data/performance-results/main-route-selected-hour-current-head.json`.

- [ ] **Step 3: Create summary script**

Create `viewer/scripts/summarize-main-route-performance.ts`.

Inputs:

- `../data/performance-results/main-route-selected-hour-current-head.json`

Output:

- `../docs/performance/main-route-selected-hour-current-head.md`

The markdown must include:

```markdown
# Main Route Selected-Hour Performance Baseline

Date: 2026-05-14

## Scope

Collected from the product main route `/`, not `/debug`.

Included analyses:
- Ben-Gurion base 2m
- Ness Tziona 2m

Excluded analyses:
- Ben-Gurion variants, because prior variant runs were close enough to base and not worth the extra collection time for this planning question.

## Proof Boundary

...

## Timing Table

...

## Memory Table

...

## Current Optimization Inference

...
```

The inference section must choose the next optimization target from measured evidence. If the fresh numbers disagree with the old May 9 doc, the fresh main-route numbers win.

- [ ] **Step 4: Add package script if useful**

If the repo already prefers scripts in `viewer/package.json`, add:

```json
"summarize:main-route-performance": "tsx scripts/summarize-main-route-performance.ts"
```

If `tsx` is not available, use the existing TypeScript script runner pattern in `viewer/package.json`. Do not add a dependency just for this summary script.

- [ ] **Step 5: Run summary script**

Run:

```powershell
cd viewer
npm run summarize:main-route-performance
```

Expected:

- Markdown summary exists at `docs/performance/main-route-selected-hour-current-head.md`.
- Summary explicitly says numbers are from `/`.
- Summary explicitly says `.bin` comparison is excluded from main-route timing.

- [ ] **Step 6: Update strategy doc**

Modify `docs/webgpu_strategy_analysis.md`:

- Add a dated section for the 2026-05-14 main-route timing baseline.
- Link to `docs/performance/main-route-selected-hour-current-head.md`.
- Replace or qualify stale debug-route timing claims where needed.
- Keep the old-vs-current contrast.
- Keep the 0.5m boundary honest: this pass does not prove 0.5m; it informs which bottleneck to attack before 0.5m.

- [ ] **Step 7: Subagent review gate for timing collection and inference**

Dispatch two review subagents in parallel:

```text
Review the main-route performance timing collection in D:\Projects\Nur\Shade\fast-utci. Do not edit files. Verify the collector uses `/`, not `/debug`, covers only BG base and Ness Tziona, excludes .bin/Python comparison from main timing, preserves proof fields, and writes a durable JSON/markdown artifact. Report blockers first.
```

```text
Review docs/performance/main-route-selected-hour-current-head.md and docs/webgpu_strategy_analysis.md in D:\Projects\Nur\Shade\fast-utci. Do not edit files. Verify the optimization inference follows the fresh main-route evidence, keeps 0.5m claims conservative, and does not overclaim total GPU memory or zero-copy behavior. Report blockers first.
```

Expected:

- No blockers.
- If reviewers disagree on the next optimization target, document the disagreement in the performance summary rather than averaging it away.

## Task 6: Full Verification And Cleanup

**Files:**
- Inspect only unless generated state needs cleanup.

- [ ] **Step 1: Run static check**

Run:

```powershell
cd viewer
npm run check
```

Expected:

- PASS with `svelte-check found 0 errors and 0 warnings`.

- [ ] **Step 2: Run focused unit tests**

Run:

```powershell
cd viewer
npx vitest run tests/performance/mainRoutePerformanceTelemetry.test.ts tests/performance/debugPerformanceComparison.test.ts tests/routes/main-route-debug-boundary-source-lock.test.ts tests/compute/onDemandDiagnostics.test.ts --reporter=dot
```

Expected:

- PASS.

- [ ] **Step 3: Run selected-hour quality suite**

Run:

```powershell
cd viewer
npm run test:quality:selected-hour
```

Expected:

- PASS, preserving selected-hour proof surfaces.

- [ ] **Step 4: Run selected-hour E2E suite**

Run:

```powershell
cd viewer
npm run test:e2e:selected-hour
```

Expected:

- PASS.

- [ ] **Step 5: Run performance-specific E2E**

Run:

```powershell
cd viewer
npx playwright test tests/e2e/main-route-performance-baseline.spec.ts tests/e2e/debug-route-parity-runtime-diagnostics.spec.ts --project=chromium --workers=1 --reporter=list --timeout=45000
```

Expected:

- PASS.
- Main route baseline still covers only BG base and Ness Tziona.

- [ ] **Step 6: Run production build**

Run:

```powershell
cd viewer
npm run build
```

Expected:

- PASS. Existing large-chunk warnings are acceptable unless this pass makes them worse.

- [ ] **Step 7: Clean generated Playwright state**

Run:

```powershell
git status --short
```

If it shows `viewer/test-results/.last-run.json`, restore it:

```powershell
git restore -- "viewer/test-results/.last-run.json"
```

Then run:

```powershell
git status --short
```

Expected:

- Only intentional source/test/doc/performance-result artifacts remain.

- [ ] **Step 8: Run diff whitespace check**

Run:

```powershell
git diff --check
```

Expected:

- PASS with no output.

- [ ] **Step 9: Final subagent review**

Dispatch one final review subagent:

```text
Final review for D:\Projects\Nur\Shade\fast-utci main-route Performance tab and timing baseline. Do not edit files. Verify no commits/worktrees, generated-state cleanup, main route remains .bin/Python/debug independent, Performance UI is end-user-first, debug comparison is debug-only, timing artifacts are from BG base and Ness Tziona on `/`, and verification evidence is complete. Report blockers first, then recommended next optimization target for 0.5m.
```

Expected:

- No blockers.
- Recommendation names one next optimization target or states why the fresh data is inconclusive.

## Completion Criteria

This plan is complete only when all are true:

- Main route sidebar shows `Performance`, not `Analytics`.
- `viewer/src/lib/components/ui/AnalyticsPanel.svelte` is deleted or fully replaced with no remaining imports.
- Main route Performance panel shows end-user runtime facts from live route telemetry:
  - visible time
  - UTCI calculation time
  - render prep time
  - scoped UTCI-owned WebGPU memory
  - grid size
- Main route Performance panel does not load validation data, `.bin`, Python reference data, or debug globals.
- Debug route has a compact Python vs WebGPU Performance comparison table with a diff column.
- Debug route clearly gates `.bin` comparison validity and does not imply `.bin` exists for every project/month.
- Fresh main-route timing baseline exists for only:
  - `Ben-Gurion/20250815_grid_2m_fullday`
  - `Ness-Tziona/exploded/nes_tziona_unblock_2`
- `docs/performance/main-route-selected-hour-current-head.md` and `docs/webgpu_strategy_analysis.md` capture the new evidence and the next optimization inference.
- `npm run check` passes.
- `npm run test:quality:selected-hour` passes.
- `npm run test:e2e:selected-hour` passes.
- Performance-specific unit and E2E tests pass.
- `npm run build` passes.
- `git diff --check` passes.
- `viewer/test-results/.last-run.json` is not left modified.

## Next Best Steps After This Plan

Use the fresh main-route evidence to pick exactly one 0.5m-enabling optimization slice:

- If `firstSelectedHourVisibleMs` is dominated by pre-scene startup, plan startup/payload/model/BVH work.
- If `oneHourDispatchMs` dominates, plan compute shader / dispatch / selected-hour pipeline optimization.
- If `renderSceneSyncTotalMs`, `renderSurfaceMeshMs`, `renderStorageInitWaitMs`, or `renderQueueDrainMs` dominates, plan render-surface reuse, prewarm, or synchronization reduction.
- If app-owned UTCI/WebGPU memory is already too high at 2m/Ness Tziona, plan memory/tiling before UI polish.
- If timings are acceptable but the Performance tab reveals confusing states, plan UX polish around loading and readiness.
