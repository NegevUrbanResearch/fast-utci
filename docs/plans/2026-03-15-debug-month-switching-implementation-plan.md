# Debug Month Switching Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add month switching to the debug WebGPU UTCI viewer: precompute all 12 months (15th of each) on load, add a Day/Month radial picker with segmented toggle, use year-gradient from Figma for month ring. Base .bin layer stays August; only live WebGPU layer responds to month changes. Month scrubbing is instant (index change only).

**Architecture:** Extend `createLiveUtciAnalysisFromCompute` to use `numMonths=12` and `startMonth=1`. Pipeline already supports multi-month; buffers are sized for `numPoints * 288`. Read back all 288 UTCI slices into `utciByHour`. Add `currentMonth` (0-11) to viewerStore. Introduce `getEffectiveHourIndex(analysis, hour, month)` so point cloud and tooltip use `monthIndex*24+hourIndex` for live 12-month analyses. RadialTimePicker gains a Day/Month segmented control; in Month mode shows 12 months with year gradient. Default `currentMonth=7` (August) to match .bin.

**Tech Stack:** SvelteKit, Svelte 5, TypeScript, WebGPU, Three.js, existing RadialTimePicker and dayGradient patterns.

---

### Task 1: Add currentMonth to viewer store and types

**Files:**
- Modify: `viewer/src/lib/types/viewer.ts`
- Modify: `viewer/src/lib/stores/viewerStore.ts`
- Test: `viewer/tests/stores/viewerStore.test.ts` (create if missing)

**Step 1: Extend ViewerState type**

In `viewer/src/lib/types/viewer.ts`, add `currentMonth` to the interface:

```ts
export interface ViewerState {
	currentHour: number;
	currentMonth: number;  // 0-11, 0=Jan, 7=Aug. Used only for multi-month live analysis.
	colorMode: ColorMode;
	metricType: MetricType;
	utciVisible: boolean;
	analysisId: string | null;
	loading: boolean;
	error: string | null;
	theme: 'dark' | 'light';
}
```

**Step 2: Add setCurrentMonth and default currentMonth in viewerStore**

In `viewer/src/lib/stores/viewerStore.ts`:

- Add `currentMonth: 7` to initial state (August = index 7).
- Add setter:

```ts
export function setCurrentMonth(month: number): void {
	const clamped = Math.max(0, Math.min(11, month));
	viewerStore.update((state) => ({ ...state, currentMonth: clamped }));
}
```

**Step 3: Run build to verify**

Run: `cd viewer && npm run build`
Expected: No type errors.

**Step 4: Review checkpoint**

Confirm ViewerState and viewerStore have currentMonth. No tests required if none exist for viewerStore; skip test file creation for this task.

---

### Task 2: Add numMonths to AnalysisMetadata and getEffectiveHourIndex helper

**Files:**
- Modify: `viewer/src/lib/types/analysis.ts`
- Create: `viewer/src/lib/utils/effectiveHourIndex.ts`
- Test: `viewer/tests/utils/effectiveHourIndex.test.ts`

**Step 1: Add numMonths to AnalysisMetadata**

In `viewer/src/lib/types/analysis.ts`, add optional field to `AnalysisMetadata`:

```ts
// In AnalysisMetadata interface, add:
/** Number of representative months when analysis has multi-month data (e.g. 12 for full year). */
num_months?: number;
```

**Step 2: Write the failing test for getEffectiveHourIndex**

Create `viewer/tests/utils/effectiveHourIndex.test.ts`:

```ts
import { describe, it, expect } from 'vitest';
import { getEffectiveHourIndex } from '$lib/utils/effectiveHourIndex';
import type { Analysis } from '$lib/types/analysis';

describe('getEffectiveHourIndex', () => {
	it('returns hourIndex when analysis has no num_months', () => {
		const analysis = {
			metadata: { num_months: undefined },
			data: { numHours: 24 }
		} as unknown as Analysis;
		expect(getEffectiveHourIndex(analysis, 12, 7)).toBe(12);
	});

	it('returns monthIndex*24 + hourIndex when analysis has num_months=12', () => {
		const analysis = {
			metadata: { num_months: 12 },
			data: { numHours: 288 }
		} as unknown as Analysis;
		expect(getEffectiveHourIndex(analysis, 12, 0)).toBe(12);   // Jan, noon
		expect(getEffectiveHourIndex(analysis, 12, 7)).toBe(180);  // Aug, noon (7*24+12)
	});
});
```

**Step 3: Run test to verify it fails**

Run: `cd viewer && npm run test -- tests/utils/effectiveHourIndex.test.ts`
Expected: FAIL (module/function not found).

**Step 4: Implement getEffectiveHourIndex**

Create `viewer/src/lib/utils/effectiveHourIndex.ts`:

```ts
import type { Analysis } from '$lib/types/analysis';

/**
 * Returns the UTCI slice index for the given hour and month.
 * For single-month analyses (e.g. .bin): returns hourIndex.
 * For 12-month analyses (live WebGPU): returns monthIndex*24 + hourIndex.
 */
export function getEffectiveHourIndex(
	analysis: Analysis | null,
	hourIndex: number,
	monthIndex: number
): number {
	if (!analysis?.metadata?.num_months || analysis.metadata.num_months <= 1) {
		return hourIndex;
	}
	return monthIndex * 24 + hourIndex;
}
```

**Step 5: Run test to verify it passes**

Run: `cd viewer && npm run test -- tests/utils/effectiveHourIndex.test.ts`
Expected: PASS.

**Step 6: Review checkpoint**

Confirm helper and types are in place.

---

### Task 3: Add year gradient from Figma

**Files:**
- Create: `viewer/src/lib/utils/yearGradient.ts`
- Test: `viewer/tests/utils/yearGradient.test.ts`

**Step 1: Extract year clock gradient from Figma**

From the Time is Relative Figma site (https://flight-swoop-66337217.figma.site/), the year clock (Mu component) uses this gradient (from the provided web search / JS):

```ts
// viewer/src/lib/utils/yearGradient.ts

export const YEAR_GRADIENT_STOPS: Array<{ stop: number; color: string }> = [
	{ stop: 0, color: '#8B6CC7' },   // Winter solstice - violet/purple
	{ stop: 0.06, color: '#6E8ED8' },
	{ stop: 0.12, color: '#5AA0E8' },
	{ stop: 0.18, color: '#48B8D8' },
	{ stop: 0.25, color: '#40C8A0' },  // March equinox - teal
	{ stop: 0.32, color: '#50D878' },
	{ stop: 0.38, color: '#78E060' },
	{ stop: 0.44, color: '#B8E848' },
	{ stop: 0.5, color: '#E8E040' },   // June solstice - bright yellow
	{ stop: 0.56, color: '#F0C840' },
	{ stop: 0.62, color: '#F0A840' },
	{ stop: 0.68, color: '#E88840' },
	{ stop: 0.75, color: '#E06848' },  // September equinox - coral-orange
	{ stop: 0.82, color: '#D84858' },
	{ stop: 0.88, color: '#C84070' },
	{ stop: 0.94, color: '#A850A0' },
	{ stop: 1, color: '#8B6CC7' }
];

export const YEAR_RING_INNER = 84;
export const YEAR_RING_OUTER = 110;

/**
 * Returns a conic-gradient() CSS value for the year ring.
 * 0 = Jan, 0.5 = Jul; winter at top (12 o'clock).
 */
export function getYearRingConicGradient(): string {
	const reordered: Array<{ pct: number; color: string }> = [];
	for (let i = 0; i < YEAR_GRADIENT_STOPS.length; i++) {
		const stop = YEAR_GRADIENT_STOPS[i];
		const rotated = (stop.stop + 0.5) % 1;
		reordered.push({ pct: rotated * 100, color: stop.color });
	}
	reordered.sort((a, b) => a.pct - b.pct);
	const janColor = reordered[0].color;
	if (reordered[reordered.length - 1].pct < 100) {
		reordered.push({ pct: 100, color: janColor });
	}
	const parts = reordered.map(({ pct, color }) => `${color} ${pct}%`);
	return `conic-gradient(from 0deg, ${parts.join(', ')})`;
}
```

**Step 2: Write minimal test**

Create `viewer/tests/utils/yearGradient.test.ts`:

```ts
import { describe, it, expect } from 'vitest';
import { getYearRingConicGradient, YEAR_GRADIENT_STOPS } from '$lib/utils/yearGradient';

describe('yearGradient', () => {
	it('returns a conic-gradient string', () => {
		const g = getYearRingConicGradient();
		expect(g).toMatch(/^conic-gradient/);
		expect(g).toContain('#8B6CC7');
	});

	it('has 17 gradient stops', () => {
		expect(YEAR_GRADIENT_STOPS.length).toBe(17);
	});
});
```

**Step 3: Run test**

Run: `cd viewer && npm run test -- tests/utils/yearGradient.test.ts`
Expected: PASS.

**Step 4: Review checkpoint**

Confirm year gradient matches Figma year clock. If the live site differs, adjust colors in YEAR_GRADIENT_STOPS. User can verify by opening https://flight-swoop-66337217.figma.site/ and checking the year clock colors.

---

### Task 4: Extend live analysis to compute 12 months

**Files:**
- Modify: `viewer/src/lib/compute/liveUtciAnalysis.ts`
- Modify: `viewer/src/routes/debug-webgpu-utci/+page.svelte` (analysisParams)
- Test: `viewer/tests/compute/live-utci-analysis.test.ts`

**Step 1: Write failing test for 12-month output**

In `viewer/tests/compute/live-utci-analysis.test.ts`, add:

```ts
it('produces 288 utciByHour slices when numMonths=12', async () => {
	const fakePipeline = createFakePipeline(); // use existing test helpers
	const result = await createLiveUtciAnalysisFromCompute(
		{
			analysisId: 'test',
			baseMetadata: mockMetadata,
			workerResult: mockWorkerResult,
			epwContent: mockEpwContent,
			startMonth: 1,
			numMonths: 12  // NEW PARAM - need to add to interface
		},
		{ pipeline: fakePipeline }
	);
	expect(result.data.utciByHour.length).toBe(288);
	expect(result.metadata.num_months).toBe(12);
});
```

Note: The LiveUtciAnalysisParams may need a `numMonths` override. If the interface does not have it, add it. The implementation will change `numMonths = 1` to `numMonths = params.numMonths ?? 1` and `startMonth` to `startMonth = params.startMonth ?? 1` when numMonths is 12.

**Step 2: Run test to verify it fails**

Run: `cd viewer && npm run test -- tests/compute/live-utci-analysis.test.ts`
Expected: FAIL or skip if test setup differs; adjust test to match existing patterns.

**Step 3: Implement 12-month compute in liveUtciAnalysis**

In `viewer/src/lib/compute/liveUtciAnalysis.ts`:

- Add to `LiveUtciAnalysisParams`: `numMonths?: number;` (default 1; pass 12 from debug page), `startMonth?: number` (already exists; use 1 when numMonths=12, 8 when numMonths=1).
- Change `const numMonths = 1` to `const numMonths = params.numMonths ?? 1`.
- When `numMonths > 1`, do NOT pass `sunVectorsFixture` to `initFromModelAndWeather`—`buildSunVectorsFixtureFromMetadata` repeats the .bin's single day for all months. Let ComputeManager compute real sun vectors from EPW for each month via `getSunVectors(location, month, 15)`.
- In the ComputeManager config, pass `numMonths` and `startMonth: 1` when numMonths is 12.
- In the readback loop, iterate over all `numMonths * numHours` slices (not just numHours):

```ts
// Replace the single loop over hourIndex with nested loops:
for (let monthOffset = 0; monthOffset < numMonths; monthOffset++) {
	for (let hourIndex = 0; hourIndex < numHours; hourIndex++) {
		const slice = await computeManager.getUtcisForMonthHour({
			monthIndex: monthOffset,
			hourIndex,
			numPoints: effectiveNumPoints,
			numMonths,
			numHours
		});
		// ... same stats and push logic ...
		utciByHour.push(effectiveSlice);
	}
}
```

- Set `liveMetadata.num_months = numMonths`.
- Set `liveData.numHours = numMonths * numHours` (288) so `utciByHour.length` matches.
- When calling `initFromModelAndWeather`, pass `sunVectorsFixture` only when `numMonths === 1` (to use .bin's sun positions). When `numMonths > 1`, omit it so ComputeManager computes per-month sun vectors from EPW.
- Update `hour_statistics` to cover 288 entries if needed for range calculation (or keep a single global range; existing code uses globalMin/globalMax across all slices).

**Step 4: Update debug page to pass numMonths=12**

In `viewer/src/routes/debug-webgpu-utci/+page.svelte`, in `computeLiveAnalysis`, change:

```ts
const analysisParams = {
	analysisId,
	baseMetadata: base.metadata,
	workerResult,
	epwContent,
	gridResolution,
	zHeight,
	numHours: base.data.numHours ?? base.metadata.hours.length ?? 24,
	startMonth: 1,
	numMonths: 12
};
```

**Step 5: Update prepareMeshPayloadForWorkerAsync call**

In the same file, ensure `numMonths: 12` is passed to `prepareMeshPayloadForWorkerAsync` if it affects preflight (it may not; verify in mergeAndBvhWorkerClient).

**Step 6: Run tests**

Run: `cd viewer && npm run test -- tests/compute/live-utci-analysis.test.ts`
Expected: PASS (after adapting the new test to the actual mock setup).

**Step 7: Review checkpoint**

Live analysis should produce 288 slices and metadata.num_months=12. Initial load will take longer (~1.2–3.6s); add progress UX in a later task if desired.

---

### Task 5: Wire getEffectiveHourIndex into point cloud and tooltip

**Files:**
- Modify: `viewer/src/lib/components/scene/UTCIPointCloud.svelte`
- Modify: `viewer/src/lib/services/pointCloudService.ts` (no API change; callers pass effective index)
- Modify: `viewer/src/routes/debug-webgpu-utci/+page.svelte` (tooltip)

**Step 1: Use getEffectiveHourIndex in UTCIPointCloud**

In `viewer/src/lib/components/scene/UTCIPointCloud.svelte`:

- Import: `import { getEffectiveHourIndex } from '$lib/utils/effectiveHourIndex';`
- In the reactive block, compute effective index before calling createUtciSurfaceMesh and updateUtciSurfaceTexture:

```ts
const effectiveHourIndex = getEffectiveHourIndex(
	analysis,
	viewerState?.currentHour ?? 0,
	viewerState?.currentMonth ?? 7
);
```

- Pass `effectiveHourIndex` instead of `viewerState.currentHour` to `createUtciSurfaceMesh` and `updateUtciSurfaceTexture`.
- Add `currentMonth` to `hasStateChanged` and `lastUpdateState` so texture updates when month changes.

**Step 2: Update tooltip in debug page**

In `viewer/src/routes/debug-webgpu-utci/+page.svelte`, `handleMouseMove` calls `getTooltipData(..., hourIndex, ...)`. Compute the correct hour index based on which analysis is hovered:

```ts
const hourForTooltip = targetAnalysis === liveAnalysis
	? getEffectiveHourIndex(targetAnalysis, $viewerStore.currentHour, $viewerStore.currentMonth ?? 7)
	: $viewerStore.currentHour;
```

Pass `hourForTooltip` to `getTooltipData` instead of `$viewerStore.currentHour`. Add `import { getEffectiveHourIndex } from '$lib/utils/effectiveHourIndex';`.

**Step 3: Verify no regressions**

Run: `cd viewer && npm run build`
Expected: No type errors. Manually test debug page: left (.bin) and right (live) both show correct UTCI; tooltip shows correct values on hover.

**Step 4: Review checkpoint**

Point cloud and tooltip use effective index for multi-month live analysis.

---

### Task 6: Add Day/Month segmented control and month radial to RadialTimePicker

**Files:**
- Modify: `viewer/src/lib/components/ui/RadialTimePicker.svelte`
- Create: `viewer/src/lib/utils/radialMonthPickerState.ts` (optional; can inline)
- Modify: `viewer/src/lib/components/ui/RadialTimePicker.svelte` styles

**Step 1: Add mode state and segmented control**

In `RadialTimePicker.svelte`:

- Add a prop or internal state: `mode: 'day' | 'month'` (default `'day'`).
- Add a segmented control (two buttons: "Day" | "Month") at the top-right inside the radial panel.
- When `mode === 'day'`: show existing hour dial (24 segments, day gradient).
- When `mode === 'month'`: show month dial (12 segments, year gradient).

**Step 2: Implement month dial**

- Use same dial structure as hour dial: `getPositionForIndex(i, 12, HANDLE_RADIUS)` for 12 segments.
- Labels: `['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']`.
- On pointer/keyboard: call `setCurrentMonth(index)` when in month mode.
- Center display: show month name when in month mode, e.g. `['Jan','Feb',...][currentMonth]`.
- Use `getYearRingConicGradient()` for the month ring background.

**Step 3: Wire mode to viewerStore**

- When mode is 'month', changing the dial updates `currentMonth`.
- When mode is 'day', changing the dial updates `currentHour`.
- Both values remain in the store; switching mode does not reset the other.

**Step 4: Conditional display of RadialTimePicker**

The RadialTimePicker is shown in the debug page sidebar when `$analysisStore && full_day && metricType === 'utci'`. Ensure it stays visible. The Day/Month toggle is only relevant when live analysis has 12 months; you can always show it (in Month mode with single-month analysis, currentMonth would just not affect the .bin layer).

**Step 5: Style the segmented control**

Add a compact segmented control at top-right of the radial panel (e.g. two pills: "Day" | "Month"), matching existing panel styling (backdrop blur, border-radius).

**Step 6: Review checkpoint**

Radial shows Day or Month mode; Month mode displays 12 months with year gradient; selection updates `currentMonth`. Day mode unchanged.

---

### Task 7: Add loading progress for 12-month compute

**Files:**
- Modify: `viewer/src/routes/debug-webgpu-utci/+page.svelte`

**Step 1: Show progress during compute**

The existing `showFullLoadOverlay` shows "Computing UTCI..." during live compute. Update the message to indicate progress when numMonths > 1. Options:

- Simple: Change text to "Computing full year UTCI…" during the 12-month run.
- Better: Emit progress from `createLiveUtciAnalysisFromCompute` (e.g. callback or store) and display "Computing month X/12…". This requires adding an optional `onProgress?: (month: number, total: number) => void` to the params.

**Step 2: Implement minimal progress (recommended)**

Add an optional progress callback to `createLiveUtciAnalysisFromCompute`:

```ts
onProgress?: (completed: number, total: number) => void;
```

In the readback loop, after each month completes, call `onProgress(monthOffset + 1, numMonths)`.

In the debug page, use a local state `liveComputeProgress: { current: number; total: number } | null` and pass the callback. Display "Computing month {current}/{total}…" in the overlay when progress is set.

**Step 3: Review checkpoint**

User sees progress during the 1–4 second full-year compute.

---

### Task 8: Parity readback and e2e compatibility (optional)

**Files:**
- Modify: `viewer/src/routes/debug-webgpu-utci/+page.svelte`

**Step 1: Ensure parity collection still works**

The `__parityResults__` and `__parityIntermediates__` are written after compute. With 12 months, `numMonths` in readback calls must be 12. Verify:

- `pipeline.readSolarExposureFull({ numPoints, numHours, numMonths: 12 })`
- `pipeline.readSkyExposure({ numPoints })`
- `pipeline.readMrtFull({ numPoints, numHours, numMonths: 12 })`

Update the readback block to use `numMonths: 12` when the live analysis has 12 months.

**Step 2: Parity report**

If e2e parity tests expect a single month, they may need to be updated to either (a) still run with numMonths=1 for parity, or (b) accept 12-month output. For this debug-only feature, (a) is simpler: add a query param or flag to force numMonths=1 for parity runs. Document this in the plan. Implementation: when `?parity=1` is in the URL, pass `numMonths: 1` to keep existing parity behavior.

**Step 3: Review checkpoint**

Parity collection and e2e tests continue to pass when using `?parity=1`.

---

### Task 9: Memory guard for large grids (optional)

**Files:**
- Modify: `viewer/src/lib/compute/liveUtciAnalysis.ts` or `viewer/src/routes/debug-webgpu-utci/+page.svelte`

**Step 1: Add grid size check**

Before starting 12-month compute, estimate memory: `numPoints * 288 * 4 * 3` (solar, MRT, UTCI) ≈ 3.5MB per 10k points. For >50k points (~175MB), consider warning the user or falling back to numMonths=1. Implementation: if `preflight.estimatedGridPoints > 50_000`, show a console warning and optionally restrict to 6 months or 1 month. This is optional and can be deferred.

**Step 2: Review checkpoint**

No mandatory memory guard for initial implementation; document as future improvement if skipped.

---

### Summary of file changes

| File | Action |
|------|--------|
| `viewer/src/lib/types/viewer.ts` | Add currentMonth to ViewerState |
| `viewer/src/lib/types/analysis.ts` | Add num_months to AnalysisMetadata |
| `viewer/src/lib/stores/viewerStore.ts` | Add currentMonth, setCurrentMonth |
| `viewer/src/lib/utils/effectiveHourIndex.ts` | Create helper |
| `viewer/src/lib/utils/yearGradient.ts` | Create year gradient |
| `viewer/src/lib/compute/liveUtciAnalysis.ts` | 12-month compute, num_months, progress |
| `viewer/src/lib/components/ui/RadialTimePicker.svelte` | Day/Month toggle, month dial |
| `viewer/src/lib/components/scene/UTCIPointCloud.svelte` | Use getEffectiveHourIndex |
| `viewer/src/lib/services/tooltipService.ts` | Caller passes effective index (or extend getTooltipData) |
| `viewer/src/routes/debug-webgpu-utci/+page.svelte` | numMonths=12, progress, tooltip, parity=1 fallback |
| `viewer/tests/utils/effectiveHourIndex.test.ts` | Create |
| `viewer/tests/utils/yearGradient.test.ts` | Create |
| `viewer/tests/compute/live-utci-analysis.test.ts` | Extend for 12-month |

---

### Verification checklist

- [ ] `cd viewer && npm run build` passes
- [ ] `npm run test` passes
- [ ] Debug page loads; live compute runs 12 months; overlay shows progress
- [ ] Month radial works; switching month updates live layer instantly
- [ ] Hour radial works; both .bin and live respond to hour
- [ ] Tooltip shows correct UTCI for both sides
- [ ] .bin (left) always shows August data; live (right) shows selected month
- [ ] Year gradient on month ring matches Figma year clock
- [ ] `?parity=1` preserves single-month parity behavior if needed

---

**Plan complete and saved to `docs/plans/2026-03-15-debug-month-switching-implementation-plan.md`.**

**Execution options:**

1. **Subagent-Driven (this session)** — Fresh subagent per task, review between tasks, fast iteration.

2. **Parallel Session (separate)** — Open a new session with executing-plans and run the plan there with checkpoints.

Which approach would you prefer?
