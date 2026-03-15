# WebGPU Pointwise UTCI + ERF/DMRT Parity Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Complete WebGPU migration parity by collecting ERF/DMRT intermediates and enforcing strict pointwise comparison for UTCI and all available intermediate stages.

**Architecture:** Keep the existing deterministic browser collection path (`debug-webgpu-utci` + Playwright collect) and tighten data contracts so both Python and WebGPU artifacts expose comparable arrays with identical length/order semantics. Extend the parity comparison/report pipeline to treat UTCI like other pointwise stages (not only range-level), while preserving stats mode for diagnosis.

**Tech Stack:** SvelteKit, TypeScript, Playwright, Node `tsx` scripts, WebGPU compute pipeline, parity report utilities.

---

### Task 1: Lock parity artifact contracts (failing tests first)

**Files:**
- Modify: `viewer/tests/e2e/collect-webgpu-parity.spec.ts`
- Create: `viewer/tests/parity/utci-pointwise-contract.test.ts`
- Modify: `viewer/src/lib/parity/loadWebgpuCollectedFromFs.ts`
- Modify: `viewer/src/lib/parity/loadReferenceIntermediatesFromFs.ts`

**Step 1: Write the failing test**

```ts
import { describe, it, expect } from 'vitest';
import { loadWebgpuCollectedFromFs } from '$lib/parity/loadWebgpuCollectedFromFs';

describe('WebGPU parity artifact contract', () => {
	it('requires pointwise UTCI arrays and optional ERF/DMRT arrays with strict shape validation', async () => {
		const data = await loadWebgpuCollectedFromFs('/tmp/fake-base');
		expect(data.utci?.utciByHour).toBeDefined();
		expect(Array.isArray(data.utci?.utciByHour)).toBe(true);
	});
});
```

**Step 2: Run test to verify it fails**

Run: `npm run test -- tests/parity/utci-pointwise-contract.test.ts`  
Expected: FAIL due to missing strict validator behavior and/or missing fixture contract.

**Step 3: Write minimal implementation**

```ts
// In loadWebgpuCollectedFromFs.ts:
// - Validate utciByHour.length === numHours
// - Validate each hour length === numPoints
// - Validate optional positions length === numPoints * 3
// - Validate optional short_erf/long_erf/short_dmrt/long_dmrt length === numPoints * numHours
// - Throw clear errors for strict contract violations
```

**Step 4: Run test to verify it passes**

Run: `npm run test -- tests/parity/utci-pointwise-contract.test.ts`  
Expected: PASS.

**Step 5: Review checkpoint**

Summarize contract decisions:
- Required UTCI pointwise fields
- Optional ERF/DMRT fields and strict length policy
- Error message format for invalid artifacts.

---

### Task 2: Ensure collection exports ERF/DMRT and pointwise UTCI deterministically

**Files:**
- Modify: `viewer/src/routes/debug-webgpu-utci/+page.svelte`
- Modify: `viewer/tests/e2e/collect-webgpu-parity.spec.ts`
- Test: `viewer/tests/e2e/collect-webgpu-parity.spec.ts`

**Step 1: Write the failing test**

```ts
test('writes _webgpu_mrt.json with MRT + available ERF/DMRT arrays and _webgpu_utci.json with pointwise utciByHour', async () => {
	// assert JSON files contain expected keys and strict lengths
});
```

**Step 2: Run test to verify it fails**

Run: `npm run parity:collect-webgpu`  
Expected: FAIL with clear assertion about missing field or invalid length.

**Step 3: Write minimal implementation**

```ts
// In +page.svelte:
// - Keep __parityIntermediates__ payload stable:
//   { solarExposure, skyExposure, mrt?, shortErf?, longErf?, shortDmrt?, longDmrt?, numPoints, numHours }
// - Keep __parityResults__ pointwise:
//   { utciByHour, positions, numPoints, numHours }
//
// In collect-webgpu-parity.spec.ts:
// - Preserve staged export flow
// - Serialize ERF/DMRT arrays into _webgpu_mrt.json when present
// - Preserve pointwise utciByHour in _webgpu_utci.json
```

**Step 4: Run test to verify it passes**

Run: `npm run parity:collect-webgpu`  
Expected: PASS, and files written with strict contract shape.

**Step 5: Review checkpoint**

Capture sample artifact snippets (keys only) and confirm no phase timeout regressions.

---

### Task 3: Add pointwise UTCI comparator and strict CLI/report support

**Files:**
- Create: `viewer/src/lib/parity/compareUtciPointwise.ts`
- Create: `viewer/tests/parity/compareUtciPointwise.test.ts`
- Modify: `viewer/scripts/compare-parity.ts`
- Modify: `viewer/src/lib/parity/buildParityReport.ts`

**Step 1: Write the failing test**

```ts
import { compareUtciPointwise } from '$lib/parity/compareUtciPointwise';

it('fails strict mode when any utcibyhour cell exceeds tolerance', () => {
	const ref = [[25, 26], [27, 28]];
	const wg = [[25, 26], [27, 30]];
	const result = compareUtciPointwise({ ref, webgpu: wg, tolerance: 0.5 });
	expect(result.pass).toBe(false);
	expect(result.maxError).toBe(2);
});
```

**Step 2: Run test to verify it fails**

Run: `npm run test -- tests/parity/compareUtciPointwise.test.ts`  
Expected: FAIL because comparator does not exist yet.

**Step 3: Write minimal implementation**

```ts
export function compareUtciPointwise(params: {
	ref: readonly (readonly number[])[];
	webgpu: readonly (readonly number[])[];
	tolerance: number;
}) {
	// validate hour count and per-hour point count
	// compute rmse, maxError, meanDiff, worstIndices
	// pass when maxError <= tolerance
}
```

**Step 4: Integrate into strict compare script**

```ts
// compare-parity.ts strict mode:
// - replace range-only UTCI pass/fail with pointwise comparator
// - keep range metrics as secondary diagnostics
// - include utcibyhour strict result in report JSON
```

Run:  
`npx tsx scripts/compare-parity.ts --base-path data/analyses/Ben-Gurion/20250815_grid_2m_fullday --mode strict --report parity-report.json`  
Expected: UTCI line reports pointwise metrics (rmse/maxError) and pass/fail based on tolerance.

**Step 5: Review checkpoint**

Confirm strict report now has:
- `utci.pointwise` details
- `utci.range` secondary diagnostics
- consistent terminal/report output.

---

### Task 4: Enforce ERF/DMRT comparison policy in strict mode

**Files:**
- Modify: `viewer/scripts/compare-parity.ts`
- Modify: `viewer/tests/parity/compareIntermediates-pointwise.test.ts`
- Modify: `viewer/tests/fixtures/parity/mrt-fixtures.json`

**Step 1: Write the failing test**

```ts
it('strict mode fails when only one side has short_erf', async () => {
	// build fixture where ref has short_erf and webgpu omits it
	// expect strict fail with explicit reason
});
```

**Step 2: Run test to verify it fails**

Run: `npm run test -- tests/parity/compareIntermediates-pointwise.test.ts`  
Expected: FAIL (current behavior skips silently).

**Step 3: Write minimal implementation**

```ts
// compare-parity.ts strict:
// - if component present in one side but absent in the other => FAIL with reason
// - if present in both => pointwise compare
// - if absent in both => SKIP
```

**Step 4: Run test to verify it passes**

Run: `npm run test -- tests/parity/compareIntermediates-pointwise.test.ts`  
Expected: PASS with explicit mismatch policy coverage.

**Step 5: Review checkpoint**

Document strict policy table:
- both present -> compare
- both absent -> skip
- one present -> fail.

---

### Task 5: Add pointwise UTCI E2E parity assertion (optional tolerances by mode)

**Files:**
- Modify: `viewer/tests/e2e/parity-ben-gurion.spec.ts`
- Optionally modify: `viewer/playwright.collect.config.ts` (timeouts only if needed)

**Step 1: Write the failing test**

```ts
test('pointwise UTCI parity compares ref vs webgpu when grid aligns', async ({ page }) => {
	// load __parityResults__, load reference, compare per-hour arrays
	// assert maxError <= UTCI_POINTWISE_TOLERANCE
});
```

**Step 2: Run test to verify it fails**

Run: `npm run test:e2e:parity -- --project=chromium`  
Expected: FAIL until pointwise comparator and/or artifact loading is wired.

**Step 3: Write minimal implementation**

```ts
// parity-ben-gurion.spec.ts:
// - wait for parity status
// - read __parityResults__.utciByHour
// - compare against loadReferenceFromFs(...).data.utciByHour
// - report worst hour/index in assertion message
```

**Step 4: Run test to verify it passes**

Run: `npm run test:e2e:parity -- --project=chromium`  
Expected: PASS or actionable fail with exact (hour, point) mismatch context.

**Step 5: Review checkpoint**

Confirm e2e failure message includes:
- phase/status
- worst UTCI index/hour
- ref/webgpu/diff values.

---

### Task 6: Final verification matrix and docs update

**Files:**
- Modify: `docs/plans/2026-03-14-webgpu-rectangular-pointwise-parity-implementation-plan.md`
- Optionally modify: `viewer/scripts/compare-parity.ts` (CLI help text only)

**Step 1: Run targeted unit tests**

Run:  
- `npm run test -- tests/parity/compareUtciPointwise.test.ts`  
- `npm run test -- tests/parity/compareIntermediates-pointwise.test.ts`  
- `npm run test -- tests/parity/gridCanonical.test.ts`  
Expected: PASS.

**Step 2: Run compute-focused tests**

Run:  
- `npm run test -- tests/compute/mrt-reference-vs-shader.test.ts`  
- `npm run test -- tests/compute/sun-vector-alignment.test.ts`  
- `npm run test -- tests/compute/weather-index-alignment.test.ts`  
Expected: PASS.

**Step 3: Run deterministic collection**

Run: `npm run parity:collect-webgpu`  
Expected: PASS with phase completion and generated artifacts.

**Step 4: Run strict pointwise compare**

Run:  
`npx tsx scripts/compare-parity.ts --base-path data/analyses/Ben-Gurion/20250815_grid_2m_fullday --mode strict --report parity-report.json`  
Expected: strict output includes pointwise UTCI + ERF/DMRT policy results.

**Step 5: Documentation checkpoint**

Update plan doc status section with:
- what now compares pointwise
- remaining known deviations (if any)
- exact commands to reproduce.

---

## Acceptance Criteria

- `collect-webgpu-parity.spec.ts` exports deterministic artifacts including pointwise UTCI and available ERF/DMRT arrays.
- `compare-parity.ts --mode strict` performs pointwise comparisons for:
  - solar
  - sky
  - mrt
  - UTCI
  - ERF/DMRT components (with explicit missing-data policy)
- Strict terminal output and strict report JSON are consistent.
- E2E parity test reports precise mismatch context (phase + index/hour values).
- No new lint/test regressions in touched test suites.

---

Plan complete and saved to `docs/plans/2026-03-14-webgpu-pointwise-utci-erf-dmrt-parity-plan.md`. Two execution options:

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

Which approach?
