# WebGPU Rectangular Pointwise Parity Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove mesh-grid logic and deliver strict point-to-point WebGPU parity against the Python baseline by enforcing one canonical rectangular grid and aligning stage semantics (solar, sky, MRT, UTCI).

**Architecture:** The parity path uses one deterministic grid source: rectangular-from-bounds only. WebGPU compute keeps running in-browser, but all parity-sensitive contracts (grid ordering, sky normalization, weather/sun indexing, domain handling) are made explicit and test-locked. Comparison shifts from distribution-only fallback to strict pointwise checks whenever canonical grid metadata is present.

**Tech Stack:** SvelteKit, TypeScript, WebGPU/WGSL, Vitest, Playwright, Node `tsx` scripts.

---

### Task 1: Delete Mesh-Grid Path (Cleanup First)

**Files:**
- Modify: `viewer/src/lib/compute/grid-generator.ts`
- Modify: `viewer/src/lib/compute/liveUtciAnalysis.ts`
- Modify: `viewer/src/lib/compute/compute-manager.ts`
- Modify: `viewer/src/routes/debug-webgpu-utci/+page.svelte`
- Test: `viewer/tests/compute/grid-generator.test.ts`
- Test: `viewer/tests/compute/live-utci-analysis.test.ts`

**Step 1: Write failing test that mesh-grid API is removed**

```ts
import { describe, it, expect } from 'vitest';
import * as grid from '$lib/compute/grid-generator';

describe('mesh-grid removal', () => {
  it('does not export generateGridFromMesh', () => {
    expect((grid as Record<string, unknown>).generateGridFromMesh).toBeUndefined();
  });
});
```

**Step 2: Run test to verify it fails**

Run: `cd viewer && npx vitest run tests/compute/grid-generator.test.ts -v`  
Expected: FAIL because `generateGridFromMesh` is still exported/used.

**Step 3: Remove mesh-grid implementation and imports**

```ts
// grid-generator.ts
// Keep only createRectangularGridFromBounds and related types.
// Delete generateGridFromMesh and BVH/raycast extension setup used only by that path.
```

**Step 4: Remove mesh fallback branches from live compute path**

```ts
// liveUtciAnalysis.ts
if (!bounds || !workerResult) {
  throw new Error('Rectangular parity path requires bounds + serialized BVH.');
}
```

**Step 5: Run focused tests and checkpoint**

Run: `cd viewer && npx vitest run tests/compute/grid-generator.test.ts tests/compute/live-utci-analysis.test.ts tests/compute/compute-manager.test.ts -v`  
Expected: PASS with mesh-grid references removed.

---

### Task 2: Canonical Rectangular Grid Contract (Count + Coordinates + Ordering)

**Files:**
- Modify: `viewer/src/lib/compute/analysisGridFromBounds.ts`
- Create: `viewer/src/lib/parity/gridCanonical.ts`
- Create: `viewer/tests/parity/gridCanonical.test.ts`
- Modify: `viewer/src/lib/compute/compute-manager.ts`

**Step 1: Write failing test for deterministic canonical ordering**

```ts
import { describe, it, expect } from 'vitest';
import { canonicalGridChecksum } from '$lib/parity/gridCanonical';

describe('canonical grid ordering', () => {
  it('produces stable checksum for fixed bounds/grid', () => {
    const checksum = canonicalGridChecksum({
      bounds: { x_min: 0, x_max: 4, y_min: 0, y_max: 4, z: 0.9 },
      gridSize: 2,
      coordinateSystem: 'xy_ground'
    });
    expect(checksum).toBe('REPLACE_WITH_EXPECTED_HASH');
  });
});
```

**Step 2: Run test to verify it fails**

Run: `cd viewer && npx vitest run tests/parity/gridCanonical.test.ts -v`  
Expected: FAIL because helper/checksum does not exist.

**Step 3: Implement canonical helper and ordering policy**

```ts
// gridCanonical.ts
export function canonicalGridPoints(...) { /* returns Float32Array x,y,z in strict order */ }
export function canonicalGridChecksum(...) { /* stable hash over rounded coords */ }
// Ordering policy: outer loop X ascending, inner loop Z ascending, inclusive upper bound epsilon.
```

**Step 4: Use helper in compute-manager for all parity grids**

```ts
// compute-manager.ts
const canonical = canonicalGridPoints({ bounds, gridSize, coordinateSystem, originOffset });
gridPoints = canonical.points;
numPoints = canonical.numPoints;
```

**Step 5: Re-run tests and checkpoint**

Run: `cd viewer && npx vitest run tests/parity/gridCanonical.test.ts tests/compute/compute-manager.test.ts -v`  
Expected: PASS and stable checksum in test fixtures.

---

### Task 3: Strict Pointwise Compare Mode (No More Silent Distribution Fallback)

**Files:**
- Modify: `viewer/src/lib/parity/compareIntermediates.ts`
- Modify: `viewer/scripts/compare-parity.ts`
- Create: `viewer/tests/parity/compareIntermediates-pointwise.test.ts`

**Step 1: Write failing test for strict mode length mismatch**

```ts
import { describe, it, expect } from 'vitest';
import { compareIntermediates } from '$lib/parity/compareIntermediates';

describe('strict pointwise parity', () => {
  it('fails immediately on length mismatch', () => {
    expect(() =>
      compareIntermediates({
        ref: new Float32Array([1, 2, 3]),
        webgpu: new Float32Array([1, 2]),
        tolerance: 1e-5
      })
    ).toThrow(/Length mismatch/);
  });
});
```

**Step 2: Run test to verify it fails**

Run: `cd viewer && npx vitest run tests/parity/compareIntermediates-pointwise.test.ts -v`  
Expected: FAIL until strict mode is wired in compare script.

**Step 3: Add CLI mode and enforce strict for parity runs**

```ts
// compare-parity.ts
// add --mode strict|stats (default strict for canonical grid path)
if (mode === 'strict') {
  // use compareIntermediates with same-length requirement
} else {
  // current stats mode fallback
}
```

**Step 4: Add clear error output for non-comparable artifacts**

```ts
console.log('solar: FAIL (strict mode requires same length/order artifacts)');
process.exitCode = 1;
```

**Step 5: Verify and checkpoint**

Run: `cd viewer && npx vitest run tests/parity/compareIntermediates-pointwise.test.ts -v`  
Run: `cd viewer && npx tsx scripts/compare-parity.ts --base-path ../data/analyses/Ben-Gurion/20250815_grid_2m_fullday --mode strict`  
Expected: test PASS; script fails fast with explicit reason if artifacts are not canonical.

---

### Task 4: Sky Normalization Contract (Exactly Once)

**Files:**
- Modify: `viewer/src/routes/debug-webgpu-utci/+page.svelte`
- Modify: `viewer/tests/e2e/collect-webgpu-parity.spec.ts`
- Modify: `viewer/src/lib/parity/skyScale.ts`
- Test: `viewer/tests/parity/skyScale.test.ts`

**Step 1: Write failing idempotency test**

```ts
import { describe, it, expect } from 'vitest';
import { normalizeSkyExposureToViewFactor } from '$lib/parity/skyScale';

describe('sky normalization', () => {
  it('is idempotent when input already in 0..1', () => {
    const once = normalizeSkyExposureToViewFactor([0, 0.5, 1]);
    const twice = normalizeSkyExposureToViewFactor(once);
    expect(twice).toEqual(once);
  });
});
```

**Step 2: Run test to verify current behavior**

Run: `cd viewer && npx vitest run tests/parity/skyScale.test.ts -v`  
Expected: FAIL if helper is not idempotent-aware.

**Step 3: Make one source of truth + remove second normalize call**

```ts
// collect-webgpu-parity.spec.ts
// remove normalizeSky(parityIntermediates.skyExposure)
// write parityIntermediates.skyExposure directly when contract says already normalized.
```

**Step 4: Add explicit schema comment in collect output**

```ts
// _webgpu_sky.json contains skyExposure in [0,1] normalized once in debug page.
```

**Step 5: Verify and checkpoint**

Run: `cd viewer && npx vitest run tests/parity/skyScale.test.ts -v`  
Run: `cd viewer && npx playwright test tests/e2e/collect-webgpu-parity.spec.ts`  
Expected: sky values stay in 0..1 with realistic mean (not ~0.005 collapse).

---

### Task 5: Sun/Weather Time Alignment Lock

**Files:**
- Modify: `viewer/src/lib/compute/compute-manager.ts`
- Create: `viewer/tests/compute/weather-index-alignment.test.ts`
- Create: `viewer/tests/compute/sun-vector-alignment.test.ts`

**Step 1: Write failing weather alignment test**

```ts
it('packs hour 0 weather from EPW hour 1 record by explicit contract', async () => {
  // build deterministic EPW with unique dryBulb per hour, assert packed[0] equals expected
});
```

**Step 2: Run test to verify baseline behavior**

Run: `cd viewer && npx vitest run tests/compute/weather-index-alignment.test.ts -v`  
Expected: FAIL until assertions match explicit contract and implementation is consistent.

**Step 3: Write failing sun-vector parity fixture test**

```ts
it('uses fixture vectors unchanged in parity mode', async () => {
  // assert uploadStaticData.sunVectors equals injected fixture Float32Array
});
```

**Step 4: Implement explicit contract comments + guard checks**

```ts
// compute-manager.ts
// Document exact month/day/hour mapping and add assertions for array lengths/time indices.
```

**Step 5: Verify and checkpoint**

Run: `cd viewer && npx vitest run tests/compute/weather-index-alignment.test.ts tests/compute/sun-vector-alignment.test.ts -v`  
Expected: PASS with deterministic packed weather/sun behavior.

---

### Task 6: MRT Equivalence Program (In-House Replacement for Ladybug Behavior)

**Files:**
- Create: `viewer/src/lib/compute/mrtReference.ts`
- Modify: `viewer/src/lib/compute/shaders/mrt_utci.wgsl`
- Create: `viewer/tests/compute/mrt-reference-vs-shader.test.ts`
- Create: `viewer/tests/fixtures/parity/mrt-fixtures.json`

**Step 1: Write failing fixture test for MRT delta bounds**

```ts
it('shader MRT matches reference MRT within epsilon on canonical fixtures', () => {
  // compare per-point-hour MRT from fixture inputs
  expect(maxAbsDiff).toBeLessThan(0.25);
});
```

**Step 2: Run test to confirm current shader mismatch**

Run: `cd viewer && npx vitest run tests/compute/mrt-reference-vs-shader.test.ts -v`  
Expected: FAIL with current simplified shader.

**Step 3: Implement TS reference MRT matching desired in-house contract**

```ts
// mrtReference.ts
export function computeMrtReference(inputs: {...}): number {
  // exact agreed formula constants and stage semantics
}
```

**Step 4: Port agreed formula to WGSL and align constants**

```wgsl
// mrt_utci.wgsl
// replace simplified terms where needed; keep constants synced to TS reference.
```

**Step 5: Verify and checkpoint**

Run: `cd viewer && npx vitest run tests/compute/mrt-reference-vs-shader.test.ts -v`  
Expected: PASS with bounded error on fixtures.

---

### Task 7: UTCI Domain/Boundary Equivalence

**Files:**
- Modify: `viewer/src/lib/compute/utci.ts`
- Modify: `viewer/src/lib/compute/shaders/mrt_utci.wgsl`
- Create: `viewer/tests/compute/utci-domain-parity.test.ts`

**Step 1: Write failing domain behavior tests**

```ts
it('returns NaN outside validity domain (policy mode)', () => {
  expect(Number.isNaN(calculateUTCI(60, 60, 1, 50))).toBe(true);
});
```

**Step 2: Run tests to capture current behavior**

Run: `cd viewer && npx vitest run tests/compute/utci-domain-parity.test.ts -v`  
Expected: FAIL where shader/TS contracts diverge.

**Step 3: Add explicit policy mode and align shader branch**

```ts
// utci.ts
export type UtciPolicy = 'strict-domain' | 'clamped-domain';
```

```wgsl
// mrt_utci.wgsl
// apply same policy logic as TS path (or explicit documented difference with tests).
```

**Step 4: Verify boundary averaging semantics**

```ts
// assert last-hour behavior matches chosen policy (single-hour vs duplicated next-hour).
```

**Step 5: Verify and checkpoint**

Run: `cd viewer && npx vitest run tests/compute/utci-domain-parity.test.ts -v`  
Expected: PASS with explicit, documented domain behavior.

---

### Task 8: Pointwise Parity E2E Harness and Visual-Complexity Diagnostics

**Files:**
- Modify: `viewer/tests/e2e/collect-webgpu-parity.spec.ts`
- Modify: `viewer/scripts/compare-parity.ts`
- Create: `viewer/src/lib/parity/spatialComplexity.ts`
- Create: `viewer/tests/parity/spatialComplexity.test.ts`

**Step 1: Write failing spatial-complexity diagnostic test**

```ts
it('computes gradient-energy and local-variance metrics for UTCI field', () => {
  const metrics = computeSpatialComplexity(field, width, height);
  expect(metrics.gradientEnergy).toBeGreaterThan(0);
});
```

**Step 2: Run test to verify missing module**

Run: `cd viewer && npx vitest run tests/parity/spatialComplexity.test.ts -v`  
Expected: FAIL until helper exists.

**Step 3: Implement diagnostics helper and report output**

```ts
// spatialComplexity.ts
export function computeSpatialComplexity(...) { return { gradientEnergy, variance, entropy }; }
```

**Step 4: Extend compare output for strict pointwise + complexity report**

```ts
// compare-parity.ts
// print RMSE/max abs/p95 per stage + complexity deltas for UTCI.
```

**Step 5: End-to-end verification checkpoint**

Run: `cd viewer && npx playwright test tests/e2e/collect-webgpu-parity.spec.ts`  
Run: `cd viewer && npx tsx scripts/compare-parity.ts --base-path ../data/analyses/Ben-Gurion/20250815_grid_2m_fullday --mode strict --report parity-report.json`  
Expected: strict pointwise output with explicit fail/pass reasons and complexity deltas.

---

## Execution Order

1. Task 1 (delete mesh-grid)
2. Task 2 (canonical rectangular grid + checksum)
3. Task 3 (strict compare mode)
4. Task 4 (sky normalization exactly once)
5. Task 5 (sun/weather alignment lock)
6. Task 6 (MRT equivalence)
7. Task 7 (UTCI domain/boundary equivalence)
8. Task 8 (E2E strict parity + complexity diagnostics)

---

## Verification Bundle (after all tasks)

Run: `cd viewer && npx vitest run tests/compute/grid-generator.test.ts tests/compute/compute-manager.test.ts tests/compute/live-utci-analysis.test.ts tests/parity/*.test.ts -v`  
Run: `cd viewer && npx playwright test tests/e2e/collect-webgpu-parity.spec.ts`  
Run: `cd viewer && npx tsx scripts/compare-parity.ts --base-path ../data/analyses/Ben-Gurion/20250815_grid_2m_fullday --mode strict --report parity-report.json`

Expected:
- No mesh-grid references in parity path.
- WebGPU and Python artifacts share canonical grid count/order.
- Strict pointwise comparison active and meaningful.
- MRT/UTCI gaps reported with deterministic diagnostics.

---

Plan complete and saved to `docs/plans/2026-03-14-webgpu-rectangular-pointwise-parity-implementation-plan.md`.

**Two execution options:**

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

**Which approach?**
