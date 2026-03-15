# WebGPU vs Python Parity: Findings and Lessons

Date: 2026-03-15  
Scope: `viewer/` parity pipeline + Python baseline artifacts for `Ben-Gurion/20250815_grid_2m_fullday`

## Purpose

This document consolidates what we learned while executing:

- `docs/plans/2026-03-14-webgpu-pointwise-utci-erf-dmrt-parity-plan.md`
- `docs/plans/2026-03-14-webgpu-rectangular-pointwise-parity-implementation-plan.md`

It captures:

1. What was implemented and validated.
2. What root causes were real versus false leads.
3. Which diagnostics were high-value.
4. Current parity status and remaining gap.
5. Practical rules for future migration/debug cycles.

## Executive Summary

We moved parity from broad aggregate checks to strict pointwise contracts and diagnostics, then discovered two distinct mismatch classes:

1. **Localized geometric/raycast edge effects** (small number of cells, high local impact).
2. **Systemic weather-hour alignment drift** (many cells, lower-to-medium spread impact).

Key breakthrough:

- Splitting weather packing semantics by channel in `compute-manager.ts`:
  - thermal channels follow Ladybug day-period semantics (`hour 0 -> previous day hour 24`, `hour 1..23 -> same-day 1..23`)
  - shortwave channels stay aligned to same-day sun-vector timeline (`hour h -> EPW h+1`)

Result:

- UTCI parity improved from broad hourly drift to almost complete agreement.
- Remaining strict failures are now highly localized and primarily shortwave/raycast boundary related.

## What Is Now Enforced

## 1) Artifact Contracts (strict)

- WebGPU collected artifacts are shape/length validated:
  - `_webgpu_solar.json`
  - `_webgpu_sky.json`
  - `_webgpu_mrt.json` (including optional `short_erf`, `long_erf`, `short_dmrt`, `long_dmrt`)
  - `_webgpu_utci.json` (`utciByHour` pointwise + positions)
- Reference intermediate loader validates optional ERF/DMRT arrays when present.
- Contract tests fail fast on malformed payloads.

## 2) Strict Compare Behavior

- UTCI strict gate is now **pointwise**, not range-only.
- Range metrics are preserved as diagnostics (`utci_range`) but are not the primary strict acceptance gate.
- ERF/DMRT policy is explicit:
  - present on both sides -> compare pointwise
  - missing on both -> skip
  - one-sided presence -> fail with clear reason

## 3) Deterministic Collection and Reporting

- Browser collection remains deterministic and phase-observable.
- Strict terminal output and JSON report are aligned.
- Worst-cell diagnostics include hour/point index and value deltas.

## Root Causes We Confirmed

## A) Real Root Cause: Weather-hour semantic mismatch (systemic)

Observed evidence:

- Python weather sample for baseline date starts with previous-day midnight-equivalent value (`24.2`), indicating Ladybug `AnalysisPeriod(..., 0..23)` boundary semantics.
- WebGPU debug weather sample showed same-day `hour 1` at slot 0 (`23.7`) before fix.
- UTCI deltas were widespread across many points and clustered by hours.

Fix:

- Updated weather packing in `viewer/src/lib/compute/compute-manager.ts`:
  - thermal fields use Ladybug day-boundary mapping
  - shortwave fields remain aligned with same-day sun timeline

Validation:

- TDD with `viewer/tests/compute/weather-index-alignment.test.ts`:
  - failing expectation first
  - minimal implementation
  - passing test after fix
- Fresh collect + strict/stats compare showed major UTCI reduction and collapse of broad spread.

## B) Real Root Cause: Solar binary edge flip(s) (localized)

Observed evidence:

- A tiny number of solar binary mismatches (`0/1` flips) can produce disproportionately large local `short_erf`, `short_dmrt`, `mrt`, and UTCI outliers.
- Shader ray/triangle epsilon tuning changed which cells flipped and affected strict outcomes.

Fix trajectory:

- Kept the epsilon change in `bvh_raycast.wgsl` (`1e-8 -> 1e-6`) after evidence showed fewer flips and better UTCI behavior.

## High-value Diagnostic Tools Added During This Work

- `viewer/scripts/diagnose-solar-flips.ts`
  - lists flip cells with coordinates and component context.
- `viewer/scripts/diagnose-mrt-worst-cell.ts`
  - decomposes worst MRT deltas by components.
- `viewer/tests/e2e/diagnose-solar-ray-oracle.spec.ts`
  - CPU-side ray oracle against browser model state.
- `viewer/src/lib/parity/mrtWorstCellDiagnostics.ts`
- `viewer/src/lib/parity/pointwiseIndex.ts`

These tools changed the workflow from "aggregate guessing" to "cell-level evidence".

## False Leads / Lessons From Reverted Experiments

The following experiments were useful but should not be treated as final fixes:

- Full weather remap to previous-day semantics for all channels:
  - improved thermal alignment but broke shortwave timing and worsened strict shortwave/MRT chain.
- Raw sun-vector mapping changes in `liveUtciAnalysis.ts`:
  - increased flip count and worsened parity.
- Ray origin bias and aggressive intersection threshold variants:
  - either neutral or regressive on parity.
- BVH stack size increase:
  - no measurable effect on mismatch count.

Lesson: preserve narrowly-scoped, evidence-backed adjustments; revert broad changes that move multiple subsystems at once.

## Current Status Snapshot (Latest Verified)

After latest weather-channel split fix and fresh artifact collection:

- Strict mode:
  - `utci`: PASS (pointwise)
  - `sky`: PASS
  - `solar`: FAIL (localized flip remains)
  - `mrt`: FAIL (localized max outlier)
  - `short_erf`: FAIL
  - `long_erf`: PASS
  - `short_dmrt`: FAIL
  - `long_dmrt`: PASS
- Stats mode:
  - `solar`: PASS
  - `sky`: PASS
  - `mrt`: PASS (tight p99)
  - `utci`: PASS
  - `utci_pointwise`: PASS

UTCI distribution after fix:

- max absolute error: ~`1.856`
- cells > `1.5` UTCI: `2`
- unique points > `1.5`: `1`
- unique hours > `1.5`: `2`

Interpretation:

- Systemic mismatch has been largely resolved.
- Remaining strict blockers are now narrow and localized.

## Practical Rules for Future Parity Work

1. **Always distinguish systemic vs localized errors early.**
   - Use distribution counts (`>0.5`, `>1.0`, `>1.5`) and unique points/hours.
2. **Treat weather and sun timelines as independent contracts.**
   - Thermal and shortwave may need different alignment semantics.
3. **Do not trust range-only UTCI success as parity completion.**
   - Keep pointwise UTCI as strict gate.
4. **Use component decomposition before shader edits.**
   - Confirm whether drift is shortwave-driven, longwave-driven, or both.
5. **Accept shader numeric tweaks only with non-regression checks.**
   - Must reduce flips/outliers and avoid new broad drift.
6. **Keep diagnostics deterministic and compatibility-checked.**
   - Validate report/base-path compatibility before using saved candidate indices.

## Suggested Next Debug Loop (Focused)

1. Keep current weather-channel split and current epsilon baseline.
2. Run:
   - `npm run parity:collect-webgpu`
   - strict compare + stats compare
   - solar flip diagnostics + MRT worst-cell diagnostics
3. Target only the remaining localized shortwave/raycast outlier(s):
   - evaluate tiny intersection threshold variants one at a time
   - accept only if:
     - flip count does not increase
     - UTCI pointwise remains pass
     - strict component regressions do not broaden

## Files Most Relevant To This Knowledge

- `viewer/src/lib/compute/compute-manager.ts`
- `viewer/tests/compute/weather-index-alignment.test.ts`
- `viewer/scripts/compare-parity.ts`
- `viewer/src/lib/parity/buildParityReport.ts`
- `viewer/src/lib/parity/compareUtciPointwise.ts`
- `viewer/scripts/diagnose-solar-flips.ts`
- `viewer/scripts/diagnose-mrt-worst-cell.ts`
- `viewer/tests/e2e/diagnose-solar-ray-oracle.spec.ts`

