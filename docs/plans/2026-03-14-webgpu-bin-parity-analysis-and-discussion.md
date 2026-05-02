# WebGPU vs .bin Parity: Final Findings

**Date:** 2026-03-14  
**Finalized:** 2026-05-03  
**Scope:** Ben-Gurion `20250815_grid_2m_fullday`, Python `.bin` baseline vs WebGPU UTCI debug pipeline.

## Executive Summary

The large daylight mismatches were caused by the TypeScript sunpath implementation, not by WebGPU numeric precision or the UTCI polynomial. The previous TS path used a simplified NOAA fractional-year approximation that differed from Ladybug/Python by about `0.38 deg` in sun direction. That is small geometrically, but enough to flip binary sun/shade classification at shadow edges and create isolated `~2 C` UTCI spikes.

The repeated hour-23 night offset was a separate boundary-averaging bug. In 12-month mode, WebGPU averaged August hour 23 with September hour 0. The shader now clamps UTCI averaging inside each representative-day block.

Current normal-mode app-visible August slice parity is very close:

| Metric | Result |
|---|---:|
| Points | `104,445` |
| Hours | `24` |
| Mean diff | `-0.040472 C` |
| Mean abs diff | `0.040475 C` |
| Max abs diff | `1.852249 C` |
| Cells over `0.5 C` | `2` |
| Cells over `2 C` | `0` |

The remaining two large cells are the same point at hours 16 and 17:

```json
[
  {
    "pointIndex": 31079,
    "hour": 16,
    "coords": { "x": -3342.616943359375, "y": -489.2862548828125, "z": 1.5 },
    "pythonUtci": 32.697750091552734,
    "webgpuUtci": 34.54999923706055,
    "diff": 1.8522491455078125
  },
  {
    "pointIndex": 31079,
    "hour": 17,
    "coords": { "x": -3342.616943359375, "y": -489.2862548828125, "z": 1.5 },
    "pythonUtci": 28.2528018951416,
    "webgpuUtci": 30.100000381469727,
    "diff": 1.847198486328125
  }
]
```

## Root Causes Found

### 1. Sunpath Drift

Python uses Ladybug's Julian-century NOAA sunpath calculation. The previous frontend implementation used a shorter fractional-year NOAA approximation. For Ben-Gurion Aug 15, the old TS vectors differed by roughly `0.38 deg` at multiple daylight hours.

This caused isolated solar exposure flips near shadow boundaries:

- Python shaded, WebGPU sunlit -> WebGPU UTCI too high by about `2 C`.
- Python sunlit, WebGPU shaded -> WebGPU UTCI too low by about `2 C`.

Fix: `viewer/src/lib/compute/sunpath.ts` now implements the Ladybug-compatible Julian-century equation chain, atmospheric refraction correction, azimuth branch, non-leap 2017 behavior, and correct ENU vector convention.

### 2. Hour-23 Boundary Averaging

The MRT/UTCI shader averaged each hour with `time_idx + 1`. In 12-month mode, that made a month/day boundary cross-talk:

```text
August hour 23 -> September hour 0
```

Fix: `mrt_utci.wgsl` now receives `num_hours_per_day` and clamps `next_idx` to the current representative-day block. Hour 23 duplicates itself for averaging, matching Python boundary behavior.

### 3. Weather Channel Semantics

Earlier parity work also confirmed an EPW/Ladybug time-series detail that should stay documented: thermal and shortwave weather channels are not the same contract.

- Thermal fields follow Ladybug day-period semantics:
  - hour 0 uses previous calendar day EPW hour 24.
  - hours 1..23 use the same representative day EPW hours 1..23.
- Shortwave fields follow the sun-vector timeline and EnergyPlus preceding-hour radiation convention:
  - hour `h` uses same-day EPW hour `h + 1`.

This split lives in `viewer/src/lib/compute/compute-manager.ts` and removed a broad weather-driven UTCI drift.

### 4. Test Path vs App Path

The old Playwright parity collector used:

```text
/debug-webgpu-utci?parity=1
```

That single-month fixture path did not match the user's visual debug path:

```text
/debug-webgpu-utci
```

Fix: `PARITY_COLLECT_MODE=normal` now loads `/debug-webgpu-utci?collect=normal` and exports the app-visible 12-month August slice to:

```text
data/analyses/Ben-Gurion/20250815_grid_2m_fullday_webgpu_normal_utci.json
```

This lets parity checks inspect the same data that is shown in the UI.

## Current Stage Results

From `viewer/parity-report-stats-latest.json`:

| Stage | Pass | Notes |
|---|---:|---|
| Solar | yes | One remaining binary flip: point `31079`, hour `17`. |
| Sky | yes | Normalized WebGPU sky exposure matches Python closely. |
| MRT | yes | Mean diff `0.0037 C`; worst cell is driven by the same solar flip. |
| UTCI | yes | Pointwise max error now comes from the same remaining solar-edge point. |

Component diagnostics are also populated and compared in the parity report:

| Component | Current status |
|---|---|
| `short_erf` | Pass; worst cell follows the same solar flip. |
| `long_erf` | Pass; no broad drift. |
| `short_dmrt` | Pass; worst cell follows the same solar flip. |
| `long_dmrt` | Pass; no broad drift. |

The MRT worst case is:

```json
{
  "pointIndex": 31079,
  "hourIndex": 17,
  "ref": 33.06828308105469,
  "webgpu": 50.31842803955078,
  "diff": 17.250144958496094
}
```

That propagates to UTCI as the remaining `~1.85 C` outlier.

## High-Value Diagnostics

The useful tools from the parity work are:

- `viewer/scripts/diagnose-solar-flips.ts`
  - Lists solar flip cells with point/hour indices and coordinates.
- `viewer/scripts/diagnose-mrt-worst-cell.ts`
  - Decomposes the worst MRT deltas by solar, sky, ERF, and DMRT terms.
- `viewer/tests/e2e/diagnose-solar-ray-oracle.spec.ts`
  - Runs a CPU ray oracle against browser model state.
- `viewer/src/lib/parity/mrtWorstCellDiagnostics.ts`
  - Shared MRT term attribution helpers.

These moved the workflow from aggregate guessing to point/hour-level evidence.

## False Leads and Lessons

Useful but non-final experiments:

- Full weather remap to previous-day semantics for all channels:
  - helped thermal alignment but broke shortwave timing.
- Raw sun-vector sign/mapping tweaks:
  - moved flips around but did not address the underlying sunpath drift.
- Broad BVH epsilon/ray-origin changes:
  - sometimes reduced one flip while creating more elsewhere.
- Range-only UTCI checks:
  - can pass while pointwise parity is still wrong.

Practical rules going forward:

1. Separate systemic drift from localized geometric edge effects early.
2. Compare stage-by-stage: solar -> sky -> MRT/DMRT/ERF -> UTCI.
3. Treat weather thermal and shortwave timelines as independent contracts.
4. Keep pointwise UTCI and component diagnostics available, even when aggregate stats pass.
5. For UI-visible parity questions, collect from the same app path the user sees, not only the special `?parity=1` fixture path.

## Verification Commands

Focused sunpath verification:

```powershell
cd viewer
npx vitest run tests/compute/sunpath.test.ts tests/compute/solar-altitude-packing.test.ts
```

Normal app-path WebGPU collection:

```powershell
cd viewer
$env:PARITY_COLLECT_MODE='normal'
npm run parity:collect-webgpu
```

Existing parity/stat report path:

```powershell
cd viewer
npm run parity:collect-webgpu
npm run parity:compare
npx tsx scripts/compare-parity.ts --mode stats --report parity-report-stats-latest.json
```

## Remaining Work

1. Investigate the single remaining solar-edge point at:
   - point `31079`
   - coords `(-3342.616943359375, -489.2862548828125, 1.5)`
   - hours `16` and `17`
2. Compare Python Embree vs WebGPU BVH ray behavior for that ray.
3. Decide whether to accept one binary shadow-edge flip, add a numerical tolerance policy, or align ray epsilon/origin/intersection behavior further.

## Conclusion

The main parity blockers are resolved. WebGPU UTCI now matches the Python baseline closely on the actual app-visible normal debug path. The remaining discrepancy is localized to one shadow-edge ray classification, not a broad UTCI, MRT, weather, or WebGPU precision issue.

## Superseded Notes

This document supersedes `docs/plans/2026-03-15-webgpu-python-parity-findings-and-lessons.md`. The durable findings from that note have been folded here so there is one canonical parity summary.
