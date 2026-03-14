# WebGPU vs .bin Parity: Root-Cause Analysis and Discussion

**Date:** 2026-03-14  
**Goal:** Analyze why WebGPU and Python .bin results differ, how to test parity properly, and how to align WebGPU with .bin—**before making any code changes**.  
**Skills used:** Brainstorming (explore context, gather evidence, propose approaches), Systematic debugging (root-cause first, no fixes without understanding).

---

## 1. Evidence Gathered

### 1.1 Intermediate data you have

| File | Purpose |
|------|--------|
| `20250815_grid_2m_fullday_solar.json` | Python solar exposure (0/1 per point×hour), point-major flat |
| `20250815_grid_2m_fullday_sky.json` | Python sky exposure (0–1 fraction per point) |
| `20250815_grid_2m_fullday_mrt.json` | Python MRT (°C per point×hour) |
| `20250815_grid_2m_fullday_weather_sample.json` | First 3 hours weather (air_temp, direct_normal, etc.) |
| `20250815_grid_2m_fullday_webgpu_inspect.json` | WebGPU readback: solar/sky/MRT stats and samples |

### 1.2 Grid and scale mismatches (evidence)

- **Point counts**
  - Python / .bin: **104,445** (rectangular grid from analysis bounds, stored in .bin).
  - WebGPU default (mesh): **105,157** (from `generateGridFromMesh`; different positions and count).
  - So **default WebGPU and .bin are on different grids**—point-to-point comparison is invalid unless you use the same grid (e.g. `?rectangularGrid=1`).

- **Sky exposure scale (root cause)**
  - **Python export:** `sky_exposure` is a **0–1 fraction** (unoccluded sky view).
  - **WebGPU readback:** The exposure shader writes the **raw Tregenza weight sum** (unoccluded weights), which has max ~**145.25** (`total_tregenza_weight` in `mrt_utci.wgsl`). The MRT shader then does `sky_exposure[i] / 145.2488` to get 0–1 for UTCI.
  - **Inspect file:** `skyExposure` values are in the 5–145 range (e.g. 114.15, 144.99), i.e. **unnormalized**.
  - **Parity test:** `compareIntermediatesStats` uses `TOLERANCE_MEAN = 0.02`, `TOLERANCE_MAX = 0.05` assuming **0–1**. Comparing WebGPU raw sum to Python 0–1 will fail (mean/max diff huge). So **sky parity failure is largely a scale/definition mismatch**, not only grid.

- **Solar:** 0/1 per (point, hour); scale is fine. Differences can still come from grid, sun vectors, BVH/raycast, or ray offset (e.g. WebGPU `origin + sun * 0.1`).

- **Weather sample:** Small float differences (e.g. 24.2 vs 23.7) are expected (EPW parsing, rounding). First-hour alignment is what matters for MRT/UTCI.

### 1.3 Exposure / e2e test “timeout”

- **Test:** `viewer/tests/e2e/parity-intermediates.spec.ts` waits for `__parityIntermediates__` (or `__parityIntermediatesError__`) with **INTERMEDIATES_WAIT_MS = 15_000** (15 s) and **test.setTimeout(INTERMEDIATES_WAIT_MS + 10_000)** (25 s).
- **Pipeline for Ben-Gurion:** Load model → worker (merge + BVH + mesh grid) → upload ~105k points, 24 hours, BVH, dome, weather → exposure passes (solar + sky) → MRT/UTCI pass → readback. On a typical machine this can exceed 15 s, so the test can **time out** before the page ever sets `__parityIntermediates__`.
- So the “exposure test failing due to timeout” is consistent with **wait too short** for the full run, not necessarily a bug in exposure itself.

### 1.4 Rectangular grid page: “WebGPU UTCI not showing / processed”

- **Flow:** With `?rectangularGrid=1` and metadata `bounds`, the debug page still runs the worker (for BVH), then passes `useRectangularGridFromBounds: true` and analysis bounds so `ComputeManager` builds the grid from `analysisBoundsToRectangularGrid` (same bounds as .bin) → **104,445 points**.
- **Possible reasons UTCI “doesn’t show”:**
  1. **Runtime:** 104k points × 24 hours is heavy; compute can take 30–60+ seconds—user may leave or think it hung.
  2. **Error path:** Any throw (e.g. WebGPU device lost, buffer size, missing EPW) sets `liveError` and no `liveAnalysis` → no visualization.
  3. **Display path:** Same code path as mesh grid; if `liveAnalysis` is set, the same point cloud/overlay should show. So if “nothing shows,” either compute never completes or an error is shown.
- **Recommendation:** Add a visible “Computing…” state and progress (or at least “Running with 104,445 points…”) when using rectangular grid, and ensure errors surface (e.g. `liveError` in UI). Check browser console for errors when loading `?rectangularGrid=1&analysis=Ben-Gurion/20250815_grid_2m_fullday`.

---

## 2. Root Causes (Summary)

| Cause | Impact | Fix direction (later) |
|-------|--------|------------------------|
| **Sky exposure scale** | WebGPU readback is weight sum (~0–145); Python/reference is 0–1. Stats comparison assumes 0–1. | Normalize WebGPU sky by `total_tregenza_weight` when exposing for parity, or compare normalized stats. |
| **Grid difference** | Default WebGPU uses mesh grid (105,157 pts); .bin uses rectangular (104,445). Different points → no point-wise match. | Use `?rectangularGrid=1` for same-grid runs; parity tests can use statistical comparison or same-grid + point-wise. |
| **E2E timeout** | 15 s wait often too short for full Ben-Gurion pipeline. | Increase INTERMEDIATES_WAIT_MS (e.g. 60–90 s) or make it configurable; optionally split “quick smoke” vs “full parity” test. |
| **Sun / BVH / ray details** | Small differences in sun vectors, ray origin offset, or BVH vs CPU intersector can change exposure and thus MRT/UTCI. | After fixing scale and grid, compare intermediates stage-by-stage (solar → sky → MRT → UTCI); align sun and ray convention if needed. |

---

## 3. How to Test Result Parity (Approaches)

**A. Stage-by-stage with current setup (recommended first)**  
- Keep **statistical** comparison (mean/max) for solar and sky, but **normalize WebGPU sky** to 0–1 before comparison (divide by same constant used in MRT shader).  
- Compare MRT with existing tolerance (e.g. mean 1 °C, max 2 °C).  
- Keep point-to-point UTCI out of scope until intermediates match.

**B. Same-grid (rectangular) for point-wise**  
- Use `?rectangularGrid=1` so WebGPU uses the same 104,445 points as .bin (in viewer coords).  
- Then you can optionally add point-wise UTCI comparison (or per-hour RMSE) with a tolerance (e.g. 1–2 °C).  
- Requires rectangular path to complete and expose results (fix timeout/UX so it’s clear when it’s still computing).

**C. Parity test robustness**  
- Increase wait for `__parityIntermediates__` (e.g. 60–90 s) for the full Ben-Gurion run.  
- Optionally: small “smoke” test (e.g. tiny analysis or mock) with short timeout; full parity test with long timeout.  
- Document that reference files must be regenerated when Python pipeline or model changes.

**D. Diagnostic harness**  
- Keep inspect test and `*_webgpu_inspect.json` output; add optional script to compare one run’s WebGPU vs Python intermediates (with normalization for sky) and print mean/max/rmse per stage.  
- Helps iterate on formula/BVH/sun alignment without running Playwright every time.

---

## 4. Who Could Help and What They’d Say

- **Thermal comfort / UTCI domain expert (e.g. Ladybug or academic)**  
  - “MRT and UTCI formulas must match the standard (e.g. ISO/ASHRAE); small differences in exposure or weather propagate. Align input definitions (sky view factor 0–1, same sun convention) first, then compare outputs.”

- **WebGPU / graphics engineer**  
  - “Ensure readback happens after `onSubmittedWorkDone` and that you’re reading the buffer the shader wrote. For parity, normalize sky to 0–1 on the CPU after readback so your comparison is like-for-like. Check for precision (f32) and any packed format.”

- **QA / test engineer**  
  - “Separate fast smoke test (short timeout) from full parity test (long timeout). Use conditional skip when reference files are missing. Log stage-wise pass/fail (solar, sky, MRT) so we know which stage regressed.”

- **Product / project owner**  
  - “We need a clear definition of ‘parity achieved’: e.g. same grid, intermediates within tolerance, UTCI RMSE &lt; 2 °C. Prioritize fixing the sky scale and timeout so the current test can pass; then tighten tolerances or add point-wise checks.”

---

## 5. Recommended Order of Work (When You Implement)

1. **Align sky scale:** Normalize WebGPU sky readback to 0–1 (divide by `total_tregenza_weight`) when populating `__parityIntermediates__` (and in any export used for parity). No change to MRT shader; only to what is exposed for comparison.
2. **E2E timeout:** Increase INTERMEDIATES_WAIT_MS (and test timeout) so the full Ben-Gurion run can complete in CI/local; optionally add a “slow” tag and document.
3. **Rectangular grid UX:** Make long-running rectangular runs visible (progress or message) and ensure errors are shown; verify in browser that `?rectangularGrid=1` completes and displays UTCI.
4. **Re-run parity test:** With normalized sky and longer wait, confirm solar/sky/MRT statistical tests pass (or document remaining gaps).
5. **Optional:** Same-grid point-wise UTCI comparison and/or diagnostic script as above.

No code changes were made in this session; this document is for analysis and discussion only.
