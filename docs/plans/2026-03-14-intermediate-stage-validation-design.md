# Intermediate-Stage Validation Design

**Goal:** Validate WebGPU pipeline intermediate stages (solar exposure, sky exposure; MRT later) against reference data so we can isolate where discrepancies come from. No point-to-point UTCI assertion; e2e remains a smoke test.

**Scope:** Ben-Gurion base case only: `data/analyses/Ben-Gurion/20250815_grid_2m_fullday` (.bin + .json) and `data/3d_models/Ben-Gurion/original_with_layers.glb`.

---

## 1. Reference data

- **Source:** Captured once from the Python pipeline (exposure code); not regenerated in CI.
- **One file per stage** (smaller files, step-by-step validation):
  - **Solar:** `data/analyses/Ben-Gurion/20250815_grid_2m_fullday_solar.json`  
    `{ "numPositions", "numHours", "solarExposure": number[] }` — point-major flat: `[p0_h0, p0_h1, …, pN_h23]`.
  - **Sky:** `data/analyses/Ben-Gurion/20250815_grid_2m_fullday_sky.json`  
    `{ "numPositions", "skyExposure": number[] }`.
  - **MRT (later):** e.g. `*_mrt.json` when added.
  - **Weather sample:** `*_weather_sample.json` (first few hours: air_temp, direct_normal, diffuse_horizontal, horiz_infrared, wind_speed, rel_humidity). Compare `window.__parityDebug__.weatherSample` with this file for alignment (manual or one-off script).
- **Producer:** Python script (e.g. `scripts/export_ben_gurion_intermediates.py`) that runs exposure for this analysis + model and writes `*_solar.json` and/or `*_sky.json` (e.g. `--stage solar`, `--stage sky`, or both). Run manually when pipeline or model changes.

---

## 2. WebGPU readback and exposure

- **Pipeline:** Add readback for solar and sky in the WebGPU pipeline (same pattern as `readUtcisSlice`): staging buffer + `mapAsync`. E.g. `readSolarExposureSlice(monthIndex, hourIndex)` and/or full solar buffer; `readSkyExposure()` for full sky.
- **Debug viewer:** After every successful compute, expose intermediates (`window.__parityIntermediates__`) with `solarExposure` and `skyExposure` as JSON-serializable arrays so Playwright can read them. No parity mode; grid sizes may differ from reference.
- **Stages:** Implement and expose solar and sky first; MRT later.

---

## 3. Comparison and assertions

- **Comparison helper:** In `viewer/src/lib/parity/`, add a helper (e.g. `compareIntermediates.ts`) that takes reference and WebGPU arrays (solar or sky), optional tolerance (e.g. 1e-5 for 0–1), returns `{ pass, rmse, maxError, numPoints }`. Reuse “length must match” and metric pattern from `compareParity`.
- **Reference loader:** Add `loadReferenceIntermediatesFromFs(basePath, stage: 'solar' | 'sky')` (Node-only) that loads `basePath + '_solar.json'` or `basePath + '_sky.json'` and returns typed arrays.
- **Where to assert:** One check per stage. Option A: Playwright test that loads reference in Node, gets `__parityIntermediates__` from the page, compares, fails if out of tolerance. Option B: Node script that runs Playwright (or receives WebGPU results from a prior run), then compares. Prefer one stage at a time (e.g. solar test, sky test); fail or log which stage failed.

---

## 4. Docs and plan updates

- Update `docs/plans/2026-03-14-webgpu-bin-parity-validation-harness.md` (or a short linked doc): intermediate validation in scope; reference = one file per stage (`*_solar.json`, `*_sky.json`); only Ben-Gurion base; how to run Python export and how to run the parity check; point-to-point UTCI remains out of scope.

---

## 5. How to run

- **Generate reference (once):** From repo root, run  
  `python scripts/export_ben_gurion_intermediates.py --base-path data/analyses/Ben-Gurion/20250815_grid_2m_fullday --stage solar --stage sky --model data/3d_models/Ben-Gurion/original_with_layers.glb`  
  to create `*_solar.json` and `*_sky.json`. Re-run when the Python pipeline or model changes.
- **Run parity checks:** From `viewer/`, run  
  `npx playwright test tests/e2e/parity-intermediates.spec.ts`  
  (or the path chosen in the implementation plan). Tests skip if reference files are missing; they assert when reference and WebGPU results are compared.

---

## 6. Debugging zero intermediates

If WebGPU readback returns all zeros for solar/sky exposure:

1. **Validation:** The debug page sets `window.__parityIntermediatesError__` when both solar and sky are all zeros, so the e2e test fails with a clear message instead of a statistical mismatch.
2. **Diagnostics:** Run the inspect test (no assertions):  
   `cd viewer && npx playwright test tests/e2e/inspect-intermediates.spec.ts`  
   It prints: `numPoints`/`numHours`, solar/sky stats (mean, min, max, non-zero count), sun vector samples (hours 0, 12, 23), and sampled indices (0%, 25%, 50%, 75%, 99%). Use this to confirm whether sun vectors are daytime (y > 0 in Y-up) and whether any cells are non-zero.
3. **Pipeline checks:**
   - Exposure buffers must include `GPUBufferUsage.COPY_SRC` so copy-to-staging succeeds (otherwise the copy is invalid and staging may stay zero).
   - `queue.onSubmittedWorkDone()` is called before readback so compute has finished.
   - Solar/sky passes run only when BVH is present; `ranExposurePassesThisRun` guards readback.
4. **Shader probe (optional):** In `exposure_solar.wgsl`, set `PROBE_FORCE_WRITE = true` temporarily. Then run the inspect test: if `solarExposure[0] === 0.5`, the compute→readback path is correct and the issue is in shader logic (e.g. sun vectors or BVH). If still 0, the buffer being read is not the one the shader writes to.

---

## 7. MRT, weather sample, and rectangular grid

- **MRT and weather sample** can be exported/compared for diagnosis (e.g. `*_mrt.json`, `*_weather_sample.json`; `__parityIntermediates__.mrt`, `__parityDebug__.weatherSample`). Rectangular grid option (`?rectangularGrid=1`) allows same-grid runs for grid-vs-formula separation and distinct "inside building" behaviour when the building layer is hidden.

## 8. Out of scope (for now)

- Other analyses beyond Ben-Gurion base.
- Point-to-point UTCI assertion in e2e.
