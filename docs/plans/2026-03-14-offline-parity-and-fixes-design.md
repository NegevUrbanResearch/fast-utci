# Offline Parity Workflow and Required Fixes — Design

**Date:** 2026-03-14  
**Status:** Design (no implementation until approved and implementation plan is written.)  
**Related:** `2026-03-14-webgpu-bin-parity-analysis-and-discussion.md` (root-cause analysis).

---

## 1. Goal

- **Parity workflow:** Run parity in two phases—**collect** (Python and WebGPU each write results to files) and **compare** (Node script reads files and asserts). No comparison inside the browser; tests stay fast and deterministic.
- **Reliability:** WebGPU results are **extracted to files** in an automatic, repeatable way (not “grab from console and hope it’s there”).
- **Fixes:** Address known causes of mismatch and broken behaviour (sky scale, UTCI stats, rectangular grid UTCI not showing) so that the new workflow can succeed.

---

## 2. Three Separate Commands

Three independent commands, run in any order that respects dependencies (Python collect and WebGPU collect can run in parallel; Compare needs both sets of files).

| Command | Responsibility | Output |
|--------|-----------------|--------|
| **Python collect** | Run the existing export script for a given analysis (e.g. Ben-Gurion base). | Reference files: `{base}_solar.json`, `{base}_sky.json`, `{base}_mrt.json`, `{base}_weather_sample.json`, plus existing `{base}.bin` and `{base}.json` (metadata with `utci_range`). |
| **WebGPU collect** | Load the debug page, run compute, read results from the page, write WebGPU outputs to disk. | One file per stage: `{base}_webgpu_solar.json`, `{base}_webgpu_sky.json`, `{base}_webgpu_mrt.json`, `{base}_webgpu_utci.json`. |
| **Compare** | Load Python ref files and WebGPU collected files from disk; run all comparisons; exit with pass/fail. | No artifact; exit code and logs (and optionally a short report file). |

- **Fully automatic:** WebGPU collect is driven by a script (e.g. Playwright): open page → wait for compute and for results on `window` → Node reads via `page.evaluate()` → Node writes files with `fs`. No manual “download” or console copy.
- **Collect and compare are separate:** The test suite does not do “wait in browser then compare in same run.” Compare runs purely in Node against files written by the two collect steps.

---

## 3. File Naming and Contents (One File per Stage)

**Convention:** For an analysis base path (e.g. `data/analyses/Ben-Gurion/20250815_grid_2m_fullday`), all files use that base with a suffix. WebGPU outputs mirror Python stage names with a `_webgpu_` infix.

### 3.1 Python reference (existing)

| File | Contents |
|------|----------|
| `{base}_solar.json` | `{ numPositions, numHours, solarExposure: number[] }` — point-major flat, 0/1 per (point, hour). |
| `{base}_sky.json` | `{ numPositions, skyExposure: number[] }` — 0–1 fraction per point. |
| `{base}_mrt.json` | `{ numPositions, numHours, mrt: number[] }` — point-major flat, °C. |
| `{base}.json` | Metadata including `utci_range: { min, max, mean, std }`. |
| `{base}.bin` | Binary UTCI (and positions); used by Compare only for reference metadata / optional point-wise later. |

### 3.2 WebGPU collected (new)

| File | Contents |
|------|----------|
| `{base}_webgpu_solar.json` | Same shape as Python solar: `{ numPositions, numHours, solarExposure: number[] }`. |
| `{base}_webgpu_sky.json` | **Normalized 0–1** (see §5.1): `{ numPositions, skyExposure: number[] }`. |
| `{base}_webgpu_mrt.json` | Same shape as Python MRT: `{ numPositions, numHours, mrt: number[] }`. |
| `{base}_webgpu_utci.json` | `{ numPoints, numHours, utciByHour: number[][], utci_range: { min, max, mean } }` — computed from WebGPU readback so Compare can check against Python `{base}.json` `utci_range`. |

All written under a **fixed path** derived from the analysis base (e.g. CLI or config specifies the base path; WebGPU collect writes into that directory). Optional: env var or CLI arg to override output directory for CI.

---

## 4. WebGPU Collect (Automatic)

- **Runner:** A dedicated script or Playwright test that:
  1. Starts the viewer (or assumes it’s already running), opens the debug page with the chosen analysis (e.g. `?analysis=Ben-Gurion/20250815_grid_2m_fullday`). The debug page always uses a bounds-based grid (same as .bin); no query param is needed.
  2. Waits for `window.__parityResults__` and `window.__parityIntermediates__` (or a single “parity ready” flag) with a **generous timeout** (e.g. 90–120 s) so the full pipeline can finish; no assertion logic here, only “data present.”
  3. Reads the full payload from the page (e.g. `page.evaluate(() => ({ parityResults: window.__parityResults__, parityIntermediates: window.__parityIntermediates__ }))`).
  4. In **Node**, normalizes sky to 0–1 (see §5.1), then writes one JSON file per stage to the chosen base path (`_webgpu_solar.json`, `_webgpu_sky.json`, `_webgpu_mrt.json`, `_webgpu_utci.json`).

- **Extract, don’t hope:** The only way WebGPU data gets to disk is this script writing it after a successful wait. No reliance on manual console copy or “hope it’s there.” If the page never sets the globals (error or hang), the script times out and can write nothing or a small error manifest; Compare can then report “WebGPU collect failed / missing files.”

- **Single analysis per run:** One run of WebGPU collect produces files for one base path. To support multiple analyses, run the script multiple times or extend it later with a list of bases.

---

## 5. Compare (Offline, Node Only)

- **Input:** A base path (e.g. `data/analyses/Ben-Gurion/20250815_grid_2m_fullday`). Compare expects:
  - Python refs: `{base}_solar.json`, `{base}_sky.json`, `{base}_mrt.json`, `{base}.json` (for `utci_range`).
  - WebGPU: `{base}_webgpu_solar.json`, `{base}_webgpu_sky.json`, `{base}_webgpu_mrt.json`, `{base}_webgpu_utci.json`.

- **Behaviour:** For each stage, load the ref and WebGPU file (if both exist), run the comparison, then:
  - **Solar / sky / MRT:** Statistical comparison (mean, max) with configurable tolerances; sky is already 0–1 in both ref and WebGPU files.
  - **UTCI:** Compare WebGPU `utci_range` (min, max, mean) from `_webgpu_utci.json` to Python `utci_range` from `{base}.json` with tolerances (e.g. min/max within 2 °C, mean within 1 °C). No point-wise comparison in this design; that can be added later when using same-grid (rectangular) and same point count.

- **Output:** Pass/fail per stage; non-zero exit code if any stage fails or required files are missing. Optional: short report file (e.g. `parity-report.json` or `.md`) with metrics and tolerances.

- **No browser:** Compare only uses Node and `fs`; it does not start Playwright or load the viewer.

---

## 6. Required Fixes (From Previous Discussion)

These changes are needed so that the offline workflow and comparisons are valid and so that rectangular-grid runs are usable.

### 6.1 Sky exposure scale (WebGPU → 0–1)

- **Problem:** WebGPU exposure shader writes the **raw Tregenza weight sum** (~0–145); Python reference is **0–1**. Comparing them directly makes sky parity fail.
- **Fix:** When preparing data for parity (both for `window.__parityIntermediates__` and when writing `_webgpu_sky.json`), normalize WebGPU sky by the same constant used in the MRT shader (`total_tregenza_weight` = 145.2488). No change to the shader; only to the value exposed and persisted. Compare then sees 0–1 on both sides.

### 6.2 UTCI min/max/mean comparison

- **Requirement:** Compare must assert WebGPU UTCI statistics against the Python reference. Python already has `utci_range: { min, max, mean, std }` in `{base}.json`.
- **Fix:** WebGPU collect writes `_webgpu_utci.json` with a computed `utci_range: { min, max, mean }` (and optionally `std`) from the read-back `utciByHour`. Compare loads `{base}.json` and `_webgpu_utci.json` and asserts that WebGPU min/max/mean lie within configured tolerances of the ref. No dependency on the .bin for this check; only metadata.

### 6.3 Rectangular grid: UTCI layer never shows

- **Problem:** With `?rectangularGrid=1`, the UTCI layer for the live (WebGPU) run never appears. This is not “user didn’t wait long enough”—the layer simply does not show.
- **Fix (design):**
  - **Diagnose:** Confirm whether `liveAnalysis` is set and whether the overlay/visibility logic differs for the rectangular path; check for thrown errors (e.g. in init, upload, or readback) that set `liveError` and prevent `liveAnalysis` from being set; verify that the same visualization path (e.g. `UTCIPointCloud` + `createUtciSurfaceMesh`) is used for both mesh and rectangular grid once `liveAnalysis` exists.
  - **Remediate:** Fix the root cause (e.g. ensure rectangular path completes and sets `liveAnalysis`; surface any error in `liveError` and in UI; ensure layout/metadata for the live analysis are valid so the overlay can render). Optionally add a clear “Computing…” or “Running with N points…” when using rectangular grid so users know the run is in progress.
  - **No assumption that rectangular is “slower”:** Mesh and rectangular grids have similar point counts (~105k vs ~104k); if rectangular ever appears much slower or never completes, that is a bug to fix, not a UX workaround.

### 6.4 No long timeout as the solution for “slow” parity

- **Decision:** We do **not** fix parity by greatly increasing a single in-browser test timeout. Instead, the collect step only waits long enough for the pipeline to finish and for results to be available on `window`, then writes files and exits. The Compare step runs offline and is fast. So “slow” is confined to the collect run, and the test that matters (Compare) does not depend on browser timing.

### 6.5 Optional: same-grid (rectangular) collect and point-wise UTCI

- **Later:** Once rectangular grid shows UTCI reliably, WebGPU collect can support `?rectangularGrid=1` to produce same-grid outputs. Compare could then optionally do point-wise or per-hour RMSE for UTCI (same point count and order). Out of scope for this design; only the structural support (one file per stage, UTCI stats comparison) is in scope.

---

## 7. Error Handling and Robustness

- **WebGPU collect:** If the page sets `__parityIntermediatesError__` or an equivalent, the script should treat that as failure and not write (or write an error stub); Compare can then report “WebGPU collect failed.”
- **Missing files:** Compare should skip stages whose ref or WebGPU file is missing and report clearly (e.g. “solar: skipped (no ref)” or “sky: failed (no webgpu file)”); overall pass only if every required stage is present and within tolerance.
- **Configurable base path:** Compare (and optionally WebGPU collect) accepts the analysis base path as an argument or env var so CI and local runs can target the same layout without hardcoding.

---

## 8. How to run

- **Python collect:** From repo root, run:
  `python scripts/export_ben_gurion_intermediates.py --base-path data/analyses/Ben-Gurion/20250815_grid_2m_fullday --model data/3d_models/Ben-Gurion/original_with_layers.glb --stage solar --stage sky --stage mrt`
  (optionally add `--stage weather`). Produces `_solar.json`, `_sky.json`, `_mrt.json`, etc.

- **WebGPU collect:** From repo root, run:
  `cd viewer && npx playwright test tests/e2e/collect-webgpu-parity.spec.ts`
  Optionally set `PARITY_BASE_PATH=data/analyses/Ben-Gurion/20250815_grid_2m_fullday` (relative to repo root). Produces `_webgpu_solar.json`, `_webgpu_sky.json`, `_webgpu_mrt.json`, `_webgpu_utci.json`.

- **Compare:** From repo root, run:
  `cd viewer && npx tsx scripts/compare-parity.ts --base-path data/analyses/Ben-Gurion/20250815_grid_2m_fullday`
  (when run from `viewer`, base path is relative to repo root). Exits 0 if all stages pass, 1 otherwise.
  To get a detailed report for debugging (per-stage diff stats, percentiles, worst indices), add:
  `--report parity-report.json`; the script writes a JSON file with `.detail.diffStats` and `.detail.worstIndices` per stage.

---

## 9. Summary

| Item | Decision |
|------|----------|
| Commands | Python collect, WebGPU collect, Compare — three separate, automatic commands. |
| WebGPU output | One file per stage: `_webgpu_solar.json`, `_webgpu_sky.json`, `_webgpu_mrt.json`, `_webgpu_utci.json`. |
| WebGPU collect | Playwright (or equivalent) loads page, waits for results on `window`, Node writes files; no manual download. |
| Compare | Node-only; reads ref + WebGPU files; asserts solar/sky/MRT stats and UTCI min/max/mean. |
| Sky scale | Normalize WebGPU sky to 0–1 when exposing and when writing `_webgpu_sky.json`. |
| UTCI | Compare WebGPU `utci_range` (from `_webgpu_utci.json`) to Python `utci_range` (from `{base}.json`). |
| Rectangular grid | Diagnose and fix so UTCI layer shows when `?rectangularGrid=1`; do not rely on “wait longer” or “slower run” as the fix. |
| Timeout | Avoid “one big browser test with long timeout”; collect writes files, compare runs offline. |

---

## 10. Next Step

After approval of this design, the next step is to produce an **implementation plan** (e.g. via the writing-plans skill) that breaks down: (1) WebGPU collect script and file writing, (2) Compare script and UTCI/min-max-mean logic, (3) sky normalization in the viewer and in collect output, (4) rectangular-grid diagnosis and fix, and (5) any updates to existing tests or docs.
