# Main Route Product Shell Cleanup Results

Date: 2026-05-12

## Scope

This pass thinned the product main route without changing the debug route, compute folder layout, selected-hour route-host architecture, or repo-wide static debt.

The work stayed focused on `viewer/src/routes/+page.svelte` and product-route modules under `viewer/src/routes/main/`. The debug route remains temporary/proof infrastructure.

## Main Route Size

- Before: `viewer/src/routes/+page.svelte` was 866 lines.
- After: `viewer/src/routes/+page.svelte` is 588 lines.
- Debug route: `viewer/src/routes/debug/+page.svelte` stayed 4539 lines.

## Extracted Boundaries

- `viewer/src/routes/main/modelLifecycle.ts`
  - Pure model-load policy for scene bounds, first camera-fit position/target, and `hasFitOnce`.
  - Tests assert output clones and input immutability for mutable Three objects.

- `viewer/src/routes/main/MainRouteViewport.svelte`
  - Owns the `Scene`, `Camera`, `Lights`, `Model`, `GridHelper`, `UTCIPointCloud`, and `ComparisonRenderer` render tree.
  - Binds route-owned canvas/camera/surface/component handles back to `+page.svelte`.
  - Forwards renderer diagnostics, UTCI surface diagnostics, render contexts, surface identities, pending GPU outputs, and accepted-output release callbacks.

- `viewer/src/routes/main/MainRouteTooltipLayer.svelte`
  - Owns tooltip state, hover throttling, motion suppression, canvas listener lifecycle, and `MetricTooltip` rendering.
  - Keeps telemetry counters bindable to the route for diagnostics.
  - Uses event-time `getComparisonUtciMesh()` inside pointer handling, covered by source lock.

- `viewer/src/routes/main/MainRouteOverlays.svelte`
  - Optional extraction was used because the route was still 644 lines after viewport and tooltip extraction.
  - Owns only loading/error overlay markup and `ComparisonCurtain` rendering.
  - Overlay gating remains in `+page.svelte`.

## Svelte Reactivity Notes

Selected-hour route-host inputs remain explicit in `+page.svelte`: selected month/hour/time index, route-host `setRouteInputs(...)`, projected scene state, overlay gating, and diagnostics publication remain route-level reactive statements.

The extracted components receive explicit props/callbacks rather than hiding selected-hour dependencies inside no-argument helpers.

## Verification

- Baseline focused Vitest: PASS, 6 files / 26 tests.
- Baseline `npm run test:quality:selected-hour`: PASS, 18 files / 158 tests.
- Baseline `npm run test:e2e:selected-hour`: PASS, 13 Chromium tests.
- Baseline `npm run check`: FAIL, 163 errors / 4 warnings in 34 files; inherited static debt.
- Task 1 source locks: PASS, 3 files / 34 tests.
- Task 2 model lifecycle: red/green verified, final PASS, 1 file / 2 tests.
- Task 3 viewport focused Vitest: PASS, 5 files / 23 tests.
- Task 3 selected-hour E2E: PASS, 13 Chromium tests.
- Task 4 tooltip focused Vitest: PASS, 37 tests.
- Task 4 tooltip source lock: PASS, 1 test.
- Task 4 main-route diagnostics Playwright: PASS, 6 Chromium tests.
- Task 5 overlay/source-lock tests: PASS, 2 files / 20 tests.
- Final focused Vitest bundle: PASS, 10 files / 65 tests.
- Final tooltip-layer source lock: PASS, 1 file / 1 test.
- Final `npm run test:quality:selected-hour`: PASS, 18 files / 158 tests.
- Final `npm run test:e2e:selected-hour`: PASS, 13 Chromium tests.
- Final `npm run build`: PASS with existing warnings in `Model.svelte`, `UTCIPointCloud.svelte`, and chunk-size/import warnings.
- Final `npm run check`: FAIL, 163 errors / 4 warnings in 34 files.
- Final `git diff --check`: PASS; Git printed CRLF conversion warnings only.

During final verification, one `ENOSPC` failure occurred because the C: drive had 0 bytes free and Vitest/transform cache writes use user temp space there. Root cause was environmental; old generated temp folders and Playwright browser profiles under `%TEMP%` were removed after path verification, freeing about 1.3 GB. The failed test command then passed.

## Static Debt

The transient touched-file check errors introduced by `MainRouteOverlays.svelte` were fixed:

- `+page.svelte` passed `string | null` route-host errors into overlay props typed as `string | undefined`.
- `MainRouteOverlays.svelte` now accepts `string | null | undefined` for live-route error props.

Final touched-file diagnostics: none found in `+page.svelte`, `viewer/src/routes/main/*`, or new route tests.

Remaining inherited debt includes:

- ArrayBufferLike / ArrayBuffer transfer typing.
- parity reference union narrowing.
- Three.js `Object3D` narrowing.
- debug-route parity intermediate window typing.
- `SunPath.svelte` store drift.
- `Model.svelte` and `UTCIPointCloud.svelte` existing Svelte/type warnings.
- older point-cloud/data-loader test fixture drift.

## Review Agents

- Main route Svelte/readability review: no blockers. Confirmed the route is visibly thinner and selected-hour reactivity remains explicit. Noted non-blocking concerns that `MainRouteViewport` and `MainRouteTooltipLayer` have broad prop surfaces and still feel like route-glue components.
- Selected-hour behavior review: no blockers. Confirmed `strongVisibleGpuPath` remains conservative and source locks/debug boundaries hold. Noted non-blocking resource-honesty concerns: accepted GPU-resident output is marked releasable on copy-complete but not destroyed until later replacement/reset/dispose, so VRAM-retention claims should remain conservative.
- Static-debt/scope review: no blockers. Confirmed no debug beautification, no compute-folder reorganization, and no repo-wide check cleanup. Confirmed final `npm run check` failures are inherited, with no touched-file diagnostics.

## Remaining Work

- Broad inherited `npm run check` debt needs a separate plan.
- Debug route temporary/proof infrastructure can be decomposed later only with separate parity/source-lock proof.
- `MainRouteViewport` and `MainRouteTooltipLayer` could get cleaner contracts in a later modularization pass, but current broad props preserve explicit dependencies and avoid hidden reactivity.
- GPU-resident memory/VRAM release timing should remain a separate resource-lifecycle follow-up; do not use this cleanup as proof of prompt buffer destruction after visible copy.
