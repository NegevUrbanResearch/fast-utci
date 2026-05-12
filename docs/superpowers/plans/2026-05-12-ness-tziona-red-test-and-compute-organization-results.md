# Ness Tziona Red Test And Compute Organization Results

Date: 2026-05-12

## Scope

This pass checked the Ness Tziona main-route selected-hour E2E before reorganizing the selected-hour compute cluster, then executed the authorized GPU/shader, core-math, on-demand, weather, and root-leftover audit follow-up slices. No commits or worktrees were created. `compute-manager.ts` and `telemetry.ts` remain intentional compute-root files.

## Root Cause

The previously reported Ness Tziona failure did not reproduce in the current run. The focused Playwright case `uses live WebGPU UTCI range independent of .bin metadata for Ness Tziona` passed twice in a row, then the full selected-hour E2E suite passed 13/13. Per the plan branch for this case, the earlier failure is documented as likely environmental or order-dependent rather than fixed here. No speculative route/host/session/controller/projection behavior fix was made before the move.

### Ness Tziona Root-Cause Hypothesis

I think the root cause is: no current root-cause boundary can be named from this run because the focused browser regression and the full selected-hour E2E suite both passed before code edits.

Evidence:
- Focused Ness Tziona Playwright run 1: PASS, 1 Chromium test, 23.5s.
- Focused Ness Tziona Playwright run 2: PASS, 1 Chromium test, 20.3s.
- Full selected-hour E2E before reorg: PASS, 13 Chromium tests, 1.1m.

The smallest test that should fail before a behavior fix is: the existing focused Playwright case in `viewer/tests/e2e/main-route-manual-diagnostics.spec.ts`. It did not fail in this implementation pass.

## Behavior Fix

No Ness Tziona route behavior fix was made because the behavior gate was already green. During the selected-hour mechanical gate, `viewer/tests/compute/live-utci-analysis.test.ts` exposed a selected-hour adapter regression in the per-slice readback fallback: fallback slices were aggregated without length validation and not stored in the point-major layout consumed later. The minimal fix validates bulk and per-slice readback lengths, stores fallback slices point-major, and preserves the existing `readUtciBulk()` and `readUtcisSlice()` paths. The main route still does not read `.bin`, Python reference output, or debug parity state.

## Compute Organization

- Selected-hour orchestration: moved to `viewer/src/lib/compute/selected-hour/`; imports and source-lock path reads were updated.
- GPU pipeline and shaders: moved to `viewer/src/lib/compute/gpu/`; imports, exact `vi.mock(...)` specifiers, shader raw imports, hard-coded source reads, and `docs/webgpu_strategy_analysis.md` path references were updated.
- Core math: moved to `viewer/src/lib/compute/core/`; imports and the `mrt_utci.wgsl` source comment were updated. Ladybug/Python parity-sensitive formulas were preserved exactly. The colocated `analysisGridFromBounds.test.ts` moved with the core module.
- On-demand diagnostics/state: moved to `viewer/src/lib/compute/on-demand/`; imports and exact test specifiers were updated. Diagnostics field names and proof-boundary semantics remain unchanged.
- Weather: moved to `viewer/src/lib/compute/weather/`; imports were updated. Ben-Gurion and Ness Tziona weather mapping behavior remains unchanged.
- Root leftovers: `compute-manager.ts` and `telemetry.ts` remain at `viewer/src/lib/compute/`. `telemetry.ts` is shared by debug route, GPU worker client, and selected-hour internals, so no one-file diagnostics folder was created.

## Verification

- Focused Ness Tziona Playwright before reorg: PASS twice, 1 Chromium test each, 23.5s then 20.3s.
- Full selected-hour E2E before reorg: PASS, 13 Chromium tests, 1.1m.
- Selected-hour mechanical Vitest gate after reorg: PASS, 10 files / 123 tests.
- `npm run test:quality:selected-hour`: PASS, 18 files / 158 tests.
- `npm run test:e2e:selected-hour`: PASS, 13 Chromium tests, 1.1m.
- `npm run build`: PASS. Existing warnings remain for Svelte module-level reassignment in `Model.svelte`, unused `UTCIPointCloud.svelte` export, large chunks, and unused external imports.
- `npm run check`: FAIL with inherited static debt, 163 errors and 4 warnings in 34 files. No errors were reported in the moved selected-hour files; touched route files still surface preexisting diagnostics elsewhere in those files.
- `git diff --check`: PASS after generated-state cleanup.
- Task 6 pre-move `npm run test:quality:selected-hour`: PASS, 18 files / 158 tests.
- Task 6 pre-move `npm run test:e2e:selected-hour`: PASS, 13 Chromium tests.
- Task 6 GPU-focused Vitest gate: PASS, 12 files / 57 tests, 4 skipped.
- Task 6 post-move `npm run test:quality:selected-hour`: PASS, 18 files / 158 tests.
- Task 6 post-move `npm run test:e2e:selected-hour`: PASS, 13 Chromium tests.
- Task 6 post-move `npm run build`: PASS with the same existing warning profile.
- Task 7 pre-move `npm run test:quality:selected-hour`: PASS, 18 files / 158 tests.
- Task 7 pre-move `npm run test:e2e:selected-hour`: PASS, 13 Chromium tests.
- Task 7 pre-move `npm run build`: PASS with the same existing warning profile.
- Task 7 core/parity-sensitive Vitest gate: PASS, 14 files / 57 tests, 1 skipped.
- Task 7 post-move `npm run test:quality:selected-hour`: PASS, 18 files / 158 tests.
- Task 7 post-move `npm run test:e2e:selected-hour`: PASS, 13 Chromium tests.
- Task 7 post-move `npm run build`: PASS with the same existing warning profile.
- Task 7 colocated source test: PASS, `src/lib/compute/core/analysisGridFromBounds.test.ts`, 1 test.
- Task 7 `npm run check`: FAIL with inherited static debt, 160 errors and 4 warnings in 33 files. The first touched-file errors remain the moved inherited `ArrayBufferLike` debt in `viewer/src/lib/compute/gpu/mergeAndBvh.worker.ts`; the core math move did not introduce formula/check errors.
- Task 7 `git diff --check`: PASS after generated-state cleanup.
- Task 8 pre-move `npm run test:quality:selected-hour`: PASS, 18 files / 158 tests.
- Task 8 pre-move `npm run test:e2e:selected-hour`: PASS, 13 Chromium tests.
- Task 8 pre-move `npm run build`: PASS with the same existing warning profile.
- Task 8 on-demand Vitest gate: PASS, 7 files / 41 tests.
- Task 8 post-move `npm run test:quality:selected-hour`: PASS, 18 files / 158 tests.
- Task 8 post-move `npm run test:e2e:selected-hour`: PASS, 13 Chromium tests.
- Task 9 pre-move `npm run test:quality:selected-hour`: PASS, 18 files / 158 tests.
- Task 9 pre-move `npm run test:e2e:selected-hour`: PASS, 13 Chromium tests.
- Task 9 pre-move `npm run build`: PASS with the same existing warning profile.
- Task 9 weather Vitest gate: PASS, 4 files / 54 tests.
- Task 9 post-move `npm run test:quality:selected-hour`: PASS, 18 files / 158 tests.
- Task 9 post-move `npm run test:e2e:selected-hour`: PASS, 13 Chromium tests.
- Task 9 post-move `npm run build`: PASS with the same existing warning profile.
- Task 10 root audit: root files are `compute-manager.ts` and `telemetry.ts`; directories are `core`, `gpu`, `on-demand`, `selected-hour`, and `weather`.
- Task 10 import-map artifact: `.compute-import-map.after.txt` generated for audit and removed; no temporary import maps remain.
- Final `npm run test:quality:selected-hour`: PASS, 18 files / 158 tests.
- Final `npm run test:e2e:selected-hour`: PASS, 13 Chromium tests.
- Final `npm run build`: PASS with the same existing warning profile.
- Final `npm run check`: FAIL with inherited static debt, 160 errors and 4 warnings in 33 files. The first touched-file errors remain the moved inherited `ArrayBufferLike` debt in `viewer/src/lib/compute/gpu/mergeAndBvh.worker.ts`; later errors are broad existing static debt across parity/services/routes/tests.
- Final `git diff --check`: PASS after generated-state cleanup.
- Final generated-state cleanup: `viewer/test-results/.last-run.json` restored; `.compute-import-map.before.txt` and `.compute-import-map.after.txt` absent.

## Subagent Reviews

- Pre-reorg behavior/scope review: no blocker; accepted proceeding because the focused case passed twice and full selected-hour E2E passed.
- Import-boundary review after reorg: no blocker.
- Selected-hour behavior/proof review after reorg: no blocker; strong visible GPU path proof boundaries remain guarded.
- Debug parity/fallback review after reorg: initial Git visibility blocker because moved files were untracked; resolved by marking new moved/doc files intent-to-add. No fallback/proof-surface blocker found.
- Task 6 import/source-lock review: no blocker; old GPU imports and hard-coded source paths were updated.
- Task 6 selected-hour proof review: no blocker; `strongVisibleGpuPath`, same-device, zero visible readback, `dataTexture`, and route diagnostics boundaries remain guarded.
- Task 6 fallback/debug parity review: flagged that the total diff still includes the already-completed Task 5 selected-hour move; fallback/proof surfaces remain available and Task 6 did not move core math, on-demand, weather, `compute-manager.ts`, or `telemetry.ts`.
- Task 7 parity/formula review: no blocker; `solarcal.ts`, `sunpath.ts`, `utci.ts`, `mrtReference.ts`, `tregenza.ts`, and `grid-generator.ts` were preserved byte-for-byte aside from the move, while `canonicalGrid.ts` and `analysisGridFromBounds.ts` changed only internal import paths.
- Task 7 scope review: noted the total working tree still includes earlier Task 5/6 moves. Within the Task 7 slice, production changes were import/comment-only outside the moved core files. The `solarcal.test.ts` assertion update is a scoped test-maintenance fix for stale expectations against removed aggregate fields; no SolarCal formulas changed.
- Task 8 on-demand implementation review: no blocker; the move was mechanical, with only an internal import rewrite in `on-demand/onDemandPrototypeStatus.ts` and consumers updated to the new bucket path.
- Task 9 weather implementation review: no blocker; `epw-parser.ts` and `projectWeather.ts` moved mechanically and the parser/mapping surface stayed unchanged.
- Task 10 root-leftover/import audit review: initial hygiene blocker because Playwright rewrote `viewer/test-results/.last-run.json`; resolved by restoring generated state. No stale on-demand/weather imports or temp import-map artifacts remained after cleanup.
- Task 10 fallback/proof-surface review: no blocker; `runAll()`, `readUtcisSlice()`, `readUtciBulk()`, `.bin`/Python reference loading, normal collect/parity flow, and legacy debug selected-hour wiring remain available.

## Remaining Work

- Keep `compute-manager.ts` at root until a dedicated facade/orchestration plan proves a better home.
- Keep `telemetry.ts` at root as the shared compute telemetry utility unless a later multi-file diagnostics bucket emerges.
- Keep broader Python/scripts/data organization as a separate migration.
- Investigate the earlier Ness Tziona timeout only if it reappears in focused or suite-order runs.
