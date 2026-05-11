# Route And Scene Cleanup Before Static Debt Results

Date: 2026-05-11

## Scope

This pass decomposed the main and debug routes, extracted the selected-hour scene sync state machine, and performed only limited module organization after the route/scene boundaries were proven.

The pass did not start broad repo-wide static-debt cleanup, did not remove fallback paths, and did not create commits or worktrees.

## Route Decomposition

Main route helpers now live under `viewer/src/routes/main/`:

- `liveSelectedHour.ts` owns selected-hour host release forwarding, diagnostics shaping, and main-route live selected-hour proof helpers.
- `modelSelection.ts` owns URL/project/model selection policy helpers.
- `tooltip.ts` owns tooltip suppression/routing and selected-hour tooltip analysis selection helpers.

Debug route helpers now live under `viewer/src/routes/debug/`:

- `queryState.ts` keeps debug query parsing at the route boundary.
- `selectedHourMode.ts` derives debug selected-hour diagnostics state and shared-host-vs-legacy mode policy.
- `sharedHostWiring.ts` owns shared-host route-host setup, release forwarding, scene projection, and shared-host diagnostics patching.
- `legacySelectedHourWiring.ts` owns the legacy debug selected-hour identity and release-forwarding boundary.

The debug-only parity runtime remained in `debug/+page.svelte` intentionally. That code still carries `.bin`, Python comparison, parity intermediates, collect, and proof-window behavior, so extracting it in this pass would have increased risk without improving the route boundary enough.

## Scene Sync Extraction

Added `viewer/src/lib/components/scene/acceptedGpuResidentSurfaceSync.ts` to own:

- sync key capture.
- active run tokens.
- controller identity and controller instance snapshots.
- supersession checks.
- copy-complete, copy-failed, and superseded release mapping.

`UTCIPointCloud.svelte` and `ComparisonRenderer.svelte` now act as scene-specific adapters around that helper. The helper run key includes controller instance identity, so same-content old-instance async copy completions cannot be mistaken for the current instance.

Added `viewer/tests/scene/acceptedGpuResidentSurfaceSync.behavior.test.ts` for the late old-instance async-copy behavior, including current-instance exactly-once release and stale completion/failure suppression.

## Folder Organization

Moved only helper files created by this plan:

- `viewer/src/routes/mainRouteLiveSelectedHour.ts` -> `viewer/src/routes/main/liveSelectedHour.ts`
- `viewer/src/routes/mainRouteModelSelection.ts` -> `viewer/src/routes/main/modelSelection.ts`
- `viewer/src/routes/mainRouteTooltip.ts` -> `viewer/src/routes/main/tooltip.ts`
- `viewer/src/routes/debug/debugRouteQueryState.ts` -> `viewer/src/routes/debug/queryState.ts`
- `viewer/src/routes/debug/debugRouteSelectedHourMode.ts` -> `viewer/src/routes/debug/selectedHourMode.ts`
- `viewer/src/routes/debug/debugRouteSharedHostWiring.ts` -> `viewer/src/routes/debug/sharedHostWiring.ts`
- `viewer/src/routes/debug/debugRouteLegacySelectedHourWiring.ts` -> `viewer/src/routes/debug/legacySelectedHourWiring.ts`

Scene selected-hour helpers were left beside the scene components. There are only two helpers there, and moving them into a folder would add import churn without much clarity.

## Static Debt Boundary

Baseline `npm run check` before this cleanup failed with 163 errors and 4 warnings in 34 files.

Final `npm run check` still fails with 163 errors and 4 warnings in 34 files. The new helper type issues found during this pass were fixed. Remaining touched-file diagnostics are inherited:

- `viewer/src/routes/debug/+page.svelte`: existing parity intermediate `Float32Array | null` vs `number[]` window typing around the debug proof export.
- `viewer/src/lib/components/scene/UTCIPointCloud.svelte`: existing unused exported `model` warning.

The rest of the failure set is inherited repo-wide debt, including ArrayBufferLike/ArrayBuffer transfer typing, parity reference union narrowing, Three `Object3D` narrowing, UTCI/grid fixture shape drift, `SunPath.svelte` store drift, and `Model.svelte` Svelte/type warnings.

## Verification

- Baseline `git status --short`: only the untracked plan file was dirty before implementation.
- Baseline `cd viewer; npm run test:quality:selected-hour`: PASS, 18 files / 158 tests.
- Baseline `cd viewer; npm run test:e2e:selected-hour`: PASS, 13 Chromium tests.
- Baseline `cd viewer; npm run check`: FAIL, 163 errors / 4 warnings in 34 files.
- Main route focused tests: PASS after extraction; one Svelte reactivity issue was found by E2E and fixed by passing explicit diagnostics dependencies into the helper.
- Debug route focused Vitest: PASS, 6 files / 57 tests.
- Debug shared-host Playwright diagnostics: PASS, 2 Chromium tests.
- Debug baseline/parity Playwright diagnostics: PASS, 5 Chromium tests.
- Normal collect diagnostic: PASS, 1 Chromium test.
- Scene sync focused tests: PASS, 5 files / 18 tests.
- Post-move focused route/scene tests: PASS, 6 files / 43 tests.
- Final `cd viewer; npm run test:quality:selected-hour`: PASS, 18 files / 158 tests.
- Final `cd viewer; npm run test:e2e:selected-hour`: PASS, 13 Chromium tests.
- Final focused route/scene check `cd viewer; npx vitest run tests/routes/main-route-live-selected-hour-source-lock.test.ts tests/routes/main-route-live-selected-hour.test.ts tests/routes/main-route-model-selection.test.ts tests/routes/main-route-tooltip.test.ts tests/debug/debug-route-decomposition-source-lock.test.ts tests/scene/acceptedGpuResidentSurfaceSync.behavior.test.ts`: PASS, 6 files / 43 tests.
- Final scene sync check `cd viewer; npx vitest run tests/scene/acceptedGpuResidentOutputRelease.test.ts tests/scene/acceptedGpuResidentOutputReleaseCallsites.test.ts tests/scene/acceptedGpuResidentSurfaceSync.behavior.test.ts tests/scene/utci-surface-sync.test.ts tests/scene/utciComputeBufferRenderBridge.test.ts`: PASS, 5 files / 18 tests.
- Final normal collect diagnostic `cd viewer; npx playwright test tests/e2e/debug-route-normal-collect-diagnostics.spec.ts --project=chromium --workers=1 --reporter=list --timeout=30000`: PASS, 1 Chromium test.
- Final `cd viewer; npm run build`: PASS with existing warnings.
- Final `cd viewer; npm run check`: FAIL, 163 errors / 4 warnings in 34 files; inherited static debt remains.
- Final `git diff --check`: PASS; Git printed CRLF conversion warnings only.

## Review Agents

- Main route reviewers: no blocking findings. Early review notes led to adding a mount/state test and tightening the exported release param type.
- Debug route reviewers: found a stale Svelte dependency around legacy counters and stale source-lock allowlists; both were fixed. No blocking behavioral findings remained. Parity runtime was intentionally left in the debug route.
- Scene lifecycle reviewers: initially found instance-only sync and duplicate pending-diagnostics risks; both were fixed. Final review then found one recreate-path blocker where scene components could overwrite helper-published `pending` diagnostics back to `idle`; this was fixed by preserving pending status/request id during compute-surface recreation. Re-review confirmed the blocker is closed.
- Final route reviewer: no blocking findings. Confirmed main-route helpers remain free of debug-only behavior, debug helpers preserve query/selected-hour/shared-host/legacy boundaries, and `strongVisibleGpuPath` remains derived from the runtime contract.
- Final static-debt boundary reviewer: no blocking findings. Confirmed the cleanup stayed scoped to route/scene helpers and targeted tests, and remaining `npm run check` failures match inherited debt rather than new cleanup debt.

## Remaining Work

- Broad inherited `npm run check` debt needs a separate plan.
- Debug parity runtime can be decomposed later, but only with narrower proof around `.bin`, Python comparison, parity intermediates, and normal collect exports.
- Scene selected-hour helper foldering can wait until there are enough scene selected-hour modules to justify a subfolder.
