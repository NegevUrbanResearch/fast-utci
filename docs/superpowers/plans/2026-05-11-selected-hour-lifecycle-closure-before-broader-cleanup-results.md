# Selected-Hour Lifecycle Closure Results

Date: 2026-05-11

## Scope

This note closes the selected-hour GPU lifecycle blockers that had to be resolved before broader route decomposition, scene sync extraction, or folder reorganization.

## Closed Contracts

- Scene releases are controller-instance safe: route-host release forwarding checks both `controllerIdentity` and `controllerInstanceId`.
- Scene async copy supersession is controller-instance safe in both `UTCIPointCloud` and `ComparisonRenderer`.
- `UTCIPointCloud` and `ComparisonRenderer` use the same exactly-once release notifier.
- Missing controller identity or instance id suppresses release emission.
- Stale rejected GPU outputs do not destroy a GPU handle still owned by the current accepted output or a retired accepted output.
- Fresh stale GPU outputs still dispose immediately.

## Verification

- `cd viewer; npm run test:quality:selected-hour` PASS: 18 files / 158 tests passed.
- `cd viewer; npm run test:e2e:selected-hour` PASS: 13 Chromium tests passed. Main route selected-hour diagnostics, hour/month changes, strong visible GPU path with comparison readbacks separated, Ness Tziona live range, debug shared-host behavior, and debug parity `.bin` scoping all passed.
- `cd viewer; npm run build` PASS: Vite/SvelteKit build completed and wrote the static site. Build emitted existing warnings, including `Model.svelte` module-level reassignment warnings and the `UTCIPointCloud.svelte` unused `model` export warning.
- `cd viewer; npm run check` FAIL: inherited static debt remains. Final run reported 163 errors and 4 warnings in 34 files. No touched-file TypeScript errors remained after fixing `ComparisonRenderer.svelte` notifier return typing and `UTCIPointCloud.svelte` controller identity narrowing. The remaining touched-file item is the preexisting `UTCIPointCloud.svelte` unused exported `model` warning.
- `cd ..; git diff --check` PASS: no whitespace errors. Git printed CRLF conversion warnings only.

## Review Agents

- Runtime ownership reviewer: REVIEW PASS. Confirmed host release forwarding is controller-instance safe, stale same-content old-instance releases are ignored, the previous async-copy supersession blocker is closed by `controllerInstanceId` checks in both scene components, stale/reused GPU handles cannot destroy current or retired accepted outputs, `strongVisibleGpuPath` was not loosened, and fallback paths remain preserved.
- Scene lifecycle reviewer: REVIEW PASS. Confirmed both scene components use the shared exactly-once notifier, missing identity or instance id cannot emit a release, async copy supersession is instance-safe, selected-hour visible GPU diagnostics remain honest, data-texture/CPU fallback paths are preserved, and no route decomposition or scene sync extraction happened.
- Non-blocking reviewer notes: there is still no mounted component behavioral test for a late old-instance async copy completion/failure; current coverage uses route-host/controller diagnostics tests plus source-level scene call-site contracts.

## Broader Cleanup Readiness

The selected-hour lifecycle layer is stable enough to begin a separate broader cleanup plan for route decomposition, scene sync extraction, and folder/module organization.

## Explicitly Deferred

- Route decomposition.
- Full scene sync state-machine extraction.
- Folder reorganization.
- 0.5m performance work.
- Removal of legacy `.bin`, `dataTexture`, parity, collect, or debug fallback paths.
