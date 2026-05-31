# Main Route Exposure Scheduler Evidence

## Baseline

- artifact: `data/performance-results/main-route-visual-freeze-map.json`
- scrub artifact: `data/performance-results/main-route-cold-start-waterfall.json`
- visual-freeze control case: `ness-tziona-0_5m`
- scrub artifact control case: `ness-tziona-0_5m-single-submit`
- chunked case: `ness-tziona-0_5m-chunked-8192`
- tuned case: `ness-tziona-0_5m-chunked-4096`
- follow-up tuned case: `ness-tziona-0_5m-chunked-2048`
- proof surface: `/`

## Result

| case | scheduler | first publication ms | first scrub ms | top rAF gap ms | top interval gap ms | exposure queue ms | max scheduler queue wait ms | render update ms | owned GPU MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ness-tziona-0_5m` | `single-submit` | 27080.099999964237 | 746.9102999999959 | 3017.100000000002 | 1634.2999999523163 | 16148.200000047684 | 16148.200000047684 | 4122.699999988079 | 685.8698692321777 |
| `ness-tziona-0_5m-chunked-8192` | `chunked`, 16 slices | 25780.69999998808 | 912.3907999999938 | 1600.2000000000007 | 1633.800000011921 | 16307.099999964237 | 1407.0999999642372 | 4011.899999976158 | 685.8698692321777 |
| `ness-tziona-0_5m-chunked-4096` | `chunked`, 32 slices | 26232.80000001192 | 1124.8364000000001 | 1587 | 1611.6000000238419 | 16636.999999940395 | 733.1999999880791 | 3954.900000035763 | 685.8698692321777 |
| `ness-tziona-0_5m-chunked-2048` | `chunked`, 63 slices | 26836.600000023842 | 764.4930000000168 | 1569.1000000000022 | 1604.0999999642372 | 16961.900000154972 | 381.30000001192093 | 4037.2000000476837 | 685.8698692321777 |

## Remaining Freeze Profile

The 2048 follow-up proves smaller slices can push the largest observed scheduler queue wait below the 500 ms frame-gap target (`381.30000001192093 ms`), while keeping first publication and first post-visible scrub within 10% of this run's control. However, the browser still recorded a `1569.1000000000022 ms` top rAF gap and `1367 ms` top long task.

The top 2048 rAF gap was `24975.1 -> 26544.2 ms`. It overlaps the initial render publication scene path:

- `controllerStatePublishedAtMs`: `25022.899999976158`
- `sceneSyncAttemptStartedAtMs`: `25025.69999998808`
- `sceneLayoutKeyCompletedAtMs`: `25281.19999998808`
- `scenePublicationPlanReadyAtMs`: `25734.5`
- `sceneSurfacePendingStorageInitAtMs`: `26325.30000001192`
- `renderStorageReadyAtMs`: `26550.899999976158`
- `sceneSyncCompletedAtMs`: `26835.80000001192`

Interpretation: scheduler slicing is doing its job, but the remaining visible freeze is now dominated by main-thread render publication / scene sync for the dense NZ 0.5m surface rather than by one oversized exposure queue wait.

## Proof Notes

- `publicationReached`: true for all four NZ 0.5m cases.
- GPU-native proof held in the cold-start artifact for control, 8192, 4096, and 2048: `rendererBackend=webgpu`, `utciRenderResolved=gpuNative`, `utciSurfaceSource=compute-buffer-selected-hour`, `baseRenderTransport=compute-buffer-selected-hour`, `baseSameDeviceForComputeAndRender=true`, `visibleSelectedHourReadbackCount=0`, `strongVisibleGpuPath=true`.
- No forbidden comparison requests or comparison fields were recorded for the NZ 0.5m cold-start cases.
- Freeze-map page errors, request failures, and crashes were zero for control, 8192, 4096, and 2048.
- App-owned GPU memory was unchanged across the compared cases.

## Decision

Keep query-gated. Do not promote chunked exposure scheduling to a default or product recommendation from this evidence.

The chunked scheduler reduced the worst exposure queue stall. At 2048, it also kept first visible and first post-visible scrub close to control in this run. Even so, the best tuned result still had a top rAF gap of `1569.1000000000022 ms` and an interval gap of `1604.0999999642372 ms`, so it fails the hard 500 ms responsiveness gate.

Recommendation for the next implementation plan: keep the query-gated scheduler as the cooperative GPU-work foundation, but shift the next optimization target to render publication / scene sync for the dense NZ 0.5m surface. Do not spend the next plan only tuning smaller exposure slices unless that work is paired with explicit render-publication profiling or interaction-priority protection.

## Verification

- `cd viewer; npm run check`
  - passed: `svelte-check found 0 errors and 0 warnings`
- `cd viewer; npm run test -- --run tests/compute/exposureScheduling.test.ts tests/compute/compute-manager-on-demand.test.ts tests/compute/live-selected-hour-session.test.ts tests/compute/live-selected-hour-route-host.test.ts tests/compute/webgpu-on-demand-source-locks.test.ts tests/compute/webgpuUtciPipeline.behavior.test.ts tests/diagnostics/main-route-utci-diagnostics.test.ts tests/e2e/main-route-exposure-scheduler-collectors-source-lock.test.ts`
  - passed: 8 test files, 112 tests
- `cd viewer; npm run test:quality:selected-hour`
  - passed: 18 test files, 196 tests
- `cd viewer; npx playwright test --config=playwright.collect.config.ts tests/e2e/main-route-visual-freeze-map.spec.ts --project=chromium --workers=1 --reporter=list --timeout=900000`
  - passed: 1 headed Chromium collector test; updated `data/performance-results/main-route-visual-freeze-map.json`
- `cd viewer; npx playwright test --config=playwright.collect.config.ts tests/e2e/main-route-cold-start-waterfall.spec.ts --project=chromium --workers=1 --reporter=list --timeout=900000`
  - passed: 1 headed Chromium collector test; updated `data/performance-results/main-route-cold-start-waterfall.json`
- `cd viewer; npm test -- --run tests/e2e/main-route-exposure-scheduler-collectors-source-lock.test.ts`
  - passed: 1 test file, 2 tests
- `cd viewer; npx playwright test --config=playwright.collect.config.ts tests/e2e/main-route-visual-freeze-map.spec.ts --list`
  - passed: 1 headed collector test listed
- `cd viewer; npx playwright test --config=playwright.collect.config.ts tests/e2e/main-route-cold-start-waterfall.spec.ts --list`
  - passed: 1 headed collector test listed
- `cd viewer; npx playwright test --config=playwright.collect.config.ts tests/e2e/main-route-visual-freeze-map.spec.ts --project=chromium --workers=1 --reporter=list --timeout=900000`
  - passed: 1 headed Chromium collector test; refreshed artifact includes `ness-tziona-0_5m-chunked-2048`
- `cd viewer; npx playwright test --config=playwright.collect.config.ts tests/e2e/main-route-cold-start-waterfall.spec.ts --project=chromium --workers=1 --reporter=list --timeout=900000`
  - passed: 1 headed Chromium collector test; refreshed artifact includes `ness-tziona-0_5m-chunked-2048`
- `git diff --check`
  - passed: exit 0; only LF-to-CRLF warnings were printed
