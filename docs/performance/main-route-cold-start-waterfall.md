# Main Route Cold-Start Waterfall Evidence

Date: 2026-05-30

This note is refreshed from the current JSON artifact only:
[data/performance-results/main-route-cold-start-waterfall.json](../../data/performance-results/main-route-cold-start-waterfall.json).

The collector measures the product route `/` with `utciRender=auto` and `utciRenderDiagnostics=1`. It records initial first visible publication and the first post-visible scrub on the accessible `Select analysis hour` slider. It does not use `/debug`, parity, Python reference data, or `.bin` comparison.

The clean rerun passed after the earlier overlapped verification failed, so this artifact is current as of 2026-05-30:

```powershell
cd viewer
npx playwright test tests/e2e/main-route-cold-start-waterfall.spec.ts --project=chromium --workers=1 --reporter=list
```

Result: `1 passed (1.0m)`.

Additional final verification on the same change set:

```powershell
cd viewer
npx vitest run tests/services/pointCloudService.surface.test.ts tests/compute/webgpu-on-demand-source-locks.test.ts tests/compute/live-selected-hour-controller.test.ts --no-file-parallelism --maxWorkers=1
npm run check
npx playwright test tests/e2e/main-route-manual-diagnostics.spec.ts --project=chromium --workers=1 --reporter=list --grep "publishes Ness Tziona 0.5m camera and tooltip interaction diagnostics"
cd ..
git diff --check
```

Results: focused Vitest `104 passed`; `npm run check` reported `0 errors and 0 warnings`; the Ness Tziona 0.5m manual interaction proof passed in `43.1s`; `git diff --check` exited `0` with only line-ending warnings.

## Summary

`Exposure queue wait ms` remains `n/a` where the JSON value is `null`. `Draw indices` uses the current indexed-surface `renderPublicationDrawIndexCount` from the initial render publication.

| Project | Grid m | Points | Initial first visible ms | Exposure queue wait ms | Initial render update ms | Position vertices | Draw indices | Initial render-owned MiB | First post-visible scrub ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ben-Gurion | 2 | 104,445 | 5,373.7 | n/a | 353.3 | 105,152 | 626,670 | 4.4 | 419.3 |
| Ness-Tziona | 2 | 511,840 | 5,440.8 | n/a | 609.3 | 513,315 | 3,071,040 | 21.5 | 143.8 |
| Ben-Gurion | 0.5 | 1,662,657 | 9,885.7 | n/a | 1,733.5 | 1,665,476 | 9,975,942 | 69.8 | 218.5 |
| Ness-Tziona | 0.5 | 8,171,761 | 26,031.2 | n/a | 5,098.9 | 8,177,652 | 49,030,566 | 343.0 | 1,112.5 |

## Proof Boundary

Both initial and first post-visible scrub diagnostics asserted the main-route WebGPU path: `rendererBackend=webgpu`, `utciRenderResolved=gpuNative`, `utciSurfaceSource=compute-buffer-selected-hour`, `baseRenderTransport=compute-buffer-selected-hour`, `dataTextureBuildCount=0`, and `selectedHourRuntimeContract.visibleSelectedHourReadbackCount=0`.
