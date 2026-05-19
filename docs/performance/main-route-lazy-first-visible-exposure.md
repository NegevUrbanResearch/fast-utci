# Main Route Lazy First-Visible Exposure

Collected: 2026-05-19

Source artifact: `data/performance-results/main-route-lazy-first-visible-exposure.json`

Route and mode: `/` with `utciRender=auto&utciRenderDiagnostics=1`; lazy rows add `utciLazyExposure=1`. This result does not use the debug route, parity comparison, or `.bin` reference loading.

## Summary

Lazy first-visible exposure improves first visible selected-hour time in every collected row, including the target Ness-Tziona 0.5m case. It does not yet look safe as a default behavior because scrub during the background full-solar fill is much slower than baseline scrub-after-visible.

The strongest current recommendation is to keep the path diagnostics-gated behind `utciLazyExposure=1`, preserve the default full-exposure path, and treat the next optimization question as background-fill contention rather than first-visible availability.

## Result Table

| Project | Grid | Mode | First visible ms | Scrub after visible ms | Scrub during background ms | Scrub after full ms | Background state before scrub | Background start delta ms |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- | ---: |
| Ben-Gurion | 2m | baseline full exposure | 1364.7 | 158.7 | n/a | n/a | n/a | n/a |
| Ben-Gurion | 2m | lazy first visible | 873.6 | 665.4 | 665.4 | 56.3 | background-full-running | 0.6 |
| Ness-Tziona | 2m | baseline full exposure | 2565.1 | 155.5 | n/a | n/a | n/a | n/a |
| Ness-Tziona | 2m | lazy first visible | 1684.4 | 1498.4 | 1498.4 | 87.4 | background-full-running | 0.6 |
| Ben-Gurion | 0.5m | baseline full exposure | 8616.5 | 216.3 | n/a | n/a | n/a | n/a |
| Ben-Gurion | 0.5m | lazy first visible | 5240.5 | 6226.0 | 6226.0 | 145.1 | background-full-running | 2.1 |
| Ness-Tziona | 0.5m | baseline full exposure | 25689.9 | 957.5 | n/a | n/a | n/a | n/a |
| Ness-Tziona | 0.5m | lazy first visible | 16227.9 | 17176.7 | 17176.7 | 369.9 | background-full-running | 0.7 |

## Proof Boundary

All collected rows stayed on the main-route GPU-resident path:

- `route=/`
- `rendererBackend=webgpu`
- `utciRenderResolved=gpuNative`
- `utciSurfaceSource=compute-buffer-selected-hour`
- `baseRenderTransport=compute-buffer-selected-hour`
- `dataTextureBuildCount=0`
- `selectedHourRuntimeContract.route=main`
- `selectedHourRuntimeContract.readbackInstrumentation=instrumented`
- `selectedHourRuntimeContract.visibleSelectedHourReadbackCount=0`
- `selectedHourRuntimeContract.strongVisibleGpuPath=true`

Lazy background sequencing was proven in every lazy row. `backgroundFullExposureStartedAtMs` was later than `visibleAcknowledgedAtMs`, with `backgroundStartDeltaMs` between 0.6 and 0.8 ms.

The scrub-during-background rows did exercise contention: every lazy row reported `preScrubExposureCoverageState=background-full-running`.

## Interpretation

The first-visible win is real: Ness-Tziona 0.5m improved from 25689.9 ms to 16227.9 ms, about 9.5 seconds faster.

The contention cost is also real: Ness-Tziona 0.5m scrub during background fill took 17176.7 ms versus 957.5 ms baseline scrub-after-visible. After full exposure completed, the same lazy row scrubbed in 369.9 ms.

So the lazy architecture answers the cold-start question positively, but the current background full-solar fill competes too strongly with active scrub. The next useful proof change is to throttle, chunk, or defer background fill while the user is actively scrubbing, then recollect this same artifact.

## Perspective Ensemble

Task classification: this is a performance architecture tradeoff, not a simple pass/fail optimization.

Panel A - Council:

- Cold-start lens: first visible time improves materially -> the gated architecture is worth keeping -> preserve the path for continued diagnostics.
- Interaction lens: scrub during background fill regresses badly -> do not promote this to default -> add a background scheduler that yields to active scrub.
- Proof-boundary lens: all rows preserve GPU-resident main-route contracts -> the proof surface is strong enough for discussion -> keep the same collector as the canonical lazy artifact.
- Maintenance lens: lazy/full exposure state now spans pipeline, session, controller, route, and E2E -> avoid adding another route-local workaround -> keep readiness state owned by the session/controller boundary.

Tensions:

- Faster first paint vs responsive scrub: this run improves cold start by seconds but can make the next interaction much worse.
- Background completeness vs foreground priority: eager background fill makes later scrubs fast, but it competes with the very first user action.
- Diagnostic truth vs product simplicity: the proof needs explicit state fields, but the default path should remain unchanged until contention is solved.

Panel B - Adversarial red cell:

- Attack target: promoting lazy first-visible exposure before background contention is controlled.
- Contention vector: background solar work monopolizes the GPU queue -> the user scrubs immediately after first visible and sees a multi-second stall -> mitigation is chunking, throttling, or cancelling/restarting background fill around active scrub.
- Evidence vector: the collector currently proves one scripted contention pattern -> real use may include multiple rapid scrubs and month changes -> add a repeated-scrub lazy contention row before any default flip.
- State-staleness vector: background full-ready diagnostics can be overtaken by a newer visible selected hour -> stale state could make the UI or report claim full coverage too early -> keep session-scoped readiness and request identity tests in place.

Strongest attack: the current lazy path can make the product feel worse exactly when the user first interacts with it. The first visible frame arrives sooner, but the next scrub on Ness-Tziona 0.5m took 17.2 seconds during background fill. That is a bad trade unless the background fill becomes polite to foreground work.

Falsifiers and early warnings:

- Lazy scrub during background is within about 25% of baseline scrub-after-visible on Ness-Tziona 0.5m.
- Repeated scrub while background fill is running does not produce a long queue-wait tail.
- Background start remains strictly after visible acknowledgement in every row.
- Proof contracts stay unchanged: `compute-buffer-selected-hour`, `dataTextureBuildCount=0`, and visible selected-hour readback count `0`.

Conditional recommendation: keep `utciLazyExposure=1` diagnostics-only, continue from this artifact, and make the next proof change about foreground-priority scheduling for background full exposure. Reconsider default lazy exposure only after the contention row stops regressing.
