# Main Route Cold-Start Waterfall Evidence

Date: 2026-05-18

## Scope

This note measures the main route `/` cold-start path. It is separate from warm-scrub selected-hour render diagnostics and does not use `/debug`, `.bin`, Python reference data, or parity comparison.

JSON source: [data/performance-results/main-route-cold-start-waterfall.json](../../data/performance-results/main-route-cold-start-waterfall.json)

Follow-up lazy-exposure evidence: [docs/performance/main-route-lazy-first-visible-exposure.md](main-route-lazy-first-visible-exposure.md)

Collector:

```powershell
cd viewer
npx playwright test tests/e2e/main-route-cold-start-waterfall.spec.ts --project=chromium --workers=1 --reporter=list
```

Result: `1 passed (1.1m)` on 2026-05-18.

## Proof Boundary

- `rendererBackend=webgpu`
- `utciRenderResolved=gpuNative`
- `utciSurfaceSource=compute-buffer-selected-hour`
- `baseRenderTransport=compute-buffer-selected-hour`
- `dataTextureBuildCount=0`
- `selectedHourRuntimeContract.route=main`
- `selectedHourRuntimeContract.readbackInstrumentation=instrumented`
- `selectedHourRuntimeContract.strongVisibleGpuPath=true`
- `selectedHourRuntimeContract.visibleSelectedHourReadbackCount=0`
- no python/bin/debug comparison fields
- no forbidden comparison requests

## Included Analyses

- `Ben-Gurion/20250815_grid_2m_fullday`
- `Ness-Tziona/exploded/nes_tziona_unblock_2`

## Timing Table

Durations below are computed from the `coldStart` start/completion stamps in the JSON, except `First visible ms`, which is computed from the route-analysis start stamp to the top-level `firstSelectedHourVisibleAtMs`. That visible stamp is sourced from `renderPublication.renderPublicationTimeline.controllerVisibleAcknowledgedAtMs`. `Selected-hour request span ms` is the wall-clock span from first selected-hour dispatch start to completion in the cold waterfall, not the lower-level `timings.oneHourDispatchMs`. `Initial render publication ms` uses the existing coarse `timings.renderUpdateMs` field and is not yet a deeper sub-bucket split.

| Project | Grid m | First visible ms | Analysis load ms | Model load ms | Model processing ms | Payload prepare ms | Worker BVH ms | Upload ms | Exposure precompute ms | Selected-hour request span ms | Initial render publication ms | App-owned memory MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ben-Gurion | 2 | 2632.5 | 56.2 | 422.6 | 628.0 | 41.7 | 123.2 | 33.5 | 779.6 | 163.2 | 354.2 | 12.4 |
| Ness-Tziona | 2 | 4770.7 | 149.9 | 251.8 | 1116.3 | 301.0 | 270.8 | 80.4 | 1523.0 | 189.6 | 844.6 | 60.5 |
| Ben-Gurion | 0.5 | 9902.4 | 34.7 | 363.7 | 549.0 | 123.2 | 103.6 | 208.0 | 6107.4 | 197.4 | 2137.6 | 196.6 |
| Ness-Tziona | 0.5 | 30276.2 | 76.0 | 229.3 | 1131.5 | 293.2 | 221.0 | 1177.4 | 17738.0 | 180.1 | 8946.5 | 966.4 |

## Exposure Split

The current exposure instrumentation splits CPU-side command encoding and workload dimensions. Solar and sky still run in one submitted command buffer, so `Exposure queue wait ms` is the combined GPU wait for both passes. It is not a separate solar-vs-sky GPU timer. `Exposure precompute ms` in the previous table is the route/session lifecycle span; `Exposure queue wait ms` is the narrower runtime queue wait reported by the WebGPU pipeline.

| Project | Grid m | Exposure queue wait ms | Command encode total ms | Exposure encode ms | Solar encode ms | Sky encode ms | Points | Total time steps | Daylight time steps | Solar ray budget | Sky ray budget | Point chunks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ben-Gurion | 2 | 722.8 | 0.4 | 0.3 | 0.1 | 0.1 | 104445 | 288 | 145 | 15144525 | 15144525 | 1 |
| Ness-Tziona | 2 | 1460.3 | 0.3 | 0.2 | 0.1 | 0.1 | 511840 | 288 | 145 | 74216800 | 74216800 | 1 |
| Ben-Gurion | 0.5 | 6053.2 | 0.1 | 0.1 | 0.1 | 0.0 | 1662657 | 288 | 145 | 241085265 | 241085265 | 1 |
| Ness-Tziona | 0.5 | 16371.6 | 2.1 | 1.9 | 1.5 | 0.4 | 8171761 | 288 | 145 | 1184905345 | 1184905345 | 2 |

## Diagnosis

- Exposure precompute is the largest confirmed cold-start bucket in all four cases: `779.6 ms` for BG 2m, `1523.0 ms` for Ness Tziona 2m, `6107.4 ms` for BG 0.5m, and `17738.0 ms` for Ness Tziona 0.5m. This is the top cold-path target.
- The new exposure split shows CPU command encoding is negligible (`0.1-2.1 ms`). The slow part is GPU queue wait: `722.8 ms`, `1460.3 ms`, `6053.2 ms`, and `16371.6 ms`. Optimizing TypeScript setup, bind-group construction, or command encoding will not materially improve cold start.
- Current workload counters show solar daylight ray budget and sky ray budget are equal in this collection because both are `points * 145`: 145 daylight time steps and 145 Tregenza sky patches. The all-time-step solar dispatch still covers 288 time steps, but nighttime/invalid vectors return early in the shader.
- Because first visible waits for full solar+sky exposure, the most direct cold-start experiment was to stop making first visible pay for all exposure work. The lazy first-visible proof confirmed this can reduce first visible time, but it also showed that background fill can make the next scrub dramatically slower. That makes lazy exposure unsafe as a product direction unless foreground scrub keeps priority.
- Initial render publication is the second major dense-grid bucket: `2137.6 ms` for BG 0.5m and `8946.5 ms` for Ness Tziona 0.5m. This is currently a coarse `renderUpdateMs` measure; it should be split after the exposure first-visible decision, not before.
- Upload is modest at 2m and BG 0.5m, but Ness Tziona 0.5m records `1177.4 ms`; this is a third-order candidate after exposure and dense render publication.
- The selected-hour request span is not the cold-start bottleneck in this pass: it is about `163.2-197.4 ms` across the four cases.
- No table field is null in this collection. The proof counters remain clean and scoped to the product route.

## Follow-up Lazy Exposure Finding

The lazy first-visible artifact collected on 2026-05-19 compared baseline full exposure with `utciLazyExposure=1` on the same main-route GPU-resident proof boundary.

It proved a real first-visible win:

| Project | Grid m | Baseline first visible ms | Lazy first visible ms | Improvement ms |
| --- | ---: | ---: | ---: | ---: |
| Ben-Gurion | 2 | 1364.7 | 873.6 | 491.1 |
| Ness-Tziona | 2 | 2565.1 | 1684.4 | 880.7 |
| Ben-Gurion | 0.5 | 8616.5 | 5240.5 | 3376.0 |
| Ness-Tziona | 0.5 | 25689.9 | 16227.9 | 9462.0 |

It also proved the current lazy approach hurts the interaction path too much:

| Project | Grid m | Baseline scrub after visible ms | Lazy scrub during background ms | Lazy scrub after full ms |
| --- | ---: | ---: | ---: | ---: |
| Ben-Gurion | 2 | 158.7 | 665.4 | 56.3 |
| Ness-Tziona | 2 | 155.5 | 1498.4 | 87.4 |
| Ben-Gurion | 0.5 | 216.3 | 6226.0 | 145.1 |
| Ness-Tziona | 0.5 | 957.5 | 17176.7 | 369.9 |

Every lazy contention row reported `exposureCoverage.state === background-full-running`, and background full exposure started only after visible acknowledgement (`backgroundStartDeltaMs` about `0.6-2.1 ms`). So the regression is not a collector artifact: the test did exercise active background-fill contention.

The practical conclusion is that first-visible lazy exposure is useful evidence, but it should not become the default path. The previous full-exposure path preserves the fast scrub contract, and that contract is more important than a faster first frame. Future cold-start work should reduce initialization while preserving fast scrub, or prove a foreground-priority background scheduler before revisiting lazy exposure.

## Next Optimization Candidates

1. Keep fast scrub as the cold-start optimization gate. The default full-exposure path currently preserves this contract, and the lazy proof showed why the gate matters. Future cold-start collectors should report first visible and first post-visible scrub together so a faster first frame cannot hide an interaction regression.
2. Full-exposure cold-start reduction without lazy foreground semantics. The lazy proof already answered the practical split question: moving full exposure into the background improves first visible, but creates an unsafe scrub contention window. The next exposure work should target ways to reduce the full pre-live exposure cost while keeping the readiness needed for fast scrub before the route becomes interactive. Candidate directions include reducing redundant all-day solar work, improving dispatch shape, caching reusable exposure inputs, or persisting/reusing exposure products where valid.
3. Dense initial render publication split. Ness Tziona 0.5m still spends `8946.5 ms` in coarse initial render update. This is the second major dense-grid bucket after exposure, and it should be split into scene sync, layout, storage init, buffer copy, queue drain, and publication acknowledgement before optimization.
4. Upload/model processing follow-up. These are real but smaller, and they do not explain the dominant dense-grid cold-start latency as directly as exposure queue wait and render publication.
5. Lazy exposure only as a bounded scheduler proof, not the main plan. If revisited, the proof must show foreground-priority background fill: lazy scrub during background should land close to baseline scrub, repeated scrubs should not create long queue-wait tails, and the path must remain gated behind `utciLazyExposure=1`.

## Non-Goals

- No overlay copy changes in this pass.
- No warm-scrub optimization in this pass.
- No `.bin` or parity comparison in this pass.
