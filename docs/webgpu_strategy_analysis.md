# WebGPU Strategy Analysis

Updated: 2026-07-10

> **2026-05-11 route-name note:** The active debug route in this checkout is now `/debug` at `viewer/src/routes/debug/+page.svelte`. Older references to `/debug-webgpu-utci` describe the same debug/parity role before the route rename and should not be copied into new execution plans.

## 2026-07-10 Current Documentation Baseline

The WebGPU route is the main user-facing route for fast-utci. Onboarding docs should describe the app from this current state.

Current framing for new docs:

- `/` is the live WebGPU user-facing route.
- `/debug` is for parity, collectors, `.bin` comparison, and diagnostics.
- Python/Ladybug was the intermediate Embree/parallel-CPU improvement over Grasshopper/Ladybug. It now supports reference calculations, legacy artifact reproduction, legacy exports, and parity checks; separate Python scripts also handle GIS postprocessing.
- Do not recommend Python/Ladybug as the normal high-throughput analysis path when a project can run through WebGPU.
- Precomputed `.bin` artifacts are useful for reference, compatibility, and debug workflows, but new interactive projects should start from GLB + metadata + weather.
- The public shade metric is Shading Availability Index (SAI), following Derech Tzel's shade-availability terminology. The code and data schema may still use `shading_index` as the implementation field name.
- Do not overstate the boundary as "zero CPU readback." The product boundary is narrower: the visible selected-hour WebGPU path renders through `compute-buffer-selected-hour`, while point tooltip, diagnostics, export, fallback, and parity paths may still do bounded CPU readback.

## 2026-06-11 Innovation District Live WebGPU Contract

Innovation District is a GLB-backed live WebGPU project. The strategy boundary for this class of project is:

- no Python or `.bin` payload is part of the product path; generated metadata plus the GLB are the source of truth for live analysis setup
- metadata `bounds` and grid resolution define the sample grid and preflight estimate; filtered occluder bounds must not shrink grid sizing or budget checks
- compute-BVH eligibility is carried on loaded mesh `userData.includeInComputeBvh`: known ground-family/context layers such as `ground`, `street`, `train_tracks`, and `district_outline` are excluded; known occluders such as `existing_buildings` and `trees_canopy` are included; an unset flag preserves legacy behavior instead of silently opting meshes out
- `has_shading_index: false` in generated metadata does not mean the main route lacks live SAI support; it only means there is no baked SAI binary, while live WebGPU may still publish SAI on the same-device GPU path

GIS export sits downstream of this route. The current Innovation District handoff path is: verified main route `/` -> live WebGPU UTCI/SAI -> collector raw active-cell arrays -> Python geospatial postprocess validation -> `cells.geoparquet`, `manifest.json`, and QA sample. It is not a separate Python/Ladybug analysis route.

Proof boundary for future performance claims: do not treat Innovation District as proof that dense live GLB-backed startup is broadly safe at scale without citing a browser-measured run for the exact route and hardware. Startup and publication cost remains a runtime risk to keep measuring, not a resolved performance fact.

## Decision Snapshot

The WebGPU/Three.js selected-hour route is now the main product path; future optimization decisions should come from measured cold-start/render-path evidence.

This claim is now backed by a fresh selected-hour baseline on the main route `/` plus the older debug-route captures below. The current-head main-route artifact is [docs/performance/main-route-selected-hour-current-head.md](performance/main-route-selected-hour-current-head.md). Older `debug-webgpu-utci` route references below are historical captures from before the route rename. It is not yet a blanket statement about every route or fallback path in the repo.

The reason is simple: the viewer already renders with Three.js `WebGPURenderer`, and the current scaling wall is no longer "can we compute UTCI on the GPU?" It is "can the renderer consume GPU-computed UTCI without CPU readback, CPU quantization, and CPU texture/color regeneration?"

Worker/LRU work is still useful as fallback or transitional plumbing, but it is no longer the strategic next step toward 0.5m resolution.

## Current Repo Status

Recent relevant commits:

| Commit | Status |
| --- | --- |
| `306bd8b` | Shipped bit-packed solar exposure and bulk UTCI readback. |
| `8d88771` | Restored incremental progress overlay during readback. |
| `0680a54` | Disabled MRT diagnostics by default to save VRAM. |
| `c4265ec` | Added 12-month parity/performance reporting. |
| `ed0284c` | Added higher-fidelity telemetry and VRAM audit work. |

Implemented as of 2026-05-09:

| Capability | Current state | Key files |
| --- | --- | --- |
| Three.js WebGPU rendering | `WebGPURenderer` is already used. | `viewer/src/lib/components/scene/Scene.svelte` |
| WebGPU compute pipeline | Solar exposure, sky exposure, and MRT/UTCI compute already run on WebGPU. | `viewer/src/lib/compute/gpu/webgpuUtciPipeline.ts` |
| Solar exposure storage | Already bit-packed as one bit per point-hour in a `u32` buffer. | `gpu/webgpuUtciPipeline.ts`, `gpu/shaders/exposure_solar.wgsl`, `gpu/shaders/mrt_utci.wgsl` |
| UTCI readback | The visible selected-hour main-route path is GPU-native when `compute-buffer-selected-hour` is available; bulk and bounded readbacks still exist for legacy, fallback, tooltip, debug, parity, and export flows. | `viewer/src/routes/main/liveSelectedHour.ts`, `viewer/src/lib/compute/selected-hour/liveUtciSelectedHourSession.ts`, `viewer/src/lib/compute/gpu/webgpuUtciPipeline.ts` |
| CPU UTCI storage | The main route avoids all-hours CPU UTCI storage for the visible selected-hour path; legacy/fallback/debug flows may still materialize CPU arrays. | `viewer/src/lib/compute/selected-hour/liveUtciSelectedHourSession.ts`, `liveUtciAnalysis.ts` |
| Rendering UTCI values | The main route can render from a GPU-native compute buffer surface (`compute-buffer-selected-hour`), while `dataTexture` remains as fallback/legacy path. | `viewer/src/lib/components/scene/UTCIPointCloud.svelte`, `viewer/src/lib/services/gpuUtciRenderBridge.ts`, `viewer/src/lib/services/pointCloudService.ts` |
| Debug route default | Plain `/debug` now defaults to on-demand `f32`; `?utciOnDemand=off` opts out, and `?collect=normal` preserves the old full-day collection harness. | `viewer/src/routes/debug/+page.svelte` |
| MRT diagnostics | Disabled by default; opt-in only when hardware supports enough storage buffers. | `webgpuUtciPipeline.ts` |

## 2026-05-15 Main-Route Selected-Hour Current-HEAD Baseline

Fresh main-route timing now lives in [docs/performance/main-route-selected-hour-current-head.md](performance/main-route-selected-hour-current-head.md).

Scope of that artifact:

- route: `/`, not `/debug`
- analyses: `Ben-Gurion/20250815_grid_2m_fullday` and `Ness-Tziona/exploded/nes_tziona_unblock_2` only
- proof boundary: `utciSurfaceSource=compute-buffer-selected-hour`, `baseRenderTransport=compute-buffer-selected-hour`, `dataTextureBuildCount=0`, `selectedHourRuntimeContract.strongVisibleGpuPath=true`
- debug/parity boundary: no `.bin`, Python, or debug comparison fields, and no forbidden comparison requests
- memory boundary: tracked app-owned UTCI/WebGPU buffers only (`persistentExposureBytes + allHoursOutputBytes + selectedHourOutputBytes + renderOwnedSelectedHourBytes` when render-owned bytes are published), not total browser/OS/device VRAM

### Fresh Main-Route Snapshot

| Metric | Ben-Gurion 2m | Ness Tziona 2m |
| --- | ---: | ---: |
| `pointCount` | 104,445 | 511,840 |
| `firstSelectedHourVisibleMs` | 1283.8 | 8465.4 |
| `exposurePrecomputeMs` | 708.3 | 6648.2 |
| `renderSceneSyncStartDelayMs` | 269.0 | 1140.1 |
| `renderSceneSyncTotalMs` | 80.9 | 340.1 |
| `oneHourDispatchMs` | 3.8 | 5.1 |
| `GPU VRAM` tracked app-owned memory | ~14.34 MiB | ~70.29 MiB |

### Current Main-Route Bottleneck Framing

Fresh `/` numbers shift the route-level timing picture in two important ways:

1. The main-route first-visible timings are the current route-level evidence, so the older 2026-05-09 debug-route table below should no longer be read as the current baseline for `/`.
2. The next optimization target is still cold-start work before first visible publication, with `exposurePrecomputeMs` dominating and `oneHourDispatchMs` staying tiny.

Important boundary: the fresh main-route capture currently leaves `payloadPrepareMs`, `workerBvhMs`, `pipelineUploadMs`, and `firstSelectedHourReadyMs` as unavailable (`null`) in the JSON artifact. We keep those fields null rather than inventing finer-grained values for `/`.

That means the current optimization inference is based first on the available 2026-05-15 main-route fields (`firstSelectedHourVisibleMs`, `exposurePrecomputeMs`, `renderSceneSyncStartDelayMs`, `renderSceneSyncTotalMs`, `oneHourDispatchMs`) and only secondarily on the older debug-route evidence below when discussing finer sub-buckets.

That is a useful narrowing, but it is **not** a 0.5m proof. This pass only says the current 2m main-route selected-hour path is still bottlenecked by cold-start compute/setup rather than selected-hour transport or `.bin` comparison.

## 2026-05-15 Main-Route 0.5m Stress Baseline

Fresh 0.5m main-route timing now lives in [docs/performance/main-route-selected-hour-0_5m-base.md](performance/main-route-selected-hour-0_5m-base.md), backed by [data/performance-results/main-route-selected-hour-0_5m-base.json](../data/performance-results/main-route-selected-hour-0_5m-base.json).

Scope of that artifact:

- route: `/`, not `/debug`
- query: `gridResolution=0.5&utciRender=auto&utciRenderDiagnostics=1`
- analyses: `Ben-Gurion/20250815_grid_2m_fullday` and `Ness-Tziona/exploded/nes_tziona_unblock_2`
- color modes: normalized/full-day and discrete/per-hour
- scrub sample: app-visible hour slider scrub from hour `0` to hour `1`
- proof boundary: `utciSurfaceSource=compute-buffer-selected-hour`, `baseRenderTransport=compute-buffer-selected-hour`, `dataTextureBuildCount=0`, `selectedHourRuntimeContract.strongVisibleGpuPath=true`
- debug/parity boundary: no `.bin`, Python, or debug comparison fields, and no forbidden comparison requests
- memory boundary: tracked app-owned UTCI/WebGPU buffers only, not total browser/OS/device VRAM

### 0.5m Main-Route Snapshot

| Metric | Ben-Gurion 0.5m | Ness Tziona 0.5m |
| --- | ---: | ---: |
| `pointCount` | 1,662,657 | 8,171,761 |
| `firstSelectedHourVisibleMs`, initial normalized | 8414.5 | 23883.8 |
| `firstSelectedHourVisibleMs`, scrub normalized | 986.9 | 4889.3 |
| `firstSelectedHourVisibleMs`, scrub discrete | 998.9 | 4707.8 |
| `exposurePrecomputeMs` | ~6040-6055 | ~16224-16266 |
| `oneHourDispatchMs`, scrub | 11.6-69.7 | 11.1-16.9 |
| `renderUpdateMs`, scrub | 916.8-987.1 | 4696.5-4871.8 |
| `renderSceneSyncStartDelayMs`, scrub | 707.9-785.1 | 3609.8-3665.4 |
| `renderSceneSyncTotalMs`, scrub | 201.8-208.6 | 1086.6-1206.0 |
| `GPU VRAM` tracked app-owned memory | ~228.33 MiB | ~1122.22 MiB |
| `renderOwnedSelectedHourBytes` | ~158.56 MiB | ~779.32 MiB |

### 0.5m Optimization Inference

The 0.5m route proves that the current selected-hour direction is memory-plausible on the tested machine, but it also shows that the user-visible bottleneck has shifted.

The UTCI selected-hour dispatch is not the main issue. At Ness Tziona 0.5m it stays around `11-17 ms` for the scrub samples, while the new hour takes about `4.7-4.9 s` to become visible. That gap points at render publication and scene synchronization, especially:

- render update time
- scene sync start delay
- scene sync total time
- render-owned selected-hour storage handling
- queue drain and first-use storage setup

At the time of this 0.5m stress baseline, the next inference was diagnostics-first render-publication work. The later init-smoothness attribution section supersedes that as the current next step: first rank early startup, exposure breathing, and render-publication freezes together. Still do not start by changing UTCI equations, color ramps, or `.bin`/debug comparison surfaces.

## 2026-05-31 Cooperative Scheduler And Manual Scrub Observations

The cooperative exposure scheduler evidence lives in [docs/performance/main-route-exposure-scheduler.md](performance/main-route-exposure-scheduler.md), backed by:

- [data/performance-results/main-route-visual-freeze-map.json](../data/performance-results/main-route-visual-freeze-map.json)
- [data/performance-results/main-route-cold-start-waterfall.json](../data/performance-results/main-route-cold-start-waterfall.json)

Current conclusion: promote chunked scheduling at `2048` max workgroups per slice as the product default. New performance collectors should compare chunked slice sizes rather than continuing to spend time on single-submit. The `2048` setting reduced the largest exposure scheduler queue wait to `381.3 ms`, but the top rAF gap stayed at `1569.1 ms`. The remaining top visual freeze overlaps initial render publication / scene sync for the dense Ness Tziona 0.5m surface, so smaller exposure slices alone should stay a measured spike rather than the next default.

The sub-2048 spike did not justify changing that default. Fresh chunked-only collectors showed `1024` lowered max scheduler queue wait from `424.6 ms` to `222.5 ms` and nudged top rAF gap from `1369.6 ms` to `1341.7 ms`, but first publication moved from `25671.5 ms` to `26447.2 ms`. The `512` case cut max queue wait to `118.8 ms` but moved first publication to `28319.9 ms` and did not beat `1024` on the visible freeze metrics. Keep `2048` as the default; treat `1024` as a tradeoff candidate only if a later product decision favors lower maximum queue waits over first-visible time.

Manual checking after the scheduler pass surfaced two open interaction observations that need a targeted diagnostic pass before any fix:

1. On Ness Tziona 0.5m, month changes are inconsistent in both normal and chunked modes. The same route can feel like about `90 ms`, `200 ms`, or multi-second publication. This should not be attributed to the exposure scheduler until a repeated month-change artifact splits `oneHourDispatchMs`, `firstSelectedHourVisibleMs`, `renderUpdateMs`, selected-hour session timing, and render-publication timeline fields for several consecutive month changes.
2. In chunked mode, arriving at Ness Tziona 0.5m after first loading Ben-Gurion can show around `200 ms` hour scrub in the Performance panel, while directly loading/reloading into Ness Tziona 0.5m can show around `70 ms`. The Performance panel's "Total calculation time" is `timings.firstSelectedHourVisibleMs`, not pure GPU compute time, so this difference may reflect warm state, route/session/cache history, selected-hour range/color-mode state, or render-publication reuse state rather than a UTCI compute regression.

These observations may be a bug, but they are not yet proven to be one. The proof collector should keep comparing direct-load NZ 0.5m, BG -> NZ 0.5m, default chunked `2048`, and explicit chunked slice-size variants in one bounded run. The collector should record repeated hour and month scrubs, plus the existing GPU-native proof boundary (`webgpu`, `compute-buffer-selected-hour`, same device, no visible selected-hour readback). If the same selection alternates between fast and multi-second runs while proof and cache state look identical, treat it as a scheduler/session invalidation bug. If the slow cases correlate with layout rebuild, cold render-owned storage setup, range scan, or selected-hour CPU materialization, keep it classified as a render-publication/session-state optimization target.

### 2026-05-31 Transition Scrub Diagnostic

Fresh targeted evidence now lives in [data/performance-results/main-route-transition-scrub-diagnostics.json](../data/performance-results/main-route-transition-scrub-diagnostics.json), collected on `/` with GPU-native proof preserved. Historical rows below include the old single-submit control, but the current collector uses chunked scheduling by default and explicit chunked `2048` cases.

Key result: the manual inconsistency is reproduced, but it does not point at the cooperative exposure scheduler. The slow hour-1 and month-8 samples are dominated by render publication / scene sync delay. Later hour scrubs are fast once render layout reuse is safe.

| Case | Initial visible | Hour 1 visible | Hour 2 visible | Month 8 visible | Month 0 visible |
| --- | ---: | ---: | ---: | ---: | ---: |
| Direct NZ, single-submit | `20929 ms` | `651 ms` | `117 ms` | `1907 ms` | `87 ms` |
| BG -> NZ, single-submit | `21386 ms` | `864 ms` | `158 ms` | `2239 ms` | `110 ms` |
| Direct NZ, chunked 2048 | `22033 ms` | `638 ms` | `111 ms` | `1929 ms` | `82 ms` |
| BG -> NZ, chunked 2048 | `22487 ms` | `878 ms` | `159 ms` | `2308 ms` | `111 ms` |

Interpretation:

- `oneHourDispatchMs` stays small for the interaction samples, about `11-21 ms`.
- Hour 1 is slower because render layout reports `build-required:canonical-mismatch`; hour 2 and hour 3 then drop to about `104-159 ms` visible with `reused:reuse-safe`.
- Month 8 is slow in this collector because it is the sampled first uncached month after initial visibility. Manual checking indicates this is not unique to month 8: the first visit to an uncached month can take seconds, while returning to the same month is usually around `100 ms`.
- Direct vs BG -> NZ adds a modest penalty, but not a new class of failure: BG -> NZ hour 2/3 stays around `134-159 ms`, and the slow month-8 pattern appears in both entry paths.
- Chunked 2048 does not materially improve or hurt these post-visible scrub/month timings. Its benefit remains visual responsiveness during exposure cold start, not render publication.

The first-scrub issue was fixed by the hot publication follow-up below. Before the compact range-summary work, the month-change issue still needed to be framed as **first-time uncached month publication cost**, upstream of scene sync queue/start, rather than as a month-8-specific issue or a scene duplicate-sync problem.

#### Hot Publication Fix Follow-Up

After the selected-hour publication hot-path change, the pre-compact refreshed `main-route-transition-scrub-diagnostics.json` showed:

| Case | Hour 1 visible | Hour 1 layout/proof | Hour 2 visible | Month 8 visible | Month 8 start delay | Scene queued split |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| Direct NZ, single-submit | `185 ms` | `reused:refreshed-proof-safe`, `refreshed-runtime-proof` | `140 ms` | `1917 ms` | `1873 ms` | `0.0 / 0.3 ms` |
| BG -> NZ, single-submit | `240 ms` | `reused:refreshed-proof-safe`, `refreshed-runtime-proof` | `193 ms` | `2680 ms` | `2627 ms` | `0.0 / 0.1 ms` |
| Direct NZ, chunked 2048 | `110 ms` | `reused:refreshed-proof-safe`, `refreshed-runtime-proof` | `106 ms` | `2272 ms` | `2227 ms` | `0.0 / 0.1 ms` |
| BG -> NZ, chunked 2048 | `150 ms` | `reused:refreshed-proof-safe`, `refreshed-runtime-proof` | `153 ms` | `2316 ms` | `2267 ms` | `0.0 / 0.3 ms` |

Conclusion:

- First post-visible hour scrub no longer rebuilds layout. It uses a full safe refreshed proof and lands in the same band as later hour scrubs.
- The sampled month-8 stall remained in this pre-compact artifact, but manual testing showed the broader pattern was first visit to an uncached month: once that month was warm, returning to it was fast.
- The new split fields show `sceneReactiveToSyncQueuedMs` and `sceneSyncQueuedToStartMs` are tiny, so the delay is not inside the scene reactive block after it observes the pending surface.
- The next month-change investigation needed to trace upstream of `pendingRenderUpdateStartedAt` / route selected-hour publication, especially per-month/day range resolution or selected-hour session cache warm-up. Do not add duplicate scene-sync queueing changes unless new evidence points back to that layer.
- Scheduler chunking is now promoted as the default cold-start exposure scheduler at `2048`; the hot publication fix remains independent of that default.

#### First-Time Month Range Investigation

The pre-compact follow-up collector sequence sampled first explicit visits and returns for months `8`, `0`, and `1`. Labels used `first` and `return`; cache state was reported separately by `sessionSelectedDayRangeCacheHit`.

That result confirmed the expensive bucket was upstream selected-day range resolution before compact range summaries:

- First explicit visits to uncached month `8` and month `1` spent about `1.8-2.4 s` in `sessionRangeResolveStartedAtMs -> sessionRangeResolveCompletedAtMs`, matching the visible delay.
- Each uncached sampled month reported `sessionSelectedDayRangeCacheHit=false`, `sessionSelectedDayRangeReadbackCount=23`, and `sessionSelectedDayRangeComputedHourCount=23`.
- Returning to month `8` and month `1` reported `sessionSelectedDayRangeCacheHit=true`, `readbackCount=0`, `computedHourCount=0`, range resolve around `0 ms`, and visible time around `74-118 ms`.
- Month `0` was already warm by the time the explicit month sequence reached it, so its first sampled change also reported a cache hit and stayed around `74-105 ms`.
- GPU-native proof remained intact in that artifact: main route `/`, `rendererBackend=webgpu`, `compute-buffer-selected-hour`, same compute/render device, and no visible selected-hour readback fallback.

Interpretation at that point: this was a per-session selected-day range cache warm-up. The slow first visit computed/read back the remaining `23` hours for the normalized display range; the return visit reused the cached range. It was not a cooperative exposure scheduler bug, and it was not month-8-specific. The next optimization target was selected-day range publication, with compact GPU reduction as the preferred direction now covered by the follow-up below.

#### GPU Compact Range Summary Follow-Up

Evidence was refreshed with:

```powershell
cd viewer
npx playwright test --config=playwright.collect.config.ts tests/e2e/main-route-transition-scrub-diagnostics.spec.ts --project=chromium --workers=1 --reporter=list --timeout=900000
```

The follow-up artifact keeps the proof surface on `/` and preserves the visible GPU path: `rendererBackend=webgpu`, `compute-buffer-selected-hour`, same compute/render device, and `visibleSelectedHourReadbackCount=0`. The artifact parser reported `badProof=0`, `forbidden=0`, `missingCompact=0`, and `uncachedMonthProofCount=8` across `4` cases and `40` samples.

Before this change, first visits to uncached months spent about `1.8-2.4 s` resolving the selected-day range and performed `23` full selected-hour value readbacks. The refreshed compact-summary run shows the uncached range-resolution bucket at `280-393 ms` (`343 ms` average). Each uncached month proof reports `sessionSelectedDayRangeResolutionPath=compact-gpu-summary`, `sessionSelectedDayRangeReadbackCount=0`, `sessionSelectedDayRangeSummaryReadbackCount=23`, `sessionSelectedDayRangeSummaryReadbackBytes=368`, and `sessionSelectedDayRangeFullReadbackAvoidedCount=23`.

Cache-hit returns remain `sessionSelectedDayRangeResolutionPath=cache-hit` with `readbackCount=0`, `computedHourCount=0`, and `summaryReadbackBytes=0`; measured range-resolution time is `0 ms` in the refreshed artifact. Wall-visible month-change time still varies (`526-1278 ms` for uncached compact months and `164-998 ms` for returns), so there is remaining non-range publication time to investigate after this fix.

Residual risks:

- GPU reduction ordering can differ from the old CPU scan by a small floating-point epsilon, though the compact parity proof covers mixed, equal, invalid, and all-invalid fixtures.
- The selected-day range bottleneck is reduced but not the only month-change cost; remaining publication/layout/storage timing should be treated as a separate follow-up.

### 2026-06-01 Cold Initial Prepared-Layout Attempt

A cold initial render-publication attempt prepared the selected-hour compute-buffer render layout before scene publication and attached it to the GPU-resident selected-hour result. The implementation was removed after the proof gate failed; keep the conclusion, not the unused runtime path.

The headed `/` collector for Ness Tziona `0.5m` showed the intended bucket improvement:

- `preparedRenderLayoutStatus=used`
- `rendererBackend=webgpu`
- `utciSurfaceSource=compute-buffer-selected-hour`
- `baseRenderTransport=compute-buffer-selected-hour`
- `baseSameDeviceForComputeAndRender=true`
- `visibleSelectedHourReadbackCount=0`
- `renderPublicationPreStorageMs` dropped to about `515-517 ms`, meeting the `<600 ms` target

But the user-visible freeze did not improve. The largest render-publication-overlapped rAF gap stayed around `1342 ms`, and the matching interval gap stayed around `1395 ms`, above the `<800 ms` target and broadly in the previous range.

The artifact explains the miss: the prepared layout build moved out of the named scene-side pre-storage bucket, but not out of the blocking main-thread window. The largest render-overlapped rAF gap began while `preparedLayoutBuild` was still running, then continued through scene publication:

| Bucket | Approx. duration |
| --- | ---: |
| Prepared layout build | `547 ms` |
| Scene/key/mesh pre-storage work | `517 ms` |
| Surface mesh creation | `268 ms` |
| Render-owned storage wait | `275 ms` |
| Copy queue drain | `379 ms` |

Inside `renderSurfaceMeshMs`, the expensive work was typed-array construction rather than Three.js object creation: position fill about `78 ms`, index fill about `127 ms`, and cell-to-point fill about `54 ms`.

Conclusion: moving layout construction earlier inside the same synchronous selected-hour publication path can make `renderPublicationPreStorageMs` look better, but it does not reduce the felt rAF freeze because the work remains on the main thread immediately before scene publication. Do not reintroduce prepared-layout plumbing as a performance fix unless the design makes the work genuinely non-blocking or removes it from the cold visible path.

### 2026-06-01 Three.js GPU-Driven Publication Research

A follow-up architecture review considered whether the viewer should imitate fully GPU-driven renderers where custom render setup is expressed as compute shaders rather than CPU callbacks.

Conclusion: keep Three.js as the scene/render owner. The useful direction is not replacing Three.js with a custom renderer; it is making the UTCI overlay behave more like a stable GPU resource whose values and small uniforms change, rather than a freshly published render object with expensive synchronous setup.

The repo is already GPU-native in the important selected-hour sense:

- UTCI selected-hour compute runs on WebGPU.
- The visible route can render from `compute-buffer-selected-hour`.
- The proven hot path keeps `visibleSelectedHourReadbackCount=0`.
- The proven hot path keeps `dataTextureBuildCount=0`.
- The material path is already storage-buffer/uniform shaped: UTCI values, cell-to-point mapping, range uniforms, and LUT sampling are expressed through Three WebGPU/TSL nodes.

The remaining architectural knot is render publication and resource ownership, not JavaScript render callbacks. Today the compute output is still synchronized into a Three-owned render storage buffer. The scene path may need to wait for Three storage initialization, copy the selected-hour compute output into render-owned storage, drain or observe queue completion, and publish visibility through Svelte/Three scene state. Dense geometry and cell mapping are also still CPU-created when a new render surface is built.

That means "more GPU-driven" should be interpreted narrowly:

| Direction | Keep | Avoid |
| --- | --- | --- |
| Stable Three-managed overlay | Three.js scene ownership, camera integration, fallbacks, diagnostics | Replacing the renderer as a first move |
| GPU-resident values and small uniforms | selected-hour UTCI buffer, range uniforms, LUT, opacity, grid constants | full selected-hour readback or `DataTexture` rebuilds |
| Publication lifecycle reduction | reuse/prewarm/directly bind resources when evidence says it removes a measured bucket | moving the same synchronous work to a different immediately adjacent bucket |
| Shader responsibility | data-parallel color/value/mapping work | month/hour selection, cache identity, fallback decisions, invalidation semantics |

Guardrails for the downstream publication plan, after init smoothness attribution decides render publication is the right target:

- Do not chase "GPU-driven" as a slogan. Each candidate must name the measured bucket it removes: layout/key/proof, mesh creation, storage init wait, buffer copy, queue drain, scene sync delay, or adjacent pre-scene publication work.
- Do not reintroduce hidden readbacks. Preserve `compute-buffer-selected-hour`, same-device proof, `visibleSelectedHourReadbackCount=0`, and `dataTextureBuildCount=0`.
- Do not make publication depend on fresh dense mesh/storage setup per interaction when the layout can be reused.
- Do not treat diagnostics/parity surfaces as the product route.
- Do not move product control-plane semantics into WGSL unless the data is truly parallel and the measurement says it matters.
- Do not call a bucket improvement successful unless the top rAF/interval gap and first-visible/first-scrub publication metrics move.

If attribution confirms render publication as the right next target, the likely high-value boundary is resource ownership between the WebGPU compute pipeline and Three's WebGPU backend: either find a supported way to bind/reuse the selected-hour GPU resource with less copy/sync churn, or prove that the copy/storage lifecycle is not the limiting bucket and target the adjacent CPU publication work instead.

### 2026-06-01 Init Smoothness And rAF Attribution Correction

A follow-up review of the existing rAF artifacts corrected the framing again: rAF is a freeze detector and phase-correlation tool, not the whole performance goal.

The current `data/performance-results/main-route-exposure-and-raf-diagnostics.json` artifact for `ness-tziona-0_5m` reports `rafGapCount=76`, but stores only the top rAF gaps rather than the complete frame distribution. The stored top gaps still contain enough timestamp data for first-pass attribution:

| Gap | Window | Duration | Overlap |
| --- | --- | ---: | --- |
| Top overall rAF gap | `4314.5 -> 5663.2 ms` | `1348.7 ms` | no exposure-slice or render-publication overlap |
| Top overall interval gap | `4284.3 -> 5669.5 ms` | `1385.2 ms` | no exposure-slice or render-publication overlap |
| Top overall long task | `4310.3 -> 5666.3 ms` | `1356 ms` | no exposure-slice or render-publication overlap |
| Largest render-publication rAF gap | `23613.8 -> 24927.9 ms` | `1314.1 ms` | `renderPublicationTotal`, `renderPublicationPreStorage`, `renderStorageFirstWaitFrame`, `renderStorageWait` |
| Render-publication tail rAF gap | `24934.9 -> 25261.5 ms` | `326.6 ms` | `renderCopyQueueDrain` |
| Largest exposure-overlapped rAF gaps | exposure slices `27-32` | about `326-355 ms` | exposure slices only |

This means the single largest page-local freeze in the current artifact is an early startup/pre-exposure stall, not render publication. Render publication is still a proven app-owned freeze near the end of the initial load, but it is not the only visible freeze owner. Exposure is the main wall-clock latency owner, but the current artifact does not prove the exposure phase is the largest page-local rAF freeze owner after chunking.

The next work should therefore be a **main-route init smoothness attribution pass**, not a direct optimization pass. It should answer two separate questions:

| Question | Metric family | Current owner candidates |
| --- | --- | --- |
| How long until the UTCI surface is visible? | `firstSelectedHourVisibleMs`, `pipelineFirstSelectedHourVisibleMs`, exposure, one-hour dispatch, render update, scene sync | full-grid exposure latency, first publication |
| Does the app visibly freeze while waiting? | rAF gaps, interval gaps, long tasks, input/paint proxies | early startup/pre-exposure stall, render publication, exposure-slice breathing |

Updated owner ranking:

1. **Total latency owner:** full-grid exposure for NZ `0.5m` remains the largest wall-clock cost (`exposurePrecomputeMs` around `17.2 s` in the current focused artifact). This is likely a mix of workload/HW reality and scheduling strategy.
2. **Largest measured contiguous page freeze:** early startup/pre-exposure work around `4.3 -> 5.7 s`, currently under-attributed.
3. **Known app-owned publication freeze:** render publication / scene sync around `23.6 -> 25.3 s`, with layout/pre-storage, storage wait, and queue drain sub-windows.
4. **Exposure roughness:** repeated exposure-overlapped rAF gaps around `326-355 ms`, which may represent GPU/driver pressure or page-local breath limits, but are smaller than the top startup/publication freezes.
5. **Non-owners for current init smoothness:** selected-hour UTCI dispatch, full visible-path readback, and `DataTexture` rebuilds remain ruled out by the current proof boundary.

Planning implication: do not pick the next optimization from the render-publication upper-bound table alone. First preserve or collect enough data to attribute the early startup/pre-exposure gap and to summarize the rAF distribution, not just the maximum. Render-publication optimization remains likely valuable, but it should be chosen after the init smoothness map shows whether it is the largest app-owned freeze, the easiest app-owned freeze, or simply one of several meaningful freezes.

### Future Analysis Boundary

Future all-hours histograms are feasible if they are implemented as async derived summaries, not as all-hours/all-points resident fields.

The safe shape is:

1. compute one selected hour or one tile/hour batch
2. reduce on GPU or in bounded tiles into bins/counts/min/max/threshold summaries
3. read back only the compact summary
4. persist the compact result
5. release or reuse the large selected-hour buffers

The unsafe shape is keeping all point values for all representative hours in browser memory. For Ness Tziona 0.5m, `8.17M points * 288 hours * f32` is roughly `9.4 GB` for one scalar field, before any render/model overhead. That remains outside the browser-product budget.

Point wind, scenario deltas, and richer climate layers should be treated as later product/data-contract decisions. They should not be allowed to pull the main route back into all-hours CPU readback or store-all browser memory.

## Historical 2026-05-09 Debug-Route Cold-Start Snapshot

This section is historical debug-route evidence, not the current route-level baseline for `/`.

The older legacy batch reports are still useful only as historical/parity evidence:

- Legacy run-all parity/performance: [data/batch-parity-results/parity_performance_report.md](../data/batch-parity-results/parity_performance_report.md)
- Run-all vs strict exposure-only on-demand: [data/batch-parity-results/parity_performance_report_run_all_vs_on_demand.md](../data/batch-parity-results/parity_performance_report_run_all_vs_on_demand.md)

But those reports do not capture the app-visible debug-route cold-start/render-path breakdown that was captured before the route was renamed from `/debug-webgpu-utci` to `/debug`. The focused timings below remain useful historical debug-route context for sub-buckets that the fresh `/` capture does not currently expose, but the 2026-05-15 main-route baseline above is the current route evidence for timing and memory decisions on `/`.

### Route Shape Used

- `http://127.0.0.1:5173/debug-webgpu-utci?analysis=Ben-Gurion%2F20250815_grid_2m_fullday&utciRender=gpu`
- `http://127.0.0.1:5173/debug-webgpu-utci?analysis=Ness-Tziona%2Fexploded%2Fnes_tziona_unblock_2&utciRender=gpu`

### Fresh Verification Anchors

- `pnpm vitest run tests/compute/onDemandDiagnostics.test.ts` -> `14/14 passed`
- `npx playwright test tests/e2e/webgpu-on-demand-prototype.spec.ts` -> `14/14 passed`

### Current Diagnostic Snapshot

Common proof from both BG and NZ runs:

- `utciRenderResolved=gpuNative`
- `renderTransport=compute-buffer-selected-hour`
- `utciSurfaceSource=compute-buffer-selected-hour`
- `sameDeviceForComputeAndRender=true`
- `selectedHourTransferCount=0`
- `selectedHourReadbackCount=0`
- `dataTextureBuildCount=0`

This means the selected-hour route is no longer bottlenecked by the old CPU readback/upload path. The remaining cold-start cost is elsewhere.

| Metric | Ben-Gurion 2m | Ness Tziona 2m |
| --- | ---: | ---: |
| `payloadPrepareMs` | 150.9 | 276.8 |
| `workerBvhMs` | 91.1 | 222.0 |
| `pipelineUploadMs` | 29.5 | 66.7 |
| `exposurePrecomputeMs` | 668.3 | 1466.5 |
| `oneHourDispatchMs` | 1.3 | 3.4 |
| `firstSelectedHourReadyMs` | 1146.3 | 2240.9 |
| `firstSelectedHourVisibleMs` | 1399.2 | 3120.3 |
| `renderUpdateMs` | 253.0 | 879.7 |
| `renderSceneSyncStartDelayMs` | 98.2 | 434.3 |
| `renderSceneSyncTotalMs` | 154.7 | 445.3 |
| `renderLayoutBuildMs` | 5.4 | 29.9 |
| `renderSurfaceMeshMs` | 37.7 | 189.1 |
| `renderStorageInitWaitMs` | 49.5 | 118.6 |
| `renderBufferCopyMs` | 0.0 | 0.0 |
| `renderQueueDrainMs` | 61.4 | 107.2 |

### Historical Debug-Route Bottleneck Framing

The measured cold-start cost is now split into two real buckets:

1. **Pre-scene-sync startup delay**
   - BG: about `98 ms`
   - NZ: about `434 ms`
2. **Scene sync itself**
   - BG: about `155 ms`
   - NZ: about `445 ms`

Inside scene sync, the largest measured contributors are:

- `renderSurfaceMeshMs`
- `renderStorageInitWaitMs`
- `renderQueueDrainMs`

The old unexplained `renderUpdateMs` gap is basically gone after adding the extra timing splits. We now have enough instrumentation to discuss optimization intentionally instead of guessing.

## Corrected Memory Footprint

Ness Tziona at 2m resolution is about 511,840 grid points and 288 representative hours.

### Current On-Demand Working Set

The current app-visible selected-hour path is no longer the old "store all UTCI/MRT hours and upload textures" design.

Focused route measurements now show this smaller working-set shape for the hot path, excluding BVH/model overhead and scene mesh/material overhead:

| Analysis | Persistent Exposure | Selected-Hour HWM | Notes |
| --- | ---: | ---: | --- |
| Ben-Gurion 2m | ~3.98 MiB | ~408.0 KiB | Strict exposure-only report baseline for the same grid scale. |
| Ness Tziona 2m | ~19.53 MiB | ~1999.4 KiB | Largest current 2m analysis grid; useful order-of-magnitude proxy for the selected-hour route. |

These figures are the important correction to the old mental model: the hot render path no longer requires a full all-hours UTCI CPU copy or selected-hour readback/upload when the `gpuNative` path is active.

### Legacy Run-All / Store-All Picture

The older all-hours design is still useful as historical contrast and as fallback context. Its GPU memory shape, excluding BVH/model overhead, looked like this:

| Buffer | Current format | Approx. size |
| --- | --- | ---: |
| Solar exposure | bit-packed `u32` | ~17.5 MB |
| Sky exposure | `f32` per point | ~2 MB |
| UTCI results | full `f32`, all point-hours | ~560 MB |
| MRT results | full `f32`, all point-hours | ~560 MB |
| Grid points | `vec3<f32>` | ~6 MB |
| Weather/sun vectors | small `f32` buffers | <1 MB |
| Total major buffers | without diagnostics | ~1.15 GB |

Legacy CPU memory shape:

| Buffer | Current format | Approx. size |
| --- | --- | ---: |
| Live UTCI storage | `Int16Array`, all point-hours | ~282 MB |
| Positions | `Float32Array`, all points | ~6 MB |
| CPU color/texture update temporaries | per visible slice/texture update | MB-scale per update |

The old warning that solar exposure alone costs ~560 MB is stale. The real issue in the run-all/store-all architecture was the all-hours UTCI/MRT buffers plus the compatibility CPU copy.

## 0.5m Planning Boundary

Moving from 2m to 0.5m increases point count by roughly 16x. Ness Tziona would move from about 512K points to about 8.2M points.

### Old Run-All / Store-All Reality

| Buffer | 0.5m, 8.2M points x 288 hours |
| --- | ---: |
| UTCI `f32`, all hours | ~9.4 GB |
| MRT `f32`, all hours | ~9.4 GB |
| CPU UTCI `Int16Array`, all hours | ~4.7 GB |

Under the old design, 0.5m was fundamentally blocked by all-hours storage, CPU compatibility copies, and the render path built around them. That remains true as historical contrast.

### Current Selected-Hour Direction

The current on-demand route changes the 0.5m question. The selected-hour debug path already proves that we can avoid hot-path selected-hour readback on the strong route. The remaining 0.5m question is no longer "can we remove selected-hour CPU transport?" It is "can we keep the persistent geometry/exposure data resident, compute one selected hour, and get the first visible render on screen quickly enough?"

Order-of-magnitude 0.5m shape for the current direction:

| Component | Old run-all/store-all | Current selected-hour direction |
| --- | ---: | ---: |
| Solar exposure bitmask | ~295 MB | ~295 MB persistent |
| Sky exposure | ~33 MB | ~33 MB persistent |
| UTCI output | ~9.4 GB all hours | ~33 MB for one selected hour |
| MRT output | ~9.4 GB all hours | ~33 MB for one selected hour if kept separately |
| CPU UTCI compatibility copy | ~4.7 GB | not on the hot path when `gpuNative` is active |

This is still a serious workload, but it is no longer obviously impossible in the same way as the old all-hours architecture.

### What 0.5m Depends On Now

For the current route, 0.5m is primarily gated by:

- persistent GPU memory for solar/sky data plus scene assets
- cold-start cost before first visible selected hour
- first-use render surface setup and synchronization cost
- target-device VRAM and weak-GPU behavior

So the bottleneck story has changed:

- **Old story:** memory blow-up from all-hours UTCI/MRT storage and CPU copies
- **Current story:** selected-hour memory is now proven plausible on the tested 0.5m main-route cases, but 0.5m scrub/render publication is still UX-poor.

Important caveat: we have collected the current main route at 0.5m and have now ruled out one narrow prepared-layout shift as an effective visible-freeze fix. The next evidence target is therefore not "prove 0.5m exists" or "move the same synchronous work earlier"; it is to attribute whether early startup/pre-exposure, exposure breathing, render publication, or another phase owns the most important visible freezes and total-latency costs.

## Recommended Next Step

The next step should be an intentional **main-route init smoothness attribution pass**, not another opportunistic optimization without a clear target.

Current planning artifact: [2026-06-01 main-route init smoothness attribution plan](superpowers/plans/2026-06-01-main-route-init-smoothness-attribution.md). That plan first preserves enough rAF/interval/long-task distribution data and early startup phase marks to rank total-latency owners separately from visible-freeze owners.

The older [2026-06-01 main-route cold publication next steps plan](superpowers/plans/2026-06-01-main-route-cold-publication-next-steps.md) remains useful after attribution, but it is now downstream of the init smoothness map. Do not choose a render-publication implementation only from the cold-publication upper-bound table until the early startup/pre-exposure gap is attributed.

Current objective:

1. Keep `/` as the canonical product proof route and keep `/debug` thin/proof-oriented.
2. Preserve or collect enough diagnostics to rank early startup/pre-exposure, exposure breathing, and render-publication freezes in the same artifact.
3. Separate total latency from visible freeze before choosing a fix.
4. Only after attribution, decide whether the first implementation should target startup/data prep, render publication, exposure scheduling/tiling, or a deliberately scoped combination with a single falsifiable gate.
5. Preserve `dataTexture` and other legacy/fallback paths until the selected-hour route is clearly good enough to replace them more broadly.

Why this is the right risk order:

| Option | What it proves | Why it is not first |
| --- | --- | --- |
| Web Worker quantization | Main thread can stay responsive while quantizing full readback. | It preserves full CPU readback/storage, so it does not explain the remaining selected-hour cold-start bottlenecks. |
| LRU decoded slices | Scrubbing can improve in the legacy CPU pipeline. | The current bottleneck is no longer selected-hour decoded-slice churn on the hot path. |
| `pack2x16float` | MRT+UTCI can use half the output-buffer memory. | It may still matter later, but the current measured pain is cold-start/render sync, not output-buffer size. |
| Cold-start/render-path analysis | We can choose the next optimization from measured route-level evidence. | This is the current highest-signal step. |

## Current On-Demand Route Architecture

This section is no longer a target-only sketch. It should describe the current route shape and how it differs from the old path.

### Old Path vs Current Path

| Aspect | Old run-all / store-all path | Current selected-hour path |
| --- | --- | --- |
| Compute scope | Many/all representative hours | One selected `timeIndex` |
| Main storage shape | All-hours UTCI/MRT buffers plus CPU compatibility storage | Persistent geometry/exposure inputs plus selected-hour output |
| Render transport | CPU-driven decode/color/`DataTexture` upload | `compute-buffer-selected-hour` when the `gpuNative` route is active |
| Selected-hour readback | Common architectural dependency | No visible selected-hour readback on the proven GPU-native hot path |
| Route status | Legacy/fallback/collection context still exists | Plain `/debug` now defaults here |

### Current Route Shape

Persistent geometry-dependent state:

| Data | Lifetime | Format / role |
| --- | --- | --- |
| BVH buffers | persistent | scene-owned geometry acceleration data |
| Grid points / layout | persistent | per-point or per-cell spatial layout |
| Solar exposure | persistent | bit-packed `u32` |
| Sky exposure | persistent | `f32` per point |
| Weather and sun vectors | persistent | small per-analysis inputs |

Selected-hour path:

1. Resolve month/hour into one selected `timeIndex`.
2. Dispatch MRT/UTCI for that selected hour only.
3. Keep the selected-hour result on the GPU when the `gpuNative` route is active.
4. Sync the scene-owned UTCI surface from that selected-hour compute result.
5. Render without selected-hour CPU readback or `DataTexture` rebuilds on the proven hot path.

Fallback and legacy paths still intentionally remain:

- `?utciOnDemand=off`
- `?collect=normal`
- `dataTexture` / CPU-oriented fallback behavior

### What The Measurements Now Say About This Architecture

The current route has already cleared the earlier proof hurdle:

- `renderTransport=compute-buffer-selected-hour`
- `utciSurfaceSource=compute-buffer-selected-hour`
- `selectedHourReadbackCount=0`
- `dataTextureBuildCount=0`

The remaining cost is now centered on:

- `exposurePrecomputeMs`
- pre-scene-sync startup delay
- scene sync itself, especially `renderSurfaceMeshMs`, `renderStorageInitWaitMs`, and `renderQueueDrainMs`

So the architecture question has shifted from "can we avoid hot-path readback?" to "how cheaply can we bring the first selected-hour surface on screen?"

## Deferred Packed Output Option

This is no longer the first optimization to chase, and it is not part of the currently proven route.

Use `f32` for all canonical computation and validation. Do not use `f16` arithmetic for the UTCI polynomial or MRT intermediate math.

Keep `pack2x16float` as an optional later output-storage optimization if 0.5m memory/bandwidth becomes the dominant bottleneck again:

```wgsl
let packed = pack2x16float(vec2<f32>(mrt, utci));
output_values[point_idx] = packed;
```

Reasons to prefer `pack2x16float` over `shader-f16` storage:

| Point | Implication |
| --- | --- |
| `pack2x16float` is a core WGSL builtin. | No `shader-f16` device feature is required. |
| It packs two `f32` values into one `u32`. | MRT and UTCI can share one 32-bit output word. |
| Arithmetic remains `f32`. | The sensitive UTCI polynomial stays stable. |
| It adds pack/unpack and half precision quantization. | Validate max error and stress-category flip rate before using it as the default analytic output. |

If we revisit output packing, the clean comparison is still these two variants behind a flag:

| Variant | Output | Purpose |
| --- | --- | --- |
| A | one `f32` UTCI buffer | simplest bridge and baseline performance |
| B | one packed `u32` MRT+UTCI buffer | validates memory/bandwidth win and visual precision |

Decision rule: keep `f32` unless memory/bandwidth pressure clearly becomes the blocker and packed output proves numerically/visually safe.

## What Worker/LRU Still Means

Worker and LRU work is not wrong; it is just no longer the leading architecture.

Keep it as:

| Use case | Role |
| --- | --- |
| No WebGPU / WebGPU failure | CPU fallback path. |
| Export/statistics/picking | Optional readback/CPU summary path. |
| Legacy/compatibility route | Short-term mitigation where the selected-hour GPU-native path is not active yet. |
| Future tiling | A tile lifecycle cache may still be useful, but for GPU-resident tiles rather than decoded CPU UTCI slices. |

Do not spend the next major effort on CPU decoded-slice LRU unless the selected-hour GPU-native path stalls out or a fallback path urgently needs stabilization.

## Next Investigation Plan

This plan is attribution-first. Render-publication optimization candidates remain downstream until the init smoothness attribution pass explains the early startup/pre-exposure gap and ranks total-latency owners separately from visible-freeze owners.

1. **Preserve the measurement baseline**
   - Keep using the main route `/` for BG/NZ 2m and 0.5m route-level proof.
   - Keep route-level proof visible: `compute-buffer-selected-hour`, no visible selected-hour readback, zero `dataTexture` rebuilds.

2. **Map init smoothness across phases**
   - Preserve or collect enough rAF/interval/long-task distribution data to distinguish one maximum gap from repeated roughness.
   - Add or preserve phase marks before exposure starts so the `4314.5 -> 5663.2 ms` startup/pre-exposure stall is no longer anonymous.
   - Keep exposure slice windows and render-publication windows in the same artifact so overlap evidence stays comparable.

3. **Rank latency and freeze owners separately**
   - Treat full-grid exposure as the current total-latency owner unless fresh evidence changes it.
   - Treat early startup/pre-exposure as the currently largest measured page-local freeze until it is attributed or falsified.
   - Treat render publication as a proven app-owned freeze near the end of init, not necessarily the only or first fix target.

4. **Use the prepared-layout failure as a downstream constraint**
   - Do not keep or reintroduce unused prepared-layout runtime code.
   - Do not count a bucket-name improvement as success if the rAF/interval gap does not move.
   - Treat synchronous work immediately before scene publication as still part of the felt freeze, even when it is owned by selected-hour session assembly rather than scene sync.

5. **Only after attribution, rank implementation candidates**
   - Treat typed-array mesh/layout construction as important evidence, not as an automatic next implementation. `renderSurfaceMeshMs` was about `268 ms`, which is meaningful but too small by itself to explain a `1342 ms` gap against an `<800 ms` target.
   - Treat the synchronous layout/prep work adjacent to publication as part of the same felt freeze even when it is not inside the scene-side bucket.
   - Treat render-owned storage readiness as a separate suspect: the failed attempt still showed a one-frame storage wait around `275 ms`.
   - Treat queue drain as a separate suspect: the failed attempt still produced a later render-publication-overlapped gap around `361 ms`.
   - Prefer candidates whose measured upper bound can move the chosen product gate, whether that gate is early startup freeze, render-publication freeze, total first-visible latency, or a combination.

## Open Risks And Boundaries

| Risk | Probe |
| --- | --- |
| We optimize the wrong bucket first. | Keep BG/NZ timing splits current and choose changes from the measured breakdown, not intuition. |
| The large remaining NZ cost may be split across both pre-scene-sync delay and scene sync. | Keep `renderSceneSyncStartDelayMs` and `renderSceneSyncTotalMs` as separate live suspects in the next design pass. |
| 0.5m is measured but still UX-poor on init smoothness. | Do not treat 0.5m as product-smooth until the route is recollected after init smoothness attribution and the measured bottlenecks are ranked. |
| Capability/status reporting can still mislead future debugging. | Keep trusting stronger runtime proof over weak `navigatorGpu`/overlay capability checks. |
| Solar bitmask at 0.5m is still hundreds of MB. | Add spatial tiling after bridge proof; keep BVH persistent and tile exposure/results. |
| GPU-only values are inconvenient for charts, exports, and picking. | Add small targeted readbacks for summaries/picked cells, not full-field readback. |
| Packed output changes classifications near UTCI thresholds. | Track stress-category flip rate, not just numeric RMSE. |
| WebGPU browser/device support remains uneven. | Keep current CPU/bin-backed viewer path as fallback. |

## Source Notes

- Three.js `StorageBufferAttribute` is intended for compute-generated buffer data and is only usable with `WebGPURenderer`: <https://threejs.org/docs/pages/StorageBufferAttribute.html>
- Three.js `StorageTexture` is available for compute-generated texture-style outputs under the WebGPU renderer: <https://threejs.org/docs/pages/StorageTexture.html>
- WGSL `pack2x16float` converts two `f32` values to binary16 and packs them into one `u32`: <https://gpuweb.github.io/gpuweb/wgsl/#pack2x16float-builtin>
- `shader-f16` is a WebGPU feature that must be supported/requested before using `f16` arithmetic in WGSL: <https://developer.mozilla.org/en-US/docs/Web/API/GPUSupportedFeatures>
- WebGPU buffer readback requires staging buffers and `mapAsync`, so it should stay off the hot render path: <https://developer.mozilla.org/en-US/docs/Web/API/GPUBuffer>
- WebGPU limits should be queried on the adapter/device and treated as target-device constraints: <https://developer.mozilla.org/en-US/docs/Web/API/GPUSupportedLimits>
