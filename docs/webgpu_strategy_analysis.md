# WebGPU Strategy Analysis

Updated: 2026-05-31

> **2026-05-11 route-name note:** The active debug route in this checkout is now `/debug` at `viewer/src/routes/debug/+page.svelte`. Older references to `/debug-webgpu-utci` describe the same debug/parity role before the route rename and should not be copied into new execution plans.

## Decision Snapshot

We should keep the selected-hour **WebGPU compute-on-demand path** as the main direction and make the next decisions from measured cold-start/render-path evidence.

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
| UTCI readback | The selected-hour on-demand debug path no longer reads back UTCI values on the hot render path; bulk readback still exists for legacy/fallback/export flows. | `liveUtciAnalysis.ts`, `viewer/src/routes/debug/+page.svelte`, `viewer/src/lib/compute/gpu/webgpuUtciPipeline.ts` |
| CPU UTCI storage | Legacy live analysis still creates a full time-major `Int16Array` copy, but the selected-hour on-demand debug route avoids all-hours CPU UTCI storage as the main path. | `liveUtciAnalysis.ts`, `viewer/src/routes/debug/+page.svelte` |
| Rendering UTCI values | The current selected-hour debug route can render from a GPU-native compute buffer surface (`compute-buffer-selected-hour`), while `dataTexture` remains as fallback/legacy path. | `viewer/src/lib/components/scene/UTCIPointCloud.svelte`, `viewer/src/lib/services/gpuUtciRenderBridge.ts`, `viewer/src/lib/services/pointCloudService.ts` |
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

The next main-route optimization should therefore be diagnostics-first render publication work. Do not start by changing UTCI equations, color ramps, or `.bin`/debug comparison surfaces.

## 2026-05-31 Cooperative Scheduler And Manual Scrub Observations

The cooperative exposure scheduler evidence lives in [docs/performance/main-route-exposure-scheduler.md](performance/main-route-exposure-scheduler.md), backed by:

- [data/performance-results/main-route-visual-freeze-map.json](../data/performance-results/main-route-visual-freeze-map.json)
- [data/performance-results/main-route-cold-start-waterfall.json](../data/performance-results/main-route-cold-start-waterfall.json)

Current conclusion: keep chunked scheduling query-gated. The `2048` setting reduced the largest exposure scheduler queue wait to `381.3 ms`, but the top rAF gap stayed at `1569.1 ms`. The remaining top visual freeze overlaps initial render publication / scene sync for the dense Ness Tziona 0.5m surface, so smaller exposure slices alone are not the default-ready answer.

Manual checking after the scheduler pass surfaced two open interaction observations that need a targeted diagnostic pass before any fix:

1. On Ness Tziona 0.5m, month changes are inconsistent in both normal and chunked modes. The same route can feel like about `90 ms`, `200 ms`, or multi-second publication. This should not be attributed to the exposure scheduler until a repeated month-change artifact splits `oneHourDispatchMs`, `firstSelectedHourVisibleMs`, `renderUpdateMs`, selected-hour session timing, and render-publication timeline fields for several consecutive month changes.
2. In chunked mode, arriving at Ness Tziona 0.5m after first loading Ben-Gurion can show around `200 ms` hour scrub in the Performance panel, while directly loading/reloading into Ness Tziona 0.5m can show around `70 ms`. The Performance panel's "Total calculation time" is `timings.firstSelectedHourVisibleMs`, not pure GPU compute time, so this difference may reflect warm state, route/session/cache history, selected-hour range/color-mode state, or render-publication reuse state rather than a UTCI compute regression.

These observations may be a bug, but they are not yet proven to be one. The next proof should compare direct-load NZ 0.5m, BG -> NZ 0.5m, normal, and `utciExposureSchedule=chunked&utciExposureMaxWorkgroupsPerSlice=2048` in one bounded collector. The collector should record repeated hour and month scrubs, plus the existing GPU-native proof boundary (`webgpu`, `compute-buffer-selected-hour`, same device, no visible selected-hour readback). If the same selection alternates between fast and multi-second runs while proof and cache state look identical, treat it as a scheduler/session invalidation bug. If the slow cases correlate with layout rebuild, cold render-owned storage setup, range scan, or selected-hour CPU materialization, keep it classified as a render-publication/session-state optimization target.

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

The older batch reports are still useful supporting evidence:

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
| Ness Tziona 2m | ~19.53 MiB | ~1999.4 KiB | Largest current batch grid; useful order-of-magnitude proxy for the selected-hour route. |

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

Important caveat: we have collected the current main route at 0.5m, but we have **not** recollected it after the planned render-publication diagnostics pass. The next evidence target is therefore not "prove 0.5m exists"; it is "explain and rank the measured 0.5m render-publication bottlenecks."

## Recommended Next Step

The next step should be an intentional **main-route render publication diagnostics pass** based on the measured 0.5m bottlenecks, not another opportunistic optimization without a clear target.

Current planning artifact: [2026-05-15 main-route render diagnostics plan](superpowers/plans/2026-05-15-main-route-render-diagnostics.md).

Current objective:

1. Keep `/` as the canonical product proof route and keep `/debug` thin/proof-oriented.
2. Add deeper modular diagnostics around the render publication path without turning `viewer/src/routes/+page.svelte` back into a debug shell.
3. Explain the multi-second 0.5m scrub gap between selected-hour compute completion and visible render publication.
4. Decide whether the first optimization should be render-owned storage reuse, layout/mesh reuse, queue-drain scheduling, scene sync handoff, or tiling.
5. Preserve `dataTexture` and other legacy/fallback paths until the selected-hour route is clearly good enough to replace them more broadly.

Historical plan trail for this route:

- Prototype implementation plan: [docs/superpowers/plans/2026-05-07-webgpu-compute-on-demand-prototype.md](superpowers/plans/2026-05-07-webgpu-compute-on-demand-prototype.md)
- Prototype results: [docs/superpowers/plans/2026-05-07-webgpu-compute-on-demand-prototype-results.md](superpowers/plans/2026-05-07-webgpu-compute-on-demand-prototype-results.md)
- F32 vertical-slice follow-up plan: [docs/superpowers/plans/2026-05-08-webgpu-f32-on-demand-vertical-slice.md](superpowers/plans/2026-05-08-webgpu-f32-on-demand-vertical-slice.md)
- Debug on-demand integration plan: [docs/superpowers/plans/2026-05-08-webgpu-on-demand-debug-integration.md](superpowers/plans/2026-05-08-webgpu-on-demand-debug-integration.md)

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
| Selected-hour readback | Common architectural dependency | Zero on the proven hot path |
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

## Next Investigation And Optimization Plan

1. **Preserve the measurement baseline**
   - Keep using the main route `/` for BG/NZ 2m and 0.5m route-level proof.
   - Keep route-level proof visible: `compute-buffer-selected-hour`, zero selected-hour readback, zero `dataTexture` rebuilds.

2. **Map the render publication gap**
   - Trace the time from selected-hour output acceptance to scene sync start.
   - Split mesh create vs mesh reuse, layout extraction, render-owned storage wait, buffer copy, queue drain, visibility publication, and invalidation.
   - Decide which parts are one-time setup, repeated scrub work, or unavoidable waits.

3. **Map the scene-sync path**
   - Treat `renderSurfaceMeshMs`, `renderStorageInitWaitMs`, and `renderQueueDrainMs` as the current measured scene-sync suspects.
   - Decide whether the likely win is reuse, prewarm, caching, synchronization changes, or something else.

4. **Prioritize boring/high-confidence optimization candidates**
   - Reuse or cache work that is currently front-loaded into the first selected-hour render.
   - Prewarm scene-owned GPU surface/storage setup if that cost is mostly first-use overhead.
   - Reduce explicit first-use synchronization / queue drain where it is not actually required.
   - Avoid repeated payload/BVH/upload work when the selected analysis has not changed.

5. **Remeasure before optimizing**
   - Recollect BG and NZ 0.5m after the diagnostics pass.
   - Produce a short evidence note that ranks the suspected render buckets.
   - Only then choose the first behavior-changing optimization.

## Open Risks And Boundaries

| Risk | Probe |
| --- | --- |
| We optimize the wrong bucket first. | Keep BG/NZ timing splits current and choose changes from the measured breakdown, not intuition. |
| The large remaining NZ cost may be split across both pre-scene-sync delay and scene sync. | Keep `renderSceneSyncStartDelayMs` and `renderSceneSyncTotalMs` as separate live suspects in the next design pass. |
| 0.5m is measured but still UX-poor on scrub/render publication. | Do not treat 0.5m as product-smooth until the route is recollected after render-publication diagnostics and the measured bottlenecks are ranked. |
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
