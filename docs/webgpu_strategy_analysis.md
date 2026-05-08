# WebGPU Strategy Analysis

Updated: 2026-05-08

## Decision Snapshot

We should leapfrog the Web Worker + decoded-slice LRU work as the main path and move next to a narrow **WebGPU compute-on-demand prototype**.

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

Implemented today:

| Capability | Current state | Key files |
| --- | --- | --- |
| Three.js WebGPU rendering | `WebGPURenderer` is already used. | `viewer/src/lib/components/scene/Scene.svelte` |
| WebGPU compute pipeline | Solar exposure, sky exposure, and MRT/UTCI compute already run on WebGPU. | `viewer/src/lib/compute/webgpuUtciPipeline.ts` |
| Solar exposure storage | Already bit-packed as one bit per point-hour in a `u32` buffer. | `webgpuUtciPipeline.ts`, `shaders/exposure_solar.wgsl`, `shaders/mrt_utci.wgsl` |
| UTCI readback | Bulk readback exists; the old 288 `mapAsync` readback loop is no longer the default path. | `liveUtciAnalysis.ts`, `webgpuUtciPipeline.ts` |
| CPU UTCI storage | Live analysis still creates a full time-major `Int16Array` copy for compatibility with the existing analysis/render pipeline. | `liveUtciAnalysis.ts` |
| Rendering UTCI values | Still CPU-driven: decode values, create colors, fill a `DataTexture`, upload texture to GPU. | `UTCIPointCloud.svelte`, `pointCloudService.ts`, `dataLoader.ts` |
| MRT diagnostics | Disabled by default; opt-in only when hardware supports enough storage buffers. | `webgpuUtciPipeline.ts` |

## Corrected Memory Picture

Ness Tziona at 2m resolution is about 511,840 grid points and 288 representative hours.

Current GPU memory shape, excluding BVH/model overhead:

| Buffer | Current format | Approx. size |
| --- | --- | ---: |
| Solar exposure | bit-packed `u32` | ~17.5 MB |
| Sky exposure | `f32` per point | ~2 MB |
| UTCI results | full `f32`, all point-hours | ~560 MB |
| MRT results | full `f32`, all point-hours | ~560 MB |
| Grid points | `vec3<f32>` | ~6 MB |
| Weather/sun vectors | small `f32` buffers | <1 MB |
| Total major buffers | without diagnostics | ~1.15 GB |

Current CPU memory shape:

| Buffer | Current format | Approx. size |
| --- | --- | ---: |
| Live UTCI storage | `Int16Array`, all point-hours | ~282 MB |
| Positions | `Float32Array`, all points | ~6 MB |
| CPU color/texture update temporaries | per visible slice/texture update | MB-scale per update |

The old warning that solar exposure alone costs ~560 MB is stale. The remaining large allocations are the all-hours UTCI and MRT buffers plus the compatibility CPU copy.

## 0.5m Target

Moving from 2m to 0.5m increases point count by roughly 16x. Ness Tziona would move from about 512K points to about 8.2M points.

Under the current compute-all/store-all design:

| Buffer | 0.5m, 8.2M points x 288 hours |
| --- | ---: |
| UTCI `f32`, all hours | ~9.4 GB |
| MRT `f32`, all hours | ~9.4 GB |
| CPU UTCI `Int16Array`, all hours | ~4.7 GB |

This is not a compression problem anymore. It is an architecture problem. The only credible path to 0.5m in-browser is to keep long-lived geometry-dependent data on the GPU and compute only the currently needed visible/hour buffer.

## Recommended Next Step

Build a small **GPU-resident render bridge prototype** before refactoring the full analysis pipeline.

Prototype implementation plan: [docs/superpowers/plans/2026-05-07-webgpu-compute-on-demand-prototype.md](superpowers/plans/2026-05-07-webgpu-compute-on-demand-prototype.md).
Prototype results: [docs/superpowers/plans/2026-05-07-webgpu-compute-on-demand-prototype-results.md](superpowers/plans/2026-05-07-webgpu-compute-on-demand-prototype-results.md).
F32 vertical-slice follow-up plan: [docs/superpowers/plans/2026-05-08-webgpu-f32-on-demand-vertical-slice.md](superpowers/plans/2026-05-08-webgpu-f32-on-demand-vertical-slice.md).
Debug on-demand integration plan: [docs/superpowers/plans/2026-05-08-webgpu-on-demand-debug-integration.md](superpowers/plans/2026-05-08-webgpu-on-demand-debug-integration.md).

Prototype goal:

1. Keep using `WebGPURenderer`.
2. Create a tiny compute output buffer, ideally one value per grid cell.
3. Feed that buffer into Three.js WebGPU rendering through `StorageBufferAttribute`, TSL `storage()`, or a `StorageTexture`, depending on which path works cleanly in Three r175.
4. Render colors from the GPU-resident values without CPU readback.
5. Measure dispatch, render update, and frame latency.

Only after this bridge works should we replace the production UTCI surface path.

Why this is the right risk order:

| Option | What it proves | Why it is not first |
| --- | --- | --- |
| Web Worker quantization | Main thread can stay responsive while quantizing full readback. | It preserves full CPU readback/storage, so it does not solve 0.5m. |
| LRU decoded slices | Scrubbing can improve in the current CPU pipeline. | It caches decoded CPU slices; compute-on-demand aims to remove that CPU path. |
| `pack2x16float` | MRT+UTCI can use half the output-buffer memory. | It is useful but secondary until the GPU-render bridge exists. |
| Compute-on-demand bridge | GPU values can flow directly into rendering. | This is the core unlock for 0.5m and future layers. |

## Compute-On-Demand Target Architecture

Split the pipeline into two stages:

### Stage 1: Persistent Geometry-Dependent Compute

Run once per model/grid/weather setup:

| Data | Lifetime | Format |
| --- | --- | --- |
| BVH buffers | persistent | existing node/index/vertex buffers |
| Grid points | persistent | `vec3<f32>` |
| Solar exposure | persistent | bit-packed `u32` |
| Sky exposure | persistent | `f32` per point |
| Weather and sun vectors | persistent | `f32`, small |

### Stage 2: Per-Hour Derived Compute

Run when the user changes month/hour, metric, or later a visible tile:

1. Dispatch MRT/UTCI for one selected time index.
2. Read persistent solar/sky/weather inputs.
3. Compute MRT and UTCI in `f32`.
4. Write one GPU-resident output buffer or texture.
5. Render from that output without CPU readback.

At 0.5m, persistent exposure buffers are plausible:

| Component | Approx. size at 8.2M x 288 |
| --- | ---: |
| Solar exposure bitmask | ~295 MB |
| Sky exposure | ~33 MB |
| One-hour UTCI `f32` | ~33 MB |
| One-hour MRT `f32` | ~33 MB |
| One-hour packed MRT+UTCI | ~33 MB total |

Note: an earlier version of this strategy understated 0.5m solar bitmask memory as ~33 MB. The corrected value is about 8.2M * 288 / 8 = ~295 MB. That is large but still fundamentally different from multi-GB all-hours UTCI/MRT storage.

## `pack2x16float` Recommendation

Use `f32` for all canonical computation and validation. Do not use `f16` arithmetic for the UTCI polynomial or MRT intermediate math.

Use `pack2x16float` as an optional output-storage optimization once the bridge prototype works:

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

For the next prototype, I would implement two variants behind a flag:

| Variant | Output | Purpose |
| --- | --- | --- |
| A | one `f32` UTCI buffer | simplest bridge and baseline performance |
| B | one packed `u32` MRT+UTCI buffer | validates memory/bandwidth win and visual precision |

Decision rule: keep `f32` if bridge complexity or category flips appear; use packed output if visual/category parity is clean.

## What Worker/LRU Still Means

Worker and LRU work is not wrong; it is just no longer the leading architecture.

Keep it as:

| Use case | Role |
| --- | --- |
| No WebGPU / WebGPU failure | CPU fallback path. |
| Export/statistics/picking | Optional readback/CPU summary path. |
| Current production UX | Short-term mitigation if the GPU bridge is delayed. |
| Future tiling | A tile lifecycle cache may still be useful, but for GPU-resident tiles rather than decoded CPU UTCI slices. |

Do not spend the next major effort on CPU decoded-slice LRU unless the bridge prototype fails or production UX urgently needs a stopgap.

## Implementation Plan For The Next Spike

1. **Add instrumentation first**
   - Record exposure compute time, one-hour MRT/UTCI dispatch time, GPU output-to-render update time, and any CPU readback time.
   - Query and log `adapter.limits.maxStorageBufferBindingSize`, `maxBufferSize`, storage buffer counts, and texture limits.

2. **Create a dummy GPU output render path**
   - Use a synthetic per-point value buffer.
   - Prove Three.js WebGPU can render colors from it without rebuilding CPU `DataTexture`s.
   - Try `StorageBufferAttribute` first for point-style rendering; try `StorageTexture` if the surface overlay path wants a 2D field.

3. **Add one-hour UTCI compute mode**
   - Extend params with `time_index`.
   - Dispatch only `numPoints` work items for the selected hour.
   - Keep the old all-hours path for parity tests during transition.

4. **Compare `f32` vs packed output**
   - Measure GPU time and render update time.
   - Validate absolute UTCI error, max error, RMSE, and stress-category flips.
   - Keep thresholds stricter than the current human-facing parity tolerance.

5. **Only then refactor production live analysis**
   - Replace CPU `Int16Array` live storage with a GPU-backed live analysis object.
   - Keep CPU readback as an explicit debug/export path, not the default render path.

## Open Risks

| Risk | Probe |
| --- | --- |
| Three.js WebGPU storage-buffer interop is awkward in r175. | Dummy bridge prototype before touching production compute. |
| One-hour compute is too slow at 8.2M points on weak GPUs. | Benchmark on the weakest target device; introduce tiling/LOD if over frame budget. |
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
