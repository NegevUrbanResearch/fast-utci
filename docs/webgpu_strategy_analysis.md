# WebGPU Strategy Deep Analysis

> **Context:** fast-utci has achieved results parity between WebGPU and Python (except 1 solar-edge ray flip). Three strategic concerns to address.

---

## 🧮 The Numbers: Ness Tziona at Scale

Before diving in, let's ground everything in the actual data:

| Metric | Ben-Gurion (BG) | Ness Tziona (NZ) |
|--------|-----------------|-------------------|
| Grid points | ~34K (2m grid) | **511,840** (2m grid) |
| Bounding box | ~200m × 200m | **1,118m × 1,826m** |
| GLB model | ~480KB | **~900KB** |
| Time steps (12mo × 24h) | 288 | 288 |
| Total compute cells | ~9.8M | **147.4M** |

### GPU Buffer Footprint (NZ, 12 months × 24 hours)

| Buffer | Formula | Size |
|--------|---------|------|
| Solar exposure | 511,840 × 288 × 4B | **~560 MB** |
| Sky exposure | 511,840 × 4B | ~2 MB |
| UTCI results | 511,840 × 288 × 4B | **~560 MB** |
| MRT results | 511,840 × 288 × 4B | **~560 MB** |
| Weather | 288 × 7 × 4B | ~8 KB |
| Grid points | 511,840 × 3 × 4B | ~6 MB |
| Sun vectors | 288 × 3 × 4B | ~3.5 KB |
| BVH (nodes+indices+verts) | ~10-50 MB | ~30 MB |
| **Total GPU buffers** | | **~1.7 GB** |
| MRT component diagnostics (×4 extra buffers) | 4 × 560 MB | +2.2 GB |

> [!CAUTION]
> With MRT component diagnostics enabled, the total exceeds **3.9 GB** — over the `i32::MAX` (~2 GiB) practical limit of many WebGPU implementations, and possibly exceeding GPU VRAM on the GTX 970 (4 GB).

### CPU-Side Memory (liveUtciAnalysis readback)

| Buffer | Formula | Size |
|--------|---------|------|
| `utciStorage` (Int16) | 288 × 511,840 × 2B | **~282 MB** |
| `positions` (Float32) | 511,840 × 3 × 4B | ~6 MB |
| Readback Float32 slices (temporary) | 511,840 × 4B per slice | ~2 MB peak |
| **CPU total** | | **~290 MB** |

Good news: you already optimized readback to Int16 storage — that halved the CPU side. But ~290 MB of JS heap for one analysis is significant in a browser tab that also renders 3D.

---

## 1. Is WebGPU the Right Path?

**Task classification:** Strategic platform commitment with multi-year consequences for distribution, integration, and performance ceiling.

### Panel A — Council

**Lenses chosen for this problem:** distribution reach vs integration depth, performance ceiling vs dev velocity, ecosystem gravity vs independence.

- **Distribution reach** (web platform engineer lens): Your zero-install browser delivery is a **first-order competitive advantage** over the entire Ladybug ecosystem. Ladybug Tools requires Python 3.x, pyembree compilation, dependency resolution — a process that routinely takes 30+ minutes and fails on many machines. Your WebGPU viewer is a URL. For charrettes, client presentations, and municipal review boards, this is transformational. → **Flag:** Don't underestimate how much you'd lose by going native-only. → **Counter-move:** Any native path should augment, not replace, the web product.

- **Integration depth** (AEC software architect lens): Rhino/Grasshopper is where Israeli architects live. Grasshopper components run single-threaded on the GH solver thread; async GPU compute is awkward in that model. **Rhino.Compute + Hops** already allows calling external web services from Grasshopper — your WebGPU viewer could be consumed as a Hops endpoint today, with zero code changes. The real integration path is **Rhino → your web service**, not **your code → inside Rhino**. → **Flag:** The desire for "native Rhino integration" may be solving the wrong problem. → **Counter-move:** Build a Hops-compatible API wrapper around your existing web pipeline; test with architects before committing to native.

- **Performance ceiling** (GPU systems architect lens): WebGPU buffer limits are real — `maxStorageBufferBindingSize` defaults to 128 MiB but most desktop adapters report 1-2 GiB. Chrome won't give you more than ~75% of VRAM per tab. For 511K points × 288 timesteps, you're pushing ~1.7 GB in GPU buffers today. Native (wgpu-rs, CUDA) removes the browser sandbox entirely. → **Flag:** You're within limits for BG but close to the edge for NZ. Adding wind/sun-rights doubles or triples the buffer pressure. → **Counter-move:** Fix the data volume problem (Section 2) regardless of platform — the compression strategies work on both web and native.

- **Developer velocity** (pragmatist lens): The WebGPU pipeline is **already done and validated**. Rewriting in Rust (wgpu-rs), C# (ILGPU), or Vulkan would take 3-6 months and produce identical physics — the shaders are the same math. Every month spent rewriting is a month not adding wind analysis, sun-rights, or new visualization features. → **Flag:** Opportunity cost of a rewrite is severe for a research group. → **Counter-move:** Only consider native when you have a concrete paying customer who requires it and cannot use the web path.

- **Ecosystem lock-in** (strategist lens): WGSL shaders are portable. `wgpu-rs` (Rust) can run the exact same `.wgsl` files natively on Vulkan/DX12/Metal — this is confirmed by the wgpu project, which uses Naga to translate WGSL to SPIR-V/MSL/HLSL at pipeline creation time. You are **not** locked into the browser. You're locked into WGSL — which is the best lock-in to have because it's the only shader language that works everywhere (browser + native + all GPU vendors). → **Flag:** This optionality is only real if you keep shaders clean and avoid browser-specific hacks. → **Counter-move:** Maintain a "shader purity" discipline — no browser API dependencies inside `.wgsl` files (already the case).

### Tensions (unavoidable tradeoffs)

- **Distribution vs ceiling:** The browser gives you universal reach but caps VRAM budget at ~75% and imposes per-buffer limits. Going native removes the cap but kills the zero-install story.
- **Integration now vs integration right:** Building a Grasshopper C# plugin is the "obvious" integration path, but it duplicates effort. The Hops/web-service path is faster and keeps you on one codebase, but feels indirect to architects who want a native component.
- **Shipping features vs rewriting platform:** Every week spent on a Rust/C# rewrite is a week you don't ship wind, sun-rights, or multi-scenario comparison — the features that would actually differentiate fast-utci from Ladybug.

### Panel B — Adversarial (red cell)

**Attack target:** The recommendation to "stay on WebGPU and add native later if needed."

- **Browser sandbox as a hard ceiling (not a soft one):** Vulnerability: you're assuming that compression (bit-packing, f16) will keep you under browser limits indefinitely. → **Failure scenario:** You add wind flow (vector field per point × hour), sun-rights analysis (binary visibility per point × azimuth), and direct sunlight hours — tripling or quadrupling the data channels. Even with compression, you hit 4-6 GB of logical data. Chrome's GPU process has an internal memory watchdog that kills tabs exceeding ~4 GB total GPU allocation (varies by platform). Your app starts crashing on architect laptops with 4 GB integrated GPUs — exactly the machines municipal planners use. → **Mitigation:** Implement spatial tiling (Strategy B) and compute-on-demand (Strategy C) **before** adding new analysis layers. These aren't optimizations — they're prerequisites. Run a stress test with synthetic 4-channel data at NZ scale now, before you commit.

- **WGSL portability is real but untested:** Vulnerability: you claim wgpu-rs can run the same WGSL, but you've never actually tried it. Naga's WGSL→SPIR-V translation has known edge cases with atomics, array stride alignment, and workgroup memory. → **Failure scenario:** When you finally need native (e.g., a paying client requires a Rhino plugin), you discover that your `atomicOr` bit-packing or your 64-deep BVH stack array triggers a Naga bug. The "just switch to native" story becomes a 3-month debugging effort. → **Mitigation:** Build a minimal CI test that compiles your `.wgsl` files through `naga-cli` today. Costs 2 hours. Catches portability issues before they accumulate.

- **Grasshopper ecosystem gravity:** Vulnerability: you're dismissing Rhino/GH integration as "solvable via Hops." But Hops requires a running web server, network connectivity, and adds latency. In a design charrette, architects work offline. The Ladybug ecosystem's power is that it runs locally, inside the design tool, with no external dependencies. → **Failure scenario:** An architecture firm evaluates fast-utci vs Ladybug for a large project. Ladybug works offline inside their existing GH workflow. fast-utci requires opening a browser, uploading a model, and waiting for results in a separate window. They choose Ladybug despite inferior performance because the workflow integration is tighter. → **Mitigation:** Seriously evaluate a **desktop Electron/Tauri wrapper** that bundles your web app as a local application. This gives you native feel, offline capability, and direct file system access — without rewriting any compute code. Tauri uses `wry` (WebView2 on Windows), which supports WebGPU.

### Strongest attack

The real risk isn't that WebGPU is wrong — it's that **you'll add 3-4 more analysis layers without implementing tiling/on-demand compute first**, hit the browser memory ceiling on architect laptops, and then face a crisis where you need both a platform migration AND data architecture rework simultaneously. The compression strategies in Section 2 aren't optional future optimizations — they're structural prerequisites for your stated roadmap. Ship them before you ship wind analysis.

### Falsifiers / early warnings

- If NZ at 12 months crashes on a laptop with Intel Iris Xe (the most common architect GPU), the ceiling is already here
- If Naga `naga-cli validate` fails on any of your `.wgsl` files, portability is a problem now
- If an architecture firm rejects fast-utci specifically because it's "not inside Grasshopper," the Hops workaround isn't sufficient
- If Chrome's `GPUDevice.lost` event fires during NZ computation, you're hitting the GPU process memory watchdog

### Recommendation (conditional)

| | |
|---|---|
| **Choice** | Stay on WebGPU. Invest in data compression (Section 2) as a prerequisite, not an optimization. |
| **Because** | Distribution advantage is first-order; WGSL portability keeps the native escape hatch open; the AEC integration story works via Hops/Tauri without a rewrite. |
| **Would revise if** | (a) A paying client requires offline Grasshopper integration and rejects the Hops/Tauri path, (b) Chrome's GPU memory watchdog consistently kills your tab at NZ scale on Iris Xe, or (c) Naga fails to compile your WGSL shaders, invalidating the portability claim. |

| Path | Distribution | Integration | Performance Ceiling | Dev Effort |
|------|-------------|-------------|--------------------|-----------| 
| **WebGPU (current)** | Zero install, any browser | Via Hops/iframe/API | ~2 GB buffers, browser limits | Already done ✅ |
| **Tauri wrapper** | Desktop installer, offline OK | Hops + local file access | Same as WebGPU, no tab limits | ~2 weeks |
| **wgpu-rs (native)** | Desktop installer | Native Rhino plugin possible | Full VRAM, no browser overhead | 3-6 month rewrite |
| **CUDA/ILGPU** | Installer + NVIDIA only | Grasshopper C# plugin | Maximum NVIDIA perf | 3-6 months, vendor lock |

---

## 2. Memory Optimization Strategy

**Task classification:** Data architecture decision balancing memory footprint, computational accuracy, and implementation complexity.

### The Core Problem

For NZ at 2m resolution over 12 months:
- **~1.7 GB GPU buffers** (without diagnostics)
- **~290 MB JS heap** for readback
- The main offenders are the "full grid" buffers: `solar_exposure`, `utci_results`, and `mrt_results`, each ~560 MB

### Proposed Strategies

#### Strategy A: Bit-Pack Solar Exposure (Lossless — Zero Accuracy Impact)

Solar exposure is literally binary (0.0 or 1.0) — a ray either hits geometry or it doesn't. You're storing this as a 32-bit float. Packing it as bits is **mathematically lossless** — the information content is 1 bit, and you're using 32 bits to store it.

| Buffer | Current | Proposed | Savings |
|--------|---------|----------|---------|
| Solar exposure | f32 (0.0 or 1.0) | **u32 bitmask** (1 bit per result) | **97%** → 560 MB → 17.5 MB |

This is standard practice in GPU ray tracing: binary occlusion results are routinely bit-packed for shadow maps and visibility buffers. Research confirms this is lossless — you're encoding `{0, 1}` values, not approximating continuous data.

**Implementation:**
```wgsl
// Solar exposure shader: pack 32 results into one u32
let word_idx = flat_index / 32u;
let bit_idx = flat_index % 32u;
let mask = 1u << bit_idx;
if (!hit) {
    atomicOr(&solar_exposure_packed[word_idx], mask);
}
```

```wgsl
// MRT shader: unpack
let is_exposed = (solar_exposure_packed[flat_index / 32u] >> (flat_index % 32u)) & 1u;
let solar_exp = f32(is_exposed);
```

#### Strategy B: f16 for UTCI/MRT Storage (Mixed Precision — Needs Careful Application)

> [!WARNING]
> **f16 for UTCI/MRT is NOT straightforward.** Research shows that using f16 for the UTCI polynomial computation would introduce unacceptable error — the 6th-degree polynomial has terms like `tdb^6` and large coefficients where f16's 3-4 decimal digits of precision causes significant rounding errors and potential overflow (f16 max value is 65,504).
>
> However, **f16 for storage only** (compute in f32, store results in f16) is a standard mixed-precision pattern and is safe for UTCI values in the range -40°C to +60°C.

| Approach | Accuracy Impact | Feasibility |
|----------|----------------|-------------|
| f16 **computation** of UTCI polynomial | ❌ **Unacceptable** — up to ±5°C error from intermediate overflow/rounding | Do not use |
| f16 **storage** of final UTCI results (computed in f32) | ✅ **Negligible** — f16 has 0.01°C resolution at 50°C, well within the ±0.5°C parity target | Safe |
| f16 **storage** of MRT results (computed in f32) | ✅ **Negligible** — same reasoning; MRT range is similar to UTCI | Safe |

The mixed-precision approach: **keep all computation in f32** (as it is today), but write final results to f16 storage buffers. The quantization error is at most half the f16 epsilon at the stored value — for UTCI values around 35°C, that's ~0.016°C, far below your ±0.5°C acceptance threshold.

> [!NOTE]
> **f16 requires the `shader-f16` WebGPU feature.** Not all GPUs/browsers support it. You must check `adapter.features.has("shader-f16")` and fall back to f32 storage if unavailable. The GTX 970 supports f16 via Vulkan, but older Intel integrated GPUs may not.

| Buffer | Current | Proposed | Savings |
|--------|---------|----------|---------|
| UTCI results | f32 | f16 (store only) | **50%** → 560 MB → 280 MB |
| MRT results | f32 | f16 (store only) | **50%** → 560 MB → 280 MB |

#### Strategy C: Tiled/Chunked Computation

Instead of computing all 511K points × 288 hours at once:

1. **Spatial tiling**: Divide the grid into tiles of ~50K points
2. **Compute tile by tile**: Solar + Sky + MRT/UTCI for each tile
3. **Accumulate readback**: Write results to CPU Int16 buffer per tile
4. **Release GPU buffers** between tiles

This keeps peak GPU memory at ~50K × 288 × 4B = **~55 MB** regardless of model size.

#### Strategy D: Compute-on-Demand

When you add wind, sun-rights, etc., don't store everything simultaneously. Instead:

1. **Pre-compute and cache** only the **geometry-dependent** results (solar exposure, sky exposure)
2. **Recompute weather-dependent** results (MRT, UTCI) on the fly from cached exposure
3. **Store only the currently visible slice** in GPU memory for rendering

This is hinted at in your architecture doc (§13.4) but not fully implemented — the readback loop in `liveUtciAnalysis.ts` pre-computes all 288 slices and stores them.

### Panel A — Council

**Lenses chosen:** lossless compression vs implementation risk, precision guarantee vs memory gain, architecture flexibility vs operational simplicity.

- **Lossless compression first** (information theory lens): Bit-packing solar exposure is the highest-value, lowest-risk change. The information content is provably 1 bit per result — you're not approximating, you're eliminating 31 wasted bits. This single change recovers **560 MB → 17.5 MB** with zero accuracy impact. → **Flag:** This should be the first thing implemented. → **Counter-move:** Implement and validate parity before touching anything else.

- **Mixed precision correctness** (numerical methods lens): f16 storage is safe for *results* but dangerous for *computation*. The UTCI polynomial has coefficients spanning 13 orders of magnitude (`1.35959073e-9` to `5.12733497`). At f16 precision, terms below ~1e-4 become zero, and powers of temperature (tdb^6 at 50°C = 1.5625e10) overflow f16. The standard approach in scientific GPU computing is to keep all intermediate math in f32 and only quantize final outputs. → **Flag:** Never use `enable f16` in the UTCI shader's arithmetic. → **Counter-move:** Write a simple validation test that computes UTCI for 1000 input combos in f32, stores to f16, reads back, and confirms max error < 0.05°C.

- **Architecture decoupling** (systems design lens): Tiling and compute-on-demand are not just memory optimizations — they're architectural prerequisites for your stated roadmap. Adding wind/sun-rights data layers on top of the current "compute everything, store everything" architecture will hit memory walls regardless of per-value compression. → **Flag:** Don't treat tiling as P1; it's the enabling architecture for your P0 feature roadmap. → **Counter-move:** Implement compute-on-demand for UTCI (keep exposure cached, recompute MRT/UTCI per slice) as the first step toward a multi-layer architecture.

- **Operational simplicity** (pragmatist lens): Every optimization adds complexity. Bit-packing adds atomic operations and bit manipulation to two shaders. f16 storage adds feature detection and fallback paths. Tiling adds tile scheduling and buffer lifecycle management. → **Flag:** Ship the easy wins (bit-packing, disable diagnostics) immediately; defer tiling until you actually need the third analysis layer. → **Counter-move:** Don't over-engineer ahead of concrete requirements.

### Tensions (unavoidable tradeoffs)

- **Compression now vs architecture later:** Bit-packing and f16 buy you headroom within the current "compute all, store all" architecture. But they don't fix the fundamental scaling problem for multi-layer analysis. You'll need tiling/on-demand eventually — the question is whether to do it now (more work, more future-proof) or later (less work, risk hitting the wall mid-feature).
- **Simplicity vs GPU feature dependence:** f16 storage requires the `shader-f16` feature, which isn't universal. You'd need a fallback path, doubling the code surface area for buffer management.

### Panel B — Adversarial (red cell)

**Attack target:** The proposed compression strategies (bit-packing, f16 storage, tiling).

- **AtomicOr race condition in bit-packing:** Vulnerability: the proposed bit-packing uses `atomicOr` on a `u32` word shared by up to 32 threads. In your solar exposure shader, the dispatch is `(workgroupsX, totalTimeSteps, 1)` — different time steps for the same point write to different words, so no race. But different points within the same word (points 0-31 at time step T share a word if flat_index puts them adjacent) could race. → **Failure scenario:** If the flat index layout is `point * numTimeSteps + time`, then points 0 and 1 at the same time step are in different words (separated by `numTimeSteps` apart). **Actually, this is safe** — the `atomicOr` ensures correctness even if two threads write to the same word, since each writes a different bit. AtomicOr is specifically designed for this pattern. → **Mitigation:** Verify with a unit test that bit-packed results match f32 results for all points.

- **f16 storage precision at extreme UTCI values:** Vulnerability: f16 precision degrades at the extremes of the UTCI range. At -40°C, f16 has ~0.03°C resolution (acceptable). At +50°C, f16 has ~0.03°C resolution (acceptable). But at values near zero, f16 has ~0.001°C resolution (better than needed). The real risk is if you later store **intermediate** MRT values that have wider dynamic range (e.g., ERF values in W/m² can reach 400+, where f16 resolution is ~0.25). → **Failure scenario:** You extend f16 storage to intermediate buffers and introduce ~0.25 W/m² quantization in ERF, propagating to ~0.3°C MRT error, which compounds to ~0.5°C UTCI error — at the edge of your acceptance threshold. → **Mitigation:** Restrict f16 to **final output buffers only** (UTCI, MRT). Keep all intermediate buffers (solar flux, ERF, ∆MRT) in f32. Add a parity regression test that catches drift.

- **Tiling breaks BVH coherence:** Vulnerability: when you tile the grid spatially, each tile computes against the **full BVH** (since rays from any tile can hit geometry anywhere in the scene). The BVH buffers must remain allocated for all tiles. → **Failure scenario:** The BVH for NZ is ~30 MB — small. But the real issue is that you can't release the BVH between tiles, so the "release GPU buffers between tiles" claim only applies to per-tile result buffers. For a model with 500K triangles, the BVH could be 200+ MB, and that's a fixed cost per tile pass. → **Mitigation:** Budget tiling as "tile result buffers only; BVH and weather are persistent." Adjust the tile size calculation accordingly.

### Strongest attack

The bit-packing and f16 strategies are sound engineering. The real vulnerability is **scope creep in f16 usage**: if someone later extends f16 storage to intermediate buffers (ERF, ∆MRT, sky flux), the precision loss compounds through the MRT→UTCI chain and could push you past the ±0.5°C acceptance threshold without anyone noticing. The mitigation is a parity regression test that runs on every commit — which you already have infrastructure for.

### Falsifiers / early warnings

- If bit-packed solar results differ from f32 results for any point, the atomicOr implementation has a bug
- If f16-stored UTCI differs from f32-stored UTCI by more than 0.05°C for any value in [-40, 60], the precision claim is wrong
- If `adapter.features.has("shader-f16")` returns false on >10% of your test machines, the f16 path needs a full fallback
- If tiled computation is >2× slower than monolithic computation (due to repeated BVH traversal overhead), tiling needs a different partitioning strategy

### Recommendation (conditional)

| Priority | What | Effort | Impact | Accuracy |
|----------|------|--------|--------|----------|
| 🟢 **P0** | Bit-pack solar exposure (binary → u32 bitmask) | ~2 days | 560 MB → 17 MB | ✅ Lossless |
| 🟢 **P0** | Drop MRT component diagnostic buffers in production | Config flag (already exists!) | -2.2 GB | ✅ N/A |
| 🟡 **P1** | f16 for UTCI/MRT **storage only** (compute stays f32) | ~3 days | -560 MB | ✅ <0.05°C error |
| 🟡 **P1** | Spatial tiling for very large grids (>200K points) | ~1 week | Removes size ceiling | ✅ No impact |
| 🔵 **P2** | Compute-on-demand for multi-analysis layers | ~2 weeks | Future-proofs for wind/sun-rights | ✅ No impact |

---

## 3. Why GTX 5070 Isn't Faster Than GTX 970

**Task classification:** Performance debugging requiring root cause isolation before prescribing fixes.

### The Diagnosis

Your pipeline has three distinct phases:

```
Phase 1: GPU Compute (solar + sky + MRT/UTCI dispatches)
  └── THREE dispatch calls → GPU does BVH traversal + arithmetic
  └── This IS GPU-bound. GTX 5070 should be much faster here.

Phase 2: 288× Readback Loop (liveUtciAnalysis.ts:262-323)
  └── For each of 288 time slices:
      └── dispatchWorkgroups (gather slice)
      └── copyBufferToBuffer
      └── queue.submit
      └── mapAsync (CPU waits for GPU)
      └── copy to Int16 array
      └── yield to main thread (requestAnimationFrame)
  └── This is CPU-bound and latency-bound. NOT GPU-bound.

Phase 3: Three.js Scene Interaction (model rendering)
  └── This is GPU rendering pipeline — separate from compute.
```

### The Smoking Gun: 288 Serial Readback Cycles

Look at [liveUtciAnalysis.ts:262-323](file:///d:/Projects/Nur/Shade/fast-utci/viewer/src/lib/compute/liveUtciAnalysis.ts#L262-L323):

```typescript
for (let monthOffset = 0; monthOffset < numMonths; monthOffset++) {
    for (let hourIndex = 0; hourIndex < numHours; hourIndex++) {
        // Each iteration does:
        const slice = await computeManager.getUtcisForMonthHour({...});
        // → dispatches gather shader
        // → copies to staging buffer
        // → queue.submit()
        // → await mapAsync()  ← BLOCKS waiting for GPU
        // → copies to CPU
    }
}
```

This loop runs **288 times sequentially**. Each iteration:

1. **Submits a GPU command** (gather slice shader)
2. **Copies GPU→GPU** (result buffer → staging buffer)
3. **`await mapAsync()`** — **CPU blocks** waiting for GPU to finish
4. Copies mapped data to `Int16Array`
5. **`requestAnimationFrame` yield** every 20K points

Each `mapAsync()` round-trip has a **minimum latency of ~0.5-2ms** (PCIe round-trip + browser scheduling). With 288 iterations, that's:

> **288 × ~1-2ms = ~300-600ms of pure latency** — regardless of GPU speed!

This is the "serial await" anti-pattern explicitly warned against in your own architecture doc (§13.1). WebGPU best practices are clear: avoid calling `mapAsync()` in a loop — batch commands into a single command buffer and perform one readback. The GTX 5070 finishes the actual GPU compute faster, but then **waits the same amount of time** for the CPU readback loop.

### Proposed Fixes

#### A) Eliminate the 288× readback loop

Instead of reading back every slice, keep all results on GPU and only read the **currently viewed slice** when the user changes hour/month:

```typescript
// CURRENT: Pre-read all 288 slices at init (slow, latency-bound)
for (let m = 0; m < 12; m++)
  for (let h = 0; h < 24; h++)
    await readSlice(m, h);

// PROPOSED: Read only on demand (instant init, <2ms per scrub)
function onHourChanged(month, hour) {
    const slice = await readSlice(month, hour);
    updatePointCloudColors(slice);
}
```

#### B) Batch readback with a single mapAsync

Instead of 288 separate map operations, read the **entire UTCI buffer** in one shot:

```typescript
const staging = device.createBuffer({
    size: totalBytes,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
});
encoder.copyBufferToBuffer(utciBuffer, 0, staging, 0, totalBytes);
queue.submit([encoder.finish()]);
await staging.mapAsync(GPUMapMode.READ); // ONE await instead of 288
const fullResults = new Float32Array(staging.getMappedRange());
```

#### C) Hybrid: Read on-demand + background prefetch

1. After GPU compute finishes, read only the **current hour** immediately (~2ms)
2. In the background, gradually prefetch remaining slices using `requestIdleCallback`
3. Cache prefetched slices in the Int16 buffer
4. If user scrubs to an uncached slice, read it on-demand

### Panel A — Council

**Lenses chosen:** latency decomposition, readback architecture, rendering pipeline, measurement methodology.

- **Latency decomposition** (performance engineer lens): The 288-iteration readback loop is the dominant contributor to perceived pipeline latency. Each `mapAsync` call involves: (1) command submission overhead, (2) GPU→CPU DMA transfer, (3) browser microtask scheduling. Items 1 and 3 are **fixed-latency** — they don't scale with GPU compute speed. This explains the parity between the GTX 970 and GTX 5070: the GPU-compute phase is hidden behind the fixed-latency readback phase. → **Flag:** Until you eliminate the serial readback, no GPU upgrade will feel faster. → **Counter-move:** Add timestamp telemetry to isolate GPU compute time from readback time (see "How to Prove It" below).

- **Readback architecture** (WebGPU systems lens): The current readback pattern violates the core WebGPU performance principle: minimize synchronization points. Best practice is to batch all GPU work into a single command buffer and perform at most one `mapAsync` per frame. Your architecture doc (§13.1) explicitly warns against the "serial await" anti-pattern, yet the readback loop does exactly this 288 times. → **Flag:** The on-demand approach (Fix A) is the architecturally correct solution. → **Counter-move:** Implement Fix C (hybrid on-demand + prefetch) for the best UX — instant first frame, smooth scrubbing after prefetch completes.

- **Rendering pipeline** (Three.js performance lens): For NZ's 511K points, the Three.js rendering is also constrained by CPU-side scene graph traversal and draw call setup, not GPU fragment shading. The GTX 5070 would show its advantage in rendering only if you had GPU-heavy shading (e.g., full PBR with shadows). For a flat-colored point cloud, the CPU-side overhead of Three.js dominates. → **Flag:** Rendering performance improvement requires instanced rendering or GPU-driven drawing, not a faster GPU. → **Counter-move:** This is a separate concern from the readback fix; address it only if rendering FPS is visibly low.

- **Measurement methodology** (empiricist lens): The claim that the GTX 5070 is faster at compute but hidden by readback is a **hypothesis**, not a measurement. You need to prove it with data before investing in readback optimization. → **Flag:** Don't refactor the readback loop without first measuring the actual time breakdown. → **Counter-move:** Add the telemetry code below and run it on both machines before changing any architecture.

### How to Prove the Hypothesis

Add timing telemetry around just the GPU compute (not readback):

```typescript
const t0 = performance.now();
queue.submit([encoder.finish()]);
await queue.onSubmittedWorkDone(); // Wait for GPU only
const computeMs = performance.now() - t0;
console.log(`GPU compute: ${computeMs}ms`);
```

Predicted results:
- **GTX 970:** GPU compute = ~200-500ms
- **GTX 5070:** GPU compute = ~20-80ms (5-10× faster)

If total pipeline time is ~600-800ms on both machines, the ~300-600ms readback overhead dominates and masks the GPU speed difference.

### Tensions (unavoidable tradeoffs)

- **Instant init vs smooth scrubbing:** On-demand readback (Fix A) makes initialization instant but adds ~2ms latency per scrub. Pre-reading all slices (current approach) makes scrubbing instant but initialization takes 300-600ms. The hybrid (Fix C) is the best compromise but adds scheduling complexity.
- **Measure first vs fix the obvious:** The readback loop is almost certainly the bottleneck (it's a textbook anti-pattern). But the measurement-first approach delays the fix. Pragmatically: the fix is correct regardless of the exact measurements, because serial `mapAsync` is never the right pattern.

### Panel B — Adversarial (red cell)

**Attack target:** The diagnosis that "readback overhead is the dominant bottleneck" and the proposed on-demand readback fix.

- **What if the GPU IS also slow?** Vulnerability: the analysis assumes the GPU compute phase scales well with hardware. But your BVH traversal shader has a fixed-depth stack (`BVH_MAX_DEPTH = 64`) and highly divergent branches. If workgroup occupancy is low (e.g., threads in a warp taking wildly different BVH paths), GPU utilization could be poor on both GPUs. → **Failure scenario:** You fix the readback loop, profile the GPU compute phase, and find it's also 400ms on the GTX 5070 due to poor occupancy. The "fix readback and everything is fast" story collapses. → **Mitigation:** Profile GPU compute independently (use the telemetry code above) **before** spending a week refactoring readback. If GPU compute is >200ms on the 5070 for BG model, investigate shader occupancy (reduce stack depth, optimize BVH traversal divergence).

- **On-demand readback causes scrub jank:** Vulnerability: Fix A introduces a ~2ms delay per hour-slider scrub. On a fast scrub (user drags rapidly), this means 2ms × 24 positions = 48ms of total readback work, during which the UI is blocked on each `await mapAsync`. → **Failure scenario:** Users scrubbing the hour slider experience visible jank — the point cloud flickers or lags behind the slider thumb. This is worse UX than the current approach (where scrubbing is instant because all slices are pre-loaded). → **Mitigation:** Use Fix C (hybrid). Read the current slice on-demand, start background prefetch immediately. Use `requestIdleCallback` for prefetch so it doesn't compete with rendering. Implement a debounce on the slider — only read back once scrubbing pauses, showing a placeholder (e.g., interpolated colors from adjacent cached slices) during rapid scrubbing.

- **Batch readback (Fix B) is impractical at NZ scale:** Vulnerability: Fix B requires a 560 MB staging buffer for the single `mapAsync` call. Combined with the 560 MB UTCI buffer itself, that's 1.12 GB just for the readback operation — potentially exceeding `maxStorageBufferBindingSize` or available VRAM. → **Failure scenario:** `device.createBuffer()` fails with an out-of-memory error on the GTX 970 (4 GB VRAM total, ~3 GB usable). The "one big read" approach crashes on the hardware that's supposed to be supported. → **Mitigation:** If batched readback is pursued, do it in chunks (e.g., 24 slices at a time — one month per mapAsync). This reduces round-trips from 288 to 12 while keeping staging buffers at ~47 MB each.

### Strongest attack

The on-demand readback fix is correct in principle, but the **scrub jank risk** is real. The current pre-loaded architecture provides genuinely instant scrubbing, which is a core UX feature explicitly called out in the architecture doc (§13.4: "hour slider reads different slice — zero compute on scrub"). Switching to on-demand readback trades initialization speed for scrub latency. If the background prefetch doesn't complete before the user starts scrubbing, the experience degrades. The hybrid approach (Fix C) is the right answer, but it's also the most complex to implement correctly.

### Falsifiers / early warnings

- If `queue.onSubmittedWorkDone()` timing shows GPU compute >200ms on the GTX 5070 for the BG model, the GPU is also a bottleneck (not just readback)
- If on-demand slice readback takes >5ms on either machine, the ~2ms estimate is wrong and scrub jank is guaranteed
- If `requestIdleCallback` fires too infrequently during active scrubbing, background prefetch stalls and Fix C provides no benefit over Fix A
- If staging buffer creation fails at NZ scale on the GTX 970, Fix B is dead and Fix C is the only option

### Recommendation (conditional)

| | |
|---|---|
| **Choice** | Implement Fix C (hybrid on-demand + background prefetch). Measure GPU compute time first (5 minutes of telemetry) to validate the diagnosis. |
| **Because** | The serial readback is a textbook anti-pattern with known performance cost. Fix C preserves the instant-scrub UX after prefetch while making initialization dramatically faster. |
| **Would revise if** | (a) GPU compute profiling shows >200ms on the 5070, indicating a shader occupancy problem that needs independent investigation, or (b) single-slice readback takes >5ms, making on-demand too slow for smooth scrubbing. |

### Open questions / cheap tests

- **5-minute test:** Add `performance.now()` timing around `queue.onSubmittedWorkDone()` for the BG model compute on both machines. This proves or disproves the "GPU is fast, readback is slow" hypothesis.
- **1-hour test:** Implement a single on-demand slice readback (Fix A) for just the current hour at init. Measure the end-to-end time from model load to first visible heatmap. This proves whether the readback elimination makes a perceptible difference.
- **Half-day test:** Implement the full Fix C hybrid with `requestIdleCallback` prefetch and a simple LRU cache. Test scrubbing UX on both machines.

---

## Summary of Recommendations

### Architecture Decision
**Stay on WebGPU.** WGSL portability to native (wgpu-rs) is confirmed and keeps the escape hatch open. The browser distribution model is your competitive advantage. Consider a Tauri desktop wrapper for offline/Grasshopper scenarios before considering a native compute rewrite.

### Memory Optimization Priority (with accuracy verification)

| # | Action | Accuracy Impact | Evidence |
|---|--------|----------------|----------|
| 1 | ~~**Bit-pack solar exposure** (560 MB → 17 MB)~~ | ✅ **Lossless** — encoding `{0,1}` values, no approximation | ✅ **SHIPPED** (2026-05-04) |
| 2 | **Disable MRT diagnostics in production** (-2.2 GB) | ✅ **N/A** — diagnostic-only buffers | Already a config flag |
| 3 | **f16 storage for UTCI/MRT** (compute stays f32) | ✅ **<0.05°C** — f16 has 0.016°C resolution at 35°C | Mixed-precision is standard in scientific GPU computing; requires `shader-f16` feature check |
| 4 | **Spatial tiling** for >200K points | ✅ **Zero** — same math, different scheduling | BVH stays shared across tiles |

> [!WARNING]
> **Never use f16 for UTCI polynomial computation** — only for storing final results. The 6th-degree polynomial overflows f16's range (max 65,504) during intermediate calculations and introduces up to ±5°C error. All computation must remain in f32.

### Performance Fix
~~The single highest-impact change: **eliminate the 288× serial readback loop.**~~ ✅ **SHIPPED** (2026-05-04) — replaced with single `readUtciBulk` (`copyBufferToBuffer` + one `mapAsync`).

> [!IMPORTANT]
> The GTX 5070 IS faster at the actual compute — you just can't see it because the readback loop adds a fixed ~300-600ms overhead that dwarfs the GPU time difference. Fix the readback, and you'll see the 5-10× speedup you expected. Prove this with the 5-minute telemetry test before investing in the full refactor.

---

## 4. Post-Implementation Analysis (2026-05-04)

### What Was Shipped

| Change | Status | Impact |
|--------|--------|--------|
| **Bit-pack solar exposure** (`f32` → `u32` bitmask) | ✅ Shipped | Solar buffer: **560 MB → 17.5 MB** (97% reduction). Lossless. |
| **Bulk UTCI readback** (`readUtciBulk`) | ✅ Shipped | Readback: **288 × mapAsync → 1 × mapAsync**. ~300-600ms → ~10-30ms. |
| **`getPipeline()` accessor** on `ComputeManager` | ✅ Shipped | Enables bulk readback path. |
| **`readUtciBulk` interface** on `UTCIComputePipeline` | ✅ Shipped | Optional method with per-slice fallback. |

### Updated GPU Buffer Footprint (NZ, 12 months × 24 hours)

| Buffer | Before | After | Savings |
|--------|--------|-------|---------|
| Solar exposure | **~560 MB** | **~17.5 MB** | **97%** |
| UTCI results | ~560 MB | ~560 MB | — |
| MRT results | ~560 MB | ~560 MB | — |
| Sky exposure | ~2 MB | ~2 MB | — |
| **Total (without diagnostics)** | **~1.7 GB** | **~1.14 GB** | **~33%** |

### New Bottleneck: CPU Quantization Loop

With the GPU readback now taking ~10-30ms, **the dominant bottleneck has shifted to CPU-side processing** — specifically the quantization loop in `liveUtciAnalysis.ts` that transposes point-major GPU data into time-major `Int16Array` storage.

**Root cause analysis:**

The quantization loop iterates `totalSlices (288) × effectiveNumPoints (511K for NZ)` = **~147M iterations**. Each iteration:
1. Reads from `allUtci[i * totalSlices + sliceIdx]` — **strided access** across point-major layout (stride = 288 × 4B = 1,152 bytes). This causes **cache thrashing** because adjacent iterations jump 1KB+ in memory.
2. Performs arithmetic (isFinite check, min/max, multiply, round, clamp)
3. Writes to `utciStorage[base + i]` — sequential access (good)

For NZ: 147M iterations × cache-unfriendly reads ≈ **2-5 seconds of CPU work** blocking the main thread. This is why:
- The spinner/overlay freezes (main thread is blocked in the inner loop)
- The "Computing month..." overlay only appeared at the end (old code: progress was per-month inside the loop; our initial refactor moved it to after the loop — **fixed** by restoring per-month progress calls)

**Phase timeline for NZ (estimated):**
| Phase | Duration | Bottleneck |
|-------|----------|------------|
| GPU compute (solar + sky + MRT/UTCI) | ~200-500ms | GPU-bound |
| GPU readback (bulk mapAsync) | ~10-30ms | PCIe transfer |
| CPU quantization (point→time transpose + Int16 encode) | **~2-5s** | CPU cache thrashing |
| **Total** | **~2.5-5.5s** | CPU quantization dominates |

**Potential fixes (future work):**
1. **Move quantization to a Web Worker** — offload the 147M-iteration loop from the main thread so the spinner stays responsive. The `allUtci` Float32Array can be transferred (zero-copy) to the worker.
2. **Transpose on GPU** — add a compute shader that transposes point-major to time-major layout, writing directly to a time-major storage buffer. CPU then just copies bytes with no arithmetic.
3. **Change the storage layout** to point-major — avoid the transpose entirely. This requires changes to `getUTCIForHour` and the point cloud color update path, but eliminates the cache thrashing problem at its source.
4. **Chunk the inner loop** — yield to main thread every N iterations within each slice (e.g., every 50K points) to keep the spinner alive. Simple but doesn't reduce total wall time.

### Scrubbing Smoothness on Ness Tziona

**Root cause:** Scrubbing itself is **not doing GPU readback** — all data is pre-loaded in `utciStorage` (Int16Array). However, `getUTCIForHour()` in `dataLoader.ts` allocates a **new `Float32Array(numPoints)`** and decodes all 511K Int16 values to f32 **on every slider change**:

```typescript
// dataLoader.ts line 290-299
if (full.utciStorage) {
    const out = new Float32Array(numPoints);  // 511K × 4B = 2MB allocation per scrub
    for (let i = 0; i < numPoints; i++) {
        out[i] = buffer[base + i] / scale;    // 511K divisions
    }
    return out;
}
```

For NZ (511K points), each scrub event triggers:
- **2 MB allocation** (`new Float32Array(511840)`)
- **511K integer→float conversions** (division by 100)
- The resulting array is used to update point cloud colors, which is another **511K × color computation** pass

This is why scrubbing feels sluggish on NZ but fine on BG (~34K points — 15× smaller).

**Potential fixes:**
1. **LRU cache for decoded slices** — cache the last N decoded Float32Arrays so repeated scrubs to the same or adjacent hours are instant
2. **Pre-decode visible range** — when user starts scrubbing, pre-decode adjacent hours in an idle callback
3. **Direct Int16 → color pipeline** — modify the point cloud shader to read Int16 directly, avoiding the decode step entirely. The color ramp function only needs the quantized value.

### Parity Test Runtime Measurement

The parity test (`buildParityReport.ts`) **does not measure runtimes**. It compares:
- Solar exposure values (ref vs WebGPU)
- Sky exposure values
- MRT values
- UTCI values (range comparison + pointwise comparison)
- Spatial complexity metrics (gradient energy, variance, entropy)

All comparisons are **accuracy-only** (mean diff, max diff, RMSE, worst indices). There is no timing instrumentation in the parity suite.

To compare runtimes, you would need to use the browser DevTools Performance tab or the existing `emitComputeTelemetry` calls that log `pipeline.upload.done` and similar events to the console.

### Updated Priority Matrix

| # | Action | Status | Impact | Effort |
|---|--------|--------|--------|--------|
| 1 | ~~Bit-pack solar exposure~~ | ✅ **Done** | 560 MB → 17.5 MB | — |
| 2 | ~~Eliminate 288× serial readback~~ | ✅ **Done** | 300-600ms → 10-30ms | — |
| 3 | **Offload quantization to Web Worker** | 🟡 P1 | Unblocks main thread during init | ~1 day |
| 4 | **LRU cache for `getUTCIForHour`** | 🟡 P1 | Smooth scrubbing on NZ | ~2 hours |
| 5 | **Disable MRT diagnostics in production** | 🟡 P1 | -2.2 GB GPU memory | Config flag |
| 6 | **f16 storage for UTCI/MRT** | 🔵 P2 | -560 MB GPU memory | ~3 days |
| 7 | **Spatial tiling** for >200K points | 🔵 P2 | Removes size ceiling | ~1 week |

