# Phase 0 Research: Accelerating MRT/Raycasting

## Context
- Current: ~8 minutes for 10 m grid UTCI heatmap on typical scenes.
- Targets: ≤ 60 s first full update (city block ≤ 1M tris), ≤ 2.0 °C UTCI RMSE vs baseline, progressive feedback.
- Baseline hardware: 8‑core CPU, integrated graphics; optional GPU.

## Decisions
1. Acceleration Structure: Integrate CPU BVH with vectorized ray batches (Embree if feasible; otherwise Python-side BVH via pyembree or a robust pure-Python/numba fallback).
   - Rationale: Orders of magnitude speedup for ray-scene intersection; mature library.
   - Alternatives: KD-tree, uniform grids/voxels only; custom C++ ext (higher effort).

2. Sampling Strategy: Use stratified/importance sampling for sky directions and hemispherical rays; reuse angular samples across grid points via rotated bases.
   - Rationale: Reduces variance; fewer rays for same error bound.
   - Alternatives: Pure Monte Carlo with higher ray counts (slower).

3. Parallelism & Batching: Multi-core parallel execution with chunked grids; batch rays per chunk to maximize cache locality; avoid per-ray Python loops.
   - Rationale: CPU cores underused; batching reduces overhead.
   - Alternatives: Single-threaded; per-point loops (slow in Python).

4. Caching & Reuse: Cache sky factor/visibility per tile and hour; memoize sun-occlusion per sun position; incremental recompute when parameters change.
   - Rationale: Many queries repeat across nearby points and hours.
   - Alternatives: Recompute from scratch each interaction.

5. Progressive Feedback: Two modes supported: (a) coarse-to-fine (lower angular and/or spatial resolution first), (b) streamed tiles as they complete.
   - Rationale: Perceived latency improvement; aligns with UX targets.
   - Alternatives: Single final update.

6. Optional GPU Path: If GPU present, use PyTorch/CuPy/OptiX-backed ray batches for high-throughput; keep CPU path as baseline.
   - Rationale: Big gains on capable machines without breaking baseline goals.
   - Alternatives: Force GPU-only (excludes baseline users).

## Key Workstreams
- Profiling: Identify current bottlenecks in raycasting and MRT aggregation.
- Geometry Prep: Build or import BVH from scene meshes; tile/voxel pre-processing for streaming.
- Ray Engine: Batched ray queries via Embree/pyembree or fallback BVH; vectorized math.
- Sampling & Weights: Stratified/importance sampling sets, deterministic seeds.
- Scheduler: Chunk grid, parallel pools, cancellation, progress reporting.
- Caching Layer: Sky factors, sun masks, tile-level memoization keyed by parameters.
- Validation Suite: Latency, throughput, RMSE vs baseline across scenes.

## Risks & Mitigations
- Embree availability on Windows/Python: Provide graceful fallback BVH; document install.
- Memory pressure on large scenes: Tile processing and bounded batch sizes.
- Accuracy drift with sampling changes: Fix seeds; RMSE monitoring; increase rays on outliers.

## Alternatives Considered
- Full analytic models (no raycasting): Not sufficient for complex occlusions.
- Pure GPU-first approach: Conflicts with baseline hardware constraint.
- Precomputed full-day sky factors: Large storage; acceptable later as an optimization.

## BVH / Intersector Evaluation

### Current
- Using `trimesh` ray queries with its default BVH/intersector.
- Observed runtime: ~8 minutes for 10 m grid city-block scenes.

### Options Compared
1. Trimesh built-in BVH
   - Pros: Zero extra deps, integrated.
   - Cons: Slower for large triangle counts; limited SIMD; fewer advanced split heuristics.

2. Embree (CPU) via `pyembree` (and `trimesh.ray.ray_pyembree`)
   - Pros: State-of-the-art CPU kernels; SAH BVH with spatial splits; BVH4/8 (QBVH/MBVH) for wide SIMD; ray packets and stream APIs; robust, cross-platform.
   - Cons: Native dependency (install friction on some systems).
   - Fit: Best general-purpose speedup on baseline CPUs.

3. LBVH/HLBVH (Morton codes) builders
   - Pros: Extremely fast build; good for dynamic scenes or frequent rebuilds; amenable to GPU.
   - Cons: Slightly lower traversal performance than SAH-built BVH on static geometry.
   - Fit: Useful if we rebuild often (e.g., interactive geometry edits like adding trees).

4. Two-level BVH (top-level instances + per-mesh BVH)
   - Pros: Fast updates when adding/removing instances (trees) without full rebuild.
   - Cons: Requires instance-aware scene management.

5. Alternatives
   - NanoRT (CPU, header-only): simple, decent performance; fewer features than Embree.
   - OptiX (GPU): excellent, but conflicts with baseline (GPU not guaranteed).

### Recommendation
- Primary: Integrate Embree-backed intersector via `pyembree` and the `trimesh.ray.ray_pyembree` path where possible. Use Embree’s SAH BVH (with spatial splits where available), BVH4/8 traversal, and packet tracing for hemispherical rays.
- Fallbacks:
  - If Embree unavailable: keep current trimesh BVH, but introduce tiling, batching, and sampling improvements to mitigate.
  - For interactive geometry edits (e.g., placing trees): consider LBVH/HLBVH (Morton) builder for very fast rebuilds, or a two-level BVH (static scene BVH + instanced tree sub-BVHs) to avoid full rebuild.

### Practical Integration Path
- Use `trimesh` API to switch to `ray_pyembree` when installed; otherwise fall back automatically.
- Batch rays per tile and use coherent angular packets; avoid per-ray Python loops.
- Reuse per-mesh BVHs; for added trees, maintain a top-level BVH over instances to minimize rebuild cost.

## Embree vs Ladybug Physics
- Embree provides fast ray–geometry intersection only.
- Sun position, sky luminance/longwave, material radiative properties, and MRT/UTCI aggregation remain in our current pipeline (including Ladybug components where used).
- Integration model: we generate ray directions/weights (same physics), query Embree for visibility/occlusion and distances, then feed results into the existing MRT/UTCI calculations. No physics rewrite required.

## Runtime Selection & Fallbacks
- Embree is optional and not limited to Intel CPUs; it targets x86 with SSE/AVX (AMD works). If `pyembree` is unavailable or fails to initialize, automatically fall back to current `trimesh` BVH.
- Selection logic (conceptually):
  1) Try `trimesh.ray.ray_pyembree.RayMeshIntersector` → use when available
  2) Else use `trimesh.ray.ray_triangle.RayMeshIntersector` (current path)
- Provide a user override/env var (e.g., `FAST_UTCI_INTERSECTOR=embree|trimesh`) to force a path for testing.
- Log which intersector is active at runtime; attach to ValidationSummary for reproducibility.
