# Python vs WebGPU Exposure Paths Comparison

Comparison of how solar and sky exposure are computed in the Python pipeline (reference) vs the WebGPU viewer pipeline. Grids and some inputs differ by design; this doc helps debug parity and interpret statistical differences.

---

## 1. Grid (positions)

| Aspect | Python (reference) | WebGPU |
|--------|--------------------|--------|
| **Source** | Precomputed grid stored in the analysis `.bin` file. Positions come from the original analysis run (e.g. Grasshopper or `run_analysis.py`). | Grid generated at runtime from the 3D model: `generateGridFromMesh(mesh, gridResolution, zHeight)` in a worker. |
| **Coordinates** | Analysis coordinate system (e.g. `xy_ground`: x, y in ground plane, z up or fixed). Stored in `.bin` as `num_positions × 3` float32. | Three.js world frame: Y-up, X East, Z North. Grid points = surface hit point + `worldNormal * zHeight` (default `zHeight = 0.9`). |
| **Height** | For exposure, Python uses **sample points** above ground: `create_human_sample_points(position, pt_count=1, height=1.7)` → one point at `position + (0, 0, height/2)` = 0.85 m above ground. | Grid points are already at “sensor” height: **hit point + normal × 0.9 m**, so no extra height offset in the exposure shader. |
| **Count** | Fixed by the analysis (e.g. `num_positions` from metadata, 104k for Ben-Gurion 2 m). | Depends on mesh bounds and resolution (2 m): number of rays that hit “walkable” surfaces; often differs from `.bin` count. |

So **grids do not match**: different generation (precomputed vs mesh raycast), different coordinate systems, and different height semantics. Statistical comparison (mean, max) is the intended validation.

---

## 2. Sun vectors

| Aspect | Python (reference) | WebGPU |
|--------|--------------------|--------|
| **Source** | Baked in analysis metadata: `sun_positions` in the analysis `.json` (from the run that produced the `.bin`). Export script uses `sun_data_from_metadata(meta)` → same vectors as the original analysis. | Recomputed in the viewer: EPW is loaded, `getSunVectors(location, month, day)` from `sunpath.ts`, then `rotateZUpToYUp(dayVectors[hour])` per hour. |
| **Convention** | Stored in metadata as `vector` per hour; Python exposure uses “ray FROM position TO sun” (direction toward sun). | Viewer: Ladybug-style sun vectors rotated from Z-up to Y-up; shader uses direction **toward sun**. |
| **Hours** | `meta["hours"]` (e.g. 0–23). One vector per hour. | Same logical day (e.g. Aug 15); `numHours` from analysis metadata or 24. |

If EPW and location match the original analysis, sun vectors should be close but may differ slightly (different implementation or rounding). Any mismatch will affect solar exposure statistics.

---

## 3. Geometry / BVH

| Aspect | Python (reference) | WebGPU |
|--------|--------------------|--------|
| **Mesh** | `load_context_meshes([glb_path])` → trimesh (single or combined mesh). | Same GLB loaded in Three.js; merged in worker; geometry used to build BVH. |
| **BVH** | Trimesh/Embree (or configured intersector). Used in `batch_ray_intersections(origins, directions, mesh_context)`. | `three-mesh-bvh` in worker; serialized and uploaded to GPU; WGSL `bvh_raycast.wgsl` does traversal and Möller–Trumbore triangle intersection. |
| **Coordinate frame** | Mesh and rays in the same frame as the analysis (e.g. Z-up or the frame in which the GLB was authored). | Mesh and BVH in Three.js world (Y-up). Grid and sun vectors are in the same frame. |

Ray/geometry coordinate mismatch (e.g. if Python uses Z-up and WebGPU uses Y-up without equivalent transform) would cause systematic differences or all-hit/no-hit behavior.

---

## 4. Solar exposure

| Aspect | Python | WebGPU |
|--------|--------|--------|
| **Per position** | One or more **sample points** per position (e.g. `position + (0,0, height/2)` for pt_count=1). | One **grid point** per position (already at sensor height). |
| **Per hour** | For each hour: ray **origins** = sample points, ray **direction** = sun vector (toward sun). `batch_ray_intersections` → hit boolean per ray. Fraction exposed = fraction of rays that **miss** (no hit). | Shader: `origin = grid_points[point_idx]`, `ray_origin = origin + sun * 0.1`, direction = `sun`. `bvh_intersects_any(ray_origin, sun)` → hit. `solar_exposure[flat_index] = select(1.0, 0.0, hit)` (1 = no hit, 0 = hit). |
| **Output** | `fract_body_exp` per hour (0–1), then flattened in export as point-major: `[p0_h0, p0_h1, ..., pN_h23]`. | Buffer: point-major flat `numPoints × numHours`; read back and exposed as `solarExposure` array. |
| **Night** | `sun_data.is_sun_up[hour_idx]` → 0 if down. | Shader: `sun.y <= 0.0` or `dot(sun,sun) < 1e-10` → write 0. |

The **0.1** offset in the WebGPU shader (`origin + sun * 0.1`) is to reduce self-intersection; Python does not necessarily use the same offset (depends on intersector and mesh scale).

---

## 5. Sky exposure

| Aspect | Python | WebGPU |
|--------|--------|--------|
| **Dome** | Tregenza-like dome (e.g. 145 patches); vectors and weights from cache. Rays from sample point toward each patch; fraction of rays that miss = sky exposure. | Same idea: `getTregenzaDome()`, vectors rotated to Y-up, uploaded as `domeVectors` and `domeWeights`. Shader loops over patches, ray from grid point + `dir * 0.1`, `bvh_intersects_any` → hit; accumulate unoccluded weight. |
| **Output** | One scalar per position (0–1). | One value per grid point in `skyExposure` buffer. |

---

## 6. Inspecting WebGPU intermediates

To dump WebGPU solar/sky stats and sample values without running the full parity test:

```bash
cd viewer && npx playwright test tests/e2e/inspect-intermediates.spec.ts
```

This loads the Ben-Gurion debug page, waits for `__parityIntermediates__` (or `__parityIntermediatesError__`), then prints:

- `numPoints`, `numHours`
- Solar and sky stats: mean, min, max, std, n
- First 20 values of solar and sky arrays
- Optionally writes a short sample to `data/analyses/Ben-Gurion/20250815_grid_2m_fullday_webgpu_inspect.json`

Ensure the dev server is running (Playwright will start it if not). Use this to confirm whether WebGPU returns non-zero exposure and to compare distributions with the Python reference.

---

## 7. Summary

- **Grid**: Python uses `.bin` positions (precomputed); WebGPU uses mesh-derived grid (different count and positions). Same grid is not expected.
- **Sun**: Python uses metadata `sun_positions`; WebGPU recomputes from EPW and rotates to Y-up. Should be close if EPW/location match.
- **Geometry**: Same GLB; different BVH implementations (CPU vs GPU) and possibly different coordinate handling.
- **Solar**: Same idea (ray toward sun, no hit = exposed); different sample height (Python ground+0.85 m vs WebGPU surface+0.9 m) and different ray offset (WebGPU uses 0.1).
- **Sky**: Same idea (dome patches, miss = unoccluded); same dome; implementation details may differ.

Statistical comparison (mean, max) is the right level for intermediate validation given these path differences.
