# Tasks: Near Real-Time MRT/Raycasting Performance

Feature Dir: D:/Projects/Nur/Shade/fast-utci/specs/main
Spec: D:/Projects/Nur/Shade/fast-utci/specs/main/spec.md
Plan: D:/Projects/Nur/Shade/fast-utci/specs/main/plan.md

Conventions:
- [P] = parallelizable with other [P] tasks (different files/subsystems)
- Paths are absolute to avoid ambiguity

## Ordered Tasks

T001. Project setup and environment switches [P]
- Add env flag selection for intersector: `FAST_UTCI_INTERSECTOR=embree|trimesh` (default auto)
- Add logging of active intersector in results/validation summary
- Files: D:/Projects/Nur/Shade/fast-utci/reader.py, D:/Projects/Nur/Shade/fast-utci/utci_calculator.py
- Dependency: none

T002. Profiling baseline runtime hotspots
- Add/modify lightweight timers around: scene load, grid gen, raycasting, MRT aggregation, IO
- Output CSV per stage per run
- Files: D:/Projects/Nur/Shade/fast-utci/viewer.py, D:/Projects/Nur/Shade/fast-utci/demo_utci_workflow_simplified_model.py
- Dependency: none

T003. Optional Embree path via trimesh.ray.ray_pyembree
- Implement try/except import and selection; on success, use Embree intersector
- Fallback automatically to existing trimesh ray intersector
- Files: D:/Projects/Nur/Shade/fast-utci/reader.py
- Dependency: T001

T004. Batched ray queries and vectorized math
- Refactor per-ray loops into batched arrays (directions X points)
- Ensure memory-bounded batches (configurable batch size)
- Files: D:/Projects/Nur/Shade/fast-utci/MRT/ray batching (likely MRT/mrt_calculator.py, MRT/mesh.py)
- Dependency: T003

T005. Stratified/importance angular sampling
- Implement deterministic stratified hemispherical samples; weight by sky model importance
- Provide presets: Fast/Balanced/Accurate → rays_per_point mapping
- Files: D:/Projects/Nur/Shade/fast-utci/MRT/solar.py, D:/Projects/Nur/Shade/fast-utci/MRT/mrt_calculator.py
- Dependency: T004

T006. Grid tiling and scheduler with cancellation
- Tile grid space; process tiles in a worker pool; support cancel-on-new-request
- Emit per-tile progress events and ETA
- Files: D:/Projects/Nur/Shade/fast-utci/enhanced_viewer.py, D:/Projects/Nur/Shade/fast-utci/MRT/grid.py
- Dependency: T004

T007. Progressive feedback modes
- Coarse-to-fine: early pass with fewer angles/rays, then refine to target
- Streaming tiles: publish completed tiles incrementally to UI/HTML
- Ensure outputs/legends explicitly label Partial vs Final results
- Files: D:/Projects/Nur/Shade/fast-utci/enhanced_viewer.py
- Dependency: T006

T008. Caching: sky factors and sun masks
- Cache visibility/occlusion per tile and sun position; invalidate on geometry changes
- Persist optionally to disk cache keyed by scene+config hash
- Files: D:/Projects/Nur/Shade/fast-utci/MRT/solar.py, D:/Projects/Nur/Shade/fast-utci/MRT/mrt_calculator.py
- Dependency: T005, T006

T008a. Guardrails for configuration safety (FR-009)
- Enforce minimum rays/point per preset; warn or block extreme grid density or angles that break accuracy budget
- Provide user-facing warnings with suggested fixes
- Files: D:/Projects/Nur/Shade/fast-utci/MRT/mrt_calculator.py, D:/Projects/Nur/Shade/fast-utci/viewer.py
- Dependency: T005

T009. Two-level BVH strategy for dynamic trees (design spike) [P]
- Prototype instance-level BVH for added tree assets without full rebuild
- Document feasibility and fallback
- Files: D:/Projects/Nur/Shade/fast-utci/research/spikes/two_level_bvh.md
- Dependency: T003

T010. Validation suite: latency/throughput/RMSE harness
- Script running baseline vs fast across sample scenes; compute RMSE (°C), latency, rays/s
- Output CSV and summary HTML
- Files: D:/Projects/Nur/Shade/fast-utci/run_automated_workflow.py
- Dependency: T005–T008

T011. Quickstart update and docs [P]
- Update quickstart.md with Embree optional install and env variable
- Add runtime selection note and troubleshooting
- Files: D:/Projects/Nur/Shade/fast-utci/specs/main/quickstart.md
- Dependency: T003

T012. Unit tests for sampling and batching [P]
- Tests for deterministic sampling sets; batch equivalence vs scalar path
- Files: tests/unit/test_sampling.py, tests/unit/test_batching.py
- Dependency: T004, T005

T013. Contract tests for progress/cancel/results [P]
- Tests for start, progress polling, cancel, partial vs final results
- Files: tests/contract/test_progress.py
- Dependency: T006, T007

T014. Performance optimization pass
- Based on profiling, tune batch sizes, thread pool, sampling presets to meet ≤ 60 s and ≤ 2.0 °C RMSE
- Produce before/after metrics (rays/s, wall time, peak RSS) and commit CSV/summary
- Files: D:/Projects/Nur/Shade/fast-utci/* (as needed)
- Dependency: T010

T015. Finalize and record metrics
- Capture benchmark results in repo (CSV + README)
- Update spec Clarifications if targets adjusted
- Include PR‑ready metrics snippet (before/after table) per constitution
- Files: D:/Projects/Nur/Shade/fast-utci/specs/main/research.md, D:/Projects/Nur/Shade/fast-utci/specs/main/spec.md
- Dependency: T014

## Parallel Execution Guidance
- Run [P] tasks in parallel where dependencies allow:
  - T001, T002, T009, T011, T012 can start early
  - After T003: T004/T005 proceed; T006 after T004; T007 after T006; T008 after T005+T006
  - T012 with T004/T005; T013 after T006/T007

## Example Agent Commands
- Run profiling: `python D:/Projects/Nur/Shade/fast-utci/demo_utci_workflow.py --profile 1`
- Force Embree: `setx FAST_UTCI_INTERSECTOR embree && python ...`
- Force fallback: `setx FAST_UTCI_INTERSECTOR trimesh && python ...`
