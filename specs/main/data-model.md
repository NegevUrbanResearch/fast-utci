# Data Model (Phase 1)

## Entities

### Scene
- id
- meshes: list of mesh descriptors (name, triangle count, material tags)
- bounds: AABB
- metadata: source file, units, date

### SamplingGrid
- id
- spacing_m: float
- extent: polygon or bbox
- points: generated lazily by tiling (tile_id, x, y, z)

### RaycastConfig
- quality_preset: enum(Fast,Balanced,Accurate)
- rays_per_point: int
- angular_scheme: enum(stratified,importance)
- seed: int (deterministic mode)

### ComputeJob
- id
- scene_id
- grid_id
- config_id
- status: enum(queued,running,cancelled,completed)
- progress: 0..1
- eta_s: float
- started_at, finished_at

### TileResult
- job_id
- tile_id
- mrt_values: array
- utc i_values: array
- partial: bool

### ValidationSummary
- job_id
- latency_s
- throughput_rays_per_s
- rmse_utci_c
- max_error_utci_c
