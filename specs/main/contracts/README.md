Contracts (Phase 1)

Scope: CLI/UI integration points for running fast MRT/UTCI compute and validation.

Endpoints (conceptual):
- start_compute(scene_id, grid_id, config): returns job_id
- get_progress(job_id): returns status, progress, eta_s
- cancel(job_id): cancels running job
- get_results(job_id, tile_id?): returns partial/final results
- validate_against_baseline(job_id, baseline_job_id): returns ValidationSummary

Note: Actual implementation may be in-process APIs or CLI commands rather than HTTP.


