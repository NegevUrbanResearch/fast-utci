# WebGPU quality and performance plan

## Goal and constraints

Improve the current WebGPU viewer before implementing the Grasshopper adapter. Work locally on `codex/webgpu-quality-2026-10-13`, based on `d3d824f`. Preserve existing untracked design documents, analysis data, scientific results, and the time picker design. Keep changes in focused commits; do not publish, merge, or implement the adapter.

The owner authorized implementation choices and requires measured performance evidence. Use the existing main-route, parity, selected-hour, interaction, and GPU ownership checks. Earlier timings are historical context; fresh captures on this machine establish the comparison.

## Design decision

Use incremental changes with separate correctness and performance gates. A broad engine rewrite would add parity risk before integration. A UI redesign would spend effort outside the demonstrated loading bottlenecks. Prioritize allocation overhead, cancellation and resource ownership, reliable quality checks, and reproducible startup evidence. Improve UI behavior when a reproducible issue supports the change.

## Work sequence

- [x] Capture the unchanged checkout: type check, unit tests, production build, Python tests, three sequential four-case main-route cold-start collections, and hardware/browser details.
- [x] Remove large intermediate JavaScript arrays from canonical grid creation and checksum calculation. Preserve float32 coordinates, axis order, offsets, point counts, and checksum values. Benchmark real 2m and 0.5m grids and verify byte identity.
- [x] Repair GPU output ownership on queue failures, cancellation, readback failure, and compatibility fallback. Keep successful output handles alive until their consumer releases them. Add regression tests before fixes.
- [x] Replace repeated CPU zero uploads with bounded GPU clears only if real-browser verification confirms correctness and no regression. Retain scheduling behavior and diagnostics.
- [x] Prevent stale asynchronous model and analysis loads from publishing after a newer selection or teardown. Verify overlap, failures, and teardown with deferred promises and browser project switching.
- [x] Separate Vitest and Playwright discovery, repair pre-existing unit failures, and restore type/unit/build checks in CI. Preserve the existing dependency versions unless a specific defect requires a change.
- [x] Constrain the local data-serving route to its data root and test valid paths and traversal attempts.
- [x] Isolate legacy Python package imports from optional heavyweight analysis dependencies and address reproduced scientific inconsistencies with parity tests. Avoid moving historical utilities without a demonstrated maintenance benefit.
- [x] Collect treatment performance sequentially, compare per-case medians and ranges with baseline artifacts, run scientific/browser checks, review the branch independently, and record limitations and remaining work.
- [x] Verify grouped and indirect BVH coverage, worker settlement, localized UTCI tooltip reads, compact CPU fallback, DataTexture compatibility, and truthful debug environment metadata through behavior regressions and real GPU checks.

## Verification and acceptance

Performance collections use the same datasets, Chromium configuration, diagnostics, machine, and initial hour. No unit/build/browser workloads overlap timing collections. Preserve raw samples and provenance. Report navigation-to-visible separately from the runtime's internal first-visible timing. A microbenchmark improvement alone does not establish faster full startup.

For the main WebGPU route, retain a shared compute/render device, GPU-resident selected-hour output, zero visible selected-hour readbacks, no selected-hour DataTexture construction, and no Python `.bin` comparison requests. Compare numerical outputs through the existing parity checks. Explain tolerance and unsupported cases rather than relaxing checks to pass.

Accept a performance change only with measured results and unchanged correctness. Treat small/noisy timing differences as inconclusive. Resource cleanup and race fixes can be accepted on regression evidence without claiming faster execution. Record all baseline failures before repairing them.

## Progress and evidence

Detailed task briefs, audit reports, command logs, and the progress ledger live under the ignored `.cache/quality-2026-10-13/` directory. Raw before-and-after performance JSON artifacts are retained locally under the ignored `data/performance-results/quality-2026-10-13/` directory. The results, final review, and evidence audit reports remain tracked. New tests cover actual behavior; avoid tests that merely mirror source text or mocked call sequences.

Completed outcomes, exact checks, raw evidence locations, and remaining performance limits are documented in [the results](2026-10-13-webgpu-quality-results.md).
