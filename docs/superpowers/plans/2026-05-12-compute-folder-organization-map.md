# Compute Folder Organization Map

Date: 2026-05-12

## Current Problem

`viewer/src/lib/compute/` currently mixes domain math, WebGPU pipeline code, selected-hour orchestration, on-demand diagnostics, weather parsing, workers, WGSL shaders, and route-facing contracts.

## Proposed Buckets

| Bucket | New path | Files |
| --- | --- | --- |
| Core/domain math | `viewer/src/lib/compute/core/` | `analysisGridFromBounds.ts`, `canonicalGrid.ts`, `grid-generator.ts`, `mrtReference.ts`, `solarcal.ts`, `sunpath.ts`, `tregenza.ts`, `utci.ts` |
| WebGPU pipeline and workers | `viewer/src/lib/compute/gpu/` | `bvhGpuUpload.ts`, `gpu-pipeline.ts`, `mergeAndBvh.worker.ts`, `mergeAndBvhWorkerClient.ts`, `meshMerger.ts`, `webgpuDeviceLimits.ts`, `webgpuUtciPipeline.ts`, `shaders/` |
| Selected-hour orchestration | `viewer/src/lib/compute/selected-hour/` | `liveSelectedHourController.ts`, `liveSelectedHourRenderContext.ts`, `liveSelectedHourRouteHost.ts`, `liveSelectedHourRouteProjection.ts`, `liveSelectedHourSurfaceIdentity.ts`, `liveUtciAnalysis.ts`, `liveUtciSelectedHour.ts`, `liveUtciSelectedHourSession.ts`, `selectedHourOutputHandle.ts` |
| On-demand diagnostics/state | `viewer/src/lib/compute/on-demand/` | `onDemandDiagnostics.ts`, `onDemandOutputFormat.ts`, `onDemandPrototypeStatus.ts`, `onDemandScrubState.ts`, `onDemandSizing.ts` |
| Weather | `viewer/src/lib/compute/weather/` | `epw-parser.ts`, `projectWeather.ts` |
| Intentional root leftovers | `viewer/src/lib/compute/` | `compute-manager.ts`, `telemetry.ts` |

## This Pass

This implementation pass first moved only the selected-hour orchestration bucket after the Ness Tziona selected-hour behavior gate was green. The focused Ness Tziona Playwright case and the full selected-hour E2E suite both passed before this map was written, so no behavior fix was bundled into that move.

Follow-up authorization in the same working tree also moved the WebGPU pipeline/shader bucket, the core/domain math bucket, the on-demand diagnostics/state bucket, and the weather bucket. The core move included the colocated `analysisGridFromBounds.test.ts`; parity-sensitive formulas remained unchanged.

## Roadmap-Only Buckets

The root-leftover bucket is intentionally not moved in this pass. `compute-manager.ts` remains the compute facade/orchestrator at root, and `telemetry.ts` remains the shared compute telemetry utility.

## Guardrails

- Preserve `runAll()`, `.bin`, Python comparison/reference paths, `readUtciBulk()`, `readUtcisSlice()`, `dataTexture`, debug parity, collect, and legacy selected-hour paths.
- Do not make the main route depend on `.bin` metadata, Python reference output, or debug globals.
- Keep `compute-manager.ts` at the compute root for this pass.
- Update exact imports, `vi.mock(...)` specifiers, and source-lock path reads for the selected-hour move only.
- Remove temporary import-map artifacts before final status.
