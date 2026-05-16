# Main Route Selected-Hour Render Diagnostics Evidence

Date: 2026-05-16

## Scope

This artifact is the first 0.5m main-route collection after adding render-publication diagnostics.
It keeps the protected pre-diagnostics baseline intact:

- `docs/performance/main-route-selected-hour-0_5m-base.md`
- `data/performance-results/main-route-selected-hour-0_5m-base.json`

JSON source: [data/performance-results/main-route-selected-hour-render-diagnostics-next.json](../../data/performance-results/main-route-selected-hour-render-diagnostics-next.json)

Focused reset-proof source: [data/performance-results/main-route-selected-hour-render-reset-diagnostics-next.json](../../data/performance-results/main-route-selected-hour-render-reset-diagnostics-next.json)

## Included Analyses

- `Ben-Gurion/20250815_grid_2m_fullday`
- `Ness-Tziona/exploded/nes_tziona_unblock_2`

## Collection Method

- Route: `/`
- Query: `gridResolution=0.5&utciRender=auto&utciRenderDiagnostics=1`
- Color modes: normalized/full-day and discrete/per-hour
- Scrub sample: app-visible hour slider scrub from hour `0` to hour `1`
- No debug route, no parity mode, no Python `.bin` comparison fields

## Proof Boundary

Both included analyses reported:

- `rendererBackend=webgpu`
- `utciRenderResolved=gpuNative`
- `utciSurfaceSource=compute-buffer-selected-hour`
- `baseRenderTransport=compute-buffer-selected-hour`
- `dataTextureBuildCount=0`
- `selectedHourRuntimeContract.strongVisibleGpuPath=true`
- `selectedHourRuntimeContract.visibleSelectedHourReadbackCount=0`
- no python/bin/debug comparison fields
- no forbidden comparison requests

Memory remains scoped to tracked app-owned UTCI/WebGPU buffers. `GPU VRAM` here means the route's tracked UTCI-owned total, not total browser, OS, or device VRAM.

## Timing Table

| Project | Mode | Phase | Points | First ready ms | First visible ms | Dispatch ms | Render update ms | Scene sync delay ms | Scene sync total ms |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ben-Gurion | normalized | initial | 1,662,657 | 9219.5 | 9805.4 | 6.1 | 3553.5 | 2570.2 | 982.8 |
| Ben-Gurion | normalized | scrub | 1,662,657 | 101.8 | 894.8 | 69.0 | 825.5 | 33.4 | 791.7 |
| Ben-Gurion | discrete | initial | 1,662,657 | 6830.8 | 7343.2 | 7.0 | 1079.3 | 101.3 | 977.5 |
| Ben-Gurion | discrete | scrub | 1,662,657 | 39.7 | 852.3 | 6.4 | 845.7 | 57.1 | 788.4 |
| Ness-Tziona | normalized | initial | 8,171,761 | 20633.2 | 23328.5 | 16.0 | 6873.2 | 2412.6 | 4460.1 |
| Ness-Tziona | normalized | scrub | 8,171,761 | 263.1 | 4242.7 | 125.0 | 4117.2 | 138.5 | 3978.4 |
| Ness-Tziona | discrete | initial | 8,171,761 | 18458.2 | 21499.6 | 15.9 | 4997.8 | 248.1 | 4749.1 |
| Ness-Tziona | discrete | scrub | 8,171,761 | 188.2 | 5028.1 | 10.9 | 5016.9 | 288.8 | 4727.7 |

## Render Publication Detail

| Project | Mode | Phase | Layout build ms | Storage init wait ms | Buffer copy ms | Queue drain ms | Mesh action | Source MiB | Target MiB |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| Ben-Gurion | normalized | initial | 100.4 | 130.5 | 0.0 | 145.0 | created | 6.34 | 6.34 |
| Ben-Gurion | normalized | scrub | 106.5 | 41.1 | 0.0 | 74.6 | created | 6.34 | 6.34 |
| Ben-Gurion | discrete | initial | 114.1 | 124.8 | 0.0 | 144.6 | created | 6.34 | 6.34 |
| Ben-Gurion | discrete | scrub | 103.0 | 42.7 | 0.0 | 61.5 | created | 6.34 | 6.34 |
| Ness-Tziona | normalized | initial | 463.9 | 530.2 | 0.0 | 712.8 | created | 31.17 | 31.17 |
| Ness-Tziona | normalized | scrub | 549.8 | 140.0 | 0.0 | 424.5 | created | 31.17 | 31.17 |
| Ness-Tziona | discrete | initial | 473.9 | 552.5 | 0.0 | 606.9 | created | 31.17 | 31.17 |
| Ness-Tziona | discrete | scrub | 476.3 | 134.2 | 0.0 | 1373.9 | created | 31.17 | 31.17 |

The diagnostics show that source and target selected-hour buffers are the same byte size. The expensive part is not the buffer copy itself; it is the render publication path around it: scene sync delay, storage init wait, layout build, and queue drain.

## Render Publication Timeline

This pass includes the nested timeline under `timings.renderPublication.renderPublicationTimeline`, including child-side first observation and the successful sync attempt start/token. `scenePendingSurfaceObservedAtMs` is first observation per sync key. `sceneSyncAttemptStartedAtMs` and `sceneSyncAttemptToken` describe the completing sync attempt, so the two fields deliberately expose retry/re-entry delay instead of pretending there is one strict single-attempt chain.

| Project | Mode | Phase | First visible ms | Dispatch ms | Accept -> pending ms | Pending -> observed ms | Observed -> attempt ms | Receive -> attempt ms | Attempt -> storage ms | Storage -> complete ms | Complete -> route ms | Attempt token | Active-window resets |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ben-Gurion | normalized | initial | 9805.4 | 6.1 | 1.6 | 1.6 | 1.1 | 0.8 | 837.6 | 145.2 | 0.7 | 2 | 0 |
| Ben-Gurion | normalized | scrub | 894.8 | 69.0 | 0.7 | 0.1 | 0.1 | 0.1 | 717.0 | 74.7 | 0.6 | 3 | 0 |
| Ben-Gurion | discrete | initial | 7343.2 | 7.0 | 0.7 | 0.3 | 0.2 | 0.2 | 832.9 | 144.6 | 0.7 | 2 | 0 |
| Ben-Gurion | discrete | scrub | 852.3 | 6.4 | 0.7 | 0.3 | 0.1 | 0.0 | 726.8 | 61.6 | 0.4 | 3 | 0 |
| Ness-Tziona | normalized | initial | 23328.5 | 16.0 | 0.7 | 0.4 | 0.5 | 0.4 | 3747.3 | 712.8 | 0.8 | 2 | 0 |
| Ness-Tziona | normalized | scrub | 4242.7 | 125.0 | 0.7 | 0.1 | 0.1 | 0.0 | 3553.9 | 424.5 | 0.4 | 3 | 0 |
| Ness-Tziona | discrete | initial | 21499.6 | 15.9 | 0.5 | 0.5 | 0.3 | 0.3 | 4142.1 | 607.0 | 1.0 | 2 | 0 |
| Ness-Tziona | discrete | scrub | 5028.1 | 10.9 | 0.7 | 0.2 | 0.2 | 0.2 | 3353.8 | 1373.9 | 0.6 | 3 | 0 |

## Memory Table

| Project | Points | Grid | Grid width | Grid height | GPU VRAM MiB | Persistent exposure MiB | Selected-hour current MiB | Render-owned selected-hour MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ben-Gurion | 1,662,657 | 0.5 | 1,977 | 841 | 228.33 | 63.43 | 6.34 | 158.56 |
| Ness-Tziona | 8,171,761 | 0.5 | 2,237 | 3,653 | 1122.22 | 311.73 | 31.17 | 779.32 |

## Current Inference

The recollected JSON now includes `scenePendingSurfaceObservedAtMs`, `sceneSyncAttemptStartedAtMs`, `sceneSyncAttemptToken`, scoped `sceneSyncActiveWindowResetHistory`, and `renderStorageWaitTrace` for every sample. That gives a cleaner split:

`controller accepted -> route pending exposed -> scene first observed -> completing sync attempt started -> storage ready -> scene complete -> route published`

The selected-hour UTCI dispatch is still small relative to the visible delay: `125.0 ms` for Ness Tziona normalized scrub and `10.9 ms` for Ness Tziona discrete scrub. The user-visible delay is therefore still not primarily the UTCI math.

The fresh scrub diagnosis is now tighter:

- Ness Tziona normalized scrub: route pending -> scene first observed `0.1 ms`; observed -> completing attempt `0.1 ms`; scene receive -> attempt start `0.0 ms`; attempt -> storage ready `3553.9 ms`; storage -> scene complete `424.5 ms`; dispatch `125.0 ms`; visible `4242.7 ms`; attempt token `3`; active-window resets `0`.
- Ness Tziona discrete scrub: route pending -> scene first observed `0.2 ms`; observed -> completing attempt `0.2 ms`; scene receive -> attempt start `0.2 ms`; attempt -> storage ready `3353.8 ms`; storage -> scene complete `1373.9 ms`; dispatch `10.9 ms`; visible `5028.1 ms`; attempt token `3`; active-window resets `0`.
- Ben-Gurion scrub control now shows the same reset-free shape at smaller scale: observed -> completing attempt is `0.1 ms` normalized and `0.1 ms` discrete, with attempt token `3`.
- Final route publication after scene sync is basically free in all scrub samples, around `0.4-1.0 ms`.
- The measured buffer copy remains effectively tiny, around `0.0-0.1 ms`.
- Mesh action is now `created` in these samples. The storage-readiness trace shows the expensive portion is mostly before the storage wait loop, in compute surface mesh setup/materialization rather than buffer copy.

Current diagnosis: the old route-to-child / first-observation / retry gap has been removed from the active publication window. `UTCIPointCloud` observes the pending surface almost immediately and starts the completing attempt almost immediately. The dominant remaining stall is now after that attempt starts, before the storage polling loop begins.

## Diagnosis Update

The recollection answers the main boundary question from the previous draft, the child-side timestamp proof, and the non-invalidating reset behavior proof:

- `scenePendingSurfaceObservedAtMs` is present in the JSON and lands almost immediately after route pending exposure.
- `sceneSyncAttemptStartedAtMs` is present in the JSON and lands almost immediately after `sceneSurfaceReceivedAtMs`.
- The old long Ness Tziona scrub delay is no longer before route exposure, between route exposure and child observation, between first observation and the completing attempt, inside the immediate successful `startSync()` handoff, or between scene completion and route publication.
- The dominant remaining gap is between the completing attempt start and render storage readiness, and the newer surface-mesh trace shows that most of this is before storage polling begins.
- `routePublishedAtMs` is now clearly just a trailing completion acknowledgement, not the primary bottleneck.

Important caveat: `scenePendingSurfaceObservedAtMs` is first observation per sync key, while `sceneSyncAttemptStartedAtMs` and `sceneSyncAttemptToken` belong to the completing attempt. The new scoped active-window reset history makes that distinction explicit. After the behavior proof, every full-matrix sample has zero active-window resets.

Top suspects are correspondingly shifted:

- compute surface mesh creation work before `waitForRenderStorageBuffer()` starts
- Three/TSL resource construction triggered by compute-surface recreation and storage attribute materialization
- less likely WebGPU device/backend absence during the wait loop; the trace shows device/backend entry are present immediately and the buffer appears after one frame once polling begins

Render surface setup before storage polling is now the first target. Buffer copy remains effectively tiny and route publication remains a trailing acknowledgement.

This is still a proof update only. No optimization is claimed here.

## Reset-Proof Follow-Up

A narrowed reset-proof collector now runs only the Ness Tziona 0.5m normalized path, initial load plus hour-0 -> hour-1 scrub. It writes the sibling artifact above instead of recollecting the full BG/Ness Tziona x normalized/discrete matrix.

Command:

```powershell
cd viewer
npx playwright test tests/e2e/main-route-performance-0_5m.spec.ts -g "reset proof" --project=chromium --workers=1 --reporter=list
```

Latest focused result after the non-invalidating compute-surface recreation behavior proof and the layout-compatible reuse proof:

| Phase | Visible ms | Dispatch ms | Pending -> observed ms | Observed -> attempt ms | Receive -> attempt ms | Attempt -> storage ms | Storage -> complete ms | Attempt token | Active-window resets |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| initial | 25294.0 | 15.1 | 1.1 | 0.8 | 0.6 | 4291.0 | 1429.3 | 2 | 0 |
| scrub | 1080.7 | 162.8 | 0.2 | 0.1 | 0.1 | 723.2 | 10.5 | 3 | 0 |

Relevant reset history from the scrub sample after the proof:

| Reset reason | Invalidates active run | Token before -> after | Reset timing relative to first observation | Reset timing before completing attempt |
| --- | --- | ---: | ---: | ---: |
| dispose-utci-surface | true | 0 -> 1 | -25324.4 ms | 25324.5 ms |

The stricter collector now also requires `sceneSyncActiveWindowResetHistory` to be present. In the scrub sample it is an emitted empty array. That means the old active-window `compute-surface-recreation` reset has been removed from the current accepted-output publication window.

Current proof outcome:

- The behavior proof succeeded for the reset/retry hypothesis: Ness Tziona normalized scrub `observed -> attempt` dropped from roughly `3.0 s` to `0.1 ms`.
- The scrub attempt token dropped from `5` to `3`, and there are no active-window resets in the completing scrub publication.
- The create-vs-update proof then showed the scrub recreate was caused by analysis object identity only: `missingSurface=false`, `notComputeBufferSurface=false`, `analysisIdentityChanged=true`, and `layoutCompatible=true`.
- The layout-compatible reuse proof changed the scrub mesh action from `created` to `updated`; after tightening the compatibility check to include point count and cell-to-point mapping, `renderSurfaceMeshMs` is `46.5 ms`, and the user-visible Ness Tziona normalized scrub sample is `1080.7 ms`.
- The remaining scrub delay is no longer compute-surface creation. It is now mostly layout build before storage polling: `attempt -> wait start` is `723.0 ms`, while storage wait is `0.1 ms` and storage -> complete is `10.5 ms`.

## Storage-Wait Trace

The storage-wait instrumentation splits `attempt -> storage ready` into:

`attempt start -> storage polling begins -> storage buffer appears -> storage ready`

In the full matrix, the WebGPU device and Three backend entry are already available on the first storage read. The render-owned buffer appears after one frame once polling begins. The long part is before the wait loop starts.

| Project | Mode | Phase | Attempt -> wait start ms | Wait loop ms | First buffer after wait start ms | Surface mesh ms | Layout build ms | Queue drain ms |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ben-Gurion | normalized | scrub | 675.9 | 41.1 | 41.1 | 569.4 | 106.5 | 74.6 |
| Ben-Gurion | discrete | scrub | 684.1 | 42.7 | 42.7 | 581.1 | 103.0 | 61.5 |
| Ness-Tziona | normalized | scrub | 3413.9 | 140.0 | 140.0 | 2864.0 | 549.8 | 424.5 |
| Ness-Tziona | discrete | scrub | 3219.6 | 134.2 | 134.2 | 2743.3 | 476.3 | 1373.9 |

Current storage diagnosis:

- The multi-second wait is not because the renderer WebGPU device is missing.
- It is not because Three's backend entry is missing throughout the wait.
- Once storage polling begins, the buffer appears after one frame: about `134-140 ms` for Ness Tziona scrub, `41-43 ms` for Ben-Gurion scrub.
- The expensive portion is before polling begins, dominated by `renderSurfaceMeshMs` and layout work after the sync attempt starts.

## Surface Mesh Trace

The latest full 0.5m collection adds `renderSurfaceMeshTrace`, splitting `renderSurfaceMeshMs` into the create/update branch and small follow-up work. The full matrix below is the pre-reuse-fix evidence that identified the recreate problem:

| Project | Mode | Phase | Visible ms | Surface mesh ms | Action | Create mesh ms | Layout build ms | Wait loop ms | Queue drain ms |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| Ben-Gurion | normalized | initial | 9805.4 | 606.2 | created | 605.8 | 100.4 | 130.5 | 145.0 |
| Ben-Gurion | normalized | scrub | 894.8 | 569.4 | created | 568.5 | 106.5 | 41.1 | 74.6 |
| Ben-Gurion | discrete | initial | 7343.2 | 593.9 | created | 593.5 | 114.1 | 124.8 | 144.6 |
| Ben-Gurion | discrete | scrub | 852.3 | 581.1 | created | 580.1 | 103.0 | 42.7 | 61.5 |
| Ness-Tziona | normalized | initial | 23328.5 | 2753.1 | created | 2749.4 | 463.9 | 530.2 | 712.8 |
| Ness-Tziona | normalized | scrub | 4242.7 | 2864.0 | created | 2863.1 | 549.8 | 140.0 | 424.5 |
| Ness-Tziona | discrete | initial | 21499.6 | 3115.5 | created | 3114.9 | 473.9 | 552.5 | 606.9 |
| Ness-Tziona | discrete | scrub | 5028.1 | 2743.3 | created | 2742.4 | 476.3 | 134.2 | 1373.9 |

The small substeps around the create call are not the cause in this run:

- `disposeResetMeshRemovalMs` is about `0.1-0.7 ms` in the full-matrix and focused reset-proof samples.
- `applySurfaceMeshStateMs`, `setCreatedSurfacePendingStorageInitMs`, `setPostSurfacePendingStorageInitMs`, `sceneAddMs`, and diagnostics publication are effectively `0.0-0.1 ms`.
- `updateComputeBufferSurfaceMeshMs` is absent because every collected sample took the `created` path, not the update/reuse path.

Code inspection explains why this bucket can scale with grid density. `createComputeBufferUtciSurfaceMesh()` in `viewer/src/lib/services/gpuUtciRenderBridge.ts` allocates and fills a full non-indexed surface geometry plus storage-backed lookup arrays:

- `createGpuNativeSurfaceGeometry(layout)` allocates `width * height * 6 * 3` float positions and loops every grid cell.
- `createVertexToPointIndexArray(layout)` allocates `width * height * 6` uint entries and loops every grid cell and surface vertex.
- the compute-buffer surface also allocates a render-owned UTCI storage attribute sized to `layout.numPositions` and builds node material/color LUT state.

For Ness Tziona 0.5m, that means rebuilding tens of millions of CPU-side vertex/lookup entries during selected-hour publication. The current evidence points at CPU-side surface recreation/materialization, not UTCI shader math, route publication, buffer copy, scene add, or the storage polling loop.

The important open question is why scrub still reaches the create branch. In `viewer/src/lib/components/scene/UTCIPointCloud.svelte`, the update path is only used when an existing `utciSurface` is compute-buffer backed and `activeAnalysis === lastAnalysis`. The collected trace shows `action: created` for every sample, so either the compute surface is being disposed/replaced between accepted outputs, or the analysis identity is changing from the scene component's point of view. That is the next proof boundary.

The focused proof answered that boundary for Ness Tziona normalized scrub: the existing surface was present, compute-buffer backed, and layout-compatible, while `analysisIdentityChanged=true`. After replacing the object-identity recreate guard with the structural layout/update safety check, the same focused scrub path updated the existing surface:

| Phase | Visible ms | Surface mesh ms | Mesh action | Create mesh ms | Update mesh ms | Layout build ms | Wait loop ms | Storage -> complete ms | Recreate decision |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| initial | 25294.0 | 3008.1 | created | 3007.1 |  | 595.5 | 686.5 | 1429.3 | `missingSurface=true`, `analysisIdentityChanged=true`, `layoutCompatible=false` |
| scrub | 1080.7 | 46.5 | updated |  | 18.3 | 676.5 | 0.1 | 10.5 | `missingSurface=false`, `notComputeBufferSurface=false`, `analysisIdentityChanged=true`, `layoutCompatible=true` |

## Next Engineering Step

The next smallest safe proof should shift to the remaining post-fix scrub cost:

1. keep the layout-compatible reuse fix if the full matrix confirms the same update path outside the focused reset-proof sample
2. split `renderLayoutBuildMs` for selected-hour scrub, because it is now the dominant focused Ness Tziona normalized scrub cost at about `676 ms`
3. check whether `extractUtciLayout(activeAnalysis)` rebuilds grid mapping from a fresh selected-hour `Analysis` object when it could reuse stable layout metadata for the same analysis/grid
4. run the full BG/Ness Tziona normalized/discrete collection after the focused proof remains stable

This pass does not claim the full selected-hour visible delay is fixed. It does claim the previous accepted-output retry/reset delay has been removed from the active publication window, and the layout-compatible reuse proof removes compute-surface recreation from the focused Ness Tziona scrub path. The next bottleneck is layout extraction/build work before storage polling.
