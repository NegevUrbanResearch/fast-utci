# Main Route Selected-Hour Render Diagnostics History

Date: 2026-05-17

## Scope

This note preserves historical diagnostic material split out from [main-route-selected-hour-render-diagnostics-next.md](main-route-selected-hour-render-diagnostics-next.md). It is supporting context for the current 0.5m main-route render-diagnostics evidence, not the live current-state summary.

For the current layout-reuse implementation recollection, next engineering step, and current JSON source, use [main-route-selected-hour-render-diagnostics-next.md](main-route-selected-hour-render-diagnostics-next.md).

## Historical Appendix: 2026-05-16 Pre-Implementation Full-Matrix Snapshot

The sections that follow preserve the earlier pre-layout-reuse diagnostic recollection that motivated the implementation. They are historical evidence only. Their scrub timings, `created` mesh actions, and "next step is layout reuse" conclusions are superseded by the 2026-05-17 implementation recollection above.

### Timing Table

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

### Render Publication Detail

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

### Render Publication Timeline

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

### Memory Table

| Project | Points | Grid | Grid width | Grid height | GPU VRAM MiB | Persistent exposure MiB | Selected-hour current MiB | Render-owned selected-hour MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ben-Gurion | 1,662,657 | 0.5 | 1,977 | 841 | 228.33 | 63.43 | 6.34 | 158.56 |
| Ness-Tziona | 8,171,761 | 0.5 | 2,237 | 3,653 | 1122.22 | 311.73 | 31.17 | 779.32 |

### Historical Inference

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

Historical diagnosis at that stage: the old route-to-child / first-observation / retry gap had been removed from the active publication window, but layout reuse was not yet implemented. `UTCIPointCloud` observed the pending surface almost immediately and started the completing attempt almost immediately. At that point the dominant remaining stall was still after that attempt started, before the storage polling loop began.

### Historical Diagnosis Update

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

This was still a proof update only. No optimization was claimed at that stage.

### Historical Reset-Proof Follow-Up

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

### Historical Storage-Wait Trace

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

### Historical Surface Mesh Trace

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

### Historical 2026-05-17 Full-Matrix Recollection Before Layout Reuse

Command:

```powershell
cd viewer
npx playwright test tests/e2e/main-route-performance-0_5m.spec.ts -g "collects BG and Ness Tziona" --project=chromium --workers=1 --reporter=list
```

Result: `1 passed`. The run refreshed [data/performance-results/main-route-selected-hour-render-diagnostics-next.json](../../data/performance-results/main-route-selected-hour-render-diagnostics-next.json).

This was the pre-implementation full-matrix recollection that confirmed the focused reuse proof generalized. At that stage every scrub sample used the update path with `layoutCompatible=true`, but the stable layout arrays were still being rebuilt on scrub.

| Project | Mode | Phase | Visible ms | Dispatch ms | Render update ms | Layout build ms | Surface mesh ms | Mesh action | Update mesh ms | Attempt -> wait start ms | Wait loop ms | Queue drain ms |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| Ben-Gurion | normalized | initial | 8101.2 | 6.8 | 1829.3 | 96.8 | 532.0 | created |  | 628.9 | 98.9 | 126.8 |
| Ben-Gurion | normalized | scrub | 228.6 | 56.2 | 166.6 | 91.4 | 31.1 | updated | 21.9 | 122.6 | 0.0 | 10.3 |
| Ben-Gurion | discrete | initial | 7295.2 | 9.8 | 999.8 | 103.0 | 556.5 | created |  | 659.6 | 95.7 | 125.3 |
| Ben-Gurion | discrete | scrub | 190.4 | 11.4 | 178.8 | 92.2 | 16.5 | updated | 6.7 | 108.7 | 0.0 | 14.6 |
| Ness-Tziona | normalized | initial | 22759.1 | 16.8 | 6449.1 | 472.4 | 2643.1 | created |  | 3115.6 | 485.5 | 503.6 |
| Ness-Tziona | normalized | scrub | 776.7 | 10.8 | 765.8 | 498.6 | 73.3 | updated | 31.5 | 571.9 | 0.0 | 3.1 |
| Ness-Tziona | discrete | initial | 20658.5 | 16.2 | 4301.0 | 462.2 | 2645.8 | created |  | 3108.0 | 423.9 | 541.8 |
| Ness-Tziona | discrete | scrub | 1433.7 | 13.7 | 1419.9 | 937.6 | 80.1 | updated | 31.5 | 1017.8 | 0.0 | 7.9 |

The final recollection also includes `renderLayoutBuildTrace` under each `renderPublicationTimeline`, splitting the layout build bucket:

| Project | Mode | Phase | Layout total ms | Array ms | Transform/bounds ms | Coordinate assignment ms | Index-to-texel ms | Cell-to-point ms | Color buffer ms |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ben-Gurion | normalized | scrub | 91.4 | 0.1 | 45.8 | 19.9 | 7.1 | 18.0 | 0.2 |
| Ben-Gurion | discrete | scrub | 92.2 | 0.1 | 46.7 | 19.6 | 7.2 | 18.1 | 0.1 |
| Ness-Tziona | normalized | scrub | 498.6 | 1.4 | 208.8 | 106.5 | 36.6 | 144.8 | 0.1 |
| Ness-Tziona | discrete | scrub | 937.0 | 2.1 | 392.6 | 235.0 | 99.4 | 206.7 | 0.7 |

Historical diagnosis after that full-matrix recollection:

- The scrub recreate problem was closed for the collected matrix. The structural reuse guard held for BG and Ness Tziona, normalized and discrete.
- The remaining Ness Tziona scrub cost at that time was mostly before storage polling started: `attempt -> wait start` was `571.9 ms` normalized and `1017.8 ms` discrete.
- The wait loop, queue drain, and storage-to-complete tail were no longer the dominant scrub costs in that recollection.
- `renderLayoutBuildMs` was the largest measured named bucket inside the remaining scrub delay: `498.6 ms` normalized and `937.6 ms` discrete for Ness Tziona.
- The split showed that the cost was not allocation or color-buffer creation. It was dominated by repeated geometry/grid derivation: transform/bounds, coordinate assignment, index-to-texel fill, and cell-to-point mapping.
- Code inspection showed `renderLayoutBuildMs` wrapping `extractUtciLayout(activeAnalysis)`, which calls `buildUtciGridLayout(activeAnalysis)`. That rebuilt stable layout arrays and mappings even when the selected-hour `Analysis` object reused the same base positions and structural grid.

### Historical Next Engineering Step

The next smallest safe proof should shift from splitting to proving safe layout reuse:

1. emit a scrub-local layout reuse/cache diagnostic keyed by stable structural inputs such as positions identity, point count, grid size, grid dimensions, normalization state, and cell-to-point mapping compatibility
2. prove whether selected-hour scrub can reuse stable layout metadata for the same analysis/grid before changing behavior
3. if that proof holds, implement a narrow layout-cache/reuse change for stable layout metadata and recollect the same matrix
4. keep cold-start/tiling separate until the scrub layout-reuse proof is resolved

This pass did not claim the full selected-hour visible delay was fixed. It claimed the previous accepted-output retry/reset delay had been removed from the active publication window, and the layout-compatible reuse proof removed compute-surface recreation from all scrub samples in the collected full matrix. That historical section is included because it explains why the later 2026-05-17 implementation work targeted layout extraction/build reuse.

### Historical 2026-05-17 Layout-Reuse Proof Recollection

This section is proof-only historical evidence from Task 2. It records the recollection that preceded the implementation and stops before any layout-reuse implementation work.

Command:

```powershell
cd viewer
npx playwright test tests/e2e/main-route-performance-0_5m.spec.ts -g "collects BG and Ness Tziona" --project=chromium --workers=1 --reporter=list
```

Result: `1 passed (1.5m)`.

Snapshot captured before recollection:

- `data/performance-results/main-route-layout-reuse-proof-before.json` (intentionally preserved worktree-local before snapshot for this change set; historical comparison aid, not a claimed tracked baseline artifact)

Collected proof fields per BG/Ness Tziona, normalized/discrete, initial/scrub sample:

- `decision`
- `canonicalRuntimeCompatibilityWouldReuse`
- `proofMatchesCanonicalRuntimeCompatibility`
- `positionsReferenceMatch`
- `pointCountMatch`
- `gridSizeMatch`
- `coordinateSystemMatch`
- `normalizationSignatureMatch`
- `constructionMode`
- `constructionModeMatch`
- `dimensionsMatch`
- `placementMatch`
- `cellToPointMappingMatch`
- `hoverCellLookupProofStatus`
- `proofCostMs`
- `estimatedRetainedCpuLayoutBytes`
- `renderLayoutBuildTrace.totalMs`

### Proof Summary

| Project | Mode | Phase | Decision | Canonical would reuse | Proof matches canonical | Positions ref | Point count | Grid size | Coord system | Normalization | Construction mode / match | Dimensions | Placement | Mapping compat | Hover proof | Proof cost ms | Retained CPU bytes | Layout total ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| Ben-Gurion | normalized | initial | `rebuild-required` |  |  |  |  |  |  |  | `world-positions` /  |  |  |  | `proof-inconclusive` | 0.0 | 33,253,140 | 98.9 |
| Ben-Gurion | normalized | scrub | `reuse-safe` | `true` | `true` | `true` | `true` | `true` | `true` | `true` | `world-positions` / `true` | `true` | `true` | `true` | `same-point-confirmed` | 0.1 | 33,253,140 | 109.5 |
| Ben-Gurion | discrete | initial | `rebuild-required` |  |  |  |  |  |  |  | `world-positions` /  |  |  |  | `proof-inconclusive` | 0.1 | 33,253,140 | 94.8 |
| Ben-Gurion | discrete | scrub | `reuse-safe` | `true` | `true` | `true` | `true` | `true` | `true` | `true` | `world-positions` / `true` | `true` | `true` | `true` | `same-point-confirmed` | 0.1 | 33,253,140 | 96.7 |
| Ness-Tziona | normalized | initial | `rebuild-required` |  |  |  |  |  |  |  | `world-positions` /  |  |  |  | `proof-inconclusive` | 0.1 | 163,435,220 | 459.0 |
| Ness-Tziona | normalized | scrub | `reuse-safe` | `true` | `true` | `true` | `true` | `true` | `true` | `true` | `world-positions` / `true` | `true` | `true` | `true` | `same-point-confirmed` | 0.0 | 163,435,220 | 641.1 |
| Ness-Tziona | discrete | initial | `rebuild-required` |  |  |  |  |  |  |  | `world-positions` /  |  |  |  | `proof-inconclusive` | 0.0 | 163,435,220 | 511.5 |
| Ness-Tziona | discrete | scrub | `reuse-safe` | `true` | `true` | `true` | `true` | `true` | `true` | `true` | `world-positions` / `true` | `true` | `true` | `true` | `same-point-confirmed` | 0.0 | 163,435,220 | 768.8 |

### Readout

- Every target scrub sample in this recollection is reuse-safe.
- Every scrub sample agrees with the canonical runtime compatibility predicate: `canonicalRuntimeCompatibilityWouldReuse=true` and `proofMatchesCanonicalRuntimeCompatibility=true`.
- Every scrub sample stays on the same construction mode, matches placement and dimensions, and confirms hover/cell lookup safety with `same-point-confirmed`.
- The initial samples remain `rebuild-required` because there is no previous layout to compare against; that is expected and is not a scrub-path falsifier.
- No scrub-path falsifiers or ambiguous scrub samples appeared in this recollection.
- The remaining scrub cost is still visible in `renderLayoutBuildTrace.totalMs`, especially for Ness Tziona at `641.1 ms` normalized and `768.8 ms` discrete.

### Historical Stop-Here Recommendation

At that point, the proof supported a later layout-reuse implementation plan. That recommendation is now historical; the implementation described in the 2026-05-17 recollection above is the follow-through on this proof surface.
