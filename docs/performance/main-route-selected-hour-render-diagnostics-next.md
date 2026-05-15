# Main Route Selected-Hour Render Diagnostics Evidence

Date: 2026-05-15

## Scope

This artifact is the first 0.5m main-route collection after adding render-publication diagnostics.
It keeps the protected pre-diagnostics baseline intact:

- `docs/performance/main-route-selected-hour-0_5m-base.md`
- `data/performance-results/main-route-selected-hour-0_5m-base.json`

JSON source: [data/performance-results/main-route-selected-hour-render-diagnostics-next.json](../../data/performance-results/main-route-selected-hour-render-diagnostics-next.json)

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
| Ben-Gurion | normalized | initial | 1,662,657 | 7870.5 | 8420.3 | 5.4 | 2109.7 | 1724.2 | 385.1 |
| Ben-Gurion | normalized | scrub | 1,662,657 | 102.5 | 1069.7 | 5.0 | 1064.2 | 833.2 | 230.7 |
| Ben-Gurion | discrete | initial | 1,662,657 | 6822.5 | 7517.8 | 6.5 | 1277.3 | 873.1 | 403.9 |
| Ben-Gurion | discrete | scrub | 1,662,657 | 47.9 | 1030.7 | 12.6 | 1017.8 | 794.5 | 223.1 |
| Ness-Tziona | normalized | initial | 8,171,761 | 20651.2 | 23896.7 | 16.5 | 7453.2 | 5748.1 | 1704.8 |
| Ness-Tziona | normalized | scrub | 8,171,761 | 195.6 | 4622.7 | 24.9 | 4597.4 | 3443.1 | 1154.1 |
| Ness-Tziona | discrete | initial | 8,171,761 | 18451.8 | 21692.2 | 16.9 | 5157.0 | 3496.4 | 1659.8 |
| Ness-Tziona | discrete | scrub | 8,171,761 | 168.8 | 4852.2 | 10.8 | 4841.2 | 3682.4 | 1158.6 |

## Render Publication Detail

| Project | Mode | Phase | Layout build ms | Storage init wait ms | Buffer copy ms | Queue drain ms | Mesh action | Source MiB | Target MiB |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| Ben-Gurion | normalized | initial | 96.2 | 132.6 | 0.0 | 156.0 | reused | 6.34 | 6.34 |
| Ben-Gurion | normalized | scrub | 117.9 | 53.7 | 0.0 | 59.1 | reused | 6.34 | 6.34 |
| Ben-Gurion | discrete | initial | 107.2 | 146.5 | 0.0 | 150.0 | reused | 6.34 | 6.34 |
| Ben-Gurion | discrete | scrub | 89.3 | 68.0 | 0.0 | 65.6 | reused | 6.34 | 6.34 |
| Ness-Tziona | normalized | initial | 492.3 | 561.4 | 0.0 | 650.7 | reused | 31.17 | 31.17 |
| Ness-Tziona | normalized | scrub | 551.7 | 165.5 | 0.0 | 436.8 | reused | 31.17 | 31.17 |
| Ness-Tziona | discrete | initial | 477.6 | 535.5 | 0.0 | 646.3 | reused | 31.17 | 31.17 |
| Ness-Tziona | discrete | scrub | 557.4 | 159.4 | 0.0 | 441.5 | reused | 31.17 | 31.17 |

The diagnostics show that source and target selected-hour buffers are the same byte size. The expensive part is not the buffer copy itself; it is the render publication path around it: scene sync delay, storage init wait, layout build, and queue drain.

## Render Publication Timeline

This pass adds the nested timeline under `timings.renderPublication.renderPublicationTimeline`. The route projection and scene receipt branches can interleave, so the table reports the two useful causal branches rather than assuming one strict total order.

| Project | Mode | Phase | First visible ms | Render update ms | Accept -> route publish ms | Accept -> scene receive ms | Effect -> storage ready ms | Storage ready -> complete ms | Accept -> complete ms |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ben-Gurion | normalized | initial | 8420.3 | 2109.7 | 1070.7 | 684.8 | 229.0 | 156.1 | 1070.0 |
| Ben-Gurion | normalized | scrub | 1069.7 | 1064.2 | 967.4 | 736.1 | 171.6 | 59.1 | 966.9 |
| Ben-Gurion | discrete | initial | 7517.8 | 1277.3 | 1165.2 | 740.6 | 253.8 | 150.1 | 1164.8 |
| Ben-Gurion | discrete | scrub | 1030.7 | 1017.8 | 959.3 | 735.7 | 157.5 | 65.6 | 958.9 |
| Ness-Tziona | normalized | initial | 23896.7 | 7453.2 | 5012.0 | 3306.7 | 1053.9 | 650.9 | 5011.5 |
| Ness-Tziona | normalized | scrub | 4622.7 | 4597.4 | 4427.1 | 3272.5 | 717.3 | 436.8 | 4426.8 |
| Ness-Tziona | discrete | initial | 21692.2 | 5157.0 | 4904.0 | 3242.7 | 1013.2 | 646.7 | 4902.8 |
| Ness-Tziona | discrete | scrub | 4852.2 | 4841.2 | 4567.2 | 3408.2 | 717.0 | 441.6 | 4566.9 |

## Memory Table

| Project | Points | Grid | Grid width | Grid height | GPU VRAM MiB | Persistent exposure MiB | Selected-hour current MiB | Render-owned selected-hour MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ben-Gurion | 1,662,657 | 0.5 | 1,977 | 841 | 228.33 | 63.43 | 6.34 | 158.56 |
| Ness-Tziona | 8,171,761 | 0.5 | 2,237 | 3,653 | 1122.22 | 311.73 | 31.17 | 779.32 |

## Current Inference

The selected-hour UTCI dispatch is still small: about `5-13 ms` for Ben-Gurion and `11-25 ms` for Ness Tziona. The user-visible delay is therefore not the UTCI math.

The deeper render diagnostics point to the handoff between accepted selected-hour output and visible scene publication:

- Ness Tziona scrub still takes about `4.6-4.9 s` to become visible.
- Of that, about `4.4-4.6 s` is from controller acceptance to scene-sync completion.
- The route-publish branch accounts for almost the whole delay: about `4.4-4.6 s` from controller acceptance to route publication on Ness Tziona scrub, and about `1.0 s` on Ben-Gurion scrub.
- The scene branch receives the surface earlier than route publication in the current diagnostics: about `3.3-3.4 s` after controller acceptance on Ness Tziona scrub, and about `0.74 s` on Ben-Gurion scrub.
- Once the scene publication effect starts, the remaining GPU/render-storage work is much smaller: about `1.16 s` on Ness Tziona scrub and about `0.22 s` on Ben-Gurion scrub.
- The measured buffer copy is effectively tiny, around `0.0-0.1 ms`.
- Mesh action is `reused` in all samples, so the slowdown is not coming from full mesh recreation.

Best next engineering target: investigate why route publication and scene receipt are delayed for the accepted selected-hour output, especially on Ness Tziona. The data still does not support optimizing the UTCI compute shader as the first move.

## Next Engineering Step

The broad delay has now been split into smaller handoff timestamps:

1. selected-hour compute finishes
2. controller accepts the selected-hour output
3. route host publishes the accepted output
4. main route projects it into scene props
5. scene component receives the selected-hour surface
6. `UTCIPointCloud` publication effect starts
7. render storage buffer is ready
8. scene sync completes

The next pass should be a focused code/architecture review of the largest new gap: accepted selected-hour output -> route publication / scene receipt. Do not start with another broad timing pass. Use small extra diagnostics only if the review finds an ambiguous boundary.

Recommended review scope for the next agent:

1. `viewer/src/lib/compute/selected-hour/liveSelectedHourController.ts`
   - Confirm accepted output is committed immediately after `runSelectedHour()` returns.
   - Check whether runtime diagnostics, output ownership, or release bookkeeping can delay publication.
2. `viewer/src/lib/compute/selected-hour/liveSelectedHourRouteHost.ts`
   - Inspect route-host subscription/notification cadence.
   - Look for throttling, batching, stale-surface guards, or scene-acknowledgement coupling that could delay `routePublishedAtMs`.
   - Verify base/comparison paths are not contending or causing extra readback/accounting work on the visible base path.
3. `viewer/src/routes/main/liveSelectedHour.ts` and `viewer/src/routes/+page.svelte`
   - Check whether Svelte reactive scheduling, derived state, or diagnostics publication waits behind expensive synchronous work.
   - Ensure the page is not passing large objects through reactive state in a way that causes expensive identity churn.
4. `viewer/src/lib/components/scene/UTCIPointCloud.svelte`
   - Review the path from scene surface receipt to publication effect start.
   - Confirm mesh reuse is real and no hidden geometry/material rebuild happens during scrub.
   - Check whether render-owned storage waits are GPU queue pressure, Three/Threlte lifecycle timing, or main-thread scheduling.

Questions the review should answer:

- Why does Ness Tziona spend about `4.4-4.6 s` from controller acceptance to route publication / completion during scrub?
- Why does Ben-Gurion show the same pattern at a smaller scale, around `1.0 s`?
- Is the delay proportional to point count, render-owned storage size, Svelte reactive work, or GPU queue pressure?
- Does cold start share the same post-acceptance bottleneck, or does it add a separate earlier exposure/BVH/upload freeze?
- What is the smallest low-risk proof change that should be tried first?

This is related to, but still not proven to be the same as, the broader 0.5m "cold start" symptom where the browser appears to keep calculating while the desktop visually stops updating until it releases. The current evidence separates two questions:

- Cold start / first load freeze: likely tied to the large exposure precompute, BVH/upload, GPU scheduling, or main-thread work during initial 0.5m setup.
- Selected-hour scrub delay: now measured as a handoff/publication delay after selected-hour compute is already quick.

The timeline suggests a shared pattern: both initial and scrub samples spend most visible delay after controller acceptance, before publication completes. But the cold-start visual freeze can still include earlier exposure/BVH/upload pressure that the scrub case avoids, so the next optimization should stay targeted and evidence-driven.
