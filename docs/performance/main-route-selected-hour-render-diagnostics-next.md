# Main Route Selected-Hour Render Diagnostics Evidence

Date: 2026-05-17

## Scope

This note is the canonical 0.5m main-route render-diagnostics evidence surface for the current 2026-05-17 layout-reuse implementation recollection.
The current-state story is the 2026-05-17 implementation section below. Older pre-implementation diagnostics were split into [main-route-selected-hour-render-diagnostics-history.md](main-route-selected-hour-render-diagnostics-history.md) and should not be read as the live scrub behavior.

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
- Scrub sample: app-visible hour slider scrub from hour `0` to hour `2`, with hour `1` as the warmup scrub that establishes previous-publication proof
- Repeated-scrub soak: Ness Tziona normalized reusable scrubs at hours `2`, `3`, and `4`, then an in-session `gridResolution=2` rebuild to stamp released-layout ownership
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

## 2026-05-17 Layout Reuse Implementation Recollection

Collector command:

```powershell
cd viewer
npx playwright test tests/e2e/main-route-performance-0_5m.spec.ts --project=chromium --workers=1 --reporter=list --grep "collects BG and Ness Tziona 0.5m main-route timing, memory, color-mode, and scrub samples"
```

Result: `1 passed (2.2m)` on 2026-05-17. The refreshed collector updated [data/performance-results/main-route-selected-hour-render-diagnostics-next.json](../../data/performance-results/main-route-selected-hour-render-diagnostics-next.json) and compared against [data/performance-results/main-route-layout-reuse-implementation-before.json](../../data/performance-results/main-route-layout-reuse-implementation-before.json), which is an intentionally preserved worktree-local before snapshot for this change set and not a claimed tracked baseline artifact.

### Before/After Scrub Comparison

| Project | Mode | Visible ms before -> after | Saved ms | Render update ms before -> after | Reuse action | Build trace before -> after | Mesh ms before -> after | Queue ms before -> after | Retained CPU layout MiB | App-owned UTCI/WebGPU MiB |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| Ben-Gurion | normalized | 220.2 -> 113.3 | 106.9 | 156.8 -> 106.1 | `build-required` -> `reused` (`reuse-safe`, decision `46.6`) | `109.5` total (`56.4` transform, `26.9` coord, `7.4` texel, `18.5` cell map) -> skipped | 0.3 -> 0.2 | 3.2 -> 2.8 | 31.7 | 196.6 |
| Ben-Gurion | discrete | 194.8 -> 122.6 | 72.2 | 180.2 -> 114.9 | `build-required` -> `reused` (`reuse-safe`, decision `46.2`) | `96.7` total (`50.1` transform, `20.2` coord, `7.4` texel, `18.5` cell map) -> skipped | 0.1 -> 0.3 | 12.8 -> 3.3 | 31.7 | 196.6 |
| Ness-Tziona | normalized | 922.9 -> 454.7 | 468.2 | 907.9 -> 438.1 | `build-required` -> `reused` (`reuse-safe`, decision `247.3`) | `641.1` total (`340.7` transform, `112.3` coord, `43.4` texel, `142.4` cell map) -> skipped | 0.3 -> 0.1 | 7.1 -> 3.8 | 155.9 | 966.4 |
| Ness-Tziona | discrete | 1133.6 -> 548.7 | 584.9 | 1121.2 -> 537.7 | `build-required` -> `reused` (`reuse-safe`, decision `242.5`) | `768.8` total (`385.9` transform, `194.8` coord, `39.6` texel, `145.5` cell map) -> skipped | 0.3 -> 0.3 | 6.8 -> 3.3 | 155.9 | 966.4 |

All four target scrub samples now keep the route proof clean: `rendererBackend=webgpu`, `utciRenderResolved=gpuNative`, `utciSurfaceSource=compute-buffer-selected-hour`, `dataTextureBuildCount=0`, forbidden comparison request count `0`, and `activeLayoutCandidateCount=1`.

### Skipped Rebuild Confirmation

- All four refreshed scrub samples now report `renderLayoutReuseAction='reused'` and `renderLayoutReuseReason='reuse-safe'`.
- `renderLayoutBuildTrace` is `null` for every refreshed scrub sample, so the old transform/bounds, coordinate assignment, index-to-texel, and cell-to-point rebuild work is fully skipped.
- The warm scrub publication now reuses the existing compute-buffer mesh instead of recreating it: `renderPublicationMeshAction='reused'` and `renderSurfaceMeshTrace.action='updated'`.
- Reused layout identities are stable per analysis/mode scrub family:
  - Ben-Gurion: `Ben-Gurion/20250815_grid_2m_fullday|v1:6b909fa3|1977|841|0|0|-0.050000000111758605`
  - Ness-Tziona: `Ness-Tziona/exploded/nes_tziona_unblock_2|v1:24c829f0|2237|3653|0|0|-0.05`

### Repeated-Scrub Retained-Bytes Soak

The refreshed collector now includes a repeated Ness Tziona normalized soak under `repeatedScrubSoak`. It performs the reusable warm scrub at hour `2` and then continues through hours `3` and `4` on the same analysis/grid before forcing a rebuild.

- Every reusable scrub sample (`2`, `3`, `4`) reports `renderLayoutReuseAction='reused'`, `renderLayoutReuseReason='reuse-safe'`, `activeLayoutCandidateCount=1`, and `hoverCellLookupProofStatus='same-point-confirmed'`.
- `reusedLayoutIdentity` stays stable across all reusable scrubs: `Ness-Tziona/exploded/nes_tziona_unblock_2|v1:24c829f0|2237|3653|0|0|-0.05`.
- Retained CPU layout bytes plateau at `163,435,220` bytes (`155.9 MiB`) across all reusable scrubs.
- Tracked app-owned UTCI/WebGPU memory also plateaus across all reusable scrubs at `1,013,299,388` bytes (`966.4 MiB`), with `renderOwnedSelectedHourBytes=653,741,904` and the same high-watermark on every reusable sample.
- The hover proof stays on the same point (`positionIndex=4034742`) while UTCI values change by hour, which is the expected "same layout, new selected-hour values" shape.
- When a rebuild replaces the active layout (forced in-session by `gridResolution=2` at hour `4`), `releasedPreviousLayout` is stamped with that same Ness Tziona identity before the new smaller layout takes over.

### Timing Interpretation

The saved time is the deleted rebuild work itself. Ness Tziona scrub now saves about half a second on the visible path: `468.2 ms` normalized and `584.9 ms` discrete. Queue drain is already small (`3.3-3.8 ms`), storage wait is effectively gone (`0.0 ms`), and surface mesh work is already tiny (`0.1-0.3 ms`).

That means the remaining warm-scrub bottleneck is no longer layout construction. The next focus should stay on the render-side warm update path, specifically the reuse-decision plus route-to-scene sync/publication window (`renderUpdateMs` still `438.1-537.7 ms` for Ness Tziona scrub), rather than pivoting to cold-start/init first.

## 2026-05-17 Remaining Warm-Scrub Window Split

Follow-up collector command:

```powershell
cd viewer
npx playwright test tests/e2e/main-route-performance-0_5m.spec.ts --project=chromium --workers=1 --reporter=list --grep "collects BG and Ness Tziona 0.5m main-route timing, memory, color-mode, and scrub samples"
```

Result: `1 passed (2.2m)` on 2026-05-17. This pass added diagnostics-only scalar stamps for selected-hour value publication start and controller visible acknowledgement, and preserved the existing reuse-key timing fields in the collected artifact.

Important interpretation caveat: `routePublishedAtMs` remains a trailing route-host acknowledgement after scene sync, not the first route-to-scene handoff. The table below therefore keeps route acknowledgement as a small tail and does not classify `controllerAcceptedAtMs -> routePublishedAtMs` as "route work".

### Warm Scrub Split

| Project | Mode | Visible ms | Render update ms | Value publish start -> controller accept ms | Accept -> pending exposed ms | Pending -> scene observed ms | Observed -> sync attempt ms | Attempt -> storage ready ms | Storage -> scene complete ms | Scene complete -> controller visible ack ms | Route ack tail ms | Reuse decision ms | Reuse key ms | Mesh ms | Queue ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ben-Gurion | normalized | 124.2 | 113.6 | 42.2 | 1.2 | 0.2 | 0.2 | 66.2 | 3.4 | 0.2 | 0.3 | 57.2 | 57.1 | 0.2 | 3.3 |
| Ben-Gurion | discrete | 155.6 | 147.6 | 72.8 | 0.6 | 0.3 | 0.2 | 69.6 | 3.9 | 0.2 | 0.2 | 45.6 | 45.5 | 0.1 | 3.9 |
| Ness-Tziona | normalized | 532.3 | 403.7 | 124.2 | 0.6 | 0.1 | 0.1 | 275.3 | 3.1 | 0.3 | 0.4 | 241.5 | 241.4 | 0.1 | 3.1 |
| Ness-Tziona | discrete | 680.9 | 530.7 | 261.9 | 0.8 | 0.2 | 0.2 | 263.8 | 3.4 | 0.4 | 0.5 | 230.4 | 230.2 | 0.2 | 3.3 |

All four scrub samples still report `renderLayoutReuseAction='reused'`, `renderLayoutReuseReason='reuse-safe'`, `renderLayoutBuildTrace=null`, `activeLayoutCandidateCount=1`, and `hoverCellLookupProofStatus='same-point-confirmed'`.

### Diagnosis

The remaining Ness Tziona warm-scrub window is now split cleanly enough to choose the next target:

- Route/controller publication is not the dominant bucket. Controller accept -> pending exposure is `0.6-0.8 ms`, route acknowledgement after scene completion is `0.4-0.5 ms`, and route projected tail is about `0.5-0.6 ms`.
- Scene receipt and final visible acknowledgement are also tiny. Pending -> scene observed and observed -> sync attempt are each around `0.1-0.2 ms`; scene complete -> controller visible acknowledgement is `0.3-0.4 ms`.
- Storage wait, mesh update, and queue drain remain small on warm scrubs: storage wait is `0.0 ms`, mesh update is `0.1-0.2 ms`, and queue drain is `3.1-3.3 ms` for Ness Tziona.
- The two dominant named buckets are reuse-key/layout-reuse decision (`230.2-241.4 ms` of reuse-key work for Ness Tziona) and the scene-side attempt -> storage-ready span (`263.8-275.3 ms`). These two buckets together explain almost all of the remaining `403.7-530.7 ms` Ness Tziona render update.
- The selected-hour value publication start -> controller accept span is `124.2 ms` normalized and `261.9 ms` discrete for Ness Tziona. Because `selectedHourValuePublicationStartedAtMs` is stamped from the session's render-publication start and `computeCompletedAtMs` is stamped after the session result returns, this span should be read as the controller/session handoff tail, not as a pure GPU compute duration.

### Recommended Next Step

The next action should be optimization, but still narrow and evidence-gated: investigate the reuse-key/frame derivation work and the scene attempt -> storage-ready span before touching cold-start/init. The leading hypothesis is that warm scrubs are paying repeated structural/frame derivation work even though layout reuse is accepted and the mesh/storage path is already cheap.

Cold-start/init remains a separate performance project. It should not be the next scrub-path move unless the product priority shifts away from warm interaction.

## 2026-05-17 Coupled Pre-Storage Split

Follow-up collector command:

```powershell
cd viewer
npx playwright test tests/e2e/main-route-performance-0_5m.spec.ts --project=chromium --workers=1 --reporter=list --grep "collects BG and Ness Tziona 0.5m main-route timing, memory, color-mode, and scrub samples"
```

Result: `1 passed (2.2m)` on 2026-05-17. This pass split the two suspected coupled buckets together: reuse-key/frame derivation and `sceneSyncAttemptStartedAtMs -> renderStorageReadyAtMs`.

### Coupled Bucket Split

| Project | Mode | Visible ms | Render update ms | Reuse key ms | Frame derivation ms | Positions signature cache hit | Scene attempt -> storage wait start ms | Storage wait start -> ready ms | Storage wait trace ms | Key complete -> plan ready ms | Storage ready -> scene complete ms |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Ben-Gurion | normalized | 111.2 | 103.6 | 45.1 | 45.0 | true | 68.3 | 0.1 | 0.0 | 23.0 | 2.9 |
| Ben-Gurion | discrete | 144.0 | 135.1 | 44.6 | 44.6 | true | 51.8 | 0.1 | 0.0 | 6.8 | 4.8 |
| Ness-Tziona | normalized | 529.7 | 401.4 | 235.0 | 235.0 | true | 268.2 | 0.1 | 0.0 | 32.9 | 3.1 |
| Ness-Tziona | discrete | 664.8 | 582.5 | 242.1 | 242.1 | true | 275.7 | 0.1 | 0.0 | 33.3 | 3.4 |

All four scrub samples still report `renderLayoutReuseAction='reused'`, `renderLayoutReuseReason='reuse-safe'`, `renderLayoutBuildTrace=null`, `activeLayoutCandidateCount=1`, and `hoverCellLookupProofStatus='same-point-confirmed'`.

### Diagnosis

This pass answers the 2+3 coupling question:

- The previous `sceneSyncAttemptStartedAtMs -> renderStorageReadyAtMs` bucket was not a separate storage-ready bottleneck. It includes reuse-key/frame derivation because `sceneSyncAttemptStartedAtMs` is stamped before layout-key work starts.
- Actual storage readiness is effectively free on warm scrubs: `renderStorageWaitStartedAtMs -> renderStorageReadyAtMs` is about `0.1 ms`, while `renderStorageWaitTrace.waitMs` remains `0.0 ms`.
- The dominant warm-scrub CPU work is repeated frame derivation inside reuse-key construction: Ness Tziona spends `235.0-242.1 ms`, and Ben-Gurion spends `44.6-45.1 ms`.
- `positionsSourceSignatureCacheHit=true` and `positionsSourceSignatureMs=0.0`, so this is not rehashing the shared positions buffer. The repeated work is frame derivation itself.
- `renderLayoutReuseFrameCacheHit=false` on reused scrubs, even though the final layout reuse decision succeeds. That means the frame cache is missing for the selected-hour `Analysis` object while the later structural proof still confirms the layout is reusable.
- The remaining named non-frame pre-wait work is much smaller: key complete -> plan ready is about `32.9-33.3 ms` for Ness Tziona and `6.8-23.0 ms` for Ben-Gurion; mesh update, pending-storage marking, storage wait, and queue drain remain small.

### Target Recommendation

The next optimization target should be structural frame reuse/cache identity, not render storage readiness and not route publication.

The smallest safe optimization proof should ask whether selected-hour scrub can reuse the already-derived layout frame by stable structural identity instead of caching only by selected-hour `Analysis` object identity. Stop conditions should remain the existing proof boundary: `reuse-safe`, same dimensions/placement/cell mapping, same hover point, no readbacks, and unchanged GPU-native transport.

### Next Engineering Step

Completed by the 2026-05-17 remaining-window split above:

1. instrument the reused scrub path around reuse-key lookup, selected-hour value publication, route controller update, scene receipt, tooltip/picking proof, and final visible acknowledgement - done
2. keep the current layout-reuse proof assertions as stop rules: `renderLayoutReuseAction='reused'`, `renderLayoutBuildTrace=null`, `activeLayoutCandidateCount=1`, `hoverCellLookupProofStatus='same-point-confirmed'` - done
3. recollect the same BG/Ness Tziona normalized/discrete matrix and the Ness Tziona repeated-scrub soak - done
4. decide whether the next optimization is controller/route publication batching, scene-side acknowledgement timing, or a smaller cold-start/init pass - current evidence points to reuse-key/frame derivation plus attempt -> storage-ready, not route publication or cold-start/init

Cold-start/init is still worth investigating, especially for weaker machines, but the new data says the highest-confidence next scrub optimization is the remaining warm-scrub reuse-key/frame derivation and attempt -> storage-ready path before changing a different phase.

## 2026-05-17 Structural Frame Cache Optimization Recollection

Collector command:

```powershell
cd viewer
npx playwright test tests/e2e/main-route-performance-0_5m.spec.ts --project=chromium --workers=1 --reporter=list --grep "collects BG and Ness Tziona 0.5m main-route timing, memory, color-mode, and scrub samples"
```

Result: `1 passed (2.2m)` on 2026-05-17.

| Project | Mode | Frame cache hit | Frame derivation ms before -> after | Reuse key ms before -> after | Pre-wait ms before -> after | Render update ms before -> after | Visible ms before -> after |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Ben-Gurion | normalized | structural | 45.0 -> 0.0 | 45.1 -> 0.1 | 68.3 -> 30.3 | 103.6 -> 65.8 | 111.2 -> 75.2 |
| Ben-Gurion | discrete | structural | 44.6 -> 0.0 | 44.6 -> 0.0 | 51.8 -> 7.7 | 135.1 -> 71.8 | 144.0 -> 79.3 |
| Ness-Tziona | normalized | structural | 235.0 -> 0.0 | 235.0 -> 0.1 | 268.2 -> 33.5 | 401.4 -> 188.8 | 529.7 -> 316.9 |
| Ness-Tziona | discrete | structural | 242.1 -> 0.0 | 242.1 -> 0.0 | 275.7 -> 33.5 | 582.5 -> 281.6 | 664.8 -> 295.0 |

Proof boundary stayed intact: all target scrub samples still report `renderLayoutReuseAction='reused'`, `renderLayoutReuseReason='reuse-safe'`, `renderLayoutBuildTrace=null`, `activeLayoutCandidateCount=1`, and `hoverCellLookupProofStatus='same-point-confirmed'`. The GPU-native boundary also stayed intact: `rendererBackend=webgpu`, `utciRenderResolved=gpuNative`, `utciSurfaceSource=compute-buffer-selected-hour`, `baseRenderTransport=compute-buffer-selected-hour`, `dataTextureBuildCount=0`, and visible selected-hour readback count `0`.

### Structural Cache Diagnosis

The structural frame cache removed the repeated frame-derivation bucket from warm scrubs. All four scrub samples now report `renderLayoutReuseFrameCacheHit=true`, `renderLayoutReuseFrameCacheKind='structural'`, and `renderLayoutReuseFrameDerivationMs=0`.

For Ness Tziona, the reuse-key/frame bucket collapsed from `235.0-242.1 ms` to `0.0-0.1 ms`. The visible warm-scrub path improved by `212.8 ms` normalized and `369.8 ms` discrete relative to the coupled pre-storage split. The remaining Ness Tziona render-update window is now `188.8-281.6 ms`.

The repeated-scrub soak still plateaus: retained CPU layout bytes remain flat at `163,435,220`, app-owned UTCI/WebGPU bytes remain flat at `1,013,299,388`, and rebuild replacement still releases the stable reused layout identity before the smaller grid takes over.

### Next Target

The next action should not be another frame/layout optimization. The structural-frame bucket is now proven gone for warm scrubs. The remaining work is the post-key scene/update path: after `renderStoragePreWaitMs` fell to `33.5 ms` for Ness Tziona, the remaining render update still spends about `155-248 ms` outside frame derivation and actual storage wait. That should be investigated as a separate render-update split before touching cold-start/init.

## Historical Diagnostics

Older pre-implementation diagnostics were split into [main-route-selected-hour-render-diagnostics-history.md](main-route-selected-hour-render-diagnostics-history.md) so this file stays focused on the current 2026-05-17 layout-reuse implementation recollection. Those historical sections preserve the earlier evidence trail but are superseded by the current measurements above.
