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

### Next Engineering Step

The next smallest useful pass should split the remaining warm-scrub render update time after layout reuse:

1. instrument the reused scrub path around reuse-key lookup, selected-hour value publication, route controller update, scene receipt, tooltip/picking proof, and final visible acknowledgement
2. keep the current layout-reuse proof assertions as stop rules: `renderLayoutReuseAction='reused'`, `renderLayoutBuildTrace=null`, `activeLayoutCandidateCount=1`, `hoverCellLookupProofStatus='same-point-confirmed'`
3. recollect the same BG/Ness Tziona normalized/discrete matrix and the Ness Tziona repeated-scrub soak
4. only then decide whether the next optimization is controller/route publication batching, scene-side acknowledgement timing, or a smaller cold-start/init pass

Cold-start/init is still worth investigating, especially for weaker machines, but the new data says the highest-confidence next scrub optimization is to explain the remaining `438.1-537.7 ms` Ness Tziona warm update window before changing a different phase.

## Historical Diagnostics

Older pre-implementation diagnostics were split into [main-route-selected-hour-render-diagnostics-history.md](main-route-selected-hour-render-diagnostics-history.md) so this file stays focused on the current 2026-05-17 layout-reuse implementation recollection. Those historical sections preserve the earlier evidence trail but are superseded by the current measurements above.
