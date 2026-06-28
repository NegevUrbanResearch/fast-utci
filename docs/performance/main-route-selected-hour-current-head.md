# Main Route Selected-Hour Current-HEAD Baseline

Date: 2026-05-15

## Scope

This artifact measures the main route `/`, not `/debug`. It captures the current selected-hour WebGPU path without `.bin`, Python, or debug comparison data in the timing baseline.

## Included Analyses

- `Ben-Gurion/20250815_grid_2m_fullday`
- `Ness-Tziona/exploded/nes_tziona_unblock_2`

## Excluded BG Variants

This baseline intentionally excludes other Ben-Gurion variants so the current-head timing pass stays limited to the BG 2m base case and Ness Tziona 2m.

## Proof Boundary

- `Ben-Gurion/20250815_grid_2m_fullday`: `utciSurfaceSource=compute-buffer-selected-hour`, `baseRenderTransport=compute-buffer-selected-hour`, `dataTextureBuildCount=0`, `selectedHourRuntimeContract.strongVisibleGpuPath=true`, no python/bin/debug comparison fields, no forbidden comparison requests.
- `Ness-Tziona/exploded/nes_tziona_unblock_2`: `utciSurfaceSource=compute-buffer-selected-hour`, `baseRenderTransport=compute-buffer-selected-hour`, `dataTextureBuildCount=0`, `selectedHourRuntimeContract.strongVisibleGpuPath=true`, no python/bin/debug comparison fields, no forbidden comparison requests.

Memory is scoped only to tracked app-owned UTCI/WebGPU buffers. The `GPU VRAM` total mirrors the main-route runtime helper: `persistentExposureBytes + allHoursOutputBytes + selectedHourOutputBytes + renderOwnedSelectedHourBytes`. Selected-hour high-watermark is reported as diagnostic context, but the displayed total is not a browser, OS, or device VRAM measurement.

## Unavailable Timing Fields

The fresh `/` capture currently exposes only the coarser main-route timing fields shown below. The JSON artifact preserves these fields as `null` for both cases instead of inventing values:

- `payloadPrepareMs`
- `workerBvhMs`
- `pipelineUploadMs`
- `firstSelectedHourReadyMs`

This means the main-route baseline is suitable for current route-level evidence around first-visible timing, exposure precompute, scene-sync delay, scene-sync total, selected-hour dispatch, and tracked app-owned GPU memory. It is not a full fine-grained cold-start sub-bucket breakdown by itself.

## Timing Table

| Project | Analysis | Points | Grid m | Month | Hour | Time index | First visible ms | Exposure precompute ms | Scene sync start delay ms | Scene sync total ms | Dispatch ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ben-Gurion | `Ben-Gurion/20250815_grid_2m_fullday` | 104,445 | 2.0 | 7 | 0 | 168 | 1283.8 | 708.3 | 269.0 | 80.9 | 3.8 |
| Ness-Tziona | `Ness-Tziona/exploded/nes_tziona_unblock_2` | 511,840 | 2.0 | 7 | 0 | 168 | 8465.4 | 6648.2 | 1140.1 | 340.1 | 5.1 |

## Memory Table

| Project | Analysis | GPU VRAM MiB | Persistent exposure MiB | Selected-hour current MiB | Selected-hour HWM MiB | Render-owned selected-hour MiB | Scope |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Ben-Gurion | `Ben-Gurion/20250815_grid_2m_fullday` | 14.34 | 3.98 | 0.40 | 0.40 | 9.96 | utci-owned-webgpu-buffers |
| Ness-Tziona | `Ness-Tziona/exploded/nes_tziona_unblock_2` | 70.29 | 19.53 | 1.95 | 1.95 | 48.81 | utci-owned-webgpu-buffers |

## Current Optimization Inference

Fresh main-route numbers point at exposure precompute / cold-start compute as the next bottleneck: it averages 3678.3 ms across BG and Ness Tziona, while selected-hour dispatch stays at 4.5 ms. That means the next optimization pass should stay focused on cold-start work before first visible publication, not on .bin comparison, selected-hour transport, or 0.5m claims yet.

The optimization inference above is intentionally conservative: it is based on the available main-route fields plus older debug-route context where finer-grained sub-bucket detail exists.

If these numbers disagree with the older 2026-05-09 strategy snapshot, this fresh main-route baseline wins for current route-level timing decisions, while the older debug-route breakdown remains supporting historical context for unavailable sub-buckets such as `payloadPrepareMs`, `workerBvhMs`, `pipelineUploadMs`, `firstSelectedHourReadyMs`.
