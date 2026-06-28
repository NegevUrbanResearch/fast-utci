# Main Route Selected-Hour 0.5m Base Baseline

Date: 2026-05-15

## Scope

This artifact measures the main route `/`, not `/debug`. It captures the selected-hour WebGPU path at `gridResolution=0.5` without `.bin`, Python, parity, or debug comparison data in the timing baseline.

JSON source: [data/performance-results/main-route-selected-hour-0_5m-base.json](../../data/performance-results/main-route-selected-hour-0_5m-base.json)

## Included Analyses

- `Ben-Gurion/20250815_grid_2m_fullday`
- `Ness-Tziona/exploded/nes_tziona_unblock_2`

## Excluded BG Variants

This 0.5m stress pass intentionally excludes other Ben-Gurion variants and uses only the BG base case plus the Ness Tziona base/exploded model.

## Collection Method

- Route: `/`
- Query: `gridResolution=0.5&utciRender=auto&utciRenderDiagnostics=1`
- Color modes: normalized/full-day and discrete/per-hour
- Scrub sample: app-visible hour slider scrub from hour `0` to hour `1`

## Proof Boundary

- `Ben-Gurion/20250815_grid_2m_fullday`: `utciSurfaceSource=compute-buffer-selected-hour`, `baseRenderTransport=compute-buffer-selected-hour`, `dataTextureBuildCount=0`, `selectedHourRuntimeContract.strongVisibleGpuPath=true`, no python/bin/debug comparison fields, no forbidden comparison requests.
- `Ness-Tziona/exploded/nes_tziona_unblock_2`: `utciSurfaceSource=compute-buffer-selected-hour`, `baseRenderTransport=compute-buffer-selected-hour`, `dataTextureBuildCount=0`, `selectedHourRuntimeContract.strongVisibleGpuPath=true`, no python/bin/debug comparison fields, no forbidden comparison requests.

Memory is scoped only to tracked app-owned UTCI/WebGPU buffers. The `GPU VRAM` total mirrors the main-route runtime helper: `persistentExposureBytes + allHoursOutputBytes + selectedHourOutputBytes + renderOwnedSelectedHourBytes`. It is not a browser, OS, or device VRAM measurement.

## Timing Table

| Project | Mode | Phase | Points | First ready ms | First visible ms | Payload ms | Worker BVH ms | Upload ms | Exposure precompute ms | Dispatch ms | Render update ms | Scene sync delay ms | Scene sync total ms |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ben-Gurion | normalized | initial | 1,662,657 | 7781.2 | 8414.5 | 124.7 | 98.5 | 190.9 | 6055.4 | 4.8 | 2176.1 | 1767.0 | 408.7 |
| Ben-Gurion | normalized | scrub | 1,662,657 | 103.5 | 986.9 | 124.7 | 98.5 | 190.9 | 6055.4 | 69.7 | 916.8 | 707.9 | 208.6 |
| Ben-Gurion | discrete | initial | 1,662,657 | 6799.2 | 7483.0 | 155.5 | 87.0 | 199.3 | 6040.2 | 8.0 | 1266.4 | 864.1 | 402.1 |
| Ben-Gurion | discrete | scrub | 1,662,657 | 46.0 | 998.9 | 155.5 | 87.0 | 199.3 | 6040.2 | 11.6 | 987.1 | 785.1 | 201.8 |
| Ness-Tziona | normalized | initial | 8,171,761 | 20665.8 | 23883.8 | 277.4 | 215.9 | 1084.8 | 16224.2 | 16.4 | 7478.5 | 5707.2 | 1771.0 |
| Ness-Tziona | normalized | scrub | 8,171,761 | 182.6 | 4889.3 | 277.4 | 215.9 | 1084.8 | 16224.2 | 16.9 | 4871.8 | 3665.4 | 1206.0 |
| Ness-Tziona | discrete | initial | 8,171,761 | 18400.8 | 21560.6 | 286.9 | 209.7 | 1074.7 | 16265.8 | 15.3 | 5103.4 | 3473.3 | 1629.7 |
| Ness-Tziona | discrete | scrub | 8,171,761 | 160.8 | 4707.8 | 286.9 | 209.7 | 1074.7 | 16265.8 | 11.1 | 4696.5 | 3609.8 | 1086.6 |

## Render Sync Detail

| Project | Mode | Phase | Layout build ms | Storage init wait ms | Buffer copy ms | Queue drain ms |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Ben-Gurion | normalized | initial | 99.1 | 139.2 | 0.0 | 170.0 |
| Ben-Gurion | normalized | scrub | 101.3 | 38.6 | 0.0 | 68.7 |
| Ben-Gurion | discrete | initial | 116.3 | 122.7 | 0.1 | 162.9 |
| Ben-Gurion | discrete | scrub | 85.0 | 53.7 | 0.0 | 63.0 |
| Ness-Tziona | normalized | initial | 480.8 | 574.1 | 0.0 | 714.5 |
| Ness-Tziona | normalized | scrub | 539.1 | 147.3 | 0.1 | 519.0 |
| Ness-Tziona | discrete | initial | 470.9 | 529.0 | 0.0 | 627.3 |
| Ness-Tziona | discrete | scrub | 523.1 | 149.1 | 0.0 | 414.2 |

## Memory Table

| Project | Points | GPU VRAM MiB | Persistent exposure MiB | Selected-hour current MiB | Render-owned selected-hour MiB | Scope |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Ben-Gurion | 1,662,657 | 228.33 | 63.43 | 6.34 | 158.56 | utci-owned-webgpu-buffers |
| Ness-Tziona | 8,171,761 | 1122.22 | 311.73 | 31.17 | 779.32 | utci-owned-webgpu-buffers |

## Current Optimization Inference

The 0.5m data keeps the UTCI selected-hour dispatch small: roughly `5-70 ms` for BG and `11-17 ms` for Ness Tziona across initial and scrub samples. That means UTCI math is not the primary bottleneck.

Initial load is dominated by exposure precompute and render publication:

- BG initial: about `6.0 s` exposure plus `1.3-2.2 s` render update.
- Ness Tziona initial: about `16.2 s` exposure plus `5.1-7.5 s` render update.

Scrubbing avoids the cold-start exposure work, but Ness Tziona still spends about `4.7-4.9 s` before the new hour is visible. The scrub bottleneck is therefore render-side publication/synchronization, especially scene sync start delay and render-owned selected-hour storage handling.

The next optimization pass should split into two tracks:

1. Scrub responsiveness: reduce render-owned storage churn, scene sync start delay, and queue drain cost.
2. 0.5m cold start: tile or cache exposure/BVH/upload work so the first visible selected hour does not require one monolithic full-grid exposure pass.
