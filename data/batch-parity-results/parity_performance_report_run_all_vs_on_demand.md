# WebGPU Parity & Performance Report: Run-All vs On-Demand

Generated on: 09/05/2026, 4:55:52

Legacy run-all-only report preserved at: `D:\Projects\Nur\Shade\fast-utci\data\batch-parity-results\parity_performance_report.md`
New combined report written to: `D:\Projects\Nur\Shade\fast-utci\data\batch-parity-results\parity_performance_report_run_all_vs_on_demand.md`

## 1. Headline Speedup vs Grasshopper 1-Hour UTCI Baseline

| Profile | Points | 1h GH Baseline | 1h On-Demand Now | 1h Speedup | 12m GH Estimate | 12m On-Demand Estimate | 12m Speedup | Notes |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Median (52 analyses) | 104,445 | 15.0 min | 1.55 s | **579.9x** | 72.0 h | 1.70 s | **152403.4x** | Strict exposure-only on-demand. 12-month estimate = precompute once + 288 one-hour dispatches. |
| Ben-Gurion/20250815_grid_2m_fullday | 104,445 | 15.0 min | 2.23 s | **404.5x** | 72.0 h | 1.73 s | **149809.3x** | Main Ben-Gurion full-day baseline analysis. |
| Ness-Tziona/exploded/nes_tziona_unblock_2 | 511,840 | 15.0 min | 4.82 s | **186.8x** | 72.0 h | 2.70 s | **95872.2x** | Largest grid in the current batch. |

Assumptions: Grasshopper baseline is fixed at ~15 minutes (900s) per UTCI hour. The 12-month estimate uses this repo's representative full-year sweep of 288 hourly evaluations (12 months x 24 hours), not an 8,760-hour annual run. On-demand estimates come from strict exposure-only diagnostics: one exposure precompute plus repeated one-hour dispatches.

## 2. Per-Analysis On-Demand Summary

| Analysis | Points | State | Ready Wall (s) | Exposure Precompute (ms) | One-Hour Dispatch (ms) | 12m Estimate (s) | Persistent Exposure (MiB) | Selected-Hour HWM (KiB) | Zero All-Hours Alloc | Path |
| :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :--- |
| Ben-Gurion/20250815_grid_2m_fullday | 104,445 | success | 2.23 | 664.6 | 3.700 | 1.730 | 3.98 | 408.0 | PASS | exposure-only-f32 |
| Ben-Gurion/existing_buildings/existing_buildings_01 | 104,445 | success | 1.54 | 343.0 | 5.700 | 1.985 | 3.98 | 408.0 | PASS | exposure-only-f32 |
| Ben-Gurion/existing_buildings/existing_buildings_02 | 104,445 | success | 1.57 | 346.4 | 4.900 | 1.758 | 3.98 | 408.0 | PASS | exposure-only-f32 |
| Ben-Gurion/existing_buildings/existing_buildings_03 | 104,445 | success | 1.53 | 371.4 | 5.500 | 1.955 | 3.98 | 408.0 | PASS | exposure-only-f32 |
| Ben-Gurion/existing_buildings/existing_buildings_04 | 104,445 | success | 1.51 | 310.8 | 5.500 | 1.895 | 3.98 | 408.0 | PASS | exposure-only-f32 |
| Ben-Gurion/existing_buildings/existing_buildings_05 | 104,445 | success | 1.59 | 377.2 | 4.600 | 1.702 | 3.98 | 408.0 | PASS | exposure-only-f32 |
| Ben-Gurion/existing_buildings/existing_buildings_06 | 104,445 | success | 1.54 | 380.1 | 5.500 | 1.964 | 3.98 | 408.0 | PASS | exposure-only-f32 |
| Ben-Gurion/existing_buildings/existing_buildings_07 | 104,445 | success | 1.56 | 367.2 | 5.100 | 1.836 | 3.98 | 408.0 | PASS | exposure-only-f32 |
| Ben-Gurion/existing_buildings/existing_buildings_08 | 104,445 | success | 1.59 | 410.8 | 4.500 | 1.707 | 3.98 | 408.0 | PASS | exposure-only-f32 |
| Ben-Gurion/existing_buildings/existing_buildings_09 | 104,445 | success | 1.59 | 374.7 | 4.600 | 1.700 | 3.98 | 408.0 | PASS | exposure-only-f32 |
| Ben-Gurion/existing_buildings/existing_buildings_10 | 104,445 | success | 1.52 | 401.2 | 4.400 | 1.668 | 3.98 | 408.0 | PASS | exposure-only-f32 |
| Ben-Gurion/existing_trees/existing_trees_01 | 104,445 | success | 1.25 | 237.6 | 4.400 | 1.505 | 3.98 | 408.0 | PASS | exposure-only-f32 |
| Ben-Gurion/existing_trees/existing_trees_02 | 104,445 | success | 1.36 | 246.2 | 5.300 | 1.773 | 3.98 | 408.0 | PASS | exposure-only-f32 |
| Ben-Gurion/existing_trees/existing_trees_03 | 104,445 | success | 1.57 | 262.1 | 4.000 | 1.414 | 3.98 | 408.0 | PASS | exposure-only-f32 |
| Ben-Gurion/existing_trees/existing_trees_04 | 104,445 | success | 1.46 | 281.5 | 6.000 | 2.010 | 3.98 | 408.0 | PASS | exposure-only-f32 |
| Ben-Gurion/existing_trees/existing_trees_05 | 104,445 | success | 1.53 | 335.3 | 5.200 | 1.833 | 3.98 | 408.0 | PASS | exposure-only-f32 |
| Ben-Gurion/existing_trees/existing_trees_06 | 104,445 | success | 1.53 | 336.0 | 5.500 | 1.920 | 3.98 | 408.0 | PASS | exposure-only-f32 |
| Ben-Gurion/existing_trees/existing_trees_07 | 104,445 | success | 1.55 | 335.2 | 5.700 | 1.977 | 3.98 | 408.0 | PASS | exposure-only-f32 |
| Ben-Gurion/existing_trees/existing_trees_08 | 104,445 | success | 1.51 | 333.9 | 3.700 | 1.400 | 3.98 | 408.0 | PASS | exposure-only-f32 |
| Ben-Gurion/existing_trees/existing_trees_09 | 104,445 | success | 1.51 | 341.3 | 4.400 | 1.609 | 3.98 | 408.0 | PASS | exposure-only-f32 |
| Ben-Gurion/existing_trees/existing_trees_10 | 104,445 | success | 1.48 | 336.4 | 3.900 | 1.460 | 3.98 | 408.0 | PASS | exposure-only-f32 |
| Ben-Gurion/new_high_buildings/new_high_buildings_01 | 104,445 | success | 1.62 | 372.4 | 4.700 | 1.726 | 3.98 | 408.0 | PASS | exposure-only-f32 |
| Ben-Gurion/new_high_buildings/new_high_buildings_02 | 104,445 | success | 1.54 | 330.3 | 5.300 | 1.857 | 3.98 | 408.0 | PASS | exposure-only-f32 |
| Ben-Gurion/new_high_buildings/new_high_buildings_03 | 104,445 | success | 1.50 | 332.2 | 5.300 | 1.859 | 3.98 | 408.0 | PASS | exposure-only-f32 |
| Ben-Gurion/new_high_buildings/new_high_buildings_04 | 104,445 | success | 1.47 | 329.6 | 5.800 | 2.000 | 3.98 | 408.0 | PASS | exposure-only-f32 |
| Ben-Gurion/new_high_buildings/new_high_buildings_05 | 104,445 | success | 1.59 | 338.0 | 4.500 | 1.634 | 3.98 | 408.0 | PASS | exposure-only-f32 |
| Ben-Gurion/new_high_buildings/new_high_buildings_06 | 104,445 | success | 1.63 | 337.1 | 4.000 | 1.489 | 3.98 | 408.0 | PASS | exposure-only-f32 |
| Ben-Gurion/new_high_buildings/new_high_buildings_07 | 104,445 | success | 1.55 | 385.8 | 5.100 | 1.855 | 3.98 | 408.0 | PASS | exposure-only-f32 |
| Ben-Gurion/new_high_buildings/new_high_buildings_08 | 104,445 | success | 1.47 | 334.8 | 5.600 | 1.948 | 3.98 | 408.0 | PASS | exposure-only-f32 |
| Ben-Gurion/new_high_buildings/new_high_buildings_09 | 104,445 | success | 1.50 | 333.7 | 4.200 | 1.543 | 3.98 | 408.0 | PASS | exposure-only-f32 |
| Ben-Gurion/new_high_buildings/new_high_buildings_10 | 104,445 | success | 1.57 | 334.5 | 2.400 | 1.026 | 3.98 | 408.0 | PASS | exposure-only-f32 |
| Ben-Gurion/new_low_buildings/new_low_buildings_01 | 104,445 | success | 1.41 | 319.2 | 5.700 | 1.961 | 3.98 | 408.0 | PASS | exposure-only-f32 |
| Ben-Gurion/new_low_buildings/new_low_buildings_02 | 104,445 | success | 1.49 | 320.3 | 4.700 | 1.674 | 3.98 | 408.0 | PASS | exposure-only-f32 |
| Ben-Gurion/new_low_buildings/new_low_buildings_03 | 104,445 | success | 1.58 | 366.1 | 5.500 | 1.950 | 3.98 | 408.0 | PASS | exposure-only-f32 |
| Ben-Gurion/new_low_buildings/new_low_buildings_04 | 104,445 | success | 1.47 | 329.3 | 4.800 | 1.712 | 3.98 | 408.0 | PASS | exposure-only-f32 |
| Ben-Gurion/new_low_buildings/new_low_buildings_05 | 104,445 | success | 1.46 | 326.3 | 4.800 | 1.709 | 3.98 | 408.0 | PASS | exposure-only-f32 |
| Ben-Gurion/new_low_buildings/new_low_buildings_06 | 104,445 | success | 1.61 | 333.0 | 4.400 | 1.600 | 3.98 | 408.0 | PASS | exposure-only-f32 |
| Ben-Gurion/new_low_buildings/new_low_buildings_07 | 104,445 | success | 1.51 | 329.3 | 4.000 | 1.481 | 3.98 | 408.0 | PASS | exposure-only-f32 |
| Ben-Gurion/new_low_buildings/new_low_buildings_08 | 104,445 | success | 1.57 | 332.8 | 4.600 | 1.658 | 3.98 | 408.0 | PASS | exposure-only-f32 |
| Ben-Gurion/new_low_buildings/new_low_buildings_09 | 104,445 | success | 1.92 | 384.4 | 4.400 | 1.652 | 3.98 | 408.0 | PASS | exposure-only-f32 |
| Ben-Gurion/new_low_buildings/new_low_buildings_10 | 104,445 | success | 1.49 | 336.1 | 4.100 | 1.517 | 3.98 | 408.0 | PASS | exposure-only-f32 |
| Ben-Gurion/new_trees/new_trees_01 | 104,445 | success | 1.56 | 336.7 | 3.600 | 1.374 | 3.98 | 408.0 | PASS | exposure-only-f32 |
| Ben-Gurion/new_trees/new_trees_02 | 104,445 | success | 1.55 | 307.0 | 5.000 | 1.747 | 3.98 | 408.0 | PASS | exposure-only-f32 |
| Ben-Gurion/new_trees/new_trees_03 | 104,445 | success | 1.56 | 317.0 | 3.800 | 1.411 | 3.98 | 408.0 | PASS | exposure-only-f32 |
| Ben-Gurion/new_trees/new_trees_04 | 104,445 | success | 1.68 | 394.9 | 3.600 | 1.432 | 3.98 | 408.0 | PASS | exposure-only-f32 |
| Ben-Gurion/new_trees/new_trees_05 | 104,445 | success | 1.73 | 432.1 | 3.600 | 1.469 | 3.98 | 408.0 | PASS | exposure-only-f32 |
| Ben-Gurion/new_trees/new_trees_06 | 104,445 | success | 1.70 | 403.5 | 3.700 | 1.469 | 3.98 | 408.0 | PASS | exposure-only-f32 |
| Ben-Gurion/new_trees/new_trees_07 | 104,445 | success | 1.65 | 406.0 | 4.300 | 1.644 | 3.98 | 408.0 | PASS | exposure-only-f32 |
| Ben-Gurion/new_trees/new_trees_08 | 104,445 | success | 1.65 | 411.4 | 3.900 | 1.535 | 3.98 | 408.0 | PASS | exposure-only-f32 |
| Ben-Gurion/new_trees/new_trees_09 | 104,445 | success | 1.65 | 416.9 | 4.200 | 1.626 | 3.98 | 408.0 | PASS | exposure-only-f32 |
| Ben-Gurion/new_trees/new_trees_10 | 104,445 | success | 1.68 | 419.9 | 3.600 | 1.457 | 3.98 | 408.0 | PASS | exposure-only-f32 |
| Ness-Tziona/exploded/nes_tziona_unblock_2 | 511,840 | success | 4.82 | 1465.2 | 4.300 | 2.704 | 19.53 | 1999.4 | PASS | exposure-only-f32 |

## 3. Existing Run-All Route Summary

| Analysis | Python Full-Day (s) | WebGPU Parity/Day (s) | Speedup vs Python | WebGPU Full-Year (s) | Speedup vs Python x12 | Solar | Sky | MRT | UTCI |
| :--- | ---: | ---: | ---: | ---: | ---: | :---: | :---: | :---: | :---: |
| Ben-Gurion/20250815_grid_2m_fullday | 195.1 | 1.89 | **103.2x** | 2.77 | **846.3x** | PASS | PASS | PASS | PASS |
| Ben-Gurion/existing_buildings/existing_buildings_01 | 27.3 | 1.78 | **15.4x** | 2.44 | **134.7x** | - | - | - | PASS |
| Ben-Gurion/existing_buildings/existing_buildings_02 | 29.9 | 1.79 | **16.7x** | 2.43 | **147.5x** | - | - | - | PASS |
| Ben-Gurion/existing_buildings/existing_buildings_03 | 28.5 | 1.92 | **14.8x** | 2.54 | **134.5x** | - | - | - | PASS |
| Ben-Gurion/existing_buildings/existing_buildings_04 | 27.8 | 1.71 | **16.3x** | 2.41 | **138.2x** | - | - | - | PASS |
| Ben-Gurion/existing_buildings/existing_buildings_05 | 28.0 | 1.79 | **15.6x** | 2.59 | **130.1x** | - | - | - | PASS |
| Ben-Gurion/existing_buildings/existing_buildings_06 | 28.1 | 1.80 | **15.6x** | 2.51 | **134.3x** | - | - | - | PASS |
| Ben-Gurion/existing_buildings/existing_buildings_07 | 28.5 | 1.75 | **16.3x** | 2.44 | **140.2x** | - | - | - | PASS |
| Ben-Gurion/existing_buildings/existing_buildings_08 | 29.1 | 1.75 | **16.6x** | 2.45 | **142.3x** | - | - | - | PASS |
| Ben-Gurion/existing_buildings/existing_buildings_09 | 29.0 | 1.75 | **16.5x** | 2.44 | **142.5x** | - | - | - | PASS |
| Ben-Gurion/existing_buildings/existing_buildings_10 | 28.6 | 1.76 | **16.2x** | 2.50 | **137.4x** | - | - | - | PASS |
| Ben-Gurion/existing_trees/existing_trees_01 | 28.9 | 1.66 | **17.4x** | 2.15 | **161.4x** | - | - | - | PASS |
| Ben-Gurion/existing_trees/existing_trees_02 | 28.6 | 1.80 | **15.9x** | 2.31 | **148.9x** | - | - | - | PASS |
| Ben-Gurion/existing_trees/existing_trees_03 | 29.1 | 1.78 | **16.4x** | 2.41 | **145.3x** | - | - | - | PASS |
| Ben-Gurion/existing_trees/existing_trees_04 | 29.4 | 1.88 | **15.6x** | 2.33 | **151.4x** | - | - | - | PASS |
| Ben-Gurion/existing_trees/existing_trees_05 | 29.5 | 1.69 | **17.5x** | 2.35 | **150.7x** | - | - | - | PASS |
| Ben-Gurion/existing_trees/existing_trees_06 | 29.1 | 1.80 | **16.2x** | 2.38 | **146.9x** | - | - | - | PASS |
| Ben-Gurion/existing_trees/existing_trees_07 | 29.4 | 1.71 | **17.2x** | 2.43 | **145.4x** | - | - | - | PASS |
| Ben-Gurion/existing_trees/existing_trees_08 | 29.6 | 1.75 | **16.9x** | 2.40 | **148.0x** | - | - | - | PASS |
| Ben-Gurion/existing_trees/existing_trees_09 | 29.5 | 1.78 | **16.6x** | 2.48 | **142.9x** | - | - | - | PASS |
| Ben-Gurion/existing_trees/existing_trees_10 | 29.8 | 1.79 | **16.6x** | 2.42 | **148.0x** | - | - | - | PASS |
| Ben-Gurion/new_high_buildings/new_high_buildings_01 | 30.0 | 1.72 | **17.4x** | 2.41 | **149.2x** | - | - | - | PASS |
| Ben-Gurion/new_high_buildings/new_high_buildings_02 | 29.8 | 1.91 | **15.7x** | 2.44 | **146.4x** | - | - | - | PASS |
| Ben-Gurion/new_high_buildings/new_high_buildings_03 | 30.1 | 1.82 | **16.5x** | 2.40 | **150.0x** | - | - | - | PASS |
| Ben-Gurion/new_high_buildings/new_high_buildings_04 | 30.7 | 1.72 | **17.8x** | 2.38 | **154.8x** | - | - | - | PASS |
| Ben-Gurion/new_high_buildings/new_high_buildings_05 | 31.3 | 1.83 | **17.1x** | 2.42 | **155.1x** | - | - | - | PASS |
| Ben-Gurion/new_high_buildings/new_high_buildings_06 | 31.7 | 1.78 | **17.8x** | 2.40 | **158.7x** | - | - | - | PASS |
| Ben-Gurion/new_high_buildings/new_high_buildings_07 | 31.5 | 1.81 | **17.4x** | 2.50 | **151.3x** | - | - | - | PASS |
| Ben-Gurion/new_high_buildings/new_high_buildings_08 | 32.4 | 1.70 | **19.0x** | 2.47 | **157.6x** | - | - | - | PASS |
| Ben-Gurion/new_high_buildings/new_high_buildings_09 | 31.2 | 1.77 | **17.6x** | 2.39 | **157.0x** | - | - | - | PASS |
| Ben-Gurion/new_high_buildings/new_high_buildings_10 | 32.5 | 1.80 | **18.1x** | 2.58 | **151.0x** | - | - | - | PASS |
| Ben-Gurion/new_low_buildings/new_low_buildings_01 | 31.0 | 1.76 | **17.6x** | 2.41 | **154.4x** | - | - | - | PASS |
| Ben-Gurion/new_low_buildings/new_low_buildings_02 | 32.4 | 1.78 | **18.2x** | 2.41 | **160.9x** | - | - | - | PASS |
| Ben-Gurion/new_low_buildings/new_low_buildings_03 | 31.8 | 1.79 | **17.8x** | 2.37 | **161.1x** | - | - | - | PASS |
| Ben-Gurion/new_low_buildings/new_low_buildings_04 | 32.0 | 1.68 | **19.0x** | 2.40 | **160.4x** | - | - | - | PASS |
| Ben-Gurion/new_low_buildings/new_low_buildings_05 | 32.0 | 1.78 | **17.9x** | 2.47 | **155.4x** | - | - | - | PASS |
| Ben-Gurion/new_low_buildings/new_low_buildings_06 | 32.2 | 1.76 | **18.3x** | 2.42 | **159.5x** | - | - | - | PASS |
| Ben-Gurion/new_low_buildings/new_low_buildings_07 | 33.1 | 1.80 | **18.4x** | 2.50 | **158.5x** | - | - | - | PASS |
| Ben-Gurion/new_low_buildings/new_low_buildings_08 | 32.4 | 1.81 | **17.9x** | 2.41 | **161.7x** | - | - | - | PASS |
| Ben-Gurion/new_low_buildings/new_low_buildings_09 | 32.1 | 1.94 | **16.5x** | 2.62 | **147.3x** | - | - | - | PASS |
| Ben-Gurion/new_low_buildings/new_low_buildings_10 | 33.1 | 1.99 | **16.6x** | 2.58 | **154.0x** | - | - | - | PASS |
| Ben-Gurion/new_trees/new_trees_01 | 33.6 | 1.74 | **19.3x** | 2.51 | **160.2x** | - | - | - | PASS |
| Ben-Gurion/new_trees/new_trees_02 | 33.9 | 1.82 | **18.7x** | 2.39 | **170.5x** | - | - | - | PASS |
| Ben-Gurion/new_trees/new_trees_03 | 32.9 | 1.81 | **18.2x** | 2.43 | **162.3x** | - | - | - | PASS |
| Ben-Gurion/new_trees/new_trees_04 | 34.4 | 1.83 | **18.8x** | 2.53 | **163.1x** | - | - | - | PASS |
| Ben-Gurion/new_trees/new_trees_05 | 34.2 | 1.81 | **18.9x** | 2.67 | **153.6x** | - | - | - | PASS |
| Ben-Gurion/new_trees/new_trees_06 | 35.8 | 1.88 | **19.0x** | 2.54 | **169.5x** | - | - | - | PASS |
| Ben-Gurion/new_trees/new_trees_07 | 35.5 | 1.88 | **18.9x** | 2.46 | **173.1x** | - | - | - | PASS |
| Ben-Gurion/new_trees/new_trees_08 | 34.5 | 1.91 | **18.0x** | 2.53 | **163.4x** | - | - | - | PASS |
| Ben-Gurion/new_trees/new_trees_09 | 34.7 | 1.77 | **19.5x** | 2.48 | **167.8x** | - | - | - | PASS |
| Ben-Gurion/new_trees/new_trees_10 | 34.9 | 1.84 | **19.0x** | 2.57 | **163.0x** | - | - | - | PASS |
| Ness-Tziona/exploded/nes_tziona_unblock_2 | 931.0 | 6.96 | **133.7x** | 10.19 | **1096.4x** | - | - | - | PASS |

## 4. Legacy "Store All" Memory Estimate

| Analysis | Grid Points | Solar (MiB) | Sky (MiB) | Results (MiB) | Total VRAM (MiB) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Ben-Gurion/20250815_grid_2m_fullday | 118,531 | 4.1 | 65.6 | 130.2 | **199.9** |
| Ben-Gurion/existing_buildings/existing_buildings_01 | 118,531 | 4.1 | 65.6 | 130.2 | **199.9** |
| Ben-Gurion/existing_buildings/existing_buildings_02 | 118,531 | 4.1 | 65.6 | 130.2 | **199.9** |
| Ben-Gurion/existing_buildings/existing_buildings_03 | 116,163 | 4.0 | 64.3 | 127.6 | **195.9** |
| Ben-Gurion/existing_buildings/existing_buildings_04 | 116,163 | 4.0 | 64.3 | 127.6 | **195.9** |
| Ben-Gurion/existing_buildings/existing_buildings_05 | 116,163 | 4.0 | 64.3 | 127.6 | **195.9** |
| Ben-Gurion/existing_buildings/existing_buildings_06 | 116,745 | 4.0 | 64.6 | 128.3 | **196.8** |
| Ben-Gurion/existing_buildings/existing_buildings_07 | 118,531 | 4.1 | 65.6 | 130.2 | **199.9** |
| Ben-Gurion/existing_buildings/existing_buildings_08 | 118,531 | 4.1 | 65.6 | 130.2 | **199.9** |
| Ben-Gurion/existing_buildings/existing_buildings_09 | 118,531 | 4.1 | 65.6 | 130.2 | **199.9** |
| Ben-Gurion/existing_buildings/existing_buildings_10 | 117,569 | 4.0 | 65.0 | 129.2 | **198.2** |
| Ben-Gurion/existing_trees/existing_trees_01 | 105,697 | 3.6 | 58.5 | 116.1 | **178.2** |
| Ben-Gurion/existing_trees/existing_trees_02 | 107,699 | 3.7 | 59.6 | 118.3 | **181.6** |
| Ben-Gurion/existing_trees/existing_trees_03 | 108,389 | 3.7 | 60.0 | 119.1 | **182.8** |
| Ben-Gurion/existing_trees/existing_trees_04 | 108,388 | 3.7 | 60.0 | 119.1 | **182.8** |
| Ben-Gurion/existing_trees/existing_trees_05 | 111,004 | 3.8 | 61.4 | 122.0 | **187.2** |
| Ben-Gurion/existing_trees/existing_trees_06 | 116,516 | 4.0 | 64.4 | 128.0 | **196.5** |
| Ben-Gurion/existing_trees/existing_trees_07 | 116,516 | 4.0 | 64.4 | 128.0 | **196.5** |
| Ben-Gurion/existing_trees/existing_trees_08 | 116,516 | 4.0 | 64.4 | 128.0 | **196.5** |
| Ben-Gurion/existing_trees/existing_trees_09 | 116,653 | 4.0 | 64.5 | 128.2 | **196.7** |
| Ben-Gurion/existing_trees/existing_trees_10 | 118,256 | 4.1 | 65.4 | 129.9 | **199.4** |
| Ben-Gurion/new_high_buildings/new_high_buildings_01 | 118,531 | 4.1 | 65.6 | 130.2 | **199.9** |
| Ben-Gurion/new_high_buildings/new_high_buildings_02 | 118,531 | 4.1 | 65.6 | 130.2 | **199.9** |
| Ben-Gurion/new_high_buildings/new_high_buildings_03 | 118,531 | 4.1 | 65.6 | 130.2 | **199.9** |
| Ben-Gurion/new_high_buildings/new_high_buildings_04 | 118,531 | 4.1 | 65.6 | 130.2 | **199.9** |
| Ben-Gurion/new_high_buildings/new_high_buildings_05 | 118,531 | 4.1 | 65.6 | 130.2 | **199.9** |
| Ben-Gurion/new_high_buildings/new_high_buildings_06 | 118,531 | 4.1 | 65.6 | 130.2 | **199.9** |
| Ben-Gurion/new_high_buildings/new_high_buildings_07 | 118,531 | 4.1 | 65.6 | 130.2 | **199.9** |
| Ben-Gurion/new_high_buildings/new_high_buildings_08 | 118,531 | 4.1 | 65.6 | 130.2 | **199.9** |
| Ben-Gurion/new_high_buildings/new_high_buildings_09 | 118,531 | 4.1 | 65.6 | 130.2 | **199.9** |
| Ben-Gurion/new_high_buildings/new_high_buildings_10 | 118,531 | 4.1 | 65.6 | 130.2 | **199.9** |
| Ben-Gurion/new_low_buildings/new_low_buildings_01 | 118,531 | 4.1 | 65.6 | 130.2 | **199.9** |
| Ben-Gurion/new_low_buildings/new_low_buildings_02 | 118,531 | 4.1 | 65.6 | 130.2 | **199.9** |
| Ben-Gurion/new_low_buildings/new_low_buildings_03 | 118,531 | 4.1 | 65.6 | 130.2 | **199.9** |
| Ben-Gurion/new_low_buildings/new_low_buildings_04 | 118,531 | 4.1 | 65.6 | 130.2 | **199.9** |
| Ben-Gurion/new_low_buildings/new_low_buildings_05 | 118,531 | 4.1 | 65.6 | 130.2 | **199.9** |
| Ben-Gurion/new_low_buildings/new_low_buildings_06 | 118,531 | 4.1 | 65.6 | 130.2 | **199.9** |
| Ben-Gurion/new_low_buildings/new_low_buildings_07 | 118,531 | 4.1 | 65.6 | 130.2 | **199.9** |
| Ben-Gurion/new_low_buildings/new_low_buildings_08 | 118,531 | 4.1 | 65.6 | 130.2 | **199.9** |
| Ben-Gurion/new_low_buildings/new_low_buildings_09 | 118,531 | 4.1 | 65.6 | 130.2 | **199.9** |
| Ben-Gurion/new_low_buildings/new_low_buildings_10 | 118,531 | 4.1 | 65.6 | 130.2 | **199.9** |
| Ben-Gurion/new_trees/new_trees_01 | 118,531 | 4.1 | 65.6 | 130.2 | **199.9** |
| Ben-Gurion/new_trees/new_trees_02 | 118,531 | 4.1 | 65.6 | 130.2 | **199.9** |
| Ben-Gurion/new_trees/new_trees_03 | 118,531 | 4.1 | 65.6 | 130.2 | **199.9** |
| Ben-Gurion/new_trees/new_trees_04 | 118,531 | 4.1 | 65.6 | 130.2 | **199.9** |
| Ben-Gurion/new_trees/new_trees_05 | 118,531 | 4.1 | 65.6 | 130.2 | **199.9** |
| Ben-Gurion/new_trees/new_trees_06 | 118,531 | 4.1 | 65.6 | 130.2 | **199.9** |
| Ben-Gurion/new_trees/new_trees_07 | 118,531 | 4.1 | 65.6 | 130.2 | **199.9** |
| Ben-Gurion/new_trees/new_trees_08 | 118,531 | 4.1 | 65.6 | 130.2 | **199.9** |
| Ben-Gurion/new_trees/new_trees_09 | 118,531 | 4.1 | 65.6 | 130.2 | **199.9** |
| Ben-Gurion/new_trees/new_trees_10 | 118,531 | 4.1 | 65.6 | 130.2 | **199.9** |
| Ness-Tziona/exploded/nes_tziona_unblock_2 | 512,637 | 17.6 | 283.6 | 563.2 | **864.4** |

## 5. Detailed Full-Year Timing Breakdown

| Analysis | Weather | Init | BVH | GPU Compute | Readback/Wait | Total (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | ---: |
| Ben-Gurion/20250815_grid_2m_fullday | 0.016s (0.6%) | 0.016s (0.6%) | 0.120s (4.3%) | **0.142s (5.1%)** | **2.330s (84.2%)** | **2.77** |
| Ben-Gurion/existing_buildings/existing_buildings_01 | 0.024s (1.0%) | 0.015s (0.6%) | 0.124s (5.1%) | **0.148s (6.1%)** | **1.961s (80.5%)** | **2.44** |
| Ben-Gurion/existing_buildings/existing_buildings_02 | 0.016s (0.7%) | 0.097s (4.0%) | 0.119s (4.9%) | **0.143s (5.9%)** | **1.972s (81.1%)** | **2.43** |
| Ben-Gurion/existing_buildings/existing_buildings_03 | 0.016s (0.6%) | 0.017s (0.7%) | 0.155s (6.1%) | **0.160s (6.3%)** | **2.009s (79.1%)** | **2.54** |
| Ben-Gurion/existing_buildings/existing_buildings_04 | 0.015s (0.6%) | 0.096s (4.0%) | 0.115s (4.8%) | **0.144s (6.0%)** | **1.960s (81.3%)** | **2.41** |
| Ben-Gurion/existing_buildings/existing_buildings_05 | 0.016s (0.6%) | 0.110s (4.3%) | 0.123s (4.8%) | **0.156s (6.0%)** | **2.096s (81.1%)** | **2.59** |
| Ben-Gurion/existing_buildings/existing_buildings_06 | 0.017s (0.7%) | 0.018s (0.7%) | 0.116s (4.6%) | **0.147s (5.9%)** | **1.971s (78.5%)** | **2.51** |
| Ben-Gurion/existing_buildings/existing_buildings_07 | 0.013s (0.5%) | 0.015s (0.6%) | 0.108s (4.4%) | **0.143s (5.9%)** | **2.001s (82.1%)** | **2.44** |
| Ben-Gurion/existing_buildings/existing_buildings_08 | 0.014s (0.6%) | 0.096s (3.9%) | 0.128s (5.2%) | **0.141s (5.7%)** | **2.000s (81.5%)** | **2.45** |
| Ben-Gurion/existing_buildings/existing_buildings_09 | 0.022s (0.9%) | 0.093s (3.8%) | 0.123s (5.0%) | **0.134s (5.5%)** | **2.002s (81.9%)** | **2.44** |
| Ben-Gurion/existing_buildings/existing_buildings_10 | 0.015s (0.6%) | 0.094s (3.8%) | 0.136s (5.4%) | **0.143s (5.7%)** | **2.034s (81.5%)** | **2.50** |
| Ben-Gurion/existing_trees/existing_trees_01 | 0.024s (1.1%) | 0.018s (0.8%) | 0.115s (5.4%) | **0.139s (6.5%)** | **1.791s (83.4%)** | **2.15** |
| Ben-Gurion/existing_trees/existing_trees_02 | 0.014s (0.6%) | 0.016s (0.7%) | 0.106s (4.6%) | **0.148s (6.4%)** | **1.859s (80.7%)** | **2.31** |
| Ben-Gurion/existing_trees/existing_trees_03 | 0.027s (1.1%) | 0.017s (0.7%) | 0.125s (5.2%) | **0.149s (6.2%)** | **1.935s (80.4%)** | **2.41** |
| Ben-Gurion/existing_trees/existing_trees_04 | 0.015s (0.6%) | 0.014s (0.6%) | 0.105s (4.5%) | **0.134s (5.7%)** | **1.910s (81.8%)** | **2.33** |
| Ben-Gurion/existing_trees/existing_trees_05 | 0.014s (0.6%) | 0.014s (0.6%) | 0.115s (4.9%) | **0.155s (6.6%)** | **1.890s (80.3%)** | **2.35** |
| Ben-Gurion/existing_trees/existing_trees_06 | 0.013s (0.5%) | 0.017s (0.7%) | 0.108s (4.5%) | **0.140s (5.9%)** | **1.948s (81.9%)** | **2.38** |
| Ben-Gurion/existing_trees/existing_trees_07 | 0.018s (0.7%) | 0.015s (0.6%) | 0.128s (5.3%) | **0.139s (5.7%)** | **1.970s (81.1%)** | **2.43** |
| Ben-Gurion/existing_trees/existing_trees_08 | 0.013s (0.5%) | 0.016s (0.7%) | 0.115s (4.8%) | **0.137s (5.7%)** | **1.954s (81.4%)** | **2.40** |
| Ben-Gurion/existing_trees/existing_trees_09 | 0.017s (0.7%) | 0.015s (0.6%) | 0.177s (7.1%) | **0.162s (6.5%)** | **1.943s (78.3%)** | **2.48** |
| Ben-Gurion/existing_trees/existing_trees_10 | 0.014s (0.6%) | 0.016s (0.7%) | 0.120s (5.0%) | **0.140s (5.8%)** | **1.952s (80.8%)** | **2.42** |
| Ben-Gurion/new_high_buildings/new_high_buildings_01 | 0.016s (0.7%) | 0.016s (0.7%) | 0.121s (5.0%) | **0.139s (5.8%)** | **1.957s (81.1%)** | **2.41** |
| Ben-Gurion/new_high_buildings/new_high_buildings_02 | 0.018s (0.7%) | 0.015s (0.6%) | 0.137s (5.6%) | **0.178s (7.3%)** | **1.918s (78.5%)** | **2.44** |
| Ben-Gurion/new_high_buildings/new_high_buildings_03 | 0.014s (0.6%) | 0.016s (0.7%) | 0.120s (5.0%) | **0.145s (6.0%)** | **1.949s (81.1%)** | **2.40** |
| Ben-Gurion/new_high_buildings/new_high_buildings_04 | 0.018s (0.8%) | 0.016s (0.7%) | 0.112s (4.7%) | **0.137s (5.8%)** | **1.943s (81.6%)** | **2.38** |
| Ben-Gurion/new_high_buildings/new_high_buildings_05 | 0.017s (0.7%) | 0.014s (0.6%) | 0.131s (5.4%) | **0.144s (5.9%)** | **1.947s (80.3%)** | **2.42** |
| Ben-Gurion/new_high_buildings/new_high_buildings_06 | 0.015s (0.6%) | 0.016s (0.7%) | 0.111s (4.6%) | **0.145s (6.0%)** | **1.949s (81.2%)** | **2.40** |
| Ben-Gurion/new_high_buildings/new_high_buildings_07 | 0.034s (1.4%) | 0.020s (0.8%) | 0.128s (5.1%) | **0.139s (5.6%)** | **1.968s (78.9%)** | **2.50** |
| Ben-Gurion/new_high_buildings/new_high_buildings_08 | 0.014s (0.6%) | 0.016s (0.6%) | 0.112s (4.5%) | **0.139s (5.6%)** | **2.025s (82.1%)** | **2.47** |
| Ben-Gurion/new_high_buildings/new_high_buildings_09 | 0.015s (0.6%) | 0.017s (0.7%) | 0.119s (5.0%) | **0.145s (6.1%)** | **1.920s (80.5%)** | **2.39** |
| Ben-Gurion/new_high_buildings/new_high_buildings_10 | 0.015s (0.6%) | 0.015s (0.6%) | 0.119s (4.6%) | **0.136s (5.3%)** | **2.136s (82.7%)** | **2.58** |
| Ben-Gurion/new_low_buildings/new_low_buildings_01 | 0.014s (0.6%) | 0.016s (0.7%) | 0.112s (4.7%) | **0.145s (6.0%)** | **1.957s (81.3%)** | **2.41** |
| Ben-Gurion/new_low_buildings/new_low_buildings_02 | 0.015s (0.6%) | 0.015s (0.6%) | 0.115s (4.8%) | **0.138s (5.7%)** | **1.943s (80.5%)** | **2.41** |
| Ben-Gurion/new_low_buildings/new_low_buildings_03 | 0.017s (0.7%) | 0.015s (0.6%) | 0.111s (4.7%) | **0.139s (5.9%)** | **1.919s (80.9%)** | **2.37** |
| Ben-Gurion/new_low_buildings/new_low_buildings_04 | 0.016s (0.7%) | 0.016s (0.7%) | 0.112s (4.7%) | **0.143s (6.0%)** | **1.932s (80.6%)** | **2.40** |
| Ben-Gurion/new_low_buildings/new_low_buildings_05 | 0.015s (0.6%) | 0.017s (0.7%) | 0.121s (4.9%) | **0.210s (8.5%)** | **1.933s (78.2%)** | **2.47** |
| Ben-Gurion/new_low_buildings/new_low_buildings_06 | 0.018s (0.7%) | 0.014s (0.6%) | 0.111s (4.6%) | **0.140s (5.8%)** | **1.971s (81.3%)** | **2.42** |
| Ben-Gurion/new_low_buildings/new_low_buildings_07 | 0.017s (0.7%) | 0.016s (0.6%) | 0.126s (5.0%) | **0.149s (6.0%)** | **2.027s (81.0%)** | **2.50** |
| Ben-Gurion/new_low_buildings/new_low_buildings_08 | 0.016s (0.7%) | 0.016s (0.7%) | 0.122s (5.1%) | **0.138s (5.7%)** | **1.948s (80.9%)** | **2.41** |
| Ben-Gurion/new_low_buildings/new_low_buildings_09 | 0.018s (0.7%) | 0.017s (0.7%) | 0.126s (4.8%) | **0.154s (5.9%)** | **2.138s (81.8%)** | **2.62** |
| Ben-Gurion/new_low_buildings/new_low_buildings_10 | 0.021s (0.8%) | 0.019s (0.7%) | 0.145s (5.6%) | **0.164s (6.4%)** | **2.046s (79.5%)** | **2.58** |
| Ben-Gurion/new_trees/new_trees_01 | 0.015s (0.6%) | 0.017s (0.7%) | 0.117s (4.7%) | **0.153s (6.1%)** | **2.042s (81.2%)** | **2.51** |
| Ben-Gurion/new_trees/new_trees_02 | 0.021s (0.9%) | 0.018s (0.8%) | 0.117s (4.9%) | **0.138s (5.8%)** | **1.936s (81.1%)** | **2.39** |
| Ben-Gurion/new_trees/new_trees_03 | 0.021s (0.9%) | 0.017s (0.7%) | 0.137s (5.6%) | **0.144s (5.9%)** | **1.948s (80.1%)** | **2.43** |
| Ben-Gurion/new_trees/new_trees_04 | 0.015s (0.6%) | 0.017s (0.7%) | 0.121s (4.8%) | **0.145s (5.7%)** | **2.068s (81.7%)** | **2.53** |
| Ben-Gurion/new_trees/new_trees_05 | 0.014s (0.5%) | 0.017s (0.6%) | 0.150s (5.6%) | **0.185s (6.9%)** | **2.131s (79.7%)** | **2.67** |
| Ben-Gurion/new_trees/new_trees_06 | 0.020s (0.8%) | 0.016s (0.6%) | 0.130s (5.1%) | **0.142s (5.6%)** | **2.067s (81.5%)** | **2.54** |
| Ben-Gurion/new_trees/new_trees_07 | 0.014s (0.6%) | 0.015s (0.6%) | 0.120s (4.9%) | **0.134s (5.4%)** | **2.020s (82.0%)** | **2.46** |
| Ben-Gurion/new_trees/new_trees_08 | 0.017s (0.7%) | 0.018s (0.7%) | 0.131s (5.2%) | **0.143s (5.6%)** | **2.053s (81.1%)** | **2.53** |
| Ben-Gurion/new_trees/new_trees_09 | 0.014s (0.6%) | 0.015s (0.6%) | 0.120s (4.8%) | **0.141s (5.7%)** | **2.030s (81.9%)** | **2.48** |
| Ben-Gurion/new_trees/new_trees_10 | 0.015s (0.6%) | 0.014s (0.5%) | 0.121s (4.7%) | **0.146s (5.7%)** | **2.103s (81.9%)** | **2.57** |
| Ness-Tziona/exploded/nes_tziona_unblock_2 | 0.025s (0.2%) | 0.021s (0.2%) | 0.270s (2.6%) | **0.200s (2.0%)** | **9.402s (92.3%)** | **10.19** |

## 6. Detailed Parity/Day Timing Breakdown

| Analysis | Weather | Init | BVH | GPU Compute | Readback/Wait | Total (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | ---: |
| Ben-Gurion/20250815_grid_2m_fullday | 0.017s (0.9%) | 0.015s (0.8%) | 0.125s (6.6%) | **0.143s (7.6%)** | **1.443s (76.3%)** | **1.89** |
| Ben-Gurion/existing_buildings/existing_buildings_01 | 0.014s (0.8%) | 0.018s (1.0%) | 0.116s (6.5%) | **0.142s (8.0%)** | **1.326s (74.6%)** | **1.78** |
| Ben-Gurion/existing_buildings/existing_buildings_02 | 0.024s (1.3%) | 0.085s (4.8%) | 0.107s (6.0%) | **0.149s (8.3%)** | **1.315s (73.6%)** | **1.79** |
| Ben-Gurion/existing_buildings/existing_buildings_03 | 0.017s (0.9%) | 0.087s (4.5%) | 0.155s (8.1%) | **0.152s (7.9%)** | **1.427s (74.4%)** | **1.92** |
| Ben-Gurion/existing_buildings/existing_buildings_04 | 0.014s (0.8%) | 0.094s (5.5%) | 0.110s (6.4%) | **0.135s (7.9%)** | **1.281s (75.0%)** | **1.71** |
| Ben-Gurion/existing_buildings/existing_buildings_05 | 0.019s (1.1%) | 0.089s (5.0%) | 0.108s (6.0%) | **0.141s (7.9%)** | **1.367s (76.3%)** | **1.79** |
| Ben-Gurion/existing_buildings/existing_buildings_06 | 0.012s (0.7%) | 0.101s (5.6%) | 0.121s (6.7%) | **0.142s (7.9%)** | **1.355s (75.1%)** | **1.80** |
| Ben-Gurion/existing_buildings/existing_buildings_07 | 0.021s (1.2%) | 0.093s (5.3%) | 0.106s (6.1%) | **0.139s (8.0%)** | **1.317s (75.4%)** | **1.75** |
| Ben-Gurion/existing_buildings/existing_buildings_08 | 0.025s (1.4%) | 0.017s (1.0%) | 0.109s (6.2%) | **0.143s (8.2%)** | **1.295s (74.0%)** | **1.75** |
| Ben-Gurion/existing_buildings/existing_buildings_09 | 0.022s (1.3%) | 0.018s (1.0%) | 0.112s (6.4%) | **0.150s (8.5%)** | **1.297s (73.9%)** | **1.75** |
| Ben-Gurion/existing_buildings/existing_buildings_10 | 0.017s (1.0%) | 0.092s (5.2%) | 0.108s (6.1%) | **0.139s (7.9%)** | **1.331s (75.5%)** | **1.76** |
| Ben-Gurion/existing_trees/existing_trees_01 | 0.017s (1.0%) | 0.043s (2.6%) | 0.102s (6.2%) | **0.129s (7.8%)** | **1.349s (81.4%)** | **1.66** |
| Ben-Gurion/existing_trees/existing_trees_02 | 0.017s (0.9%) | 0.017s (0.9%) | 0.112s (6.2%) | **0.143s (8.0%)** | **1.345s (74.8%)** | **1.80** |
| Ben-Gurion/existing_trees/existing_trees_03 | 0.047s (2.6%) | 0.020s (1.1%) | 0.112s (6.3%) | **0.132s (7.4%)** | **1.259s (70.7%)** | **1.78** |
| Ben-Gurion/existing_trees/existing_trees_04 | 0.014s (0.7%) | 0.017s (0.9%) | 0.107s (5.7%) | **0.138s (7.3%)** | **1.407s (74.7%)** | **1.88** |
| Ben-Gurion/existing_trees/existing_trees_05 | 0.028s (1.7%) | 0.017s (1.0%) | 0.106s (6.3%) | **0.139s (8.2%)** | **1.239s (73.3%)** | **1.69** |
| Ben-Gurion/existing_trees/existing_trees_06 | 0.048s (2.7%) | 0.019s (1.1%) | 0.112s (6.2%) | **0.131s (7.3%)** | **1.334s (74.1%)** | **1.80** |
| Ben-Gurion/existing_trees/existing_trees_07 | 0.013s (0.8%) | 0.016s (0.9%) | 0.103s (6.0%) | **0.137s (8.0%)** | **1.290s (75.4%)** | **1.71** |
| Ben-Gurion/existing_trees/existing_trees_08 | 0.046s (2.6%) | 0.016s (0.9%) | 0.116s (6.6%) | **0.134s (7.6%)** | **1.282s (73.1%)** | **1.75** |
| Ben-Gurion/existing_trees/existing_trees_09 | 0.013s (0.7%) | 0.016s (0.9%) | 0.101s (5.7%) | **0.133s (7.5%)** | **1.351s (75.9%)** | **1.78** |
| Ben-Gurion/existing_trees/existing_trees_10 | 0.016s (0.9%) | 0.015s (0.8%) | 0.120s (6.7%) | **0.149s (8.3%)** | **1.338s (74.6%)** | **1.79** |
| Ben-Gurion/new_high_buildings/new_high_buildings_01 | 0.019s (1.1%) | 0.017s (1.0%) | 0.114s (6.6%) | **0.129s (7.5%)** | **1.274s (74.0%)** | **1.72** |
| Ben-Gurion/new_high_buildings/new_high_buildings_02 | 0.015s (0.8%) | 0.016s (0.8%) | 0.123s (6.5%) | **0.132s (6.9%)** | **1.452s (76.2%)** | **1.91** |
| Ben-Gurion/new_high_buildings/new_high_buildings_03 | 0.048s (2.6%) | 0.019s (1.0%) | 0.116s (6.4%) | **0.137s (7.5%)** | **1.342s (73.7%)** | **1.82** |
| Ben-Gurion/new_high_buildings/new_high_buildings_04 | 0.014s (0.8%) | 0.017s (1.0%) | 0.102s (5.9%) | **0.133s (7.7%)** | **1.294s (75.2%)** | **1.72** |
| Ben-Gurion/new_high_buildings/new_high_buildings_05 | 0.015s (0.8%) | 0.017s (0.9%) | 0.151s (8.2%) | **0.132s (7.2%)** | **1.356s (74.0%)** | **1.83** |
| Ben-Gurion/new_high_buildings/new_high_buildings_06 | 0.058s (3.3%) | 0.016s (0.9%) | 0.113s (6.3%) | **0.139s (7.8%)** | **1.289s (72.3%)** | **1.78** |
| Ben-Gurion/new_high_buildings/new_high_buildings_07 | 0.018s (1.0%) | 0.018s (1.0%) | 0.115s (6.4%) | **0.144s (8.0%)** | **1.351s (74.8%)** | **1.81** |
| Ben-Gurion/new_high_buildings/new_high_buildings_08 | 0.014s (0.8%) | 0.019s (1.1%) | 0.102s (6.0%) | **0.145s (8.5%)** | **1.255s (73.8%)** | **1.70** |
| Ben-Gurion/new_high_buildings/new_high_buildings_09 | 0.014s (0.8%) | 0.015s (0.8%) | 0.114s (6.4%) | **0.149s (8.4%)** | **1.305s (73.6%)** | **1.77** |
| Ben-Gurion/new_high_buildings/new_high_buildings_10 | 0.015s (0.8%) | 0.015s (0.8%) | 0.112s (6.2%) | **0.138s (7.7%)** | **1.305s (72.7%)** | **1.80** |
| Ben-Gurion/new_low_buildings/new_low_buildings_01 | 0.016s (0.9%) | 0.015s (0.9%) | 0.106s (6.0%) | **0.131s (7.5%)** | **1.318s (75.0%)** | **1.76** |
| Ben-Gurion/new_low_buildings/new_low_buildings_02 | 0.015s (0.8%) | 0.016s (0.9%) | 0.104s (5.8%) | **0.132s (7.4%)** | **1.358s (76.2%)** | **1.78** |
| Ben-Gurion/new_low_buildings/new_low_buildings_03 | 0.016s (0.9%) | 0.016s (0.9%) | 0.182s (10.2%) | **0.133s (7.4%)** | **1.282s (71.7%)** | **1.79** |
| Ben-Gurion/new_low_buildings/new_low_buildings_04 | 0.014s (0.8%) | 0.018s (1.1%) | 0.107s (6.4%) | **0.136s (8.1%)** | **1.252s (74.3%)** | **1.68** |
| Ben-Gurion/new_low_buildings/new_low_buildings_05 | 0.018s (1.0%) | 0.018s (1.0%) | 0.119s (6.7%) | **0.134s (7.5%)** | **1.336s (74.9%)** | **1.78** |
| Ben-Gurion/new_low_buildings/new_low_buildings_06 | 0.014s (0.8%) | 0.015s (0.9%) | 0.108s (6.1%) | **0.133s (7.6%)** | **1.327s (75.4%)** | **1.76** |
| Ben-Gurion/new_low_buildings/new_low_buildings_07 | 0.023s (1.3%) | 0.018s (1.0%) | 0.105s (5.8%) | **0.165s (9.2%)** | **1.323s (73.5%)** | **1.80** |
| Ben-Gurion/new_low_buildings/new_low_buildings_08 | 0.018s (1.0%) | 0.018s (1.0%) | 0.105s (5.8%) | **0.152s (8.4%)** | **1.355s (74.7%)** | **1.81** |
| Ben-Gurion/new_low_buildings/new_low_buildings_09 | 0.017s (0.9%) | 0.014s (0.7%) | 0.129s (6.6%) | **0.157s (8.1%)** | **1.438s (74.0%)** | **1.94** |
| Ben-Gurion/new_low_buildings/new_low_buildings_10 | 0.018s (0.9%) | 0.016s (0.8%) | 0.123s (6.2%) | **0.162s (8.1%)** | **1.473s (73.9%)** | **1.99** |
| Ben-Gurion/new_trees/new_trees_01 | 0.015s (0.9%) | 0.015s (0.9%) | 0.128s (7.4%) | **0.134s (7.7%)** | **1.285s (74.0%)** | **1.74** |
| Ben-Gurion/new_trees/new_trees_02 | 0.027s (1.5%) | 0.019s (1.0%) | 0.164s (9.0%) | **0.139s (7.6%)** | **1.286s (70.7%)** | **1.82** |
| Ben-Gurion/new_trees/new_trees_03 | 0.017s (0.9%) | 0.016s (0.9%) | 0.121s (6.7%) | **0.140s (7.7%)** | **1.350s (74.7%)** | **1.81** |
| Ben-Gurion/new_trees/new_trees_04 | 0.014s (0.8%) | 0.016s (0.9%) | 0.151s (8.2%) | **0.135s (7.4%)** | **1.351s (73.8%)** | **1.83** |
| Ben-Gurion/new_trees/new_trees_05 | 0.016s (0.9%) | 0.016s (0.9%) | 0.118s (6.5%) | **0.136s (7.5%)** | **1.351s (74.8%)** | **1.81** |
| Ben-Gurion/new_trees/new_trees_06 | 0.014s (0.7%) | 0.015s (0.8%) | 0.114s (6.1%) | **0.134s (7.1%)** | **1.340s (71.3%)** | **1.88** |
| Ben-Gurion/new_trees/new_trees_07 | 0.017s (0.9%) | 0.015s (0.8%) | 0.141s (7.5%) | **0.150s (8.0%)** | **1.378s (73.4%)** | **1.88** |
| Ben-Gurion/new_trees/new_trees_08 | 0.016s (0.8%) | 0.016s (0.8%) | 0.180s (9.4%) | **0.135s (7.1%)** | **1.366s (71.5%)** | **1.91** |
| Ben-Gurion/new_trees/new_trees_09 | 0.016s (0.9%) | 0.017s (1.0%) | 0.117s (6.6%) | **0.139s (7.8%)** | **1.330s (74.9%)** | **1.77** |
| Ben-Gurion/new_trees/new_trees_10 | 0.016s (0.9%) | 0.015s (0.8%) | 0.127s (6.9%) | **0.137s (7.5%)** | **1.367s (74.5%)** | **1.84** |
| Ness-Tziona/exploded/nes_tziona_unblock_2 | 0.017s (0.2%) | 0.015s (0.2%) | 0.275s (3.9%) | **0.170s (2.4%)** | **6.212s (89.2%)** | **6.96** |
