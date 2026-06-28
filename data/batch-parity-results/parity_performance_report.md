# WebGPU vs Python Parity & Performance Report

Generated on: 05/05/2026, 1:58:39

## 1. Summary Comparison

| Analysis | Python 1m (s) | WebGPU 1m (s) | Speedup 1m | WebGPU 12m (s) | Speedup 12m | Solar | Sky | MRT | UTCI |
| :--- | ---: | ---: | ---: | ---: | ---: | :---: | :---: | :---: | :---: |
| Ben-Gurion/existing_buildings/existing_buildings_06 | 28.1 | 1.82 | **15.4x** | 2.51 | **134.3x** | - | - | - | ✅ |
| Ben-Gurion/existing_buildings/existing_buildings_01 | 27.3 | 1.72 | **15.9x** | 2.42 | **135.7x** | - | - | - | ✅ |
| Ben-Gurion/existing_buildings/existing_buildings_03 | 28.5 | 1.73 | **16.5x** | 2.49 | **137.4x** | - | - | - | ✅ |
| Ben-Gurion/existing_buildings/existing_buildings_05 | 28.0 | 1.75 | **16.0x** | 2.48 | **135.3x** | - | - | - | ✅ |
| Ben-Gurion/existing_buildings/existing_buildings_04 | 27.8 | 1.69 | **16.5x** | 2.59 | **128.7x** | - | - | - | ✅ |
| Ben-Gurion/existing_buildings/existing_buildings_02 | 29.9 | 1.66 | **18.0x** | 2.47 | **145.4x** | - | - | - | ✅ |
| Ben-Gurion/20250815_grid_2m_fullday | 195.1 | 1.83 | **106.7x** | 2.92 | **800.8x** | ✅ | ✅ | ✅ | ✅ |
| Ben-Gurion/existing_buildings/existing_buildings_07 | 28.5 | 1.70 | **16.7x** | 2.45 | **139.2x** | - | - | - | ✅ |
| Ben-Gurion/existing_buildings/existing_buildings_10 | 28.6 | 1.79 | **15.9x** | 2.48 | **138.4x** | - | - | - | ✅ |
| Ben-Gurion/existing_buildings/existing_buildings_08 | 29.1 | 1.75 | **16.6x** | 2.45 | **142.5x** | - | - | - | ✅ |
| Ben-Gurion/existing_trees/existing_trees_02 | 28.6 | 1.79 | **16.0x** | 2.29 | **149.6x** | - | - | - | ✅ |
| Ben-Gurion/existing_trees/existing_trees_01 | 28.9 | 1.45 | **20.0x** | 2.28 | **152.0x** | - | - | - | ✅ |
| Ben-Gurion/existing_buildings/existing_buildings_09 | 29.0 | 1.65 | **17.6x** | 2.48 | **140.2x** | - | - | - | ✅ |
| Ben-Gurion/existing_trees/existing_trees_03 | 29.1 | 1.69 | **17.3x** | 2.35 | **148.9x** | - | - | - | ✅ |
| Ben-Gurion/existing_trees/existing_trees_07 | 29.4 | 1.90 | **15.5x** | 2.43 | **145.1x** | - | - | - | ✅ |
| Ben-Gurion/existing_trees/existing_trees_09 | 29.5 | 1.76 | **16.8x** | 2.49 | **142.6x** | - | - | - | ✅ |
| Ben-Gurion/existing_trees/existing_trees_08 | 29.6 | 1.70 | **17.4x** | 2.44 | **145.2x** | - | - | - | ✅ |
| Ben-Gurion/existing_trees/existing_trees_05 | 29.5 | 1.70 | **17.4x** | 2.40 | **147.4x** | - | - | - | ✅ |
| Ben-Gurion/existing_trees/existing_trees_04 | 29.4 | 1.67 | **17.6x** | 2.37 | **148.9x** | - | - | - | ✅ |
| Ben-Gurion/existing_trees/existing_trees_10 | 29.8 | 1.69 | **17.6x** | 2.47 | **144.6x** | - | - | - | ✅ |
| Ben-Gurion/existing_trees/existing_trees_06 | 29.1 | 1.67 | **17.4x** | 2.45 | **142.7x** | - | - | - | ✅ |
| Ben-Gurion/new_high_buildings/new_high_buildings_01 | 30.0 | 1.75 | **17.2x** | 2.44 | **147.8x** | - | - | - | ✅ |
| Ben-Gurion/new_high_buildings/new_high_buildings_03 | 30.1 | 1.66 | **18.1x** | 2.45 | **147.4x** | - | - | - | ✅ |
| Ben-Gurion/new_high_buildings/new_high_buildings_07 | 31.5 | 1.75 | **18.0x** | 2.60 | **145.4x** | - | - | - | ✅ |
| Ben-Gurion/new_high_buildings/new_high_buildings_04 | 30.7 | 1.75 | **17.6x** | 2.44 | **151.2x** | - | - | - | ✅ |
| Ben-Gurion/new_high_buildings/new_high_buildings_06 | 31.7 | 1.77 | **17.9x** | 2.42 | **157.3x** | - | - | - | ✅ |
| Ben-Gurion/new_high_buildings/new_high_buildings_02 | 29.8 | 1.64 | **18.2x** | 2.53 | **141.2x** | - | - | - | ✅ |
| Ben-Gurion/new_high_buildings/new_high_buildings_05 | 31.3 | 1.65 | **19.0x** | 2.55 | **147.3x** | - | - | - | ✅ |
| Ben-Gurion/new_high_buildings/new_high_buildings_08 | 32.4 | 1.71 | **19.0x** | 2.44 | **159.5x** | - | - | - | ✅ |
| Ben-Gurion/new_high_buildings/new_high_buildings_10 | 32.5 | 1.67 | **19.4x** | 2.48 | **157.5x** | - | - | - | ✅ |
| Ben-Gurion/new_low_buildings/new_low_buildings_02 | 32.4 | 1.70 | **19.0x** | 2.50 | **155.5x** | - | - | - | ✅ |
| Ben-Gurion/new_high_buildings/new_high_buildings_09 | 31.2 | 1.76 | **17.8x** | 2.52 | **148.8x** | - | - | - | ✅ |
| Ben-Gurion/new_low_buildings/new_low_buildings_01 | 31.0 | 1.76 | **17.5x** | 2.38 | **156.0x** | - | - | - | ✅ |
| Ben-Gurion/new_low_buildings/new_low_buildings_03 | 31.8 | 1.68 | **18.9x** | 2.48 | **154.1x** | - | - | - | ✅ |
| Ben-Gurion/new_low_buildings/new_low_buildings_04 | 32.0 | 1.68 | **19.1x** | 2.41 | **159.2x** | - | - | - | ✅ |
| Ben-Gurion/new_low_buildings/new_low_buildings_06 | 32.2 | 1.73 | **18.7x** | 2.40 | **160.7x** | - | - | - | ✅ |
| Ben-Gurion/new_low_buildings/new_low_buildings_10 | 33.1 | 1.76 | **18.8x** | 2.56 | **154.8x** | - | - | - | ✅ |
| Ben-Gurion/new_low_buildings/new_low_buildings_09 | 32.1 | 1.80 | **17.8x** | 2.41 | **159.6x** | - | - | - | ✅ |
| Ben-Gurion/new_trees/new_trees_01 | 33.6 | 1.69 | **19.9x** | 2.45 | **164.6x** | - | - | - | ✅ |
| Ben-Gurion/new_low_buildings/new_low_buildings_08 | 32.4 | 1.77 | **18.3x** | 2.55 | **152.7x** | - | - | - | ✅ |
| Ben-Gurion/new_low_buildings/new_low_buildings_07 | 33.1 | 1.68 | **19.7x** | 2.55 | **155.5x** | - | - | - | ✅ |
| Ben-Gurion/new_trees/new_trees_02 | 33.9 | 1.80 | **18.9x** | 2.56 | **159.1x** | - | - | - | ✅ |
| Ben-Gurion/new_low_buildings/new_low_buildings_05 | 32.0 | 1.71 | **18.7x** | 2.41 | **159.4x** | - | - | - | ✅ |
| Ben-Gurion/new_trees/new_trees_03 | 32.9 | 1.67 | **19.7x** | 2.53 | **156.1x** | - | - | - | ✅ |
| Ben-Gurion/new_trees/new_trees_04 | 34.4 | 1.75 | **19.6x** | 2.48 | **166.3x** | - | - | - | ✅ |
| Ben-Gurion/new_trees/new_trees_08 | 34.5 | 1.79 | **19.3x** | 2.76 | **150.0x** | - | - | - | ✅ |
| Ben-Gurion/new_trees/new_trees_07 | 35.5 | 1.73 | **20.5x** | 2.52 | **169.1x** | - | - | - | ✅ |
| Ben-Gurion/new_trees/new_trees_05 | 34.2 | 1.77 | **19.3x** | 2.52 | **163.4x** | - | - | - | ✅ |
| Ben-Gurion/new_trees/new_trees_06 | 35.8 | 1.70 | **21.1x** | 2.53 | **169.7x** | - | - | - | ✅ |
| Ben-Gurion/new_trees/new_trees_09 | 34.7 | 1.71 | **20.3x** | 2.52 | **165.5x** | - | - | - | ✅ |
| Ben-Gurion/new_trees/new_trees_10 | 34.9 | 1.72 | **20.3x** | 2.53 | **165.7x** | - | - | - | ✅ |
| Ness-Tziona/exploded/nes_tziona_unblock_2 | 931.0 | 6.29 | **147.9x** | 10.09 | **1107.1x** | - | - | - | ✅ |

## 2. Memory Usage (Current "Store All" Arch)

| Analysis | Grid Points | Solar (MB) | Sky (MB) | Results (MB) | **Total VRAM (MB)** |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Ben-Gurion/existing_buildings/existing_buildings_06 | 116,745 | 4.0 | 64.6 | 128.3 | **196.8** |
| Ben-Gurion/existing_buildings/existing_buildings_01 | 118,531 | 4.1 | 65.6 | 130.2 | **199.9** |
| Ben-Gurion/existing_buildings/existing_buildings_03 | 116,163 | 4.0 | 64.3 | 127.6 | **195.9** |
| Ben-Gurion/existing_buildings/existing_buildings_05 | 116,163 | 4.0 | 64.3 | 127.6 | **195.9** |
| Ben-Gurion/existing_buildings/existing_buildings_04 | 116,163 | 4.0 | 64.3 | 127.6 | **195.9** |
| Ben-Gurion/existing_buildings/existing_buildings_02 | 118,531 | 4.1 | 65.6 | 130.2 | **199.9** |
| Ben-Gurion/20250815_grid_2m_fullday | 118,531 | 4.1 | 65.6 | 130.2 | **199.9** |
| Ben-Gurion/existing_buildings/existing_buildings_07 | 118,531 | 4.1 | 65.6 | 130.2 | **199.9** |
| Ben-Gurion/existing_buildings/existing_buildings_10 | 117,569 | 4.0 | 65.0 | 129.2 | **198.2** |
| Ben-Gurion/existing_buildings/existing_buildings_08 | 118,531 | 4.1 | 65.6 | 130.2 | **199.9** |
| Ben-Gurion/existing_trees/existing_trees_02 | 107,699 | 3.7 | 59.6 | 118.3 | **181.6** |
| Ben-Gurion/existing_trees/existing_trees_01 | 105,697 | 3.6 | 58.5 | 116.1 | **178.2** |
| Ben-Gurion/existing_buildings/existing_buildings_09 | 118,531 | 4.1 | 65.6 | 130.2 | **199.9** |
| Ben-Gurion/existing_trees/existing_trees_03 | 108,389 | 3.7 | 60.0 | 119.1 | **182.8** |
| Ben-Gurion/existing_trees/existing_trees_07 | 116,516 | 4.0 | 64.4 | 128.0 | **196.5** |
| Ben-Gurion/existing_trees/existing_trees_09 | 116,653 | 4.0 | 64.5 | 128.2 | **196.7** |
| Ben-Gurion/existing_trees/existing_trees_08 | 116,516 | 4.0 | 64.4 | 128.0 | **196.5** |
| Ben-Gurion/existing_trees/existing_trees_05 | 111,004 | 3.8 | 61.4 | 122.0 | **187.2** |
| Ben-Gurion/existing_trees/existing_trees_04 | 108,388 | 3.7 | 60.0 | 119.1 | **182.8** |
| Ben-Gurion/existing_trees/existing_trees_10 | 118,256 | 4.1 | 65.4 | 129.9 | **199.4** |
| Ben-Gurion/existing_trees/existing_trees_06 | 116,516 | 4.0 | 64.4 | 128.0 | **196.5** |
| Ben-Gurion/new_high_buildings/new_high_buildings_01 | 118,531 | 4.1 | 65.6 | 130.2 | **199.9** |
| Ben-Gurion/new_high_buildings/new_high_buildings_03 | 118,531 | 4.1 | 65.6 | 130.2 | **199.9** |
| Ben-Gurion/new_high_buildings/new_high_buildings_07 | 118,531 | 4.1 | 65.6 | 130.2 | **199.9** |
| Ben-Gurion/new_high_buildings/new_high_buildings_04 | 118,531 | 4.1 | 65.6 | 130.2 | **199.9** |
| Ben-Gurion/new_high_buildings/new_high_buildings_06 | 118,531 | 4.1 | 65.6 | 130.2 | **199.9** |
| Ben-Gurion/new_high_buildings/new_high_buildings_02 | 118,531 | 4.1 | 65.6 | 130.2 | **199.9** |
| Ben-Gurion/new_high_buildings/new_high_buildings_05 | 118,531 | 4.1 | 65.6 | 130.2 | **199.9** |
| Ben-Gurion/new_high_buildings/new_high_buildings_08 | 118,531 | 4.1 | 65.6 | 130.2 | **199.9** |
| Ben-Gurion/new_high_buildings/new_high_buildings_10 | 118,531 | 4.1 | 65.6 | 130.2 | **199.9** |
| Ben-Gurion/new_low_buildings/new_low_buildings_02 | 118,531 | 4.1 | 65.6 | 130.2 | **199.9** |
| Ben-Gurion/new_high_buildings/new_high_buildings_09 | 118,531 | 4.1 | 65.6 | 130.2 | **199.9** |
| Ben-Gurion/new_low_buildings/new_low_buildings_01 | 118,531 | 4.1 | 65.6 | 130.2 | **199.9** |
| Ben-Gurion/new_low_buildings/new_low_buildings_03 | 118,531 | 4.1 | 65.6 | 130.2 | **199.9** |
| Ben-Gurion/new_low_buildings/new_low_buildings_04 | 118,531 | 4.1 | 65.6 | 130.2 | **199.9** |
| Ben-Gurion/new_low_buildings/new_low_buildings_06 | 118,531 | 4.1 | 65.6 | 130.2 | **199.9** |
| Ben-Gurion/new_low_buildings/new_low_buildings_10 | 118,531 | 4.1 | 65.6 | 130.2 | **199.9** |
| Ben-Gurion/new_low_buildings/new_low_buildings_09 | 118,531 | 4.1 | 65.6 | 130.2 | **199.9** |
| Ben-Gurion/new_trees/new_trees_01 | 118,531 | 4.1 | 65.6 | 130.2 | **199.9** |
| Ben-Gurion/new_low_buildings/new_low_buildings_08 | 118,531 | 4.1 | 65.6 | 130.2 | **199.9** |
| Ben-Gurion/new_low_buildings/new_low_buildings_07 | 118,531 | 4.1 | 65.6 | 130.2 | **199.9** |
| Ben-Gurion/new_trees/new_trees_02 | 118,531 | 4.1 | 65.6 | 130.2 | **199.9** |
| Ben-Gurion/new_low_buildings/new_low_buildings_05 | 118,531 | 4.1 | 65.6 | 130.2 | **199.9** |
| Ben-Gurion/new_trees/new_trees_03 | 118,531 | 4.1 | 65.6 | 130.2 | **199.9** |
| Ben-Gurion/new_trees/new_trees_04 | 118,531 | 4.1 | 65.6 | 130.2 | **199.9** |
| Ben-Gurion/new_trees/new_trees_08 | 118,531 | 4.1 | 65.6 | 130.2 | **199.9** |
| Ben-Gurion/new_trees/new_trees_07 | 118,531 | 4.1 | 65.6 | 130.2 | **199.9** |
| Ben-Gurion/new_trees/new_trees_05 | 118,531 | 4.1 | 65.6 | 130.2 | **199.9** |
| Ben-Gurion/new_trees/new_trees_06 | 118,531 | 4.1 | 65.6 | 130.2 | **199.9** |
| Ben-Gurion/new_trees/new_trees_09 | 118,531 | 4.1 | 65.6 | 130.2 | **199.9** |
| Ben-Gurion/new_trees/new_trees_10 | 118,531 | 4.1 | 65.6 | 130.2 | **199.9** |
| Ness-Tziona/exploded/nes_tziona_unblock_2 | 512,637 | 17.6 | 283.6 | 563.2 | **864.4** |

## 3. Detailed 12-Month Timing Breakdown

| Analysis | Weather | Init | BVH | **GPU Compute** | **Readback/Wait** | **Total (s)** |
| :--- | :--- | :--- | :--- | :--- | :--- | ---: |
| Ben-Gurion/existing_buildings/existing_buildings_06 | 0.019s (0.8%) | 0.103s (4.1%) | 0.097s (3.9%) | **0.183s (7.3%)** | **2.013s (80.2%)** | **2.51** |
| Ben-Gurion/existing_buildings/existing_buildings_01 | 0.016s (0.7%) | 0.015s (0.6%) | 0.081s (3.4%) | **0.160s (6.6%)** | **1.971s (81.5%)** | **2.42** |
| Ben-Gurion/existing_buildings/existing_buildings_03 | 0.023s (0.9%) | 0.017s (0.7%) | 0.087s (3.5%) | **0.160s (6.4%)** | **2.013s (80.9%)** | **2.49** |
| Ben-Gurion/existing_buildings/existing_buildings_05 | 0.020s (0.8%) | 0.113s (4.5%) | 0.100s (4.0%) | **0.157s (6.3%)** | **2.005s (80.7%)** | **2.48** |
| Ben-Gurion/existing_buildings/existing_buildings_04 | 0.016s (0.6%) | 0.015s (0.6%) | 0.095s (3.7%) | **0.170s (6.6%)** | **2.115s (81.8%)** | **2.59** |
| Ben-Gurion/existing_buildings/existing_buildings_02 | 0.015s (0.6%) | 0.108s (4.4%) | 0.110s (4.5%) | **0.153s (6.2%)** | **1.994s (80.8%)** | **2.47** |
| Ben-Gurion/20250815_grid_2m_fullday | 0.018s (0.6%) | 0.015s (0.5%) | 0.156s (5.3%) | **0.216s (7.4%)** | **2.351s (80.4%)** | **2.92** |
| Ben-Gurion/existing_buildings/existing_buildings_07 | 0.021s (0.9%) | 0.098s (4.0%) | 0.095s (3.9%) | **0.162s (6.6%)** | **1.993s (81.2%)** | **2.45** |
| Ben-Gurion/existing_buildings/existing_buildings_10 | 0.015s (0.6%) | 0.102s (4.1%) | 0.093s (3.8%) | **0.161s (6.5%)** | **2.026s (81.7%)** | **2.48** |
| Ben-Gurion/existing_buildings/existing_buildings_08 | 0.016s (0.7%) | 0.015s (0.6%) | 0.088s (3.6%) | **0.157s (6.4%)** | **1.991s (81.2%)** | **2.45** |
| Ben-Gurion/existing_trees/existing_trees_02 | 0.017s (0.7%) | 0.017s (0.7%) | 0.082s (3.6%) | **0.154s (6.7%)** | **1.850s (80.6%)** | **2.29** |
| Ben-Gurion/existing_trees/existing_trees_01 | 0.021s (0.9%) | 0.052s (2.3%) | 0.083s (3.6%) | **0.187s (8.2%)** | **1.912s (83.9%)** | **2.28** |
| Ben-Gurion/existing_buildings/existing_buildings_09 | 0.023s (0.9%) | 0.015s (0.6%) | 0.096s (3.9%) | **0.161s (6.5%)** | **2.006s (80.7%)** | **2.48** |
| Ben-Gurion/existing_trees/existing_trees_03 | 0.017s (0.7%) | 0.015s (0.6%) | 0.095s (4.0%) | **0.156s (6.6%)** | **1.880s (80.1%)** | **2.35** |
| Ben-Gurion/existing_trees/existing_trees_07 | 0.013s (0.5%) | 0.016s (0.7%) | 0.085s (3.5%) | **0.154s (6.3%)** | **1.982s (81.5%)** | **2.43** |
| Ben-Gurion/existing_trees/existing_trees_09 | 0.021s (0.8%) | 0.020s (0.8%) | 0.095s (3.8%) | **0.153s (6.2%)** | **2.019s (81.2%)** | **2.49** |
| Ben-Gurion/existing_trees/existing_trees_08 | 0.036s (1.5%) | 0.018s (0.7%) | 0.090s (3.7%) | **0.152s (6.2%)** | **1.967s (80.4%)** | **2.44** |
| Ben-Gurion/existing_trees/existing_trees_05 | 0.021s (0.9%) | 0.019s (0.8%) | 0.095s (4.0%) | **0.148s (6.2%)** | **1.931s (80.3%)** | **2.40** |
| Ben-Gurion/existing_trees/existing_trees_04 | 0.019s (0.8%) | 0.016s (0.7%) | 0.093s (3.9%) | **0.155s (6.5%)** | **1.902s (80.2%)** | **2.37** |
| Ben-Gurion/existing_trees/existing_trees_10 | 0.015s (0.6%) | 0.017s (0.7%) | 0.109s (4.4%) | **0.161s (6.5%)** | **1.988s (80.4%)** | **2.47** |
| Ben-Gurion/existing_trees/existing_trees_06 | 0.017s (0.7%) | 0.016s (0.7%) | 0.089s (3.6%) | **0.156s (6.4%)** | **1.989s (81.3%)** | **2.45** |
| Ben-Gurion/new_high_buildings/new_high_buildings_01 | 0.026s (1.1%) | 0.014s (0.6%) | 0.092s (3.8%) | **0.160s (6.6%)** | **1.961s (80.5%)** | **2.44** |
| Ben-Gurion/new_high_buildings/new_high_buildings_03 | 0.014s (0.6%) | 0.016s (0.7%) | 0.085s (3.5%) | **0.161s (6.6%)** | **1.991s (81.4%)** | **2.45** |
| Ben-Gurion/new_high_buildings/new_high_buildings_07 | 0.015s (0.6%) | 0.015s (0.6%) | 0.086s (3.3%) | **0.158s (6.1%)** | **2.143s (82.6%)** | **2.60** |
| Ben-Gurion/new_high_buildings/new_high_buildings_04 | 0.016s (0.7%) | 0.015s (0.6%) | 0.087s (3.6%) | **0.153s (6.3%)** | **1.959s (80.4%)** | **2.44** |
| Ben-Gurion/new_high_buildings/new_high_buildings_06 | 0.015s (0.6%) | 0.016s (0.7%) | 0.094s (3.9%) | **0.156s (6.4%)** | **1.946s (80.4%)** | **2.42** |
| Ben-Gurion/new_high_buildings/new_high_buildings_02 | 0.021s (0.8%) | 0.020s (0.8%) | 0.165s (6.5%) | **0.163s (6.4%)** | **1.965s (77.5%)** | **2.53** |
| Ben-Gurion/new_high_buildings/new_high_buildings_05 | 0.030s (1.2%) | 0.018s (0.7%) | 0.157s (6.1%) | **0.179s (7.0%)** | **1.966s (77.0%)** | **2.55** |
| Ben-Gurion/new_high_buildings/new_high_buildings_08 | 0.039s (1.6%) | 0.019s (0.8%) | 0.095s (3.9%) | **0.157s (6.4%)** | **1.943s (79.7%)** | **2.44** |
| Ben-Gurion/new_high_buildings/new_high_buildings_10 | 0.025s (1.0%) | 0.018s (0.7%) | 0.089s (3.6%) | **0.181s (7.3%)** | **1.980s (79.9%)** | **2.48** |
| Ben-Gurion/new_low_buildings/new_low_buildings_02 | 0.020s (0.8%) | 0.019s (0.8%) | 0.146s (5.8%) | **0.163s (6.5%)** | **1.939s (77.6%)** | **2.50** |
| Ben-Gurion/new_high_buildings/new_high_buildings_09 | 0.017s (0.7%) | 0.015s (0.6%) | 0.114s (4.5%) | **0.175s (7.0%)** | **2.019s (80.2%)** | **2.52** |
| Ben-Gurion/new_low_buildings/new_low_buildings_01 | 0.014s (0.6%) | 0.017s (0.7%) | 0.088s (3.7%) | **0.151s (6.3%)** | **1.931s (81.1%)** | **2.38** |
| Ben-Gurion/new_low_buildings/new_low_buildings_03 | 0.015s (0.6%) | 0.016s (0.6%) | 0.098s (4.0%) | **0.175s (7.1%)** | **1.967s (79.4%)** | **2.48** |
| Ben-Gurion/new_low_buildings/new_low_buildings_04 | 0.015s (0.6%) | 0.015s (0.6%) | 0.096s (4.0%) | **0.158s (6.5%)** | **1.938s (80.3%)** | **2.41** |
| Ben-Gurion/new_low_buildings/new_low_buildings_06 | 0.016s (0.7%) | 0.016s (0.7%) | 0.098s (4.1%) | **0.159s (6.6%)** | **1.933s (80.4%)** | **2.40** |
| Ben-Gurion/new_low_buildings/new_low_buildings_10 | 0.040s (1.6%) | 0.017s (0.7%) | 0.156s (6.1%) | **0.230s (9.0%)** | **1.940s (75.7%)** | **2.56** |
| Ben-Gurion/new_low_buildings/new_low_buildings_09 | 0.015s (0.6%) | 0.016s (0.7%) | 0.097s (4.0%) | **0.161s (6.7%)** | **1.948s (80.7%)** | **2.41** |
| Ben-Gurion/new_trees/new_trees_01 | 0.021s (0.9%) | 0.019s (0.8%) | 0.090s (3.7%) | **0.159s (6.5%)** | **1.975s (80.7%)** | **2.45** |
| Ben-Gurion/new_low_buildings/new_low_buildings_08 | 0.018s (0.7%) | 0.017s (0.7%) | 0.105s (4.1%) | **0.180s (7.1%)** | **2.021s (79.3%)** | **2.55** |
| Ben-Gurion/new_low_buildings/new_low_buildings_07 | 0.018s (0.7%) | 0.015s (0.6%) | 0.146s (5.7%) | **0.236s (9.3%)** | **1.942s (76.1%)** | **2.55** |
| Ben-Gurion/new_trees/new_trees_02 | 0.016s (0.6%) | 0.016s (0.6%) | 0.098s (3.8%) | **0.154s (6.0%)** | **2.105s (82.3%)** | **2.56** |
| Ben-Gurion/new_low_buildings/new_low_buildings_05 | 0.014s (0.6%) | 0.015s (0.6%) | 0.092s (3.8%) | **0.159s (6.6%)** | **1.943s (80.7%)** | **2.41** |
| Ben-Gurion/new_trees/new_trees_03 | 0.020s (0.8%) | 0.019s (0.8%) | 0.151s (6.0%) | **0.175s (6.9%)** | **1.960s (77.5%)** | **2.53** |
| Ben-Gurion/new_trees/new_trees_04 | 0.016s (0.6%) | 0.015s (0.6%) | 0.093s (3.7%) | **0.155s (6.2%)** | **2.020s (81.4%)** | **2.48** |
| Ben-Gurion/new_trees/new_trees_08 | 0.019s (0.7%) | 0.019s (0.7%) | 0.109s (4.0%) | **0.167s (6.1%)** | **2.261s (81.9%)** | **2.76** |
| Ben-Gurion/new_trees/new_trees_07 | 0.016s (0.6%) | 0.014s (0.6%) | 0.094s (3.7%) | **0.161s (6.4%)** | **2.058s (81.6%)** | **2.52** |
| Ben-Gurion/new_trees/new_trees_05 | 0.015s (0.6%) | 0.016s (0.6%) | 0.094s (3.7%) | **0.155s (6.2%)** | **2.063s (82.0%)** | **2.52** |
| Ben-Gurion/new_trees/new_trees_06 | 0.015s (0.6%) | 0.015s (0.6%) | 0.096s (3.8%) | **0.150s (5.9%)** | **2.070s (81.8%)** | **2.53** |
| Ben-Gurion/new_trees/new_trees_09 | 0.015s (0.6%) | 0.016s (0.6%) | 0.097s (3.9%) | **0.160s (6.4%)** | **2.043s (81.2%)** | **2.52** |
| Ben-Gurion/new_trees/new_trees_10 | 0.015s (0.6%) | 0.016s (0.6%) | 0.099s (3.9%) | **0.151s (6.0%)** | **2.054s (81.3%)** | **2.53** |
| Ness-Tziona/exploded/nes_tziona_unblock_2 | 0.016s (0.2%) | 0.017s (0.2%) | 0.248s (2.5%) | **0.236s (2.3%)** | **9.281s (92.0%)** | **10.09** |

## 4. Detailed 1-Month Timing Breakdown

| Analysis | Weather | Init | BVH | **GPU Compute** | **Readback/Wait** | **Total (s)** |
| :--- | :--- | :--- | :--- | :--- | :--- | ---: |
| Ben-Gurion/existing_buildings/existing_buildings_06 | 0.019s (1.0%) | 0.095s (5.2%) | 0.105s (5.8%) | **0.237s (13.0%)** | **1.280s (70.2%)** | **1.82** |
| Ben-Gurion/existing_buildings/existing_buildings_01 | 0.015s (0.9%) | 0.113s (6.6%) | 0.108s (6.3%) | **0.209s (12.1%)** | **1.194s (69.3%)** | **1.72** |
| Ben-Gurion/existing_buildings/existing_buildings_03 | 0.016s (0.9%) | 0.097s (5.6%) | 0.095s (5.5%) | **0.195s (11.3%)** | **1.213s (70.2%)** | **1.73** |
| Ben-Gurion/existing_buildings/existing_buildings_05 | 0.020s (1.1%) | 0.104s (5.9%) | 0.148s (8.4%) | **0.221s (12.6%)** | **1.167s (66.5%)** | **1.75** |
| Ben-Gurion/existing_buildings/existing_buildings_04 | 0.025s (1.5%) | 0.017s (1.0%) | 0.099s (5.9%) | **0.179s (10.6%)** | **1.174s (69.7%)** | **1.69** |
| Ben-Gurion/existing_buildings/existing_buildings_02 | 0.015s (0.9%) | 0.102s (6.1%) | 0.107s (6.4%) | **0.166s (10.0%)** | **1.183s (71.3%)** | **1.66** |
| Ben-Gurion/20250815_grid_2m_fullday | 0.019s (1.0%) | 0.014s (0.8%) | 0.109s (6.0%) | **0.173s (9.5%)** | **1.342s (73.4%)** | **1.83** |
| Ben-Gurion/existing_buildings/existing_buildings_07 | 0.015s (0.9%) | 0.107s (6.3%) | 0.122s (7.2%) | **0.157s (9.2%)** | **1.212s (71.3%)** | **1.70** |
| Ben-Gurion/existing_buildings/existing_buildings_10 | 0.023s (1.3%) | 0.095s (5.3%) | 0.111s (6.2%) | **0.169s (9.4%)** | **1.262s (70.4%)** | **1.79** |
| Ben-Gurion/existing_buildings/existing_buildings_08 | 0.019s (1.1%) | 0.019s (1.1%) | 0.105s (6.0%) | **0.176s (10.1%)** | **1.247s (71.3%)** | **1.75** |
| Ben-Gurion/existing_trees/existing_trees_02 | 0.019s (1.1%) | 0.018s (1.0%) | 0.148s (8.3%) | **0.236s (13.2%)** | **1.164s (65.0%)** | **1.79** |
| Ben-Gurion/existing_trees/existing_trees_01 | 0.016s (1.1%) | 0.045s (3.1%) | 0.086s (6.0%) | **0.150s (10.4%)** | **1.113s (77.0%)** | **1.45** |
| Ben-Gurion/existing_buildings/existing_buildings_09 | 0.015s (0.9%) | 0.098s (5.9%) | 0.093s (5.6%) | **0.178s (10.8%)** | **1.188s (72.0%)** | **1.65** |
| Ben-Gurion/existing_trees/existing_trees_03 | 0.017s (1.0%) | 0.014s (0.8%) | 0.087s (5.2%) | **0.178s (10.6%)** | **1.216s (72.1%)** | **1.69** |
| Ben-Gurion/existing_trees/existing_trees_07 | 0.024s (1.3%) | 0.019s (1.0%) | 0.117s (6.2%) | **0.186s (9.8%)** | **1.349s (71.1%)** | **1.90** |
| Ben-Gurion/existing_trees/existing_trees_09 | 0.017s (1.0%) | 0.017s (1.0%) | 0.123s (7.0%) | **0.153s (8.7%)** | **1.237s (70.4%)** | **1.76** |
| Ben-Gurion/existing_trees/existing_trees_08 | 0.016s (0.9%) | 0.015s (0.9%) | 0.086s (5.1%) | **0.179s (10.5%)** | **1.213s (71.5%)** | **1.70** |
| Ben-Gurion/existing_trees/existing_trees_05 | 0.015s (0.9%) | 0.014s (0.8%) | 0.102s (6.0%) | **0.170s (10.0%)** | **1.198s (70.3%)** | **1.70** |
| Ben-Gurion/existing_trees/existing_trees_04 | 0.015s (0.9%) | 0.013s (0.8%) | 0.094s (5.6%) | **0.171s (10.2%)** | **1.186s (71.1%)** | **1.67** |
| Ben-Gurion/existing_trees/existing_trees_10 | 0.016s (0.9%) | 0.015s (0.9%) | 0.105s (6.2%) | **0.155s (9.2%)** | **1.217s (72.0%)** | **1.69** |
| Ben-Gurion/existing_trees/existing_trees_06 | 0.015s (0.9%) | 0.015s (0.9%) | 0.089s (5.3%) | **0.180s (10.8%)** | **1.194s (71.3%)** | **1.67** |
| Ben-Gurion/new_high_buildings/new_high_buildings_01 | 0.022s (1.3%) | 0.016s (0.9%) | 0.094s (5.4%) | **0.174s (10.0%)** | **1.245s (71.2%)** | **1.75** |
| Ben-Gurion/new_high_buildings/new_high_buildings_03 | 0.018s (1.1%) | 0.020s (1.2%) | 0.101s (6.1%) | **0.170s (10.3%)** | **1.164s (70.2%)** | **1.66** |
| Ben-Gurion/new_high_buildings/new_high_buildings_07 | 0.020s (1.1%) | 0.018s (1.0%) | 0.106s (6.1%) | **0.169s (9.7%)** | **1.244s (71.2%)** | **1.75** |
| Ben-Gurion/new_high_buildings/new_high_buildings_04 | 0.017s (1.0%) | 0.018s (1.0%) | 0.108s (6.2%) | **0.155s (8.9%)** | **1.260s (72.1%)** | **1.75** |
| Ben-Gurion/new_high_buildings/new_high_buildings_06 | 0.014s (0.8%) | 0.015s (0.8%) | 0.101s (5.7%) | **0.171s (9.7%)** | **1.276s (72.2%)** | **1.77** |
| Ben-Gurion/new_high_buildings/new_high_buildings_02 | 0.015s (0.9%) | 0.015s (0.9%) | 0.090s (5.5%) | **0.173s (10.5%)** | **1.155s (70.4%)** | **1.64** |
| Ben-Gurion/new_high_buildings/new_high_buildings_05 | 0.022s (1.3%) | 0.018s (1.1%) | 0.099s (6.0%) | **0.181s (10.9%)** | **1.146s (69.3%)** | **1.65** |
| Ben-Gurion/new_high_buildings/new_high_buildings_08 | 0.044s (2.6%) | 0.016s (0.9%) | 0.114s (6.7%) | **0.159s (9.3%)** | **1.181s (69.2%)** | **1.71** |
| Ben-Gurion/new_high_buildings/new_high_buildings_10 | 0.049s (2.9%) | 0.015s (0.9%) | 0.106s (6.3%) | **0.158s (9.4%)** | **1.149s (68.7%)** | **1.67** |
| Ben-Gurion/new_low_buildings/new_low_buildings_02 | 0.026s (1.5%) | 0.018s (1.1%) | 0.106s (6.2%) | **0.171s (10.0%)** | **1.193s (70.1%)** | **1.70** |
| Ben-Gurion/new_high_buildings/new_high_buildings_09 | 0.017s (1.0%) | 0.017s (1.0%) | 0.187s (10.6%) | **0.179s (10.2%)** | **1.180s (67.2%)** | **1.76** |
| Ben-Gurion/new_low_buildings/new_low_buildings_01 | 0.018s (1.0%) | 0.018s (1.0%) | 0.115s (6.5%) | **0.150s (8.5%)** | **1.236s (70.1%)** | **1.76** |
| Ben-Gurion/new_low_buildings/new_low_buildings_03 | 0.016s (1.0%) | 0.015s (0.9%) | 0.095s (5.6%) | **0.180s (10.7%)** | **1.194s (70.9%)** | **1.68** |
| Ben-Gurion/new_low_buildings/new_low_buildings_04 | 0.017s (1.0%) | 0.019s (1.1%) | 0.100s (6.0%) | **0.153s (9.1%)** | **1.204s (71.8%)** | **1.68** |
| Ben-Gurion/new_low_buildings/new_low_buildings_06 | 0.019s (1.1%) | 0.015s (0.9%) | 0.117s (6.8%) | **0.150s (8.7%)** | **1.227s (71.1%)** | **1.73** |
| Ben-Gurion/new_low_buildings/new_low_buildings_10 | 0.017s (1.0%) | 0.015s (0.9%) | 0.114s (6.5%) | **0.185s (10.5%)** | **1.227s (69.6%)** | **1.76** |
| Ben-Gurion/new_low_buildings/new_low_buildings_09 | 0.024s (1.3%) | 0.015s (0.8%) | 0.096s (5.3%) | **0.172s (9.6%)** | **1.308s (72.6%)** | **1.80** |
| Ben-Gurion/new_trees/new_trees_01 | 0.017s (1.0%) | 0.019s (1.1%) | 0.095s (5.6%) | **0.176s (10.4%)** | **1.180s (70.0%)** | **1.69** |
| Ben-Gurion/new_low_buildings/new_low_buildings_08 | 0.023s (1.3%) | 0.018s (1.0%) | 0.114s (6.4%) | **0.160s (9.0%)** | **1.267s (71.5%)** | **1.77** |
| Ben-Gurion/new_low_buildings/new_low_buildings_07 | 0.016s (1.0%) | 0.015s (0.9%) | 0.112s (6.7%) | **0.170s (10.1%)** | **1.182s (70.6%)** | **1.68** |
| Ben-Gurion/new_trees/new_trees_02 | 0.035s (1.9%) | 0.019s (1.1%) | 0.155s (8.6%) | **0.200s (11.1%)** | **1.190s (66.3%)** | **1.80** |
| Ben-Gurion/new_low_buildings/new_low_buildings_05 | 0.016s (0.9%) | 0.015s (0.9%) | 0.116s (6.8%) | **0.160s (9.3%)** | **1.194s (69.7%)** | **1.71** |
| Ben-Gurion/new_trees/new_trees_03 | 0.014s (0.8%) | 0.017s (1.0%) | 0.111s (6.7%) | **0.168s (10.1%)** | **1.179s (70.6%)** | **1.67** |
| Ben-Gurion/new_trees/new_trees_04 | 0.016s (0.9%) | 0.015s (0.9%) | 0.123s (7.0%) | **0.192s (10.9%)** | **1.230s (70.1%)** | **1.75** |
| Ben-Gurion/new_trees/new_trees_08 | 0.020s (1.1%) | 0.016s (0.9%) | 0.115s (6.4%) | **0.151s (8.4%)** | **1.295s (72.4%)** | **1.79** |
| Ben-Gurion/new_trees/new_trees_07 | 0.014s (0.8%) | 0.017s (1.0%) | 0.099s (5.7%) | **0.175s (10.1%)** | **1.243s (71.7%)** | **1.73** |
| Ben-Gurion/new_trees/new_trees_05 | 0.019s (1.1%) | 0.018s (1.0%) | 0.115s (6.5%) | **0.148s (8.4%)** | **1.239s (70.0%)** | **1.77** |
| Ben-Gurion/new_trees/new_trees_06 | 0.016s (0.9%) | 0.015s (0.9%) | 0.104s (6.1%) | **0.151s (8.9%)** | **1.203s (70.9%)** | **1.70** |
| Ben-Gurion/new_trees/new_trees_09 | 0.015s (0.9%) | 0.017s (1.0%) | 0.116s (6.8%) | **0.169s (9.9%)** | **1.202s (70.3%)** | **1.71** |
| Ben-Gurion/new_trees/new_trees_10 | 0.017s (1.0%) | 0.016s (0.9%) | 0.106s (6.2%) | **0.154s (8.9%)** | **1.226s (71.2%)** | **1.72** |
| Ness-Tziona/exploded/nes_tziona_unblock_2 | 0.033s (0.5%) | 0.015s (0.2%) | 0.220s (3.5%) | **0.222s (3.5%)** | **5.528s (87.8%)** | **6.29** |
