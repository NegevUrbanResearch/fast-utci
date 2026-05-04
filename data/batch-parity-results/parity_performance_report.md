# WebGPU vs Python Parity & Performance Report

Generated on: 05/05/2026, 0:22:25

| Analysis | Python (s) | WebGPU 1m (s) | Speedup 1m | WebGPU 12m (s) | Speedup 12m | Solar | Sky | MRT | UTCI | Collect (s) |
| :--- | ---: | ---: | ---: | ---: | ---: | :---: | :---: | :---: | :---: | ---: |
| Ben-Gurion/20250815_grid_2m_fullday | 195.1 | 1.95 | **100.1x** | 2.94 | **796.2x** | ✅ | ✅ | ✅ | ✅ | 0.0 |
| Ben-Gurion/existing_buildings/existing_buildings_01 | 27.3 | 1.73 | **15.8x** | 2.49 | **131.8x** | - | - | - | ✅ | 0.0 |
| Ben-Gurion/existing_buildings/existing_buildings_02 | 29.9 | 1.67 | **18.0x** | 2.51 | **142.9x** | - | - | - | ✅ | 0.0 |
| Ben-Gurion/existing_buildings/existing_buildings_03 | 28.5 | 1.76 | **16.2x** | 2.75 | **124.3x** | - | - | - | ✅ | 0.0 |
| Ben-Gurion/existing_buildings/existing_buildings_04 | 27.8 | 1.62 | **17.1x** | 2.42 | **137.7x** | - | - | - | ✅ | 0.0 |
| Ben-Gurion/existing_buildings/existing_buildings_05 | 28.0 | 1.69 | **16.6x** | 2.49 | **135.0x** | - | - | - | ✅ | 0.0 |
| Ben-Gurion/existing_buildings/existing_buildings_06 | 28.1 | 1.71 | **16.4x** | 2.49 | **135.6x** | - | - | - | ✅ | 0.0 |
| Ben-Gurion/existing_buildings/existing_buildings_07 | 28.5 | 1.68 | **17.0x** | 2.53 | **134.9x** | - | - | - | ✅ | 0.0 |
| Ben-Gurion/existing_buildings/existing_buildings_08 | 29.1 | 1.66 | **17.5x** | 2.63 | **132.5x** | - | - | - | ✅ | 0.0 |
| Ben-Gurion/existing_buildings/existing_buildings_09 | 29.0 | 1.76 | **16.5x** | 2.52 | **138.4x** | - | - | - | ✅ | 0.0 |
| Ben-Gurion/existing_buildings/existing_buildings_10 | 28.6 | 1.77 | **16.1x** | 2.55 | **134.3x** | - | - | - | ✅ | 0.0 |
| Ben-Gurion/existing_trees/existing_trees_01 | 28.9 | 1.45 | **19.9x** | 2.22 | **156.4x** | - | - | - | ✅ | 0.0 |
| Ben-Gurion/existing_trees/existing_trees_02 | 28.6 | 1.61 | **17.7x** | 2.32 | **148.0x** | - | - | - | ✅ | 0.0 |
| Ben-Gurion/existing_trees/existing_trees_03 | 29.1 | 1.57 | **18.5x** | 2.35 | **148.8x** | - | - | - | ✅ | 0.0 |
| Ben-Gurion/existing_trees/existing_trees_04 | 29.4 | 1.72 | **17.1x** | 2.68 | **132.0x** | - | - | - | ✅ | 0.0 |
| Ben-Gurion/existing_trees/existing_trees_05 | 29.5 | 1.66 | **17.8x** | 2.46 | **144.0x** | - | - | - | ✅ | 0.0 |
| Ben-Gurion/existing_trees/existing_trees_06 | 29.1 | 1.69 | **17.2x** | 2.46 | **142.0x** | - | - | - | ✅ | 0.0 |
| Ben-Gurion/existing_trees/existing_trees_07 | 29.4 | 1.66 | **17.7x** | 2.48 | **142.5x** | - | - | - | ✅ | 0.0 |
| Ben-Gurion/existing_trees/existing_trees_08 | 29.6 | 1.64 | **18.0x** | 2.46 | **144.3x** | - | - | - | ✅ | 0.0 |
| Ben-Gurion/existing_trees/existing_trees_09 | 29.5 | 1.63 | **18.1x** | 2.56 | **138.4x** | - | - | - | ✅ | 0.0 |
| Ben-Gurion/existing_trees/existing_trees_10 | 29.8 | 1.68 | **17.7x** | 2.48 | **144.1x** | - | - | - | ✅ | 0.0 |
| Ben-Gurion/new_high_buildings/new_high_buildings_01 | 30.0 | 1.69 | **17.7x** | 2.50 | **144.2x** | - | - | - | ✅ | 0.0 |
| Ben-Gurion/new_high_buildings/new_high_buildings_02 | 29.8 | 1.77 | **16.8x** | 2.53 | **141.6x** | - | - | - | ✅ | 0.0 |
| Ben-Gurion/new_high_buildings/new_high_buildings_03 | 30.1 | 1.67 | **18.0x** | 2.48 | **145.7x** | - | - | - | ✅ | 0.0 |
| Ben-Gurion/new_high_buildings/new_high_buildings_04 | 30.7 | 1.63 | **18.8x** | 2.44 | **151.2x** | - | - | - | ✅ | 0.0 |
| Ben-Gurion/new_high_buildings/new_high_buildings_05 | 31.3 | 1.70 | **18.4x** | 2.65 | **141.7x** | - | - | - | ✅ | 0.0 |
| Ben-Gurion/new_high_buildings/new_high_buildings_06 | 31.7 | 1.68 | **18.9x** | 2.46 | **154.5x** | - | - | - | ✅ | 0.0 |
| Ben-Gurion/new_high_buildings/new_high_buildings_07 | 31.5 | 1.63 | **19.3x** | 2.44 | **154.4x** | - | - | - | ✅ | 0.0 |
| Ben-Gurion/new_high_buildings/new_high_buildings_08 | 32.4 | 1.74 | **18.6x** | 2.42 | **160.9x** | - | - | - | ✅ | 0.0 |
| Ben-Gurion/new_high_buildings/new_high_buildings_09 | 31.2 | 1.62 | **19.3x** | 2.48 | **151.1x** | - | - | - | ✅ | 0.0 |
| Ben-Gurion/new_high_buildings/new_high_buildings_10 | 32.5 | 1.74 | **18.7x** | 2.48 | **157.2x** | - | - | - | ✅ | 0.0 |
| Ben-Gurion/new_low_buildings/new_low_buildings_01 | 31.0 | 1.75 | **17.7x** | 2.47 | **150.2x** | - | - | - | ✅ | 0.0 |
| Ben-Gurion/new_low_buildings/new_low_buildings_02 | 32.4 | 1.64 | **19.8x** | 2.45 | **158.4x** | - | - | - | ✅ | 0.0 |
| Ben-Gurion/new_low_buildings/new_low_buildings_03 | 31.8 | 1.64 | **19.5x** | 2.45 | **156.1x** | - | - | - | ✅ | 0.0 |
| Ben-Gurion/new_low_buildings/new_low_buildings_04 | 32.0 | 1.70 | **18.9x** | 2.43 | **158.3x** | - | - | - | ✅ | 0.0 |
| Ben-Gurion/new_low_buildings/new_low_buildings_05 | 32.0 | 1.73 | **18.4x** | 2.42 | **158.9x** | - | - | - | ✅ | 0.0 |
| Ben-Gurion/new_low_buildings/new_low_buildings_06 | 32.2 | 1.66 | **19.4x** | 2.64 | **146.3x** | - | - | - | ✅ | 0.0 |
| Ben-Gurion/new_low_buildings/new_low_buildings_07 | 33.1 | 1.68 | **19.7x** | 2.45 | **161.8x** | - | - | - | ✅ | 0.0 |
| Ben-Gurion/new_low_buildings/new_low_buildings_08 | 32.4 | 1.67 | **19.4x** | 2.45 | **158.8x** | - | - | - | ✅ | 0.0 |
| Ben-Gurion/new_low_buildings/new_low_buildings_09 | 32.1 | 1.64 | **19.5x** | 2.44 | **158.0x** | - | - | - | ✅ | 0.0 |
| Ben-Gurion/new_low_buildings/new_low_buildings_10 | 33.1 | 1.66 | **19.9x** | 2.72 | **145.7x** | - | - | - | ✅ | 0.0 |
| Ben-Gurion/new_trees/new_trees_01 | 33.6 | 1.67 | **20.1x** | 2.46 | **163.8x** | - | - | - | ✅ | 0.0 |
| Ben-Gurion/new_trees/new_trees_02 | 33.9 | 1.73 | **19.6x** | 2.42 | **167.9x** | - | - | - | ✅ | 0.0 |
| Ben-Gurion/new_trees/new_trees_03 | 32.9 | 1.70 | **19.4x** | 2.48 | **159.5x** | - | - | - | ✅ | 0.0 |
| Ben-Gurion/new_trees/new_trees_04 | 34.4 | 1.67 | **20.6x** | 2.60 | **158.9x** | - | - | - | ✅ | 0.0 |
| Ben-Gurion/new_trees/new_trees_05 | 34.2 | 1.80 | **19.0x** | 2.56 | **160.5x** | - | - | - | ✅ | 0.0 |
| Ben-Gurion/new_trees/new_trees_06 | 35.8 | 1.75 | **20.5x** | 2.53 | **170.1x** | - | - | - | ✅ | 0.0 |
| Ben-Gurion/new_trees/new_trees_07 | 35.5 | 1.92 | **18.5x** | 2.73 | **155.9x** | - | - | - | ✅ | 0.0 |
| Ben-Gurion/new_trees/new_trees_08 | 34.5 | 1.72 | **20.0x** | 2.63 | **157.2x** | - | - | - | ✅ | 0.0 |
| Ben-Gurion/new_trees/new_trees_09 | 34.7 | 1.83 | **18.9x** | 2.58 | **161.0x** | - | - | - | ✅ | 0.0 |
| Ben-Gurion/new_trees/new_trees_10 | 34.9 | 1.80 | **19.4x** | 2.55 | **164.4x** | - | - | - | ✅ | 0.0 |
| Ness-Tziona/exploded/nes_tziona_unblock_2 | 931.0 | 6.36 | **146.4x** | 10.12 | **1104.5x** | - | - | - | ✅ | 0.0 |
