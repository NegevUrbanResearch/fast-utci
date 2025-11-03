# fast-utci

**Rapidly compute 2D UTCI maps from 3D models**

This package calculates Universal Thermal Climate Index (UTCI) for outdoor thermal comfort analysis by combining physical 3D geometry with weather data.

## What is UTCI?

UTCI (Universal Thermal Climate Index) represents how hot or cold it *feels* to a human outdoors, accounting for:
- **Air temperature** - ambient temperature
- **Wind speed** - convective cooling
- **Humidity** - evaporative cooling
- **Mean Radiant Temperature (MRT)** - radiation from sun, sky, and surfaces

UTCI values range from extreme cold (< -40°C) to extreme heat (> 46°C), with comfortable conditions between 9-26°C.

## How It Works

```
3D Model + Weather → MRT (ray tracing) → UTCI (thermal comfort)
```

### Two-Stage Process

**Stage 1: MRT Calculation** (`fast_utci.mrt`)
1. Load 3D context geometry (buildings, trees, terrain)
2. Create analysis grid of sample points
3. Ray-trace sun/sky visibility for each point and hour
4. Compute Mean Radiant Temperature using SolarCal

**Stage 2: UTCI Calculation** (`fast_utci.utci`)
1. Load weather data (air temp, wind, humidity)
2. Combine MRT results with weather using boundary averaging
3. Calculate UTCI thermal comfort index
4. Export results for analysis/visualization

## Quick Start

```python
from fast_utci import MRTCalculator, UTCICalculator
from fast_utci.mrt import create_rectangular_grid, create_analysis_period
from fast_utci.shared.io import (
    read_project_data, 
    get_combined_mesh, 
    get_ground_bounds
)
from ladybug.epw import EPW
import numpy as np

# 1. Setup MRT calculator with context geometry
# Load model and get combined mesh
scene, _, _ = read_project_data('buildings.glb', 'weather.epw')
model = get_combined_mesh(scene)

mrt_calc = MRTCalculator(context_meshes=[model])
mrt_calc.set_location_from_epw('weather.epw')

# 2. Create analysis grid
model_bounds = get_ground_bounds(scene)

grid = create_rectangular_grid(
    bounds_min=model_bounds[0][:2],  # X, Y minimum
    bounds_max=model_bounds[1][:2],  # X, Y maximum
    grid_size=2.0,
    z_height=1.5  # pedestrian height
)

# 3. Define analysis period (e.g., August 15, full day)
period = create_analysis_period(
    start_month=8, start_day=15, start_hour=0,
    end_month=8, end_day=15, end_hour=23
)

# 4. Compute MRT (ray tracing + solar calculations)
epw = EPW('weather.epw')
exposure_results = mrt_calc.compute_exposure(
    positions=grid.points,
    analysis_period=period
)
mrt_results = mrt_calc.compute_mrt(epw, exposure_results, period)

# 5. Compute UTCI (thermal comfort)
utci_calc = UTCICalculator(weather_data='weather.epw')
utci_results = utci_calc.compute_utci(mrt_results)

# 6. Export results
utci_calc.to_csv(utci_results, 'utci_results.csv')

# 7. Get summary statistics
summary = utci_calc.summary_statistics(utci_results)
print(f"UTCI range: {summary['utci_stats']['min']:.1f} to {summary['utci_stats']['max']:.1f}°C")
```

## Architecture

The package is organized into three modules:

```
fast_utci/
├── mrt/        # Mean Radiant Temperature calculations
│   ├── Ray tracing for sun/sky visibility
│   ├── SolarCal MRT computation
│   └── Parallel processing & BVH acceleration
│
├── utci/       # UTCI thermal comfort calculations
│   ├── Boundary averaging algorithm
│   ├── Weather data management
│   └── Thermal comfort classification
│
└── shared/     # Common utilities
    ├── Parallel processing
    ├── Weather data adapters
    └── Configuration management
```

See individual module READMEs for details:
- [`mrt/README.md`](mrt/README.md) - MRT calculation details
- [`utci/README.md`](utci/README.md) - UTCI calculation details  
- [`shared/README.md`](shared/README.md) - Shared utilities

## Key Features

### Fast Ray Tracing
- **BVH acceleration** for geometry intersection
- **Embree backend** support for 10-100x speedup
- **Parallel processing** for large grids
- **Batch optimization** for memory efficiency

### Accurate MRT Calculation
- **Tregenza sky dome** for diffuse sky radiation
- **SolarCal algorithm** (ladybug-comfort) for MRT
- **Boundary averaging** for temporal interpolation

### Misc
- **Progress tracking** with time estimates
- **Memory efficient** batch processing
- **CSV/JSON export** for analysis

## Boundary Averaging

For accurate temporal interpolation, UTCI uses boundary averaging:

For each hour N:
- Calculate UTCI₀ using MRT₀[N] + weather[N]
- Calculate UTCI₁ using MRT₁[N] + weather[N+1]
- Average: **UTCI = (UTCI₀ + UTCI₁) / 2**

This matches the temporal interpolation used in Grasshopper OutdoorSolarMRT.

## Performance

Performance scales with:
- Grid density (fewer points = faster)
- CPU cores (parallel processing)
- Ray intersector (Embree >> trimesh)

## Configuration

Primary configuration is the single TOML file at the repository root: `fast_utci.toml`.

Example:

```toml
[parallel]
n_workers = "auto"      # auto = CPU count - 1
show_progress = true
parallel_threshold = 50

[performance]
batch_size = 10000
ray_max_distance = 1000.0

[engine]
intersector = "auto"    # auto|embree|trimesh
embree_quality = "medium"
embree_build_bvh = true
embree_packet_size = 0
intersects_any = true

[features]
vectorized_solar = true
batch_positions = false
include_weather_in_results = true
include_datetime_in_results = true

[mrt]
human_height = 1.8
pt_count = 1
absorptivity = 0.7
emissivity = 0.95
ground_reflectance = 0.25
north_degrees = 0.0
csv_encoding = "utf-8"
csv_index = false

[utci]
enable_vectorized = true
csv_encoding = "utf-8"
csv_index = false
```

Programmatic overrides:

```python
from fast_utci.mrt import MRTConfig
from fast_utci.shared import ParallelConfig

config = MRTConfig(
    human_height=1.8,
    ground_reflectance=0.25,
    parallel=ParallelConfig(n_workers=8, show_progress=True)
)

mrt_calc = MRTCalculator(context_meshes=['model.glb'], config=config)
```

## Dependencies

**Core:**
- `numpy`, `pandas` - data processing
- `trimesh` - 3D geometry and BVH
- `ladybug-core` - solar calculations
- `ladybug-comfort` - SolarCal MRT
- `pythermalcomfort` - UTCI calculation

**Optional:**
- `pyembree` - fast ray tracing (highly recommended)
- `psutil` - memory optimization
- `tqdm` - progress bars

Install with:
```bash
pip install numpy pandas trimesh ladybug-core ladybug-comfort pythermalcomfort
pip install pyembree  # optional but recommended for speed
```

## Validation

The implementation is currently validated against Grasshopper's OutdoorSolarMRT component:
- MRT values match within ±0.1°C
- UTCI values match within ±0.05°C
- Exposure calculations match exactly

See `tests/` for validation test cases.