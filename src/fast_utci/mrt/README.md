# MRT Calculator Module

This module is part of the legacy Python/Ladybug CPU pipeline. It provides exposure, Mean Radiant Temperature (MRT), and Shading Availability Index (SAI) tools for reference calculations, parity checks, and legacy exports.

This code was an intermediate performance step between Grasshopper/Ladybug and the current WebGPU viewer: it keeps Ladybug's sun/sky/SolarCal semantics, but replaces manual Grasshopper workflows with scriptable Embree-accelerated, parallel CPU execution. New high-throughput analysis should use the WebGPU/Three.js route when possible.

## What This Does

Computes Mean Radiant Temperature (MRT) for the legacy CPU reference path. It uses Ladybug sun/sky helpers and `ladybug-comfort` SolarCal to replicate Grasshopper's OutdoorSolarMRT component, with Embree/BVH-backed CPU ray tracing and parallel execution as the historical performance improvement. For current production analysis, prefer the WebGPU/Three.js viewer path.

## How It Works

1. **Create analysis grid** → Sample points on surfaces
2. **Calculate sun positions** → Solar vectors for each hour
3. **Ray-trace exposure** → Test if sun/sky is visible from each point
4. **Compute MRT** → Use SolarCal with weather data + exposure

## Key Modules

All modules are located in `fast_utci/mrt/`:

### Core Modules

| Module | Purpose |
|--------|---------|
| `mrt_calculator.py` | Main orchestrator - coordinates everything |
| `grid.py` | Generate analysis points from surfaces |
| `solar.py` | Calculate sun positions and vectors |
| `exposure.py` | Ray-trace to find sun/sky visibility |
| `solarcal.py` | Compute MRT using Ladybug SolarCal |
| `mesh.py` | Handle context geometry (buildings, trees) |
| `period.py` | Filter data to specific time periods |
| shared `config.py` | TOML-backed parameters and configuration |

### Utility Modules

| Module | Purpose |
|--------|---------|
| `exceptions.py` | Custom exception hierarchy for better error handling |
| `cache.py` | Thread-safe cache management for expensive computations |
| `performance.py` | Performance optimization utilities (batch sizing, memory) |
| `adapters.py` | Ray intersector strategies (weather adapters moved to `fast_utci.shared.weather`) |
| `shading_index.py` | SAI calculation (proportion of sunlight hours fully shaded) |

## Quick Start

```python
from fast_utci.mrt import MRTCalculator, create_validation_period_filter, create_rectangular_grid
from fast_utci.shared import load_config
from fast_utci.shared.io import read_project_data, get_combined_mesh, get_ground_bounds

cfg = load_config()

# Load model and get combined mesh
scene, _, _ = read_project_data('buildings.glb', 'weather.epw')
model = get_combined_mesh(scene)

# Setup MRT calculator
calc = MRTCalculator(context_meshes=[model], config=cfg.mrt)
calc.set_location_from_epw('weather.epw')

# Create analysis grid
model_bounds = get_ground_bounds(scene)
grid = create_rectangular_grid(
    bounds_min=model_bounds[0][:2],  # X, Y minimum
    bounds_max=model_bounds[1][:2],  # X, Y maximum
    grid_size=10.0,
    z_height=1.5
)

# Get validation period (Aug 15, hour 13)
period, hours = create_validation_period_filter()

# Compute exposure and MRT
from ladybug.epw import EPW
epw = EPW('weather.epw')
exposure_results = calc.compute_exposure(grid.points, period, hours)
mrt_results = calc.compute_mrt(epw, exposure_results, period, hours)

# Calculate SAI (optional; implementation module is still named shading_index)
from fast_utci.mrt.shading_index import calculate_shading_index
sun_data = calc.get_sun_data(period, hours)
shading_indices = calculate_shading_index(exposure_results, sun_data)

# Export results (via MRT calculator)
calc.to_csv(mrt_results, 'mrt_results.csv')
```

## Key Concepts

### Ray Tracing Algorithm
- Creates vertical sample points representing a person
- Tests if sun/sky is visible from each point using ray-mesh intersections
- Uses BVH acceleration for fast geometry testing
- Returns exposure fractions (0-1) for each timestep

### SolarCal Integration
- Uses Ladybug's OutdoorSolarCal for MRT computation
- Inputs: weather data + exposure fractions
- Outputs: MRT values + component breakdowns

### Performance Features
- Parallel processing for large grids
- BVH acceleration for ray intersections
- Solar data caching
- Batch processing for memory efficiency

## Configuration

Configuration is loaded from the repo-level `fast_utci.toml` through `fast_utci.shared.load_config()`:

```python
from fast_utci.mrt import MRTCalculator
from fast_utci.shared import load_config

cfg = load_config()
calc = MRTCalculator(context_meshes=[model], config=cfg.mrt)
```

Common settings live under `[mrt]`, `[performance]`, `[engine]`, and `[parallel]`. Prefer the TOML config path in examples and scripts.

## Validation

Validates against Grasshopper OutdoorSolarMRT:
- **Test case**: August 15th, hour 13, Beer Sheva, Israel
- **Context**: Building geometry for shading
- **Output**: CSV format compatible with GH validation

## Advanced Features

### Custom Weather Data Adapters

`fast_utci.shared.weather` for use across both MRT and UTCI modules:

```python
from fast_utci.shared import create_weather_adapter, EPWAdapter, DataFrameAdapter

# Automatically detects EPW or DataFrame
adapter = create_weather_adapter(weather_data)

# Or use specific adapters
epw_adapter = EPWAdapter(epw_object)
df_adapter = DataFrameAdapter(weather_df)

# All adapters provide consistent interface
temperature = adapter.get_temperature()
radiation = adapter.get_direct_radiation()
```
### Performance Optimization

Fine-tune performance for your hardware:

```python
from fast_utci.mrt import PerformanceOptimizer

optimizer = PerformanceOptimizer(
    memory_fraction=0.3,  # Use 30% of available RAM
    min_solar_batch=100,
    min_sky_batch=500
)

batch_size = optimizer.calculate_batch_size(n_rays=10000, ray_type="solar")
```

### Cache Management

Control global caches for testing or memory management:

```python
from fast_utci.mrt import CacheManager

cache = CacheManager.get_instance()

# Get cached sky vectors (computed once, reused everywhere)
sky_vectors, sky_weights = cache.get_sky_vectors()

# Clear cache when needed (e.g., testing)
cache.clear()
```

### Custom Intersector Strategies

Extend with custom ray intersection backends:

```python
from fast_utci.mrt.adapters import RayIntersectorStrategy

class CustomIntersectorStrategy(RayIntersectorStrategy):
    def initialize(self) -> bool:
        # Your custom initialization
        return True
```

### Parallel Processing Utilities
`fast_utci.shared.parallel_utils` for use across both MRT and UTCI modules:

```python
from fast_utci.shared import ParallelProcessor, SpatialChunkStrategy

processor = ParallelProcessor(n_workers=8, show_progress=True)

def process_chunk(chunk_data):
    # Your processing logic
    return results

results = processor.process_chunks(
    data=positions,
    worker_fn=process_chunk,
    chunk_strategy=SpatialChunkStrategy(),
    description="Processing positions"
)
```

## Dependencies

- ladybug-core (solar calculations)
- ladybug-comfort (SolarCal)
- trimesh (3D geometry + BVH)
- numpy, pandas (data processing)
- psutil (memory optimization, optional)
