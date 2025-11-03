# MRT Calculator Module

## What This Does

Computes Mean Radiant Temperature (MRT) for outdoor thermal comfort analysis. This module replicates Grasshopper's OutdoorSolarMRT component with optimized performance.

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
| `config.py` | Centralized parameters and configuration |

### Utility Modules

| Module | Purpose |
|--------|---------|
| `exceptions.py` | Custom exception hierarchy for better error handling |
| `cache.py` | Thread-safe cache management for expensive computations |
| `performance.py` | Performance optimization utilities (batch sizing, memory) |
| `adapters.py` | Ray intersector strategies (weather adapters moved to `fast_utci.shared.weather`) |

## Quick Start

```python
from fast_utci.mrt import MRTCalculator, create_validation_period_filter, create_rectangular_grid
from fast_utci.shared.io import read_project_data, get_combined_mesh, get_ground_bounds

# Load model and get combined mesh
scene, _, _ = read_project_data('buildings.glb', 'weather.epw')
model = get_combined_mesh(scene)

# Setup MRT calculator
calc = MRTCalculator(context_meshes=[model])
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

### Basic Parameters

Key parameters in `config.py`:
- `DEFAULT_HUMAN_HEIGHT`: 1.8m (person height)
- `DEFAULT_GRID_SIZE`: 10.0m (analysis spacing)
- `DEFAULT_BATCH_SIZE`: 10000 (ray processing batch)
- `DEFAULT_N_WORKERS`: Auto (parallel processing)

### Environment Variables

Control performance optimizations via environment variables:

```python
# Ray intersection backend
FAST_UTCI_INTERSECTOR=auto|embree|trimesh

# Embree-specific settings
FAST_UTCI_EMBREE_QUALITY=auto|low|medium|high
FAST_UTCI_EMBREE_BUILD_BVH=true|false
FAST_UTCI_EMBREE_PACKET_SIZE=0|4|8|16
FAST_UTCI_INTERSECTS_ANY=true|false

# Performance optimizations
FAST_UTCI_VECTORIZED_SOLAR=true|false
FAST_UTCI_BATCH_POSITIONS=true|false
```

All environment variables are centralized in `EnvironmentConfig` for type-safe access:

```python
from fast_utci.mrt import get_env_config

env = get_env_config()
print(f"Using intersector: {env.intersector}")
print(f"Vectorized solar: {env.vectorized_solar}")
```

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
