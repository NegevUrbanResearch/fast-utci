# MRT Calculator Module

## What This Does

Computes Mean Radiant Temperature (MRT) for outdoor thermal comfort analysis. This module replicates Grasshopper's OutdoorSolarMRT component with optimized performance.

## How It Works

1. **Create analysis grid** → Sample points on surfaces
2. **Calculate sun positions** → Solar vectors for each hour
3. **Ray-trace exposure** → Test if sun/sky is visible from each point
4. **Compute MRT** → Use SolarCal with weather data + exposure

## Key Modules

| Module | Purpose |
|--------|---------|
| `mrt_calculator.py` | Main orchestrator - coordinates everything |
| `grid.py` | Generate analysis points from surfaces |
| `solar.py` | Calculate sun positions and vectors |
| `exposure.py` | Ray-trace to find sun/sky visibility |
| `solarcal.py` | Compute MRT using Ladybug SolarCal |
| `mesh.py` | Handle context geometry (buildings, trees) |
| `period.py` | Filter data to specific time periods |
| `config.py` | Centralized parameters |

## Quick Start

```python
from MRT import MRTCalculator, create_validation_period_filter
from MRT.grid import create_rectangular_grid

# Setup
calc = MRTCalculator(context_meshes=['buildings.obj'])
calc.set_location_from_epw('weather.epw')

# Create analysis grid
grid = create_rectangular_grid([0, 0], [100, 100], grid_size=10.0)

# Get validation period (Aug 15, hour 13)
period, hours = create_validation_period_filter()

# Compute exposure and MRT
exposure_results = calc.compute_exposure(grid.points, period, hours)
mrt_results = calc.compute_mrt(epw_data, exposure_results)

# Export results
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

Key parameters in `config.py`:
- `DEFAULT_HUMAN_HEIGHT`: 1.8m (person height)
- `DEFAULT_GRID_SIZE`: 10.0m (analysis spacing)
- `DEFAULT_BATCH_SIZE`: 10000 (ray processing batch)
- `DEFAULT_N_WORKERS`: Auto (parallel processing)

## Validation

Validates against Grasshopper OutdoorSolarMRT:
- **Test case**: August 15th, hour 13, Beer Sheva, Israel
- **Context**: Building geometry for shading
- **Output**: CSV format compatible with GH validation

## Dependencies

- ladybug-core (solar calculations)
- ladybug-comfort (SolarCal)
- trimesh (3D geometry + BVH)
- numpy, pandas (data processing)
