# fast-utci

**Rapidly compute 2D UTCI maps from 3D models**

This package calculates Universal Thermal Climate Index (UTCI) for outdoor thermal comfort analysis by combining physical 3D geometry with weather data.

**Live Demo:** [3D UTCI Viewer](https://negevurbanresearch.github.io/fast-utci/)

## What is UTCI?

UTCI (Universal Thermal Climate Index) represents how hot or cold it *feels* to a human outdoors, accounting for:
- **Air temperature** - ambient conditions
- **Wind speed** - convective cooling  
- **Humidity** - evaporative cooling
- **Mean Radiant Temperature (MRT)** - radiation from sun, sky, and surfaces

UTCI values range from extreme cold (< -40°C) to extreme heat (> 46°C), with comfortable conditions between 9-26°C.

## What is Shading Index?

Shading Index measures the proportion of sunlight hours during which each point is fully shaded (no direct solar radiation). This metric complements UTCI by providing a direct measure of shading availability, following Israeli shading metrics guidelines for urban planning.

**Shading Index values:**
- **0.0-0.5**: Poor Shading (always or mostly exposed)
- **0.5-0.7**: Acceptable Shading
- **0.7-0.9**: Good Shading
- **0.9-1.0**: Excellent Shading (always or mostly shaded)

The viewer allows toggling between UTCI and Shading Index visualizations to analyze both thermal comfort and shading patterns.

## How It Works

```
3D Model + Weather → MRT (ray tracing) → UTCI (thermal comfort)
```

**Two-stage process:**

1. **MRT Calculation**: Ray-trace sun/sky visibility from 3D geometry to compute Mean Radiant Temperature using [Ladybug Tools](https://github.com/ladybug-tools) SolarCal
2. **UTCI Calculation**: Combine MRT with weather data (air temp, wind, humidity) using [pythermalcomfort](https://github.com/center-for-the-built-environment/pythermalcomfort)

Uses **Embree-accelerated ray tracing** via trimesh (pyembree) for fast occlusion testing.

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd fast-utci

# Create/activate venv (Windows PowerShell)
..\.venv\Scripts\Activate.ps1

# Install package (editable)
pip install -e .

# Or install with optional dependencies:
pip install -e .[dev]      # Development tools (testing, linting)
pip install -e .[gpu]       # GPU acceleration
pip install -e .[profile]   # Performance profiling
pip install -e .[all]       # Everything
```

## Quick Start

### Configure

Edit `fast_utci.toml` at the repo root to set workers, performance, engine, and MRT/UTCI options.

### Running Example Scripts

```bash
# Quick automated workflow with predefined parameters (full day)
python quick_analysis.py

# Interactive workflow with date selection (full day analysis)
python run_analysis.py

# Batch process all 50 scenario variations (5 categories × 10 variants)
python batch_scenarios.py [--grid-size 10.0] [--month 8] [--day 15]
```

**Note**: 
- `quick_analysis.py` and `run_analysis.py` perform single analysis runs
- `batch_scenarios.py` processes all 50 scenario variations sequentially with consistent settings
- Modify `ANALYSIS_CONFIGS` in `quick_analysis.py` to batch-run multiple custom analyses

**Output**:
- Binary (`.bin`) + metadata (`.json`) for web viewer
  - Includes both UTCI and Shading Index data
  - Backward compatible with older analyses (Shading Index optional)
- CSV export (optional, set `export_csv=True` in config)

### Batch Processing

The `batch_scenarios.py` script processes all 50 scenario variations:

```bash
# Process all scenarios with default settings (10m grid, Aug 15)
python batch_scenarios.py

# Custom settings
python batch_scenarios.py --grid-size 5.0 --month 7 --day 20
```

**Output Structure:**
```
data/analyses/
├── existing_buildings/
│   ├── existing_buildings_01.bin
│   ├── existing_buildings_01.json
│   └── ... (10 variants)
├── existing_trees/
│   └── ... (10 variants)
├── new_high_buildings/
│   └── ... (10 variants)
├── new_low_buildings/
│   └── ... (10 variants)
├── new_trees/
│   └── ... (10 variants)
└── manifest.json
```

### View Results

```bash
python -m http.server 8000
# Open http://localhost:8000/viewer/
```

### Using the API

For programmatic access, see [`src/fast_utci/README.md`](src/fast_utci/README.md) for complete API examples.


## Project Structure

```
fast-utci/
├── src/fast_utci/          # Main package (src layout)
│   ├── mrt/                # MRT calculation (ray tracing, solar)
│   ├── utci/               # UTCI calculation (thermal comfort)
│   ├── shared/             # Shared utilities (parallel, weather, config)
│   │   └── io/             # 3D model & EPW file loading (GLB/GLTF, scene graph)
│   └── ...
├── viewer/                 # Web-based 3D viewer (three.js)
├── data/                   # Models, weather, analyses, validation
├── scripts/                # Utility scripts
├── quick_analysis.py       # Quick automated workflow
└── run_analysis.py         # Interactive analysis workflow
```

See module READMEs for detailed structure.

## Documentation

### API Documentation

For detailed API usage and programmatic access:
- **[`src/fast_utci/README.md`](src/fast_utci/README.md)** - Complete API guide with code examples
- **[`src/fast_utci/mrt/README.md`](src/fast_utci/mrt/README.md)** - MRT calculation details
- **[`src/fast_utci/utci/README.md`](src/fast_utci/utci/README.md)** - UTCI calculation details
- **[`src/fast_utci/shared/README.md`](src/fast_utci/shared/README.md)** - Shared utilities

### Module Overview

- **`fast_utci.mrt`** - MRT calculation with ray tracing and parallel processing
- **`fast_utci.utci`** - UTCI calculation using pythermalcomfort
- **`fast_utci.shared`** - Parallel processing, weather data, and configuration
- **`fast_utci.shared.io`** - 3D model (GLTF/GLB) and EPW file loading
- **`viewer/`** - Separate web-based 3D viewer (SvelteKit + Three.js) for visualizing results

## Requirements

- Python 3.11+
- See `pyproject.toml` for full dependency list
- **Recommended**: Install `pyembree` for 10-100x speedup in ray tracing
