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

# Install core dependencies
pip install -e .

# Or install with optional dependencies:
pip install -e .[dev]      # Development tools (testing, linting)
pip install -e .[gpu]       # GPU acceleration
pip install -e .[profile]   # Performance profiling
pip install -e .[all]       # Everything
```

## Quick Start

### Running Example Scripts

```bash
# Quick automated workflow with predefined parameters (full day)
python quick_analysis.py

# Interactive workflow with date selection (full day analysis)
python run_analysis.py
```

**Note**: Both scripts perform full-day (24-hour) UTCI analysis. `quick_analysis.py` is a lightweight wrapper that calls `run_analysis_core()` with predefined configurations. Modify `ANALYSIS_CONFIGS` in `quick_analysis.py` to batch-run multiple analyses.

**Output**:
- Binary (`.bin`) + metadata (`.json`) for web viewer
- CSV export (optional, set `export_csv=True` in config)

### View Results

```bash
python -m http.server 8000
# Open http://localhost:8000/viewer/
```

### Using the API

For programmatic access, see [`fast_utci/README.md`](fast_utci/README.md) for complete API examples.


## Project Structure

```
fast-utci/
├── fast_utci/              # Main package
│   ├── mrt/                # MRT calculation (ray tracing, solar)
│   ├── utci/               # UTCI calculation (thermal comfort)
│   ├── shared/             # Shared utilities (parallel, weather, config)
│   ├── model_reader.py     # 3D model & EPW file loading
│   └── ...
├── viewer/                 # Web-based 3D viewer (three.js)
├── data/                   # Models, weather, analyses, validation
├── scripts/                # Utility scripts
├── tests/                  # Test suite with validation
├── quick_analysis.py       # Quick automated workflow
└── run_analysis.py         # Interactive analysis workflow
```

See module READMEs for detailed structure.

## Documentation

### API Documentation

For detailed API usage and programmatic access:
- **[`fast_utci/README.md`](fast_utci/README.md)** - Complete API guide with code examples
- **[`fast_utci/mrt/README.md`](fast_utci/mrt/README.md)** - MRT calculation details
- **[`fast_utci/utci/README.md`](fast_utci/utci/README.md)** - UTCI calculation details
- **[`fast_utci/shared/README.md`](fast_utci/shared/README.md)** - Shared utilities

### Module Overview

- **`fast_utci.mrt`** - MRT calculation with ray tracing and parallel processing
- **`fast_utci.utci`** - UTCI calculation using pythermalcomfort
- **`fast_utci.shared`** - Parallel processing, weather data, and configuration
- **`fast_utci.model_reader`** - 3D model (GLTF/GLB) and EPW file loading
- **`fast_utci.viewer`** - 3D visualization with UTCI heatmaps (three.js)

## Requirements

- Python 3.11+
- See `pyproject.toml` for full dependency list
- **Recommended**: Install `pyembree` for 10-100x speedup in ray tracing
