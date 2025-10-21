# fast-utci

**fast-utci** is a Python package designed to rapidly compute 2D Universal Thermal Climate Index (UTCI) maps from 3D models.

For the UTCI calculations, we use the UTCI calculator from [pythermalcomfort](https://github.com/center-for-the-built-environment/pythermalcomfort). UTCI calculations take as inputs the Mean Radiant Temp, Air Temp, Wind Speed, and Relative Humidity. We also use [Ladybug Tools](https://github.com/ladybug-tools) for other calculations such as retrieving the angle of the sun.

For calculating the Mean Radiant Temperature at a given point or mesh edge, we conduct raytracing to find where direct sunlight and reflected solar radiation will land depending on both the angle of the sun and the 3D objects modeled. We use Embree-accelerated ray tests via trimesh (pyembree) for occlusion and visibility.

### Web Viewer

View live thermal comfort analysis: **[3D UTCI Viewer](https://negevurbanresearch.github.io/fast-utci/)**

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

**Note**: Both scripts perform full-day (24-hour) UTCI analysis. `quick_analysis.py` is a lightweight wrapper that calls `run_analysis_core()` with predefined configurations. Modify `ANALYSIS_CONFIGS` in `quick_analysis.py` to batch-run multiple analyses with different parameters.

**Output Files**:
- Binary data (`.bin`) and metadata (`.json`) for web viewer (always generated)
- CSV export with detailed results (optional, set `export_csv=True` in config)

### View Results

Run locally:
```bash
python -m http.server 8000
# Open http://localhost:8000/viewer/
```


## Project Structure

```
fast-utci/
├── fast_utci/              # Main package
│   ├── mrt/                # MRT calculation modules
│   │   ├── mrt_calculator.py
│   │   ├── solar.py
│   │   ├── exposure.py
│   │   ├── config.py       # MRT-specific config
│   │   └── ...
│   ├── utci/               # UTCI calculation modules (NEW)
│   │   ├── calculator.py   # Main UTCICalculator
│   │   ├── calculation.py  # Core computation logic
│   │   ├── weather.py      # Weather data management
│   │   ├── statistics.py   # Thermal comfort analysis
│   │   ├── export.py       # Export functionality
│   │   └── config.py       # UTCI-specific config
│   ├── shared/             # Shared utilities (NEW)
│   │   ├── config.py       # Parallel & performance config
│   │   ├── parallel_utils.py  # Parallel processing
│   │   └── weather.py      # Weather loading & filtering (consolidated)
│   ├── model_reader.py     # 3D model and EPW reading
│   ├── viewer.py           # 3D visualization
│   └── colors.py           # UTCI color scales
├── scripts/                # Utility scripts
│   ├── export_for_viewer.py
│   ├── generate_manifest.py
│   └── ...
├── viewer/                 # Web-based 3D viewer
│   ├── index.html          # Analysis selector
│   ├── viewer.html         # Main viewer
│   └── js/                 # Viewer components
├── data/                   # Data files
│   ├── 3d_models/          # GLTF/GLB files
│   ├── weather/            # EPW weather files
│   ├── analyses/           # Generated analysis results
│   └── validation/         # Validation data
├── docs/                   # Documentation
├── tests/                  # Test suite
│   ├── fixtures/           # Test data and baselines
│   └── test_*.py           # Integration tests
├── quick_analysis.py       # Quick config-based analysis wrapper
├── run_analysis.py         # Interactive full-day analysis script
└── pyproject.toml          # Package configuration
```

## Modules

### `fast_utci.mrt`
Core MRT calculation functionality with parallel processing support.
See `fast_utci/mrt/README.md` for detailed documentation.

### `fast_utci.utci`
UTCI calculation from MRT results and weather data using pythermalcomfort.
Modular architecture with clean separation of concerns.
See `fast_utci/utci/README.md` for detailed documentation.

### `fast_utci.shared`
Shared utilities for parallel processing, configuration, and weather data handling.
Consolidates common functionality used by both MRT and UTCI calculators.
See `fast_utci/shared/README.md` for detailed documentation.

### `fast_utci.model_reader`
Read and parse 3D models (GLTF/GLB) and EPW weather files.
Delegates weather loading to `fast_utci.shared.weather` for consistency.

### `fast_utci.viewer`
Enhanced 3D visualization with UTCI heatmaps using three.js


## Requirements

- Python 3.11
- See `pyproject.toml` for full dependency list
