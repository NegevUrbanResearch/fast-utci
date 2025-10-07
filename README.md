# fast-utci

**fast-utci** is a Python package designed to rapidly compute 2D Universal Thermal Climate Index (UTCI) maps from 3D models.

For the UTCI calculations, we use the UTCI calculator from [pythermalcomfort](https://github.com/center-for-the-built-environment/pythermalcomfort). UTCI calculations take as inputs the Mean Radiant Temp, Air Temp, Wind Speed, and Relative Humidity. We also use [Ladybug Tools](https://github.com/ladybug-tools) for other calculations such as retrieving the angle of the sun.

For calculating the Mean Radiant Temperature at a given point or mesh edge, we conduct raytracing to find where direct sunlight and reflected solar radiation will land depending on both the angle of the sun and the 3D objects modeled. We use Embree-accelerated ray tests via trimesh (pyembree) for occlusion and visibility.

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
# Interactive demo workflow
python examples/demo_workflow.py

# Automated workflow with default settings
python examples/automated_workflow.py
```


## Project Structure

```
fast-utci/
├── fast_utci/              # Main package
│   ├── mrt/                # MRT calculation modules
│   │   ├── mrt_calculator.py
│   │   ├── solar.py
│   │   ├── exposure.py
│   │   └── ...
│   ├── utci_calculator.py  # UTCI calculations
│   ├── model_reader.py     # 3D model and EPW reading
│   ├── viewer.py           # 3D visualization
│   └── colors.py           # UTCI color scales
├── examples/               # Example scripts
│   ├── demo_workflow.py
│   └── automated_workflow.py
├── data/                   # Data files
│   ├── 3d_models/          # GLTF/GLB files
│   ├── weather/            # EPW weather files
│   └── validation/         # Validation data
├── tests/                  # Test suite (TBD)
├── docs/                   # Documentation
└── pyproject.toml          # Package configuration
```

## Modules

### `fast_utci.mrt`
Core MRT calculation functionality with parallel processing support.

### `fast_utci.utci_calculator`
UTCI calculation from MRT results and weather data using pythermalcomfort.

### `fast_utci.model_reader`
Read and parse 3D models (GLTF/GLB) and EPW weather files.

### `fast_utci.viewer`
Enhanced 3D visualization with UTCI heatmaps using Plotly.


## Requirements

- Python 3.11
- See `pyproject.toml` for full dependency list
