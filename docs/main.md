# Product Requirements Document (PRD)
## fast-utci: Universal Thermal Climate Index Calculator

### 1. Executive Summary

**fast-utci** is a Python-based computational tool designed to rapidly generate 2D Universal Thermal Climate Index (UTCI) maps from 3D architectural models. The tool integrates multiple specialized libraries to perform comprehensive thermal comfort analysis by combining solar radiation modeling, raytracing simulations, and meteorological data processing.

### 2. Problem Statement

Current UTCI calculation methods for architectural applications are either:
- Too slow for real-time or near-real-time applications
- Require manual setup and complex software configurations
- Lack integration between 3D modeling and thermal comfort analysis
- Don't provide rapid visualization capabilities for design iterations

### 3. Solution Overview

fast-utci provides an automated pipeline that:
- Processes 3D models (GLB/GLTF formats) using trimesh
- Integrates weather data from EPW files via Ladybug Tools
- Performs raytracing simulations using Radiance for Mean Radiant Temperature (MRT) calculations
- Computes UTCI values using pythermalcomfort
- Generates visualizations and exports results in multiple formats

### 4. Technical Architecture

#### Core Dependencies
- **Python 3.8+**
- **trimesh**: 3D model processing and mesh manipulation
- **pythermalcomfort**: UTCI calculations
- **Ladybug Tools**: Weather data processing and solar calculations
- **Radiance**: Raytracing engine for solar radiation analysis
- **honeybee-radiance**: Python wrapper for Radiance (optional)

#### Input Data Specifications

##### 3D Models
- **Supported formats**: GLB, GLTF
- **Requirements**: 
  - Valid mesh geometry
  - Proper coordinate system (preferably georeferenced)
  - Material properties for surface reflectance analysis

##### Weather Data
- **Primary format**: EPW (EnergyPlus Weather files)
- **Additional formats**: CLM, WEA, DDY files for comprehensive analysis
- **Required parameters**:
  - Air Temperature (hourly)
  - Wind Speed (hourly) 
  - Relative Humidity (hourly)
  - Solar radiation data
  - Geographic location and timezone

#### Current Data Structure
```
data/
├── ISR_D_Beer.Sheva.401900_TMYx/
│   ├── ISR_D_Beer.Sheva.401900_TMYx.epw    # Primary weather data
│   ├── ISR_D_Beer.Sheva.401900_TMYx.clm    # Climate file
│   ├── ISR_D_Beer.Sheva.401900_TMYx.ddy    # Design day data
│   ├── ISR_D_Beer.Sheva.401900_TMYx.pvsyst # PVsyst format
│   ├── ISR_D_Beer.Sheva.401900_TMYx.rain   # Rainfall data
│   ├── ISR_D_Beer.Sheva.401900_TMYx.stat   # Weather statistics
│   └── ISR_D_Beer.Sheva.401900_TMYx.wea    # Weather file
├── rec_model_no_curve.glb                   # 3D model (GLB format)
└── rec_model_no_curve.gltf                  # 3D model (GLTF format)
```

### 5. Core Modules Specification

#### 5.1 reader.py
**Purpose**: Data ingestion and preprocessing
**Responsibilities**:
- Parse GLB/GLTF 3D models using trimesh
- Extract mesh geometry, materials, and metadata
- Read and validate EPW weather files using ladybug-core
- Convert weather data to required formats
- Validate coordinate systems and units
- Handle multiple weather file formats (CLM, WEA, DDY)

**Input**: 3D model files, weather files
**Output**: Structured data objects for downstream processing

#### 5.2 mrt_calculator.py
**Purpose**: Mean Radiant Temperature computation
**Responsibilities**:
- Set up Radiance simulation environment
- Generate scene files from 3D models
- Configure raytracing parameters
- Execute solar radiation calculations
- Compute MRT at specified points/mesh edges
- Handle multiple sun positions (time-series analysis)
- Optimize raytracing for performance

**Dependencies**: Radiance, honeybee-radiance
**Input**: 3D mesh data, solar angles, weather parameters
**Output**: MRT values for analysis points

#### 5.3 utci_calculator.py
**Purpose**: UTCI computation and thermal comfort analysis
**Responsibilities**:
- Integrate MRT, air temperature, wind speed, and humidity
- Compute UTCI using pythermalcomfort or ladybug-comfort
- Handle time-series calculations
- Apply thermal comfort categories
- Perform statistical analysis (percentiles, extremes)
- Validate input parameter ranges

**Input**: MRT, air temp, wind speed, relative humidity
**Output**: UTCI values and comfort classifications

#### 5.4 output.py
**Purpose**: Results export and visualization
**Responsibilities**:
- Export results to CSV format
- Generate 2D heat maps (PNG)
- Create interactive visualizations (HTML)
- Produce summary statistics and reports
- Handle different output resolutions
- Support batch processing outputs

**Output formats**:
- CSV: Point-wise UTCI data
- PNG: 2D thermal maps
- HTML: Interactive visualizations

#### 5.5 main.py
**Purpose**: Orchestration and automation
**Responsibilities**:
- Coordinate workflow between modules
- Handle command-line interface
- Manage configuration parameters
- Support batch processing
- Integrate with external APIs (weather data, 3D models)
- Provide logging and error handling
- Enable production automation

### 6. Output Specifications

#### 6.1 Data Outputs
- **CSV files**: Point-wise UTCI data with coordinates, timestamps, and comfort categories
- **Statistical summaries**: Min/max/average UTCI values, comfort hour analysis
- **Time-series data**: Hourly/daily/monthly UTCI variations

#### 6.2 Visualization Outputs
- **2D thermal maps**: PNG images showing UTCI distribution
- **Interactive plots**: HTML files with zoomable, time-animated visualizations
- **Comfort charts**: Bar charts and heat maps for comfort categories

#### 6.3 Performance Targets
- **Processing speed**: Complete analysis for 1000-point mesh in <5 minutes
- **Accuracy**: UTCI calculations within 0.1°C of reference implementations
- **Memory efficiency**: Handle models up to 1M triangles
- **Scalability**: Support batch processing of multiple scenarios


### 9. Success Criteria

#### Functional Requirements
- Successfully process GLB/GLTF 3D models
- Accurately compute UTCI from EPW weather data
- Generate 2D thermal comfort maps
- Export results in multiple formats

#### Performance Requirements
- Process standard architectural models (<100K triangles) in <2 minutes
- Handle weather data for full year (8760 hours) without memory issues
- Generate visualizations in <30 seconds
- Maintain UTCI calculation accuracy within 0.1°C

#### Usability Requirements
- Simple command-line interface for basic operations
- Clear documentation and examples
- Error handling with meaningful messages
- Support for common architectural modeling workflows