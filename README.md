# fast-utci

**fast-utci** is a Python tool designed to rapidly compute 2D Universal Thermal Climate Index (UTCI) maps from 3D models.

For the UTCI calculations, we use the UTCI calculator from [pythermalcomfort](https://github.com/center-for-the-built-environment/pythermalcomfort). UTCI calculations take as inputs the Mean Radiant Temp, Air Temp, Wind Speed, and Relative Humidity. We will also use [Ladybug Tools](https://github.com/ladybug-tools) for other calculations that come up such as retrieving the angle of the sun

For calulating the Mean Radiant Temperature at a given point or mesh edge, we need to conduct raytracing to find where direct sunlight and reflected solar radiation will land depending on both the angle of the sun and the 3d objects modeled. To do these calculations, we use Embree-accelerated ray tests via trimesh (pyembree) for occlusion and visibility, and may use [Radiance](https://www.radiance-online.org/download-install) where applicable.
## Performance toggles

You can enable additional performance features via environment variables (non-breaking defaults):

- FAST_UTCI_VECTORIZED_SOLAR=1 — enable vectorized solar exposure batching
- FAST_UTCI_INTERSECTOR=embree|trimesh — select ray intersector backend
- FAST_UTCI_EMBREE_QUALITY=low|medium|high — tune Embree quality (if supported)
- FAST_UTCI_EMBREE_BUILD_BVH=true|false — pre-build BVH (if supported)
- FAST_UTCI_EMBREE_PACKET_SIZE=0|4|8|16 — hint packet size (if supported)

Air Temp, Wind Speed, and Relative Humidity are all stored in the .epw weather files.

**Proposed project structure:**

`reader.py`: A script to read and parse the 3D model (glb or gltf) using trimesh library and EPW file inputs with ladybug-core.

`mrt_calculator.py`: A module that contains functions for setting up and running Radiance simulations to compute the MRT. It could use honeybee-radiance which requires Radiance simulation engine.

`utci_calculator.py`: A module that uses either pythermalcomfort or ladybug-comfort to run the final UTCI calculation for each set of inputs.

`output.py`: A script for writing the results to a file (CSV) and generating visualizations (PNG or HTML).

`main.py`: Script that orchestrates the entire process. It will handle calling APIs for pulling new weather data or 3D files and automating this pipeline in production.
