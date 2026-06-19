# Innovation District GIS Extraction Design Record

Date: 2026-06-16

## Purpose

Capture the current design discussion for getting Innovation District microclimate results from the Rhino/GLB/fast-utci workflow onto a GIS map in `innovation_dashboard`.

This is not the implementation plan. It records the coordinate, shape, output-format, and dashboard decisions that the implementation should preserve.

## Core Decision

The GIS product must be shape-aware.

The rectangular canonical grid used by `fast-utci` is an internal compute scaffold. It is useful for indexing and GPU layout, but it is not a valid GIS surface by itself. Cells outside the actual modeled/sampled surface must not appear as UTCI or shading results on the dashboard map.

The export contract should therefore distinguish:

- canonical grid: rectangular internal indexing domain
- active sampled cells/points: values that correspond to the actual simulation surface
- no-data/outside cells: excluded from display and statistics
- source shape: the model/sampling surface that defines where values are meaningful

## Validated Rhino Findings

The first Rhino extraction pass against `D:\Projects\Nur\Shade\Innovation_District.3dm` produced:

- model units: meters
- declared/provisional CRS: EPSG:2039
- model bounds: approximately `x=180591..183214`, `y=573585..575928`
- object count: 11,973
- layer inventory:
  - `ground`: sampling inventory, 176 objects
  - `street`: sampling inventory, 11 objects
  - `train_tracks`: sampling hint only, 2 objects
  - `existing_buildings`: occluder, 840 objects
  - `trees_canopy`: occluder, 8,329 objects
  - `trees_point`: ignored/context, 2,614 objects
  - `district_outline`: ignored/context, 1 object

The `district_outline` Rhino polyline was transformed from EPSG:2039 to WGS84 and visually checked on an OpenStreetMap/Leaflet map. It lines up with the intended Innovation District area, so the projected model coordinates are the current geospatial source of truth.

Rhino's `EarthAnchorPoint` is not useful for placement in this file: it is exposed through `rhino3dm`, but reports latitude/longitude `0,0`. Treat it as unset or placeholder metadata.

## Rhino And GLB Georeferencing

`Extract with origin` is not enough as the georeferencing contract.

It may preserve a useful local origin for geometry export, but GLB should be treated as visual geometry and scene transforms, not as authoritative CRS metadata. The Rhino `.3dm` should be inspected directly for geospatial metadata.

The extractor should try to capture:

- model units
- document tolerances
- EarthAnchorPoint data, if present and exposed by the available API
- model base point
- model north/east vectors
- latitude, longitude, and elevation, if present
- layer/object inventory
- layer/object bounding boxes
- warnings for missing EarthAnchorPoint, unknown CRS, suspiciously large coordinates, or missing shape layers

If the model coordinates are already in projected meters, the CRS must still be explicit. The current Innovation District bounds look like Israeli projected coordinates, likely EPSG:2039, but that must be verified from Rhino/GIS provenance before becoming a declared contract.

Normal Python `rhino3dm` may not expose every Rhino document georeferencing field. If it cannot read EarthAnchorPoint, the report should distinguish API-unavailable metadata from metadata that is definitely unset in the file. A Rhino/RhinoCommon script may be needed for the final authoritative extraction.

## Shape-Aware Surface Contract

The model shape used for GIS export should come from the same semantic source as the simulation sampling surface.

For the current live WebGPU route, relevant layer roles already exist in the metadata generator and runtime, but they are not all the same contract:

- metadata/bounds sampling layers: likely `ground`, `street`, `train_tracks`, or equivalent project-specific names
- active result mask source: the runtime policy that decides which sampled cells are valid UTCI/shading result cells
- occluder layers: buildings, trees, canopies, and other shade-blocking geometry
- ignored/context layers: outlines, point markers, and non-surface context

The shape-aware export should not export every canonical cell in the rectangular bounds. It should export only cells/points that are active according to the study-area/sampling-surface mask.

Train tracks or other context layers must not become GIS UTCI cells merely because they participated in metadata bounds or visual context. They should become valid result surfaces only if the runtime active-mask policy explicitly includes them.

For the classified Innovation District export contract, active rows remain sampled-surface rows only. Sampled-surface membership comes from `ground` plus street-family geometry when present (`street`, `streets`, `road`, `roads`, `sidewalk`, `sidewalks`). Building footprints are an overlapping classification/exclusion overlay only; they must never create active rows on their own, and building-only cells must stay outside the exported active set.

Triangle inclusion must be defined once using the shared sample-center point-in-triangle rule and reused for both sampled-surface rasterization and building-footprint overlays so boundary behavior cannot drift across phases.

Durable classification contract for classified active exports:

- the classified export is sampled-surface-only
- building footprints are overlay/exclusion flags, not active-row creators
- street/road/sidewalk-family classification depends on matching sampled geometry
- `unknown` is not legal for classified active export rows
- shared sample-center boundary semantics apply to both sampled-surface and building-footprint rasterization

For GIS display this can become downstream derivations:

- point samples for debug and validation
- cell polygons for small samples or simplified layers
- raster no-data masks for downstream surface tiles
- contours or classified polygons as derived vector products

The no-data mask is part of correctness, not a visualization detail.

## UTCI And Shading Export

Innovation District is currently a live WebGPU analysis, not a precomputed `.bin` analysis.

The export path should therefore reuse the live WebGPU session rather than assuming a static binary already exists.

For the default discussion target:

- date: August 15
- UTCI: all 24 hourly values
- Shading Index: August/day-level or month-index output, depending on the current runtime contract
- values: one `f32` per active sampled point/cell

The future export table should include enough identity to join values back to both compute and GIS:

- canonical index
- active index, if compacted
- projected x/y/z or local x/y/z plus transform
- WGS84 lon/lat for GIS/debug
- UTCI values for hours 00-23
- shading index value
- no-data/valid flag
- source layer or surface classification when available

If surface classification is present in the classified export, it should be derived from raw multi-hot flags rather than a raw primary enum/class array. The durable contract is:

- every classified active row has at least one sampled-surface flag (`ground` or `street_surface`)
- raw classification is `surfaceFlags` bitflags only; there is no raw primary enum/class array
- `surface_class = unknown` is not legal for classified active export rows
- `surface_class` priority is display convenience only: `building_footprint > street_surface > ground`
- street/building overlap is preserved in flags/booleans, while public-realm stats exclude building-footprint rows
- `include_in_outdoor_surface_stats` means active sampled rows excluding building footprints
- `include_in_public_realm_stats` means street/road/sidewalk-family rows excluding building footprints

The export must keep active and inactive semantics aligned with the viewer: inactive cells should be no-data, absent from tooltips, absent from statistics, and absent from dashboard histograms.

## Output Formats And Ownership

Different artifacts serve different jobs.

### Debug And QA

This repo should use GeoJSON only for small samples, outlines, control points, and validation layers.

Full-resolution GeoJSON is not appropriate for the complete 2 m Innovation District surface. It would be heavy and slow, especially if each cell becomes a polygon with 24 UTCI values.

### Final Reusable Handoff From This Repo

The final reusable bundle produced by `fast-utci` should be:

- `cells.geoparquet`
- `manifest.json`
- `qa/debug-sample.geojson`

`cells.geoparquet` is the authoritative reusable handoff in this repo. It should contain one row per active cell, WGS84 point geometry for GIS/debug, projected source coordinates for traceability, hourly UTCI columns, and Shading Index values. `manifest.json` carries provenance, counts, timings, checksums, and the downstream boundary note. `qa/debug-sample.geojson` is a capped local validation aid only.

This repo should not make PMTiles, raster tiles, vector tiles, `geometry.parquet`, `values.parquet`, or required summary JSON part of the default contract.

## Dashboard Integration Handoff

`innovation_dashboard` already has MapLibre, Chart.js, D3, and deep-dive overlay patterns.

This repo should produce the reusable handoff bundle and a handoff note that another agent can consume in `D:\Projects\Nur\innovation_dashboard`.

The microclimate dashboard can later become a dedicated deep dive under:

`src/executive-overview/deep-dives/microclimate/`

Expected UI data:

- map overlay mode: UTCI or shading
- hour selector for UTCI
- legend aligned with UTCI/shading classes
- histogram of UTCI distribution for the selected hour
- comfort-class breakdown
- percent shaded or exposed
- summary metrics such as mean UTCI, p90/p95 UTCI, min/max, and hot-area share

Statistics must be computed from valid active cells only.

## Completed First Slice

The first script extracted Rhino georeferencing and model inventory so we could see what the `.3dm` actually contains.

It did not attempt to compute UTCI or build dashboard tiles.

Deliverable:

- `scripts/extract_rhino_georef.py`
- output JSON sidecar, for example:
  - `data/3d_models/Innovation-District/innovation_district.georef.json`

The script should be runnable from normal Python with `rhino3dm` installed. If `rhino3dm` is missing, it should fail with a clear install message. If Rhino-specific metadata is unavailable through `rhino3dm`, the output should report that explicitly rather than guessing.

If normal Python cannot expose the required EarthAnchorPoint fields, the next slice should add a Rhino/RhinoCommon script path rather than weakening the georeferencing contract.

## Current Implementation Slice

The next implementation slice should export shape-aware UTCI and Shading Index artifacts for Innovation District.

It should:

- reuse the existing live WebGPU selected-hour path
- use the runtime active-mask contract as the source of valid GIS result cells
- write full-resolution raw active-cell data as binary arrays plus metadata JSON
- let Python post-processing own EPSG:2039 to WGS84 conversion, `cells.geoparquet`, `manifest.json`, and sampled debug GeoJSON
- produce a handoff document for the `innovation_dashboard` agent

It should not:

- implement dashboard UI in this repo
- export inactive rectangular canonical cells as valid result cells
- produce full-resolution GeoJSON as the primary artifact
- make PMTiles, raster tiles, vector tiles, or required chart-summary JSON part of this repo's default output contract
- make default all-hours GeoJSON or 24 hourly GeoJSON files part of the default output contract

## Validation Expectations

Before trusting any UTCI-to-GIS export, validate:

- the Rhino model units
- the CRS or source projection
- EarthAnchorPoint presence and values
- layer inventory matches expected sampling/occluder layers
- model bounds align with the current GLB/analysis metadata
- transformed bounds/control points land on the Innovation District site
- shape-aware active area excludes the rectangular outside region

Only after those checks should the live WebGPU UTCI/shading export be wired to GIS artifacts.

## Downstream Boundary

`fast-utci` produces the source-of-truth GIS handoff for this flow: the exported artifact set centered on `cells.geoparquet`, plus `manifest.json` and `qa/debug-sample.geojson`. Downstream repositories and consumers, including `innovation_dashboard`, derive PMTiles, map payloads, chart JSON, and other display-oriented products from `cells.geoparquet` outside this repo.

Georeferencing validation and authoritative source-layer identification remain implementation tasks for this export flow, but they do not change the artifact boundary or move downstream tiling/chart derivation into `fast-utci`.
