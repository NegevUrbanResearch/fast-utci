# Innovation District GIS Handoff

This note is the handoff surface for downstream work that consumes the reusable Innovation District GIS export bundle produced by `fast-utci`.

## Source And Coordinate Notes

- Source Rhino model: `D:\Projects\Nur\Shade\Innovation_District.3dm`
- Generated georef sidecar: `data/3d_models/Innovation-District/innovation_district.georef.json`
- Generated raw export metadata: `data/gis/Innovation-District/2025-08-15_2m/raw/2025-08-15_2m_active-cells.metadata.json`
- Final reusable bundle root: `data/gis/Innovation-District/2025-08-15_2m/`

Coordinate contract:

- Raw exported `x`, `y`, `z` are projected analysis coordinates in `EPSG:2039`.
- Final `cells.geoparquet` point geometry is `EPSG:4326` WKB.
- `lon` and `lat` columns duplicate the `EPSG:4326` point coordinates for QA and downstream joins.

EarthAnchorPoint caveat:

- Rhino metadata extraction reported `earth_basepoint_latitude = 0.0` and `earth_basepoint_longitude = 0.0`.
- Do not use EarthAnchorPoint for map placement in this model.
- The projected model coordinates and the `district_outline` visual check are the current geospatial source of truth.

## Final Bundle Layout

Canonical reusable outputs:

- `data/gis/Innovation-District/2025-08-15_2m/cells.geoparquet`
- `data/gis/Innovation-District/2025-08-15_2m/manifest.json`
- `data/gis/Innovation-District/2025-08-15_2m/qa/debug-sample.geojson`

Retained raw collector inputs:

- `data/gis/Innovation-District/2025-08-15_2m/raw/2025-08-15_2m_active-cells.metadata.json`
- `data/gis/Innovation-District/2025-08-15_2m/raw/2025-08-15_2m_active-cells.canonical.u32.bin`
- `data/gis/Innovation-District/2025-08-15_2m/raw/2025-08-15_2m_active-cells.positions.f32.bin`
- `data/gis/Innovation-District/2025-08-15_2m/raw/2025-08-15_2m_active-cells.utci.f32.bin`
- `data/gis/Innovation-District/2025-08-15_2m/raw/2025-08-15_2m_active-cells.shading.f32.bin`
- `data/gis/Innovation-District/2025-08-15_2m/raw/2025-08-15_2m_active-cells.surface-flags.u8.bin`
- `data/gis/Innovation-District/2025-08-15_2m/raw/innovation-district-collector-run-log.json`

Raw folder caveat:

- `raw/` contains the current extraction inputs for this bundle.
- These are not stale leftovers; they are the source artifacts that produced the reusable GeoParquet bundle.
- Downstream consumers should treat `cells.geoparquet` as the canonical handoff surface and use `raw/` only for provenance or reprocessing.

There is also an `optional/` directory at the bundle root. It is empty in the current export and is not part of the default handoff contract.

## Raw Export Schema

Raw metadata contract from `2025-08-15_2m_active-cells.metadata.json`:

- `schemaVersion`: `innovation-district-raw-export/v1`
- `declaredCrs`: `EPSG:2039`
- `gridSize`: `2`
- `coordinateSystem`: `projected-analysis`
- `canonicalCount`: `1492551`
- `activeCount`: `638688`
- `hourCount`: `24`
- `hours`: `0..23`
- `activeMask.source`: `base+road`
- `activeMask.checksum`: `d4e1236d`
- `activeMask.signature`: `xy_ground:2000000:1299x1149:ecdbc359:d4e1236d`

Raw binary layouts:

- `canonicalIndices`: `u32`, shape `[638688]`, layout `point-major`
- `positions`: `f32`, shape `[638688, 3]`, layout `point-major-xyz`
- `utci`: `f32`, shape `[638688, 24]`, layout `point-major-hour`
- `shadingIndex`: `f32`, shape `[638688]`, layout `point-major`
- `surfaceFlags`: `u8`, shape `[activeCount]` (current bundle: `[638688]`), layout `point-major`

Raw classification contract:

- Raw classification is `surfaceFlags` bitflags only. There is no raw primary enum/class array.
- `surfaceFlags` is row-aligned with `canonicalIndices`, `positions`, `utci`, and `shadingIndex`.
- Bit values are `ground = 1`, `street_surface = 2`, `building_footprint = 4`.
- Street, road, roads, sidewalk, and sidewalks are one street-surface family in the classified export.
- Building footprints are overlay/exclusion flags only; building-only cells do not become active rows.
- Train tracks remain excluded from active sampled cells.

## `cells.geoparquet` Schema

`cells.geoparquet` has `638688` rows, one per active cell only.

Columns:

- `active_index` (`int64`)
- `canonical_index` (`uint32`)
- `geometry` (`binary`, WKB Point)
- `lon` (`double`)
- `lat` (`double`)
- `x` (`float`)
- `y` (`float`)
- `z` (`float`)
- `shading_index` (`float`)
- `surface_flags` (`uint8`): raw per-row bitflags copied from `surfaceFlags` with `ground = 1`, `street_surface = 2`, `building_footprint = 4`
- `surface_class` (`string`): display-priority class derived from `surface_flags` with `building_footprint > street_surface > ground`; `unknown` is not legal for classified active export rows
- `is_street_surface` (`bool`): true when the row matches street/road/sidewalk-family sampled geometry
- `is_building_footprint` (`bool`): true when the sample center falls inside the building-footprint overlay rasterization
- `include_in_public_realm_stats` (`bool`): true for street/road/sidewalk-family rows excluding any building-footprint rows
- `include_in_outdoor_surface_stats` (`bool`): true for active sampled rows excluding any building-footprint rows
- `utci_00` through `utci_23` (`float`, exactly one column per hour)

GeoParquet notes:

- Primary geometry column: `geometry`
- Geometry type: `Point`
- Geometry CRS: `EPSG:4326`
- GeoParquet metadata is stored under the schema `geo` key
- Classified export rows are sampled-surface-only active rows. Building footprints never create rows on their own.

## Active Vs Canonical

The canonical grid is the rectangular compute/indexing domain used by the live analysis runtime. It contains `1492551` canonical rows.

The GIS handoff does not export that full rectangle as valid results. Only active sampled cells are exported. The current bundle therefore contains `638688` active rows, with `rowCountEqualsActiveRows = true` and `activeRatio = 0.42791703600077985`.

That distinction matters downstream:

- `canonical_index` is useful for traceability back to runtime indexing.
- `cells.geoparquet` rows are the valid GIS result surface.
- Inactive canonical cells are outside the reusable GIS result contract and should remain no-data in downstream maps and statistics.

## Timing Breakdown

Collector timings from raw metadata and manifest:

- `routeLoad`: `751.9355 ms`
- `liveSessionReady`: `10678.1642 ms`
- `utciCollection`: `32879.9269 ms`
- `shadingCollection`: `1481.4859 ms`
- `binarySerialization`: `350.2259 ms`
- `collector total`: `55923.0294 ms`

Postprocessor timings from manifest:

- `binaryLoadValidation`: `322.5447 ms`
- `epsg2039ToWgs84`: `274.7572 ms`
- `cellsGeoparquetWrite`: `1919.7126 ms`
- `debugSampleWrite`: `768.3986 ms`
- `manifestWrite`: `394.9381 ms`
- `postprocessor total`: `3286.7725 ms`

Combined export runtime:

- `totalExportRuntimeMs`: `59209.8019 ms`

Timing contract:

- Raw collector metadata keeps the collector payload under `timingsMs`, including `timingsMs.total`.
- `manifest.json` copies that payload to `collectorTimingsMs`.
- `manifest.json` records Python runtime at `postprocessorTimingsMs.totalPostprocessorRuntime`.
- `totalExportRuntimeMs = collectorTimingsMs.total + postprocessorTimingsMs.totalPostprocessorRuntime`.

## QA Sample And Map Check

`qa/debug-sample.geojson` is a small visual QA overlay only:

- current row count: `5000`
- sample strategy: `evenly-spaced-active-rows`
- purpose: local alignment and spot-checking of the active GIS surface against the map base and district outline

It is not production dashboard data and should not be treated as a complete analysis layer.

The existing local map check page is:

- `data/3d_models/Innovation-District/innovation_district_map_check.html`

It keeps the original outline/layer bounds check and can now optionally overlay:

- `../../gis/Innovation-District/2025-08-15_2m/qa/debug-sample.geojson`

Open it through a local server so `fetch(...)` works.

## Migration Notes

This bundle already uses the reusable per-export layout under `data/gis/Innovation-District/2025-08-15_2m/`.

For this bundle:

- the canonical final outputs are at bundle root plus `qa/`
- raw collector artifacts live under `raw/`
- no legacy flat root-level `*_active-cells.*` files are the intended steady-state handoff surface

## Downstream Boundary

`fast-utci` stops at the reusable GIS handoff bundle. It does not implement dashboard UI.

Downstream responsibility belongs to `innovation_dashboard` and stays downstream of `fast-utci`:

- derive PMTiles or other map artifacts from `cells.geoparquet`
- derive summary/chart JSON from `cells.geoparquet`
- decide MapLibre/Chart.js presentation details in that repo

Recommended downstream path:

- `cells.geoparquet -> PMTiles/vector/raster tiles for MapLibre`
- `cells.geoparquet -> summary/chart JSON for charts/cards`

## Downstream Filter Examples

Common downstream filters should operate on `cells.geoparquet`:

```sql
WHERE include_in_outdoor_surface_stats = true
```

```sql
WHERE include_in_public_realm_stats = true
```

```sql
WHERE is_building_footprint = false
```

## Pasteable Dashboard Handoff

```text
Innovation District GIS handoff is ready in fast-utci:

- Bundle root: data/gis/Innovation-District/2025-08-15_2m/
- Canonical artifact: cells.geoparquet
- Supporting files: manifest.json, qa/debug-sample.geojson
- CRS: source x/y/z in EPSG:2039; GeoParquet geometry and lon/lat in EPSG:4326
- Rows: 638688 active cells only
- Canonical grid rows: 1492551 (do not render as valid GIS results)
- Hours: utci_00..utci_23 plus shading_index
- Active mask provenance: base+road, checksum d4e1236d
- EarthAnchorPoint is unusable here (0,0); use projected coordinates/district outline contract instead
- Downstream boundary: innovation_dashboard should derive PMTiles/map artifacts and summary/chart JSON from cells.geoparquet
- QA-only sample: qa/debug-sample.geojson (5000 rows, not production data)
```
