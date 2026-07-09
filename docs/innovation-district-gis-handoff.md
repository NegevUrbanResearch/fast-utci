# Innovation District GIS Handoff

This doc explains the GIS bundle produced from the Innovation District live WebGPU route.

The important thing to know: GIS export is downstream of a verified viewer run. It is not a separate Python/Ladybug UTCI simulation and it is not a Python `.bin` product path.

## Flow

```text
viewer main route `/`
  -> live WebGPU UTCI / SAI
  -> collector raw active-cell arrays
  -> Python geospatial postprocess validation
  -> GeoParquet + manifest + QA sample
```

The collector accepts only the live-route GPU-native path. The raw files are provenance and postprocessing inputs; the handoff artifact for GIS users is `cells.geoparquet`.

## Current Bundle

```text
data/gis/Innovation-District/2025-08-15_2m/
  cells.geoparquet
  manifest.json
  qa/debug-sample.geojson
  raw/
```

Use `manifest.json` as the first file to inspect. It records the schema version, CRS, GeoParquet layout, and column names.

## CRS

The raw analysis is tied to the project georeference sidecar:

```text
data/3d_models/Innovation-District/innovation_district.georef.json
```

The final GIS bundle uses:

- source projected CRS: `EPSG:2039`
- output geometry CRS: `EPSG:4326`
- duplicate `lon` / `lat` columns for easy QA



## Active-Cell Policy

The raw export contains active cells only, not every canonical grid cell.

Current bundle counts:

- canonical cells: `1,492,551`
- active cells: `638,688`
- hours: `24`

The raw metadata records:

- `activeMask.source`
- `activeMask.checksum`
- `canonicalIndices`
- point positions
- UTCI values
- SAI values
- surface flags

That active-row-only policy matters when joining back to a canonical grid or comparing against route diagnostics.

## Main Files

- `cells.geoparquet`: primary handoff file for GIS and downstream map/chart work.
- `manifest.json`: bundle contract, CRS, schema, and QA context.
- `qa/debug-sample.geojson`: small readable QA sample.
- `raw/*.bin` and raw metadata: collector provenance and postprocess inputs.



## Regenerate / Postprocess

The raw collector is run from the viewer script:

```powershell
cd viewer
node --import tsx ./scripts/export-innovation-district-gis.ts
```

Postprocessing is handled by Python geospatial validation/export code. This step packages and checks live-route results; it does not rerun the legacy Ladybug-backed UTCI analysis.

```powershell
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe scripts/postprocess_innovation_district_gis.py --metadata data/gis/Innovation-District/2025-08-15_2m/raw/2025-08-15_2m_active-cells.metadata.json --georef data/3d_models/Innovation-District/innovation_district.georef.json --out-dir data/gis/Innovation-District
```



## Verification

Before handing off a regenerated bundle, run the focused Python checks:

```powershell
python -m pytest tests/test_innovation_district_gis_raw.py tests/test_innovation_district_gis_qa_manifest.py tests/test_postprocess_innovation_district_gis.py
```

If the live collector changed, also verify the viewer-side export contract:

```powershell
cd viewer
npm run check
npm test -- --run tests/gis/innovationDistrictExport.test.ts
```

