from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any


LEGACY_ALL_HOURS_GEOJSON_DEFAULT_MAX_ROWS = 100_000

_WGS84_PROJJSON = {
    "$schema": "https://proj.org/schemas/v0.7/projjson.schema.json",
    "type": "GeographicCRS",
    "name": "WGS 84",
    "datum_ensemble": {
        "name": "World Geodetic System 1984 ensemble",
        "members": [
            {"name": "World Geodetic System 1984 (Transit)"},
            {"name": "World Geodetic System 1984 (G730)"},
            {"name": "World Geodetic System 1984 (G873)"},
            {"name": "World Geodetic System 1984 (G1150)"},
            {"name": "World Geodetic System 1984 (G1674)"},
            {"name": "World Geodetic System 1984 (G1762)"},
            {"name": "World Geodetic System 1984 (G2139)"},
        ],
        "ellipsoid": {
            "name": "WGS 84",
            "semi_major_axis": 6378137,
            "inverse_flattening": 298.257223563,
        },
        "accuracy": "2.0",
        "id": {"authority": "EPSG", "code": 6326},
    },
    "coordinate_system": {
        "subtype": "ellipsoidal",
        "axis": [
            {
                "name": "Geodetic longitude",
                "abbreviation": "Lon",
                "direction": "east",
                "unit": "degree",
            },
            {
                "name": "Geodetic latitude",
                "abbreviation": "Lat",
                "direction": "north",
                "unit": "degree",
            },
        ],
    },
    "id": {"authority": "EPSG", "code": 4326},
}


@dataclass(frozen=True)
class GeoParquetContract:
    metadata_version: str = "1.0.0"
    primary_column: str = "geometry"
    encoding: str = "WKB"
    geometry_types: tuple[str, ...] = ("Point",)
    crs: str = "EPSG:4326"
    note: str = (
        "GeoParquet metadata is stored under the schema 'geo' key; geometry is WKB Point "
        "in EPSG:4326 and lon/lat columns duplicate the point coordinates for easy QA."
    )

    def geo_metadata(self) -> dict[str, Any]:
        return {
            "version": self.metadata_version,
            "primary_column": self.primary_column,
            "columns": {
                self.primary_column: {
                    "encoding": self.encoding,
                    "geometry_types": list(self.geometry_types),
                    "crs": copy.deepcopy(_WGS84_PROJJSON),
                }
            },
        }

    def manifest_note(self) -> dict[str, Any]:
        return {
            "primaryColumn": self.primary_column,
            "encoding": self.encoding,
            "geometryTypes": list(self.geometry_types),
            "crs": self.crs,
            "note": self.note,
        }


GEOPARQUET_CONTRACT = GeoParquetContract()
