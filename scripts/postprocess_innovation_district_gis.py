from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-process Innovation District raw GIS artifacts.")
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--georef", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--debug-geojson-limit", type=int, default=5000)
    return parser.parse_args()


def _ensure_src_on_sys_path() -> None:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    src_dir = repo_root / "src"
    if Path(sys.path[0]).resolve() == script_dir and str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def main() -> None:
    args = _parse_args()
    _ensure_src_on_sys_path()
    from fast_utci.innovation_district_gis.orchestrator import postprocess_active_cells

    outputs = postprocess_active_cells(
        metadata_path=args.metadata,
        georef_path=args.georef,
        out_dir=args.out_dir,
        debug_geojson_limit=args.debug_geojson_limit,
    )
    print(
        json.dumps(
            {
                key: str(value)
                for key, value in asdict(outputs).items()
                if value is not None and (not isinstance(value, Path) or value.exists())
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
