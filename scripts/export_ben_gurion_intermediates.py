"""
Export solar, sky, and MRT intermediates for the Ben-Gurion base case.

Produces *_solar.json, *_sky.json, *_mrt.json (and optionally *_weather_sample.json)
next to the analysis .bin so the WebGPU parity harness can compare intermediate
stages. Run once when the Python pipeline or model changes; no Python in CI.

Usage (from repo root):
  python scripts/export_ben_gurion_intermediates.py \\
    --base-path data/analyses/Ben-Gurion/20250815_grid_2m_fullday \\
    --model data/3d_models/Ben-Gurion/original_with_layers.glb \\
    [--stage solar] [--stage sky] [--stage mrt] [--stage weather]
  Default: export solar, sky, and MRT.

Reference format:
  *_solar.json: { "numPositions", "numHours", "solarExposure": number[] }
    Point-major flat: [p0_h0, p0_h1, ..., p0_h23, p1_h0, ...]
  *_sky.json:   { "numPositions", "skyExposure": number[] }
  *_mrt.json:   { "numPositions", "numHours", "mrt": number[], "short_erf"?, "long_erf"?, "short_dmrt"?, "long_dmrt"? }
    Point-major flat, same layout as solar (per-hour MRT in °C).
  *_weather_sample.json: { "numHours": 3, "weather": [ { "air_temp", ... }, ... ] }
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

DEFAULT_STAGES = ["solar", "sky", "mrt"]

# Add project root for imports
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fast_utci.mrt.exposure import compute_exposure_batch
from fast_utci.mrt.mesh import load_context_meshes
from fast_utci.mrt.solar import SunData
from fast_utci.mrt.solarcal import create_solar_body_parameters


def load_metadata(base_path: Path) -> dict:
    meta_path = Path(str(base_path) + ".json")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)


def load_positions_from_bin(base_path: Path, num_positions: int) -> np.ndarray:
    bin_path = Path(str(base_path) + ".bin")
    if not bin_path.exists():
        raise FileNotFoundError(f"Binary not found: {bin_path}")
    with open(bin_path, "rb") as f:
        data = f.read()
    # Format: 4 num_positions, 4 num_hours, then positions (num_positions * 3 * 4 bytes)
    if len(data) < 8 + num_positions * 3 * 4:
        raise ValueError(f"Binary too short for {num_positions} positions")
    n = struct.unpack_from("<I", data, 0)[0]
    if n != num_positions:
        raise ValueError(f"Binary num_positions {n} != metadata {num_positions}")
    positions = np.frombuffer(data, dtype=np.float32, offset=8, count=num_positions * 3)
    return positions.reshape(num_positions, 3).copy()


def sun_data_from_metadata(meta: dict) -> SunData:
    """Build SunData from analysis JSON sun_positions (same as used to generate .bin)."""
    sun_positions = meta.get("sun_positions")
    if not sun_positions:
        raise ValueError("Metadata has no sun_positions")
    # Sort by hour and build arrays
    by_hour = sorted(sun_positions, key=lambda x: x["hour"])
    sun_vectors = np.array([sp["vector"] for sp in by_hour], dtype=np.float64)
    is_sun_up = np.array([sp["is_up"] for sp in by_hour], dtype=bool)
    # SunData expects solar_times and hoys; we don't need them for exposure math
    from ladybug.dt import DateTime
    hours = meta.get("hours", list(range(24)))
    solar_times = [DateTime(8, 15, h) for h in hours]  # Aug 15 placeholder
    hoys = np.array([t.hoy for t in solar_times], dtype=np.float64)
    return SunData(
        sun_vectors=sun_vectors,
        is_sun_up=is_sun_up,
        solar_times=solar_times,
        hoys=hoys,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export solar/sky intermediates for parity.")
    parser.add_argument(
        "--base-path",
        type=Path,
        default=REPO_ROOT / "data/analyses/Ben-Gurion/20250815_grid_2m_fullday",
        help="Base path without extension (e.g. .../20250815_grid_2m_fullday)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=REPO_ROOT / "data/3d_models/Ben-Gurion/original_with_layers.glb",
        help="Path to GLB context model",
    )
    parser.add_argument(
        "--stage",
        choices=["solar", "sky", "mrt", "weather"],
        action="append",
        default=[],
        help="Stage(s) to export (default: solar, sky, mrt)",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars")
    args = parser.parse_args()

    base_path = args.base_path
    if not base_path.is_absolute():
        base_path = (REPO_ROOT / base_path).resolve()
    model_path = args.model
    if not model_path.is_absolute():
        model_path = (REPO_ROOT / model_path).resolve()

    stages = args.stage if args.stage else DEFAULT_STAGES
    show_progress = not args.no_progress
    need_exposure = "solar" in stages or "sky" in stages or "mrt" in stages

    meta = load_metadata(base_path)
    num_positions = int(meta["num_positions"])
    hours = meta.get("hours", list(range(24)))
    num_hours = len(hours)

    positions = load_positions_from_bin(base_path, num_positions)
    sun_data = sun_data_from_metadata(meta)

    mesh_context = None
    if model_path.exists():
        mesh_context = load_context_meshes([str(model_path)])
    else:
        print(f"Warning: model not found {model_path}, exporting with no occlusion (full exposure)")

    config = None
    try:
        from fast_utci.shared import load_config
        config = load_config().mrt
    except Exception:
        pass  # use pt_count and height below

    results = []
    if need_exposure:
        results = compute_exposure_batch(
            positions,
            sun_data,
            mesh_context=mesh_context,
            pt_count=1 if config is None else None,
            height=1.7 if config is None else None,
            show_progress=show_progress,
            config=config,
        )
        if len(results) != num_positions:
            raise ValueError(f"Exposure returned {len(results)} results, expected {num_positions}")

    if "solar" in stages:
        solar_flat = []
        for r in results:
            solar_flat.extend(r.fract_body_exp.tolist())
        out_path = Path(str(base_path) + "_solar.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {"numPositions": num_positions, "numHours": num_hours, "solarExposure": solar_flat},
                f,
                separators=(",", ":"),
            )
        print(f"Wrote {out_path}")

    if "sky" in stages:
        sky_exposure = [r.sky_exposure for r in results]
        out_path = Path(str(base_path) + "_sky.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {"numPositions": num_positions, "skyExposure": sky_exposure},
                f,
                separators=(",", ":"),
            )
        print(f"Wrote {out_path}")

    if "mrt" in stages:
        from ladybug.epw import EPW
        from ladybug.analysisperiod import AnalysisPeriod as LBAnalysisPeriod
        from fast_utci.mrt.mrt_calculator import _create_solarcal_from_epw

        epw_path = meta.get("epw_file")
        if not epw_path:
            raise ValueError("Metadata has no epw_file; cannot export MRT")
        epw_full = REPO_ROOT / epw_path if not Path(epw_path).is_absolute() else Path(epw_path)
        if not epw_full.exists():
            raise FileNotFoundError(f"EPW not found: {epw_full}")
        epw_data = EPW(str(epw_full))
        # Analysis day from metadata date (e.g. 20250815 -> Aug 15)
        date_str = meta.get("date", "20250815")
        month = int(date_str[4:6])
        day = int(date_str[6:8])
        day_period = LBAnalysisPeriod(month, day, 0, month, day, 23)
        target_hours = hours
        ground_reflectance = 0.25
        if config is not None and hasattr(config, "ground_reflectance"):
            ground_reflectance = getattr(config, "ground_reflectance", 0.25)
        body_params = create_solar_body_parameters()
        mrt_flat = []
        short_erf_flat = []
        long_erf_flat = []
        short_dmrt_flat = []
        long_dmrt_flat = []
        for i, exp in enumerate(results):
            if show_progress and (i + 1) % 10000 == 0:
                print(f"  MRT {i + 1}/{num_positions}")
            mrt_result = _create_solarcal_from_epw(
                epw_data, exp, None, target_hours, ground_reflectance, body_params
            )
            mrt_flat.extend(mrt_result.mrt.tolist())
            short_erf_flat.extend(mrt_result.short_erf.tolist())
            long_erf_flat.extend(mrt_result.long_erf.tolist())
            short_dmrt_flat.extend(mrt_result.short_dmrt.tolist())
            long_dmrt_flat.extend(mrt_result.long_dmrt.tolist())
        out_path = Path(str(base_path) + "_mrt.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "numPositions": num_positions,
                    "numHours": num_hours,
                    "mrt": mrt_flat,
                    "short_erf": short_erf_flat,
                    "long_erf": long_erf_flat,
                    "short_dmrt": short_dmrt_flat,
                    "long_dmrt": long_dmrt_flat,
                },
                f,
                separators=(",", ":"),
            )
        print(f"Wrote {out_path}")

    if "weather" in stages:
        from ladybug.epw import EPW
        from ladybug.analysisperiod import AnalysisPeriod as LBAnalysisPeriod

        epw_path = meta.get("epw_file")
        if not epw_path:
            raise ValueError("Metadata has no epw_file; cannot export weather sample")
        epw_full = REPO_ROOT / epw_path if not Path(epw_path).is_absolute() else Path(epw_path)
        if not epw_full.exists():
            raise FileNotFoundError(f"EPW not found: {epw_full}")
        epw_data = EPW(str(epw_full))
        date_str = meta.get("date", "20250815")
        month = int(date_str[4:6])
        day = int(date_str[6:8])
        day_period = LBAnalysisPeriod(month, day, 0, month, day, 23)
        air_temp = epw_data.dry_bulb_temperature.filter_by_analysis_period(day_period).values
        direct_normal = epw_data.direct_normal_radiation.filter_by_analysis_period(day_period).values
        diffuse_horizontal = epw_data.diffuse_horizontal_radiation.filter_by_analysis_period(day_period).values
        horiz_ir = epw_data.horizontal_infrared_radiation_intensity.filter_by_analysis_period(day_period).values
        wind = epw_data.wind_speed.filter_by_analysis_period(day_period).values
        rel_humidity = epw_data.relative_humidity.filter_by_analysis_period(day_period).values
        n_hrs = min(3, len(air_temp))
        weather_sample = []
        for h in range(n_hrs):
            weather_sample.append({
                "air_temp": float(air_temp[h]),
                "direct_normal": float(direct_normal[h]),
                "diffuse_horizontal": float(diffuse_horizontal[h]),
                "horiz_infrared": float(horiz_ir[h]),
                "wind_speed": float(wind[h]),
                "rel_humidity": float(rel_humidity[h]),
            })
        out_path = Path(str(base_path) + "_weather_sample.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"numHours": n_hrs, "weather": weather_sample}, f, separators=(",", ":"))
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
