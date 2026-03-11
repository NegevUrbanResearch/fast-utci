"""
Fetch IMS Envista stations and export only those with data suitable for MRT/UTCI analysis.

Required for UTCI: air_temp (TD), wind_speed (WS), relative_humidity (RH)
Required for MRT/solar: at least one of global_horizontal (Grad), direct_normal (NIP), diffuse (DiffR)

The list endpoint GET /stations returns ALL stations with monitors embedded - no per-station fetching.
"""

import json
from pathlib import Path

import requests

# Channels we need for MRT/UTCI (IMS monitor names)
REQUIRED_WEATHER = {"TD", "WS", "RH"}  # temp, wind, humidity - minimum for UTCI
REQUIRED_RADIATION = {"Grad", "NIP", "DiffR"}  # at least one for solar/MRT


def get_monitor_names(station: dict) -> set:
    """Extract monitor names from station's monitors array (present in list response)."""
    monitors = station.get("monitors") or station.get("channels") or []
    return {str(m.get("name", "")).strip() for m in monitors if m.get("name")}


def has_required_data(monitor_names: set) -> tuple[bool, bool]:
    """
    Check if station has minimum data for MRT/UTCI.
    Returns (has_weather, has_radiation).
    """
    has_weather = all(ch in monitor_names for ch in REQUIRED_WEATHER)
    has_rad = any(r in monitor_names for r in REQUIRED_RADIATION)
    return has_weather, has_rad


def load_token() -> str:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        raise FileNotFoundError(f".env not found at {env_path}")
    token = env_path.read_text(encoding="utf-8").strip()
    if not token:
        raise ValueError(".env is empty - add IMS_API_KEY or API token")
    return token


def main():
    token = load_token()
    headers = {"Authorization": f"ApiToken {token}"}
    base_url = "https://api.ims.gov.il/v1/envista"

    resp = requests.get(f"{base_url}/stations", headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    stations = data if isinstance(data, list) else data.get("data", [])
    print(f"Total stations from API: {len(stations)}")

    # List endpoint includes monitors - no per-station fetching needed
    full_data = []
    weather_only = []
    all_summary = []

    for s in stations:
        names = get_monitor_names(s)
        has_weather, has_rad = has_required_data(names)
        loc = s.get("location") or {}
        row = {
            "stationId": s.get("stationId"),
            "name": s.get("name"),
            "shortName": s.get("shortName"),
            "regionId": s.get("regionId"),
            "lat": loc.get("latitude"),
            "lon": loc.get("longitude"),
            "timebase": s.get("timebase"),
            "active": s.get("active"),
            "TD": "TD" in names,
            "WS": "WS" in names,
            "RH": "RH" in names,
            "radiation": has_rad,
            "monitors": sorted(names),
        }
        all_summary.append(row)
        if has_weather and has_rad:
            full_data.append(s)
        elif has_weather:
            weather_only.append(s)

    output = {
        "meta": {
            "source": "IMS Envista API",
            "url": base_url,
            "total_stations": len(stations),
            "stations_with_full_data": len(full_data),
            "stations_weather_only": len(weather_only),
            "note": "Full = TD+WS+RH + at least one of Grad/NIP/DiffR. "
            "Weather-only can be used with IR estimation.",
        },
        "stations_full": [
            {
                "stationId": s.get("stationId"),
                "name": s.get("name"),
                "shortName": s.get("shortName"),
                "lat": (s.get("location") or {}).get("latitude"),
                "lon": (s.get("location") or {}).get("longitude"),
                "regionId": s.get("regionId"),
                "monitors": sorted(get_monitor_names(s)),
            }
            for s in full_data
        ],
        "stations_weather_only": [
            {
                "stationId": s.get("stationId"),
                "name": s.get("name"),
                "shortName": s.get("shortName"),
                "lat": (s.get("location") or {}).get("latitude"),
                "lon": (s.get("location") or {}).get("longitude"),
                "regionId": s.get("regionId"),
                "monitors": sorted(get_monitor_names(s)),
            }
            for s in weather_only
        ],
        "all_stations_summary": all_summary,
    }

    out_dir = Path(__file__).resolve().parent.parent / "data" / "ims"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "stations_filtered.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved to {out_path}")
    print(f"  Full data (temp+wind+RH+radiation): {len(output['stations_full'])}")
    print(f"  Weather only (IR estimatable): {len(output['stations_weather_only'])}")
    print(f"  All stations summary: {len(output['all_stations_summary'])}")

    # CSV for quick scanning (escape commas in names)
    csv_path = out_dir / "stations_filtered.csv"
    rows = []
    for st in output["all_stations_summary"]:
        name = (st.get("name") or "").replace('"', '""')
        short = (st.get("shortName") or "").replace('"', '""')
        rows.append(
            f'{st["stationId"]},"{name}","{short}",{st["regionId"]},'
            f'{st["lat"]},{st["lon"]},{st["TD"]},{st["WS"]},{st["RH"]},{st["radiation"]}'
        )
    header = "stationId,name,shortName,regionId,lat,lon,TD,WS,RH,radiation"
    csv_path.write_text(header + "\n" + "\n".join(rows), encoding="utf-8")
    print(f"CSV saved to {csv_path}")


if __name__ == "__main__":
    main()
