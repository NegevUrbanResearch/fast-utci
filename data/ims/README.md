# IMS Envista Station Data

Fetched from https://api.ims.gov.il/v1/envista using `scripts/fetch_ims_stations.py`.

The list endpoint `GET /stations` returns all 187 stations with monitors embedded in one call.

## Files

- **stations_filtered.json** – Full structured data:
  - `stations_full`: TD+WS+RH + radiation (Grad/NIP/DiffR) – ready for MRT/UTCI
  - `stations_weather_only`: TD+WS+RH only – can use with IR estimation
  - `all_stations_summary`: All 187 stations

- **stations_filtered.csv** – Flat table for quick scanning (stationId, name, lat, lon, TD, WS, RH, radiation)

## Notes

- Stations with `_1m` suffix are often radiation-only or specialized; many lack TD/WS/RH.
- **Ness Tziona**: Not in IMS API. Use Bet Dagan (54) or Tel Aviv Coast (178) as nearby proxies.

## Best nearby stations for Ness Tziona (central coastal plain)

- **Bet Dagan** (54): 32.0073, 34.8138 – TD, WS, RH (weather only; pair with BET DAGAN RAD 85 for radiation)
- **Tel Aviv Coast** (178): 32.058, 34.7588 – TD, WS, RH (weather only)
- **Beer Sheva** (59) + **Beer Sheva UNI** (60): 59 has temp/RH/wind; 60 has radiation – pair for full data

For EPW-based analysis, use **Bet Dagan TMYx** from Climate.OneBuilding (closest EPW to Ness Tziona).
