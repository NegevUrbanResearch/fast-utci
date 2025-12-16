<!-- d8ea0dc4-6f6e-4bca-8cca-0216b1cc36a2 016fe2f7-2a8d-4721-b9d4-c051884c39e1 -->
# IMS Weather API Integration Plan

## Overview

Replace EPW file loading with IMS weather station API integration. The API requires fetching data from two stations (59 for basic weather, 60 for radiation) and combining them. Missing horizontal infrared radiation will be estimated using empirical formulas.

## API Details (from testing)

**Base URL:** `https://api.ims.gov.il/v1/envista/`

**Authentication:** `Authorization: ApiToken {API_KEY}`

**Endpoints:**

- `/stations` - List all stations
- `/stations/{stationId}` - Station metadata
- `/stations/{stationId}/data?from={date}&to={date}` - Weather data

**Response Structure:**

```json
{
  "stationId": int,
  "data": [{
    "datetime": "ISO8601+timezone",
    "channels": [{
      "id": int,
      "name": "TD|WS|RH|Grad|NIP|DiffR|...",
      "value": float,
      "status": int,  // 1=valid, 2/4=invalid
      "valid": bool
    }]
  }]
}
```

**Station Configuration:**

- **Station 59** (BEER SHEVA_20250506): TD, WS, RH, BP, TG, etc. (basic weather, NO radiation)
- **Station 60** (BEER SHEVA UNI): Grad, NIP, DiffR (radiation only)
- **Location:** Station 59: lat=31.2515, lon=34.7995

**Field Mapping:**

- `TD` → `air_temp` (degC)
- `WS` → `wind_speed` (m/sec)
- `RH` → `relative_humidity` (%)
- `NIP` → `direct_normal_radiation` (w/m2)
- `DiffR` → `diffuse_horizontal_radiation` (w/m2)
- `Grad` → `global_horizontal_radiation` (w/m2, for validation)
- `horizontal_infrared_radiation_intensity` → **MISSING** (must estimate)

## Implementation Steps

### 1. API Key Management

- Add `IMS_API_KEY` to environment variables (recommended) or `fast_utci.toml` under `[api]` section
- Update `.gitignore` to exclude any local API key files
- Create helper function `get_api_key()` that checks env var first, then config file

### 2. Create API Fetching Function

**File:** `src/fast_utci/shared/weather.py`

Add `fetch_weather_from_api()`:

- Accept: `station_ids: List[int]`, `from_date: str`, `to_date: str`, `api_key: str`
- Fetch data from multiple stations (59 and 60)
- Parse JSON response, filter channels where `valid: true` or `status: 1`
- Convert datetime strings to Python datetime objects
- Return combined DataFrame with all fields merged by datetime

### 3. Create Infrared Radiation Estimator

**File:** `src/fast_utci/shared/weather.py`

Add `estimate_infrared_radiation()`:

- Implement Brunt-type model: `IR = εσTₐ⁴(0.56 - 0.08√e)(1 + 0.006C²)`
  - Where: ε=0.95, σ=5.67e-8, Tₐ=air_temp (K), e=vapor pressure (from RH), C=cloud factor
- Alternative: Swinbank model if simpler: `IR = 0.94σTₐ⁴ - 171`
- Use air_temp and relative_humidity from API data

### 4. Create APIAdapter Class

**File:** `src/fast_utci/shared/weather.py`

Implement `APIAdapter(WeatherDataSource)`:

- Store DataFrame from API fetch
- Implement all protocol methods (`get_temperature`, `get_wind_speed`, etc.)
- Call `estimate_infrared_radiation()` for `get_infrared_radiation()`
- Support filtering via `filter_by_period()` method

### 5. Update load_weather_data()

**File:** `src/fast_utci/shared/weather.py`

Modify to accept either:

- File path (existing EPW behavior)
- API config dict: `{"source": "api", "station_ids": [59, 60], "from_date": "...", "to_date": "...", "api_key": "..."}`

### 6. Update WeatherDataManager

**File:** `src/fast_utci/shared/weather.py`

Modify `_load_weather_data()` to handle API config dict in addition to file paths and DataFrames.

### 7. Update read_project_data()

**File:** `src/fast_utci/shared/io/model.py`

Modify `weather_path` parameter to accept either:

- File path (existing behavior)
- API config dict

### 8. Update MRTCalculator Location Handling

**File:** `src/fast_utci/mrt/mrt_calculator.py`

Add `set_location()` method:

- Accept: `latitude: float`, `longitude: float`, `time_zone: str`, `elevation: float`
- Create Location object from ladybug.location
- Update `set_location_from_epw()` to call `set_location()` internally

### 9. Update run_analysis.py

**File:** `run_analysis.py`

Modify `run_analysis_core()`:

- Change `epw_file` parameter to `weather_source: Union[str, Dict]`
- If dict, pass to `read_project_data()` as API config
- Update `mrt_calc.set_location_from_epw()` to handle API case:
  - If API config, use station 59 location (31.2515, 34.7995) directly
  - If EPW file, use existing behavior

### 10. Testing & Validation

- Test API fetching with 2017-08-15 data (matching `data/analyses/20250815_grid_2m_fullday.bin`)
- Compare UTCI results between EPW and API data
- Validate infrared radiation estimates are reasonable (typically 200-500 W/m² range)
- Ensure datetime alignment between stations 59 and 60

## Configuration Example

```toml
# fast_utci.toml
[api]
# Optional: API key (prefer environment variable IMS_API_KEY)
# api_key = "your-key-here"
```

## Notes

- API returns 10-minute interval data; may need to resample to hourly for compatibility
- Handle missing/invalid data gracefully (status != 1 or valid != true)
- Timezone handling: API returns `+03:00` (Israel time), ensure proper conversion
- Station 60 has limited fields (only radiation + inactive TD), so combining is necessary

### To-dos

- [ ] Create secure API key storage (.env file, environment variable loading, .env.example template)
- [ ] Implement horizontal infrared radiation estimation using Brunt formula (vapor pressure + sky emissivity)
- [ ] Create fetch_weather_from_ims_api() function to get weather data from IMS API and map fields
- [ ] Extend load_weather_data() and WeatherDataManager to support API config alongside file paths
- [ ] Add set_location() method to MRTCalculator for dict-based location (API data)
- [ ] Update read_project_data() and run_analysis_core() to accept API config
- [ ] Create test script to compare API-based results vs existing 20250815 binary data