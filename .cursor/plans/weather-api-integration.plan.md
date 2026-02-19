# Weather API Integration Plan

## Overview

Extend fast-utci to support real-time weather station API integration alongside existing EPW file support. The implementation follows a **provider-agnostic architecture** designed for global scalability—starting with Israel Meteorological Service (IMS) as the pilot, but architected for worldwide expansion.

### Key Design Principles

1. **Dual-mode operation**: EPW files for "typical" climate analysis, API for "actual" weather analysis
2. **Provider abstraction**: Weather provider interface allows adding new APIs (OpenWeather, Meteomatics, etc.)
3. **Cloud-aware estimation**: Horizontal infrared radiation estimated using Brutsaert-Crawford-Duchon method
4. **TDD approach**: Tests written before implementation, validation against EPW baseline

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     WeatherDataSource Protocol                      │
│  get_temperature() | get_wind_speed() | get_infrared_radiation()   │
└─────────────────────────────────────────────────────────────────────┘
           ▲                    ▲                    ▲
           │                    │                    │
    ┌──────┴──────┐     ┌───────┴───────┐    ┌──────┴──────┐
    │ EPWAdapter  │     │ APIAdapter    │    │ Future APIs │
    │ (existing)  │     │ (new)         │    │             │
    └─────────────┘     └───────────────┘    └─────────────┘
                               │
                    ┌──────────┴──────────┐
                    │ WeatherProvider     │
                    │ Protocol            │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
       ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
       │ IMSProvider │  │OpenWeather  │  │ Meteomatics │
       │ (pilot)     │  │ (future)    │  │ (future)    │
       └─────────────┘  └─────────────┘  └─────────────┘
```

---

## Phase 1: Infrastructure & Core Components

### 1.1 API Key Management

**File:** `src/fast_utci/shared/config.py` (extend existing)

```python
def get_api_key(provider: str = "ims") -> Optional[str]:
    """
    Get API key with priority: env var > config file > None.
    
    Env vars: IMS_API_KEY, OPENWEATHER_API_KEY, etc.
    """
```

**Files to create/update:**
- `.env.example` - Template with placeholder keys
- `.gitignore` - Ensure `.env` is excluded
- `fast_utci.toml` - Add `[api.ims]` section

### 1.2 Infrared Radiation Estimation Module

**File:** `src/fast_utci/shared/infrared.py` (NEW)

This is the critical component for global compatibility. Uses the **Brutsaert-Crawford-Duchon** method which:
- Works for any climate (desert to tropical)
- Derives cloud cover from measured solar radiation (no separate cloud data needed)
- Validated in peer-reviewed literature

```python
"""
Horizontal infrared radiation estimation for locations without direct measurements.

Implements:
- Brutsaert (1975): Clear-sky atmospheric emissivity
- Crawford-Duchon (1999): Cloud cover correction using solar radiation ratio

Reference: Crawford, T.M. and Duchon, C.E. (1999). An improved parameterization 
for estimating effective atmospheric emissivity for use in calculating daytime 
downwelling longwave radiation. J. Appl. Meteor., 38, 474-480.
"""

def estimate_vapor_pressure(air_temp_c: np.ndarray, relative_humidity: np.ndarray) -> np.ndarray:
    """
    Calculate water vapor pressure using Magnus-Tetens formula.
    
    Args:
        air_temp_c: Air temperature in Celsius
        relative_humidity: Relative humidity in percent (0-100)
        
    Returns:
        Vapor pressure in hPa (millibars)
    """

def calculate_clear_sky_emissivity(air_temp_k: np.ndarray, vapor_pressure_hpa: np.ndarray) -> np.ndarray:
    """
    Brutsaert (1975) clear-sky atmospheric emissivity.
    
    Formula: ε_clear = 1.24 * (e_a / T_a)^(1/7)
    
    Args:
        air_temp_k: Air temperature in Kelvin
        vapor_pressure_hpa: Vapor pressure in hPa
        
    Returns:
        Clear-sky emissivity (dimensionless, 0-1)
    """

def estimate_cloud_fraction(
    measured_ghi: np.ndarray,
    solar_altitude_deg: np.ndarray,
    day_of_year: np.ndarray,
    latitude: float
) -> np.ndarray:
    """
    Derive cloud fraction from measured vs theoretical solar radiation.
    
    Uses Crawford-Duchon (1999) approach:
    cloud_fraction = 1 - (measured_solar / potential_clear_sky_solar)
    
    Args:
        measured_ghi: Measured global horizontal irradiance (W/m²)
        solar_altitude_deg: Sun altitude angle in degrees
        day_of_year: Day of year (1-366)
        latitude: Site latitude in degrees
        
    Returns:
        Cloud fraction (0=clear, 1=overcast)
    """

def calculate_clear_sky_ghi(
    solar_altitude_deg: np.ndarray,
    day_of_year: np.ndarray,
    latitude: float
) -> np.ndarray:
    """
    Estimate theoretical clear-sky global horizontal irradiance.
    
    Uses simplified Bird clear-sky model.
    """

def estimate_infrared_radiation(
    air_temp_c: np.ndarray,
    relative_humidity: np.ndarray,
    global_horizontal_rad: Optional[np.ndarray] = None,
    solar_altitude_deg: Optional[np.ndarray] = None,
    day_of_year: Optional[np.ndarray] = None,
    latitude: Optional[float] = None
) -> np.ndarray:
    """
    Estimate horizontal infrared radiation intensity.
    
    Full cloud-aware estimation when solar data provided,
    falls back to clear-sky assumption otherwise.
    
    Args:
        air_temp_c: Air temperature in Celsius
        relative_humidity: Relative humidity in percent
        global_horizontal_rad: Optional measured GHI for cloud detection
        solar_altitude_deg: Optional sun altitude for cloud calculation
        day_of_year: Optional DOY for clear-sky reference
        latitude: Optional latitude for clear-sky reference
        
    Returns:
        Horizontal infrared radiation intensity (W/m²)
        
    Note:
        For Beer Sheva in summer, clear-sky vs cloud-aware typically
        differs by <5% due to predominantly clear conditions.
    """
```


---

## Phase 2: Weather Provider Abstraction

### 2.1 Provider Protocol

**File:** `src/fast_utci/shared/weather_providers/__init__.py` (NEW package)

```python
"""Weather data provider abstraction for multi-source support."""

from typing import Protocol, Dict, Any, Optional
from datetime import datetime
import pandas as pd

class WeatherProvider(Protocol):
    """Protocol for weather data providers."""
    
    @property
    def provider_name(self) -> str:
        """Human-readable provider name."""
        ...
    
    @property
    def requires_api_key(self) -> bool:
        """Whether this provider requires authentication."""
        ...
    
    def fetch_data(
        self,
        latitude: float,
        longitude: float,
        start_date: datetime,
        end_date: datetime,
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch weather data for location and time range.
        
        Returns DataFrame with standardized columns:
        - datetime: timezone-aware datetime
        - air_temp: °C
        - wind_speed: m/s
        - relative_humidity: %
        - direct_normal_radiation: W/m² (if available)
        - diffuse_horizontal_radiation: W/m² (if available)
        - global_horizontal_radiation: W/m² (if available)
        """
        ...
    
    def get_station_location(self, station_id: Any) -> Dict[str, float]:
        """Get lat/lon/elevation for a station."""
        ...
```

### 2.2 IMS Provider Implementation

**File:** `src/fast_utci/shared/weather_providers/ims.py` (NEW)

```python
"""Israel Meteorological Service (IMS) weather data provider."""

class IMSProvider:
    """
    IMS API provider for Israeli weather stations.
    
    API Documentation: https://ims.gov.il/en/ObservationDataAPI
    
    Station pairs (basic weather + radiation):
    - Beer Sheva: stations 59 + 60
    - Tel Aviv: stations X + Y (future)
    - Haifa: stations X + Y (future)
    """
    
    BASE_URL = "https://api.ims.gov.il/v1/envista"
    
    # Station configurations for Israeli cities
    STATION_CONFIGS = {
        "beer_sheva": {
            "weather_station": 59,
            "radiation_station": 60,
            "location": {"lat": 31.2515, "lon": 34.7995, "elevation": 280, "timezone": "Asia/Jerusalem"}
        },
        # Future: add more cities
    }
    
    # Channel name to standardized field mapping
    CHANNEL_MAP = {
        "TD": "air_temp",
        "WS": "wind_speed", 
        "RH": "relative_humidity",
        "NIP": "direct_normal_radiation",
        "DiffR": "diffuse_horizontal_radiation",
        "Grad": "global_horizontal_radiation",
    }
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or get_api_key("ims")
        if not self.api_key:
            raise ValueError("IMS API key required. Set IMS_API_KEY env var.")
    
    def fetch_data(
        self,
        latitude: float,
        longitude: float,
        start_date: datetime,
        end_date: datetime,
        station_ids: Optional[List[int]] = None,
        resample_to_hourly: bool = True
    ) -> pd.DataFrame:
        """
        Fetch and combine data from weather + radiation stations.
        
        IMS provides 10-minute data; resampled to hourly by default.
        """
    
    def _fetch_station_data(self, station_id: int, from_date: str, to_date: str) -> pd.DataFrame:
        """Fetch raw data from single station."""
    
    def _merge_station_data(self, weather_df: pd.DataFrame, radiation_df: pd.DataFrame) -> pd.DataFrame:
        """Merge weather and radiation data by datetime."""
    
    def _resample_to_hourly(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resample 10-minute data to hourly averages."""
```

---

## Phase 3: APIAdapter Integration

### 3.1 APIAdapter Class

**File:** `src/fast_utci/shared/weather.py` (extend)

```python
class APIAdapter:
    """
    Adapter for API-sourced weather data.
    
    Implements WeatherDataSource protocol with automatic
    infrared radiation estimation.
    """
    
    def __init__(
        self,
        df_data: pd.DataFrame,
        location: Dict[str, float],  # lat, lon, elevation, timezone
        estimate_infrared: bool = True
    ):
        """
        Initialize API adapter.
        
        Args:
            df_data: DataFrame with standardized weather columns
            location: Location dict for solar calculations
            estimate_infrared: Whether to estimate IR (True for API data)
        """
        self.df_data = df_data.copy()
        self.location = location
        self._filter_indices = None
        
        if estimate_infrared:
            self._compute_infrared_radiation()
    
    def _compute_infrared_radiation(self):
        """Compute IR using Brutsaert-Crawford-Duchon method."""
        from fast_utci.shared.infrared import estimate_infrared_radiation
        from fast_utci.mrt.sun import compute_sun_positions  # Reuse existing
        
        # Get solar altitudes for cloud fraction calculation
        solar_data = compute_sun_positions(
            self.df_data['datetime'].tolist(),
            self.location['lat'],
            self.location['lon']
        )
        
        # Estimate IR with cloud awareness
        ir_values = estimate_infrared_radiation(
            air_temp_c=self.df_data['air_temp'].values,
            relative_humidity=self.df_data['relative_humidity'].values,
            global_horizontal_rad=self.df_data.get('global_horizontal_radiation', {}).values,
            solar_altitude_deg=solar_data.altitudes,
            day_of_year=self.df_data['datetime'].dt.dayofyear.values,
            latitude=self.location['lat']
        )
        
        self.df_data['horizontal_infrared_radiation_intensity'] = ir_values
    
    def get_infrared_radiation(self) -> np.ndarray:
        """Get estimated horizontal infrared radiation."""
        return self._apply_filter(
            self.df_data['horizontal_infrared_radiation_intensity'].values
        )
    
    # ... implement other WeatherDataSource methods ...
```

### 3.2 Update WeatherDataManager

**File:** `src/fast_utci/shared/weather.py` (extend)

```python
def load_weather_data(
    source: Union[str, Path, pd.DataFrame, Dict[str, Any]]
) -> WeatherDataSource:
    """
    Load weather data from various sources.
    
    Args:
        source: One of:
            - str/Path: EPW file path
            - DataFrame: Pre-loaded weather data
            - Dict: API configuration
                {
                    "provider": "ims",  # or "openweather", etc.
                    "latitude": 31.2515,
                    "longitude": 34.7995,
                    "start_date": "2024-08-15",
                    "end_date": "2024-08-15",
                    "api_key": "..." (optional, uses env var if missing)
                }
    
    Returns:
        WeatherDataSource adapter
    """
    if isinstance(source, dict):
        return _load_from_api(source)
    elif isinstance(source, pd.DataFrame):
        return DataFrameAdapter(source)
    else:
        return _load_from_epw(source)

def _load_from_api(config: Dict[str, Any]) -> APIAdapter:
    """Load weather data from API provider."""
    provider_name = config.get("provider", "ims")
    
    if provider_name == "ims":
        from fast_utci.shared.weather_providers.ims import IMSProvider
        provider = IMSProvider(api_key=config.get("api_key"))
    else:
        raise ValueError(f"Unknown weather provider: {provider_name}")
    
    df = provider.fetch_data(
        latitude=config["latitude"],
        longitude=config["longitude"],
        start_date=parse_date(config["start_date"]),
        end_date=parse_date(config["end_date"])
    )
    
    location = {
        "lat": config["latitude"],
        "lon": config["longitude"],
        "elevation": config.get("elevation", 0),
        "timezone": config.get("timezone", "UTC")
    }
    
    return APIAdapter(df, location)
```

---

## Phase 4: MRT Calculator Updates

### 4.1 Add Direct Location Setting

**File:** `src/fast_utci/mrt/mrt_calculator.py` (extend)

```python
def set_location(
    self,
    latitude: float,
    longitude: float,
    timezone: str = "UTC",
    elevation: float = 0.0
) -> None:
    """
    Set location directly without EPW file.
    
    Required for API-based weather data.
    
    Args:
        latitude: Decimal degrees (-90 to 90)
        longitude: Decimal degrees (-180 to 180)
        timezone: IANA timezone string (e.g., "Asia/Jerusalem")
        elevation: Meters above sea level
    """
    from ladybug.location import Location
    self.location = Location(
        city="API Location",
        latitude=latitude,
        longitude=longitude,
        time_zone=self._timezone_to_offset(timezone),
        elevation=elevation
    )

def set_location_from_epw(self, epw_file: Union[str, Path]) -> None:
    """Set location from EPW file (existing behavior)."""
    epw = EPW(str(epw_file))
    self.set_location(
        latitude=epw.location.latitude,
        longitude=epw.location.longitude,
        timezone=epw.location.time_zone,
        elevation=epw.location.elevation
    )
```

---

## Phase 5: Testing Strategy (TDD)

### 5.1 Unit Tests - Infrared Estimation

**File:** `tests/test_infrared.py` (NEW)

```python
"""Tests for infrared radiation estimation module."""

import pytest
import numpy as np
from fast_utci.shared.infrared import (
    estimate_vapor_pressure,
    calculate_clear_sky_emissivity,
    estimate_cloud_fraction,
    estimate_infrared_radiation
)

class TestVaporPressure:
    """Test Magnus-Tetens vapor pressure calculation."""
    
    def test_known_values(self):
        """Verify against published psychrometric tables."""
        # At 20°C, 50% RH, vapor pressure ≈ 11.7 hPa
        vp = estimate_vapor_pressure(
            air_temp_c=np.array([20.0]),
            relative_humidity=np.array([50.0])
        )
        assert 11.5 < vp[0] < 12.0
    
    def test_saturation(self):
        """At 100% RH, vapor pressure equals saturation pressure."""
        # At 30°C, saturation pressure ≈ 42.4 hPa
        vp = estimate_vapor_pressure(
            air_temp_c=np.array([30.0]),
            relative_humidity=np.array([100.0])
        )
        assert 42.0 < vp[0] < 43.0

class TestClearSkyEmissivity:
    """Test Brutsaert clear-sky emissivity."""
    
    def test_typical_range(self):
        """Emissivity should be 0.6-0.9 for typical conditions."""
        # Hot humid: T=35°C, RH=80%
        vp = estimate_vapor_pressure(np.array([35.0]), np.array([80.0]))
        eps = calculate_clear_sky_emissivity(np.array([308.15]), vp)
        assert 0.7 < eps[0] < 0.9
        
        # Hot dry: T=35°C, RH=20%
        vp = estimate_vapor_pressure(np.array([35.0]), np.array([20.0]))
        eps = calculate_clear_sky_emissivity(np.array([308.15]), vp)
        assert 0.6 < eps[0] < 0.75

class TestCloudFraction:
    """Test cloud detection from solar radiation."""
    
    def test_clear_sky(self):
        """High solar radiation → low cloud fraction."""
        # At solar noon in summer, GHI could be ~900 W/m²
        cf = estimate_cloud_fraction(
            measured_ghi=np.array([900.0]),
            solar_altitude_deg=np.array([70.0]),
            day_of_year=np.array([200]),
            latitude=31.25
        )
        assert cf[0] < 0.15
    
    def test_overcast(self):
        """Low solar radiation → high cloud fraction."""
        cf = estimate_cloud_fraction(
            measured_ghi=np.array([100.0]),
            solar_altitude_deg=np.array([70.0]),
            day_of_year=np.array([200]),
            latitude=31.25
        )
        assert cf[0] > 0.7
    
    def test_night(self):
        """Sun below horizon → cloud fraction 0 (no measurement possible)."""
        cf = estimate_cloud_fraction(
            measured_ghi=np.array([0.0]),
            solar_altitude_deg=np.array([-10.0]),
            day_of_year=np.array([200]),
            latitude=31.25
        )
        assert cf[0] == 0.0

class TestInfraredEstimation:
    """Test full infrared radiation estimation."""
    
    def test_typical_range(self):
        """IR should be 250-450 W/m² for typical conditions."""
        ir = estimate_infrared_radiation(
            air_temp_c=np.array([30.0]),
            relative_humidity=np.array([50.0])
        )
        assert 300 < ir[0] < 420
    
    def test_hot_dry_desert(self):
        """Beer Sheva summer conditions."""
        # T=38°C, RH=25%, clear sky
        ir = estimate_infrared_radiation(
            air_temp_c=np.array([38.0]),
            relative_humidity=np.array([25.0]),
            global_horizontal_rad=np.array([850.0]),
            solar_altitude_deg=np.array([75.0]),
            day_of_year=np.array([227]),  # Aug 15
            latitude=31.25
        )
        assert 320 < ir[0] < 400  # Clear sky, moderate humidity
    
    def test_cloud_correction_increases_ir(self):
        """Clouds should increase IR (warmer emitting layer)."""
        ir_clear = estimate_infrared_radiation(
            air_temp_c=np.array([25.0]),
            relative_humidity=np.array([50.0]),
            global_horizontal_rad=np.array([800.0]),  # Clear
            solar_altitude_deg=np.array([60.0]),
            day_of_year=np.array([200]),
            latitude=31.25
        )
        ir_cloudy = estimate_infrared_radiation(
            air_temp_c=np.array([25.0]),
            relative_humidity=np.array([50.0]),
            global_horizontal_rad=np.array([200.0]),  # Overcast
            solar_altitude_deg=np.array([60.0]),
            day_of_year=np.array([200]),
            latitude=31.25
        )
        assert ir_cloudy[0] > ir_clear[0]
```

### 5.2 Integration Tests - API Provider

**File:** `tests/test_ims_provider.py` (NEW)

```python
"""Integration tests for IMS weather provider."""

import pytest
from datetime import datetime
from fast_utci.shared.weather_providers.ims import IMSProvider

@pytest.fixture
def ims_provider():
    """Create IMS provider (skipped if no API key)."""
    import os
    api_key = os.environ.get("IMS_API_KEY")
    if not api_key:
        pytest.skip("IMS_API_KEY not set")
    return IMSProvider(api_key)

class TestIMSProvider:
    """Test IMS API integration."""
    
    def test_fetch_beer_sheva_data(self, ims_provider):
        """Fetch real data from Beer Sheva stations."""
        df = ims_provider.fetch_data(
            latitude=31.2515,
            longitude=34.7995,
            start_date=datetime(2024, 8, 15),
            end_date=datetime(2024, 8, 15)
        )
        
        # Should have ~24 hourly records
        assert 20 <= len(df) <= 25
        
        # Required columns present
        assert "air_temp" in df.columns
        assert "wind_speed" in df.columns
        assert "relative_humidity" in df.columns
        assert "direct_normal_radiation" in df.columns
        
    def test_data_ranges_reasonable(self, ims_provider):
        """Verify data values are in reasonable ranges."""
        df = ims_provider.fetch_data(
            latitude=31.2515,
            longitude=34.7995,
            start_date=datetime(2024, 8, 15),
            end_date=datetime(2024, 8, 15)
        )
        
        # August in Beer Sheva
        assert df["air_temp"].min() > 20  # Hot even at night
        assert df["air_temp"].max() < 45  # Not impossibly hot
        assert df["relative_humidity"].min() >= 0
        assert df["relative_humidity"].max() <= 100
```

### 5.3 Validation Tests - EPW Comparison

**File:** `tests/test_validation.py` (NEW)

```python
"""
Validation tests comparing API results against EPW baseline.

IMPORTANT: These tests compare "actual" weather (API) against "typical" weather (EPW).
Some difference is EXPECTED - the goal is to verify:
1. Processing pipeline works correctly
2. Infrared estimation produces reasonable values
3. UTCI calculations are in similar range (not identical)
"""

import pytest
import numpy as np
from pathlib import Path

class TestInfraredValidation:
    """Validate IR estimation against EPW measured values."""
    
    @pytest.fixture
    def epw_data(self):
        """Load Beer Sheva EPW file."""
        from ladybug.epw import EPW
        epw_path = Path("data/weather/ISR_D_Beer.Sheva.401900_TMYx/ISR_D_Beer.Sheva.401900_TMYx.epw")
        return EPW(str(epw_path))
    
    def test_estimated_vs_epw_infrared(self, epw_data):
        """
        Compare estimated IR against EPW-provided IR values.
        
        EPW files contain measured/modeled IR data. Our estimation
        should be within ±15% on average for the same conditions.
        """
        from fast_utci.shared.infrared import estimate_infrared_radiation
        
        # Extract August data from EPW
        temps = np.array(epw_data.dry_bulb_temperature.values)
        rh = np.array(epw_data.relative_humidity.values)
        ir_epw = np.array(epw_data.horizontal_infrared_radiation_intensity.values)
        ghi = np.array(epw_data.global_horizontal_radiation.values)
        
        # Filter to August (month 8)
        aug_start = 31+28+31+30+31+30+31  # DOY for Aug 1
        aug_end = aug_start + 31
        aug_slice = slice(aug_start*24, aug_end*24)
        
        # Estimate IR for August
        ir_estimated = estimate_infrared_radiation(
            air_temp_c=temps[aug_slice],
            relative_humidity=rh[aug_slice],
            global_horizontal_rad=ghi[aug_slice],
            # Note: simplified - real test would compute solar altitudes
        )
        
        ir_epw_aug = ir_epw[aug_slice]
        
        # Calculate mean absolute percentage error
        mape = np.mean(np.abs(ir_estimated - ir_epw_aug) / ir_epw_aug) * 100
        
        # Should be within 15% on average
        assert mape < 15, f"IR estimation MAPE {mape:.1f}% exceeds 15% threshold"
        
        # Correlation should be strong
        correlation = np.corrcoef(ir_estimated, ir_epw_aug)[0, 1]
        assert correlation > 0.8, f"IR correlation {correlation:.2f} below 0.8 threshold"
    
    def test_ir_range_matches_epw(self, epw_data):
        """Estimated IR should have similar min/max range as EPW."""
        from fast_utci.shared.infrared import estimate_infrared_radiation
        
        temps = np.array(epw_data.dry_bulb_temperature.values)
        rh = np.array(epw_data.relative_humidity.values)
        ir_epw = np.array(epw_data.horizontal_infrared_radiation_intensity.values)
        
        ir_estimated = estimate_infrared_radiation(temps, rh)
        
        # Ranges should be similar (within 10%)
        assert abs(ir_estimated.min() - ir_epw.min()) / ir_epw.min() < 0.15
        assert abs(ir_estimated.max() - ir_epw.max()) / ir_epw.max() < 0.15

class TestUTCIValidation:
    """
    Validate full UTCI pipeline with API data.
    
    Note: We cannot expect identical results between API and EPW because:
    1. EPW is "typical year" (statistical), API is "actual year"
    2. The specific day's weather may have been unusual
    
    We verify the pipeline works and produces reasonable results.
    """
    
    def test_api_utci_in_reasonable_range(self):
        """UTCI from API data should be in physically plausible range."""
        # This would be an integration test using actual API data
        # UTCI for Beer Sheva summer should be 30-50°C (very hot)
        pass
    
    def test_api_vs_epw_same_pipeline(self):
        """
        Verify API and EPW data go through identical processing.
        
        Use synthetic data to confirm pipeline consistency.
        """
        pass
```

---

## Phase 6: run_analysis.py Updates

### 6.1 Dual-Mode Analysis Support

**File:** `run_analysis.py` (extend)

```python
def run_analysis_core(
    month: int = DEFAULT_MONTH,
    day: int = DEFAULT_DAY,
    year: Optional[int] = None,  # NEW: year for API data
    grid_size: float = DEFAULT_GRID_SIZE,
    model_file: str = "data/3d_models/100_test.glb",
    weather_source: Union[str, Dict] = "data/weather/ISR_D_Beer.Sheva.401900_TMYx/ISR_D_Beer.Sheva.401900_TMYx.epw",
    # ... rest of parameters
) -> Dict[str, Any]:
    """
    Run full day UTCI analysis.
    
    Args:
        month, day: Analysis date
        year: If specified, uses API for actual weather. If None, uses weather_source.
        weather_source: EPW file path OR API config dict
        
    Weather Source Examples:
        # EPW file (typical year)
        weather_source="path/to/file.epw"
        
        # API config (actual weather)
        weather_source={
            "provider": "ims",
            "latitude": 31.2515,
            "longitude": 34.7995,
            "start_date": "2024-08-15",
            "end_date": "2024-08-15"
        }
        
        # Shorthand: just specify year (uses default Beer Sheva location)
        year=2024  # Fetches actual weather for Aug 15, 2024
    """
```

---

## Implementation Order & Checklist

### Phase 1: Infrastructure (Week 1)
- [ ] Create `src/fast_utci/shared/infrared.py` with full estimation logic
- [ ] Write `tests/test_infrared.py` (TDD - tests first!)
- [ ] Implement infrared estimation functions
- [ ] Verify tests pass

### Phase 2: Provider Abstraction (Week 1-2)
- [ ] Create `src/fast_utci/shared/weather_providers/` package
- [ ] Write `tests/test_ims_provider.py` 
- [ ] Implement IMSProvider class
- [ ] Add API key management to config

### Phase 3: Integration (Week 2)
- [ ] Create APIAdapter class in `weather.py`
- [ ] Update `load_weather_data()` for dict config
- [ ] Add `MRTCalculator.set_location()` method
- [ ] Write integration tests

### Phase 4: Validation (Week 2-3)
- [ ] Write validation tests comparing IR estimation vs EPW
- [ ] Run full pipeline with API data
- [ ] Document any discrepancies and acceptable thresholds

### Phase 5: Polish (Week 3)
- [ ] Update `run_analysis.py` with year parameter
- [ ] Create `.env.example` template
- [ ] Update README with API usage instructions
- [ ] Add CLI flags for API vs EPW mode

---

## Configuration Example

```toml
# fast_utci.toml

[api]
# Default weather provider
default_provider = "ims"

[api.ims]
# API key (prefer IMS_API_KEY environment variable)
# api_key = "your-key-here"

# Default stations for Beer Sheva
default_weather_station = 59
default_radiation_station = 60

[api.ims.locations]
# Pre-configured Israeli cities
beer_sheva = { weather = 59, radiation = 60, lat = 31.2515, lon = 34.7995 }
# tel_aviv = { weather = X, radiation = Y, lat = 32.0853, lon = 34.7818 }
```

```bash
# .env.example
IMS_API_KEY=your_ims_api_key_here
# OPENWEATHER_API_KEY=your_openweather_key_here  # Future
```

---

## Notes & Considerations

### Data Quality
- API returns 10-minute intervals → resample to hourly (mean for temps, max for radiation)
- Handle missing/invalid data: `status != 1` or `valid != true` → interpolate or skip
- Timezone: API returns `+03:00` (Israel) → convert to UTC internally

### Performance
- Cache API responses locally for repeated analyses of same date
- Consider rate limiting for bulk historical fetches

### Future Extensibility
- `WeatherProvider` protocol allows easy addition of:
  - OpenWeather API (global coverage)
  - Meteomatics (high-resolution)
  - NOAA (US coverage)
  - Local CSV/JSON files

### Validation Strategy
Since we're comparing "actual" (API) vs "typical" (EPW) weather:
1. **IR Estimation**: Compare against EPW IR values → should correlate 0.8+
2. **Full Pipeline**: Verify UTCI is in expected range (don't expect identical values)
3. **Sanity Checks**: Temperature/humidity within physical limits