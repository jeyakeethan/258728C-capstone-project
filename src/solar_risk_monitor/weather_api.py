from __future__ import annotations

import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
NOMINATIM_REVERSE_URL = "https://nominatim.openstreetmap.org/reverse"
USER_AGENT = "solar-fuel-readiness-capstone/12.0 (academic project)"

WMO_LABELS = {
    0: ("Clear sky", "☀️"),
    1: ("Mainly clear", "🌤️"), 2: ("Partly cloudy", "⛅"), 3: ("Overcast", "☁️"),
    45: ("Fog", "🌫️"), 48: ("Rime fog", "🌫️"),
    51: ("Light drizzle", "🌦️"), 53: ("Drizzle", "🌦️"), 55: ("Heavy drizzle", "🌧️"),
    61: ("Light rain", "🌦️"), 63: ("Rain", "🌧️"), 65: ("Heavy rain", "🌧️"),
    71: ("Light snow", "🌨️"), 73: ("Snow", "🌨️"), 75: ("Heavy snow", "❄️"),
    80: ("Rain showers", "🌦️"), 81: ("Rain showers", "🌧️"), 82: ("Heavy showers", "🌧️"),
    95: ("Thunderstorm", "⛈️"), 96: ("Thunderstorm", "⛈️"), 99: ("Thunderstorm", "⛈️"),
}


def describe_weather_code(code) -> tuple[str, str]:
    try:
        code = int(code)
    except (TypeError, ValueError):
        return "Current weather", "🌤️"
    return WMO_LABELS.get(code, ("Current weather", "🌤️"))


def reverse_geocode(latitude: float, longitude: float) -> str:
    """Best-effort coordinate -> human-readable location using Nominatim."""
    params = {"lat": float(latitude), "lon": float(longitude), "format": "jsonv2", "zoom": 12, "addressdetails": 1}
    req = Request(f"{NOMINATIM_REVERSE_URL}?{urlencode(params)}", headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=10) as response:
        payload = json.load(response)
    address = payload.get("address") or {}
    locality = (
        address.get("suburb") or address.get("city_district") or address.get("city")
        or address.get("town") or address.get("village") or address.get("municipality")
    )
    region = address.get("state") or address.get("region")
    country = address.get("country")
    parts = []
    for value in (locality, region, country):
        if value and value not in parts:
            parts.append(value)
    return ", ".join(parts) or payload.get("display_name") or f"{latitude:.4f}, {longitude:.4f}"


def fetch_open_meteo_hourly(latitude: float, longitude: float, forecast_days: int = 8) -> pd.DataFrame:
    """Fetch current + hourly weather and map it to UNISOLAR-compatible inputs."""
    fields = ["shortwave_radiation", "cloud_cover", "temperature_2m", "relative_humidity_2m"]
    params = {
        "latitude": float(latitude),
        "longitude": float(longitude),
        "current": ",".join([*fields, "weather_code"]),
        "hourly": ",".join([*fields, "weather_code"]),
        "forecast_days": int(forecast_days),
        "timezone": "auto",
    }
    req = Request(f"{OPEN_METEO_URL}?{urlencode(params)}", headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=15) as response:
        payload = json.load(response)

    hourly = payload.get("hourly") or {}
    time_values = hourly.get("time") or []
    if not time_values:
        raise RuntimeError("Weather API returned no hourly forecast values.")

    frame = pd.DataFrame({
        "timestamp": pd.to_datetime(time_values, errors="coerce"),
        "Ghi": pd.to_numeric(hourly.get("shortwave_radiation"), errors="coerce"),
        "CloudOpacity": pd.to_numeric(hourly.get("cloud_cover"), errors="coerce"),
        "AirTemperature": pd.to_numeric(hourly.get("temperature_2m"), errors="coerce"),
        "RelativeHumidity": pd.to_numeric(hourly.get("relative_humidity_2m"), errors="coerce"),
        "WeatherCode": pd.to_numeric(hourly.get("weather_code"), errors="coerce"),
    })
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    frame.attrs["timezone"] = payload.get("timezone")
    frame.attrs["timezone_abbreviation"] = payload.get("timezone_abbreviation")
    frame.attrs["utc_offset_seconds"] = payload.get("utc_offset_seconds")
    frame.attrs["current"] = payload.get("current") or {}
    return frame
