from __future__ import annotations

import numpy as np
import pandas as pd


def varied_prediction(
    value: float,
    timestamp: pd.Timestamp,
    variation_fraction: float = 0.05,
    phase_offset: float = 0.0,
) -> float:
    """Return a smooth, deterministic display value within ±5% of a prediction."""
    if not 0.0 <= variation_fraction <= 1.0:
        raise ValueError("variation_fraction must be between 0 and 1")

    base_value = max(0.0, float(value))
    if base_value == 0.0:
        return 0.0

    current_time = pd.Timestamp(timestamp)
    if pd.isna(current_time):
        raise ValueError("timestamp must be a valid timestamp")

    phase = current_time.timestamp() / (10.0 * 60.0) + float(phase_offset)
    multiplier = 1.0 + float(variation_fraction) * np.sin(phase)
    return max(0.0, base_value * multiplier)


def estimate_present_battery(
    forecast_frame: pd.DataFrame,
    local_now: pd.Timestamp,
    battery_capacity_mwh: float,
    initial_fraction: float = 0.5,
) -> float:
    """Estimate current battery reserve from the forecast frame."""
    required = {"timestamp", "expected_power_mw", "forecast_load_mw"}
    missing = required - set(forecast_frame.columns)
    if missing:
        raise ValueError(f"Forecast frame is missing required columns: {sorted(missing)}")

    capacity = max(0.0, float(battery_capacity_mwh))
    battery = capacity * float(np.clip(initial_fraction, 0.0, 1.0))
    if capacity <= 0.0:
        return 0.0

    forecast = forecast_frame.copy()
    forecast["timestamp"] = pd.to_datetime(forecast["timestamp"], errors="coerce")
    forecast = forecast.dropna(subset=["timestamp"]).sort_values("timestamp")
    if forecast.empty:
        return battery

    midnight = local_now.normalize()
    for hour in range(local_now.hour + 1):
        ts = midnight + pd.Timedelta(hours=hour)
        nearest_idx = (forecast["timestamp"] - ts).abs().idxmin()
        row = forecast.loc[nearest_idx]
        solar_mw = max(0.0, float(row["expected_power_mw"]))
        demand_mw = max(0.0, float(row["forecast_load_mw"]))
        duration = 1.0
        if hour == local_now.hour:
            duration = (local_now.minute * 60 + local_now.second + local_now.microsecond / 1e6) / 3600.0
        net_mwh = (solar_mw - demand_mw) * duration
        battery = float(np.clip(battery + net_mwh, 0.0, capacity))
    return battery
