from __future__ import annotations

import numpy as np
import pandas as pd


def _cyclic_interpolate(values: np.ndarray, position: float) -> float:
    if len(values) == 0:
        return 0.0
    pos = float(position) % len(values)
    lo = int(np.floor(pos))
    hi = (lo + 1) % len(values)
    frac = pos - lo
    return float(values[lo] * (1.0 - frac) + values[hi] * frac)


def synthetic_live_values(
    solar_data: pd.DataFrame,
    demand_data: pd.DataFrame,
    local_now: pd.Timestamp,
    installed_capacity_mw: float,
) -> tuple[float, float]:
    """Return smooth synthetic present solar and demand values.

    The source datasets are cycled as simulation traces. Five-second dashboard
    heartbeats interpolate between hourly records instead of generating random
    jumps. These values are demonstration telemetry, not project evaluation data.
    """
    seconds_into_hour = local_now.minute * 60 + local_now.second + local_now.microsecond / 1e6
    frac = seconds_into_hour / 3600.0

    # Solar demo data is an arbitrary 0-100 synthetic trace; treat it as a percentage
    # of rated site capacity in MW when adapting it to the configured deployment site.
    solar_values = pd.to_numeric(solar_data["solar_power"], errors="coerce").fillna(0.0).to_numpy(float)
    solar_hour_index = ((local_now.dayofyear - 1) * 24 + local_now.hour) % max(len(solar_values), 1)
    solar_base_mw = _cyclic_interpolate(solar_values, solar_hour_index + frac)
    solar_mw = max(0.0, solar_base_mw / 100.0 * float(installed_capacity_mw))

    demand_values = pd.to_numeric(demand_data["site_load_kw"], errors="coerce").fillna(0.0).to_numpy(float)
    demand_hour_index = ((local_now.dayofyear - 1) * 24 + local_now.hour) % max(len(demand_values), 1)
    demand_mw = max(0.0, _cyclic_interpolate(demand_values, demand_hour_index + frac))
    return solar_mw, demand_mw


def estimate_present_battery(
    solar_data: pd.DataFrame,
    demand_data: pd.DataFrame,
    local_now: pd.Timestamp,
    installed_capacity_mw: float,
    battery_capacity_mwh: float,
    initial_fraction: float = 0.5,
) -> float:
    """Estimate current reserve by replaying today's synthetic site state.

    Because no physical battery telemetry is available, the demonstration starts
    each local day at a documented fraction of usable capacity and synchronizes
    the estimate from midnight to the current heartbeat.
    """
    capacity = max(0.0, float(battery_capacity_mwh))
    battery = capacity * float(np.clip(initial_fraction, 0.0, 1.0))
    if capacity <= 0.0:
        return 0.0

    midnight = local_now.normalize()
    for hour in range(local_now.hour + 1):
        ts = midnight + pd.Timedelta(hours=hour)
        solar_mw, demand_mw = synthetic_live_values(solar_data, demand_data, ts, installed_capacity_mw)
        duration = 1.0
        if hour == local_now.hour:
            duration = (local_now.minute * 60 + local_now.second + local_now.microsecond / 1e6) / 3600.0
        net_mwh = (solar_mw - demand_mw) * duration
        battery = float(np.clip(battery + net_mwh, 0.0, capacity))
    return battery
