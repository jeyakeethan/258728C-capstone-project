from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ShortagePeriod:
    start: pd.Timestamp
    end: pd.Timestamp
    fuel_energy_mwh: float


def interval_hours(frame: pd.DataFrame) -> float:
    """Return the representative interval duration in hours."""
    if len(frame) < 2:
        return 1.0
    ts = pd.to_datetime(frame["timestamp"], errors="coerce").dropna().sort_values()
    if len(ts) < 2:
        return 1.0
    dt = ts.diff().dropna().median().total_seconds() / 3600.0
    return float(dt) if dt > 0 else 1.0


def simulate_battery_and_fuel(
    frame: pd.DataFrame,
    *,
    battery_capacity_mwh: float,
    initial_battery_mwh: float,
    first_interval_hours: float | None = None,
) -> pd.DataFrame:
    """Chronologically simulate solar gain, demand loss, battery reserve and fuel-backed energy.

    Required columns: timestamp, expected_power_mw, forecast_load_mw.

    ``battery_reserve_mwh`` is the reserve available at the BEGINNING of each
    displayed interval. ``fuel_required_mwh`` is generator-backed electrical
    energy required within that interval after solar and battery are used.

    Battery charge/discharge efficiency and power limits are intentionally not
    modelled in this capstone version; they are documented assumptions.
    """
    if battery_capacity_mwh < 0:
        raise ValueError("battery_capacity_mwh must be >= 0")
    if initial_battery_mwh < 0:
        raise ValueError("initial_battery_mwh must be >= 0")

    out = frame.copy().sort_values("timestamp").reset_index(drop=True)
    if out.empty:
        for col in (
            "interval_hours", "solar_energy_mwh", "demand_energy_mwh",
            "battery_reserve_mwh", "battery_end_mwh", "fuel_required_mwh",
        ):
            out[col] = pd.Series(dtype=float)
        return out

    base_dt = interval_hours(out)
    durations = np.full(len(out), base_dt, dtype=float)
    if first_interval_hours is not None:
        durations[0] = float(np.clip(first_interval_hours, 0.0, base_dt))

    capacity = float(battery_capacity_mwh)
    battery = float(np.clip(initial_battery_mwh, 0.0, capacity))

    battery_before = []
    battery_after = []
    fuel_required = []
    solar_energy = []
    demand_energy = []

    for row, dt in zip(out.itertuples(index=False), durations):
        solar = max(0.0, float(getattr(row, "expected_power_mw"))) * dt
        demand = max(0.0, float(getattr(row, "forecast_load_mw"))) * dt
        battery_before.append(battery)
        solar_energy.append(solar)
        demand_energy.append(demand)

        net = solar - demand
        if net >= 0.0:
            battery = min(capacity, battery + net)
            fuel = 0.0
        else:
            deficit = -net
            from_battery = min(battery, deficit)
            battery -= from_battery
            fuel = deficit - from_battery

        # Avoid floating point noise in UI/status logic.
        if battery < 1e-9:
            battery = 0.0
        if fuel < 1e-9:
            fuel = 0.0

        battery_after.append(battery)
        fuel_required.append(fuel)

    out["interval_hours"] = durations
    out["solar_energy_mwh"] = solar_energy
    out["demand_energy_mwh"] = demand_energy
    out["battery_reserve_mwh"] = battery_before
    out["battery_end_mwh"] = battery_after  # internal state; not required in main UI
    out["fuel_required_mwh"] = fuel_required
    return out


def shortage_periods(frame: pd.DataFrame) -> list[ShortagePeriod]:
    """Group consecutive fuel-required hours into operator warning periods."""
    if frame.empty or "fuel_required_mwh" not in frame:
        return []

    periods: list[ShortagePeriod] = []
    active_start: pd.Timestamp | None = None
    active_end: pd.Timestamp | None = None
    active_fuel = 0.0

    for row in frame.itertuples(index=False):
        ts = pd.Timestamp(getattr(row, "timestamp"))
        fuel = max(0.0, float(getattr(row, "fuel_required_mwh")))
        dt = float(getattr(row, "interval_hours", 1.0))

        if fuel > 1e-9:
            if active_start is None:
                active_start = ts
                active_fuel = 0.0
            active_fuel += fuel
            active_end = ts + pd.Timedelta(hours=dt)
        elif active_start is not None:
            periods.append(ShortagePeriod(active_start, active_end or ts, active_fuel))
            active_start = None
            active_end = None
            active_fuel = 0.0

    if active_start is not None:
        periods.append(ShortagePeriod(active_start, active_end or active_start, active_fuel))

    return periods


def summarize_energy(frame: pd.DataFrame) -> dict[str, float]:
    """Aggregate an already simulated interval for simple daily presentation."""
    if frame.empty:
        return {
            "solar_energy_mwh": 0.0,
            "demand_energy_mwh": 0.0,
            "fuel_required_mwh": 0.0,
        }
    return {
        "solar_energy_mwh": float(frame["solar_energy_mwh"].sum()),
        "demand_energy_mwh": float(frame["demand_energy_mwh"].sum()),
        "fuel_required_mwh": float(frame["fuel_required_mwh"].sum()),
    }
