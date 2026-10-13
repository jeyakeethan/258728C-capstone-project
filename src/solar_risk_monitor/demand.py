from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd

from .models import evaluate, make_model, save_bundle


GENERIC_HOLIDAYS = {(1, 1), (5, 1), (12, 25), (12, 26)}
DEMAND_FEATURES = ["hour", "hour_sin", "hour_cos", "is_holiday", "day_of_week", "month", "day_of_year"]


def add_demand_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ts = pd.to_datetime(out["timestamp"], errors="coerce")
    # Fractional hour preserves minute/second information at inference time.
    # Historical training observations may still be hourly; deployment can
    # evaluate the trained model for the exact current timestamp.
    out["hour"] = (
        ts.dt.hour
        + ts.dt.minute / 60.0
        + ts.dt.second / 3600.0
        + ts.dt.microsecond / 3_600_000_000.0
    )
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24.0)
    out["day_of_week"] = ts.dt.dayofweek
    out["month"] = ts.dt.month
    out["day_of_year"] = ts.dt.dayofyear
    out["is_holiday"] = [
        int((t.month, t.day) in GENERIC_HOLIDAYS) if not pd.isna(t) else 0
        for t in ts
    ]
    return out


def _hourly_base_load(hour: int, base_kw: float) -> float:
    """Smooth, intentionally simple commercial-site load profile."""
    anchors_h = np.array([0, 4, 6, 8, 10, 12, 14, 16, 18, 20, 23, 24], dtype=float)
    anchors_ratio = np.array([0.54, 0.51, 0.60, 0.76, 0.92, 1.00, 0.97, 0.88, 0.74, 0.64, 0.56, 0.54])
    return float(np.interp(float(hour), anchors_h, anchors_ratio) * base_kw)


def generate_synthetic_demand(
    start: str = "2024-01-01 00:00",
    end: str = "2025-12-31 23:00",
    base_peak_kw: float = 120.0,
    noise_fraction: float = 0.025,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a transparent synthetic historical site-demand dataset.

    Historical samples are stored hourly for a compact, reproducible training
    series. The deployed model can still be evaluated for an exact timestamp,
    including minutes and seconds, through continuous time features.

    The series is deliberately smooth: time-of-day is dominant, weekends and
    holidays reduce demand, and month/day-of-year add a mild seasonal effect.
    Random noise is small and reproducible.
    """
    ts = pd.date_range(start=start, end=end, freq="h")
    frame = add_demand_calendar_features(pd.DataFrame({"timestamp": ts}))
    rng = np.random.default_rng(seed)

    load = []
    for row in frame.itertuples(index=False):
        value = _hourly_base_load(float(row.hour), float(base_peak_kw))

        # Reduced activity on weekends and generic holidays.
        if row.day_of_week >= 5:
            value *= 0.82
        if row.is_holiday:
            value *= 0.68

        # Mild smooth annual pattern, intentionally smaller than hourly variation.
        seasonal = 1.0 + 0.05 * np.cos(2 * np.pi * (row.day_of_year - 25) / 365.25)
        value *= seasonal

        # Low-amplitude noise avoids an unrealistically deterministic series.
        value *= 1.0 + rng.normal(0.0, noise_fraction)
        load.append(max(0.0, value))

    frame["site_load_kw"] = np.round(load, 3)
    return frame[["timestamp", *DEMAND_FEATURES, "site_load_kw"]]


def chronological_demand_split(df: pd.DataFrame, train_ratio: float = 0.70, val_ratio: float = 0.15):
    ordered = df.sort_values("timestamp").reset_index(drop=True)
    n = len(ordered)
    t1 = int(n * train_ratio)
    t2 = int(n * (train_ratio + val_ratio))
    return ordered.iloc[:t1].copy(), ordered.iloc[t1:t2].copy(), ordered.iloc[t2:].copy()


def train_demand_forecaster(df: pd.DataFrame, out_path: Path):
    train, val, test = chronological_demand_split(df)
    results = []
    for name in ["linear", "hgb"]:
        model = make_model(name)
        model.fit(train[DEMAND_FEATURES], train["site_load_kw"])
        result = evaluate(model, val[DEMAND_FEATURES], val["site_load_kw"], name)
        results.append(result)

    best = min(results, key=lambda r: r.mae)
    train_val = pd.concat([train, val], ignore_index=True)
    model = make_model(best.model_name)
    model.fit(train_val[DEMAND_FEATURES], train_val["site_load_kw"])
    test_result = evaluate(model, test[DEMAND_FEATURES], test["site_load_kw"], best.model_name)

    metadata = {
        "dataset_name": "synthetic_historical_site_demand",
        "selected_model": best.model_name,
        "model_purpose": "estimate site demand for a supplied timestamp from historical calendar/time patterns",
        "feature_columns": DEMAND_FEATURES,
        "target": "site_load_kw",
        "validation": best.__dict__,
        "test": test_result.__dict__,
        "synthetic_data_notice": (
            "This synthetic historical site-demand dataset is a scenario/demo training component "
            "and must not be presented as measured site consumption data. Deployment inference "
            "is evaluated for the supplied current timestamp, including minute/second information."
        ),
    }
    save_bundle(model, DEMAND_FEATURES, out_path, metadata=metadata)
    out_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def future_demand_features(timestamps: pd.Series | pd.DatetimeIndex) -> pd.DataFrame:
    return add_demand_calendar_features(pd.DataFrame({"timestamp": pd.to_datetime(timestamps)}))
