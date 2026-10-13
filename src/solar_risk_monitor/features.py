import numpy as np
import pandas as pd


TIME_FEATURES = [
    "hour", "day_of_week", "day_of_year", "month",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
]

DATASET_EXOGENOUS = {
    "unisolar": ["Ghi", "CloudOpacity", "AirTemperature", "RelativeHumidity"],
    "demo": ["irradiance", "cloud_cover", "temperature", "humidity"],
}



def infer_step_hours(df: pd.DataFrame) -> float:
    """Infer the median sampling interval in hours from timestamp."""
    ts = pd.to_datetime(df["timestamp"], errors="coerce").dropna().sort_values()
    if len(ts) < 2:
        return 1.0
    delta = ts.diff().dropna().median()
    if pd.isna(delta) or delta.total_seconds() <= 0:
        return 1.0
    return float(delta.total_seconds() / 3600.0)



def hours_to_rows(hours: float, step_hours: float) -> int:
    """Convert a semantic duration in hours into dataset rows."""
    if step_hours <= 0:
        step_hours = 1.0
    return max(1, int(round(float(hours) / step_hours)))



def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ts = pd.to_datetime(out["timestamp"], errors="coerce")
    out["hour"] = ts.dt.hour + ts.dt.minute / 60.0
    out["day_of_week"] = ts.dt.dayofweek
    out["day_of_year"] = ts.dt.dayofyear
    out["month"] = ts.dt.month
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24.0)
    out["doy_sin"] = np.sin(2 * np.pi * out["day_of_year"] / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * out["day_of_year"] / 365.25)
    return out



def add_lag_features(
    df: pd.DataFrame,
    target: str = "solar_power",
    lags=(1, 2, 3, 6, 12, 24),
    rolling_windows=(3, 6, 12, 24),
) -> pd.DataFrame:
    """Add target-history features where suffixes represent hours, not rows."""
    out = df.copy()
    step_hours = infer_step_hours(out)

    for lag_h in lags:
        rows = hours_to_rows(lag_h, step_hours)
        out[f"{target}_lag_{lag_h}"] = out[target].shift(rows)

    shifted = out[target].shift(1)
    for window_h in rolling_windows:
        rows = hours_to_rows(window_h, step_hours)
        out[f"{target}_roll_mean_{window_h}"] = shifted.rolling(rows, min_periods=rows).mean()
        out[f"{target}_roll_std_{window_h}"] = shifted.rolling(rows, min_periods=rows).std()

    return out



def engineered_history_columns(
    target="solar_power",
    lags=(1, 2, 3, 6, 12, 24),
    rolling_windows=(3, 6, 12, 24),
):
    cols = [f"{target}_lag_{h}" for h in lags]
    for h in rolling_windows:
        cols.extend([f"{target}_roll_mean_{h}", f"{target}_roll_std_{h}"])
    return cols



def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Build time/history features while preserving missing exogenous weather values."""
    out = add_time_features(df)
    out = add_lag_features(out)
    required = ["timestamp", "solar_power"] + engineered_history_columns()
    return out.dropna(subset=required).reset_index(drop=True)



def _base_exclusions():
    return {
        "timestamp", "solar_power", "solar_capacity_ratio", "site_capacity_kwp",
        "latitude", "longitude", "site_id", "campus_id", "ZONEID",
    }



def numeric_feature_columns(df: pd.DataFrame):
    """Generic leakage-safe numeric columns; retained for compatibility/tests."""
    excluded = _base_exclusions()
    return [
        c for c in df.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
    ]



def model_feature_columns(df: pd.DataFrame, dataset_name: str):
    """Return explicit leakage-safe features for supported solar datasets.

    The active capstone path is UNISOLAR. This compatibility helper retains
    optional history features for offline experiments, while the deployed
    weather-only model uses ``weather_only_feature_columns`` below.
    """
    name = (dataset_name or "").strip().lower().replace(" ", "_")
    allowed = DATASET_EXOGENOUS.get(name)
    if allowed is None:
        return numeric_feature_columns(df)
    exogenous = [c for c in allowed if c in df.columns]
    missing_flags = [f"{c}_was_missing" for c in exogenous if f"{c}_was_missing" in df.columns]
    history = engineered_history_columns()
    selected = exogenous + missing_flags + [c for c in TIME_FEATURES if c in df.columns] + [
        c for c in history if c in df.columns
    ]
    return list(dict.fromkeys(
        c for c in selected if pd.api.types.is_numeric_dtype(df[c])
    ))



def required_history_rows(df: pd.DataFrame, max_history_hours: float = 24.0) -> int:
    return hours_to_rows(max_history_hours, infer_step_hours(df))


def weather_only_feature_columns(df: pd.DataFrame):
    """UNISOLAR deployment features available from a weather forecast API.

    Target-history lags are intentionally excluded so a future hour can be
    predicted without accidentally using future observed solar generation.
    """
    exogenous = [
        c for c in DATASET_EXOGENOUS["unisolar"]
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]
    return exogenous + [
        c for c in TIME_FEATURES
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]
