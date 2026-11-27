"""Reproducible exploratory analysis for the UNISOLAR solar dataset.

This script intentionally analyses only the four UNISOLAR source files:
generation, weather, irradiance, and site details. It does not read or use any
synthetic demand or demo-solar data.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "app" / "utils"))

from data import load_canonical_csv


SELECTED_FEATURES = [
    "Ghi",
    "CloudOpacity",
    "AirTemperature",
    "RelativeHumidity",
    "hour",
    "day_of_week",
    "day_of_year",
    "month",
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
]

CANDIDATE_FEATURES = [
    "ApparentTemperature",
    "AirTemperature",
    "DewPointTemperature",
    "RelativeHumidity",
    "WindSpeed",
    "WindDirection",
    "CloudOpacity",
    "Ghi",
    "hour",
    "day_of_week",
    "day_of_year",
    "month",
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
]

FEATURE_RATIONALES = {
    "Ghi": "Direct irradiance signal and the strongest physical driver of PV output.",
    "CloudOpacity": "Cloud proxy available from the deployment weather API.",
    "AirTemperature": "Available at deployment and related to PV operating conditions.",
    "RelativeHumidity": "Available at deployment and may capture atmospheric conditions.",
    "ApparentTemperature": "Historical weather candidate, but no direct deployment model input is used.",
    "DewPointTemperature": "Historical weather candidate, but no direct deployment model input is used.",
    "WindSpeed": "Historical candidate with high missingness and no selected deployment mapping.",
    "WindDirection": "Historical candidate with high missingness and no selected deployment mapping.",
    "hour": "Captures the solar-day operating pattern.",
    "day_of_week": "Retained as a calendar control; physical contribution should be tested.",
    "day_of_year": "Captures annual solar-position and seasonal effects.",
    "month": "Coarse annual-season indicator.",
    "hour_sin": "Continuous daily cycle representation.",
    "hour_cos": "Continuous daily cycle representation.",
    "doy_sin": "Continuous annual cycle representation.",
    "doy_cos": "Continuous annual cycle representation.",
}


def add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    timestamp = pd.to_datetime(out["timestamp"], errors="coerce")
    out["hour"] = timestamp.dt.hour + timestamp.dt.minute / 60.0
    out["day_of_week"] = timestamp.dt.dayofweek
    out["day_of_year"] = timestamp.dt.dayofyear
    out["month"] = timestamp.dt.month
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24.0)
    out["doy_sin"] = np.sin(2 * np.pi * out["day_of_year"] / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * out["day_of_year"] / 365.25)
    return out


def raw_file_summary(raw_dir: Path) -> pd.DataFrame:
    files = [
        "Solar_Generation_sub.csv",
        "Weather_data_sub.csv",
        "Solar_Irradiance.csv",
        "Solar_Site_Details.csv",
    ]
    rows = []
    for name in files:
        path = raw_dir / name
        frame = pd.read_csv(path)
        date_column = next(
            (column for column in ("Timestamp", "Timestamp_UTC", "timestamp") if column in frame),
            None,
        )
        date_min = date_max = None
        if date_column:
            dates = pd.to_datetime(frame[date_column], errors="coerce")
            date_min = dates.min()
            date_max = dates.max()
        rows.append(
            {
                "file": name,
                "rows": len(frame),
                "columns": len(frame.columns),
                "date_column": date_column or "",
                "date_min": date_min,
                "date_max": date_max,
                "missing_cells": int(frame.isna().sum().sum()),
            }
        )
    return pd.DataFrame(rows)


def feature_associations(frame: pd.DataFrame) -> pd.DataFrame:
    target = pd.to_numeric(frame["normalized_solar"], errors="coerce")
    rows = []
    for feature in CANDIDATE_FEATURES:
        if feature not in frame.columns:
            continue
        pair = pd.DataFrame(
            {
                "feature": pd.to_numeric(frame[feature], errors="coerce"),
                "target": target,
            }
        ).dropna()
        if len(pair) < 10 or pair["feature"].nunique() < 2:
            continue
        mi = float(
            mutual_info_regression(
                pair[["feature"]],
                pair["target"],
                random_state=42,
                n_neighbors=min(3, len(pair) - 1),
            )[0]
        )
        rows.append(
            {
                "feature": feature,
                "observations": len(pair),
                "pearson_correlation": pair["feature"].corr(pair["target"], method="pearson"),
                "spearman_correlation": pair["feature"].corr(pair["target"], method="spearman"),
                "mutual_information": mi,
                "deployment_selected": feature in SELECTED_FEATURES,
                "rationale": FEATURE_RATIONALES.get(feature, "Calendar or deployment feature."),
            }
        )
    return pd.DataFrame(rows).sort_values("mutual_information", ascending=False)


def create_figures(frame: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))
    plt.hist(frame["normalized_solar"].clip(lower=0), bins=40, color="#2f75b5", alpha=0.85)
    plt.xlabel("Solar generation / installed capacity")
    plt.ylabel("Observations")
    plt.title("UNISOLAR normalized solar-generation distribution")
    plt.tight_layout()
    plt.savefig(out_dir / "solar_generation_distribution.png", dpi=160)
    plt.close()

    hourly = frame.groupby("hour", as_index=False)["normalized_solar"].mean()
    plt.figure(figsize=(9, 5))
    plt.plot(hourly["hour"], hourly["normalized_solar"], marker="o", color="#2f75b5")
    plt.xlabel("Hour of day")
    plt.ylabel("Mean normalized solar generation")
    plt.title("UNISOLAR generation profile by hour")
    plt.xticks(range(0, 24, 2))
    plt.tight_layout()
    plt.savefig(out_dir / "solar_generation_by_hour.png", dpi=160)
    plt.close()

    missing = frame.isna().mean().sort_values(ascending=False)
    missing = missing[missing > 0]
    plt.figure(figsize=(9, 5))
    if missing.empty:
        plt.text(0.5, 0.5, "No missing values remain in the processed dataset", ha="center", va="center")
        plt.axis("off")
    else:
        plt.bar(missing.index, missing.values, color="#c0504d")
        plt.ylabel("Missing fraction")
        plt.xticks(rotation=45, ha="right")
        plt.title("Processed UNISOLAR missingness")
    plt.tight_layout()
    plt.savefig(out_dir / "processed_missingness.png", dpi=160)
    plt.close()

    plot_features = [feature for feature in SELECTED_FEATURES[:4] if feature in frame.columns]
    sample = frame.sample(min(5000, len(frame)), random_state=42)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for axis, feature in zip(axes.flat, plot_features):
        axis.scatter(sample[feature], sample["normalized_solar"], s=5, alpha=0.18, color="#2f75b5")
        axis.set_xlabel(feature)
        axis.set_ylabel("Normalized solar generation")
        axis.set_title(f"{feature} versus target")
    for axis in axes.flat[len(plot_features):]:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "feature_relationships.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run UNISOLAR-only exploratory analysis.")
    parser.add_argument("--raw", type=Path, default=ROOT / "data/raw/unisolar")
    parser.add_argument("--processed", type=Path, default=ROOT / "data/processed/unisolar.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "reports/unisolar_eda")
    parser.add_argument("--refresh-processed", action="store_true")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    summary = raw_file_summary(args.raw)
    summary.to_csv(args.out / "raw_file_summary.csv", index=False)

    if args.refresh_processed or not args.processed.exists():
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "workflows" / "prepare_unisolar.py"),
                "--raw",
                str(args.raw),
                "--out",
                str(args.processed),
            ],
            check=True,
        )

    frame = load_canonical_csv(args.processed)
    frame = add_time_features(frame)
    capacity = pd.to_numeric(frame["site_capacity_kwp"], errors="coerce")
    solar_power = pd.to_numeric(frame["solar_power"], errors="coerce")
    frame["normalized_solar"] = solar_power / capacity

    frame.isna().sum().rename("missing_count").to_frame().assign(
        missing_fraction=lambda table: table["missing_count"] / len(frame)
    ).to_csv(args.out / "processed_missingness.csv")

    associations = feature_associations(frame)
    associations.to_csv(args.out / "feature_associations.csv", index=False)

    frame.groupby("hour", as_index=False).agg(
        mean_solar_power=("solar_power", "mean"),
        mean_normalized_solar=("normalized_solar", "mean"),
        observations=("solar_power", "size"),
    ).to_csv(args.out / "hourly_profile.csv", index=False)
    frame.groupby("month", as_index=False).agg(
        mean_solar_power=("solar_power", "mean"),
        mean_normalized_solar=("normalized_solar", "mean"),
        observations=("solar_power", "size"),
    ).to_csv(args.out / "monthly_profile.csv", index=False)

    create_figures(frame, args.out)

    metadata = {
        "dataset_scope": "UNISOLAR source files only",
        "processed_rows": len(frame),
        "processed_columns": list(frame.columns),
        "target": "normalized_solar = solar_power / site_capacity_kwp",
        "selected_features": SELECTED_FEATURES,
        "candidate_features": CANDIDATE_FEATURES,
        "association_note": "Pearson, Spearman, and mutual information describe association; they do not establish causation.",
        "outputs": sorted(path.name for path in args.out.iterdir()),
    }
    (args.out / "eda_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    print(json.dumps(metadata, indent=2, default=str))


if __name__ == "__main__":
    main()
