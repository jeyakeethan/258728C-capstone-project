from pathlib import Path
import argparse
from datetime import timedelta
import json
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "app" / "utils"))

from data import load_canonical_csv
from features import add_time_features, weather_only_feature_columns
from models import evaluate, make_model, metrics_from_predictions, save_bundle


def day_grouped_split(frame: pd.DataFrame, train_ratio=0.70, val_ratio=0.15, random_state=42):
    """Split records by calendar date so all sites and intervals on a day stay in one partition."""
    timestamps = pd.to_datetime(frame["timestamp"])
    groups = timestamps.dt.date
    if groups.nunique() < 10:
        raise ValueError("Not enough unique days for day-grouped train/validation/test splitting.")

    test_ratio = 1.0 - train_ratio - val_ratio
    if train_ratio <= 0 or val_ratio <= 0 or test_ratio <= 0:
        raise ValueError("train_ratio, val_ratio, and implied test_ratio must all be positive.")

    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=random_state)
    train_val_idx, test_idx = next(gss_test.split(frame, groups=groups))
    train_val = frame.iloc[train_val_idx].copy()
    test = frame.iloc[test_idx].copy()

    val_share_of_remaining = val_ratio / (train_ratio + val_ratio)
    train_val_groups = pd.to_datetime(train_val["timestamp"]).dt.date
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_share_of_remaining, random_state=random_state)
    train_idx, val_idx = next(gss_val.split(train_val, groups=train_val_groups))
    train = train_val.iloc[train_idx].copy()
    val = train_val.iloc[val_idx].copy()

    sort_cols = ["timestamp", "site_id"] if "site_id" in frame.columns else ["timestamp"]
    train = train.sort_values(sort_cols).reset_index(drop=True)
    val = val.sort_values(sort_cols).reset_index(drop=True)
    test = test.sort_values(sort_cols).reset_index(drop=True)
    return train, val, test


def partition_summary(part: pd.DataFrame, total_rows: int) -> dict:
    """Summarize row count, share, day count, and timestamp span for a split partition."""
    ts = pd.to_datetime(part["timestamp"])
    return {
        "records": int(len(part)),
        "record_share": round(len(part) / total_rows, 4) if total_rows else 0.0,
        "unique_days": int(ts.dt.date.nunique()),
        "months_covered": int(ts.dt.month.nunique()),
        "start_timestamp": str(ts.min()),
        "end_timestamp": str(ts.max()),
    }


def per_site_metrics(model, test, features):
    if "site_id" not in test.columns:
        return {}
    result = {}
    for site, group in test.groupby("site_id"):
        if len(group) < 2:
            continue
        pred = model.predict(group[features])
        m = metrics_from_predictions(group["normalized_solar"], pred, "selected")
        result[str(site)] = m.__dict__
    return result


parser = argparse.ArgumentParser(
    description="Train a generalized UNISOLAR multi-site weather model for current/future API inference."
)
parser.add_argument("--data", default=str(ROOT / "data/processed/unisolar.csv"))
parser.add_argument("--out", default=str(ROOT / "models/unisolar_weather_forecaster.joblib"))
args = parser.parse_args()

raw = load_canonical_csv(args.data)
if "site_capacity_kwp" not in raw.columns:
    raise ValueError("UNISOLAR processed data must contain site_capacity_kwp metadata.")

frame = add_time_features(raw)
capacity = pd.to_numeric(frame["site_capacity_kwp"], errors="coerce")
frame = frame.loc[capacity > 0].copy()
frame["normalized_solar"] = (
    pd.to_numeric(frame["solar_power"], errors="coerce") /
    pd.to_numeric(frame["site_capacity_kwp"], errors="coerce")
)
frame = frame.replace([np.inf, -np.inf], np.nan).dropna(
    subset=["timestamp", "normalized_solar"]
).sort_values(["timestamp", "site_id"] if "site_id" in frame.columns else ["timestamp"]).reset_index(drop=True)

features = weather_only_feature_columns(frame)
if not features:
    raise ValueError("No weather/time features were found in the UNISOLAR processed data.")

train, val, test = day_grouped_split(frame, train_ratio=0.70, val_ratio=0.15, random_state=42)
train_val = pd.concat([train, val], ignore_index=True)
split_summary = {
    "total": partition_summary(frame, len(frame)),
    "train": partition_summary(train, len(frame)),
    "validation": partition_summary(val, len(frame)),
    "test": partition_summary(test, len(frame)),
    "train_val": partition_summary(train_val, len(frame)),
}
print(f"DAY-GROUPED SPLIT SUMMARY: {json.dumps(split_summary, indent=2)}")

hgb_grid = [
    {"learning_rate": lr, "max_iter": it, "max_leaf_nodes": leaves,
     "min_samples_leaf": min_leaf, "l2_regularization": l2}
    for lr in (0.03, 0.05, 0.10)
    for it in (200, 300)
    for leaves in (15, 31)
    for min_leaf in (20, 40)
    for l2 in (0.0, 0.1)
]
hgb_trials = []
for params in hgb_grid:
    candidate = make_model("hgb", hgb_params=params)
    candidate.fit(train[features], train["normalized_solar"])
    result = evaluate(candidate, val[features], val["normalized_solar"], "hgb")
    hgb_trials.append({"params": params, "metrics": result})

best_hgb_trial = min(hgb_trials, key=lambda t: (t["metrics"].mae, t["metrics"].rmse))
best_hgb_params = best_hgb_trial["params"]
hgb_result = best_hgb_trial["metrics"]
print(f"BEST HGB PARAMS: {json.dumps(best_hgb_params, indent=2)}")
print(f"VALIDATION HGB: {json.dumps(hgb_result.__dict__, indent=2)}")

model = make_model("hgb", hgb_params=best_hgb_params)
model.fit(train_val[features], train_val["normalized_solar"])
test_result = evaluate(model, test[features], test["normalized_solar"], "hgb")
site_results = per_site_metrics(model, test, features)

end_cutoff = pd.to_datetime(test["timestamp"]).max() - timedelta(days=7)
end_fit = test.loc[pd.to_datetime(test["timestamp"]) >= end_cutoff].copy()
end_fit["predicted_generation_proportion"] = model.predict(end_fit[features])
end_fit["actual_generation_proportion"] = end_fit["normalized_solar"]
end_fit_metrics = metrics_from_predictions(
    end_fit["actual_generation_proportion"],
    end_fit["predicted_generation_proportion"],
    "hgb",
)

evidence_columns = ["timestamp"]
if "site_id" in end_fit.columns:
    evidence_columns.append("site_id")
evidence_columns += ["actual_generation_proportion", "predicted_generation_proportion"]
end_fit_csv = Path(args.out).with_name("unisolar_end_fit.csv")
end_fit[evidence_columns].to_csv(end_fit_csv, index=False)

plot_frame = (
    end_fit.groupby("timestamp", as_index=False)[
        ["actual_generation_proportion", "predicted_generation_proportion"]
    ]
    .mean()
    .sort_values("timestamp")
)
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(
    plot_frame["timestamp"],
    plot_frame["actual_generation_proportion"],
    label="Actual UNISOLAR generation proportion",
    linewidth=1.5,
)
ax.plot(
    plot_frame["timestamp"],
    plot_frame["predicted_generation_proportion"],
    label="Model prediction",
    linewidth=1.5,
)
ax.set_title("UNISOLAR end-of-dataset fit (final 7 days)")
ax.set_xlabel("Timestamp")
ax.set_ylabel("Generation proportion of installed capacity")
ax.legend()
fig.autofmt_xdate()
fig.tight_layout()
end_fit_plot = Path(args.out).with_name("unisolar_end_fit.png")
fig.savefig(end_fit_plot, dpi=160, bbox_inches="tight")
plt.close(fig)

metadata = {
    "dataset_name": "unisolar_multi_site",
    "selected_model": "hgb",
    "model_purpose": "estimate expected normalized PV generation from weather and time",
    "feature_columns": features,
    "target": "solar_power / site_capacity_kwp",
    "target_scale": "capacity_ratio",
    "sites": int(frame["site_id"].nunique()) if "site_id" in frame.columns else None,
    "validation": hgb_result.__dict__,
    "hgb_tuning": {
        "selection_metric": "lowest validation MAE, RMSE tie-breaker",
        "search_type": "small explicit grid on day-grouped validation partition",
        "candidate_count": len(hgb_trials),
        "best_params": best_hgb_params,
        "test_partition_used_for_tuning": False,
    },
    "test": test_result.__dict__,
    "test_by_site": site_results,
    "end_of_dataset_fit": {
        "period": "final 7 calendar days of the day-grouped test partition",
        "metrics": end_fit_metrics.__dict__,
        "plot": str(end_fit_plot),
        "evidence_csv": str(end_fit_csv),
    },
    "split": (
        "day-grouped random split (70% train / 15% validation / 15% test, random_state=42) "
        "using GroupShuffleSplit on calendar date; all sites and intervals on a day stay in the same partition"
    ),
    "split_summary": split_summary,
    "deployment_assumption": (
        "Future GHI, cloud cover, temperature and humidity are supplied by an external weather API. "
        "API cloud cover is treated as a proxy for UNISOLAR CloudOpacity."
    ),
}

out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
save_bundle(model, features, out, metadata=metadata)
out.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
print("BEST MODEL: hgb")
print(f"TEST: {json.dumps(test_result.__dict__, indent=2)}")
print(f"END-OF-DATASET FIT (FINAL 7 DAYS): {json.dumps(end_fit_metrics.__dict__, indent=2)}")
print(f"Saved end-of-dataset fit plot: {end_fit_plot}")
print(f"Saved end-of-dataset fit evidence: {end_fit_csv}")
print(f"Saved {out}")
