from pathlib import Path
import argparse
from datetime import timedelta
import json
import sys

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from demand import (
    DEMAND_FEATURES,
    chronological_demand_split,
    train_demand_forecaster,
)
from models import load_bundle, metrics_from_predictions

parser = argparse.ArgumentParser(description="Train the demand model from the synthetic historical site-demand dataset.")
parser.add_argument("--data", default=str(ROOT / "data/processed/synthetic_site_demand.csv"))
parser.add_argument("--out", default=str(ROOT / "models/site_demand_forecaster.joblib"))
args = parser.parse_args()

data = pd.read_csv(args.data, parse_dates=["timestamp"])
out = Path(args.out)
metadata = train_demand_forecaster(data, out)

# Generate the same kind of end-of-dataset evidence as the UNISOLAR workflow.
# The final seven days are selected from the untouched chronological test
# partition, so the plot shows genuine out-of-sample predictions.
_, _, test = chronological_demand_split(data)
bundle = load_bundle(out)
model = bundle["model"]
test = test.sort_values("timestamp").reset_index(drop=True)
test["predicted_site_load_kw"] = model.predict(test[DEMAND_FEATURES])

end_cutoff = pd.to_datetime(test["timestamp"]).max() - timedelta(days=7)
end_fit = test.loc[pd.to_datetime(test["timestamp"]) >= end_cutoff].copy()
end_fit["actual_site_load_kw"] = end_fit["site_load_kw"]
end_fit["residual_kw"] = (
    end_fit["actual_site_load_kw"] - end_fit["predicted_site_load_kw"]
)
end_fit_metrics = metrics_from_predictions(
    end_fit["actual_site_load_kw"],
    end_fit["predicted_site_load_kw"],
    metadata["selected_model"],
)

end_fit_csv = out.with_name("demand_end_fit.csv")
end_fit[[
    "timestamp",
    "actual_site_load_kw",
    "predicted_site_load_kw",
    "residual_kw",
]].to_csv(end_fit_csv, index=False)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(
    end_fit["timestamp"],
    end_fit["actual_site_load_kw"],
    label="Actual site demand",
    linewidth=1.5,
)
ax.plot(
    end_fit["timestamp"],
    end_fit["predicted_site_load_kw"],
    label="Model prediction",
    linewidth=1.5,
)
ax.set_title("Synthetic demand end-of-dataset fit (final 7 test days)")
ax.set_xlabel("Timestamp")
ax.set_ylabel("Site demand (kW)")
ax.legend()
fig.autofmt_xdate()
fig.tight_layout()
end_fit_plot = out.with_name("demand_end_fit.png")
fig.savefig(end_fit_plot, dpi=160, bbox_inches="tight")
plt.close(fig)

metadata["end_of_dataset_fit"] = {
    "period": "final 7 days of the chronological test partition",
    "metrics": end_fit_metrics.__dict__,
    "plot": str(end_fit_plot),
    "evidence_csv": str(end_fit_csv),
}
out.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

print(json.dumps(metadata, indent=2))
print(f"Saved end-of-dataset fit plot: {end_fit_plot}")
print(f"Saved end-of-dataset fit evidence: {end_fit_csv}")
