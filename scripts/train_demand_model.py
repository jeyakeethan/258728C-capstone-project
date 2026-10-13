from pathlib import Path
import argparse
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from solar_risk_monitor.demand import train_demand_forecaster

parser = argparse.ArgumentParser(description="Train the demand model from the synthetic historical site-demand dataset.")
parser.add_argument("--data", default=str(ROOT / "data/processed/synthetic_site_demand.csv"))
parser.add_argument("--out", default=str(ROOT / "models/site_demand_forecaster.joblib"))
args = parser.parse_args()

data = pd.read_csv(args.data, parse_dates=["timestamp"])
metadata = train_demand_forecaster(data, Path(args.out))
print(json.dumps(metadata, indent=2))
