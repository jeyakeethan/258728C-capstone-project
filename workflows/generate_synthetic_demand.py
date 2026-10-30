from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from solar_risk_monitor.demand import generate_synthetic_demand

parser = argparse.ArgumentParser(description="Generate a transparent synthetic historical site-demand training dataset.")
parser.add_argument("--start", default="2024-01-01 00:00")
parser.add_argument("--end", default="2025-12-31 23:00")
parser.add_argument("--base-peak-kw", type=float, default=120.0)
parser.add_argument("--noise-fraction", type=float, default=0.025)
parser.add_argument("--out", default=str(ROOT / "data/processed/synthetic_site_demand.csv"))
args = parser.parse_args()

out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
df = generate_synthetic_demand(args.start, args.end, args.base_peak_kw, args.noise_fraction)
df.to_csv(out, index=False)
print(f"Saved {len(df):,} rows to {out}")
print(df.head().to_string(index=False))
