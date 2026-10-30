from pathlib import Path
import argparse
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DEFAULT = ROOT / "data" / "raw" / "unisolar"
OUT_DEFAULT = ROOT / "data" / "processed" / "unisolar.csv"

CAMPUS_NAME_MAP = {
    "1": "Bundoora",
    "2": "Wodonga",
    "3": "Bendigo",
    "4": "Mildura",
    "5": "Shepparton",
}
WEATHER_FEATURES = [
    "ApparentTemperature", "AirTemperature", "DewPointTemperature",
    "RelativeHumidity", "WindSpeed", "WindDirection",
]
IRRADIANCE_FEATURES = ["CloudOpacity", "Ghi"]


def parse_ts(series):
    return pd.to_datetime(series, errors="coerce", utc=True).dt.tz_localize(None)


def _interpolate_one_site(group: pd.DataFrame, columns, limit_steps: int) -> pd.DataFrame:
    out = group.sort_values("timestamp").copy()
    for col in columns:
        if col not in out.columns:
            continue
        missing = out[col].isna()
        out[col] = out[col].interpolate(method="linear", limit=limit_steps, limit_area="inside")
        out[f"{col}_was_missing"] = missing.astype("int8")
    return out


def prepare(raw_dir: Path, out_path: Path, site_key=None, max_interp_minutes=60):
    gen_path = raw_dir / "Solar_Generation_sub.csv"
    weather_path = raw_dir / "Weather_data_sub.csv"
    irradiance_path = raw_dir / "Solar_Irradiance.csv"
    details_path = raw_dir / "Solar_Site_Details.csv"

    for p in [gen_path, weather_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    gen = pd.read_csv(gen_path)
    required = {"CampusKey", "SiteKey", "Timestamp", "SolarGeneration"}
    missing = required - set(gen.columns)
    if missing:
        raise ValueError(f"{gen_path} missing columns {sorted(missing)}")

    gen["timestamp"] = parse_ts(gen["Timestamp"])
    gen["solar_power"] = pd.to_numeric(gen["SolarGeneration"], errors="coerce")
    gen["CampusKey"] = gen["CampusKey"].astype(str)
    gen["SiteKey"] = gen["SiteKey"].astype(str)
    gen = gen.dropna(subset=["timestamp", "solar_power"])
    if site_key is not None:
        site_key = str(site_key)
        if site_key not in set(gen["SiteKey"]):
            raise ValueError(f"SiteKey {site_key!r} not found")
        gen = gen[gen["SiteKey"] == site_key].copy()
        print(f"Preparing selected SiteKey {site_key}")
    else:
        print(f"Preparing all available sites ({gen['SiteKey'].nunique()} sites)")

    base = gen[["timestamp", "CampusKey", "SiteKey", "solar_power"]].copy()

    weather = pd.read_csv(weather_path)
    weather["timestamp"] = parse_ts(weather["Timestamp"])
    weather["CampusKey"] = weather["CampusKey"].astype(str)
    weather_features = [c for c in WEATHER_FEATURES if c in weather.columns]
    for c in weather_features:
        weather[c] = pd.to_numeric(weather[c], errors="coerce")
    weather = (
        weather[["timestamp", "CampusKey"] + weather_features]
        .dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .drop_duplicates(["timestamp", "CampusKey"], keep="last")
    )
    base = base.merge(weather, on=["timestamp", "CampusKey"], how="left", validate="many_to_one")

    irr_features = []
    if irradiance_path.exists():
        irr = pd.read_csv(irradiance_path)
        if {"Timestamp_UTC", "Campus"}.issubset(irr.columns):
            irr["timestamp"] = parse_ts(irr["Timestamp_UTC"])
            irr["Campus"] = irr["Campus"].astype(str)
            irr_features = [c for c in IRRADIANCE_FEATURES if c in irr.columns]
            for c in irr_features:
                irr[c] = pd.to_numeric(irr[c], errors="coerce")
            name_to_key = {v.casefold(): k for k, v in CAMPUS_NAME_MAP.items()}
            irr["CampusKey"] = irr["Campus"].str.casefold().map(name_to_key)
            irr = (
                irr[["timestamp", "CampusKey"] + irr_features]
                .dropna(subset=["timestamp", "CampusKey"])
                .sort_values("timestamp")
                .drop_duplicates(["timestamp", "CampusKey"], keep="last")
            )
            base = base.merge(irr, on=["timestamp", "CampusKey"], how="left", validate="many_to_one")

    out = base.rename(columns={"SiteKey": "site_id", "CampusKey": "campus_id"}).copy()
    out["dataset"] = "unisolar"
    out["campus_name"] = out["campus_id"].map(CAMPUS_NAME_MAP).fillna("unknown")

    if details_path.exists():
        details = pd.read_csv(details_path)
        if "SiteKey" in details.columns:
            details["site_id"] = details["SiteKey"].astype(str)
            keep = ["site_id"]
            rename = {}
            for source, target in [("kWp", "site_capacity_kwp"), ("lat", "latitude"), ("Lon", "longitude")]:
                if source in details.columns:
                    details[source] = pd.to_numeric(details[source], errors="coerce")
                    keep.append(source)
                    rename[source] = target
            meta = details[keep].drop_duplicates("site_id").rename(columns=rename)
            out = out.merge(meta, on="site_id", how="left", validate="many_to_one")

    out = out.dropna(subset=["timestamp", "solar_power"]).sort_values(["site_id", "timestamp"])
    # Estimate cadence from within-site differences to avoid zero deltas across sites.
    deltas = out.groupby("site_id")["timestamp"].diff().dropna()
    step_minutes = deltas.median().total_seconds() / 60.0 if not deltas.empty else 15.0
    limit_steps = max(1, int(np.floor(float(max_interp_minutes) / step_minutes)))
    dynamic_cols = weather_features + irr_features
    missing_before = out[dynamic_cols].isna().sum() if dynamic_cols else pd.Series(dtype=int)
    if dynamic_cols:
        parts = [_interpolate_one_site(g, dynamic_cols, limit_steps) for _, g in out.groupby("site_id", sort=False)]
        out = pd.concat(parts, ignore_index=True)
    missing_after = out[dynamic_cols].isna().sum() if dynamic_cols else pd.Series(dtype=int)

    ordered = [
        "timestamp", "solar_power", "dataset", "site_id", "campus_id", "campus_name",
        "site_capacity_kwp", "latitude", "longitude",
    ]
    ordered += [c for c in weather_features + irr_features if c in out.columns]
    ordered += [f"{c}_was_missing" for c in dynamic_cols if f"{c}_was_missing" in out.columns]
    out = out[[c for c in ordered if c in out.columns]].sort_values(["timestamp", "site_id"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")
    print(f"Sites: {out['site_id'].nunique()}")
    print(f"Campuses: {out['campus_id'].nunique()}")
    print(f"Rows: {len(out):,}")
    print(f"Range: {out.timestamp.min()} -> {out.timestamp.max()}")
    print(f"Median within-site sampling interval: {step_minutes:g} minutes")
    if dynamic_cols:
        print(pd.DataFrame({"before": missing_before, "after_short_interpolation": missing_after}).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare UNISOLAR multi-site data into one canonical CSV.")
    parser.add_argument("--raw", type=Path, default=RAW_DEFAULT)
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--site", default=None, help="Optional SiteKey. Omit to prepare all available sites.")
    parser.add_argument("--max-interp-minutes", type=int, default=60)
    args = parser.parse_args()
    prepare(args.raw, args.out, args.site, args.max_interp_minutes)
