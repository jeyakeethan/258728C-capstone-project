from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = {"timestamp", "solar_power"}

def load_canonical_csv(path):
    path = Path(path)
    df = pd.read_csv(path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"{path} is missing required columns: {sorted(missing)}. "
            "Expected at least timestamp and solar_power."
        )

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["solar_power"] = pd.to_numeric(df["solar_power"], errors="coerce")
    df = df.dropna(subset=["timestamp", "solar_power"])
    df = df.sort_values("timestamp")
    if "site_id" in df.columns:
        df["site_id"] = df["site_id"].astype(str)
        df = df.drop_duplicates(["timestamp", "site_id"], keep="last")
    else:
        df = df.drop_duplicates("timestamp", keep="last")
    return df.reset_index(drop=True)

