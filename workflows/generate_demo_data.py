from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

rng = np.random.default_rng(42)
n = 24 * 120
ts = pd.date_range("2024-01-01", periods=n, freq="h")

hour = ts.hour.to_numpy()
day = ts.dayofyear.to_numpy()

daylight = np.maximum(0, np.sin(np.pi * (hour - 6) / 12))
seasonal = 0.82 + 0.18 * np.sin(2 * np.pi * (day - 30) / 365.25)

cloud_cover = np.clip(
    45 + 28 * np.sin(np.arange(n) / 37) + rng.normal(0, 15, n),
    0, 100
)

irradiance = np.maximum(
    0,
    950 * daylight * seasonal * (1 - 0.72 * cloud_cover / 100)
    + rng.normal(0, 25, n),
)

temperature = (
    22
    + 7 * np.sin(2 * np.pi * (hour - 8) / 24)
    + 3 * np.sin(2 * np.pi * day / 365.25)
    + rng.normal(0, 1.2, n)
)

humidity = np.clip(
    72 - 0.9 * (temperature - 20) + rng.normal(0, 6, n),
    20, 100
)

wind_speed = np.clip(rng.gamma(2.0, 1.6, n), 0, 15)
precipitation = np.where(cloud_cover > 78, rng.gamma(1.5, 0.7, n), 0)

solar_power = (
    0.105 * irradiance
    * (1 - 0.0025 * np.maximum(0, temperature - 25))
    + rng.normal(0, 2.2, n)
)
solar_power = np.clip(solar_power, 0, 100)

df = pd.DataFrame({
    "timestamp": ts,
    "solar_power": solar_power,
    "irradiance": irradiance,
    "cloud_cover": cloud_cover,
    "temperature": temperature,
    "humidity": humidity,
    "wind_speed": wind_speed,
    "precipitation": precipitation,
})

out = ROOT / "data" / "processed" / "demo_solar.csv"
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
print(f"Wrote {len(df):,} rows to {out}")
