# Weather-Aware Solar & Fuel Readiness Planner

An end-to-end energy-risk planning application that combines a **weather-based solar prediction**, a **synthetic-demand estimation component**, and an **hourly battery energy simulation** to identify when generator-backed electrical energy may be required.

## Final operational logic

```text
Weather + time + installed solar capacity
                ↓
        Expected solar forecast
                │
                │       Time/calendar features
                │                ↓
                │       Expected demand forecast
                └────────┬───────┘
                         ↓
              Hourly energy balance
                         ↓
              Battery reserve state
                         ↓
          Fuel-backed energy requirement
                         ↓
      Today: warning by shortage start time
      Future: whole-day fuel readiness forecast
```

The system reports **fuel-backed electrical energy (MWh)**, not litres of fuel. Fuel-volume conversion requires generator-specific consumption/efficiency data and is outside the current scope.


## Deployment units and defaults

The deployment application uses **MW for power** and **MWh for energy** throughout its live and planning calculations. The configured values are native MW/MWh values. Default deployment configuration:

```text
Latitude: 6.8741
Longitude: 79.8605
Installed solar capacity: 2000 MW
Usable battery capacity: 1500 MWh
```

The historical UNISOLAR preparation files retain the source dataset's original field names/units for reproducibility. The trained solar target is normalized, so deployment predictions are scaled directly by the configured **2000 MW** capacity. Because UNISOLAR is an Australian dataset (four seasons, 6–8 PM sunset), it is used here as an initial conceptual baseline and is adaptable to Sri Lanka or other target locations once local site data is collected for retraining.

## Dashboard

The main panel shows the auto-resolved location, weather, **Solar Power**, **Site Demand**, and **Battery Reserve**. Solar Power and Site Demand are smooth five-second display values derived from the current model predictions.

For **today**, the simulation starts from the entered current battery reserve and runs from the current time to the end of the day. Consecutive hours where solar plus available battery cannot meet demand are grouped into shortage periods. A warning such as `47 MWh fuel-backed energy required by 13:00` means that 47 MWh of generator-backed electrical energy should be available before the shortage begins at 13:00 and covers that continuous shortage period until forecast supply can meet demand again. Separate shortage periods are not shown cumulatively.

For the **next seven forecast days**, battery state is continuous across midnight. Tomorrow starts with today’s forecast ending reserve, and every later day starts with the preceding day’s ending reserve. The system simulates every hour. Solar surplus can charge the battery up to its usable capacity; demand consumes solar and then available battery; only remaining unmet demand becomes fuel-backed energy. The dashboard presents one wide row per day with expected solar energy, expected demand, fuel-backed energy required, and readiness status.

Hourly calculations are hidden in collapsible sections. The operator-facing table shows `Time`, `Solar Power (MW)`, `Solar Energy (MWh)`, `Grid Demand (MWh)`, `Battery reserve (MWh)`, and `Fuel required (MWh)`.

## Battery simulation

For each hourly interval:

```text
net_energy = solar_energy - demand_energy
```

If net energy is positive:

```text
battery_next = min(battery_capacity, battery + net_energy)
fuel_required = 0
```

If net energy is negative, the deficit uses available battery first. Any remaining deficit is generator-backed energy:

```text
deficit = demand_energy - solar_energy
from_battery = min(battery, deficit)
fuel_required = deficit - from_battery
battery_next = battery - from_battery
```

This must be chronological. Daily `solar - demand` arithmetic is not sufficient because an empty battery can cause an early-morning shortage even if strong solar later makes total daily solar exceed total daily demand.

## Solar model

Place UNISOLAR files in `data/raw/unisolar/`, then run:

```bash
python workflows/prepare_unisolar.py
python workflows/train_unisolar_weather_model.py
```

The generalized target is capacity-normalized solar output:

```text
normalized_solar = solar_power / site_capacity_kwp
expected_power_mw = predicted_normalized_solar × configured_site_capacity_mw
```

Deployment weather uses Open-Meteo mappings: `shortwave_radiation → Ghi`, `cloud_cover → CloudOpacity` proxy, `temperature_2m → AirTemperature`, and `relative_humidity_2m → RelativeHumidity`.

### Solar model features

`prepare_unisolar.py` writes `data/processed/unisolar.csv`. Weather and irradiance gaps are filled with short within-site linear interpolation (default at most 60 minutes). Remaining NaNs are left in place. The processed table currently has:

```text
timestamp, solar_power, dataset, site_id, campus_id, campus_name,
site_capacity_kwp, latitude, longitude,
ApparentTemperature, AirTemperature, DewPointTemperature, RelativeHumidity,
WindSpeed, WindDirection, CloudOpacity, Ghi
```

Training (`train_unisolar_weather_model.py`) uses only weather/time predictors that a forecast API can supply. Time features are built from `timestamp` in `add_time_features()`:

| Role | Columns |
|------|---------|
| Weather / irradiance | `Ghi`, `CloudOpacity`, `AirTemperature`, `RelativeHumidity` |
| Time (engineered) | `hour`, `day_of_week`, `day_of_year`, `month`, `hour_sin`, `hour_cos`, `doy_sin`, `doy_cos` |
| Label | `normalized_solar = solar_power / site_capacity_kwp` |

`site_capacity_kwp` is used to normalize the training target. At deployment the operator supplies installed capacity and the predicted ratio is scaled to power. It is not a model feature.

Kept in the processed CSV but **not** used as predictors: `dataset`, `site_id`, `campus_id`, `campus_name`, `latitude`, `longitude`, `ApparentTemperature`, `DewPointTemperature`, `WindSpeed`, and `WindDirection`. Solar history lags are also excluded so a future hour can be predicted from weather and time only.

### Solar HGB hyperparameter tuning

The UNISOLAR training script performs a small, reproducible HistGradientBoostingRegressor hyperparameter search using the chronological validation partition only. It searches `learning_rate`, `max_iter`, `max_leaf_nodes`, `min_samples_leaf`, and `l2_regularization`, then evaluates the selected configuration on the untouched chronological test partition. The selected parameters and tuning protocol are written into the solar model metadata JSON.

## Demand model

The project contains a reproducible **synthetic historical site-demand dataset** with time-of-day, weekday/weekend, holiday, seasonal and small random effects. Generate and train it with:

```bash
python workflows/generate_synthetic_demand.py
python workflows/train_demand_model.py
```

The demand workflow deploys a `HistGradientBoostingRegressor` model. It also writes `models/demand_end_fit.png` and `models/demand_end_fit.csv`, showing actual versus predicted demand for the final seven days of the chronological test partition.

Synthetic demand performance demonstrates the planning workflow; it is **not evidence of real-site demand accuracy**.

## Run

```bash
pip install -r requirements.txt
streamlit run app/dashboard.py
```

If packaged joblib files were created under an incompatible scikit-learn version, retrain the models in the active environment instead of relying on the packaged binaries.

## Important assumptions and limitations

- Installed solar peak capacity may exceed site peak demand, but weather and time-of-day can still reduce available solar below demand.
- Future weather is forecast rather than observed, so weather forecast error propagates into solar estimates.
- UNISOLAR is an Australian dataset (four seasons, 6–8 PM sunset) and cannot be applied directly to Sri Lanka; it serves as an initial conceptual baseline that can be adapted by retraining with locally collected site data.
- API cloud cover is a proxy for the UNISOLAR cloud variable.
- Forecast battery state is carried continuously across day boundaries; today begins from the synchronized estimated present reserve.
- Battery efficiency, degradation, minimum state of charge and charge/discharge power limits are not modelled.
- Generator fuel litres are not estimated.

## Dashboard synchronization

The Streamlit dashboard uses a **5-second heartbeat**. **Solar Power** uses the weather-based ML forecast from the UNISOLAR model, and **Site Demand is produced by the trained demand model for the exact current timestamp**, including minute/second information, rather than replaying a row from the historical demand dataset. Present Battery Reserve is an estimated value because no physical battery telemetry is available. Demand remains synthetic-origin until measured site demand is available.

All downstream today/fuel-readiness calculations are re-evaluated on each heartbeat. External weather responses and trained model objects are cached independently to avoid unnecessary API/model reloads. Future-day energy balance is still calculated hourly internally and shown only as daily readiness outcomes.

## How to get it work
### Activate venv environment
```bash
.venv\Scripts\activate
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Prepare Datasets
```bash
python .\app\workflows\generate_synthetic_demand.py
python .\app\workflows\generate_demo_data.py
python .\app\workflows\prepare_unisolar.py
```

### Train the demand model
```bash
python .\app\workflows\train_demand_model.py
```

### Exploratory data analysis
```bash
python .\app\analysis\eda_unisolar.py
```

### Train the solar model
```bash
python .\app\workflows\train_unisolar_weather_model.py
```

### Run the application
```bash
streamlit run app/dashboard.py
```