# Weather-Aware Solar & Fuel Readiness Planner

An end-to-end CS5998 capstone prototype that combines a **weather-aware solar forecast**, a **time-series site-demand forecast**, and an **hourly battery energy simulation** to warn when generator-backed electrical energy must be prepared.

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
Latitude: 51.5085
Longitude: -0.12574
Installed solar capacity: 1500 MW
Usable battery capacity: 2500 MWh
```

The historical UNISOLAR preparation files retain the source dataset's original field names/units for reproducibility. The trained solar target is normalized, so deployment predictions are scaled directly by the configured **1500 MW** capacity.

## Dashboard

The main panel shows the auto-resolved location, weather, **Expected Solar Now**, **Expected Site Load**, and **Current Battery Reserve**. Expected solar is also shown as a percentage of installed solar capacity.

For **today**, the simulation starts from the entered current battery reserve and runs from the current time to the end of the day. Consecutive hours where solar plus available battery cannot meet demand are grouped into shortage periods. A warning such as `47 MWh fuel-backed energy required by 13:00` means that 47 MWh of generator-backed electrical energy should be available before the shortage begins at 13:00 and covers that continuous shortage period until forecast supply can meet demand again. Separate shortage periods are not shown cumulatively.

For the **next seven forecast days**, battery state is continuous across midnight. Tomorrow starts with today’s forecast ending reserve, and every later day starts with the preceding day’s ending reserve. The system simulates every hour. Solar surplus can charge the battery up to its usable capacity; demand consumes solar and then available battery; only remaining unmet demand becomes fuel-backed energy. The dashboard presents one wide row per day with expected solar energy, expected demand, fuel-backed energy required, and readiness status.

Hourly calculations are hidden in collapsible sections. The operator-facing table intentionally shows only `Time`, `Expected Solar`, `Expected Demand`, `Battery Reserve`, and `Fuel Required`. Battery end-state is calculated internally but is not displayed.

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

### Solar HGB hyperparameter tuning

The UNISOLAR training script performs a small, explicit and reproducible HGB hyperparameter search using the chronological validation partition only. It searches `learning_rate`, `max_iter`, `max_leaf_nodes`, `min_samples_leaf`, and `l2_regularization`, selects the HGB configuration with the lowest validation MAE (RMSE tie-breaker), and leaves the final chronological test partition untouched. The selected parameters and tuning protocol are written into the solar model metadata JSON. Linear Regression remains the untuned baseline.

## Demand model

The project contains a reproducible **synthetic historical site-demand dataset** with time-of-day, weekday/weekend, holiday, seasonal and small random effects. Generate and train it with:

```bash
python workflows/generate_synthetic_demand.py
python workflows/train_demand_model.py
```

Generate the synthetic solar trace used only for the dashboard's simulated present-power display:

```bash
python workflows/generate_demo_data.py
```

This synthetic trace is not a solar forecast model and must not be used as UNISOLAR research data. The dashboard requires the trained UNISOLAR model.

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
- UNISOLAR is geographically limited; transfer to other regions requires separate validation.
- API cloud cover is a proxy for the UNISOLAR cloud variable.
- Forecast battery state is carried continuously across day boundaries; today begins from the synchronized estimated present reserve.
- Battery efficiency, degradation, minimum state of charge and charge/discharge power limits are not modelled.
- Generator fuel litres are not estimated.

## Dashboard synchronization

The Streamlit dashboard uses a **5-second heartbeat**. Present Solar Power remains a clearly labelled simulated value when inverter telemetry is unavailable. **Expected Site Demand is produced by the trained demand model for the exact current timestamp**, including minute/second information, rather than replaying a row from the historical demand dataset. Present Battery Reserve is an estimated value because no physical battery telemetry is available. **Expected Solar Generation** is the weather-based ML forecast and remains the main live forecasting outcome.

All downstream today/fuel-readiness calculations are re-evaluated on each heartbeat. External weather responses and trained model objects are cached independently to avoid unnecessary API/model reloads. Future-day energy balance is still calculated hourly internally and shown only as daily readiness outcomes.
