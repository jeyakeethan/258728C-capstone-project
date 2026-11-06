# How the system is useful

## Purpose

This project is a software based forecasting and risk monitoring model. It does not require physical sensors, batteries, generators, or other hardware to demonstrate its value. The operational screen is only the delivery layer. The main contribution is the design and evaluation of a data driven method for estimating solar generation and identifying periods in which available renewable energy may be insufficient.

The system combines two related forecasts:

1. expected site demand from calendar and time information; and
2. expected solar generation from weather, time, and installed capacity.

The purpose is to prevent an interruption in electricity availability by giving the responsible team enough time to confirm and secure the required backup supply. The alert supports operational preparation; it does not automatically control the generator.

## Why demand forecasting is useful

Site demand is not normally available as a simple lookup table for every future timestamp. A previous day can be useful, but the same clock time may have a different demand because it falls on a weekend, holiday, season, or a different point in the annual cycle. A forecasting model provides a repeatable way to estimate demand for a new timestamp using patterns learned from historical observations.

The current project uses a generated dataset because a physical deployment site and measured load history are not yet available. The generated series represents a plausible commercial site profile and is intended to test the complete software workflow. Its results must not be described as evidence of real site demand accuracy.

Once a real site is available, the generated series should be replaced or supplemented with measured interval load data. The demand model interface can remain the same: timestamp features are supplied and the model returns expected load.

## Demand attributes and their purpose

The demand dataset is generated in [demand.py](../src/solar_risk_monitor/demand.py) and contains the following main fields:

| Attribute | Why it may influence demand |
|---|---|
| `hour` | Represents the daily operating pattern. Commercial and industrial demand commonly rises during operating hours and falls overnight. The implementation preserves fractional hours at inference time. |
| `hour_sin`, `hour_cos` | Encode the daily cycle continuously. They avoid treating 23:00 and 00:00 as far apart and help models represent the circular nature of time. |
| `day_of_week` | Distinguishes working days from weekends and captures recurring weekly behaviour. |
| `is_holiday` | Represents reduced or altered activity on known holidays. The current implementation uses a small generic holiday set and should be replaced with the target region's official calendar. |
| `month` | Provides a coarse seasonal signal, such as changes in operating patterns or weather related demand. |
| `day_of_year` | Provides a more detailed annual position. In the generated data it represents a mild seasonal effect. |
| `site_load_kw` | The prediction target: the electrical load expected at the site for the timestamp. |

The current feature set is suitable for a first model, but it is not a complete demand model for a real site. Potentially important missing variables include:

* previous load values and rolling load averages;
* outside temperature, humidity, and cooling or heating degree indicators;
* production schedule, occupancy, or operating hours;
* shift patterns and planned shutdowns;
* equipment status and major industrial loads;
* electricity price or demand response events;
* rainfall or other weather variables where they affect operations;
* special events and the complete local holiday calendar.

The most important future change would be adding lagged demand features such as load one hour ago, one day ago, and one week ago. These features should only be used when the corresponding historical measurement is available. They must be created using time based validation to avoid leakage.

## How to train the demand model effectively

It is not necessary to wait one full year before making the first prediction. A practical deployment process is:

1. Begin with the historical data that is available, even if it covers only several weeks or months.
2. Train an initial model using chronological training, validation, and test periods.
3. Use the model operationally while continuing to collect measured demand.
4. Retrain periodically with the enlarged history, for example weekly or monthly.
5. Monitor forecast error and retrain earlier when error or demand behaviour changes materially.

The first year is valuable because it exposes the model to annual seasonality, but it is not a prerequisite for starting the system. During the cold start period, the model should show a lower confidence status and use conservative planning thresholds.

The current HGB implementation does not support true incremental updates through `partial_fit`. Therefore, “train as it goes” should mean periodic batch retraining on the latest accumulated history, not silently updating the model after every row. A rolling training window can be used when old behaviour is no longer representative. A real deployment should retain a fixed recent holdout period so that each retraining cycle is evaluated fairly.

## Local circuit versus public grid operation

The meaning of demand depends on the electrical boundary of the system.

For a local circuit or private site, the demand forecast is directly relevant. The system can compare expected local load with expected solar generation and identify whether the site may need imported electricity or backup generation.

If the solar plant exports to the public grid, local site demand is not necessarily the main target. The important quantities become:

* expected solar generation at the connection point;
* export capacity and network constraints;
* local consumption, if any;
* curtailment risk;
* grid price or dispatch requirements;
* imbalance between scheduled and actual generation.

In that case, the project should forecast **net export or available generation**, not assume that all generated electricity serves a local load. The current demand model is therefore appropriate only for a behind the meter or local circuit scenario. For an export scenario connected to the grid, demand should be treated as a separate optional module rather than the central forecast target.

## Solar generation model

The solar model uses UNISOLAR historical data and is deployed through the following conceptual flow:

```text
Weather forecast + time features + installed capacity
                         ↓
             Expected solar generation
                         ↓
                 Solar risk alert
```

The model predicts normalised solar generation rather than memorising one plant's absolute output. This allows the prediction to be scaled by a configured installed capacity. The deployment model intentionally uses variables that can be obtained from the weather service:

| Attribute | Why it influences solar generation |
|---|---|
| `Ghi` | Global horizontal irradiance is the direct measure of incoming solar radiation and is usually the strongest immediate predictor of PV output. |
| `CloudOpacity` | Clouds reduce and vary incoming radiation. Cloud movement can create rapid changes and is important for short term forecast risk. |
| `AirTemperature` | PV efficiency changes with module temperature. Air temperature is an indirect but useful thermal indicator. |
| `RelativeHumidity` | Humidity can be associated with cloud, haze, or atmospheric conditions that affect incoming radiation. It is usually weaker than irradiance but may improve generalisation. |
| `hour` | Captures the solar day and helps distinguish similar weather values at night, morning, and afternoon. |
| `day_of_week` | Usually has little physical effect on sunlight, but it may represent systematic data collection or operational patterns. Its contribution should be tested rather than assumed. |
| `day_of_year` and `month` | Represent seasonal solar position and annual variation. |
| `hour_sin`, `hour_cos`, `doy_sin`, `doy_cos` | Represent daily and annual cycles without artificial jumps at midnight or year boundaries. |
| installed capacity | Converts the predicted generation proportion into an expected plant output. It is a scaling variable, not a weather predictor. |

UNISOLAR contains additional fields such as apparent temperature, dew point, wind speed, wind direction, coordinates, and site identifiers. Some are not currently used because the deployment weather API does not provide a directly equivalent, consistently mapped input. They should not be added automatically: each candidate feature needs an ablation experiment and a matching deployment source.

## Adapting UNISOLAR to another location

UNISOLAR is useful because it provides historical weather and solar generation examples, but it does not guarantee that a model trained at one location will behave identically elsewhere. The current processed dataset identifies the UNISOLAR site as Bundoora, Australia. A different climate, latitude, panel orientation, shading pattern, technology, or data collection process can change the relationship between weather and generation.

The model should therefore be treated as a transferable starting point, not a universally valid model. Adaptation should follow this process:

1. Map local weather provider fields to the same conceptual variables.
2. Confirm units, time zones, sampling intervals, and missing value behaviour.
3. Scale output using the local installed capacity.
4. Collect a local calibration period of measured generation.
5. Evaluate the transferred model on a chronological local holdout.
6. Retrain or recalibrate using local observations when systematic bias is found.
7. Report performance separately for the original and adapted locations.

Useful adaptation methods include local bias correction, retraining with a mixture of UNISOLAR and local data, adding site metadata, and training a local model after sufficient observations have accumulated. A model should not be declared feasible for a new region until it has passed local validation.

## What the alert means

The system is intended to maintain backup availability and help prevent an interruption in electricity supply. It does not need to control a generator, but it should provide an early warning when forecast renewable supply may be insufficient. This gives the responsible team time to verify that adequate backup fuel and generation capacity are available before the risk becomes an interruption.

For example:

> **Today: confirm backup availability before 13:00.** Forecast solar generation is expected to fall below the relevant energy requirement, and the supply risk may begin at 13:00.

The alert should ideally include the forecast horizon, expected generation, confidence or scenario level, and the reason for the warning. A conservative alert can be generated before the worst case occurs, giving an operator time to confirm reserve availability.

## Feasibility conclusion

The software model is feasible as a forecasting and early warning prototype. Its value is not that it replaces physical energy infrastructure; its value is that it converts available weather and historical generation data into an interpretable planning signal.

The main conditions for industry use are:

* measured local generation data for adaptation;
* a correct local holiday and operating calendar;
* real demand data if the system is behind the meter;
* clear separation between local load and grid export use cases;
* continuous error monitoring and periodic retraining;
* conservative treatment of uncertainty and missing data.

The current generated demand data is appropriate for demonstrating the model pipeline, but the final evaluation must clearly distinguish software feasibility from proven operational accuracy.
