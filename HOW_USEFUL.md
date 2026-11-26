# System purpose and scope

## Purpose

This project is an energy-risk planning application. It predicts solar generation from weather data, estimates site demand, and simulates the hourly balance between solar, demand, battery reserve, and generator-backed energy.

The system combines two related forecasts:

1. expected site demand from calendar and time information; and
2. expected solar generation from weather, time, and installed capacity.

The purpose is to identify forecast shortage periods early enough for an operator to prepare backup supply. The application provides planning information; it does not control a generator.

## Why demand forecasting is useful

Site demand is not normally available as a simple lookup table for every future timestamp. A previous day can be useful, but the same clock time may have a different demand because it falls on a weekend, holiday, season, or a different point in the annual cycle. A forecasting model provides a repeatable way to estimate demand for a new timestamp using patterns learned from historical observations.

Measured site-demand data is not available in the current implementation. The demand model therefore uses reproducible synthetic hourly data for workflow development and demonstration. Its results are not evidence of real-site demand accuracy.

For deployment, the synthetic demand series should be replaced or supplemented with measured interval load data. The model interface can remain the same: timestamp features are supplied and the model returns estimated load.

## Demand attributes and their purpose

The demand dataset is generated in [demand.py](src/demand.py) and contains the following main fields:

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

Future demand training should use measured lagged load features, such as the previous hour, previous day, and previous week, when those observations are available. These features must be generated within chronological validation boundaries.

## Demand model deployment

The current workflow uses chronological training, validation, and test partitions. A deployment process should retrain periodically as measured demand data accumulates and should retain a fixed recent holdout period for fair evaluation.

The current HistGradientBoostingRegressor is batch-trained. It is not updated incrementally after each new observation.

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

## Weather data sources

For deployment and future forecasting, the application requests hourly weather forecasts from the Open Meteo Forecast API. The request is made using the selected latitude and longitude and returns current and future weather values. The application maps the API fields to the names expected by the trained model:

| Open Meteo field | Model field | Purpose |
|---|---|---|
| `shortwave_radiation` | `Ghi` | Incoming solar radiation available to the site. |
| `cloud_cover` | `CloudOpacity` | Cloud coverage used as a proxy for the historical cloud variable. |
| `temperature_2m` | `AirTemperature` | Air temperature associated with panel operating conditions. |
| `relative_humidity_2m` | `RelativeHumidity` | Atmospheric moisture information. |
| `time` | `timestamp` | Determines the hour and annual calendar features. |

The application also uses the Open Street Map Nominatim service only to convert the selected coordinates into a readable location name. It is not used as a source of weather values or model features.

The weather retrieval and field mapping are implemented in [weather_api.py](src/weather_api.py). Weather responses are cached for a limited period so that the dashboard does not repeatedly request the same forecast during its refresh cycle. If the weather service is unavailable, the application cannot produce a new forecast and must display an error rather than presenting old values as current information.

The weather forecast source is important to feasibility. The model is trained on historical observations from UNISOLAR, while deployment uses forecast values from Open Meteo. Differences between the two sources can create prediction error. Therefore, the model should be recalibrated and validated with local weather and generation records before it is used for a real site.

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

## Operational alert

The alert identifies when forecast solar and battery energy may not meet demand and reports the generator-backed energy required for the shortage period. It supports operator preparation and does not control physical equipment.

## Scope conclusion

The current implementation demonstrates the software pipeline and planning workflow. Operational accuracy requires measured local solar and demand data, local validation, appropriate calendar information, continued error monitoring, and explicit treatment of uncertainty.

Synthetic demand results demonstrate the pipeline only; they do not establish real-site demand accuracy.
