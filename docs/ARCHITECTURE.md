```text
TRAINING

UNISOLAR multi-site solar + environment
        ↓ capacity normalization
Linear Regression vs tuned HistGradientBoosting
        ↓
Generalized solar forecaster

Synthetic historical site-demand dataset
        ↓ temporal features
Linear Regression vs HistGradientBoosting
        ↓
Demand forecaster

────────────────────────────────────────────────────────
DEPLOYMENT

Latitude + longitude ──→ reverse geocoded display location
        │
        └──→ Open-Meteo current + hourly weather
                       ↓
               Solar ML forecaster
                       ↓
              Expected solar (kW)
                       │
                       ├───────────────┐
                       │               │
                       │         Demand ML forecaster
                       │               ↓
                       │       Expected site load (kW)
                       └───────┬───────┘
                               ↓
                    Hourly energy conversion
                               ↓
                    Battery-state simulation
                               ↓
               Fuel-backed energy required (kWh)
                         ┌─────┴─────┐
                         ↓           ↓
                  Today/live     Future 7 days
                  shortage       daily totals
                  periods        (start battery=0)
                         ↓           ↓
                  "X kWh by     "Prepare X kWh"
                   HH:MM"
```

## Presentation rule

**Calculate hourly, present simply.** Battery end-state and intermediate net-energy values remain internal. The collapsible audit table exposes only time, expected solar energy, expected demand energy, battery reserve at interval start, and fuel-backed energy required.

## No day/night split

There is no separate daylight/night backup concept. The same chronological balance runs every hour. Solar naturally approaches zero outside generating hours.
