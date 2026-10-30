from pathlib import Path
import html
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import streamlit as st

from solar_risk_monitor.demand import future_demand_features
from solar_risk_monitor.features import add_time_features
from solar_risk_monitor.models import load_bundle
from solar_risk_monitor.live_simulation import synthetic_live_values, estimate_present_battery
from solar_risk_monitor.planning import simulate_battery_and_fuel, shortage_periods, summarize_energy
from solar_risk_monitor.weather_api import describe_weather_code, fetch_open_meteo_hourly, reverse_geocode

st.set_page_config(page_title="Solar & Fuel Readiness Planner", page_icon="☀️", layout="wide")
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.0rem; padding-bottom: 3rem; max-width: 1500px;}
      div[data-testid="stMetricValue"] {font-size:1.55rem;}
      div[data-testid="stMetricLabel"] {font-size:.86rem;}
      .weather-card {border:1px solid rgba(120,120,120,.22); border-radius:18px; padding:18px 22px; margin:4px 0 18px 0; background:rgba(250,250,250,.55);}
      .location {font-size:1.12rem; font-weight:700;}
      .local-time {font-size:.82rem; opacity:.68;}
      .weather-main {display:flex; gap:14px; align-items:center; margin-top:10px;}
      .weather-icon {font-size:2.8rem; line-height:1;}
      .temp {font-size:2.8rem; font-weight:650; line-height:1;}
      .condition {font-size:1.02rem; font-weight:600;}
      .weather-details {display:grid; grid-template-columns:repeat(5,minmax(110px,1fr)); gap:10px; margin-top:14px;}
      .detail {padding:9px 11px; border-radius:10px; background:rgba(127,127,127,.07);}
      .detail-label {font-size:.72rem; opacity:.65;}
      .detail-value {font-size:1.00rem; font-weight:650;}
      .warning-box {border-left:5px solid #d97706; border-radius:10px; padding:13px 16px; margin:10px 0; background:rgba(217,119,6,.08); font-size:1.05rem;}
      .ok-box {border-left:5px solid #16a34a; border-radius:10px; padding:13px 16px; margin:10px 0; background:rgba(22,163,74,.07); font-size:1.05rem;}
      .forecast-wrap {overflow-x:auto; margin-top:8px;}
      table.forecast {width:100%; border-collapse:separate; border-spacing:0 9px; font-size:1.02rem;}
      table.forecast th {text-align:left; padding:8px 16px; font-size:.82rem; opacity:.68; white-space:nowrap;}
      table.forecast td {padding:17px 16px; background:rgba(127,127,127,.055); border-top:1px solid rgba(120,120,120,.14); border-bottom:1px solid rgba(120,120,120,.14); white-space:nowrap;}
      table.forecast td:first-child {border-left:1px solid rgba(120,120,120,.14); border-radius:12px 0 0 12px; font-weight:700;}
      table.forecast td:last-child {border-right:1px solid rgba(120,120,120,.14); border-radius:0 12px 12px 0; font-weight:700;}
      .status-warn {font-weight:800;}
      .status-ok {font-weight:750;}
      .small-note {font-size:.80rem; opacity:.70;}
      @media (max-width:800px) {.weather-details {grid-template-columns:repeat(2,1fr);}}
    </style>
    """,
    unsafe_allow_html=True,
)

SOLAR_MODEL_PATH = ROOT / "models" / "unisolar_weather_forecaster.joblib"
DEMAND_MODEL_PATH = ROOT / "models" / "site_demand_forecaster.joblib"
SYNTHETIC_SOLAR_PATH = ROOT / "data" / "processed" / "demo_solar.csv"
SYNTHETIC_DEMAND_PATH = ROOT / "data" / "processed" / "synthetic_site_demand.csv"

st.sidebar.markdown("### Deployment site")
latitude = st.sidebar.number_input("Latitude", value=51.5085, format="%.4f")
longitude = st.sidebar.number_input("Longitude", value=-0.12574, format="%.5f")
site_capacity_mw = st.sidebar.number_input("Installed solar capacity (MW)", min_value=0.001, value=400.0, step=10.0)
site_capacity_mw = float(site_capacity_mw)

st.sidebar.markdown("### Battery")
battery_capacity_mwh = st.sidebar.number_input("Usable battery capacity (MWh)", min_value=0.0, value=2500.0, step=10.0)
battery_capacity_mwh = float(battery_capacity_mwh)
st.sidebar.caption("Present battery reserve is estimated by the synchronized simulation because live battery telemetry is not available. Forecast battery state is carried continuously from today into each following day.")

@st.cache_data(ttl=900, show_spinner=False)
def get_weather(lat: float, lon: float):
    return fetch_open_meteo_hourly(lat, lon, forecast_days=8)

@st.cache_data(ttl=86400, show_spinner=False)
def get_location_name(lat: float, lon: float):
    try:
        return reverse_geocode(lat, lon)
    except Exception:
        return f"{lat:.4f}, {lon:.4f}"

@st.cache_resource(show_spinner=False)
def get_model(path: str):
    return load_bundle(path)

@st.cache_data(show_spinner=False)
def get_simulation_data(path: str):
    return pd.read_csv(path)

try:
    weather = get_weather(float(latitude), float(longitude))
except Exception as exc:
    st.error(f"Could not fetch live/forecast weather: {exc}")
    st.info("Internet access is required for the weather forecast.")
    st.stop()

location_name = get_location_name(float(latitude), float(longitude))
if not SOLAR_MODEL_PATH.exists() or not DEMAND_MODEL_PATH.exists():
    st.error("Forecasting models are not available. Train the solar and demand models first.")
    st.code("python workflows/generate_synthetic_demand.py\npython workflows/train_demand_model.py\npython workflows/prepare_unisolar.py\npython workflows/train_unisolar_weather_model.py")
    st.stop()

try:
    solar_bundle = get_model(str(SOLAR_MODEL_PATH))
    demand_bundle = get_model(str(DEMAND_MODEL_PATH))
except Exception as exc:
    st.error(f"Could not load forecasting models: {exc}")
    st.info("If the joblib files were created with another scikit-learn version, retrain both models in the current environment.")
    st.code("python workflows/generate_synthetic_demand.py\npython workflows/train_demand_model.py\npython workflows/prepare_unisolar.py\npython workflows/train_unisolar_weather_model.py")
    st.stop()

if not SYNTHETIC_SOLAR_PATH.exists() or not SYNTHETIC_DEMAND_PATH.exists():
    st.error("Synthetic simulation traces are missing. Generate the demo solar and demand data first.")
    st.stop()
solar_sim_data = get_simulation_data(str(SYNTHETIC_SOLAR_PATH))
demand_sim_data = get_simulation_data(str(SYNTHETIC_DEMAND_PATH))

@st.fragment(run_every=5.0)
def render_synchronized_dashboard():
    live_weather = get_weather(float(latitude), float(longitude))
    current_payload = live_weather.attrs.get("current", {}) or {}
    utc_offset_seconds = float(live_weather.attrs.get("utc_offset_seconds") or 0.0)
    local_now = (pd.Timestamp.now(tz="UTC") + pd.Timedelta(seconds=utc_offset_seconds)).tz_localize(None)

    solar_model = solar_bundle["model"]
    solar_features = solar_bundle["feature_columns"]
    frame = add_time_features(live_weather.copy())
    missing = [c for c in solar_features if c not in frame.columns]
    if missing:
        st.error(f"Weather/solar-model feature mismatch. Missing: {missing}")
        return
    ratio = np.asarray(solar_model.predict(frame[solar_features]), dtype=float)
    ratio = np.clip(ratio, 0.0, 1.10)
    ratio = np.where(frame["Ghi"].fillna(0).to_numpy() <= 1.0, 0.0, ratio)
    frame["expected_ratio"] = ratio
    frame["expected_power_mw"] = ratio * float(site_capacity_mw)

    demand_model = demand_bundle["model"]
    demand_features = demand_bundle["feature_columns"]
    demand_frame = future_demand_features(frame["timestamp"])
    missing_demand = [c for c in demand_features if c not in demand_frame.columns]
    if missing_demand:
        st.error(f"Demand-model feature mismatch. Missing: {missing_demand}")
        return
    frame["forecast_load_mw"] = np.maximum(0.0, np.asarray(demand_model.predict(demand_frame[demand_features]), dtype=float))
    frame["date"] = frame["timestamp"].dt.date

    current_hour = local_now.floor("h")
    nearest_idx = (frame["timestamp"] - current_hour).abs().idxmin()
    current = frame.loc[nearest_idx]

    simulated_solar_mw, _ = synthetic_live_values(
        solar_sim_data, demand_sim_data, local_now, float(site_capacity_mw)
    )
    current_demand_features = future_demand_features(pd.DatetimeIndex([local_now]))
    current_demand_mw = max(
        0.0,
        float(demand_model.predict(current_demand_features[demand_features])[0]),
    )
    estimated_battery_mwh = estimate_present_battery(
        solar_sim_data, demand_sim_data, local_now,
        float(site_capacity_mw), float(battery_capacity_mwh), initial_fraction=0.5,
    )

    def current_or_row(api_key: str, row_key: str, default=0.0):
        value = current_payload.get(api_key)
        if value is None:
            value = current.get(row_key, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    current_temp = current_or_row("temperature_2m", "AirTemperature")
    current_humidity = current_or_row("relative_humidity_2m", "RelativeHumidity")
    current_cloud = current_or_row("cloud_cover", "CloudOpacity")
    current_ghi = current_or_row("shortwave_radiation", "Ghi")
    condition, weather_icon = describe_weather_code(current_payload.get("weather_code"))
    timezone_abbr = live_weather.attrs.get("timezone_abbreviation") or "local"
    local_time_text = local_now.strftime("%A, %d %B %Y · %H:%M:%S")

    st.markdown(
        f"""
        <div class="weather-card">
          <div class="location">📍 {html.escape(location_name)}</div>
          <div class="local-time">{local_time_text} {html.escape(str(timezone_abbr))} · 5-second synchronization heartbeat</div>
          <div class="weather-main"><div class="weather-icon">{weather_icon}</div><div class="temp">{current_temp:.0f}°C</div><div><div class="condition">{html.escape(condition)}</div><div class="local-time">Weather inputs for expected solar generation</div></div></div>
          <div class="weather-details">
            <div class="detail"><div class="detail-label">GHI</div><div class="detail-value">{current_ghi:.0f} W/m²</div></div>
            <div class="detail"><div class="detail-label">Cloud cover</div><div class="detail-value">{current_cloud:.0f}%</div></div>
            <div class="detail"><div class="detail-label">Humidity</div><div class="detail-value">{current_humidity:.0f}%</div></div>
            <div class="detail"><div class="detail-label">Temperature</div><div class="detail-value">{current_temp:.0f} °C</div></div>
            <div class="detail"><div class="detail-label">Installed solar</div><div class="detail-value">{site_capacity_mw:.2f} MW</div></div>
          </div>
        </div>
        """, unsafe_allow_html=True,
    )

    st.title("Solar & Fuel Readiness")
    st.caption("Forecast expected solar generation, combine it with demand and battery state, and warn in advance when fuel-backed electrical energy is required.")

    expected_now = float(current["expected_power_mw"])
    simulated_solar_pct = 100.0 * simulated_solar_mw / float(site_capacity_mw) if site_capacity_mw > 0 else 0.0
    expected_pct = 100.0 * expected_now / float(site_capacity_mw) if site_capacity_mw > 0 else 0.0

    st.markdown("### Live — present status")
    st.caption(f"Synchronized at {local_now.strftime('%H:%M:%S')} · refresh heartbeat: 5 seconds")
    cols = st.columns(4)
    with cols[0]:
        st.metric("Solar Power", f"{simulated_solar_mw:.3f} MW")
        st.markdown('<div class="small-note">(simulated value)</div>', unsafe_allow_html=True)
        st.caption(f"{simulated_solar_pct:.1f}% of installed capacity")
    with cols[1]:
        st.metric("Expected Site Demand", f"{current_demand_mw:.3f} MW")
        st.markdown('<div class="small-note"><b>(ML estimate at current timestamp)</b></div>', unsafe_allow_html=True)
    with cols[2]:
        st.metric("Battery Reserve", f"{estimated_battery_mwh:.3f} / {battery_capacity_mwh:.3f} MWh")
        st.markdown('<div class="small-note">(estimated value)</div>', unsafe_allow_html=True)
    with cols[3]:
        st.metric("Expected Solar Generation", f"{expected_now:.3f} MW")
        st.markdown('<div class="small-note"><b>(ML forecast)</b></div>', unsafe_allow_html=True)
        st.caption(f"{expected_pct:.1f}% of installed capacity")

    today = local_now.date()
    today_frame = frame[(frame["date"] == today) & (frame["timestamp"] >= current_hour)].copy()
    remaining_first_hour = max(0.0, min(1.0, 1.0 - (local_now.minute * 60 + local_now.second) / 3600.0))
    today_sim = simulate_battery_and_fuel(
        today_frame,
        battery_capacity_mwh=float(battery_capacity_mwh),
        initial_battery_mwh=float(estimated_battery_mwh),
        first_interval_hours=remaining_first_hour,
    )
    today_periods = shortage_periods(today_sim)
    today_summary = summarize_energy(today_sim)

    st.markdown("### Today — fuel readiness")
    summary_cols = st.columns(3)
    summary_cols[0].metric("Expected solar energy remaining", f"{today_summary['solar_energy_mwh']:.3f} MWh")
    summary_cols[1].metric("Expected demand remaining", f"{today_summary['demand_energy_mwh']:.3f} MWh")
    summary_cols[2].metric("Fuel-backed energy required", f"{today_summary['fuel_required_mwh']:.3f} MWh")

    if today_periods:
        for period in today_periods:
            st.markdown(
                f'<div class="warning-box">⚠ <b>{period.fuel_energy_mwh:.3f} MWh fuel-backed energy required by {period.start.strftime("%H:%M")}</b><br>'
                f'<span class="small-note">Covers forecast shortage until {period.end.strftime("%H:%M")}</span></div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown('<div class="ok-box">✓ <b>No fuel-backed energy forecast for the remaining hours today.</b></div>', unsafe_allow_html=True)

    with st.expander("View today's hourly calculation"):
        if today_sim.empty:
            st.info("No remaining hourly forecast is available for today.")
        else:
            hourly = today_sim[["timestamp", "solar_energy_mwh", "demand_energy_mwh", "battery_reserve_mwh", "fuel_required_mwh"]].copy()
            hourly.columns = ["Time", "Expected solar (MWh)", "Expected demand (MWh)", "Battery reserve (MWh)", "Fuel required (MWh)"]
            hourly["Time"] = pd.to_datetime(hourly["Time"]).dt.strftime("%H:%M")
            for c in hourly.columns[1:]:
                hourly[c] = (hourly[c]).round(3)
            st.dataframe(hourly, use_container_width=True, hide_index=True)
            st.caption("Battery reserve is the energy available at the beginning of each forecast interval. End-state is calculated internally.")

    future_dates = sorted(d for d in frame["date"].unique() if d > today)[:7]
    rows = []
    carry_battery_mwh = (
        float(today_sim["battery_end_mwh"].iloc[-1])
        if not today_sim.empty else float(estimated_battery_mwh)
    )
    for i, day in enumerate(future_dates, start=1):
        day_frame = frame[frame["date"] == day].copy()
        day_start_battery_mwh = carry_battery_mwh
        sim = simulate_battery_and_fuel(
            day_frame,
            battery_capacity_mwh=float(battery_capacity_mwh),
            initial_battery_mwh=day_start_battery_mwh,
        )
        summary = summarize_energy(sim)
        carry_battery_mwh = (
            float(sim["battery_end_mwh"].iloc[-1])
            if not sim.empty else day_start_battery_mwh
        )
        label = "Tomorrow" if i == 1 else pd.Timestamp(day).strftime("%a")
        rows.append({
            "label": label,
            "date": pd.Timestamp(day).strftime("%d/%m/%Y"),
            "start_battery_mwh": day_start_battery_mwh,
            "end_battery_mwh": carry_battery_mwh,
            "hourly": sim,
            **summary,
        })

    st.markdown("### 7-day fuel readiness forecast")
    st.caption("Battery reserve is carried continuously across day boundaries. Calculations are hourly internally and presented as daily outcomes.")

    if rows:
        st.caption("Each day shows its summary before expansion. Turn on hourly evidence to inspect the weather, forecast, battery and fuel calculations for that day.")
        for r in rows:
            fuel = r["fuel_required_mwh"]
            readiness = f"⚠ PREPARE {fuel:.3f} MWh" if fuel > 0.5 else "✓ No fuel-backed energy forecast"

            st.markdown(f"#### {r['label']} · {r['date']} — {readiness}")
            daily_cols = st.columns(5)
            daily_cols[0].metric("Expected solar energy", f"{r['solar_energy_mwh']:.3f} MWh")
            daily_cols[1].metric("Expected demand", f"{r['demand_energy_mwh']:.3f} MWh")
            daily_cols[2].metric("Start battery", f"{r['start_battery_mwh']:.3f} MWh")
            daily_cols[3].metric("End battery", f"{r['end_battery_mwh']:.3f} MWh")
            daily_cols[4].metric("Fuel-backed energy", f"{fuel:.3f} MWh")

            day_key = pd.to_datetime(r["date"], dayfirst=True).strftime("%Y%m%d")
            show_hourly = st.toggle(
                "Show hourly weather & calculation evidence",
                key=f"show_hourly_evidence_{day_key}",
            )

            if show_hourly:
                hourly = r["hourly"].copy()
                if hourly.empty:
                    st.info("No hourly forecast evidence is available for this day.")
                else:
                    hourly["Condition"] = hourly.get("WeatherCode", pd.Series(index=hourly.index, dtype=float)).apply(
                        lambda code: describe_weather_code(code)[0] if pd.notna(code) else "Unknown"
                    )
                    evidence = hourly[[
                        "timestamp", "Condition", "Ghi", "CloudOpacity", "AirTemperature", "RelativeHumidity",
                        "expected_power_mw", "forecast_load_mw", "solar_energy_mwh", "demand_energy_mwh",
                        "battery_reserve_mwh", "battery_end_mwh", "fuel_required_mwh",
                    ]].copy()
                    evidence.columns = [
                        "Time", "Weather", "GHI (W/m²)", "Cloud (%)", "Temp (°C)", "Humidity (%)",
                        "Expected solar (MW)", "Expected demand (MW)", "Solar energy (MWh)",
                        "Demand energy (MWh)", "Battery start (MWh)", "Battery end (MWh)", "Fuel required (MWh)",
                    ]
                    evidence["Time"] = pd.to_datetime(evidence["Time"]).dt.strftime("%H:%M")
                    numeric_columns = [c for c in evidence.columns if c not in ("Time", "Weather")]
                    evidence[numeric_columns] = evidence[numeric_columns].round(3)
                    st.dataframe(evidence, use_container_width=True, hide_index=True)
                    st.caption(
                        "Hourly evidence uses the weather-provider forecast for that hour. "
                        "Expected solar and demand are model outputs; battery and fuel values are calculated sequentially from those forecasts."
                    )
            st.divider()
    else:
        st.info("The weather provider did not return enough future calendar days.")

    with st.expander("Calculation & model assumptions"):
        st.markdown(
            """
**5-second heartbeat.** The dashboard re-synchronizes present simulated solar, the demand-model estimate for the exact current timestamp, estimated battery reserve, expected solar generation and downstream readiness calculations every five seconds. Weather API responses and model objects are cached so synchronization does not imply unnecessary external API calls.

**Simulation traces are not the research scope.** Synthetic solar and demand records provide demonstration telemetry where physical site sensors are unavailable. They are not used as evidence for the solar-forecasting research contribution.

**Hourly calculation, daily presentation.** Expected solar and expected demand are converted to hourly energy. Solar surplus charges the battery up to usable capacity. When demand exceeds solar, available battery energy is used first. Only remaining unmet energy is counted as **fuel-backed electrical energy required (MWh)** in the dashboard.

For **today**, planning starts from the heartbeat's estimated present battery reserve. The resulting battery end-state is carried into tomorrow, and the reserve continues across every subsequent day boundary.

The system does **not** estimate litres of fuel. Battery efficiency, degradation and charge/discharge power limits are outside the current capstone scope.
            """
        )
        st.write("Solar forecast source:", solar_bundle.get("metadata", {}).get("dataset_name", "unknown"))
        st.write("Solar forecast model:", solar_bundle.get("metadata", {}).get("selected_model", "unknown"))
        st.write("Demand planning model:", demand_bundle.get("metadata", {}).get("selected_model", "unknown"))

    st.caption("Forecast assumptions: deployment cloud cover proxies the UNISOLAR cloud variable; forecast weather differs from observed training weather; geographical transfer beyond UNISOLAR sites requires separate validation.")

    # Training evidence is generated by workflows/train_unisolar_weather_model.py.
    # Keep it at the very end of the dashboard and collapsed by default.
    with st.expander("UNISOLAR end-of-dataset model fit", expanded=False):
        end_fit_plot = ROOT / "models" / "unisolar_end_fit.png"
        end_fit_csv = ROOT / "models" / "unisolar_end_fit.csv"

        if end_fit_plot.exists():
            st.image(str(end_fit_plot), caption="Actual vs predicted generation proportion — final 7 days of UNISOLAR test data", use_container_width=True)

        if end_fit_csv.exists():
            end_fit = pd.read_csv(end_fit_csv)
            if {"timestamp", "actual_generation_proportion", "predicted_generation_proportion"}.issubset(end_fit.columns):
                end_fit["timestamp"] = pd.to_datetime(end_fit["timestamp"])
                fit_chart = (
                    end_fit.groupby("timestamp", as_index=False)[
                        ["actual_generation_proportion", "predicted_generation_proportion"]
                    ].mean().set_index("timestamp")
                )
                fit_chart.columns = ["Actual generation proportion", "Predicted generation proportion"]
                st.line_chart(fit_chart, use_container_width=True)

        if not end_fit_plot.exists() and not end_fit_csv.exists():
            st.info("End-of-dataset fit evidence is not available yet. Run python workflows/train_unisolar_weather_model.py to generate it.")


render_synchronized_dashboard()
