"""Single-screen, real-time replay of a predefined LiU-ICE engine dataset."""

from __future__ import annotations

from html import escape
from pathlib import Path
import random

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from enginetwin.simulator import simulate_window
from engine_lab.liu_ice import fit_sensor_consistency_models, sensor_consistency_evidence


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "data" / "liu_ice" / "processed_features.csv.gz"
MODEL_BUNDLE = ROOT / "data" / "liu_ice" / "model_bundle.joblib"
TICK_SECONDS = 0.75
VISIBLE_WINDOWS = 32
BASELINE_RUN_ID = "ftp75city2_NF"
ENGINEGUARD_THRESHOLD = 1.0
CONFIRMATION_WINDOWS = 3
CONFIRMATION_HITS = 2
CONFIRMATION_PEAK = 1.25
EVIDENCE_DISPLAY_CAP = 4.0

SCENARIOS = {
    "wltp_f_pim_105": {
        "name": "Intake manifold pressure drift",
        "code": "PIM · +5%",
        "message": "The intake manifold pressure sensor is biased high by 5%.",
        "fault_sensor": "intake_manifold_pressure",
        "failure_delay_s": 75.0,
        "extra_drift": 0.42,
    },
    "ftp75city2_f_pic_115": {
        "name": "Intercooler pressure drift",
        "code": "PIC · +15%",
        "message": "The intercooler pressure sensor is biased high by 15%.",
        "fault_sensor": "Intercooler_pressure",
        "failure_delay_s": 60.0,
        "extra_drift": 0.55,
    },
    "wltp_f_waf_115": {
        "name": "Air mass flow over-read",
        "code": "WAF · +15%",
        "message": "The air mass flow sensor is reporting 15% above reference.",
        "fault_sensor": "air_mass_flow",
        "failure_delay_s": 70.0,
        "extra_drift": 0.50,
    },
}

SENSORS = {
    "Intercooler_pressure": ("Intercooler pressure", "kPa", lambda x: x / 1000),
    "intercooler_temperature": ("Intercooler outlet", "°C", lambda x: x - 273.15),
    "intake_manifold_pressure": ("Manifold pressure", "kPa", lambda x: x / 1000),
    "air_mass_flow": ("Air mass flow", "g/s", lambda x: x * 1000),
    "engine_speed": ("Engine speed", "rpm", lambda x: x * 60 / (2 * np.pi)),
    "throttle_position": ("Throttle", "°", lambda x: x),
    "wastegate_position": ("Wastegate", "%", lambda x: x * 100),
    "injected_fuel_mass": ("Injected fuel", "g/s", lambda x: x * 1000),
    "ambient_pressure": ("Ambient pressure", "kPa", lambda x: x / 1000),
    "ambient_temperature": ("Ambient temperature", "°C", lambda x: x - 273.15),
}

SENSOR_CODES = {
    "Intercooler_pressure": "f_pic",
    "intercooler_temperature": "f_itc",
    "intake_manifold_pressure": "f_pim",
    "air_mass_flow": "f_waf",
    "engine_speed": "f_rpm",
    "throttle_position": "f_thr",
    "wastegate_position": "f_wg",
    "injected_fuel_mass": "f_fuel",
    "ambient_pressure": "f_ambp",
    "ambient_temperature": "f_ambt",
}
MANUAL_FAULT_SIGNATURES = {
    "Intercooler_pressure": {"intake_manifold_pressure": -0.08, "air_mass_flow": -0.06, "engine_speed": -0.03},
    "intercooler_temperature": {"Intercooler_pressure": -0.04, "air_mass_flow": -0.05, "engine_speed": -0.02},
    "intake_manifold_pressure": {"air_mass_flow": -0.07, "engine_speed": -0.04, "injected_fuel_mass": -0.03},
    "air_mass_flow": {"injected_fuel_mass": 0.06, "engine_speed": -0.03, "intake_manifold_pressure": -0.04},
    "engine_speed": {"injected_fuel_mass": -0.05, "air_mass_flow": -0.04, "intake_manifold_pressure": -0.03},
    "throttle_position": {"injected_fuel_mass": 0.08, "engine_speed": 0.05, "air_mass_flow": 0.04},
    "wastegate_position": {"intake_manifold_pressure": -0.05, "air_mass_flow": 0.03, "engine_speed": -0.02},
    "injected_fuel_mass": {"engine_speed": 0.03, "air_mass_flow": 0.02, "intake_manifold_pressure": -0.02},
    "ambient_pressure": {"intake_manifold_pressure": 0.03, "air_mass_flow": 0.02, "engine_speed": -0.01},
    "ambient_temperature": {"intercooler_temperature": 0.03, "air_mass_flow": -0.02, "injected_fuel_mass": 0.02},
}
MANUAL_SCENARIO = {
    "name": "Manual sensor fault",
    "code": "MANUAL",
    "message": "Manual relative bias applied with a deterministic coupled sensor signature.",
    "failure_delay_s": 65.0,
    "extra_drift": 0.50,
}

PLOT_COLORS = ["#5ee5ff", "#f7bd66", "#8c9eff", "#64f2c8"]
DETECTOR_LABELS = {
    "f_pic": "Intercooler pressure",
    "f_pim": "Manifold pressure",
    "f_waf": "Air mass flow",
    "f_itc": "Intercooler outlet temperature",
    "f_rpm": "Engine speed",
    "f_thr": "Throttle",
    "f_wg": "Wastegate position",
    "f_fuel": "Injected fuel",
    "f_ambp": "Ambient pressure",
    "f_ambt": "Ambient temperature",
}
DETECTOR_BY_SENSOR = SENSOR_CODES


def _page_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=Manrope:wght@400;500;600&display=swap');
        :root {
          --void:#06121b; --panel:#0b1c28; --panel-hi:#102735; --line:#244052;
          --ice:#d7f5ff; --cyan:#5ee5ff; --mint:#64f2c8; --amber:#f7bd66;
          --red:#ff6878; --muted:#7e9aaa;
        }
        html, body, [class*="css"] { font-family:'Manrope',sans-serif; }
        .stApp, [data-testid="stAppViewContainer"] {
          background:
            radial-gradient(circle at 72% -20%, rgba(49,128,164,.20), transparent 38%),
            linear-gradient(145deg, #071722 0%, var(--void) 54%, #081925 100%);
          color:var(--ice);
        }
        [data-testid="stHeader"], [data-testid="stToolbar"], footer { display:none !important; }
        .block-container { max-width:1540px; padding:1.05rem 1.6rem 1.2rem; }
        .stMarkdown, .stMarkdown p, label, [data-testid="stCaptionContainer"] { color:var(--ice) !important; }
        h1,h2,h3 { font-family:'Barlow Condensed',sans-serif !important; color:var(--ice) !important; }
        div[data-testid="stHorizontalBlock"] { gap:.75rem; }
        .cockpit-head { display:flex; align-items:flex-end; justify-content:space-between; gap:1rem; margin-bottom:.65rem; }
        .brand { font:600 .69rem 'IBM Plex Mono'; letter-spacing:.22em; color:var(--cyan); text-transform:uppercase; }
        .title { font:700 2.15rem/1 'Barlow Condensed'; letter-spacing:.01em; color:var(--ice); margin:.22rem 0 .16rem; }
        .subtitle { font-size:.82rem; color:var(--muted); }
        .live-pill { display:flex; align-items:center; gap:.5rem; border:1px solid #24617a; background:#0a2634; padding:.42rem .68rem; font:600 .67rem 'IBM Plex Mono'; letter-spacing:.08em; color:var(--cyan); }
        .live-dot { width:7px; height:7px; border-radius:50%; background:var(--cyan); box-shadow:0 0 12px var(--cyan); animation:pulse 1.25s infinite; }
        @keyframes pulse { 50% { opacity:.35; transform:scale(.72); } }
        .control-rail { border:1px solid var(--line); background:rgba(10,28,40,.88); padding:.55rem .7rem; margin-bottom:.7rem; }
        [data-testid="stButton"] button { width:100%; min-height:2.15rem; border:1px solid #2d5267 !important; border-radius:2px; background:#0e2837 !important; color:var(--ice) !important; font:600 .72rem 'IBM Plex Mono'; }
        [data-testid="stButton"] button:hover { border-color:var(--cyan) !important; color:var(--cyan) !important; }
        [data-testid="stButton"] button:focus-visible { outline:2px solid var(--cyan); outline-offset:2px; }
        .clock { height:100%; display:flex; flex-direction:column; justify-content:center; border-left:1px solid var(--line); padding-left:1rem; }
        .clock strong { font:600 1.25rem 'IBM Plex Mono'; color:var(--ice); }
        .clock span { font:500 .61rem 'IBM Plex Mono'; letter-spacing:.12em; color:var(--muted); text-transform:uppercase; }
        .kpi-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:.6rem; margin-bottom:.7rem; }
        .kpi { position:relative; overflow:hidden; min-height:74px; padding:.68rem .78rem; border:1px solid var(--line); background:linear-gradient(135deg, rgba(16,39,53,.96), rgba(9,27,39,.96)); }
        .kpi:after { content:''; position:absolute; inset:auto 0 0; height:2px; background:var(--accent,var(--cyan)); box-shadow:0 0 14px var(--accent,var(--cyan)); }
        .k-label { font:500 .59rem 'IBM Plex Mono'; letter-spacing:.13em; color:var(--muted); text-transform:uppercase; }
        .k-value { font:700 1.72rem/1 'Barlow Condensed'; color:var(--ice); margin:.28rem 0 .15rem; }
        .k-note { font-size:.67rem; color:var(--muted); }
        .panel { border:1px solid var(--line); background:rgba(8,25,36,.90); padding:.7rem; }
        .panel-label { font:600 .62rem 'IBM Plex Mono'; color:var(--muted); letter-spacing:.14em; text-transform:uppercase; margin-bottom:.35rem; }
        .engine-core { display:grid; place-items:center; min-height:162px; position:relative; overflow:hidden; }
        .engine-core:before { content:''; position:absolute; width:245px; height:245px; border:1px solid rgba(94,229,255,.12); border-radius:50%; box-shadow:0 0 0 30px rgba(94,229,255,.02),0 0 0 60px rgba(94,229,255,.015); }
        .health-ring { --score:80; --ring:var(--cyan); width:138px; height:138px; display:grid; place-items:center; border-radius:50%; background:conic-gradient(var(--ring) calc(var(--score)*1%), #17313f 0); filter:drop-shadow(0 0 16px rgba(94,229,255,.18)); }
        .health-ring:before { content:''; position:absolute; width:116px; height:116px; border-radius:50%; background:#081a25; border:1px solid #274556; }
        .health-copy { position:relative; z-index:2; text-align:center; }
        .health-copy strong { display:block; font:700 2.3rem/1 'Barlow Condensed'; }
        .health-copy span { font:500 .57rem 'IBM Plex Mono'; letter-spacing:.13em; color:var(--muted); text-transform:uppercase; }
        .state-line { display:flex; align-items:center; justify-content:center; gap:.4rem; margin-top:.55rem; font:600 .65rem 'IBM Plex Mono'; color:var(--state); letter-spacing:.08em; }
        .bus { display:grid; grid-template-columns:120px repeat(10,minmax(0,1fr)); align-items:stretch; border:1px solid var(--line); background:#071923; margin:.7rem 0; overflow:hidden; }
        .bus-title { padding:.58rem .68rem; border-right:1px solid var(--line); font:600 .58rem 'IBM Plex Mono'; letter-spacing:.12em; color:var(--cyan); text-transform:uppercase; }
        .packet { position:relative; overflow:hidden; padding:.48rem .35rem; border-right:1px solid rgba(36,64,82,.7); text-align:center; font:500 .52rem 'IBM Plex Mono'; color:#86a8b8; white-space:nowrap; }
        .packet:after { content:''; position:absolute; left:-45%; bottom:0; width:38%; height:2px; background:var(--packet); box-shadow:0 0 9px var(--packet); animation:packet 1.8s linear infinite; animation-delay:var(--delay); }
        @keyframes packet { to { left:112%; } }
        .sensor-grid { display:grid; grid-template-columns:repeat(5,1fr); gap:.52rem; }
        .sensor { position:relative; min-height:74px; border:1px solid #203d4d; background:rgba(10,30,42,.92); padding:.62rem .66rem; overflow:hidden; }
        .sensor:before { content:''; position:absolute; left:0; top:0; bottom:0; width:2px; background:var(--status); box-shadow:0 0 10px var(--status); }
        .sensor-name { font:500 .56rem 'IBM Plex Mono'; letter-spacing:.08em; color:#7f9bab; text-transform:uppercase; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
        .sensor-value { margin:.22rem 0 .12rem; font:600 1.27rem/1 'Barlow Condensed'; color:var(--ice); }
        .sensor-meta { display:flex; justify-content:space-between; font:500 .54rem 'IBM Plex Mono'; color:#627f8e; }
        .sensor-status { color:var(--status); }
        .event { border-left:2px solid var(--event); padding:.48rem .62rem; background:rgba(15,38,51,.78); margin:.4rem 0; }
        .event strong { display:block; font:600 .65rem 'IBM Plex Mono'; color:var(--event); }
        .event span { font-size:.68rem; color:#88a4b3; }
        .guard-alert { display:flex; align-items:center; justify-content:space-between; gap:1rem; min-height:72px; margin:.1rem 0 .7rem; padding:.72rem 1rem; border:2px solid var(--red); background:linear-gradient(90deg,rgba(104,12,25,.97),rgba(38,8,17,.96)); box-shadow:0 0 28px rgba(255,104,120,.28),inset 0 0 28px rgba(255,104,120,.08); animation:guard-pulse 1.15s ease-in-out infinite alternate; }
        .guard-alert strong { display:block; font:700 1.65rem/1 'Barlow Condensed'; letter-spacing:.05em; color:var(--red); }
        .guard-alert span { display:block; margin-top:.25rem; font:500 .66rem 'IBM Plex Mono'; color:#ffc3ca; letter-spacing:.04em; }
        .guard-alert .guard-mark { flex:0 0 auto; font:700 1rem 'IBM Plex Mono'; color:#fff; border:1px solid #ff9aa5; padding:.55rem .75rem; text-align:center; }
        @keyframes guard-pulse { from { border-color:#ff6878; } to { border-color:#ffb0b8; box-shadow:0 0 42px rgba(255,104,120,.42),inset 0 0 30px rgba(255,104,120,.12); } }
        [data-testid="stToggle"] { padding:.15rem .7rem .2rem; border-left:1px solid var(--line); }
        [data-testid="stToggle"] label p { font:600 .69rem 'IBM Plex Mono' !important; color:var(--cyan) !important; }
        div[data-testid="stPlotlyChart"] { border:1px solid var(--line); background:rgba(8,25,36,.90); }
        .stProgress > div > div > div { background:linear-gradient(90deg,var(--cyan),var(--mint)); }
        [data-testid="stProgressBar"] { background:#122d3a; }
        @media(max-width:900px) { .sensor-grid{grid-template-columns:repeat(2,1fr)} .bus{grid-template-columns:90px repeat(5,1fr)} .packet:nth-child(n+7){display:none} .kpi-grid{grid-template-columns:repeat(2,1fr)} }
        @media(prefers-reduced-motion:reduce){ .live-dot,.packet:after,.guard-alert{animation:none!important} }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_guard_bundle() -> tuple[dict, dict]:
    if MODEL_BUNDLE.is_file():
        bundle = joblib.load(MODEL_BUNDLE)
        active_model = bundle["active"]
    else:
        raise FileNotFoundError("The fitted LiU-ICE model bundle is missing.")
    consistency = bundle.get("consistency")
    targets = {code: sensor for sensor, code in SENSOR_CODES.items()}
    if consistency is None:
        consistency = fit_sensor_consistency_models(active_model["training_windows"], target_sensors=targets)
    else:
        existing_targets = {
            code: card["target_sensor"] for code, card in consistency.get("models", {}).items()
        }
        missing_targets = {code: sensor for code, sensor in targets.items() if code not in existing_targets}
        if missing_targets:
            additional = fit_sensor_consistency_models(
                active_model["training_windows"], target_sensors=missing_targets
            )
            consistency = {
                **consistency,
                "models": {**consistency.get("models", {}), **additional["models"]},
                "target_sensors": {**existing_targets, **missing_targets},
            }
    return active_model, consistency


@st.cache_data(show_spinner=False)
def load_replay(_run_id: str = BASELINE_RUN_ID) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, float]:
    active_model, consistency_model = load_guard_bundle()
    frame = active_model["test_windows"].loc[
        active_model["test_windows"]["trajectory_id"].eq(BASELINE_RUN_ID)
    ].sort_values("window_end_s").reset_index(drop=True).copy()
    if frame.empty:
        raise ValueError(f"Healthy baseline {BASELINE_RUN_ID} is missing from the held-out split.")
    frame["injection_time_s"] = np.nan

    healthy = active_model["training_windows"].loc[active_model["training_windows"]["fault_active"].eq(0)]
    feature_columns = [column for column in healthy.columns if "__" in column]
    centre = healthy[feature_columns].median()
    scale = healthy[feature_columns].std().replace(0, 1e-9)
    robust_z = ((frame[feature_columns] - centre) / scale).abs().clip(0, 12)

    evidence = pd.DataFrame(index=frame.index)
    for sensor in SENSORS:
        columns = [column for column in feature_columns if column.startswith(f"{sensor}__")]
        evidence[sensor] = robust_z[columns].max(axis=1)

    guard_evidence = sensor_consistency_evidence(consistency_model, frame)
    confirmed = pd.DataFrame(False, index=frame.index, columns=guard_evidence.columns)
    for code in guard_evidence:
        hits = guard_evidence[code].ge(ENGINEGUARD_THRESHOLD).astype(int)
        sustained = hits.rolling(CONFIRMATION_WINDOWS, min_periods=1).sum().ge(CONFIRMATION_HITS)
        confirmed[code] = sustained & guard_evidence[code].ge(CONFIRMATION_PEAK)
    peak = guard_evidence.max(axis=1)
    health = (100 - 62 * np.clip((peak - 0.5) / 1.5, 0, 1)).clip(18, 100)

    # The replay stays healthy; its measured consistency residuals still vary by window.
    return (
        frame,
        frame.copy(),
        evidence,
        guard_evidence,
        confirmed,
        health,
        pd.Series(0.0, index=frame.index),
        float("inf"),
    )


def _value(sensor: str, raw: float) -> tuple[str, str]:
    _, unit, convert = SENSORS[sensor]
    value = float(convert(raw))
    if abs(value) >= 100:
        return f"{value:,.0f}", unit
    if abs(value) >= 10:
        return f"{value:,.1f}", unit
    return f"{value:,.2f}", unit


def _guard_confirmation(scores: pd.DataFrame) -> pd.DataFrame:
    confirmed = pd.DataFrame(False, index=scores.index, columns=scores.columns)
    for code in scores:
        hits = scores[code].ge(ENGINEGUARD_THRESHOLD).astype(int)
        sustained = hits.rolling(CONFIRMATION_WINDOWS, min_periods=1).sum().ge(CONFIRMATION_HITS)
        confirmed[code] = sustained & scores[code].ge(CONFIRMATION_PEAK)
    return confirmed


def _create_triggered_event(frame: pd.DataFrame, scenario_id: str, consistency_model: dict, start: int) -> dict:
    scenario = SCENARIOS[scenario_id]
    sensor = scenario["fault_sensor"]
    code = DETECTOR_BY_SENSOR[sensor]
    card = consistency_model["models"][code]
    start = min(max(0, int(start)), max(0, len(frame) - CONFIRMATION_WINDOWS))
    return {
        "scenario_id": scenario_id,
        "sensor": sensor,
        "code": code,
        "relative_bias": max(.15, 1.6 * float(card["residual_threshold"])),
        "start": start,
        "onset": float(frame["window_end_s"].iloc[start]),
    }


def _manual_relative_changes(biases: dict[str, float]) -> tuple[str | None, dict[str, float]]:
    active = {sensor: float(value) for sensor, value in biases.items() if abs(float(value)) > 1e-9}
    if not active:
        return None, {}
    primary = max(active, key=lambda sensor: abs(active[sensor]))
    changes = dict(active)
    for sensor, change in MANUAL_FAULT_SIGNATURES[primary].items():
        changes.setdefault(sensor, change)
    return primary, changes


def _manual_candidate(frame: pd.DataFrame, start: int, changes: dict[str, float]) -> pd.DataFrame:
    candidate = frame.copy()
    for sensor, relative_change in changes.items():
        column = f"{sensor}__mean"
        candidate.loc[start:, column] *= 1 + float(relative_change)
    return candidate


def _preview_manual_fault(
    frame: pd.DataFrame,
    consistency_model: dict,
    start: int,
    changes: dict[str, float],
    primary_sensor: str,
) -> tuple[bool, str | None, pd.DataFrame]:
    start = min(max(0, int(start)), max(0, len(frame) - CONFIRMATION_WINDOWS))
    code = SENSOR_CODES[primary_sensor]
    baseline_scores = sensor_consistency_evidence(
        consistency_model, frame, fault_codes=[code]
    ).iloc[start : start + CONFIRMATION_WINDOWS]
    if bool(_guard_confirmation(baseline_scores)[code].any()):
        return False, None, baseline_scores
    candidate = _manual_candidate(frame, start, changes)
    scores = sensor_consistency_evidence(
        consistency_model, candidate, fault_codes=[code]
    ).iloc[start : start + CONFIRMATION_WINDOWS]
    return bool(_guard_confirmation(scores)[code].any()), code, scores


def _create_manual_event(
    frame: pd.DataFrame,
    start: int,
    primary_sensor: str,
    relative_changes: dict[str, float],
    code: str,
    direct_sensors: list[str] | None = None,
) -> dict:
    start = min(max(0, int(start)), max(0, len(frame) - CONFIRMATION_WINDOWS))
    return {
        "kind": "manual",
        "sensor": primary_sensor,
        "code": code,
        "relative_changes": dict(relative_changes),
        "direct_sensors": list(direct_sensors or [primary_sensor]),
        "start": start,
        "onset": float(frame["window_end_s"].iloc[start]),
    }


def _apply_triggered_event(
    frame: pd.DataFrame,
    evidence: pd.DataFrame,
    consistency_model: dict,
    event: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, float]:
    is_manual = event.get("kind") == "manual"
    scenario = MANUAL_SCENARIO if is_manual else SCENARIOS[event["scenario_id"]]
    sensor, start = event["sensor"], int(event["start"])
    column = f"{sensor}__mean"
    if is_manual:
        source = _manual_candidate(frame, start, event["relative_changes"])
    else:
        source = frame.copy()
        card = consistency_model["models"][event["code"]]
        expected = card["estimator"].predict(frame.loc[start:, card["predictors"]])
        source.loc[start:, column] = expected * (1 + float(event["relative_bias"]))
    all_scores = sensor_consistency_evidence(consistency_model, source)
    if is_manual:
        scores = all_scores.copy()
        scores.iloc[:start] = 0.0
    else:
        scores = pd.DataFrame(0.0, index=all_scores.index, columns=all_scores.columns)
        scores[event["code"]] = all_scores[event["code"]]
    confirmed = _guard_confirmation(scores)

    onset = float(event["onset"])
    failure_time = onset + float(scenario["failure_delay_s"])
    progress = pd.Series(
        np.clip((frame["window_end_s"].to_numpy() - onset) / max(failure_time - onset, 1e-9), 0, 1),
        index=frame.index,
    )
    simulated = source.copy()
    p = progress.to_numpy()
    terminal = np.clip((p - .58) / .42, 0, 1) ** 1.5
    simulated.loc[:, column] *= 1 + float(scenario["extra_drift"]) * p**1.45
    simulated["engine_speed__mean"] *= np.clip((1 - .35 * p**2) * (1 - .90 * terminal), 0, None)
    simulated["intake_manifold_pressure__mean"] *= np.clip(1 - .15 * p**2, .08, None)
    simulated["air_mass_flow__mean"] *= np.clip(1 - .15 * p**2, .03, None)
    simulated["injected_fuel_mass__mean"] *= np.clip((1 + .15 * p) * (1 - .85 * terminal), .08, None)
    simulated.loc[progress.ge(1), "engine_speed__mean"] = 0.0

    updated_evidence = evidence.copy()
    changed_sensors = event["relative_changes"] if is_manual else {sensor: None}
    for changed_sensor in changed_sensors:
        updated_evidence.loc[start:, changed_sensor] = 12.0
    peak = scores.max(axis=1)
    health = pd.concat([100 - 62 * np.clip((peak - .5) / 1.5, 0, 1), 100 * (1 - progress**1.35)], axis=1).min(axis=1).clip(0, 100)
    return source, simulated, updated_evidence, scores, confirmed, health, progress, failure_time


def _state(score: float, confirmed: bool, active: bool) -> tuple[str, str, str]:
    if confirmed:
        return "FAULT CONFIRMED", "#ff6878", "sensor consistency failed"
    if score >= ENGINEGUARD_THRESHOLD:
        return "VERIFYING", "#f7bd66", "one residual threshold hit"
    if active:
        return "FAULT UNDER TEST", "#f7bd66", "waiting for corroboration"
    return "NOMINAL", "#64f2c8", "stable"


def _timeline(frame: pd.DataFrame, guard_evidence: pd.DataFrame, confirmed: pd.DataFrame, cursor: int, guard_enabled: bool, failure_time: float, triggered_event: dict | None = None) -> go.Figure:
    start = max(0, cursor - VISIBLE_WINDOWS + 1)
    view = frame.iloc[start : cursor + 1]
    x = view["window_end_s"]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[.72, .28], vertical_spacing=.08)
    selected = ["engine_speed", "intake_manifold_pressure", "air_mass_flow", "throttle_position"]
    if triggered_event and triggered_event["sensor"] not in selected:
        selected[-1] = triggered_event["sensor"]
    for sensor, color in zip(selected, PLOT_COLORS):
        values = frame[f"{sensor}__mean"].iloc[start : cursor + 1]
        # Use one fixed scale for the entire prerecorded run. Rescaling only the
        # visible window made every historical trace jump when a fault arrived.
        all_values = frame[f"{sensor}__mean"]
        lo, hi = all_values.min(), all_values.max()
        normalized = (values - lo) / max(float(hi - lo), 1e-9)
        fig.add_trace(go.Scatter(x=x, y=normalized, mode="lines", name=SENSORS[sensor][0], line=dict(color=color, width=2)), row=1, col=1)
    peak = guard_evidence.max(axis=1)
    visible_evidence = peak.iloc[start : cursor + 1]
    fig.add_trace(go.Scatter(
        x=x,
        y=visible_evidence.clip(upper=EVIDENCE_DISPLAY_CAP),
        customdata=visible_evidence,
        hovertemplate="fault evidence %{customdata:.2f}×<extra></extra>",
        mode="lines",
        name="fault evidence",
        line=dict(color="#ff6878", width=2),
        fill="tozeroy",
        fillcolor="rgba(255,104,120,.10)",
    ), row=2, col=1)
    fig.add_hline(y=ENGINEGUARD_THRESHOLD, line_color="#ff6878", line_dash="dash", opacity=.65, row=2, col=1)
    onset = float(frame["injection_time_s"].iloc[0])
    candidates = frame.loc[confirmed.any(axis=1), "window_end_s"]
    detection_at = float(candidates.iloc[0]) if not candidates.empty else None
    if guard_enabled and detection_at is not None and len(x) and x.min() <= detection_at <= x.max():
        fig.add_vline(x=detection_at, line_color="#ff6878", line_dash="dash", annotation_text="fault confirmed", annotation_position="top right", annotation_font_color="#ff6878", row=1, col=1)
    if len(x) and x.min() <= onset <= x.max():
        fig.add_vline(x=onset, line_color="#f7bd66", line_dash="dot", annotation_text="INJECTION", annotation_position="top left", annotation_font_color="#f7bd66", row=1, col=1)
    if triggered_event:
        event_at = float(triggered_event["onset"])
        if len(x) and x.min() <= event_at <= x.max():
            label = "MANUAL SENSOR INJECTION" if triggered_event.get("kind") == "manual" else "TRIGGERED FAILURE EVENT"
            fig.add_vline(x=event_at, line_color="#f7bd66", line_dash="dash", annotation_text=label, annotation_position="top left", annotation_font_color="#f7bd66", row=1, col=1)
    if len(x) and x.min() <= failure_time <= x.max():
        fig.add_vline(x=failure_time, line_color="#ff6878", line_dash="dot", annotation_text="SIMULATED FAILURE", annotation_position="top right", annotation_font_color="#ff6878", row=1, col=1)
    title = "MEASURED HEALTHY REPLAY · LIVE SENSOR CONSISTENCY" if triggered_event is None else "MEASURED BASELINE · INJECTED FAULT + SIMULATED RESPONSE"
    fig.update_layout(
        height=324, margin=dict(l=18, r=12, t=42, b=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#081923",
        font=dict(family="IBM Plex Mono", size=10, color="#7e9aaa"), title=dict(text=title, x=.018, font=dict(size=12, color="#d7f5ff")),
        legend=dict(orientation="h", y=1.09, x=.98, xanchor="right", font=dict(size=9)), hovermode="x unified",
    )
    fig.update_xaxes(gridcolor="#173342", zeroline=False, title_text="dataset time · seconds", row=2, col=1)
    fig.update_yaxes(gridcolor="#173342", zeroline=False, range=[-.05, 1.05], showticklabels=False, row=1, col=1)
    fig.update_yaxes(gridcolor="#173342", zeroline=False, range=[0, EVIDENCE_DISPLAY_CAP * 1.05], title_text="× limit", row=2, col=1)
    return fig


@st.cache_data(show_spinner=False, max_entries=80)
def _acoustic_figure(rpm: int, load_percent: int, cursor: int) -> go.Figure:
    window = simulate_window(rpm=max(700, rpm), load=np.clip(load_percent / 100, .05, .95), fault="healthy", severity=0, seed=cursor, duration_s=.25)
    samples = min(len(window.sound), 1600)
    time_ms = np.arange(samples) / window.sound_fs * 1000
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=time_ms, y=window.sound[:samples], mode="lines", line=dict(color="#5ee5ff", width=1), name="mic"))
    fig.add_trace(go.Scattergl(x=time_ms, y=window.vibration[:samples] - .65, mode="lines", line=dict(color="#f7bd66", width=1), name="accel"))
    fig.update_layout(
        height=190, margin=dict(l=16, r=12, t=38, b=24), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#081923",
        font=dict(family="IBM Plex Mono", size=9, color="#7e9aaa"), title=dict(text="SYNTHETIC COMBUSTION CARRIER", x=.025, font=dict(size=11, color="#d7f5ff")),
        xaxis=dict(title="milliseconds", gridcolor="#173342", zeroline=False), yaxis=dict(showticklabels=False, gridcolor="#173342", zeroline=False),
        legend=dict(orientation="h", y=1.14, x=1, xanchor="right", font=dict(size=9)),
    )
    return fig


def _box_mesh(x0, x1, y0, y1, z0, z1, color, opacity=1.0) -> go.Mesh3d:
    vertices = [
        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
    ]
    faces = [
        (0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6), (0, 4, 5), (0, 5, 1),
        (1, 5, 6), (1, 6, 2), (2, 6, 7), (2, 7, 3), (3, 7, 4), (3, 4, 0),
    ]
    return go.Mesh3d(
        x=[point[0] for point in vertices], y=[point[1] for point in vertices], z=[point[2] for point in vertices],
        i=[face[0] for face in faces], j=[face[1] for face in faces], k=[face[2] for face in faces],
        color=color, opacity=opacity, flatshading=True, hoverinfo="skip", showscale=False,
        lighting=dict(ambient=.65, diffuse=.75, specular=.25, roughness=.65, fresnel=.08),
    )


def _cylinder_mesh(start, end, radius, color, *, opacity=1.0, segments=14) -> go.Mesh3d:
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    axis = end - start
    axis /= max(float(np.linalg.norm(axis)), 1e-9)
    reference = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < .9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, reference)
    u /= max(float(np.linalg.norm(u)), 1e-9)
    v = np.cross(axis, u)
    ring = [np.cos(2 * np.pi * index / segments) * u + np.sin(2 * np.pi * index / segments) * v for index in range(segments)]
    points = [start + radius * vector for vector in ring] + [end + radius * vector for vector in ring]
    points += [start, end]
    face_i, face_j, face_k = [], [], []
    for index in range(segments):
        following = (index + 1) % segments
        face_i.extend([index, index, 2 * segments, 2 * segments + 1])
        face_j.extend([following, segments + index, following, segments + following])
        face_k.extend([segments + index, segments + following, index, segments + index])
    return go.Mesh3d(
        x=[point[0] for point in points], y=[point[1] for point in points], z=[point[2] for point in points],
        i=face_i, j=face_j, k=face_k, color=color, opacity=opacity,
        flatshading=True, hoverinfo="skip", showscale=False,
        lighting=dict(ambient=.6, diffuse=.8, specular=.3, roughness=.5),
    )


def _engine_cutaway_figure(elapsed: float, *, stopped: bool, failed: bool) -> go.Figure:
    """Animated conceptual inline-four cutaway, synchronized to replay time."""
    if failed:
        block, head, liner, piston, rod, crank = "#b72f35", "#dc4145", "#f08080", "#8f2028", "#a72a31", "#cf343b"
        title = "<b><span style='color:#ff6878'>SIMULATED ENGINE FAILURE · POWER LOST</span></b>"
    elif stopped:
        block, head, liner, piston, rod, crank = "#a34b42", "#cc655d", "#d68d84", "#8f3535", "#a74740", "#cf514b"
        title = "<b><span style='color:#ff6878'>ENGINEGUARD STOP · FAULT CONFIRMED</span></b>"
    else:
        block, head, liner, piston, rod, crank = "#526e7d", "#8399a4", "#b4c3c8", "#d6dde0", "#bd744b", "#17627b"
        title = "<span style='color:#d7f5ff'>INLINE-FOUR CUTAWAY · RUNNING</span>"

    fig = go.Figure()
    fig.add_trace(_box_mesh(-2.25, 2.25, -.62, .62, .36, 2.12, block, .30))
    fig.add_trace(_box_mesh(-2.28, 2.28, -.70, .70, 2.12, 2.42, head, .96))
    for index, x in enumerate([-1.5, -.5, .5, 1.5]):
        phase = 2 * np.pi * (elapsed / 12.0) + index * np.pi / 2
        crank_z = .34 + .26 * np.cos(phase)
        piston_z = 1.42 + .30 * np.cos(phase)
        fig.add_trace(_cylinder_mesh((x, 0, .72), (x, 0, 1.98), .34, liner, opacity=.27))
        fig.add_trace(_box_mesh(x - .27, x + .27, -.26, .26, piston_z - .13, piston_z + .13, piston, .98))
        fig.add_trace(_cylinder_mesh((x, 0, piston_z - .13), (x, 0, crank_z), .075, rod))
        fig.add_trace(_cylinder_mesh((x, 0, crank_z - .07), (x, 0, crank_z + .07), .16, crank))
    fig.add_trace(_cylinder_mesh((-2.35, 0, .34), (2.35, 0, .34), .12, crank))
    fig.update_layout(
        height=290, margin=dict(l=0, r=0, t=42, b=0), paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Mono", size=9, color="#7e9aaa"),
        title=dict(text=title, x=.03, xanchor="left", font=dict(size=11)), showlegend=False,
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            aspectmode="data", camera=dict(eye=dict(x=1.55, y=1.8, z=1.1)), bgcolor="rgba(0,0,0,0)",
        ),
        uirevision="live-engine-cutaway",
    )
    return fig


def _sensor_cards(row: pd.Series, zrow: pd.Series, guard_row: pd.Series, confirmed_row: pd.Series) -> str:
    cards = []
    for sensor, (label, _, _) in SENSORS.items():
        value, unit = _value(sensor, row[f"{sensor}__mean"])
        code = DETECTOR_BY_SENSOR.get(sensor)
        if code not in guard_row:
            code = None
        if code is not None:
            model_value = float(guard_row[code])
            evidence_text = f"guard {model_value:.2f}×"
            if bool(confirmed_row[code]):
                color, status = "#ff6878", "confirmed"
            elif model_value >= ENGINEGUARD_THRESHOLD:
                color, status = "#f7bd66", "single hit"
            else:
                color, status = "#64f2c8", "consistent"
        else:
            model_value = float(abs(zrow[sensor]))
            evidence_text = f"baseline {model_value:.1f}σ"
            if model_value >= 4:
                color, status = "#ff6878", "deviating"
            elif model_value >= 2:
                color, status = "#f7bd66", "watch"
            else:
                color, status = "#64f2c8", "streaming"
        cards.append(
            f'<div class="sensor" style="--status:{color}"><div class="sensor-name">{escape(label)}</div>'
            f'<div class="sensor-value">{value} <small>{escape(unit)}</small></div>'
            f'<div class="sensor-meta"><span>{evidence_text}</span><span class="sensor-status">{status}</span></div></div>'
        )
    return '<div class="sensor-grid">' + "".join(cards) + "</div>"


def _data_bus(zrow: pd.Series) -> str:
    packets = []
    for i, sensor in enumerate(SENSORS):
        color = "#ff6878" if abs(float(zrow[sensor])) >= 4 else "#f7bd66" if abs(float(zrow[sensor])) >= 2 else "#5ee5ff"
        short = SENSORS[sensor][0].replace(" pressure", "").replace(" temperature", " temp")
        packets.append(f'<div class="packet" style="--packet:{color};--delay:-{i * .16:.2f}s">{escape(short)}</div>')
    return '<div class="bus"><div class="bus-title">MODEL FEATURE BUS<br>10 streams</div>' + "".join(packets) + "</div>"


def _events(elapsed: float, onset: float, active: bool, score: float, detected_code: str, scenario: dict, guard_enabled: bool, guard_tripped: bool, degradation: float, engine_failed: bool) -> str:
    entries = [
        ("#5ee5ff", "10 streams synchronized", "Five-second windows arriving from the predefined LiU-ICE replay."),
        ("#64f2c8", "Consistency model online", "Ten healthy-only sensor-consistency signatures are monitored against the replay."),
    ]
    if guard_enabled:
        entries.append(("#5ee5ff", "EngineGaurd armed", f"Shutdown requires two of three limit hits and at least {CONFIRMATION_PEAK:.2f}× evidence in the current window."))
    if guard_tripped:
        entries.append(("#ff6878", "Sensor fault confirmed", f"{DETECTOR_LABELS[detected_code]} residual reached {score:.2f}× its learned limit. Manual inspection required."))
        entries.append(("#f7bd66", "Evidence rule satisfied", f"Two of three windows crossed the limit and current evidence reached at least {CONFIRMATION_PEAK:.2f}×."))
    if active:
        entries.append(("#ff6878", f"{scenario['name']} injected", scenario["message"]))
    else:
        entries.append(("#64f2c8", "Triggered failure event off", "Healthy source replay; no fault or failure progression is active."))
    if degradation > 0 and not guard_tripped:
        entries.append(("#f7bd66", "Simulated degradation", f"Virtual damage progression is at {degradation:.0%}; coupled airflow, pressure, fuel, and speed are deteriorating."))
    if engine_failed:
        entries.append(("#ff6878", "Simulated engine failure", "Health reached zero and the engine stalled. Restart or select a new run."))
    return "".join(f'<div class="event" style="--event:{c}"><strong>{escape(t)}</strong><span>{escape(d)}</span></div>' for c, t, d in entries[-4:])


def render() -> None:
    st.set_page_config(page_title="EngineTwin · Live Simulation", page_icon="◉", layout="wide", initial_sidebar_state="collapsed")
    _page_css()
    if not DATASET.is_file():
        st.error("The prepared LiU-ICE replay is missing. Run `python prepare_liu_ice.py` and reload the dashboard.")
        return

    defaults = {
        "scenario_id": random.choice(list(SCENARIOS)), "sim_cursor": 0, "sim_running": True,
        "guard_enabled": True, "guard_tripped": False, "engine_failed": False,
        "triggered_failure": False, "triggered_event": None, "manual_event": None, "guard_code": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if st.session_state.get("guard_model_version") != "manual-all-sensors-v1":
        st.session_state.guard_model_version = "manual-all-sensors-v1"
        st.session_state.triggered_failure = False
        st.session_state.triggered_event = None
        st.session_state.manual_event = None
        st.session_state.guard_tripped = False
        st.session_state.engine_failed = False
        st.session_state.guard_code = None
        st.session_state.sim_cursor = 0
        st.session_state.sim_running = True

    frame, _, base_evidence, base_scores, _, base_health, _, _ = load_replay()
    active_model, consistency_model = load_guard_bundle()

    st.markdown(
        '<div class="cockpit-head"><div><div class="brand">EngineTwin / unified simulation</div>'
        '<div class="title">LIVE ENGINE TELEMETRY</div><div class="subtitle">LiU-ICE replay · inject a manual sensor bias or arm a benchmark preset</div></div>'
        '<div class="live-pill"><span class="live-dot"></span> DATA LINK ACTIVE</div></div>',
        unsafe_allow_html=True,
    )

    def toggle_replay() -> None:
        if st.session_state.guard_tripped or st.session_state.engine_failed:
            st.session_state.sim_cursor = 0
            st.session_state.guard_tripped = False
            st.session_state.guard_code = None
            st.session_state.engine_failed = False
            st.session_state.triggered_failure = False
            st.session_state.triggered_event = None
            st.session_state.manual_event = None
            st.session_state.sim_running = True
        else:
            st.session_state.sim_running = not st.session_state.sim_running

    def reset_replay() -> None:
        st.session_state.sim_cursor = 0
        st.session_state.sim_running = False
        st.session_state.guard_tripped = False
        st.session_state.guard_code = None
        st.session_state.engine_failed = False
        st.session_state.triggered_failure = False
        st.session_state.triggered_event = None
        st.session_state.manual_event = None

    def randomize_run() -> None:
        choices = [key for key in SCENARIOS if key != st.session_state.scenario_id]
        st.session_state.scenario_id = random.choice(choices)
        st.session_state.sim_cursor = 0
        st.session_state.guard_tripped = False
        st.session_state.guard_code = None
        st.session_state.engine_failed = False
        st.session_state.triggered_failure = False
        st.session_state.triggered_event = None
        st.session_state.manual_event = None
        st.session_state.sim_running = True

    controls = st.columns([.68, .68, .78, 1.05, 1.45, .75, 1.8], gap="small")
    play_label = "↻ RESTART" if st.session_state.engine_failed else "↻ REARM" if st.session_state.guard_tripped else "▶ RUN" if not st.session_state.sim_running else "Ⅱ PAUSE"
    controls[0].button(play_label, key="run_pause", on_click=toggle_replay)
    controls[1].button("↺ RESET", key="reset", on_click=reset_replay)
    controls[2].button("⟳ NEW RUN", key="new_run", on_click=randomize_run)
    controls[3].toggle("EngineGaurd", key="guard_enabled")
    controls[4].toggle("Triggered failure event", key="triggered_failure")
    controls[5].markdown('<div class="clock"><span>replay rate</span><strong>6.7×</strong></div>', unsafe_allow_html=True)
    scenario_id = st.session_state.scenario_id
    scenario = SCENARIOS[scenario_id]
    if st.session_state.triggered_failure and st.session_state.manual_event is not None:
        st.session_state.manual_event = None
    manual_event_active = st.session_state.manual_event is not None
    with st.expander("INJECT A SENSOR FAULT", expanded=False):
        st.caption("Set a relative bias for any stream. Injection adds a small, fixed companion signature and arms only when that sensor's own guard model meets the two-of-three rule.")
        slider_columns = st.columns(2, gap="large")
        manual_biases: dict[str, float] = {}
        for index, (sensor, (label, unit, _)) in enumerate(SENSORS.items()):
            with slider_columns[index % 2]:
                manual_biases[sensor] = st.slider(
                    f"{label} · bias",
                    min_value=-80,
                    max_value=200,
                    value=0,
                    step=1,
                    format="%d%%",
                    key=f"manual_bias_{sensor}",
                    disabled=manual_event_active,
                    help=f"Relative change to this {unit} reading from the selected replay window onward.",
                ) / 100
        primary_sensor, relative_changes = _manual_relative_changes(manual_biases)
        direct_sensors = [sensor for sensor, value in manual_biases.items() if abs(value) > 1e-9]
        preview_ok, preview_code, preview_scores = (False, None, pd.DataFrame())
        preview_start = min(int(st.session_state.sim_cursor), max(0, len(frame) - CONFIRMATION_WINDOWS))
        if primary_sensor is not None and not manual_event_active:
            preview_ok, preview_code, preview_scores = _preview_manual_fault(
                frame, consistency_model, preview_start, relative_changes, primary_sensor
            )
            st.caption(
                f"Guard target: {SENSORS[primary_sensor][0]} (largest absolute slider)."
                + (" Other nonzero sliders also apply their own bias." if len(direct_sensors) > 1 else "")
            )
            coupled = [
                f"{SENSORS[sensor][0]} {change:+.0%}"
                for sensor, change in relative_changes.items()
                if sensor not in direct_sensors
            ]
            if coupled:
                st.caption("Fixed companion shifts: " + " · ".join(coupled))
            if preview_ok:
                peak = float(preview_scores[preview_code].max())
                if st.session_state.guard_enabled:
                    st.success(f"Guard preview confirms {DETECTOR_LABELS[preview_code]} at {peak:.2f}× its learned limit. EngineGaurd will stop the replay when the evidence rule is reached.")
                else:
                    st.warning(f"Guard preview confirms {DETECTOR_LABELS[preview_code]} at {peak:.2f}×, but EngineGaurd is off. Enable it to stop the replay.")
            else:
                st.warning("This sensor's bias does not yet satisfy its own guard rule, or its healthy replay already trips that limit here. Increase its magnitude, change direction, or wait for a cleaner replay window.")
        elif manual_event_active:
            st.info("Manual fault is active. Use Reset, New Run, or Rearm after a guard stop to clear it.")
        else:
            st.caption("Move one or more sensor sliders away from 0% to preview an injection.")

        inject_disabled = (
            manual_event_active
            or st.session_state.triggered_failure
            or primary_sensor is None
            or not preview_ok
        )
        if st.session_state.triggered_failure:
            st.caption("Turn off the benchmark preset before injecting a manual fault.")
        if st.button("⚡ INJECT MANUAL FAULT · RUN", key="inject_manual_fault", disabled=inject_disabled, type="primary"):
            st.session_state.manual_event = _create_manual_event(
                frame, preview_start, primary_sensor, relative_changes, str(preview_code), direct_sensors
            )
            st.session_state.triggered_event = None
            st.session_state.guard_tripped = False
            st.session_state.guard_code = None
            st.session_state.engine_failed = False
            st.session_state.sim_running = True
    if st.session_state.triggered_failure and st.session_state.triggered_event is None:
        cursor_start = min(int(st.session_state.sim_cursor), len(frame) - CONFIRMATION_WINDOWS)
        try:
            st.session_state.triggered_event = _create_triggered_event(frame, scenario_id, consistency_model, cursor_start)
            st.session_state.guard_tripped = False
            st.session_state.guard_code = None
            st.session_state.engine_failed = False
            st.session_state.sim_running = True
        except Exception as exc:
            st.session_state.triggered_failure = False
            st.error(f"Could not arm the event: {exc}")
    elif not st.session_state.triggered_failure and st.session_state.triggered_event is not None:
        st.session_state.triggered_event = None
        st.session_state.guard_tripped = False
        st.session_state.guard_code = None
        st.session_state.engine_failed = False

    event = st.session_state.manual_event or (st.session_state.triggered_event if st.session_state.triggered_failure else None)
    scenario = MANUAL_SCENARIO if event and event.get("kind") == "manual" else SCENARIOS[scenario_id]
    if event and event.get("kind") == "manual":
        primary_bias = float(event["relative_changes"][event["sensor"]])
        companions = [
            f"{SENSORS[sensor][0]} {change:+.0%}"
            for sensor, change in event["relative_changes"].items()
            if sensor not in event.get("direct_sensors", [event["sensor"]])
        ]
        coupled_message = "; companion shifts: " + ", ".join(companions) if companions else ""
        scenario = {
            **MANUAL_SCENARIO,
            "message": f"{SENSORS[event['sensor']][0]} biased {primary_bias:+.0%}{coupled_message}.",
        }
    if event:
        frame, simulated_frame, evidence, guard_evidence, confirmed, health, degradation, failure_time = _apply_triggered_event(frame, base_evidence, consistency_model, event)
    else:
        simulated_frame = frame.copy()
        evidence = base_evidence.copy()
        guard_evidence = base_scores.copy()
        confirmed = pd.DataFrame(False, index=frame.index, columns=base_scores.columns)
        health = base_health.copy()
        degradation = pd.Series(0.0, index=frame.index)
        failure_time = float("inf")
    if event is None:
        frame = frame.copy()
        frame["fault_active"] = 0
        frame["injection_time_s"] = np.nan
    run_number = list(SCENARIOS).index(scenario_id) + 1
    controls[6].markdown(
        f'<div class="clock"><span>{escape(scenario["name"]) if event else "healthy baseline"}{" · preset " + str(run_number) + "/3" if event and event.get("kind") != "manual" else ""}</span><strong>{escape(scenario["code"] if event else "NO FAULT")} &nbsp; {len(frame)} FRAMES</strong></div>',
        unsafe_allow_html=True,
    )

    @st.fragment(run_every=TICK_SECONDS)
    def cockpit() -> None:
        cursor = min(int(st.session_state.sim_cursor), len(frame) - 1)
        row, display_row, zrow = frame.iloc[cursor], simulated_frame.iloc[cursor], evidence.iloc[cursor]
        elapsed = float(row["window_end_s"])
        onset = float(event["onset"]) if event else float("inf")
        active = bool(event and elapsed >= onset)
        score_row = guard_evidence.iloc[cursor]
        confirmed_row = confirmed.iloc[cursor]
        peak_score = float(score_row.max())
        confirmed_now = bool(confirmed_row.any())
        detected_code = str(score_row.loc[confirmed_row].idxmax() if confirmed_now else score_row.idxmax())
        decision_score = float(score_row[detected_code])
        health_score = float(health.iloc[cursor])
        state, state_color, note = _state(peak_score, confirmed_now, active)

        if (
            st.session_state.guard_enabled
            and st.session_state.sim_running
            and confirmed_now
        ):
            st.session_state.guard_tripped = True
            st.session_state.guard_code = detected_code
            st.session_state.sim_running = False
            st.rerun()

        if st.session_state.sim_running and elapsed >= failure_time:
            st.session_state.engine_failed = True
            st.session_state.sim_running = False
            st.rerun()

        guard_tripped = bool(st.session_state.guard_tripped)
        engine_failed = bool(st.session_state.engine_failed)
        if guard_tripped:
            detected_code = str(st.session_state.get("guard_code", detected_code))
            decision_score = float(score_row[detected_code])
            state, state_color, note = "ENGINE STOPPED", "#ff6878", "EngineGaurd evidence intervention"
            delay = elapsed - onset
            if delay >= 0:
                timing = f"Injection occurred {delay:.1f}s earlier. Simulated failure prevented with {failure_time - elapsed:.1f}s remaining."
            else:
                timing = f"The inconsistency appeared {-delay:.1f}s before injection and is flagged as a possible false alarm."
            st.markdown(
                f'<div class="guard-alert"><div><strong>⚠ SENSOR FAULT CONFIRMED · ENGINE STOPPED</strong>'
                f'<span>{escape(DETECTOR_LABELS[detected_code])} residual reached {decision_score:.2f}× its learned limit with two of three windows over limit. {timing}</span></div>'
                f'<div class="guard-mark">EVIDENCE-GATED<br>LOCKOUT</div></div>',
                unsafe_allow_html=True,
            )
        elif engine_failed:
            state, state_color, note = "ENGINE FAILED", "#ff6878", "simulated terminal stall"
            health_score = 0.0
            st.markdown(
                '<div class="guard-alert"><div><strong>⚠ ENGINE FAILURE · POWER LOST</strong>'
                '<span>The injected fault progressed through the simulation layer until health reached zero. Engine speed and airflow collapsed.</span></div>'
                '<div class="guard-mark">SIMULATED<br>FAILURE</div></div>',
                unsafe_allow_html=True,
            )

        measured_rpm = int(float(display_row["engine_speed__mean"]) * 60 / (2 * np.pi))
        rpm = 0 if guard_tripped or engine_failed else measured_rpm
        load_percent = int(np.clip(float(display_row["throttle_position__mean"]) * 2.2, 0, 100))
        airflow_value, airflow_unit = _value("air_mass_flow", display_row["air_mass_flow__mean"])
        kpis = [
            ("Engine speed", f"{rpm:,} rpm", "crankshaft estimate", "#5ee5ff"),
            ("Manifold pressure", f"{display_row['intake_manifold_pressure__mean'] / 1000:.1f} kPa", "intake tract", "#8c9eff"),
            ("Air mass flow", f"{airflow_value} {airflow_unit}", "measured + simulated degradation", "#ff6878" if engine_failed else "#f7bd66" if float(degradation.iloc[cursor]) >= .55 else "#64f2c8"),
            ("System state", state, f"{elapsed:07.1f}s · frame {cursor + 1:03d}/{len(frame):03d}", state_color),
        ]
        st.markdown('<div class="kpi-grid">' + "".join(
            f'<div class="kpi" style="--accent:{color}"><div class="k-label">{label}</div><div class="k-value">{value}</div><div class="k-note">{note}</div></div>'
            for label, value, note, color in kpis
        ) + '</div>', unsafe_allow_html=True)

        main, side = st.columns([1.85, 1.15], gap="small")
        with main:
            st.plotly_chart(_timeline(simulated_frame, guard_evidence, confirmed, cursor, st.session_state.guard_enabled, failure_time, event), width="stretch", config={"displayModeBar": False})
        with side:
            ring_color = "#ff6878" if health_score < 50 else "#f7bd66" if health_score < 72 else "#64f2c8"
            st.markdown(
                f'<div class="panel"><div class="panel-label">Health index · model consistency</div><div class="engine-core">'
                f'<div><div class="health-ring" style="--score:{health_score:.0f};--ring:{ring_color}"><div class="health-copy"><strong>{health_score:.0f}</strong><span>health index</span></div></div>'
                f'<div class="state-line" style="--state:{state_color}">● {state}</div></div></div></div>',
                unsafe_allow_html=True,
            )
            st.caption("Consistency heuristic; it does not estimate wear or remaining life.")
            st.progress((cursor + 1) / len(frame), text=f"Replay {100 * (cursor + 1) / len(frame):.1f}%")
            st.markdown('<div class="panel-label">ANIMATED INLINE-FOUR CUTAWAY</div>', unsafe_allow_html=True)
            st.plotly_chart(
                _engine_cutaway_figure(elapsed, stopped=guard_tripped or engine_failed, failed=engine_failed),
                width="stretch", config={"displayModeBar": False},
            )
            st.caption("Conceptual moving assembly; red bold status marks a confirmed stop or simulated failure.")

        st.markdown(_data_bus(zrow), unsafe_allow_html=True)
        lower_left, lower_right = st.columns([1.5, .9], gap="small")
        with lower_left:
            st.markdown('<div class="panel-label">LIVE SENSOR ARRAY · MAX STANDARDIZED MODEL EVIDENCE ACROSS MEAN / STD / RANGE / SLOPE</div>', unsafe_allow_html=True)
            st.markdown(_sensor_cards(display_row, zrow, score_row, confirmed_row), unsafe_allow_html=True)
        with lower_right:
            st.markdown('<div class="panel-label">LIVE EVENTS</div>', unsafe_allow_html=True)
            st.markdown(_events(elapsed, onset, active, decision_score, detected_code, scenario, st.session_state.guard_enabled, guard_tripped, float(degradation.iloc[cursor]), engine_failed), unsafe_allow_html=True)
            st.markdown('<div class="panel-label">SYNTHETIC COMBUSTION SIGNAL</div>', unsafe_allow_html=True)
            st.plotly_chart(_acoustic_figure(measured_rpm, load_percent, cursor), width="stretch", config={"displayModeBar": False})
            st.caption("Carrier follows measured RPM and load; fault evidence comes from the LiU-ICE sensors.")

        if st.session_state.sim_running:
            st.session_state.sim_cursor = 0 if cursor >= len(frame) - 1 else cursor + 1

    cockpit()
