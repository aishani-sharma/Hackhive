from __future__ import annotations

from io import BytesIO
import numpy as np
import pandas as pd
from pathlib import Path
import time
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy.signal import welch, spectrogram
from scipy.stats import kurtosis, skew
from scipy.io import wavfile

from engine_lab.core import (
    PERFORMANCE_CONTROLS, PERFORMANCE_TARGETS,
    build_live_features, build_rul_features, condition_deviation, fit_rul_model,
    load_gengine_trajectory, load_safety_bounds,
    load_performance_data, propose_operating_points, select_features,
)
from engine_lab.liu_ice import (
    FAULT_LABELS, SIGNALS as LIU_SIGNALS, SOURCE_URL, fit_fault_isolator,
    fit_probability_model, simulate_fault_scenario, summarize_dataset,
)
from engine_lab.marine import fit_anomaly_model as fit_marine_model, load_runs as load_marine_runs, summarize as summarize_marine
from engine_lab.signatures import extract_window_signatures, fit_recording_heldout_svm
from engine_lab.cmapss import DATA_DIR as CMAPSS_DATA_DIR, load_fd001

from enginetwin.simulator import simulate_window as simulate_acoustic_window
from enginetwin.signals import (
    order_spectrum, envelope_spectrum, per_cylinder_energy,
    half_order_ratio, band_energy_fraction,
)
from enginetwin.schema import FAULTS as TWIN_FAULTS

st.set_page_config(page_title="EngineTwin · Combustion Lab", page_icon="◉", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700;800&family=DM+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
:root { --ink:#142b3b; --blue:#17627b; --cyan:#007d87; --amber:#b95118; --paper:#f4f7f9; --white:#ffffff; --muted:#506371; --line:#cbd5dc; }
html, body, [class*="css"] { font-family:'DM Sans', sans-serif; color:var(--ink) !important; }
.stApp, [data-testid="stAppViewContainer"] { background:var(--paper) !important; color:var(--ink) !important; }
[data-testid="stHeader"] { background:rgba(244,247,249,.96); }
.stMarkdown, .stMarkdown p, .stMarkdown li, label, [data-testid="stCaptionContainer"], [data-testid="stWidgetLabel"] p { color:var(--ink) !important; }
.stMarkdown p, .stMarkdown li { font-size:1rem; line-height:1.55; }
.block-container { max-width:1500px; padding-top:1.15rem; padding-bottom:1.5rem; }
[data-testid="stSidebar"] { background:#e6eef2; border-right:1px solid #cbd5dc; }
[data-testid="stSidebar"] * { color:var(--ink); }
h1,h2,h3 { font-family:'Barlow Condensed', sans-serif !important; letter-spacing:.01em; color:var(--ink) !important; }
h1 { font-size:2.7rem !important; line-height:1 !important; font-weight:700 !important; margin-bottom:.25rem !important; }
h2 { font-size:1.8rem !important; font-weight:600 !important; }
h3 { font-size:1.18rem !important; font-weight:600 !important; }
.eyebrow { text-transform:uppercase; letter-spacing:.16em; font:600 .7rem 'IBM Plex Mono',monospace; color:var(--blue); }
.lede { font-size:1.02rem; color:#53656d; max-width:760px; line-height:1.55; }
.rule { border-top:1px solid var(--line); margin:.8rem 0 .7rem; }
.metric-card { background:var(--white); border:1px solid var(--line); padding:.75rem .9rem; min-height:84px; }
.metric-label { color:#435866; font:600 .72rem 'IBM Plex Mono',monospace; text-transform:uppercase; letter-spacing:.08em; }
.metric-value { font:700 1.95rem 'Barlow Condensed',sans-serif; color:var(--ink); margin-top:.12rem; line-height:1; }
.metric-note { color:#4b5d68; font-size:.83rem; margin-top:.4rem; }
.signal-card { border-left:3px solid var(--amber); background:#fff3e8; padding:.65rem .8rem; color:#273e4c !important; font-size:.9rem; line-height:1.4; }
.source-card { background:var(--white); border:1px solid var(--line); padding:1rem; height:100%; }
.source-name { font:600 1.2rem 'Barlow Condensed',sans-serif; }
.source-tag { font:600 .7rem 'IBM Plex Mono',monospace; color:var(--blue); text-transform:uppercase; letter-spacing:.09em; }
.stButton > button, .stDownloadButton > button { border-radius:2px; border:1px solid var(--blue); background:var(--blue); color:#fffefa; font-weight:600; }
.stButton > button:hover, .stDownloadButton > button:hover { border-color:var(--amber); background:var(--amber); color:white; }
.stTabs [data-baseweb="tab"] { font-weight:600; }
.stDataFrame { border:1px solid var(--line); }
div[data-testid="stAlert"] { border-radius:2px; }
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap:.55rem; }
[data-testid="stSidebar"] h2 { font-size:1.45rem !important; }
small, .caption { font-size:.84rem; color:#435866; }
code, pre { font-family:'IBM Plex Mono',monospace !important; }
@media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation-duration:.01ms !important; transition-duration:.01ms !important; } }
@media (max-width: 760px) { h1 { font-size:2.1rem !important; } .block-container { padding:1rem .8rem 1rem; } }
</style>
""", unsafe_allow_html=True)

PLOT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#ffffff", font=dict(family="DM Sans", color="#142b3b", size=12),
    margin=dict(l=14, r=14, t=20, b=14), xaxis=dict(gridcolor="#e9ece8", zerolinecolor="#d9ddd8"),
    yaxis=dict(gridcolor="#e9ece8", zerolinecolor="#d9ddd8"),
)

SIGNAL_COLORS = ["#007d87", "#b95118", "#17627b", "#7756a5", "#52804e", "#9f3c32", "#64748b"]


def signal_panels(frame: pd.DataFrame, channels: list[str], *, title: str, height_per_channel: int = 170) -> go.Figure:
    """Aligned signal panels retain each channel's own scale and all source samples."""
    fig = make_subplots(
        rows=len(channels), cols=1, shared_xaxes=True, vertical_spacing=min(.04, .18 / max(len(channels), 1)),
        subplot_titles=channels,
    )
    x = frame["time_s"]
    for row, channel in enumerate(channels, start=1):
        fig.add_trace(go.Scattergl(
            x=x, y=frame[channel], mode="lines", name=channel,
            line=dict(color=SIGNAL_COLORS[(row - 1) % len(SIGNAL_COLORS)], width=1.25),
            hovertemplate=f"{channel}<br>%{{x:.2f}} s<br>%{{y:.3f}} z<extra></extra>",
        ), row=row, col=1)
        fig.update_yaxes(title_text="z-score", row=row, col=1, automargin=True)
    fig.update_xaxes(showgrid=True, gridcolor="#e3e9ec", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#e3e9ec", zerolinecolor="#cbd5dc")
    fig.update_xaxes(
        title_text="Elapsed time · seconds (10 samples/s)",
        rangeselector=dict(buttons=[
            dict(count=30, label="30 s", step="second", stepmode="backward"),
            dict(count=2, label="2 min", step="minute", stepmode="backward"),
            dict(step="all", label="All"),
        ]),
        rangeslider=dict(visible=True, thickness=.08),
        row=len(channels), col=1,
    )
    fig.update_layout(
        **(PLOT | {"margin": dict(l=18, r=20, t=48, b=22)}), title=dict(text=title, x=0, xanchor="left"),
        height=max(330, 90 + height_per_channel * len(channels)),
        showlegend=False,
        hovermode="x unified", uirevision="bosch-trace",
    )
    return fig


def telemetry_panels(frame: pd.DataFrame, channels: list[str], *, title: str) -> go.Figure:
    """Plot sensor-window means on aligned axes without implying common units."""
    fig = make_subplots(
        rows=len(channels), cols=1, shared_xaxes=True,
        vertical_spacing=min(.04, .18 / max(len(channels), 1)), subplot_titles=channels,
    )
    for row, channel in enumerate(channels, start=1):
        fig.add_trace(go.Scattergl(
            x=frame["simulation_time_s"], y=frame[channel], mode="lines",
            line=dict(color=SIGNAL_COLORS[(row - 1) % len(SIGNAL_COLORS)], width=1.5),
            hovertemplate=f"{channel} · 5 s window mean<br>%{{x:.0f}} s<br>%{{y:.3f}} source units<extra></extra>",
        ), row=row, col=1)
        fig.update_yaxes(title_text="Window mean", row=row, col=1, automargin=True)
    fig.update_xaxes(showgrid=True, gridcolor="#e3e9ec", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#e3e9ec", zerolinecolor="#cbd5dc")
    fig.update_layout(**(PLOT | {"margin": dict(l=18, r=20, t=48, b=22)}),
                      title=dict(text=title, x=0, xanchor="left"), height=max(370, 100 + 150 * len(channels)),
                      showlegend=False, hovermode="x unified")
    fig.update_xaxes(title_text="Simulation time · seconds", row=len(channels), col=1)
    return fig


def _box_mesh(x0, x1, y0, y1, z0, z1, color, opacity=1.0):
    vertices = [(x, y, z) for x, y, z in [
        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
    ]]
    faces = [(0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6), (0, 4, 5), (0, 5, 1),
             (1, 5, 6), (1, 6, 2), (2, 6, 7), (2, 7, 3), (3, 7, 4), (3, 4, 0)]
    return go.Mesh3d(
        x=[v[0] for v in vertices], y=[v[1] for v in vertices], z=[v[2] for v in vertices],
        i=[f[0] for f in faces], j=[f[1] for f in faces], k=[f[2] for f in faces],
        color=color, opacity=opacity, flatshading=True, hoverinfo="skip", showscale=False,
        lighting=dict(ambient=.65, diffuse=.75, specular=.25, roughness=.65, fresnel=.08),
    )


def _cylinder_mesh(start, end, radius, color, *, opacity=1.0, segments=14):
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    axis = end - start
    axis /= max(float(np.linalg.norm(axis)), 1e-9)
    reference = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < .9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, reference); u /= max(float(np.linalg.norm(u)), 1e-9)
    v = np.cross(axis, u)
    ring = [np.cos(2 * np.pi * q / segments) * u + np.sin(2 * np.pi * q / segments) * v for q in range(segments)]
    points = [start + radius * vec for vec in ring] + [end + radius * vec for vec in ring]
    points += [start, end]
    faces_i, faces_j, faces_k = [], [], []
    for q in range(segments):
        nxt = (q + 1) % segments
        faces_i.extend([q, q, 2 * segments, 2 * segments + 1])
        faces_j.extend([nxt, segments + q, nxt, segments + nxt])
        faces_k.extend([segments + q, segments + nxt, q, segments + q])
    return go.Mesh3d(
        x=[p[0] for p in points], y=[p[1] for p in points], z=[p[2] for p in points],
        i=faces_i, j=faces_j, k=faces_k, color=color, opacity=opacity,
        flatshading=True, hoverinfo="skip", showscale=False,
        lighting=dict(ambient=.6, diffuse=.8, specular=.3, roughness=.5),
    )


def engine_cutaway_figure(simulation_time_s: float, failed: bool) -> go.Figure:
    """Stylized animated inline-four cutaway; geometry is illustrative, not to scale."""
    if failed:
        block, head, liner, piston, rod, crank = "#b72f35", "#dc4145", "#f08080", "#8f2028", "#a72a31", "#cf343b"
        title = "INLINE-FOUR CUTAWAY · SIMULATED FAILURE"
    else:
        block, head, liner, piston, rod, crank = "#526e7d", "#8399a4", "#b4c3c8", "#d6dde0", "#bd744b", "#17627b"
        title = "INLINE-FOUR CUTAWAY · RUNNING"
    fig = go.Figure()
    # Semi-transparent block and cylinder sleeves keep the moving assembly visible.
    fig.add_trace(_box_mesh(-2.25, 2.25, -.62, .62, .36, 2.12, block, .30))
    fig.add_trace(_box_mesh(-2.28, 2.28, -.70, .70, 2.12, 2.42, head, .96))
    for idx, x in enumerate([-1.5, -.5, .5, 1.5]):
        phase = 2 * np.pi * (simulation_time_s / 12.0) + idx * np.pi / 2
        crank_z = .34 + .26 * np.cos(phase)
        piston_z = 1.42 + .30 * np.cos(phase)
        fig.add_trace(_cylinder_mesh((x, 0, .72), (x, 0, 1.98), .34, liner, opacity=.27))
        fig.add_trace(_box_mesh(x - .27, x + .27, -.26, .26, piston_z - .13, piston_z + .13, piston, .98))
        fig.add_trace(_cylinder_mesh((x, 0, piston_z - .13), (x, 0, crank_z), .075, rod, opacity=1.0))
        fig.add_trace(_cylinder_mesh((x, 0, crank_z - .07), (x, 0, crank_z + .07), .16, crank, opacity=1.0))
    fig.add_trace(_cylinder_mesh((-2.35, 0, .34), (2.35, 0, .34), .12, crank, opacity=1.0))
    fig.update_layout(
        **(PLOT | {"margin": dict(l=2, r=2, t=42, b=2)}),
        title=dict(text=title, x=.03, xanchor="left", font=dict(size=15)), height=430,
        showlegend=False,
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            aspectmode="data", camera=dict(eye=dict(x=1.55, y=1.8, z=1.1)),
            bgcolor="rgba(0,0,0,0)",
        ),
        uirevision="engine-cutaway",
    )
    return fig


def metric(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


def heading(eyebrow: str, title: str, lede: str) -> None:
    st.markdown(f'<div class="eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="lede">{lede}</div>', unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def cached_performance(engine: str) -> pd.DataFrame:
    return load_performance_data(engine)


@st.cache_data(show_spinner=False)
def cached_safety(engine: str) -> dict[str, float]:
    return load_safety_bounds(engine)


@st.cache_data(show_spinner=False)
def cached_trace() -> pd.DataFrame:
    return load_gengine_trajectory()


@st.cache_data(show_spinner=False)
def cached_liu_windows() -> pd.DataFrame:
    feature_file = Path(__file__).parent / "data" / "liu_ice" / "processed_features.csv.gz"
    return pd.read_csv(feature_file)


@st.cache_resource(show_spinner=False)
def cached_liu_models() -> tuple[dict, dict, dict]:
    windows = cached_liu_windows()
    active = fit_probability_model(windows, "fault_active")
    early = fit_probability_model(windows, "fault_within_horizon", horizon_seconds=30.0)
    isolator = fit_fault_isolator(windows, active["selected_features"])
    return active, early, isolator


@st.cache_data(show_spinner=False)
def cached_marine_rows() -> pd.DataFrame:
    return load_marine_runs()


@st.cache_resource(show_spinner=False)
def cached_marine_model() -> dict:
    return fit_marine_model(cached_marine_rows())


@st.cache_data(show_spinner=False)
def cached_cmapss_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_fd001()


@st.cache_resource(show_spinner=False)
def cached_cmapss_model() -> tuple[dict, pd.DataFrame]:
    train, test = cached_cmapss_data()
    sensor_channels = [c for c in train.select_dtypes(include=np.number).columns if c not in {"engine_id", "cycle", "rul"}]
    X, y, groups, _ = build_rul_features(train, "engine_id", "cycle", "rul")
    result = fit_rul_model(X, y, groups)
    test_X, test_ids = build_live_features(test, "engine_id", "cycle", sensor_channels)
    selected = result["selected_features"]
    predictions = result["production_model"].predict(test_X[selected])
    actual = test.groupby("engine_id").tail(1).set_index("engine_id")["rul"].reindex(test_ids)
    estimates = pd.DataFrame({"Engine": test_ids, "Observed RUL · cycles": actual.to_numpy(),
                              "Predicted RUL · cycles": predictions})
    estimates["Absolute error · cycles"] = np.abs(estimates["Observed RUL · cycles"] - estimates["Predicted RUL · cycles"])
    return result, estimates


@st.cache_data(show_spinner=False)
def cached_acoustic_window(fault: str, severity: float, rpm: float, load: float, seed: int):
    return simulate_acoustic_window(fault=fault, severity=severity, rpm=rpm, load=load, seed=seed, duration_s=2.0)


with st.sidebar:
    st.markdown("<div class='eyebrow'>Engine systems · R&D</div>", unsafe_allow_html=True)
    st.markdown("## ENGINE TWIN<br>COMBUSTION LAB", unsafe_allow_html=True)
    page = st.selectbox(
        "Workbench",
        ["Overview", "Simulation", "Acoustic twin", "Sweet spot", "Time series", "Signatures", "Live OBD-II", "Failure probability", "65-channel diesel", "RUL prognosis"],
        label_visibility="collapsed",
    )
    source_by_page = {
        "Overview": "Bosch Research · gasoline-engine maps + trace",
        "Simulation": "LiU-ICE · measured no-fault baseline + injected what-if fault",
        "Acoustic twin": "EngineTwin · simulated sound, vibration + OBD signatures",
        "Sweet spot": "Bosch Research · measured engine maps",
        "Time series": "Bosch Research · 13-channel trace",
        "Signatures": "Uploaded audio, vibration, or current traces",
        "Live OBD-II": "Read-only standard PID capture · adapter required",
        "Failure probability": "LiU-ICE · 10 automotive sensor/actuator streams",
        "65-channel diesel": "Zenodo marine diesel · 65 usable channels",
        "RUL prognosis": "User-supplied run-to-failure histories",
    }
    st.markdown("<div class='rule'></div>", unsafe_allow_html=True)
    st.markdown("**Data source**  ")
    st.caption(source_by_page[page])
    st.markdown("<div class='rule'></div>", unsafe_allow_html=True)
    st.caption("EngineTwin × Combustion Lab · research workbench")


if page == "Overview":
    heading("Combustion systems · measured data", "Engine overview.",
            "A compact view of measured operating points, synchronized telemetry, and independent fault benchmarks.")
    engine = st.selectbox("Bosch performance map", ["engine1", "engine2"], format_func=lambda x: f"{x} · measured map")
    frame, trace = cached_performance(engine), cached_trace()
    spot = frame.copy()
    safety = cached_safety(engine)
    guards = [c for c in ["engine_roughness_s", "temperature_exhaust_manifold", "temperature_in_catalyst", "HC", "NOx"] if c in spot]
    for guard in guards:
        if guard in safety and np.isfinite(safety[guard]):
            spot = spot.loc[spot[guard] <= safety[guard]]
    best = spot.loc[spot["specific_fuel_consumption"].idxmin()] if len(spot) else None
    fit = select_features(frame, PERFORMANCE_CONTROLS, "specific_fuel_consumption")[2]
    snapshot, map_tab, sources_tab = st.tabs(["Snapshot", "Performance map", "Data sources"])
    with snapshot:
        cards = st.columns(4)
        with cards[0]: metric("Measured map points", f"{len(frame):,}", f"Bosch {engine}")
        with cards[1]: metric("Telemetry channels", "13", "Bosch · measured · 10 Hz")
        with cards[2]: metric("Automotive signals", "10", "LiU-ICE benchmark")
        with cards[3]: metric("Diesel channels", "65", "independent benchmark")
        left, right = st.columns([.8, 1.2], gap="medium")
        with left:
            st.subheader("Best measured operating point")
            if best is not None:
                metric("Fuel-use score", f"{best['specific_fuel_consumption']:.2f} z", f"{len(spot):,} rows under available caps")
                chosen_cols = PERFORMANCE_CONTROLS + ["specific_fuel_consumption"]
                st.dataframe(pd.DataFrame([best[chosen_cols]]).style.format("{:.2f}"), hide_index=True, width="stretch")
            else:
                st.info("No measured point meets the available safety caps. Adjust them on Sweet spot.")
            st.caption("Workbook values are normalized z-scores, not physical setpoints.")
        with right:
            trace_channels = [c for c in trace.columns if c != "time_s"]
            defaults = [c for c in ["speed", "load", "exhaust_manifold_temperature"] if c in trace_channels]
            selected = st.multiselect("Bosch trace channels", trace_channels, default=defaults, key="overview_trace_channels")
            if selected:
                fig = signal_panels(trace, selected, title="Measured Bosch trace")
                fig.update_layout(height=max(270, 80 + 105 * len(selected)))
                st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
                st.caption(f"{len(trace):,} samples · 10 Hz · source-standardized values")
    with map_tab:
        left, right = st.columns([1.35, .65], gap="medium")
        with left:
            plot = px.scatter(frame, x="engine_speed", y="engine_load", color="specific_fuel_consumption",
                              color_continuous_scale=["#8ec5ca", "#d77d37"], opacity=.8,
                              labels={"engine_speed": "Engine speed · z", "engine_load": "Engine load · z", "specific_fuel_consumption": "Fuel-use score · z"})
            if best is not None:
                plot.add_trace(go.Scatter(x=[best["engine_speed"]], y=[best["engine_load"]], mode="markers",
                                          name="Best feasible", marker=dict(symbol="star", size=15, color="#142b3b", line=dict(color="#fff", width=2))))
            plot.update_layout(**PLOT, height=390, coloraxis_colorbar=dict(title="Fuel use · z"))
            st.plotly_chart(plot, width="stretch", config={"displayModeBar": False})
        with right:
            importance = fit["importance"].rename_axis("Control").reset_index(name="Importance").sort_values("Importance")
            bars = px.bar(importance, x="Importance", y="Control", orientation="h", color_discrete_sequence=["#17627b"])
            bars.update_layout(**PLOT, height=320, showlegend=False)
            st.plotly_chart(bars, width="stretch", config={"displayModeBar": False})
            st.caption(f"Exploratory holdout: MAE {fit['mae']:.2f} z · R² {fit['r2']:.2f}")
    with sources_tab:
        source_rows = [
            ("Bosch engine maps", "Gasoline bench", f"{len(frame):,} measured points", "Performance / Sweet spot"),
            ("Bosch trajectory", "Gasoline bench", "13 channels · 10 Hz", "Time series"),
            ("LiU-ICE", "Automotive injected faults", "10 streams · 41 trajectories", "Fault probability / replay"),
            ("Marine Engine Fault", "Marine diesel bench", "65 selected channels", "High-channel monitoring"),
            ("EngineTwin simulator", "Simulated acoustic patterns", "Sound · vibration · OBD", "Acoustic twin"),
            ("NASA C-MAPSS FD001", "Simulated aircraft turbofan", "Optional RUL benchmark", "RUL prognosis"),
        ]
        st.dataframe(pd.DataFrame(source_rows, columns=["Source", "Engine / evidence", "Coverage", "Workbench"]), hide_index=True, width="stretch")
        st.caption("Sources stay separate: engine populations and labels are not fused into one training set. Real and simulated evidence is identified in each workflow.")


elif page == "Acoustic twin":
    heading("EngineTwin · acoustic sandbox", "Explore a fault signature.",
            "Generate synchronized inline-four sound, vibration, and OBD signals, then inspect the frequencies linked to each simulated fault.")
    st.caption("All signals on this page are simulated textbook-style patterns calibrated around a Kia Soul operating point. They are not measured fault recordings or a trained diagnosis.")
    control_cols = st.columns([1.25, 1, 1, 1, .8], gap="small")
    with control_cols[0]:
        twin_fault = st.selectbox("Fault pattern", TWIN_FAULTS, format_func=lambda x: x.replace("_", " ").replace("+", " + "), key="twin_fault")
    with control_cols[1]:
        twin_severity = st.slider("Severity", 0.0, 1.0, 0.7, 0.05, disabled=twin_fault == "healthy", key="twin_severity")
    with control_cols[2]:
        twin_rpm = st.slider("Engine speed · rpm", 800, 5000, 2000, 100, key="twin_rpm")
    with control_cols[3]:
        twin_load = st.slider("Load · %", 10, 95, 40, 5, key="twin_load")
    with control_cols[4]:
        twin_seed = st.number_input("Seed", min_value=0, max_value=9999, value=7, key="twin_seed")

    twin_window = cached_acoustic_window(twin_fault, 0.0 if twin_fault == "healthy" else float(twin_severity),
                                         float(twin_rpm), float(twin_load) / 100, int(twin_seed))
    half_ratio = float(half_order_ratio(twin_window.sound, twin_window.sound_fs, twin_rpm))
    knock_fraction = float(band_energy_fraction(twin_window.sound, twin_window.sound_fs, 6000, 7000))
    summary_cols = st.columns(4)
    with summary_cols[0]: metric("Firing rate", f"{twin_rpm / 30:.1f} Hz", "four-cylinder firing events")
    with summary_cols[1]: metric("Half-order ratio", f"{half_ratio:.3f}", "misfire cue · simulated")
    with summary_cols[2]: metric("6–7 kHz energy", f"{knock_fraction:.1%}", "knock band · simulated")
    with summary_cols[3]: metric("Signal window", f"{twin_window.duration_s:.0f} s", "16 kHz sound + vibration")

    signature_tab, component_tab, telemetry_tab = st.tabs(["Signal shape", "Valve and cylinder cues", "OBD values"])
    with signature_tab:
        wave_col, order_col = st.columns(2, gap="medium")
        sample_count = min(len(twin_window.sound), int(.14 * twin_window.sound_fs))
        sample_time = np.arange(sample_count) / twin_window.sound_fs
        wave = go.Figure()
        wave.add_trace(go.Scattergl(x=sample_time, y=twin_window.sound[:sample_count], name="Microphone", line=dict(color="#17627b", width=1.2)))
        wave.add_trace(go.Scattergl(x=sample_time, y=twin_window.vibration[:sample_count], name="Accelerometer", line=dict(color="#b95118", width=1.2)))
        wave.update_layout(**(PLOT | {"height": 320}), title="Synchronized sensor windows", xaxis_title="Time · s", yaxis_title="Relative amplitude", legend=dict(orientation="h", y=1.12))
        wave_col.plotly_chart(wave, width="stretch", config={"displayModeBar": False})

        orders, amplitudes = order_spectrum(twin_window.sound, twin_window.sound_fs, twin_rpm, max_order=8.0)
        order_fig = go.Figure(go.Scattergl(x=orders, y=amplitudes, mode="lines", line=dict(color="#007d87", width=1.5), name="Order amplitude"))
        order_fig.add_vline(x=.5, line_dash="dot", line_color="#b95118")
        order_fig.add_vline(x=2, line_dash="dot", line_color="#17627b")
        order_fig.update_layout(**(PLOT | {"height": 320}), title="Microphone order spectrum", xaxis_title="Order · crank frequency", yaxis_title="Amplitude", showlegend=False)
        order_col.plotly_chart(order_fig, width="stretch", config={"displayModeBar": False})
        order_col.caption("0.5 order is a misfire cue; order 2 is the four-cylinder firing frequency.")
    with component_tab:
        valve_col, cylinder_col = st.columns(2, gap="medium")
        env_freq, env_amp = envelope_spectrum(twin_window.vibration, twin_window.vibration_fs, band=(2500, 3500))
        env_keep = env_freq <= 300
        env_fig = go.Figure(go.Scattergl(x=env_freq[env_keep], y=env_amp[env_keep], mode="lines", line=dict(color="#b95118", width=1.5)))
        env_fig.update_layout(**(PLOT | {"height": 320}), title="Valve-band envelope spectrum", xaxis_title="Repetition frequency · Hz", yaxis_title="Envelope amplitude", showlegend=False)
        valve_col.plotly_chart(env_fig, width="stretch", config={"displayModeBar": False})
        cylinder_energy = per_cylinder_energy(twin_window.vibration, twin_window.vibration_fs, twin_rpm)
        cylinder_fig = px.bar(x=[f"Cylinder {n}" for n in cylinder_energy], y=list(cylinder_energy.values()),
                              color=[f"Cylinder {n}" for n in cylinder_energy], color_discrete_sequence=["#17627b", "#007d87", "#b95118", "#64748b"])
        cylinder_fig.update_layout(**(PLOT | {"height": 320}), title="Energy grouped by firing cylinder", xaxis_title=None, yaxis_title="Signal energy", showlegend=False)
        cylinder_col.plotly_chart(cylinder_fig, width="stretch", config={"displayModeBar": False})
    with telemetry_tab:
        obd = pd.DataFrame([{"Signal": key.replace("_", " ").title(), "Simulated reading": value}
                            for key, value in twin_window.obd.items() if not key.startswith("_")])
        left, right = st.columns([1, 1.3], gap="medium")
        left.dataframe(obd, hide_index=True, width="stretch")
        with right:
            st.markdown("**How to read this view**")
            st.write("Misfire reduces one cylinder’s firing energy and can add a half-order line. Knock adds short bursts in the 6–7 kHz band. Valve wear adds a high-frequency tick whose envelope repeats with the valve cycle.")
            st.caption("These effects and OBD shifts are simulator assumptions. Scores here are signal measurements, not model probabilities.")


elif page == "Simulation":
    heading("Digital twin preview / what-if replay", "Run an engine-fault simulation.",
            "Replay a held-out real drive, inject a data-derived fault signature, and watch the benchmark warning scores react across time.")
    st.markdown("<div class='signal-card'><b>Simulation scope:</b> a three-minute measured-telemetry replay plays in one minute at 3× speed. Sensor baselines come from a held-out LiU-ICE no-fault run; fault changes are learned from training runs only. The animated inline-four is a conceptual cutaway. The red state marks the configured scenario event; benchmark scores are model outputs and do not predict catastrophic engine failure.</div>", unsafe_allow_html=True)
    st.caption("The default demonstration preset uses a held-out intake-manifold-leak case where the onset model crosses its validation threshold before full fault strength. This illustrates benchmark behavior; it is not an unbiased estimate of real-world engine-failure risk.")
    feature_path = Path(__file__).parent / "data" / "liu_ice" / "processed_features.csv.gz"
    if not feature_path.exists():
        st.warning("The LiU-ICE feature file is not prepared yet. Run `python prepare_liu_ice.py` from the project folder, then refresh this page.")
    else:
        with st.spinner("Preparing held-out telemetry and fault models for simulation…"):
            windows = cached_liu_windows()
            active_model, early_model, _ = cached_liu_models()
        heldout = active_model["test_windows"]
        baseline_ids = sorted(heldout.loc[heldout["scenario"] == "No fault", "trajectory_id"].unique())
        if not baseline_ids:
            st.error("No held-out no-fault trajectory is available for a leak-free simulation baseline.")
        else:
            left, right = st.columns(2, gap="large")
            with left:
                baseline_id = st.selectbox(
                    "Measured baseline · held-out no-fault run", baseline_ids,
                    format_func=lambda run: f"{run} · {heldout.loc[heldout['trajectory_id'] == run, 'drive_cycle'].iloc[0].upper()} drive",
                )
                baseline_rows = heldout.loc[heldout["trajectory_id"] == baseline_id].sort_values("window_end_s")
                window_seconds = float(active_model["window_seconds"])
                replay_windows = 36
                max_start = max(0, len(baseline_rows) - replay_windows)
                if max_start:
                    start_window = st.slider("Baseline start · source window", 0, max_start, min(72, max_start), 1)
                else:
                    start_window = 0
                    st.caption(f"The selected source run provides {len(baseline_rows)} windows; the replay uses that available segment.")
            with right:
                scenario_codes = [None, *FAULT_LABELS.keys()]
                fault_code = st.selectbox(
                    "Fault scenario · injected after start", scenario_codes,
                    index=scenario_codes.index("f_iml"),
                    format_func=lambda code: "No injected fault · control replay" if code is None else FAULT_LABELS[code],
                )
                severity_percent = st.slider("Fault signature strength · % of observed training shift", 0, 150, 150, 10,
                                              disabled=fault_code is None,
                                              help="100% applies the median feature shift measured between training fault windows and normal windows.")
                duration_seconds = min(replay_windows, len(baseline_rows)) * window_seconds
                onset_max = max(15, int(duration_seconds - 60))
                onset_seconds = st.slider("Fault reaches full strength · simulation seconds", 15, onset_max, min(90, onset_max), 5,
                                           disabled=fault_code is None)
                failure_min = min(int(duration_seconds - 5), int(onset_seconds + 15))
                failure_seconds = st.slider("Scripted failure event · simulation seconds", failure_min,
                                             max(failure_min, int(duration_seconds - 5)),
                                             min(max(failure_min, int(onset_seconds + 60)), int(duration_seconds - 5)), 5,
                                             disabled=fault_code is None,
                                             help="A visible scenario event for the use-case demonstration; it is not triggered by the model prediction.")
            selected_channels = st.multiselect(
                "Sensor channels · five-second window means", LIU_SIGNALS,
                default=[c for c in ["engine_speed", "air_mass_flow", "intake_manifold_pressure", "wastegate_position"] if c in LIU_SIGNALS],
                key="simulation_channels",
            )
            settings = (baseline_id, fault_code, severity_percent, onset_seconds, failure_seconds, start_window)
            run_clicked = st.button("Run simulation · 3× speed", type="primary", use_container_width=False)
            if run_clicked:
                try:
                    with st.spinner("Replaying measured windows and scoring the injected scenario…"):
                        simulated = simulate_fault_scenario(
                            baseline_rows, active_model["training_windows"], active_model, early_model,
                            fault_code=fault_code, severity=severity_percent / 100.0,
                            onset_seconds=float(onset_seconds), start_window=start_window,
                            n_windows=min(replay_windows, len(baseline_rows)),
                        )
                    st.session_state["engine_simulation"] = {"settings": settings, "frame": simulated}
                    st.session_state["engine_simulation_started_at"] = time.monotonic()
                    st.session_state["engine_simulation_elapsed"] = 0.0
                    st.session_state["engine_simulation_running"] = True
                except (ValueError, KeyError) as exc:
                    st.error(f"Could not run this simulation: {exc}")
            payload = st.session_state.get("engine_simulation")
            if payload and payload["settings"] == settings:
                simulated = payload["frame"]
                duration_seconds = len(simulated) * window_seconds

                @st.fragment(run_every=1.0 if st.session_state.get("engine_simulation_running") else None)
                def live_simulation_panel():
                    running = st.session_state.get("engine_simulation_running", False)
                    started = st.session_state.get("engine_simulation_started_at", time.monotonic())
                    elapsed = time.monotonic() - started if running else st.session_state.get("engine_simulation_elapsed", duration_seconds / 3)
                    sim_time = min(duration_seconds, elapsed * 3.0)
                    cursor = min(len(simulated) - 1, int(sim_time / window_seconds))
                    current = simulated.iloc[cursor]
                    failed = fault_code is not None and sim_time >= failure_seconds
                    signature_start = float(simulated.attrs.get("warning_signature_start_s", onset_seconds))
                    visible = simulated.iloc[:cursor + 1]

                    st.markdown("<div class='rule'></div>", unsafe_allow_html=True)
                    st.subheader("Live run · 180 simulated seconds in 60 seconds")
                    st.progress(min(1.0, sim_time / duration_seconds), text=f"{sim_time:05.1f} / {duration_seconds:.0f} simulated seconds · 3× playback")
                    engine_col, status_col = st.columns([1.15, .85], gap="large")
                    with engine_col:
                        st.plotly_chart(engine_cutaway_figure(sim_time, failed), width="stretch", config={"displayModeBar": False})
                        st.caption("Conceptual inline-four cutaway · moving parts are illustrative, not a thermodynamic model.")
                    with status_col:
                        state = ("FAILURE EVENT ACTIVE" if failed else
                                 "FAULT AT FULL STRENGTH" if fault_code is not None and sim_time >= onset_seconds else
                                 "PRECURSOR SIGNATURE RAMP" if fault_code is not None and sim_time >= signature_start else
                                 "NORMAL REPLAY")
                        st.markdown(f"### {'🔴' if failed else '🟠' if fault_code is not None and sim_time >= signature_start else '🟢'} {state}")
                        if fault_code is not None:
                            st.caption(f"Scenario: {FAULT_LABELS[fault_code]} · {severity_percent}% training shift")
                            st.caption(f"Precursor ramp: {signature_start:.0f}s · full fault: {onset_seconds}s · scripted failure: {failure_seconds}s")
                        score_cols = st.columns(2)
                        with score_cols[0]: metric("P(fault active)", f"{current['fault_probability']:.1%}", f"benchmark score · threshold {active_model['threshold']:.2f}")
                        with score_cols[1]: metric("P(onset ≤30 s)", f"{current['onset_probability']:.1%}", f"benchmark score · threshold {early_model['threshold']:.2f}")
                        active_alert = current["fault_probability"] >= active_model["threshold"]
                        onset_alert = current["onset_probability"] >= early_model["threshold"]
                        st.markdown("**Prediction status**")
                        st.write(f"{'✓' if active_alert else '○'} Active-fault threshold {'crossed' if active_alert else 'not crossed'}")
                        st.write(f"{'✓' if onset_alert else '○'} Early-warning threshold {'crossed' if onset_alert else 'not crossed'}")
                        crossings = np.flatnonzero(simulated["onset_probability"].to_numpy() >= early_model["threshold"])
                        if len(crossings):
                            first_warning_s = float(simulated.iloc[int(crossings[0])]["simulation_time_s"])
                            if first_warning_s <= sim_time:
                                st.info(f"First onset warning at {first_warning_s:.0f}s · {max(0, onset_seconds - first_warning_s):.0f}s before full fault strength and {max(0, failure_seconds - first_warning_s):.0f}s before the scripted event. This model warns on the benchmark fault label, not engine failure.")
                        st.caption("These are fault benchmark classifications. The scripted red event is independent of model output.")
                    if running and sim_time >= duration_seconds:
                        st.session_state["engine_simulation_running"] = False
                        st.session_state["engine_simulation_elapsed"] = duration_seconds / 3.0

                    probability = go.Figure()
                    probability.add_trace(go.Scatter(x=visible["simulation_time_s"], y=visible["onset_probability"], name="P(onset within 30 s)", line=dict(color="#b95118", width=2.4), fill="tozeroy", fillcolor="rgba(185,81,24,.10)", hovertemplate="%{x:.0f} s<br>Onset score %{y:.1%}<extra></extra>"))
                    probability.add_trace(go.Scatter(x=visible["simulation_time_s"], y=visible["fault_probability"], name="P(fault active)", line=dict(color="#17627b", width=2.2), hovertemplate="%{x:.0f} s<br>Active-fault score %{y:.1%}<extra></extra>"))
                    probability.add_trace(go.Scatter(x=[0, duration_seconds], y=[active_model["threshold"]] * 2,
                        mode="lines", name=f"Active-fault limit · {active_model['threshold']:.2f}",
                        line=dict(color="#17627b", width=1.4, dash="dash"), hoverinfo="skip"))
                    probability.add_trace(go.Scatter(x=[0, duration_seconds], y=[early_model["threshold"]] * 2,
                        mode="lines", name=f"30 s onset limit · {early_model['threshold']:.2f}",
                        line=dict(color="#b95118", width=1.4, dash="dot"), hoverinfo="skip"))
                    if fault_code is not None:
                        probability.add_vline(x=signature_start, line_dash="dot", line_color="#667780", annotation_text="precursor ramp", annotation_position="top left")
                        probability.add_vline(x=onset_seconds, line_dash="dot", line_color="#142b3b", annotation_text="full fault", annotation_position="top right")
                        probability.add_vline(x=failure_seconds, line_dash="dash", line_color="#b72f35", annotation_text="failure event", annotation_position="bottom right")
                    probability.add_vline(x=sim_time, line_color="#435866", line_width=1.5)
                    probability.update_layout(**(PLOT | {"height": 430, "margin": dict(l=64, r=28, t=68, b=92), "yaxis": dict(range=[0, 1], gridcolor="#e3e9ec")}), title=dict(text="Live warning scores · model output vs scenario events", x=.02, xanchor="left"), xaxis_title="Simulation time · seconds", yaxis_title="Benchmark score", legend=dict(orientation="h", y=-.22, yanchor="top", x=0, xanchor="left"), hovermode="x unified", uirevision=f"sim-{baseline_id}-{fault_code}")
                    probability.update_xaxes(range=[0, duration_seconds], rangeslider_visible=False)
                    st.plotly_chart(probability, width="stretch", config={"displayModeBar": False})
                    if selected_channels:
                        telemetry = telemetry_panels(visible, selected_channels, title="Live sensor response · measured baseline plus synthetic shift")
                        telemetry.update_xaxes(range=[0, duration_seconds])
                        st.plotly_chart(telemetry, width="stretch", config={"displayModeBar": False})
                    else:
                        st.info("Choose sensor channels to plot the five-second telemetry window means.")
                    st.caption("Fault shifts use training-run medians matched by drive cycle where possible. The synthetic precursor ramps over the benchmark's 30-second warning horizon. Model scores are not independent empirical failure probabilities.")

                live_simulation_panel()
            elif not run_clicked:
                st.info("Choose a measured baseline and scenario, then run the three-minute replay. Playback lasts one minute at 3× speed.")


elif page == "Sweet spot":
    heading("Performance calibration", "Find a measured sweet spot.",
            "Rank recorded operating points, see which controls explain the response, and keep only measurements that meet your chosen guardrails.")
    engine = st.selectbox("Map source", ["engine1", "engine2"], key="map_engine", format_func=lambda x: f"Bosch {x} · experimental data")
    frame = cached_performance(engine)
    safety = cached_safety(engine)
    target = st.selectbox("Optimize response", PERFORMANCE_TARGETS, index=0,
                          format_func=lambda x: {"specific_fuel_consumption":"Specific fuel consumption · minimize", "engine_roughness_s":"Cylinder pressure roughness · minimize", "temperature_exhaust_manifold":"Exhaust manifold temperature · minimize", "temperature_in_catalyst":"Catalyst temperature · minimize", "HC":"Hydrocarbons · minimize", "NOx":"Nitrogen oxides · minimize"}[x])
    st.markdown("<div class='signal-card'>Values and constraints use the source dataset's standardized z-score scale. The best points below are measured responses at observed settings; the ranking does not optimize predictions or extrapolate between measurements.</div>", unsafe_allow_html=True)
    st.subheader("Guardrails · Bosch safety bounds")
    guarded = [c for c in ["engine_roughness_s", "temperature_exhaust_manifold", "temperature_in_catalyst", "HC", "NOx"] if c != target]
    limit_cols = st.columns(min(5, len(guarded)))
    limits = {}
    for i, col in enumerate(guarded):
        p85 = float(frame[col].quantile(.85))
        safety_max = safety.get(col, float(frame[col].quantile(.99)))
        if not np.isfinite(safety_max):
            safety_max = float(frame[col].quantile(.99))
        safety_max = max(safety_max, float(frame[col].quantile(.05)))
        default = min(p85, safety_max)
        limits[col] = limit_cols[i].slider(col.replace("_", " "),
            min_value=float(frame[col].quantile(.05)), max_value=float(safety_max),
            value=default, step=.05, help="Maximum allowed standardized response for this exploration.")
    with st.spinner("Ranking observed settings and fitting response models…"):
        candidates, fit = propose_operating_points(frame, PERFORMANCE_CONTROLS, target, limits)
    st.markdown("<div class='rule'></div>", unsafe_allow_html=True)
    cols = st.columns(3)
    with cols[0]: metric("Measured points", f"{len(frame):,}", "source settings available to search")
    with cols[1]: metric("Candidate settings", f"{len(candidates):,}", "observed controls only")
    with cols[2]: metric("Holdout prediction error", f"{fit['mae']:.3f} z", f"R² {fit['r2']:.2f} · random holdout")
    left, right = st.columns([1.3, .7], gap="large")
    with left:
        st.subheader("Operating envelope / candidate rank")
        display = candidates.copy()
        display["status"] = np.where(display["feasible"], "Within guardrails", "Best available; guardrail exceeded")
        fig = px.scatter(display, x="engine_speed", y="engine_load", color="objective", symbol="feasible",
            color_continuous_scale=["#8ec5ca", "#d77d37"], hover_data=PERFORMANCE_CONTROLS + ["objective", "status"],
            labels={"engine_speed":"Engine speed · z", "engine_load":"Engine load · z", "objective":f"{target} · measured z"})
        feasible = display.loc[display["feasible"]]
        if len(feasible):
            candidate = feasible.loc[feasible["objective"].idxmin()]
            fig.add_trace(go.Scatter(
                x=[candidate["engine_speed"]], y=[candidate["engine_load"]], mode="markers+text",
                text=["Best feasible"], textposition="top center", name="Best feasible measured point",
                marker=dict(symbol="star", size=17, color="#142b3b", line=dict(color="#ffffff", width=2)),
                customdata=[[candidate["objective"]]],
                hovertemplate="Best feasible measured point<br>Speed %{x:.3f} z<br>Load %{y:.3f} z<br>Response %{customdata[0]:.3f} z<extra></extra>",
            ))
        fig.update_layout(**PLOT, height=460, coloraxis_colorbar=dict(title="Response · z-score"),
                           legend=dict(orientation="h", y=1.08), hovermode="closest")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.caption("The star marks the lowest measured response among candidates that meet every selected guardrail. Background points retain the full ranked candidate set.")
    with right:
        st.subheader("Controls selected by permutation")
        imp = fit["importance"].rename_axis("Control").reset_index(name="Score")
        fig = px.bar(imp.sort_values("Score"), x="Score", y="Control", orientation="h", color_discrete_sequence=["#285a70"])
        fig.update_layout(**PLOT, height=300)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.caption("Selected predictors: " + ", ".join(fit["selected_controls"]) + ". Permutation importance is computed on held-out rows; negative scores mean shuffling did not hurt this split's MAE.")
    st.subheader("Top measured settings")
    show = [*PERFORMANCE_CONTROLS, target, *[c for c in guarded if c in candidates], "feasible"]
    show = list(dict.fromkeys(show))
    st.dataframe(candidates[show].head(12).style.format({c: "{:.3f}" for c in show if c != "feasible"}), width="stretch", hide_index=True)
    st.caption("The optimum is conditional on the measured dataset, the selected single-response objective, and the stated standardized caps. Multi-objective tradeoffs and a physical engine calibration still require domain validation.")


elif page == "Time series":
    heading("Condition monitoring", "Spot drift in the signature.",
            "Compare multichannel traces, see which channels move together, and flag observations that sit far from the measured operating envelope.")
    trace = cached_trace()
    sensors = [c for c in trace.columns if c != "time_s"]
    selected = st.multiselect("Channels in view · each gets its own aligned panel", sensors, default=sensors[:4])
    if not selected:
        st.info("Select at least one channel to display its standardized trace.")
    else:
        fig = signal_panels(trace, selected, title="Aligned source signals · full 10 Hz trace", height_per_channel=185)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.caption(f"{len(trace):,} source samples shown for {len(selected)} channels. Panels share the time axis while retaining separate z-score scales, so a high-amplitude channel cannot flatten the others.")
        scores = condition_deviation(trace, selected)
        trace_view = trace.loc[scores.index].copy()
        trace_view["Relative deviation"] = scores.values
        threshold = float(scores.quantile(.95))
        st.markdown(f"**Deviation score:** Isolation Forest score normalized to the trace's 95th percentile. **Exploratory threshold:** {threshold:.2f} (not a failure probability).")
        fig = go.Figure(go.Scattergl(
            x=trace_view["time_s"], y=trace_view["Relative deviation"], mode="lines",
            name="Relative deviation", line=dict(color="#17627b", width=1.4),
            hovertemplate="%{x:.1f} s<br>Deviation %{y:.3f}<extra></extra>",
        ))
        fig.add_hline(y=threshold, line_dash="dot", line_color="#d77d37", annotation_text="trace P95")
        fig.update_layout(**PLOT, height=270, showlegend=False, title="Full-resolution deviation timeline",
                          xaxis_title="Elapsed time · seconds", yaxis_title="Relative deviation", hovermode="x")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        corr = trace[selected].corr(method="spearman")
        fig = px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale=["#d77d37", "#fffefa", "#285a70"], aspect="auto")
        fig.update_layout(**PLOT, height=350, title="Spearman correlation · selected channels")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.caption("This sample is one normalized 10 Hz measurement. Its 5 Hz Nyquist limit is too low for combustion knock or high-frequency vibration diagnosis.")


elif page == "Signatures":
    heading("DSP workbench", "Extract the signature.",
            "Inspect microphone audio, accelerometer vibration, electrical current, or uploaded ECU telemetry in time and frequency domains. Extract transparent window-level features before model fitting.")
    st.markdown("<div class='signal-card'><b>Slide deliverables:</b> microphone signatures for knock, misfire, and valve-train wear; accelerometer oscillations across the combustion cycle; and electrical/current signatures. Upload synchronized channels as CSV or acoustic recordings as WAV. CSV can also carry ECU/OBD-II/CAN telemetry such as RPM, coolant temperature, throttle, and fuel trims. Live vehicle capture requires an adapter; none is connected in this workspace.</div>", unsafe_allow_html=True)
    inspect_tab, training_tab = st.tabs(["Inspect recording", "Train from labels"])
    with inspect_tab:
        file = st.file_uploader("Microphone / accelerometer / current recording · CSV or WAV", type=["csv", "wav"], help="CSV: one row per sample and one column per sensor channel. WAV: mono or multichannel PCM audio; its embedded sample rate is used automatically.")
        if file is not None:
            try:
                detected_sample_rate = None
                if file.name.lower().endswith(".wav"):
                    detected_sample_rate, audio = wavfile.read(file)
                    audio = np.asarray(audio)
                    if audio.ndim == 1:
                        audio = audio[:, None]
                    raw = pd.DataFrame(audio, columns=[f"microphone_{i + 1}" for i in range(audio.shape[1])])
                else:
                    raw = pd.read_csv(file)
                numeric = list(raw.select_dtypes(include=np.number).columns)
                channel_options = [c for c in numeric if c.lower() not in {"time", "time_s", "timestamp", "cycle", "engine_id"}]
                if not channel_options:
                        st.error("No numeric sensor channels found in this file.")
                else:
                    control_a, control_b = st.columns(2)
                    with control_a:
                        sample_rate = st.number_input("Sample rate · Hz", min_value=1.0, max_value=2_000_000.0,
                            value=float(detected_sample_rate or 25_600.0), step=100.0, disabled=detected_sample_rate is not None,
                            help="Read from the WAV header for audio. Set manually for CSV recordings.")
                    with control_b:
                        window_seconds = st.number_input("Analysis window · seconds", min_value=.01, max_value=60.0, value=1.0, step=.25)
                    chosen = st.multiselect("Channels", channel_options, default=channel_options[:min(4, len(channel_options))])
                    if chosen:
                        stats = []
                        first_spec = None
                        first_wave = None
                        first_psd = None
                        window_samples = max(32, int(sample_rate * window_seconds))
                        truncated = False
                        for col in chosen:
                            s = pd.to_numeric(raw[col], errors="coerce").dropna().to_numpy(dtype=float)
                            if len(s) < 32:
                                continue
                            analysis_samples = min(window_samples, len(s))
                            step_samples = max(1, int(analysis_samples * .5))
                            window_count = 1 + (len(s) - analysis_samples) // step_samples
                            truncated |= window_count > 2000
                            starts = list(range(0, len(s) - analysis_samples + 1, step_samples))[:2000]
                            for window_id, start in enumerate(starts):
                                signal = s[start:start + analysis_samples]
                                signal = signal - np.mean(signal)
                                rms = float(np.sqrt(np.mean(signal ** 2)))
                                f, pxx = welch(signal, fs=sample_rate, nperseg=min(2048, len(signal)))
                                dom = float(f[np.argmax(pxx[1:]) + 1]) if len(f) > 1 else 0.0
                                centroid = float(np.sum(f * pxx) / max(np.sum(pxx), 1e-12))
                                stats.append({"Channel": col, "Window": window_id + 1,
                                              "Start · s": start / sample_rate, "End · s": (start + analysis_samples) / sample_rate,
                                              "RMS": rms, "Peak frequency · Hz": dom,
                                              "Spectral centroid · Hz": centroid, "Skewness": float(skew(signal)),
                                              "Kurtosis · excess": float(kurtosis(signal)), "Crest factor": float(np.max(np.abs(signal)) / max(rms, 1e-12))})
                            if first_spec is None:
                                signal = s[:min(len(s), max(analysis_samples, int(3 * sample_rate)))]
                                wave_count = min(len(signal), max(32, int(min(2.0, window_seconds) * sample_rate)))
                                first_wave = (col, np.arange(wave_count, dtype=float) / sample_rate, signal[:wave_count])
                                pf, pxx = welch(signal - np.mean(signal), fs=sample_rate, nperseg=min(4096, len(signal)))
                                first_psd = (col, pf, 10 * np.log10(pxx + 1e-15))
                                segment = min(256, len(signal))
                                sf, stime, Sxx = spectrogram(signal, fs=sample_rate, nperseg=segment,
                                                             noverlap=min(3 * segment // 4, segment - 1))
                                first_spec = (col, sf, stime, 10 * np.log10(Sxx + 1e-15))
                        if stats:
                            table = pd.DataFrame(stats)
                            st.subheader("Window signatures · 50% overlap")
                            st.dataframe(table.style.format({c: "{:.4g}" for c in table.select_dtypes("number").columns}), width="stretch", hide_index=True)
                            time_col, freq_col = st.columns(2, gap="large")
                            if first_wave is not None:
                                name, wave_time, wave_value = first_wave
                                wave_fig = go.Figure(go.Scatter(x=wave_time, y=wave_value, mode="lines", line=dict(color="#007d87", width=1.4), name=name))
                                wave_fig.update_layout(**(PLOT | {"height": 300}), title=f"Time-domain waveform · {name}", xaxis_title="Time · s", yaxis_title="Amplitude / source units")
                                time_col.plotly_chart(wave_fig, width="stretch", config={"displayModeBar": False})
                            if first_psd is not None:
                                name, freq, power = first_psd
                                positive = freq > 0
                                psd_fig = go.Figure(go.Scattergl(x=freq[positive], y=power[positive], mode="lines", line=dict(color="#b95118", width=1.8), name="Welch PSD"))
                                psd_fig.update_layout(**(PLOT | {"height": 320}), title=f"Welch power spectrum · {name}", xaxis_title="Frequency · Hz (log scale)", yaxis_title="Power · dB / Hz", hovermode="x")
                                psd_fig.update_xaxes(type="log")
                                freq_col.plotly_chart(psd_fig, width="stretch", config={"displayModeBar": False})
                            if first_spec is not None:
                                name, freq, times, power = first_spec
                                positive = freq > 0
                                fig = go.Figure(go.Heatmap(x=times, y=freq[positive], z=power[positive, :], colorscale=[[0, "#152b3b"], [.55, "#285a70"], [1, "#d77d37"]], colorbar=dict(title="dB / Hz"), hovertemplate="%{x:.3f} s<br>%{y:.1f} Hz<br>%{z:.1f} dB / Hz<extra></extra>"))
                                fig.update_layout(**PLOT, height=430, title=f"Short-time spectrum · {name}", xaxis_title="Time · seconds", yaxis_title="Frequency · Hz (log scale)")
                                fig.update_yaxes(type="log")
                                st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
                            st.download_button("Download extracted features", table.to_csv(index=False).encode(), "engine_window_features.csv", "text/csv")
                            if truncated:
                                st.caption("Feature extraction is limited to the first 2,000 overlapping windows per channel.")
                            st.caption("Welch PSD, short-time Fourier transform, RMS, excess kurtosis, skewness, spectral centroid, and crest factor are computed per window. These describe signals; they are not fault labels.")
                        else:
                            st.warning("Each selected channel needs at least 32 usable samples.")
            except Exception as exc:
                st.error(f"Could not read this recording: {exc}")
        else:
            st.markdown("**Expected format**")
            st.code("accel_x, accel_y, current\n0.013, -0.020, 0.004\n0.012, -0.018, 0.007\n…", language="csv")
            st.caption("CSV uses one sample per row. WAV accepts mono or multichannel microphone PCM. State labels, if present, remain in your source file but are excluded from signal features.")

    with training_tab:
        st.subheader("Train a fault-signature classifier from labeled recordings")
        st.caption("Upload multiple recordings with the same channels and sample rate. CSV columns must be sample-aligned; resample files to a common rate before upload. Assign one label to each complete recording. Evaluation holds out whole recordings so overlapping windows from one file cannot leak into both training and validation.")
        training_files = st.file_uploader(
            "Labeled recordings · one file per run", type=["csv", "wav"], accept_multiple_files=True,
            key="signature_training_files",
            help="Each recording needs a known condition label. Use at least two distinct recordings per label.",
        )
        if training_files:
            try:
                loaded = []
                shared_channels = None
                for uploaded in training_files:
                    if uploaded.name.lower().endswith(".wav"):
                        sr, values = wavfile.read(BytesIO(uploaded.getvalue()))
                        values = np.asarray(values)
                        if values.ndim == 1:
                            values = values[:, None]
                        source = pd.DataFrame(values, columns=[f"microphone_{i + 1}" for i in range(values.shape[1])])
                    else:
                        sr = None
                        source = pd.read_csv(BytesIO(uploaded.getvalue()))
                    numeric = [c for c in source.select_dtypes(include=np.number).columns
                               if str(c).lower() not in {"time", "time_s", "timestamp", "timestamp_s", "time_ms", "cycle", "engine_id", "label", "fault_label"}]
                    if not numeric:
                        raise ValueError(f"{uploaded.name} has no numeric sensor channels.")
                    shared_channels = set(numeric) if shared_channels is None else shared_channels.intersection(numeric)
                    recording_id = f"{len(loaded) + 1}:{uploaded.name}"
                    loaded.append((recording_id, uploaded, source, sr))

                channel_choices = sorted(shared_channels or [])
                if not channel_choices:
                    raise ValueError("The recordings need at least one numeric channel name in common.")
                label_choices = ["normal", "knock", "misfire", "valve_train_wear", "sensor_fault", "other_fault"]
                with st.form("signature_classifier_form"):
                    labels = {}
                    for file_index, (recording_id, uploaded, _, _) in enumerate(loaded):
                        selected_label = st.selectbox(
                            f"Condition label · {uploaded.name}", [*label_choices, "Custom label"],
                            key=f"signature_label_{file_index}",
                        )
                        labels[recording_id] = (
                            st.text_input(f"Custom condition · {uploaded.name}", key=f"signature_custom_label_{file_index}").strip()
                            if selected_label == "Custom label" else selected_label
                        )
                    channel_a, channel_b, channel_c = st.columns(3)
                    with channel_a:
                        model_channels = st.multiselect("Synchronized channels", channel_choices,
                            default=channel_choices[:min(4, len(channel_choices))], key="signature_model_channels")
                    with channel_b:
                        csv_rate = st.number_input("CSV sample rate · Hz", min_value=1.0, max_value=2_000_000.0,
                            value=25_600.0, step=100.0, key="signature_model_rate")
                    with channel_c:
                        model_window = st.number_input("Window · seconds", min_value=.01, max_value=60.0,
                            value=1.0, step=.25, key="signature_model_window")
                    train_signatures = st.form_submit_button("Train and evaluate SVM", type="primary")

                if train_signatures:
                    if any(not label for label in labels.values()):
                        st.error("Enter a condition label for each custom-labeled recording.")
                    elif not model_channels:
                        st.error("Select at least one shared signal channel.")
                    else:
                        rates = {float(embedded_rate) if embedded_rate is not None else float(csv_rate)
                                 for _, _, _, embedded_rate in loaded}
                        if len(rates) != 1:
                            st.error("All recordings need the same sample rate. Resample files first, then upload again.")
                        else:
                            feature_frames = []
                            for recording_id, uploaded, source, embedded_rate in loaded:
                                rate = float(embedded_rate) if embedded_rate is not None else float(csv_rate)
                                channels = {channel: pd.to_numeric(source[channel], errors="coerce").to_numpy(dtype=float)
                                            for channel in model_channels}
                                feature_frame = extract_window_signatures(
                                    channels, rate, float(model_window), overlap=.5, recording_id=recording_id,
                                )
                                feature_frame["fault_label"] = labels[recording_id]
                                feature_frames.append(feature_frame)
                            signature_rows = pd.concat(feature_frames, ignore_index=True)
                            result = fit_recording_heldout_svm(signature_rows)
                            result["sample_rate"] = next(iter(rates))
                            result["channels"] = list(model_channels)
                            result["window_seconds"] = float(model_window)
                            st.session_state["signature_classifier_result"] = result

                result = st.session_state.get("signature_classifier_result")
                if result is not None:
                    cards = st.columns(4)
                    with cards[0]: metric("Recording-held-out accuracy", f"{result['accuracy']:.1%}", "single grouped split")
                    with cards[1]: metric("Macro F1", f"{result['macro_f1']:.3f}", "all fault labels equally weighted")
                    with cards[2]: metric("Validation recordings", str(result["test_recordings"]), "kept out of fitting")
                    with cards[3]: metric("Training recordings", str(result["train_recordings"]), "whole files")
                    report_table = pd.DataFrame({
                        label: metrics for label, metrics in result["classification_report"].items()
                        if isinstance(metrics, dict)
                    }).T
                    st.dataframe(report_table, width="stretch")
                    st.dataframe(pd.DataFrame(result["confusion_matrix"], index=result["classes"], columns=result["classes"]), width="stretch")
                    st.dataframe(result["test_predictions"], width="stretch", hide_index=True)
                    st.caption("This is an experimental model trained only on your uploaded labels. A small number of recordings or recordings from one engine can produce optimistic or unstable results; it is not a validated vehicle fault detector.")
                    inference_file = st.file_uploader("Score a new unlabeled recording · CSV or WAV", type=["csv", "wav"], key="signature_inference_file")
                    if inference_file is not None:
                        if inference_file.name.lower().endswith(".wav"):
                            inference_rate, inference_audio = wavfile.read(BytesIO(inference_file.getvalue()))
                            inference_audio = np.asarray(inference_audio)
                            if inference_audio.ndim == 1:
                                inference_audio = inference_audio[:, None]
                            inference_data = pd.DataFrame(inference_audio, columns=[f"microphone_{i + 1}" for i in range(inference_audio.shape[1])])
                        else:
                            inference_rate = float(st.number_input("New CSV sample rate · Hz", min_value=1.0, max_value=2_000_000.0,
                                value=float(result["sample_rate"]), step=100.0, key="signature_inference_rate"))
                            inference_data = pd.read_csv(BytesIO(inference_file.getvalue()))
                        if float(inference_rate) != float(result["sample_rate"]):
                            st.error("The new recording's sample rate must match the training recordings.")
                        elif any(channel not in inference_data for channel in result["channels"]):
                            st.error("The new recording is missing one or more training channels: " + ", ".join(result["channels"]))
                        else:
                            inference_signals = {channel: pd.to_numeric(inference_data[channel], errors="coerce").to_numpy(dtype=float)
                                                 for channel in result["channels"]}
                            inference_rows = extract_window_signatures(
                                inference_signals, result["sample_rate"], result["window_seconds"], recording_id=inference_file.name,
                            )
                            prediction = result["model"].predict(inference_rows[result["feature_columns"]])
                            scores = result["model"].predict_proba(inference_rows[result["feature_columns"]])
                            scored = inference_rows[["window", "start_s", "end_s"]].copy()
                            scored["predicted_condition"] = prediction
                            for score_index, label in enumerate(result["classes"]):
                                scored[f"score__{label}"] = scores[:, score_index]
                            st.dataframe(scored, width="stretch", hide_index=True)
            except Exception as exc:
                st.error(f"Could not train the recording-level classifier: {exc}")


elif page == "Live OBD-II":
    heading("Vehicle telemetry / read-only adapter", "Capture supported engine PIDs.",
            "Poll standard OBD-II sensor values through a connected ELM327-compatible adapter, inspect the captured trace, and export it for synchronized analysis with uploaded recordings.")
    st.markdown("<div class='signal-card'><b>Connection needed:</b> connect an OBD-II adapter to the vehicle and this computer, turn the ignition on, then select its serial port. Capture sends read-only standard sensor requests. Vehicle support varies; standard OBD-II does not guarantee 30 channels, and high-rate microphone/accelerometer data still needs separate acquisition hardware.</div>", unsafe_allow_html=True)
    try:
        import obd
        try:
            detected_ports = obd.scan_serial()
        except Exception:
            detected_ports = []
        port_choices = ["Auto-detect", *[str(p) for p in detected_ports]]
        controls = st.columns(3)
        with controls[0]:
            selected_port = st.selectbox("Adapter port", port_choices)
        with controls[1]:
            capture_seconds = st.number_input("Capture duration · seconds", min_value=5, max_value=300, value=30, step=5)
        with controls[2]:
            interval = st.number_input("Target sample interval · seconds", min_value=.5, max_value=10.0, value=1.0, step=.5)
        progress_slot = st.empty()
        if st.button("Connect and capture", type="primary"):
            from engine_lab.obd_live import capture_obd
            progress_bar = progress_slot.progress(0, text="Connecting to adapter…")
            def update_capture_progress(elapsed, duration, samples):
                fraction = min(1.0, elapsed / duration)
                progress_bar.progress(fraction, text=f"Capturing · {samples} samples · {elapsed:.0f}/{duration} s")
            try:
                with st.spinner("Reading supported vehicle sensors…"):
                    captured, supported = capture_obd(
                        None if selected_port == "Auto-detect" else selected_port,
                        int(capture_seconds), float(interval), progress=update_capture_progress,
                    )
                st.session_state["obd_capture"] = captured
                st.session_state["obd_supported_pids"] = supported
            except Exception as exc:
                st.error(str(exc))
        captured = st.session_state.get("obd_capture")
        if captured is not None and not captured.empty:
            supported = st.session_state.get("obd_supported_pids", [])
            st.caption(f"Supported channels from this vehicle: {', '.join(supported)}. Rows are live samples collected during this session.")
            display = captured.copy()
            display["timestamp_utc"] = pd.to_datetime(display["timestamp_utc"], utc=True)
            channels = [c for c in display.select_dtypes(include=np.number).columns]
            if channels:
                fig = px.line(display, x="timestamp_utc", y=channels, title="Captured OBD-II sensor trace")
                fig.update_layout(**(PLOT | {"height": 430}), xaxis_title="Time · UTC", yaxis_title="Reported PID value · native units")
                st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
            st.dataframe(display, width="stretch", hide_index=True)
            st.download_button("Download captured telemetry", captured.to_csv(index=False).encode(), "obd_telemetry.csv", "text/csv")
    except ImportError:
        st.info("Live capture needs the optional Python-OBD package. Install it with `python -m pip install obd`, then restart the app.")


elif page == "Failure probability":
    heading("Labeled fault benchmark", "Probability before the fault spreads.",
            "A calibrated warning model trained on official fault-injection runs. It estimates the probability that a known benchmark fault is active, with a separate 30-second onset warning and fault-mode isolation.")
    feature_path = Path(__file__).parent / "data" / "liu_ice" / "processed_features.csv.gz"
    if not feature_path.exists():
        st.warning("The LiU-ICE features are not prepared yet. Run `python prepare_liu_ice.py` from the project folder, then refresh this page.")
        st.markdown(f"Benchmark source: [{SOURCE_URL}]({SOURCE_URL})")
    else:
        with st.spinner("Loading labeled runs and evaluating trajectory-held-out models…"):
            windows = cached_liu_windows()
            active_model, early_model, isolator = cached_liu_models()
        summary = summarize_dataset(windows)
        active_metrics = active_model["metrics"]
        early_metrics = early_model["metrics"]
        metrics = st.columns(4)
        with metrics[0]: metric("Onset ≤30 s ROC AUC", f"{early_metrics['roc_auc']:.3f}", "held-out trajectories")
        with metrics[1]: metric("Onset ≤30 s PR AUC", f"{early_metrics['average_precision']:.3f}", f"baseline prevalence {early_metrics['positive_prevalence']:.1%}")
        with metrics[2]: metric("Onset probability Brier", f"{early_metrics['brier']:.3f}", "lower is better")
        with metrics[3]: metric("Alert threshold", f"{early_model['threshold']:.3f}", "maximum F2 on grouped OOF")
        st.markdown("<div class='rule'></div>", unsafe_allow_html=True)
        st.markdown("<div class='signal-card'><b>Interpretation:</b> the primary forecast is probability of one of four injected LiU-ICE benchmark faults (three sensor faults and one intake leak) beginning within 30 seconds. It is not probability of catastrophic engine failure. The benchmark has no run-to-failure lifetime labels, and its three normal runs are too few to establish real-world calibration.</div>", unsafe_allow_html=True)
        st.subheader("Held-out trajectory replay")
        test_rows = active_model["test_windows"].copy()
        run_ids = sorted(test_rows["trajectory_id"].unique())
        chosen_run = st.selectbox("Test trajectory (never used for model fitting)", run_ids,
                                  format_func=lambda x: f"{x} · {test_rows.loc[test_rows.trajectory_id == x, 'scenario'].iloc[0]}")
        replay = test_rows.loc[test_rows["trajectory_id"] == chosen_run].sort_values("window_start_s").copy()
        early_rows = early_model["test_windows"]
        early_replay = early_rows.loc[early_rows["trajectory_id"] == chosen_run].set_index("window_start_s")
        replay["Onset ≤30 s probability"] = replay["window_start_s"].map(early_replay["fault_probability"]).fillna(0.0)
        replay["Observed fault state"] = replay["fault_active"].map({0: "No injected fault active", 1: "Injected fault active"})
        onset_values = replay["injection_time_s"].dropna()
        default_step = len(replay) - 1
        if not onset_values.empty:
            default_step = int(np.argmin(np.abs(replay["window_end_s"].to_numpy() - (float(onset_values.iloc[0]) - 15))))
        selected_step = st.slider("Replay position", 0, len(replay) - 1, default_step,
                                  format="Window %d", help="Move through the held-out run to inspect the score at each five-second window.")
        latest = replay.iloc[selected_step]
        truth = int(latest["fault_active"])
        c1, c2, c3, c4 = st.columns(4)
        with c1: metric("Onset probability ≤30 s", f"{latest['Onset ≤30 s probability']:.1%}", "calibrated forecast")
        with c2: metric("Observed benchmark state", "FAULT" if truth else "NORMAL", str(latest["scenario"]))
        with c3: metric("Replay time", f"{latest['window_end_s']:.0f} s", "5-second feature windows")
        with c4: metric("Held-out engine runs", str(active_metrics["test_trajectories"]), f"{active_metrics['train_trajectories']} train trajectories")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=replay["window_end_s"], y=replay["Onset ≤30 s probability"], name="P(onset within 30 s)", line=dict(color="#d77d37", width=2.5), fill="tozeroy", fillcolor="rgba(215,125,55,.12)"))
        fig.add_trace(go.Scatter(x=replay["window_end_s"], y=replay["fault_probability"], name="P(fault active)", line=dict(color="#285a70", width=2, dash="dot")))
        fig.add_hline(y=early_model["threshold"], line_dash="dash", line_color="#d77d37", annotation_text="early-warning threshold")
        injection = replay["injection_time_s"].dropna()
        if not injection.empty:
            fig.add_vline(x=float(injection.iloc[0]), line_dash="dot", line_color="#9f3c32", annotation_text="injected onset")
        fig.update_layout(**(PLOT | {"height": 410, "yaxis": dict(range=[0, 1], gridcolor="#e3e9ec")}), xaxis_title="Time · seconds from run start", yaxis_title="Calibrated probability", legend=dict(orientation="h", y=1.12), hovermode="x unified", uirevision=f"liu-{chosen_run}")
        fig.update_xaxes(rangeslider_visible=True, rangeselector=dict(buttons=[dict(step="all", label="Full run")]))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.caption(f"Group split is by full trajectory. Onset-model test: ROC AUC {early_metrics['roc_auc']:.3f}, PR AUC {early_metrics['average_precision']:.3f} (base rate {early_metrics['positive_prevalence']:.1%}), Brier {early_metrics['brier']:.3f}, recall {early_metrics['recall']:.1%}, precision {early_metrics['precision']:.1%}, false-alarm rate {early_metrics['false_alarm_rate']:.1%}. Fault-active model: ROC AUC {active_metrics['roc_auc']:.3f}, recall {active_metrics['recall']:.1%}, false-alarm rate {active_metrics['false_alarm_rate']:.1%} at the 5%-healthy-OOF threshold. Thresholds are selected from grouped out-of-fold training predictions. This is one deterministic grouped holdout and needs external-engine validation.")
        left, right = st.columns([1, 1], gap="large")
        with left:
            st.subheader("Fault-mode isolation")
            st.caption(f"Conditional mode classifier · held-out active-fault accuracy {isolator['accuracy']:.1%} · macro F1 {isolator['macro_f1']:.3f}. Scores below are conditional on a fault being active; they are not calibrated probabilities of failure.")
            selected_window = replay.iloc[selected_step:selected_step + 1]
            mode_prob = isolator["model"].predict_proba(selected_window[active_model["selected_features"]])[0]
            mode_frame = pd.DataFrame({"Fault mode": isolator["model"].classes_, "Conditional score": mode_prob})
            mode_frame["Fault mode"] = mode_frame["Fault mode"].map(lambda x: FAULT_LABELS.get(x, x))
            st.dataframe(mode_frame.sort_values("Conditional score", ascending=False).style.format({"Conditional score": "{:.1%}"}), width="stretch", hide_index=True)
        with right:
            st.subheader("Selected signal signatures")
            importance = early_model["feature_importance"].loc[early_model["selected_features"]].head(12).sort_values()
            imp = importance.rename_axis("Window feature").reset_index(name="RF importance")
            fig = px.bar(imp, x="RF importance", y="Window feature", orientation="h", color_discrete_sequence=["#285a70"])
            fig.update_layout(**PLOT, height=350)
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
            st.caption(f"Onset-warning model selected {len(early_model['selected_features'])} of {len(early_model['features'])} window features using training-run random-forest importance until 95% cumulative importance (bounded to 8–24 features). No held-out trajectories were used for selection.")
        with st.expander("Dataset coverage and scope"):
            st.write(f"{summary['runs']} trajectories · {summary['windows']:,} five-second windows · {summary['channels']} synchronized signals · {summary['features']} engineered features · {summary['classes']} labeled run types.")
            st.write("Each channel contributes window mean, standard deviation, range, and linear slope. Fault labels use the benchmark's published injection timestamps. Windows are non-overlapping to limit near-duplicate samples; the split keeps complete trajectories together.")
            st.markdown(f"Official source: [LiU-ICE Diagnostic XAI benchmark]({SOURCE_URL}) · 20 Hz engine-bench data · four injected fault types.")
            st.write("The Bosch gasoline-engine maps remain the source for operating-point and best-performance analysis. These independently collected benchmark sources are not fused into one model. Acoustic knock and high-frequency vibration remain uncovered because neither dataset includes the required synchronized microphone/accelerometer channels at suitable rates.")


elif page == "65-channel diesel":
    heading("Separate marine-diesel benchmark", "65 synchronized channels. Four pre-fault transitions.",
            "A second, physically fault-injected engine dataset gives the workbench genuine high-dimensional signals. It remains separate from the gasoline and automotive datasets because engine type, sensors, units, and fault definitions differ.")
    with st.spinner("Loading the controlled fault runs and fitting a trajectory-held-out anomaly model…"):
        marine_rows = cached_marine_rows()
        marine_model = cached_marine_model()
    marine_summary = summarize_marine(marine_rows)
    mm = marine_model["metrics"]
    mc = st.columns(4)
    with mc[0]: metric("Synchronous channels", str(marine_summary["channels"]), "usable per fault run")
    with mc[1]: metric("Transition runs modeled", str(mm["train_runs"] + mm["test_runs"]), "whole runs held out")
    with mc[2]: metric("Held-out ROC AUC", f"{mm['roc_auc']:.3f}", f"{mm['test_runs']} test runs · {mm['train_runs']} train")
    with mc[3]: metric("Held-out PR AUC", f"{mm['average_precision']:.3f}", f"{mm['prevalence']:.1%} anomaly prevalence")
    st.markdown("<div class='signal-card'><b>Scope:</b> this is a marine diesel test bench, not a car engine. The calibrated output is probability of the dataset's labeled induced anomaly state. It does not mean probability of catastrophic engine failure or lifetime. This channel set is the project's 30+ stream high-dimensional track; do not combine its model with the gasoline/automotive model.</div>", unsafe_allow_html=True)
    st.subheader("Held-out physical-run replay")
    marine_test = marine_model["test_rows"]
    marine_ids = sorted(marine_test.trajectory_id.unique())
    chosen_marine = st.selectbox("Test recording (entire file withheld during fitting)", marine_ids,
                                 format_func=lambda x: f"{x} · {marine_test.loc[marine_test.trajectory_id == x, 'fault_mode'].iloc[0]}")
    marine_replay = marine_test.loc[marine_test.trajectory_id == chosen_marine].sort_values("time_s")
    l, r = st.columns(2)
    with l: metric("False-alarm rate", f"{mm['false_alarm_rate']:.1%}", "threshold from maximum grouped-OOF F2")
    with r: metric("Anomaly recall", f"{mm['recall']:.1%}", f"precision {mm['precision']:.1%} · Brier {mm['brier']:.3f}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=marine_replay["time_s"], y=marine_replay["anomaly_probability"], name="P(induced anomaly active)", line=dict(color="#285a70", width=2), fill="tozeroy", fillcolor="rgba(40,90,112,.12)"))
    fig.add_trace(go.Scatter(x=marine_replay["time_s"], y=marine_replay["anomaly_active"], name="Observed anomaly state", line=dict(color="#d77d37", width=1.5, shape="hv")))
    fig.add_hline(y=marine_model["threshold"], line_dash="dash", line_color="#9f3c32", annotation_text="alert threshold")
    fig.update_layout(**(PLOT | {"height": 390, "yaxis": dict(range=[0, 1], gridcolor="#e3e9ec")}), xaxis_title="Time since run start · seconds", yaxis_title="Probability / observed state", legend=dict(orientation="h", y=1.12), hovermode="x unified", uirevision=f"marine-{chosen_marine}")
    fig.update_xaxes(rangeslider_visible=True, rangeselector=dict(buttons=[dict(step="all", label="Full run")]))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.caption(f"One deterministic trajectory-level holdout: {mm['test_rows']:,} samples across {mm['test_runs']} unseen physical runs. At the grouped-OOF maximum-F2 threshold, ROC AUC {mm['roc_auc']:.3f}, PR AUC {mm['average_precision']:.3f}, recall {mm['recall']:.1%}, precision {mm['precision']:.1%}, false-alarm rate {mm['false_alarm_rate']:.1%}, Brier {mm['brier']:.3f}. High event prevalence and repeated rows within runs make PR AUC and per-run replay essential.")
    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.subheader("Most informative channels")
        top = marine_model["importance"].head(18).sort_values()
        bars = px.bar(top.rename_axis("Channel").reset_index(name="RF importance"), x="RF importance", y="Channel", orientation="h", color_discrete_sequence=["#285a70"])
        bars.update_layout(**PLOT, height=440)
        st.plotly_chart(bars, width="stretch", config={"displayModeBar": False})
        st.caption(f"{len(marine_model['selected_features'])} of {len(marine_model['features'])} numeric channels retained using training-only RF importance (95% cumulative, bounded to 12–30). Missing channels are median-imputed from training runs.")
    with col2:
        st.subheader("Fault coverage")
        coverage = pd.DataFrame({"Fault mode": list(marine_summary["fault_modes"].keys()), "Independent runs": list(marine_summary["fault_modes"].values())})
        st.dataframe(coverage, width="stretch", hide_index=True)
        st.markdown(f"**{marine_summary['samples']:,} labeled samples** across 15 scenario recordings. Each fault CSV has time, anomaly-state label, and up to 70 additional channels; 65 channels passed the ≥50% coverage filter. Two injector-nozzle recordings contain anomaly-only samples and are excluded from pre-fault transition training. The official [Zenodo dataset](https://zenodo.org/records/19857425) is CC BY 4.0. Its [data descriptor](https://arxiv.org/abs/2607.19444) describes the test-bench setup and physically induced anomalies.")


elif page == "RUL prognosis":
    heading("Predictive maintenance", "Train on failures that happened.",
            "Bring complete run-to-failure histories or explicit RUL labels. The holdout split keeps entire engines out of training, so validation measures transfer to unseen units.")
    st.markdown("<div class='signal-card'>Bosch's public engine workbooks do not include fault events or time-to-failure labels. Upload labeled history to enable a prognosis model. We will not infer failure labels from anomaly scores.</div>", unsafe_allow_html=True)
    with st.expander("Built-in benchmark · NASA simulated turbofan run-to-failure", expanded=False):
        st.caption("This is an external prognostics benchmark for exercising the RUL workflow. C-MAPSS is a simulated aircraft turbofan, so its scores do not transfer to automotive gasoline engines.")
        if not (CMAPSS_DATA_DIR / "train_FD001.txt").is_file():
            st.info("Prepare the public NASA data once from the project folder:")
            st.code("python prepare_cmapss.py", language="powershell")
            st.markdown("Source: [NASA Prognostics Center of Excellence data repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/). The preparation script downloads and safely extracts FD001.")
        elif st.button("Fit and score NASA FD001", key="fit_cmapss"):
            with st.spinner("Fitting on complete engine histories and scoring held-out engines…"):
                st.session_state["cmapss_result"] = cached_cmapss_model()
        cmapss_state = st.session_state.get("cmapss_result")
        if cmapss_state:
            nasa_result, nasa_estimates = cmapss_state
            cards = st.columns(4)
            with cards[0]: metric("Engine-held-out MAE", f"{nasa_result['mae']:.1f}", "cycles · train partition")
            with cards[1]: metric("NASA test MAE", f"{nasa_estimates['Absolute error · cycles'].mean():.1f}", "cycles · official FD001 test units")
            with cards[2]: metric("Train engines", str(nasa_result["train_engines"]), "complete run-to-failure")
            with cards[3]: metric("Held-out engines", str(nasa_result["test_engines"]), "group split")
            st.dataframe(nasa_estimates.style.format({c: "{:.1f}" for c in nasa_estimates.select_dtypes(number).columns}), width="stretch", hide_index=True)
            st.caption("The model uses trailing sensor statistics. These are benchmark results on simulated aircraft engines; use vehicle-specific labeled runs to make automotive estimates.")
    uploaded = st.file_uploader("Run-to-failure data · CSV", type=["csv"], key="rul_upload",
        help="Long format: each row is one cycle/time point; include engine ID, cycle/time, numeric sensor channels, and either RUL or a failure-event flag.")
    if uploaded is not None:
        try:
            data = pd.read_csv(uploaded)
            cols = list(data.columns)
            numeric_cols = list(data.select_dtypes(include=np.number).columns)
            if len(data) < 100:
                st.warning("This is a small table for temporal modeling. Continue only if the records form complete run-to-failure histories.")
            id_guess = next((c for c in cols if c.lower() in {"engine_id", "unit", "unit_number", "machine_id"}), cols[0])
            time_guess = next((c for c in cols if c.lower() in {"cycle", "time", "time_s", "timestamp", "operating_time"}), numeric_cols[0] if numeric_cols else cols[0])
            id_col = st.selectbox("Engine ID", cols, index=cols.index(id_guess))
            time_col = st.selectbox("Cycle / time", numeric_cols or cols, index=(numeric_cols or cols).index(time_guess) if time_guess in (numeric_cols or cols) else 0)
            label_choices = ["(derive from failure event)"] + numeric_cols
            target_choice = st.selectbox("RUL target (optional)", label_choices)
            failure_choices = ["(none)"] + numeric_cols
            failure_choice = st.selectbox("Final failure event column", failure_choices,
                help="If no RUL target is supplied, mark the final failure row as 1 for each complete engine history.")
            if st.button("Fit and evaluate on held-out engines", type="primary"):
                target_col = None if target_choice.startswith("(") else target_choice
                failure_col = None if failure_choice == "(none)" else failure_choice
                if target_col is None and failure_col is None:
                    st.error("Select an RUL target or a final failure event column.")
                else:
                    X, y, groups, _ = build_rul_features(data, id_col, time_col, target_col, failure_col)
                    with st.spinner("Splitting by engine and fitting a temporal-feature model…"):
                        result = fit_rul_model(X, y, groups)
                    excluded = {id_col, time_col}
                    if target_col:
                        excluded.add(target_col)
                    if failure_col:
                        excluded.add(failure_col)
                    sensor_channels = [c for c in data.select_dtypes(include=np.number).columns if c not in excluded]
                    st.session_state["rul_model_state"] = {"result": result, "sensors": sensor_channels,
                        "engine_col": id_col, "time_col": time_col, "window": 8, "target_col": target_col}
            model_state = st.session_state.get("rul_model_state")
            if model_state:
                result = model_state["result"]
                c1, c2, c3 = st.columns(3)
                with c1: metric("Held-out RUL MAE", f"{result['mae']:.2f}", "cycles or time units")
                with c2: metric("Held-out engines", str(result["test_engines"]), "never used for training")
                with c3: metric("Held-out R²", f"{result['r2']:.2f}", "interpret with sample size")
                left, right = st.columns(2, gap="large")
                with left:
                    st.subheader("Sensor features selected by permutation")
                    imp = result["importance"].head(15).sort_values().rename_axis("Feature").reset_index(name="Importance")
                    fig = px.bar(imp, x="Importance", y="Feature", orientation="h", color_discrete_sequence=["#285a70"])
                    fig.update_layout(**PLOT, height=420)
                    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
                    st.caption(f"{len(result['selected_features'])} features selected on an inner engine-group validation split; scored on engines held out from both fitting and selection.")
                with right:
                    st.subheader("Held-out engine estimates")
                    pred = pd.DataFrame({"Observed RUL": result["y_test"].to_numpy(), "Predicted RUL": result["pred_test"]})
                    fig = px.scatter(pred, x="Observed RUL", y="Predicted RUL", opacity=.65, color_discrete_sequence=["#d77d37"])
                    edge = [float(min(pred.min())), float(max(pred.max()))]
                    fig.add_shape(type="line", x0=edge[0], y0=edge[0], x1=edge[1], y1=edge[1], line=dict(color="#285a70", dash="dot"))
                    fig.update_layout(**PLOT, height=420)
                    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
                st.caption(f"Engine-group split: {result['train_engines']} train / {result['test_engines']} test. Labels are capped/uncapped exactly as supplied. Rolling means, standard deviations, and trailing slopes use past samples only. The full data model is not scored on its training rows.")
                st.markdown("<div class='rule'></div>", unsafe_allow_html=True)
                st.subheader("Estimate current engines")
                live_file = st.file_uploader("Unlabeled current history · same sensor columns", type=["csv"], key="live_upload",
                    help="Provide engine ID, cycle/time, and every sensor channel used to train. RUL or failure labels are not required.")
                if live_file is not None:
                    live = pd.read_csv(live_file)
                    try:
                        live_X, live_ids = build_live_features(live, model_state["engine_col"], model_state["time_col"], model_state["sensors"])
                        selected = result["selected_features"]
                        estimates = result["production_model"].predict(live_X[selected])
                        output = pd.DataFrame({"Engine": live_ids, "Estimated remaining life": estimates})
                        st.dataframe(output.style.format({"Estimated remaining life": "{:.2f}"}), width="stretch", hide_index=True)
                        st.caption("Estimates are in the units of the uploaded RUL/time target. Apply only after checking the held-out error and confirming the new engines share the training population.")
                    except Exception as exc:
                        st.error(f"Could not score current history: {exc}")
        except Exception as exc:
            st.error(f"Could not prepare this history: {exc}")
    st.markdown("<div class='rule'></div>", unsafe_allow_html=True)
    st.markdown("**CSV schema**  ")
    st.code("engine_id,cycle,rul,oil_pressure,coolant_temp,vibration_rms\nA,1,120,3.2,82.1,0.18\nA,2,119,3.1,82.3,0.19\n…", language="csv")
    st.caption("Those example values describe column structure only. They are not bundled training data.")
