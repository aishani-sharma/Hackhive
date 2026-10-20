"""Chart builders and small tables for the dashboard. Every chart carries a Real, Simulated or Mixed tag in its title."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from enginetwin.plots import efficiency_heatmap
from enginetwin.schema import SOURCES

ACTUAL, OPTIMISED, ALERT, NORMAL = "#7f8c9a", "#1f9d6b", "#d9534f", "#4a7fb5"
KEY_ORDERS = ((0.5, "0.5 (misfire)"), (1.0, "1"), (2.0, "2 (firing)"), (4.0, "4"))
DRIVE_MODES = {  # engine speed and load a driver in each mode typically holds (assumed)
    "Eco": {"rpm": 1800.0, "load": 0.35, "speed_kmh": 60.0},
    "Normal": {"rpm": 2400.0, "load": 0.45, "speed_kmh": 80.0},
    "Sport": {"rpm": 3400.0, "load": 0.65, "speed_kmh": 100.0},
}


def tagged_title(title: str, source: str) -> str:
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}, got {source!r}")
    return f"{title} [{source.capitalize()}]"


def drive_mode_point(mode: str) -> dict:
    if mode not in DRIVE_MODES:
        raise ValueError(f"unknown drive mode {mode!r}; expected one of {list(DRIVE_MODES)}")
    return dict(DRIVE_MODES[mode])


def gauge_figure(name, value, low, high, unit, source) -> go.Figure:
    fig = go.Figure(go.Indicator(mode="gauge+number", value=value, number={"suffix": f" {unit}"},
                                 gauge={"axis": {"range": [low, high]}, "bar": {"color": NORMAL}}))
    fig.update_layout(title=tagged_title(name, source), height=220, margin={"t": 60, "b": 10, "l": 20, "r": 20})
    return fig


def spectrogram_figure(spec, duration_s, channel, source) -> go.Figure:
    """`spec` is (frequency band, time frame) as built by enginetwin.spectrograms (64 bands up to 8 kHz)."""
    bands, frames = spec.shape
    fig = go.Figure(go.Heatmap(z=spec, x=np.linspace(0, duration_s, frames), y=np.linspace(0, 8000, bands), colorscale="Viridis",
                               colorbar={"title": "log10 level"}))
    fig.update_layout(title=tagged_title(f"{channel} spectrogram", source), xaxis_title="Time (s)", yaxis_title="Frequency (Hz)")
    return fig


def order_spectrum_figure(orders, amplitudes, source) -> go.Figure:
    fig = go.Figure(go.Scatter(x=orders, y=amplitudes, mode="lines", line={"color": NORMAL}))
    for order, label in KEY_ORDERS:
        fig.add_vline(x=order, line={"dash": "dot", "color": ALERT})
        fig.add_annotation(x=order, y=1, yref="paper", text=label, showarrow=False, yanchor="bottom")
    fig.update_layout(title=tagged_title("Order spectrum", source), xaxis_title="Order (multiples of crank speed)", yaxis_title="Amplitude")
    return fig


def envelope_figure(freqs, amplitudes, rpm, source) -> go.Figure:
    fig = go.Figure(go.Scatter(x=freqs, y=amplitudes, mode="lines", line={"color": NORMAL}))
    for hz, label in ((rpm / 120.0, "once per cycle (valve)"), (rpm / 30.0, "once per firing")):
        fig.add_vline(x=hz, line={"dash": "dot", "color": ALERT}, annotation_text=label)
    fig.update_layout(title=tagged_title("Envelope spectrum of the valve band", source), xaxis_title="Repeat rate (Hz)", yaxis_title="Amplitude")
    return fig


def cylinder_figure(energy: dict, source) -> go.Figure:
    weakest = min(energy, key=energy.get)
    colours = [ALERT if c == weakest else NORMAL for c in energy]
    fig = go.Figure(go.Bar(x=[f"Cylinder {c}" for c in energy], y=list(energy.values()), marker={"color": colours}))
    fig.update_layout(title=tagged_title(f"Energy per cylinder (weakest: cylinder {weakest})", source), yaxis_title="Firing energy")
    return fig


def probability_figure(probabilities: dict, source) -> go.Figure:
    fig = go.Figure(go.Bar(y=list(probabilities), x=list(probabilities.values()), orientation="h",
                           marker={"color": [ALERT if p > 0.5 else NORMAL for p in probabilities.values()]},
                           text=[f"{p:.0%}" for p in probabilities.values()], textposition="outside"))
    fig.update_layout(title=tagged_title("Fault probabilities", source), xaxis={"range": [0, 1.1], "title": "Probability"})
    return fig


def heat_overlay_figure(spec, heat, source) -> go.Figure:
    """The spectrogram with the occlusion heat map on top: bright areas are what the network relied on."""
    fig = go.Figure(
        [
            go.Heatmap(z=spec, colorscale="Greys", showscale=False),
            go.Heatmap(z=np.where(heat > 0, heat, np.nan), colorscale="Hot", opacity=0.7, colorbar={"title": "Importance"}),
        ]
    )
    fig.update_layout(title=tagged_title("What triggered the diagnosis", source), xaxis_title="Time frame", yaxis_title="Frequency band")
    return fig


def efficiency_map_figure(cells, current, optimal, source) -> go.Figure:
    fig = efficiency_heatmap(cells, source)
    fig.layout.title.text = tagged_title("Engine efficiency map", source)
    for name, (rpm, torque), symbol, colour in (("Current point", current, "circle", "white"), ("Optimal point", optimal, "diamond", "#ffd166")):
        fig.add_trace(go.Scatter(x=[rpm], y=[torque], mode="markers", name=name,
                                 marker={"symbol": symbol, "size": 14, "color": colour, "line": {"color": "black", "width": 1.5}}))
    return fig


def series_figure(replay: pd.DataFrame, source) -> go.Figure:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=("Efficiency", "Fuel rate (L/h)", "Engine speed (rpm)"))
    for row, (actual, optimised) in enumerate((("actual_eff", "opt_eff"), ("actual_fuel_l_h", "opt_fuel_l_h"), ("actual_rpm", "opt_rpm")), start=1):
        for name, column, colour in (("Actual", actual, ACTUAL), ("Optimised", optimised, OPTIMISED)):
            fig.add_trace(go.Scatter(x=replay["time_s"], y=replay[column], name=name, legendgroup=name, showlegend=row == 1,
                                     line={"color": colour, "width": 1.5}), row=row, col=1)
    fig.update_layout(title=tagged_title("Actual against optimised", source), height=640)
    fig.update_xaxes(title_text="Time (s)", row=3, col=1)
    return fig


def health_figure(days, true_health, estimated_health, service_day, source) -> go.Figure:
    fig = go.Figure(
        [
            go.Scatter(x=days, y=true_health, name="True health (simulated wear)", line={"color": ACTUAL, "dash": "dot"}),
            go.Scatter(x=days, y=estimated_health, name="Estimated from vibration", line={"color": NORMAL}),
        ]
    )
    fig.add_hline(y=20, line={"dash": "dash", "color": ALERT}, annotation_text="service threshold")
    if np.isfinite(service_day):
        fig.add_vline(x=service_day, line={"dash": "dot", "color": ALERT}, annotation_text=f"predicted service: day {service_day:.0f}")
    fig.update_layout(title=tagged_title("Valve health index", source), xaxis_title="Day", yaxis={"title": "Health (100 new, 0 service)", "range": [0, 105]})
    return fig


def confusion_figure(matrix, labels, title, source) -> go.Figure:
    fig = go.Figure(go.Heatmap(z=matrix, x=labels, y=labels, colorscale="Blues", text=matrix, texttemplate="%{text}", showscale=False))
    fig.update_layout(title=tagged_title(title, source), xaxis_title="Predicted", yaxis_title="True", yaxis={"autorange": "reversed"})
    return fig


def data_provenance() -> pd.DataFrame:
    """Which data trained which model, and which data the demonstration uses (never the same data)."""
    rows = [
        ("Ocslab drivers A, B, D, E, F, G, I, J", "Real", "efficiency model (boosted trees and network)", "", "8 of 10 Kia Soul drivers; 72,090 usable 1 Hz rows"),
        ("Ocslab drivers C, H", "Real", "", "efficiency validation and the Ocslab trip replay", "held out of training entirely"),
        ("VED (ICE cars)", "Real", "", "VED trip replay", "213 cars, never trained on; no torque, so efficiency there rests on an assumption and did not validate"),
        ("EPA ALPHA engine map", "Real", "", "reference sweet spot", "a different engine (Mazda SkyActiv 2.0), used only for comparison"),
        ("Simulated sound and vibration", "Simulated", "SVM, softmax, CNN, autoencoder, severity models", "live engine and signatures", "no real engine audio exists in the project; 2 s windows with a fault label"),
        ("Simulated OBD readings", "Mixed", "OBD network, CNN, all feature models", "live gauges", "calibrated to real Ocslab Kia Soul rows, fault effects assumed"),
        ("Simulated splits: validation, tests (same, unseen rpm, 15/5/0 dB)", "Simulated", "", "results tables", "never used for training; unseen rpm is 3000 to 5000 rpm, absent from training"),
        ("Uploaded recording", "Real", "", "Real recording tab", "diagnosed by models trained on simulated faults only"),
    ]
    return pd.DataFrame(rows, columns=["data", "kind", "trains", "demonstrates", "notes"])
