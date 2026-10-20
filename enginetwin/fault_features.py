"""The hand-made fault feature vector: one fixed-length row per window, built from the signal tools in signals.py."""
from pathlib import Path

import numpy as np

from enginetwin.schema import Window
from enginetwin.signals import (
    band_energy_fraction,
    envelope_spectrum,
    half_order_ratio,
    order_amplitude,
    per_cylinder_energy,
)

KNOCK_BAND = (6000, 7000)
VALVE_BAND = (2500, 3500)
KNOCK_FLOOR = ((4000, 5000), (7500, 8000))  # nearby bands with no fault resonance: the local noise floor
VALVE_FLOOR = ((1600, 2200), (4000, 5000))
CYLINDER_FIRINGS = 12  # firings used for the per-cylinder energy; short, so a small speed error cannot drift the windows
RPM_SEARCH = 0.15  # the firing line is looked for within +-15% of the OBD speed (a misfire dips it by up to ~11% at 800 rpm)
ENVELOPE_FLOOR_HZ = (5.0, 500.0)
OBD_KEYS = (
    "engine_speed", "calculated_load_value", "engine_coolant_temperature", "maf", "timing_advance",
    "short_term_fuel_trim_bank1", "long_term_fuel_trim_bank1",
)
_CHANNEL_FEATURES = (
    "half_order_ratio", "order1_ratio", "order2_amp", "knock_band_fraction", "valve_band_fraction",
    "valve_env_cycle", "valve_env_firing", "knock_env_firing", "cyl_cv", "cyl_min_ratio",
    "centroid_hz", "rolloff_hz", "flatness", "rms", "kurtosis", "crest", "knock_vs_floor", "valve_vs_floor",
)
FEATURE_NAMES = (
    [f"{channel}_{name}" for channel in ("sound", "vibration") for name in _CHANNEL_FEATURES]
    + [f"obd_{k}" for k in OBD_KEYS]
    + ["obd_total_fuel_trim"]
)


def _spectrum(signal, fs):
    return np.fft.rfftfreq(len(signal), 1 / fs), np.abs(np.fft.rfft(signal * np.hanning(len(signal)))) / len(signal)


def refine_rpm(sound, fs, rpm_obd) -> float:
    """Engine speed from the strongest firing-frequency line within 15% of the OBD figure (parabolic peak fit)."""
    freqs, spectrum = _spectrum(sound, fs)
    target = rpm_obd / 30.0
    near = np.flatnonzero((freqs >= target * (1 - RPM_SEARCH)) & (freqs <= target * (1 + RPM_SEARCH)))
    k = near[np.argmax(spectrum[near])]
    peak = freqs[k]
    if 0 < k < len(spectrum) - 1:
        left, mid, right = np.log(spectrum[k - 1] + 1e-30), np.log(spectrum[k] + 1e-30), np.log(spectrum[k + 1] + 1e-30)
        curvature = left - 2 * mid + right
        if curvature >= 0:  # no peak (silence or a flat spectrum): trust the OBD figure
            return float(rpm_obd)
        peak += (freqs[1] - freqs[0]) * 0.5 * (left - right) / curvature
    return float(30.0 * peak)


def _band_vs_floor(freqs, power, band, floor_bands) -> float:
    """log10 of the mean power in `band` over the mean power in the floor bands: 0 when the band is just noise."""
    inside = lambda low, high: (freqs >= low) & (freqs <= high)
    floor = np.concatenate([power[inside(*b)] for b in floor_bands])
    return float(np.log10((power[inside(*band)].mean() + 1e-30) / (floor.mean() + 1e-30)))


def _line_over_floor(freqs, amplitudes, hz) -> float:
    """Envelope line at `hz` relative to the median envelope level (a peak-to-floor ratio)."""
    floor = np.median(amplitudes[(freqs >= ENVELOPE_FLOOR_HZ[0]) & (freqs <= ENVELOPE_FLOOR_HZ[1])]) + 1e-12
    return float(amplitudes[np.abs(freqs - hz) <= 1.0].max() / floor)


def _channel_features(signal, fs, rpm) -> list:
    order2 = order_amplitude(signal, fs, rpm, 2.0)
    valve_freqs, valve_env = envelope_spectrum(signal, fs, band=VALVE_BAND)
    knock_freqs, knock_env = envelope_spectrum(signal, fs, band=KNOCK_BAND)
    energy = np.array(list(per_cylinder_energy(signal[: int(CYLINDER_FIRINGS * 30.0 / rpm * fs)], fs, rpm).values()))
    mean_energy = energy.mean() + 1e-30

    freqs, spectrum = _spectrum(signal, fs)
    power = spectrum**2
    total = power.sum() + 1e-30
    centroid = float((freqs * power).sum() / total)
    rolloff = float(freqs[min(np.searchsorted(np.cumsum(power), 0.85 * total), len(freqs) - 1)])
    flatness = float(np.exp(np.mean(np.log(power + 1e-30))) / (power.mean() + 1e-30))
    rms = float(np.sqrt(np.mean(signal**2)))
    centred = signal - signal.mean()
    variance = np.mean(centred**2) + 1e-30
    return [
        half_order_ratio(signal, fs, rpm),
        order_amplitude(signal, fs, rpm, 1.0) / (order2 + 1e-30),
        order2,
        band_energy_fraction(signal, fs, *KNOCK_BAND),
        band_energy_fraction(signal, fs, *VALVE_BAND),
        _line_over_floor(valve_freqs, valve_env, rpm / 120.0),
        _line_over_floor(valve_freqs, valve_env, rpm / 30.0),
        _line_over_floor(knock_freqs, knock_env, rpm / 30.0),
        float(energy.std() / mean_energy),
        float(energy.min() / mean_energy),
        centroid,
        rolloff,
        flatness,
        rms,
        float(np.mean(centred**4) / variance**2),
        float(np.abs(signal).max() / (rms + 1e-30)),
        _band_vs_floor(freqs, power, KNOCK_BAND, KNOCK_FLOOR),
        _band_vs_floor(freqs, power, VALVE_BAND, VALVE_FLOOR),
    ]


def extract_features(window: Window):
    """(feature vector, FEATURE_NAMES). Speed comes from the OBD `engine_speed` reading, refined by the sound."""
    rpm = refine_rpm(window.sound, window.sound_fs, window.obd["engine_speed"])
    obd = [float(window.obd[k]) for k in OBD_KEYS]
    vector = (
        _channel_features(window.sound, window.sound_fs, rpm)
        + _channel_features(window.vibration, window.vibration_fs, rpm)
        + obd
        + [obd[OBD_KEYS.index("short_term_fuel_trim_bank1")] + obd[OBD_KEYS.index("long_term_fuel_trim_bank1")]]
    )
    return np.nan_to_num(np.array(vector), nan=0.0, posinf=0.0, neginf=0.0), FEATURE_NAMES


def feature_matrix(labelled_set, cache_path=None) -> np.ndarray:
    """Feature rows for every window of a labelled set. With `cache_path` the matrix is saved on the first call and
    read back later (a file whose shape does not match the set is rebuilt)."""
    shape = (len(labelled_set), len(FEATURE_NAMES))
    if cache_path is not None and Path(cache_path).exists():
        cached = np.load(cache_path)
        if cached.shape == shape:
            return cached
    X = np.array([extract_features(labelled_set.window(i))[0] for i in range(len(labelled_set))])
    if cache_path is not None:
        np.save(cache_path, X)
    return X
