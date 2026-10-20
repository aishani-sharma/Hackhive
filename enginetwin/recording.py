"""Turn an uploaded recording (and optional OBD log) into a window the models can read."""
import io
from math import gcd

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly

from enginetwin.fault_features import OBD_KEYS, refine_rpm
from enginetwin.schema import Window

FS = 16000
WINDOW_S = 2.0
MIN_SECONDS = 0.25
# Used when no OBD log comes with the recording: values typical of a warm, healthy engine in the simulator's calibration.
NEUTRAL_OBD = {
    "engine_speed": 2000.0, "calculated_load_value": 40.0, "engine_coolant_temperature": 85.0, "maf": 10.0,
    "timing_advance": 30.0, "short_term_fuel_trim_bank1": 0.0, "long_term_fuel_trim_bank1": 2.5,
}
MISSING_FLAG = "_obd_missing"
FIRING_HZ_RANGE = (20.0, 200.0)  # 600 to 6000 rpm
HARMONIC_LIMIT_HZ = 1500.0


def has_obd(window: Window) -> bool:
    return MISSING_FLAG not in window.obd


def obd_from_csv(source) -> dict:
    """Median of each OBD reading found in the CSV (column names as in the project's loaders); absent ones stay neutral."""
    frame = pd.read_csv(source)
    obd = dict(NEUTRAL_OBD)
    for key in OBD_KEYS:
        if key in frame and frame[key].notna().any():
            obd[key] = float(frame[key].median())
    return obd


def estimate_rpm_from_audio(sound, fs) -> float:
    """Engine speed from the sound alone: the firing frequency whose harmonic comb (lines at 1x, 2x, 3x ... up to 1.5 kHz)
    carries the most energy, preferring the highest candidate that reaches 60% of the best (a lower one also fits, with empty
    slots). Strong misfire can still be read at half the true speed, so an OBD log is more reliable."""
    spectrum = np.abs(np.fft.rfft(sound * np.hanning(len(sound)))) / len(sound)
    freqs = np.fft.rfftfreq(len(sound), 1 / fs)
    step = freqs[1]
    candidates = np.arange(FIRING_HZ_RANGE[0], FIRING_HZ_RANGE[1], 0.5)
    totals = np.zeros(len(candidates))
    for i, f0 in enumerate(candidates):
        for k in range(1, int(HARMONIC_LIMIT_HZ / f0) + 1):
            centre = int(round(k * f0 / step))
            reach = int(np.ceil((0.003 * k * f0 + 1.0) / step))  # tolerance grows with the harmonic number
            totals[i] += spectrum[max(centre - reach, 0) : centre + reach + 1].max()
    best = candidates[np.flatnonzero(totals >= 0.6 * totals.max()).max()]
    return refine_rpm(sound, fs, rpm_obd=30.0 * best)


def _load(source):
    data, fs = sf.read(io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else source, always_2d=True)
    mono = data.mean(axis=1)
    if fs != FS:
        g = gcd(int(fs), FS)
        mono = resample_poly(mono, FS // g, int(fs) // g)
    return mono


def window_from_wav(source, obd=None) -> Window:
    """A 2 s, 16 kHz, mono window tagged "real" from a WAV (bytes or path). A longer recording gives its loudest 2 s, a
    shorter one is padded with silence. One microphone stands in for both channels. The fault label is a placeholder:
    nothing is known about a recording until it is diagnosed. Without `obd` the speed comes from the audio."""
    mono = _load(source)
    n = int(WINDOW_S * FS)
    if len(mono) < MIN_SECONDS * FS:
        raise ValueError(f"recording is too short: {len(mono) / FS:.2f} s, need at least {MIN_SECONDS} s")
    if len(mono) <= n:
        mono = np.pad(mono, (0, n - len(mono)))
    else:
        hop = FS // 10
        starts = np.arange(0, len(mono) - n + 1, hop)
        energy = [np.mean(mono[s : s + n] ** 2) for s in starts]
        mono = mono[starts[int(np.argmax(energy))] : starts[int(np.argmax(energy))] + n]
    if obd is None:
        obd = {**NEUTRAL_OBD, "engine_speed": estimate_rpm_from_audio(mono, FS), MISSING_FLAG: 1.0}
    return Window(sound=mono, sound_fs=FS, vibration=mono.copy(), vibration_fs=FS, obd=dict(obd), fault="healthy", severity=0.0, source="real")
