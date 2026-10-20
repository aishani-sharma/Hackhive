"""Noise level of a recording, and alarm levels that depend on it."""
from dataclasses import dataclass

import numpy as np

from enginetwin.schema import Window

FLOOR_BAND = (7500.0, 8000.0)  # above every simulated fault resonance, so it holds only background noise
HANN_POWER = 0.375  # mean of hanning(N)^2


def _channel_noise_db(signal, fs) -> float:
    spectrum = np.fft.rfft(signal * np.hanning(len(signal)))
    freqs = np.fft.rfftfreq(len(signal), 1 / fs)
    band = (freqs >= FLOOR_BAND[0]) & (freqs <= min(FLOOR_BAND[1], fs / 2))
    noise_power = np.mean(np.abs(spectrum[band]) ** 2) / (HANN_POWER * len(signal))  # white noise: E|X|^2 = sigma^2 x sum(w^2)
    signal_power = max(np.mean(signal**2) - noise_power, 0.0)
    return float(10 * np.log10((noise_power + 1e-20) / (signal_power + 1e-20)))


def estimate_noise_db(window: Window) -> float:
    """Noise-to-signal ratio in dB (minus the SNR; higher is noisier), averaged over sound and vibration. The noise
    power is read from the 7.5-8 kHz floor and assumed white; the rest of the signal's power counts as signal."""
    return (_channel_noise_db(window.sound, window.sound_fs) + _channel_noise_db(window.vibration, window.vibration_fs)) / 2


def noise_bins(noise_db, edges) -> np.ndarray:
    """Bin number of each noise level: 0 below the first edge, `len(edges)` at or above the last."""
    return np.digitize(np.asarray(noise_db, dtype=float), edges)


@dataclass
class NoiseThresholds:
    edges: tuple
    thresholds: np.ndarray  # one alarm level per bin

    def threshold_for(self, noise_db):
        """The alarm level for a noise level (or an array of them)."""
        level = self.thresholds[noise_bins(noise_db, self.edges)]
        return float(level) if np.ndim(noise_db) == 0 else level


def fit_noise_thresholds(scores, noise_db, labels, edges, q=0.99, min_windows=10) -> NoiseThresholds:
    """Alarm level per noise bin: the `q` quantile of healthy scores in it. A bin with fewer than `min_windows` healthy
    windows borrows from the nearest calibrated bin (the noisier one on a tie)."""
    scores, labels = np.asarray(scores, dtype=float), np.asarray(labels)
    bins = noise_bins(noise_db, edges)
    n_bins = len(edges) + 1
    levels = np.full(n_bins, np.nan)
    for b in range(n_bins):
        healthy = scores[(bins == b) & (labels == "healthy")]
        if len(healthy) >= min_windows:
            levels[b] = np.quantile(healthy, q)
    calibrated = np.flatnonzero(~np.isnan(levels))
    if len(calibrated) == 0:
        raise ValueError("no noise bin has enough healthy windows to calibrate")
    for b in range(n_bins):
        if np.isnan(levels[b]):
            distance = np.abs(calibrated - b)
            nearest = calibrated[distance == distance.min()].max()  # the noisier of equally near bins
            levels[b] = levels[nearest]
    return NoiseThresholds(edges=tuple(edges), thresholds=levels)
