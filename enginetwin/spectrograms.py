"""Fixed-size log spectrogram images of the sound and vibration, the input of the CNN."""
from pathlib import Path

import numpy as np

from enginetwin.schema import Window
from enginetwin.signals import spectrogram

BANDS = 64  # 125 Hz-wide bands from 0 to 8 kHz
FRAMES = 64
WINDOW_SAMPLES = 512  # 32 ms at 16 kHz
HOP_SAMPLES = 496  # gives exactly 64 frames over a 2 s window
FLOOR = 1e-5  # added before the log so silence is a finite number


def _channel(signal, fs) -> np.ndarray:
    _, _, magnitude = spectrogram(signal, fs, WINDOW_SAMPLES / fs, HOP_SAMPLES / fs)
    magnitude = magnitude[: BANDS * 4].reshape(BANDS, 4, -1).mean(axis=1)  # 4 FFT bins of 31.25 Hz per band
    return np.log10(magnitude[:, :FRAMES] + FLOOR)


def window_spectrogram(window: Window) -> np.ndarray:
    """(2, 64, 64): log10 magnitude of the sound then the vibration, as (channel, frequency band, time frame)."""
    return np.stack([_channel(window.sound, window.sound_fs), _channel(window.vibration, window.vibration_fs)])


def standardise_spectrogram(spec) -> np.ndarray:
    """Each channel of one (2, 64, 64) spectrogram to zero mean and unit spread. Overall level and broadband noise shift
    every log bin together, so this keeps the pattern of the event and drops the level it sits at."""
    spec = np.asarray(spec, dtype=float)
    mean = spec.mean(axis=(1, 2), keepdims=True)
    std = spec.std(axis=(1, 2), keepdims=True)
    return np.where(std > 1e-9, (spec - mean) / np.where(std > 1e-9, std, 1.0), 0.0)


def spectrogram_stack(labelled_set, cache_path=None) -> np.ndarray:
    """Standardised spectrograms for every window of a labelled set, as float16 (N, 2, 64, 64), cached when `cache_path` is given."""
    shape = (len(labelled_set), 2, BANDS, FRAMES)
    if cache_path is not None and Path(cache_path).exists():
        cached = np.load(cache_path)
        if cached.shape == shape:
            return cached
    stack = np.empty(shape, dtype=np.float16)
    for i in range(len(labelled_set)):
        stack[i] = standardise_spectrogram(window_spectrogram(labelled_set.window(i)))
    if cache_path is not None:
        np.save(cache_path, stack)
    return stack
