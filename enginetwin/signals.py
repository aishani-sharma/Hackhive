import numpy as np


def _amplitude_spectrum(signal, fs):
    """Hann-windowed magnitude spectrum / N: a sine of amplitude A peaks at A/4. Returns (freqs, magnitudes)."""
    spectrum = np.abs(np.fft.rfft(signal * np.hanning(len(signal)))) / len(signal)
    return np.fft.rfftfreq(len(signal), 1 / fs), spectrum


def order_amplitude(signal, fs, rpm, order):
    """Strength of the spectral line at `order` x crank speed (order 2 = firing frequency of a 4-cylinder)."""
    freqs, spectrum = _amplitude_spectrum(signal, fs)
    target = order * rpm / 60
    return spectrum[np.abs(freqs - target) <= 1.0].max()  # +-1 Hz: the line is spread by the window


def order_spectrum(signal, fs, rpm, max_order=8.0):
    """Spectrum on an order axis (frequency / crank frequency), up to `max_order`. Returns (orders, amplitudes)."""
    freqs, spectrum = _amplitude_spectrum(signal, fs)
    orders = freqs / (rpm / 60)
    keep = orders <= max_order
    return orders[keep], spectrum[keep]


def half_order_ratio(signal, fs, rpm):
    """Misfire indicator: the order-0.5 line relative to the firing line (order 2) of a 4-cylinder."""
    return order_amplitude(signal, fs, rpm, 0.5) / order_amplitude(signal, fs, rpm, 2.0)


def bandpass(signal, fs, low, high):
    """Brick-wall FFT band-pass: keep only components between `low` and `high` Hz."""
    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), 1 / fs)
    spectrum[(freqs < low) | (freqs > high)] = 0
    return np.fft.irfft(spectrum, len(signal))


def band_energy_fraction(signal, fs, low, high):
    """Share of the signal's energy that lies between `low` and `high` Hz (0 for a silent signal)."""
    total = np.sum(signal**2)
    if total == 0:
        return 0.0
    return float(np.sum(bandpass(signal, fs, low, high) ** 2) / total)


def spectrogram(signal, fs, win_s, hop_s):
    """Short-time Fourier magnitudes on the same A/4 scale as the other spectra. Returns (freqs, times, magnitude),
    magnitude shaped (len(freqs), n_frames); times are the frame centres in seconds."""
    win, hop = int(round(win_s * fs)), int(round(hop_s * fs))
    starts = np.arange(0, len(signal) - win + 1, hop)
    frames = [_amplitude_spectrum(signal[s : s + win], fs)[1] for s in starts]
    freqs = np.fft.rfftfreq(win, 1 / fs)
    return freqs, (starts + win / 2) / fs, np.array(frames).T


def envelope_spectrum(signal, fs, band=None):
    """Spectrum of the signal's amplitude envelope: how often resonance ringing repeats (valve ticks), not where it
    rings. `band=(low, high)` first isolates that resonance. Returns (freqs, amplitudes)."""
    if band is not None:
        signal = bandpass(signal, fs, *band)
    spectrum = np.fft.fft(signal)
    h = np.zeros(len(signal))  # analytic signal: keep positive frequencies, doubled; DC and Nyquist once
    h[0] = 1
    if len(signal) % 2 == 0:
        h[len(signal) // 2] = 1
        h[1 : len(signal) // 2] = 2
    else:
        h[1 : (len(signal) + 1) // 2] = 2
    envelope = np.abs(np.fft.ifft(spectrum * h))
    return _amplitude_spectrum(envelope - envelope.mean(), fs)


def per_cylinder_energy(signal, fs, rpm):
    """Signal energy in each cylinder's own firing window. Cylinder 1 fires at t=0, then 3, 4, 2 every 180 degrees."""
    samples_per_firing = 30.0 / rpm * fs
    energy = {c: 0.0 for c in (1, 2, 3, 4)}
    for k in range(int(len(signal) / samples_per_firing + 1e-9)):
        start, stop = int(round(k * samples_per_firing)), int(round((k + 1) * samples_per_firing))
        energy[(1, 3, 4, 2)[k % 4]] += float(np.sum(signal[start:stop] ** 2))
    return energy
