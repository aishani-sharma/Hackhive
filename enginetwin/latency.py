"""Alert latency: how long after a fault starts do the models raise a confirmed alert on a live 0.5 s chunk stream."""
import time
from dataclasses import dataclass

import numpy as np

from enginetwin.fault_features import extract_features
from enginetwin.schema import Window
from enginetwin.simulator import simulate_window
from enginetwin.spectrograms import standardise_spectrogram, window_spectrogram


@dataclass
class Stream:
    """Continuous sound and vibration in which a fault starts at `onset_s`."""

    sound: np.ndarray
    vibration: np.ndarray
    fs: int
    onset_s: float
    obd_before: dict
    obd_after: dict
    fault: str
    severity: float

    def window(self, end_s, window_s) -> Window:
        """The latest `window_s` seconds up to `end_s`, with the OBD readings and label of that moment."""
        stop = int(round(end_s * self.fs))
        start = stop - int(round(window_s * self.fs))
        faulty = end_s > self.onset_s
        return Window(
            sound=self.sound[start:stop],
            sound_fs=self.fs,
            vibration=self.vibration[start:stop],
            vibration_fs=self.fs,
            obd=self.obd_after if faulty else self.obd_before,
            fault=self.fault if faulty else "healthy",
            severity=self.severity if faulty else 0.0,
            source="simulated",
        )


def simulate_onset(fault, severity, onset_s, duration_s, rpm, seed, load=0.4, snr_db=20.0, cylinder=2, worn_valves=(1,)) -> Stream:
    """A healthy engine that develops `fault` at `onset_s`. Both halves come from the same seed, so firings and noise
    carry on without a jump at the splice."""
    common = dict(rpm=rpm, seed=seed, duration_s=duration_s, load=load, snr_db=snr_db, cylinder=cylinder, worn_valves=worn_valves)
    healthy = simulate_window(**common)
    faulty = simulate_window(fault=fault, severity=severity, **common)
    cut = int(round(onset_s * healthy.sound_fs))
    return Stream(
        sound=np.concatenate([healthy.sound[:cut], faulty.sound[cut:]]),
        vibration=np.concatenate([healthy.vibration[:cut], faulty.vibration[cut:]]),
        fs=healthy.sound_fs,
        onset_s=onset_s,
        obd_before=healthy.obd,
        obd_after=faulty.obd,
        fault=fault,
        severity=severity,
    )


def alert_latency(predict_fn, stream, chunk_s=0.5, window_s=2.0, confirm=2) -> dict:
    """Run `predict_fn(window) -> label` every `chunk_s` on the latest `window_s` of the stream. An alert is confirmed
    after `confirm` non-healthy results in a row once the fault has started. Latency is the end of the confirming chunk
    minus the onset, plus the time the model took to answer that chunk. `latency_s` is None if no alert was confirmed.
    `false_alarm_rate` is the share of pre-onset windows flagged."""
    duration = len(stream.sound) / stream.fs
    ends = [window_s + k * chunk_s for k in range(int(round((duration - window_s) / chunk_s)) + 1)]
    before = [e for e in ends if e <= stream.onset_s + 1e-9]
    after = [e for e in ends if e > stream.onset_s + 1e-9]
    flagged_before = [predict_fn(stream.window(e, window_s)) != "healthy" for e in before]
    streak = 0
    for end in after:
        started = time.perf_counter()
        label = predict_fn(stream.window(end, window_s))
        compute = time.perf_counter() - started
        streak = streak + 1 if label != "healthy" else 0
        if streak >= confirm:
            return {
                "detected_at_s": end,
                "compute_s": compute,
                "latency_s": end - stream.onset_s + compute,
                "false_alarm_rate": float(np.mean(flagged_before)) if flagged_before else 0.0,
            }
    return {
        "detected_at_s": None,
        "compute_s": None,
        "latency_s": None,
        "false_alarm_rate": float(np.mean(flagged_before)) if flagged_before else 0.0,
    }


def predict_with_features(model):
    """Adaptor for the SVM and softmax models: window -> label from the 40 hand-made features."""
    return lambda window: str(model.predict(extract_features(window)[0][None])[0])


def predict_with_obd(model):
    """Adaptor for the OBD network: it reads the last eight features, which are the OBD values."""
    return lambda window: str(model.predict(extract_features(window)[0][-8:][None])[0])


def predict_with_cnn(model):
    from enginetwin.cnn import labels_from_proba

    def predict(window):
        obd = extract_features(window)[0][-8:][None]
        return str(labels_from_proba(model.predict_proba(standardise_spectrogram(window_spectrogram(window))[None], obd))[0])

    return predict
