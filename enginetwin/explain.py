"""Plain-language reasons for a diagnosis: which readings moved away from a healthy engine, and by how much."""
from dataclasses import dataclass

import numpy as np

from enginetwin.fault_features import FEATURE_NAMES

MIN_Z = 3.0  # a reading is cited only when it is this many healthy standard deviations away, in the telling direction


@dataclass
class Baseline:
    mean: np.ndarray
    std: np.ndarray


def healthy_baseline(X_healthy) -> Baseline:
    X = np.asarray(X_healthy, dtype=float)
    return Baseline(mean=X.mean(axis=0), std=X.std(axis=0) + 1e-12)


def _times(value, normal):
    return value / normal if normal else np.inf


# feature -> (direction that points to the fault, wording given the value, the normal value and the channel)
_CHANNEL = lambda name: "vibration" if name.startswith("vibration") else "sound"
_ONCE = {"cyl_min_ratio", "knock_env_firing", "valve_env_cycle"}
_RULES = {
    "misfire": {
        "half_order_ratio": (+1, lambda c, v, n: f"half-order {c} {_times(v, n):.0f} times normal"),
        "cyl_min_ratio": (-1, lambda c, v, n: f"weakest cylinder carries only {v * 100:.0f}% of the average firing energy (normal {n * 100:.0f}%)"),
        "obd_total_fuel_trim": (+1, lambda c, v, n: f"fuel trim up {v - n:.0f}%"),
    },
    "knock": {
        "knock_vs_floor": (+1, lambda c, v, n: f"{c} energy at 6-7 kHz {10 ** (v - n):.0f} times the background"),
        "knock_env_firing": (+1, lambda c, v, n: f"knock bursts after every firing {_times(v, n):.0f} times normal"),
        "obd_timing_advance": (-1, lambda c, v, n: f"ignition timing pulled back {n - v:.0f} degrees"),
    },
    "valve_wear": {
        "valve_vs_floor": (+1, lambda c, v, n: f"{c} ticks at 2.5-3.5 kHz {10 ** (v - n):.0f} times the background"),
        "valve_env_cycle": (+1, lambda c, v, n: f"valve tick rhythm once per two revolutions {_times(v, n):.0f} times normal"),
    },
}


def reasons(vector, label, baseline: Baseline, top=3) -> list:
    """Up to `top` sentences, strongest change first, for the faults named in `label`."""
    if label == "healthy":
        return ["all readings are within their normal range"]
    found = {}
    for fault in label.split("+"):
        for name, index in ((n, i) for i, n in enumerate(FEATURE_NAMES)):
            for key, (direction, wording) in _RULES[fault].items():
                if name == key or name.endswith("_" + key):
                    z = direction * (vector[index] - baseline.mean[index]) / baseline.std[index]
                    if z >= MIN_Z:
                        text = wording(_CHANNEL(name), vector[index], baseline.mean[index])
                        slot = (fault, key) if key in _ONCE else (fault, name)  # sentences that name no channel appear once
                        found[slot] = max(found.get(slot, (0, "")), (z, text))
    ranked = sorted(found.values(), key=lambda item: -item[0])
    return [text for _, text in ranked[:top]] or ["no single reading stands out strongly"]
