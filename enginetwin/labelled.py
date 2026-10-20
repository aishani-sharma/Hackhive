import hashlib
import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
from scipy.spatial import cKDTree

from enginetwin.losses import degraded_efficiency
from enginetwin.schema import FAULTS, Window
from enginetwin.simulator import FS, simulate_window

DATASET_VERSION = 2  # bump when the generator changes, so old cache files are not reused
SEVERITY_RANGE = (0.3, 1.0)  # faulty windows are never so mild that the label is a guess
LOAD_RANGE = (0.2, 0.8)
TRAIN_RPM_RANGE = (800.0, 3000.0)
UNSEEN_RPM_RANGE = (3000.0, 5000.0)
NORMAL_SNR_DB = (10.0, 25.0)  # training noise: drawn per window, wide enough to teach the models to ignore noise
WIDE_RPM_RANGE = (800.0, 5000.0)
WIDE_SNR_DB = (5.0, 25.0)
TEST_SNR_DB = {"test_noise_15db": 15.0, "test_heavy_noise": 5.0, "test_noise_0db": 0.0}  # fixed, never trained on
OBD_NAMES = (
    "engine_speed", "calculated_load_value", "engine_coolant_temperature", "maf", "timing_advance",
    "short_term_fuel_trim_bank1", "long_term_fuel_trim_bank1",
)


class LabelledSet:
    """Windows stored as arrays (sound and vibration in float16 to keep 8 GB laptops comfortable)."""

    _ARRAYS = ("sound", "vibration", "fault", "severity", "efficiency", "rpm", "load", "snr_db", "worn_valves")

    def __init__(self, sound, vibration, obd, fault, severity, efficiency, rpm, load, snr_db, worn_valves, fs=FS):
        self.sound, self.vibration, self.obd = sound, vibration, obd
        self.fault, self.severity, self.efficiency = fault, severity, efficiency
        self.rpm, self.load, self.snr_db, self.worn_valves, self.fs = rpm, load, snr_db, worn_valves, fs

    def __len__(self):
        return len(self.fault)

    def window(self, i) -> Window:
        return Window(
            sound=self.sound[i].astype(np.float64),
            sound_fs=self.fs,
            vibration=self.vibration[i].astype(np.float64),
            vibration_fs=self.fs,
            obd={k: float(v[i]) for k, v in self.obd.items()},
            fault=str(self.fault[i]),
            severity=float(self.severity[i]),
            source="simulated",
        )


class LabelledSplits(NamedTuple):
    train: LabelledSet
    validation: LabelledSet
    validation_wide: LabelledSet  # healthy-threshold calibration only: speeds 800-5000 rpm, noise 5-25 dB
    test_same: LabelledSet  # same conditions as training
    test_unseen_rpm: LabelledSet  # speeds never seen in training
    test_noise_15db: LabelledSet  # training speeds, 15 dB: inside the trained noise range
    test_heavy_noise: LabelledSet  # training speeds, 5 dB: beyond the trained range
    test_noise_0db: LabelledSet  # training speeds, 0 dB: far beyond it


def make_labelled_set(
    n_per_class, rpm_range, snr_db_range, seed, healthy_efficiency, loss_rates=None, duration_s=2.0
) -> LabelledSet:
    """`n_per_class` windows for every fault label. Window i has the same speed, load and noise level in every class
    (they are drawn from the index alone), so the operating point can never tell the classes apart."""
    n_samples = int(duration_s * FS)
    rows = []
    for fault_index, fault in enumerate(FAULTS):
        for i in range(n_per_class):
            point = np.random.default_rng([seed, 0, i])
            rpm = float(point.uniform(*rpm_range))
            load = float(point.uniform(*LOAD_RANGE))
            snr_db = float(point.uniform(*snr_db_range))
            severity = float(point.uniform(*SEVERITY_RANGE))
            cylinder = int(point.integers(1, 5))
            valves = tuple(int(v) for v in point.choice([1, 2, 3, 4], size=int(point.integers(1, 3)), replace=False))
            own = np.random.default_rng([seed, 1, fault_index, i])
            w = simulate_window(
                rpm=rpm,
                fault=fault,
                severity=0.0 if fault == "healthy" else severity,
                seed=int(own.integers(2**31)),
                duration_s=duration_s,
                cylinder=cylinder,
                load=load,
                snr_db=snr_db,
                worn_valves=valves,
            )
            efficiency = degraded_efficiency(healthy_efficiency(rpm, load), fault, w.severity, loss_rates)
            rows.append((w, rpm, load, snr_db, valves, efficiency))
    sound = np.empty((len(rows), n_samples), dtype=np.float16)
    vibration = np.empty_like(sound)
    for k, row in enumerate(rows):
        sound[k], vibration[k] = row[0].sound, row[0].vibration
    return LabelledSet(
        sound=sound,
        vibration=vibration,
        obd={name: np.array([r[0].obd[name] for r in rows]) for name in OBD_NAMES},
        fault=np.array([r[0].fault for r in rows]),
        severity=np.array([r[0].severity for r in rows]),
        efficiency=np.array([r[5] for r in rows]),
        rpm=np.array([r[1] for r in rows]),
        load=np.array([r[2] for r in rows]),
        snr_db=np.array([r[3] for r in rows]),
        worn_valves=np.array([list(r[4]) + [0] * (2 - len(r[4])) for r in rows]),
    )


def make_labelled_splits(n_train, n_val, n_test, seed, healthy_efficiency, loss_rates=None) -> LabelledSplits:
    """Train, validation, a wide validation set for calibrating alarm thresholds, and five test sets: same conditions,
    unseen speeds and three fixed noise levels. Each split has its own seed."""
    common = dict(healthy_efficiency=healthy_efficiency, loss_rates=loss_rates)
    fixed = lambda name, k: make_labelled_set(n_test, TRAIN_RPM_RANGE, (TEST_SNR_DB[name],) * 2, seed + k, **common)
    return LabelledSplits(
        train=make_labelled_set(n_train, TRAIN_RPM_RANGE, NORMAL_SNR_DB, seed, **common),
        validation=make_labelled_set(n_val, TRAIN_RPM_RANGE, NORMAL_SNR_DB, seed + 1, **common),
        validation_wide=make_labelled_set(n_val, WIDE_RPM_RANGE, WIDE_SNR_DB, seed + 7, **common),
        test_same=make_labelled_set(n_test, TRAIN_RPM_RANGE, NORMAL_SNR_DB, seed + 2, **common),
        test_unseen_rpm=make_labelled_set(n_test, UNSEEN_RPM_RANGE, NORMAL_SNR_DB, seed + 3, **common),
        test_noise_15db=fixed("test_noise_15db", 5),
        test_heavy_noise=fixed("test_heavy_noise", 4),
        test_noise_0db=fixed("test_noise_0db", 6),
    )


def _arrays_of(s: LabelledSet, prefix="") -> dict:
    arrays = {prefix + name: getattr(s, name) for name in LabelledSet._ARRAYS}
    arrays.update({f"{prefix}obd_{k}": v for k, v in s.obd.items()})
    return arrays


def _set_from(f, prefix="") -> LabelledSet:
    obd = {k[len(prefix) + 4 :]: f[k] for k in f.files if k.startswith(prefix + "obd_")}
    return LabelledSet(obd=obd, fs=int(f["fs"]), **{a: f[prefix + a] for a in LabelledSet._ARRAYS})


def save_labelled_set(s: LabelledSet, path) -> None:
    np.savez_compressed(path, fs=s.fs, **_arrays_of(s))


def load_labelled_set(path) -> LabelledSet:
    with np.load(path, allow_pickle=False) as f:
        return _set_from(f)


def cache_key(n_train, n_val, n_test, seed, healthy_efficiency, loss_rates=None) -> str:
    """Names the cache entry: it changes with the sizes, seed, healthy surface, loss rates and every SNR/rpm range."""
    probe = [float(healthy_efficiency(r, l)) for r in (1000, 2000, 3000, 4000) for l in (0.2, 0.5, 0.8)]
    ranges = [NORMAL_SNR_DB, WIDE_SNR_DB, WIDE_RPM_RANGE, TRAIN_RPM_RANGE, UNSEEN_RPM_RANGE, TEST_SNR_DB]
    text = json.dumps([DATASET_VERSION, n_train, n_val, n_test, seed, probe, loss_rates, ranges], sort_keys=True)
    return hashlib.sha1(text.encode()).hexdigest()[:12]


def cached_labelled_splits(cache_dir, n_train, n_val, n_test, seed, healthy_efficiency, loss_rates=None) -> LabelledSplits:
    """Build the five splits once; later calls with the same parameters read them back from `cache_dir`."""
    key = cache_key(n_train, n_val, n_test, seed, healthy_efficiency, loss_rates)
    path = Path(cache_dir) / f"labelled_{key}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as f:
            return LabelledSplits(*[_set_from(f, f"{name}.") for name in LabelledSplits._fields])
    splits = make_labelled_splits(n_train, n_val, n_test, seed, healthy_efficiency, loss_rates)
    arrays = {"fs": splits.train.fs}
    for name, s in zip(LabelledSplits._fields, splits):
        arrays.update(_arrays_of(s, f"{name}."))
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return splits


def healthy_efficiency_surface(cells, max_torque_nm):
    """A function (rpm, load) -> healthy efficiency, read from the nearest cell of a real efficiency map
    (load is the share of `max_torque_nm`). Axes are scaled by the map's own ranges before measuring distance."""
    rpm_span = max(cells["rpm"].max() - cells["rpm"].min(), 1.0)
    torque_span = max(cells["torque"].max() - cells["torque"].min(), 1.0)
    tree = cKDTree(np.column_stack([cells["rpm"] / rpm_span, cells["torque"] / torque_span]))
    efficiency = cells["efficiency"].to_numpy()

    def surface(rpm, load):
        _, index = tree.query([rpm / rpm_span, load * max_torque_nm / torque_span])
        return float(efficiency[index])

    return surface
