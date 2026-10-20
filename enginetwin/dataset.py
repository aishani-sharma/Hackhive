from typing import NamedTuple

import numpy as np

from enginetwin.schema import FAULTS, Window
from enginetwin.simulator import NOISE_SIGMA, simulate_window

SEVERITY_RANGE = (0.3, 1.0)  # faulty windows are never so mild that the label is a guess
LOAD_RANGE = (0.3, 0.6)


class Splits(NamedTuple):
    train: list
    test_unseen_rpm: list  # normal noise, speeds never seen in training
    test_noisy: list  # training speeds, much more background noise


def make_dataset(n_per_class, rpms, seed, noise_sigma=NOISE_SIGMA, duration_s=2.0) -> list:
    """`n_per_class` simulated windows for every fault label, cycling through `rpms`. Misfiring cylinder, severity,
    load and the window's own noise seed are drawn from `seed`."""
    rng = np.random.default_rng(seed)
    windows = []
    for fault in FAULTS:
        for i in range(n_per_class):
            windows.append(
                simulate_window(
                    rpm=rpms[i % len(rpms)],
                    fault=fault,
                    severity=0.0 if fault == "healthy" else float(rng.uniform(*SEVERITY_RANGE)),
                    seed=int(rng.integers(2**31)),
                    duration_s=duration_s,
                    cylinder=int(rng.integers(1, 5)),
                    load=float(rng.uniform(*LOAD_RANGE)),
                    noise_sigma=noise_sigma,
                )
            )
    return windows


def make_splits(train_rpms, test_rpms, n_train, n_test, seed, noisy_sigma=0.05, duration_s=2.0) -> Splits:
    """A train set, a test set at held-out speeds and a noisy test set at the training speeds. Each split uses its own seed."""
    if set(train_rpms) & set(test_rpms):
        raise ValueError("train and unseen-test rpms overlap")
    return Splits(
        train=make_dataset(n_train, train_rpms, seed, duration_s=duration_s),
        test_unseen_rpm=make_dataset(n_test, test_rpms, seed + 1, duration_s=duration_s),
        test_noisy=make_dataset(n_test, train_rpms, seed + 2, noise_sigma=noisy_sigma, duration_s=duration_s),
    )
