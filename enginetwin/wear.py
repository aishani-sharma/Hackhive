from typing import NamedTuple

import numpy as np

from enginetwin.simulator import simulate_window

WEAR_EXPONENT = 1.5  # assumed: wear accelerates, so the first weeks change little and the last weeks change a lot
DAILY_JITTER = 0.01  # day-to-day scatter in how worn the valve looks (measurement variation, not recovery)


class WearSeries(NamedTuple):
    days: np.ndarray
    windows: list


def wear_series(n_days, seed, rpm=2000.0, load=0.4, worn_valves=(1,), duration_s=2.0) -> WearSeries:
    """One window per day while one valve wears from new (severity 0) to fully worn (1). The engine is held at one
    operating point so the only thing that changes from day to day is the wear. Severity never falls back."""
    rng = np.random.default_rng(seed)
    days = np.arange(n_days)
    trend = (days / (n_days - 1)) ** WEAR_EXPONENT
    severity = np.clip(np.maximum.accumulate(trend + DAILY_JITTER * rng.standard_normal(n_days)), 0.0, 1.0)
    severity[0], severity[-1] = 0.0, 1.0
    windows = [
        simulate_window(
            rpm=rpm,
            fault="healthy" if s == 0.0 else "valve_wear",
            severity=float(s),
            seed=int(rng.integers(2**31)),
            duration_s=duration_s,
            load=load,
            worn_valves=worn_valves,
        )
        for s in severity
    ]
    return WearSeries(days=days, windows=windows)
