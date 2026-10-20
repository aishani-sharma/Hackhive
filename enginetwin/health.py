"""Valve health: wear estimated from the vibration, a 100-to-0 health index and a time-to-service forecast."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from enginetwin.fault_features import FEATURE_NAMES

VALVE_FEATURES = [
    f"{channel}_{name}" for channel in ("vibration", "sound") for name in ("valve_band_fraction", "valve_env_cycle", "valve_env_firing")
]
_COLUMNS = [FEATURE_NAMES.index(n) for n in VALVE_FEATURES]


class SeverityRegressor:
    def __init__(self, model):
        self.model = model

    def predict(self, X) -> np.ndarray:
        """Estimated valve wear between 0 (new) and 1 (fully worn) from the full 40-feature rows."""
        logs = np.log10(np.asarray(X, dtype=float)[:, _COLUMNS] + 1e-6)
        return np.clip(self.model.predict(logs), 0.0, 1.0)


def fit_severity_regressor(X, severity, seed=0) -> SeverityRegressor:
    """Learn wear from the valve-related features only (a window without valve wear has severity 0)."""
    logs = np.log10(np.asarray(X, dtype=float)[:, _COLUMNS] + 1e-6)
    model = HistGradientBoostingRegressor(max_iter=150, max_depth=3, learning_rate=0.1, random_state=seed)
    return SeverityRegressor(model.fit(logs, np.asarray(severity, dtype=float)))


def health_index(severity):
    """100 for a new valve down to 0 for a fully worn one."""
    return 100.0 * (1.0 - np.asarray(severity, dtype=float)) if np.ndim(severity) else 100.0 * (1.0 - severity)


def time_to_service(days, health, threshold=20.0, last=14) -> float:
    """Days from the last day until the recent trend (a line through the last `last` days) reaches `threshold`.
    0 when it is already there, infinity when health is not falling."""
    days, health = np.asarray(days, dtype=float)[-last:], np.asarray(health, dtype=float)[-last:]
    slope, intercept = np.polyfit(days, health, 1)
    now = intercept + slope * days[-1]
    if now <= threshold:
        return 0.0
    if slope >= -1e-9:
        return float(np.inf)
    return float((now - threshold) / -slope)
