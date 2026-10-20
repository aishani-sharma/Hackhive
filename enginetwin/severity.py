"""How bad is each fault: one small regressor per fault type on the hand-made features."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

FAULT_TYPES = ("misfire", "knock", "valve_wear")


def fit_severity_models(X, labels, severity, seed=0) -> dict:
    """fault type -> regressor. Windows without that fault count as severity 0, combined labels as the shared severity."""
    X, severity = np.asarray(X, dtype=float), np.asarray(severity, dtype=float)
    models = {}
    for fault in FAULT_TYPES:
        target = np.array([s if fault in str(label).split("+") else 0.0 for label, s in zip(labels, severity)])
        models[fault] = HistGradientBoostingRegressor(max_iter=150, max_depth=3, learning_rate=0.1, random_state=seed).fit(X, target)
    return models


def estimate_severity(models, x, fault) -> float:
    """Estimated severity (0 to 1) of `fault` in one feature vector."""
    if fault not in models:
        raise ValueError(f"unknown fault {fault!r}; expected one of {sorted(models)}")
    return float(np.clip(models[fault].predict(np.asarray(x, dtype=float)[None])[0], 0.0, 1.0))
