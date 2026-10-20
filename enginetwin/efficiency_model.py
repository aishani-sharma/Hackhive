"""Predict brake thermal efficiency from sensor readings both datasets share (gradient boosting and a small network)."""
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from torch import nn

from enginetwin.efficiency import brake_thermal_efficiency
from enginetwin.features import MAX_PLAUSIBLE_BTE

FEATURES = ("engine_speed", "load", "vehicle_speed", "rpm_per_kmh", "total_fuel_trim")
MIN_SPEED_KMH = 5.0  # floor for the rpm / speed ratio so standstill does not divide by zero
KIA_SOUL_MAX_TORQUE_NM = 157.0  # assumption: peak torque of the 1.6 L Kia Soul in the Ocslab data


def _common(df, load) -> pd.DataFrame:
    trim = df["short_term_fuel_trim_bank1"].fillna(0) + df["long_term_fuel_trim_bank1"].fillna(0)
    return pd.DataFrame(
        {
            "engine_speed": df["engine_speed"],
            "load": load,
            "vehicle_speed": df["vehicle_speed"],
            "rpm_per_kmh": df["engine_speed"] / df["vehicle_speed"].clip(lower=MIN_SPEED_KMH),
            "total_fuel_trim": trim,
        },
        index=df.index,
    )


def ocslab_frame(df) -> pd.DataFrame:
    """Features (and the measured efficiency, when the torque and fuel columns are there) for Ocslab rows."""
    out = _common(df, df["calculated_load_value"])
    if {"engine_torque", "fuel_consumption"} <= set(df.columns):
        out["efficiency"] = brake_thermal_efficiency(df["engine_torque"], df["engine_speed"], df["fuel_consumption"])
    if "driver" in df:
        out["group"] = df["driver"]
    return out


def ved_frame(df, max_torque_nm=KIA_SOUL_MAX_TORQUE_NM) -> pd.DataFrame:
    """Features and an efficiency estimate for VED rows. VED has no torque, so torque is assumed to be the absolute
    load (%) of `max_torque_nm`, and fuel comes from the MAF-based estimate in the loader."""
    out = _common(df, df["absolute_load"])
    torque = df["absolute_load"] / 100.0 * max_torque_nm
    out["efficiency"] = brake_thermal_efficiency(torque, df["engine_speed"], df["fuel_consumption"])
    if "vehid" in df:
        out["group"] = df["vehid"]
    return out


def build_xy(frame):
    """Complete rows only, with the efficiency inside the physically possible range."""
    ok = frame[list(FEATURES) + ["efficiency"]].notna().all(axis=1)
    ok &= (frame["efficiency"] > 0) & (frame["efficiency"] <= MAX_PLAUSIBLE_BTE)
    kept = frame[ok]
    return kept[list(FEATURES)].to_numpy(dtype=float), kept["efficiency"].to_numpy(dtype=float)


def group_split(groups, held_out):
    """Boolean masks (train, test): every row of a held-out group goes to test, never split."""
    test = np.asarray(groups.isin(held_out))
    return ~test, test


def evaluate_regression(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    sse = np.sum((y_true - y_pred) ** 2)
    sst = np.sum((y_true - y_true.mean()) ** 2)
    return {
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "r2": float(1 - sse / sst),
        "correlation": float(np.corrcoef(y_true, y_pred)[0, 1]),
    }


def fit_gbm_efficiency(X, y, seed=0):
    return HistGradientBoostingRegressor(max_iter=200, max_depth=5, learning_rate=0.1, random_state=seed).fit(X, y)


class _AnnRegressor:
    def __init__(self, network, mean, std, y_mean, y_std):
        self.network, self.mean, self.std, self.y_mean, self.y_std = network, mean, std, y_mean, y_std

    def predict(self, X):
        x = torch.tensor((np.asarray(X, dtype=float) - self.mean) / self.std, dtype=torch.float32)
        with torch.no_grad():
            return self.network(x).squeeze(1).numpy().astype(float) * self.y_std + self.y_mean


def fit_ann_efficiency(X, y, seed=0, epochs=60, batch=512):
    """A small network (about 700 weights) on standardised inputs and target, trained in mini-batches."""
    torch.manual_seed(seed)
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    mean, std, y_mean, y_std = X.mean(axis=0), X.std(axis=0) + 1e-9, y.mean(), y.std() + 1e-9
    x = torch.tensor((X - mean) / std, dtype=torch.float32)
    t = torch.tensor((y - y_mean) / y_std, dtype=torch.float32).unsqueeze(1)
    network = nn.Sequential(nn.Linear(X.shape[1], 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))
    optimiser = torch.optim.Adam(network.parameters(), lr=0.01)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        order = torch.randperm(len(x), generator=generator)
        for start in range(0, len(x), batch):
            idx = order[start : start + batch]
            optimiser.zero_grad()
            nn.functional.mse_loss(network(x[idx]), t[idx]).backward()
            optimiser.step()
    return _AnnRegressor(network, mean, std, y_mean, y_std)


def feature_importance(model, X, y, names, seed=0) -> dict:
    """Permutation importance: how much the mean absolute error grows when one feature is shuffled. Largest first."""
    rng = np.random.default_rng(seed)
    base = np.mean(np.abs(model.predict(X) - y))
    gains = {}
    for j, name in enumerate(names):
        shuffled = np.array(X, dtype=float, copy=True)
        shuffled[:, j] = rng.permutation(shuffled[:, j])
        gains[name] = float(np.mean(np.abs(model.predict(shuffled) - y)) - base)
    return dict(sorted(gains.items(), key=lambda kv: -kv[1]))
