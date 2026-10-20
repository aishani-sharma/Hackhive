"""Data loading and model helpers for the engine diagnostics workbench."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupShuffleSplit, train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "bosch"

PERFORMANCE_CONTROLS = ["engine_speed", "engine_load", "intake_valve_opening", "air_fuel_ratio"]
PERFORMANCE_TARGETS = [
    "specific_fuel_consumption", "temperature_exhaust_manifold", "temperature_in_catalyst",
    "engine_roughness_s", "HC", "NOx",
]
GENGINE1_COLUMNS = [
    "speed", "load", "lambda", "ignition_angle", "fuel_cutoff", "particle_numbers",
    "CO", "CO2", "HC", "NOx", "O2", "exhaust_manifold_temperature", "catalyst_temperature",
]
GENGINE2_COLUMNS = [
    "speed", "load", "lambda", "ignition_angle", "particle_numbers", "HC", "NOx",
    "exhaust_manifold_temperature", "catalyst_temperature",
]


def load_performance_data(engine: str = "engine1") -> pd.DataFrame:
    """Load Bosch's normalized operating-map measurements (not SI-unit values)."""
    path = DATA / f"{engine}_normalized.xlsx"
    if not path.exists():
        raise FileNotFoundError(f"Missing Bosch dataset: {path}")
    frame = pd.read_excel(path, sheet_name="data")
    frame = frame.dropna(how="all").dropna(axis=1, how="all")
    frame = frame.loc[:, [c for c in frame.columns if not str(c).startswith("Unnamed:")]]
    frame = frame.dropna(subset=PERFORMANCE_CONTROLS + ["specific_fuel_consumption"])
    return frame.reset_index(drop=True)


def load_safety_bounds(engine: str = "engine1") -> dict[str, float]:
    """Read Bosch's suggested upper bounds on the standardized response scale."""
    path = DATA / "safety_constraint.xlsx"
    sheet = f"{engine}_normalized"
    frame = pd.read_excel(path, sheet_name=sheet).dropna(how="all").dropna(axis=1, how="all")
    frame = frame.set_index("Unnamed: 0")
    if "upper_bound" not in frame.index:
        return {}
    return {str(col): float(value) for col, value in frame.loc["upper_bound"].items() if pd.notna(value)}


def load_gengine_trajectory(path: str | Path | None = None, engine: int = 1) -> pd.DataFrame:
    """Read one 10 Hz Bosch trajectory. Values are standardized by the source."""
    if path is None:
        path = DATA / ("gengine1_trajectory_10.csv" if engine == 1 else "gengine2_trajectory_00.csv")
    frame = pd.read_csv(path, header=None)
    names = GENGINE1_COLUMNS if frame.shape[1] == 13 else GENGINE2_COLUMNS
    if frame.shape[1] != len(names):
        names = [f"channel_{i + 1}" for i in range(frame.shape[1])]
    frame.columns = names
    frame.insert(0, "time_s", np.arange(len(frame), dtype=float) / 10.0)
    return frame


def select_features(
    frame: pd.DataFrame,
    controls: list[str],
    target: str,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Rank controls with held-out permutation importance; preserve full fit scores."""
    clean = frame[controls + [target]].replace([np.inf, -np.inf], np.nan).dropna()
    X, y = clean[controls], clean[target]
    if len(clean) < 30:
        raise ValueError("At least 30 complete measured rows are needed for a stable operating-map fit.")
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state
    )
    ranker = RandomForestRegressor(
        n_estimators=350, min_samples_leaf=3, max_features=0.9,
        random_state=random_state, n_jobs=-1,
    )
    ranker.fit(x_train, y_train)
    perm = permutation_importance(
        ranker, x_test, y_test, n_repeats=20, random_state=random_state,
        scoring="neg_mean_absolute_error", n_jobs=-1,
    )
    importance = pd.Series(perm.importances_mean, index=controls).sort_values(ascending=False)
    uncertainty = pd.Series(perm.importances_std / np.sqrt(perm.importances.shape[1]), index=controls)
    selected = [c for c in controls if importance[c] > 1.96 * uncertainty[c]]
    if not selected:
        selected = [str(importance.index[0])]
    model = RandomForestRegressor(
        n_estimators=350, min_samples_leaf=3, max_features=0.9,
        random_state=random_state + 1, n_jobs=-1,
    )
    model.fit(x_train[selected], y_train)
    scores = {
        "mae": float(np.mean(np.abs(model.predict(x_test[selected]) - y_test))),
        "r2": float(model.score(x_test[selected], y_test)),
        "importance": importance,
        "selected_controls": selected,
        "model": model,
        "X": X[selected],
        "y": y,
        "validation_rows": len(y_test),
    }
    return X, y, scores


def propose_operating_points(
    frame: pd.DataFrame,
    controls: list[str],
    target: str,
    limits: dict[str, float],
    top_n: int = 80,
) -> tuple[pd.DataFrame, dict]:
    """Rank observed response rows; filter by bounds without optimizing predictions."""
    metrics = list(dict.fromkeys([target, *limits.keys()]))
    candidates = frame[controls + metrics].replace([np.inf, -np.inf], np.nan).dropna()
    candidates = candidates.groupby(controls, as_index=False)[metrics].mean()
    candidates["objective"] = candidates[target]
    feasible = pd.Series(True, index=candidates.index)
    for metric, limit in limits.items():
        feasible &= candidates[metric] <= limit
    candidates["feasible"] = feasible
    chosen = candidates.loc[feasible].sort_values("objective").head(top_n)
    if chosen.empty:
        chosen = candidates.sort_values("objective").head(top_n)
    return chosen.reset_index(drop=True), fit


def build_rul_features(
    uploaded: pd.DataFrame,
    engine_col: str,
    time_col: str,
    target_col: str | None = None,
    failure_col: str | None = None,
    window: int = 8,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str]]:
    """Build causal trailing signal features and engine-level RUL labels.

    If RUL is not supplied, failure_col must mark the final failure event of
    complete run-to-failure histories. RUL is derived from event time per engine.
    """
    df = uploaded.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df[engine_col] = df[engine_col].astype(str)
    df = df.sort_values([engine_col, time_col]).reset_index(drop=True)
    excluded = {engine_col, time_col}
    if target_col:
        excluded.add(target_col)
    if failure_col:
        excluded.add(failure_col)
    sensors = [c for c in df.select_dtypes(include=np.number).columns if c not in excluded]
    if not sensors:
        raise ValueError("No numeric sensor channels found after excluding IDs, time, and targets.")
    if target_col:
        y = pd.to_numeric(df[target_col], errors="coerce")
        keep = y.notna()
    elif failure_col:
        event = pd.to_numeric(df[failure_col], errors="coerce").fillna(0).astype(bool)
        end_time = df.loc[event].groupby(engine_col)[time_col].max()
        complete_ids = set(end_time.index)
        keep = df[engine_col].isin(complete_ids)
        y = df[engine_col].map(end_time) - df[time_col]
    else:
        raise ValueError("Provide a remaining-useful-life target or a final failure-event column.")

    pieces = []
    feature_names = []
    for sensor in sensors:
        g = df.groupby(engine_col, sort=False)[sensor]
        current = pd.to_numeric(df[sensor], errors="coerce")
        mean = g.transform(lambda s: s.rolling(window, min_periods=1).mean())
        std = g.transform(lambda s: s.rolling(window, min_periods=2).std()).fillna(0)
        lag = g.shift(window - 1)
        slope = (current - lag) / max(window - 1, 1)
        names = [sensor, f"{sensor}__mean{window}", f"{sensor}__std{window}", f"{sensor}__slope{window}"]
        pieces.extend([current.rename(names[0]), mean.rename(names[1]), std.rename(names[2]), slope.rename(names[3])])
        feature_names.extend(names)
    X = pd.concat(pieces, axis=1).replace([np.inf, -np.inf], np.nan)
    keep &= X.notna().all(axis=1) & y.notna()
    return X.loc[keep].reset_index(drop=True), y.loc[keep].reset_index(drop=True), df.loc[keep, engine_col].reset_index(drop=True), feature_names


def fit_rul_model(X: pd.DataFrame, y: pd.Series, groups: pd.Series) -> dict:
    """Fit an engine-held-out RUL regressor and rank features on held-out engines."""
    if groups.nunique() < 3:
        raise ValueError("At least 3 complete engine histories are needed for an engine-held-out evaluation.")
    split = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(split.split(X, y, groups))
    inner_split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=43)
    inner_train, inner_val = next(inner_split.split(X.iloc[train_idx], y.iloc[train_idx], groups.iloc[train_idx]))
    ranker = RandomForestRegressor(n_estimators=350, min_samples_leaf=3, max_features=0.7, random_state=42, n_jobs=-1)
    ranker.fit(X.iloc[train_idx].iloc[inner_train], y.iloc[train_idx].iloc[inner_train])
    selection_perm = permutation_importance(
        ranker, X.iloc[train_idx].iloc[inner_val], y.iloc[train_idx].iloc[inner_val], n_repeats=10,
        random_state=43, scoring="neg_mean_absolute_error", n_jobs=-1,
    )
    selection_importance = pd.Series(selection_perm.importances_mean, index=X.columns).sort_values(ascending=False)
    selection_error = pd.Series(selection_perm.importances_std / np.sqrt(selection_perm.importances.shape[1]), index=X.columns)
    selected = [c for c in X.columns if selection_importance[c] > 1.96 * selection_error[c]]
    if not selected:
        selected = [str(selection_importance.index[0])]
    model = RandomForestRegressor(n_estimators=350, min_samples_leaf=3, max_features=0.7, random_state=44, n_jobs=-1)
    model.fit(X.iloc[train_idx][selected], y.iloc[train_idx])
    pred = model.predict(X.iloc[test_idx][selected])
    heldout_perm = permutation_importance(
        model, X.iloc[test_idx][selected], y.iloc[test_idx], n_repeats=8,
        random_state=45, scoring="neg_mean_absolute_error", n_jobs=-1,
    )
    importance = pd.Series(heldout_perm.importances_mean, index=selected).sort_values(ascending=False)
    production_model = RandomForestRegressor(n_estimators=350, min_samples_leaf=3, max_features=0.7, random_state=43, n_jobs=-1)
    production_model.fit(X[selected], y)
    return {
        "model": model,
        "production_model": production_model,
        "mae": float(np.mean(np.abs(pred - y.iloc[test_idx]))),
        "r2": float(model.score(X.iloc[test_idx][selected], y.iloc[test_idx])),
        "importance": importance,
        "selected_features": selected,
        "test_engines": int(groups.iloc[test_idx].nunique()),
        "train_engines": int(groups.iloc[train_idx].nunique()),
        "y_test": y.iloc[test_idx],
        "pred_test": pred,
    }


def build_live_features(
    live: pd.DataFrame,
    engine_col: str,
    time_col: str,
    sensors: list[str],
    window: int = 8,
) -> tuple[pd.DataFrame, pd.Series]:
    """Create the same trailing signal features for unlabeled current histories."""
    df = live.copy().replace([np.inf, -np.inf], np.nan)
    df[engine_col] = df[engine_col].astype(str)
    df = df.sort_values([engine_col, time_col]).reset_index(drop=True)
    pieces = []
    for sensor in sensors:
        if sensor not in df:
            raise ValueError(f"Live history is missing the training sensor channel '{sensor}'.")
        values = pd.to_numeric(df[sensor], errors="coerce")
        grouped = values.groupby(df[engine_col], sort=False)
        rolling_mean = grouped.transform(lambda s: s.rolling(window, min_periods=1).mean())
        rolling_std = grouped.transform(lambda s: s.rolling(window, min_periods=2).std()).fillna(0)
        lag = grouped.shift(window - 1)
        slope = (values - lag) / max(window - 1, 1)
        names = [sensor, f"{sensor}__mean{window}", f"{sensor}__std{window}", f"{sensor}__slope{window}"]
        pieces.extend([values.rename(names[0]), rolling_mean.rename(names[1]), rolling_std.rename(names[2]), slope.rename(names[3])])
    X = pd.concat(pieces, axis=1).replace([np.inf, -np.inf], np.nan)
    keep = X.notna().all(axis=1)
    latest = df.loc[keep].groupby(engine_col, sort=False).tail(1).index
    return X.loc[latest].reset_index(drop=True), df.loc[latest, engine_col].reset_index(drop=True)


def condition_deviation(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Robust standardized distance to the measured operating envelope.

    This is a deviation score, not a failure probability or RUL estimate.
    """
    X = frame[columns].replace([np.inf, -np.inf], np.nan).dropna()
    if len(X) < 10:
        return pd.Series(dtype=float)
    model = IsolationForest(n_estimators=300, contamination="auto", random_state=42)
    model.fit(X)
    scores = -model.score_samples(X)
    scale = np.nanpercentile(scores, 95)
    return pd.Series(np.clip(scores / max(scale, 1e-9), 0, 2), index=X.index, name="relative_deviation")
