"""LiU-ICE data preparation and group-held-out fault probability models."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    fbeta_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, train_test_split

SIGNALS = [
    "Intercooler_pressure", "intercooler_temperature", "intake_manifold_pressure",
    "air_mass_flow", "engine_speed", "throttle_position", "wastegate_position",
    "injected_fuel_mass", "ambient_pressure", "ambient_temperature",
]
FAULT_LABELS = {
    "f_pic": "Intercooler pressure sensor",
    "f_pim": "Intake manifold pressure sensor",
    "f_waf": "Air mass flow sensor",
    "f_iml": "Intake manifold leak",
}
SENSOR_FAULT_TARGETS = {
    "f_pic": "Intercooler_pressure",
    "f_pim": "intake_manifold_pressure",
    "f_waf": "air_mass_flow",
}
SOURCE_URL = "https://vehsys.gitlab-pages.liu.se/dx_benchmarks/liuice/competition/trainingdata/trainingdata.zip"


def _fault_times(directory: Path) -> dict[str, float]:
    onset: dict[str, float] = {}
    for name in ("wltp_fault_time.txt", "ftp75city2_fault_time.txt"):
        path = directory / name
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = re.match(r"^\s*(\S+)\s+([0-9]+(?:\.[0-9]+)?)\s*$", line)
            if match:
                onset[match.group(1).lower()] = float(match.group(2))
    return onset


def _scenario(stem: str) -> tuple[str, str, str | None]:
    lower = stem.lower()
    cycle = "ftp75city2" if lower.startswith("ftp75city2") else "wltp"
    if "_nf" in lower:
        return "No fault", cycle, None
    for code, label in FAULT_LABELS.items():
        if code in lower:
            return label, cycle, code
    raise ValueError(f"Could not identify fault scenario in '{stem}'.")


def _lookup_onset(stem: str, times: dict[str, float]) -> float:
    key = stem.lower()
    aliases = {key, key.replace("_f_iml_", "_fiml_")}
    for alias in aliases:
        if alias in times:
            return times[alias]
    raise ValueError(f"No official fault-injection time is listed for '{stem}'.")


def _window_features(block: pd.DataFrame, seconds: float) -> dict[str, float]:
    values: dict[str, float] = {}
    for col in SIGNALS:
        x = pd.to_numeric(block[col], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(x).all():
            x = pd.Series(x).interpolate(limit_direction="both").to_numpy(dtype=float)
        values[f"{col}__mean"] = float(np.mean(x))
        values[f"{col}__std"] = float(np.std(x, ddof=0))
        values[f"{col}__range"] = float(np.max(x) - np.min(x))
        values[f"{col}__slope"] = float((x[-1] - x[0]) / max(seconds, 1e-9))
    return values


def build_feature_windows(directory: str | Path, window_seconds: float = 5.0, sample_hz: float = 20.0) -> pd.DataFrame:
    """Create causal, non-overlapping telemetry features and known fault labels."""
    root = Path(directory)
    files = sorted(root.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No LiU-ICE CSV runs found in {root}")
    onset_by_name = _fault_times(root)
    window_samples = int(round(window_seconds * sample_hz))
    if window_samples < 2:
        raise ValueError("Window must contain at least two samples.")
    rows: list[dict] = []
    for path in files:
        scenario, cycle, code = _scenario(path.stem)
        data = pd.read_csv(path)
        missing = set(["time", *SIGNALS]) - set(data.columns)
        if missing:
            raise ValueError(f"{path.name} is missing required columns: {', '.join(sorted(missing))}")
        data = data[["time", *SIGNALS]].dropna(subset=["time"]).reset_index(drop=True)
        injection_time = None if code is None else _lookup_onset(path.stem, onset_by_name)
        for start in range(0, len(data) - window_samples + 1, window_samples):
            block = data.iloc[start:start + window_samples]
            t_start = float(block["time"].iloc[0])
            t_end = float(block["time"].iloc[-1])
            t_mid = (t_start + t_end) / 2
            active = int(injection_time is not None and t_mid >= injection_time)
            rows.append({
                "trajectory_id": path.stem,
                "drive_cycle": cycle,
                "scenario": scenario,
                "fault_code": code or "normal",
                "window_start_s": t_start,
                "window_end_s": t_end,
                "injection_time_s": injection_time,
                "fault_active": active,
                **_window_features(block, window_seconds),
            })
    result = pd.DataFrame(rows)
    if result.empty or result["fault_active"].nunique() < 2:
        raise ValueError("The training runs must include both fault-free and fault-active windows.")
    return result


def add_early_warning_target(windows: pd.DataFrame, horizon_seconds: float = 30.0) -> pd.DataFrame:
    """Label windows whose next horizon contains a known injected fault onset."""
    result = windows.copy()
    before_event = result["injection_time_s"].notna() & (result["window_end_s"] < result["injection_time_s"])
    time_to_event = result["injection_time_s"] - result["window_end_s"]
    result["fault_within_horizon"] = (before_event & time_to_event.le(horizon_seconds)).astype(int)
    return result


def _classifier(seed: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=280, max_depth=14, min_samples_leaf=4, max_features=0.75,
        class_weight="balanced_subsample", random_state=seed, n_jobs=-1,
    )


def _select_features(train: pd.DataFrame, features: list[str], target: str) -> tuple[list[str], pd.Series]:
    """Select a compact subset using importance learned only from training runs."""
    ranker = _classifier(41)
    ranker.fit(train[features], train[target])
    importance = pd.Series(ranker.feature_importances_, index=features).sort_values(ascending=False)
    cumulative = importance.cumsum() / max(float(importance.sum()), 1e-12)
    n_keep = int(np.searchsorted(cumulative.to_numpy(), 0.95) + 1)
    selected = importance.head(max(8, min(n_keep, 24))).index.tolist()
    return selected, importance


def _group_split(data: pd.DataFrame, seed: int = 17) -> tuple[np.ndarray, np.ndarray]:
    group_table = data[["trajectory_id", "scenario"]].drop_duplicates("trajectory_id")
    counts = group_table["scenario"].value_counts()
    if counts.min() < 2:
        raise ValueError("Each scenario needs at least two distinct trajectories for a held-out evaluation.")
    train_ids, test_ids = train_test_split(
        group_table["trajectory_id"].to_numpy(), test_size=0.25, random_state=seed,
        stratify=group_table["scenario"],
    )
    return data["trajectory_id"].isin(train_ids).to_numpy(), data["trajectory_id"].isin(test_ids).to_numpy()


def fit_probability_model(
    windows: pd.DataFrame,
    target: str = "fault_active",
    horizon_seconds: float = 30.0,
) -> dict:
    """Fit a group-held-out RF probability model with grouped OOF sigmoid calibration."""
    data = add_early_warning_target(windows, horizon_seconds) if target == "fault_within_horizon" else windows.copy()
    if target not in data:
        raise ValueError(f"Unknown probability target: {target}")
    feature_cols = [c for c in data.columns if "__" in c]
    train_mask, test_mask = _group_split(data)
    train = data.loc[train_mask].reset_index(drop=True)
    test = data.loc[test_mask].reset_index(drop=True)
    if train[target].nunique() < 2 or test[target].nunique() < 2:
        raise ValueError("The trajectory split did not leave both classes in train and test.")
    selected, feature_importance = _select_features(train, feature_cols, target)

    group_count = train["trajectory_id"].nunique()
    n_splits = min(3, group_count)
    splitter = GroupKFold(n_splits=n_splits)
    oof = np.full(len(train), np.nan, dtype=float)
    models = []
    for fold, (fit_idx, val_idx) in enumerate(splitter.split(train[selected], train[target], groups=train["trajectory_id"])):
        fold_train = train.iloc[fit_idx]
        fold_selected, _ = _select_features(fold_train, feature_cols, target)
        model = _classifier(100 + fold)
        model.fit(fold_train[fold_selected], fold_train[target])
        if 1 not in model.classes_:
            raise ValueError("A grouped calibration fold had no positive fault examples.")
        oof[val_idx] = model.predict_proba(train.iloc[val_idx][fold_selected])[:, list(model.classes_).index(1)]
        models.append({"estimator": model, "features": fold_selected})
    valid_oof = np.isfinite(oof)
    if train.loc[valid_oof, target].nunique() < 2:
        raise ValueError("Grouped calibration predictions do not contain both target classes.")
    calibrator = LogisticRegression(C=1e3, max_iter=2000, class_weight=None)
    calibrator.fit(oof[valid_oof].reshape(-1, 1), train.loc[valid_oof, target])
    positive_index = list(calibrator.classes_).index(1)

    base_test = np.mean([
        fold["estimator"].predict_proba(test[fold["features"]])[:, list(fold["estimator"].classes_).index(1)]
        for fold in models
    ], axis=0)
    calibrated_test = calibrator.predict_proba(base_test.reshape(-1, 1))[:, positive_index]
    calibrated_oof = calibrator.predict_proba(oof[valid_oof].reshape(-1, 1))[:, positive_index]
    y_oof = train.loc[valid_oof, target].to_numpy()
    y_test = test[target].to_numpy()
    if target == "fault_active":
        # Prioritize a bounded healthy-run false-alarm rate. This exposes the
        # severe sensitivity limit of the active-state model in this dataset.
        healthy_oof = calibrated_oof[y_oof == 0]
        threshold = float(np.quantile(healthy_oof, 0.95)) if len(healthy_oof) else 0.5
        threshold_method = "95th percentile of grouped healthy OOF probabilities"
    else:
        candidates = np.unique(calibrated_oof)
        threshold = float(max(
            candidates,
            key=lambda t: fbeta_score(y_oof, calibrated_oof >= t, beta=2, zero_division=0),
        )) if len(candidates) else 0.5
        threshold_method = "maximum F2 on grouped OOF probabilities"
    prediction = calibrated_test >= threshold
    metrics = {
        "threshold": threshold,
        "threshold_method": threshold_method,
        "roc_auc": float(roc_auc_score(y_test, calibrated_test)),
        "average_precision": float(average_precision_score(y_test, calibrated_test)),
        "brier": float(brier_score_loss(y_test, calibrated_test)),
        "f1": float(f1_score(y_test, prediction, zero_division=0)),
        "precision": float(precision_score(y_test, prediction, zero_division=0)),
        "recall": float(recall_score(y_test, prediction, zero_division=0)),
        "false_alarm_rate": float(np.mean(prediction[y_test == 0])) if np.any(y_test == 0) else float("nan"),
        "test_rows": int(len(test)),
        "test_trajectories": int(test["trajectory_id"].nunique()),
        "train_trajectories": int(train["trajectory_id"].nunique()),
        "positive_prevalence": float(np.mean(y_test)),
    }
    test["fault_probability"] = calibrated_test
    test["predicted_fault"] = prediction.astype(int)
    return {
        "target": target,
        "window_seconds": float(np.median(windows["window_end_s"] - windows["window_start_s"])) + 1 / 20,
        "horizon_seconds": float(horizon_seconds),
        "features": feature_cols,
        "selected_features": selected,
        "feature_importance": feature_importance,
        "models": models,
        "calibrator": calibrator,
        "calibrator_positive_index": positive_index,
        "threshold": threshold,
        "metrics": metrics,
        "test_windows": test,
        "training_windows": train,
    }


def predict_fault_probability(model_card: dict, frame: pd.DataFrame) -> np.ndarray:
    base = np.mean([
        fold["estimator"].predict_proba(frame[fold["features"]])[:, list(fold["estimator"].classes_).index(1)]
        for fold in model_card["models"]
    ], axis=0)
    return model_card["calibrator"].predict_proba(base.reshape(-1, 1))[:, model_card["calibrator_positive_index"]]


def fit_sensor_consistency_models(
    training_windows: pd.DataFrame,
    false_alarm_quantile: float = 0.985,
    target_sensors: dict[str, str] | None = None,
) -> dict:
    """Fit redundant-sensor regressors on healthy windows.

    Each model estimates one selected sensor from the other nine sensor means.
    The alarm threshold is learned from healthy relative residuals only. This is
    suited to detecting a present sensor deviation and makes no claim that a
    scheduled bias can be predicted before it exists.
    """
    if not 0.9 <= false_alarm_quantile < 1:
        raise ValueError("false_alarm_quantile must be in [0.9, 1).")
    healthy = training_windows.loc[training_windows["fault_active"].eq(0)].copy()
    mean_features = [column for column in training_windows if column.endswith("__mean")]
    models: dict[str, dict] = {}
    targets = SENSOR_FAULT_TARGETS if target_sensors is None else target_sensors
    for fault_code, target_sensor in targets.items():
        target = f"{target_sensor}__mean"
        predictors = [column for column in mean_features if column != target]
        params = dict(
            n_estimators=300,
            min_samples_leaf=3,
            max_features=0.9,
            random_state=42,
            n_jobs=-1,
        )
        estimator = ExtraTreesRegressor(**params)
        if "trajectory_id" in healthy and healthy["trajectory_id"].nunique() >= 5:
            fit_idx, calibration_idx = next(
                GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(
                    healthy, groups=healthy["trajectory_id"]
                )
            )
            calibration_model = ExtraTreesRegressor(**{**params, "n_estimators": 120})
            calibration_model.fit(healthy.iloc[fit_idx][predictors], healthy.iloc[fit_idx][target])
            calibration_expected = calibration_model.predict(healthy.iloc[calibration_idx][predictors])
            calibration_observed = healthy.iloc[calibration_idx][target].to_numpy()
            calibration_relative_residual = np.abs(calibration_observed - calibration_expected) / np.maximum(np.abs(calibration_expected), 1e-9)
        else:
            calibration_expected = estimator.fit(healthy[predictors], healthy[target]).predict(healthy[predictors])
            calibration_observed = healthy[target].to_numpy()
            calibration_relative_residual = np.abs(calibration_observed - calibration_expected) / np.maximum(np.abs(calibration_expected), 1e-9)
        estimator.fit(healthy[predictors], healthy[target])
        models[fault_code] = {
            "fault_code": fault_code,
            "target_sensor": target_sensor,
            "target": target,
            "predictors": predictors,
            "estimator": estimator,
            "residual_threshold": float(np.quantile(calibration_relative_residual, false_alarm_quantile)),
        }
    return {
        "models": models,
        "target_sensors": {code: card["target_sensor"] for code, card in models.items()},
        "false_alarm_quantile": float(false_alarm_quantile),
        "confirmation_rule": "2 of 3 consecutive five-second windows",
    }


def sensor_consistency_evidence(
    model_card: dict,
    frame: pd.DataFrame,
    fault_codes: list[str] | None = None,
) -> pd.DataFrame:
    """Return each sensor residual as a multiple of its learned alarm threshold."""
    result = pd.DataFrame(index=frame.index)
    selected = model_card["models"] if fault_codes is None else {
        code: model_card["models"][code] for code in fault_codes
    }
    for fault_code, card in selected.items():
        expected = card["estimator"].predict(frame[card["predictors"]])
        observed = frame[card["target"]].to_numpy(dtype=float)
        relative_residual = np.abs(observed - expected) / np.maximum(np.abs(expected), 1e-9)
        result[fault_code] = relative_residual / max(float(card["residual_threshold"]), 1e-12)
    return result


def simulate_fault_scenario(
    baseline_rows: pd.DataFrame,
    training_rows: pd.DataFrame,
    active_model: dict,
    onset_model: dict,
    *,
    fault_code: str | None,
    severity: float,
    onset_seconds: float,
    start_window: int = 0,
    n_windows: int = 36,
) -> pd.DataFrame:
    """Create an explicitly synthetic feature-level fault replay from measured runs.

    The baseline must be a held-out no-fault trajectory. Fault deltas are learned
    from training trajectories only, within the same drive cycle when available.
    This is a what-if model demonstration, not a physical engine simulation.
    """
    if not 0 <= severity <= 1.5:
        raise ValueError("Fault severity must be between 0 and 1.5x the observed signature.")
    feature_cols = active_model["features"]
    normal = baseline_rows.loc[baseline_rows["fault_active"] == 0].sort_values("window_end_s")
    if start_window < 0 or start_window + n_windows > len(normal):
        raise ValueError("The selected baseline segment is shorter than the requested replay.")
    source = normal.iloc[start_window:start_window + n_windows].reset_index(drop=True)
    simulated = source[feature_cols].astype(float).copy()
    dt = float(active_model["window_seconds"])
    time_s = np.arange(n_windows, dtype=float) * dt
    warning_horizon = float(onset_model["horizon_seconds"])
    warning_start = max(0.0, onset_seconds - warning_horizon)
    # Let the modeled warning horizon see a gradual pre-event signature before
    # the scheduled injection point, then hold the full injected shift.
    ramp = np.clip((time_s - warning_start) / max(warning_horizon, dt), 0.0, 1.0)

    if fault_code is not None:
        train_normal = training_rows.loc[training_rows["fault_active"] == 0]
        train_fault = training_rows.loc[
            (training_rows["fault_active"] == 1) & (training_rows["fault_code"] == fault_code)
        ]
        cycle = str(source["drive_cycle"].iloc[0])
        same_cycle_normal = train_normal.loc[train_normal["drive_cycle"] == cycle]
        same_cycle_fault = train_fault.loc[train_fault["drive_cycle"] == cycle]
        if len(same_cycle_normal) >= 10 and len(same_cycle_fault) >= 10:
            train_normal, train_fault = same_cycle_normal, same_cycle_fault
        if train_fault.empty:
            raise ValueError(f"No training examples are available for fault code {fault_code}.")
        delta = train_fault[feature_cols].median() - train_normal[feature_cols].median()
        simulated = simulated.add(pd.DataFrame(np.outer(ramp * severity, delta.to_numpy()), columns=feature_cols), fill_value=0.0)
        for col in feature_cols:
            if col.endswith(("__std", "__range")):
                simulated[col] = simulated[col].clip(lower=0.0)

    result = pd.DataFrame({
        "simulation_time_s": time_s,
        "fault_ramp": ramp * severity,
        "fault_probability": predict_fault_probability(active_model, simulated),
        "onset_probability": predict_fault_probability(onset_model, simulated),
    })
    for signal in SIGNALS:
        result[signal] = simulated[f"{signal}__mean"].to_numpy()
    result.attrs["window_seconds"] = dt
    result.attrs["drive_cycle"] = str(source["drive_cycle"].iloc[0])
    result.attrs["warning_signature_start_s"] = warning_start
    return result


def fit_fault_isolator(windows: pd.DataFrame, selected_features: list[str]) -> dict:
    """Fit mode probabilities on active faults; report mode performance by held-out run."""
    active = windows.loc[windows["fault_active"] == 1].copy()
    train_mask, test_mask = _group_split(windows)
    train_ids = set(windows.loc[train_mask, "trajectory_id"])
    test_ids = set(windows.loc[test_mask, "trajectory_id"])
    fit = active.loc[active["trajectory_id"].isin(train_ids)]
    test = active.loc[active["trajectory_id"].isin(test_ids)]
    model = _classifier(71)
    model.fit(fit[selected_features], fit["scenario"])
    predicted = model.predict(test[selected_features])
    return {
        "model": model,
        "classes": list(model.classes_),
        "accuracy": float(np.mean(predicted == test["scenario"])),
        "macro_f1": float(f1_score(test["scenario"], predicted, average="macro", zero_division=0)),
        "test_rows": int(len(test)),
        "test_trajectories": int(test["trajectory_id"].nunique()),
    }


def summarize_dataset(windows: pd.DataFrame) -> dict:
    runs = windows.groupby("trajectory_id").agg(
        scenario=("scenario", "first"), cycle=("drive_cycle", "first"),
        windows=("trajectory_id", "size"), fault_active=("fault_active", "max"),
    )
    return {
        "runs": int(len(runs)),
        "windows": int(len(windows)),
        "channels": len(SIGNALS),
        "features": len([c for c in windows if "__" in c]),
        "classes": runs["scenario"].value_counts().to_dict(),
        "cycles": runs["cycle"].value_counts().to_dict(),
        "normal_windows": int((windows["fault_active"] == 0).sum()),
        "fault_windows": int((windows["fault_active"] == 1).sum()),
    }
