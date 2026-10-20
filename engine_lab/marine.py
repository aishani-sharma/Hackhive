"""High-channel marine-diesel benchmark, kept separate from automotive models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "marine_engine" / "raw" / "Marine_Engine_Fault_Data"
SOURCE_URL = "https://zenodo.org/records/19857425"
FAULT_NAMES = {
    "AC_Fouling": "Air-cooler fouling",
    "AF_Clogging": "Air-filter clogging",
    "Injector_Nozzle": "Injector-nozzle clogging",
    "Pump_Cavitation": "Cooling-water pump cavitation",
    "Turbine_Degradation": "Turbine degradation",
}


def load_runs(directory: str | Path = DATA_DIR) -> pd.DataFrame:
    """Read 15 controlled fault runs, keeping all numeric channels and file groups."""
    root = Path(directory)
    files = sorted(path for path in root.glob("*/*.csv") if path.name != "Reference_Data.csv")
    if not files:
        raise FileNotFoundError(f"No marine-engine fault runs found under {root}")
    runs = []
    for path in files:
        frame = pd.read_csv(path, skiprows=[1, 2], low_memory=False)
        frame["Anomaly State"] = pd.to_numeric(frame["Anomaly State"], errors="coerce")
        frame = frame.loc[frame["Anomaly State"].isin([0, 1])].copy()
        frame["trajectory_id"] = path.stem
        frame["fault_mode"] = FAULT_NAMES.get(path.parent.name, path.parent.name)
        runs.append(frame)
    all_rows = pd.concat(runs, ignore_index=True)
    excluded = {"Time_abs", "Time_rel", "Anomaly State", "trajectory_id", "fault_mode"}
    channel_cols = [col for col in all_rows.columns if col not in excluded]
    for col in channel_cols:
        all_rows[col] = pd.to_numeric(all_rows[col], errors="coerce")
    keep = [col for col in channel_cols if all_rows[col].notna().mean() >= 0.5 and all_rows[col].nunique(dropna=True) > 1]
    if len(keep) < 30:
        raise ValueError(f"Only {len(keep)} usable synchronous channels found; expected at least 30.")
    all_rows = all_rows[["trajectory_id", "fault_mode", "Time_rel", "Anomaly State", *keep]]
    all_rows = all_rows.rename(columns={"Time_rel": "time_s", "Anomaly State": "anomaly_active"})
    all_rows["anomaly_active"] = all_rows["anomaly_active"].astype(int)
    return all_rows


def _select(train: pd.DataFrame, features: list[str], target: str) -> tuple[list[str], pd.Series, pd.Series]:
    medians = train[features].median().fillna(0.0)
    model = RandomForestClassifier(
        n_estimators=100, max_depth=12, min_samples_leaf=8,
        class_weight="balanced_subsample", max_features=0.65, random_state=81, n_jobs=-1,
        max_samples=0.7,
    )
    weights = 1 / train.groupby("trajectory_id")[target].transform("size")
    model.fit(train[features].fillna(medians), train[target], sample_weight=weights)
    importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    n = min(30, max(12, int(np.searchsorted((importance.cumsum() / importance.sum()).to_numpy(), 0.95) + 1)))
    return importance.head(n).index.tolist(), importance, medians


def fit_anomaly_model(rows: pd.DataFrame) -> dict:
    """Calibrated fault-state probability with leave-whole-run-out validation."""
    state_counts = rows.groupby("trajectory_id").anomaly_active.nunique()
    eligible = set(state_counts[state_counts == 2].index)
    rows = rows.loc[rows.trajectory_id.isin(eligible)].copy()
    if len(eligible) < 6:
        raise ValueError("At least six runs with both pre-anomaly and anomaly segments are needed for grouped validation.")
    features = [col for col in rows if col not in {"trajectory_id", "fault_mode", "time_s", "anomaly_active"}]
    groups = rows[["trajectory_id", "fault_mode"]].drop_duplicates("trajectory_id")
    train_ids, test_ids = train_test_split(
        groups.trajectory_id.to_numpy(), test_size=0.34, random_state=29,
        stratify=groups.fault_mode,
    )
    train = rows.loc[rows.trajectory_id.isin(train_ids)].reset_index(drop=True)
    test = rows.loc[rows.trajectory_id.isin(test_ids)].reset_index(drop=True)
    selected, importance, _ = _select(train, features, "anomaly_active")
    oof = np.full(len(train), np.nan)
    ensemble = []
    for fold_id, (fit_idx, val_idx) in enumerate(GroupKFold(3).split(train, train.anomaly_active, train.trajectory_id)):
        fold_train = train.iloc[fit_idx]
        fold_features, _, medians = _select(fold_train, features, "anomaly_active")
        clf = RandomForestClassifier(
            n_estimators=100, max_depth=12, min_samples_leaf=8,
            class_weight="balanced_subsample", max_features=0.65, random_state=100 + fold_id,
            n_jobs=-1, max_samples=0.7,
        )
        weights = 1 / fold_train.groupby("trajectory_id").anomaly_active.transform("size")
        clf.fit(fold_train[fold_features].fillna(medians), fold_train.anomaly_active, sample_weight=weights)
        if 1 not in clf.classes_:
            raise ValueError("A grouped calibration fold had no anomaly examples.")
        oof[val_idx] = clf.predict_proba(train.iloc[val_idx][fold_features].fillna(medians))[:, list(clf.classes_).index(1)]
        ensemble.append({"model": clf, "features": fold_features, "medians": medians})
    valid = np.isfinite(oof)
    y_oof = train.loc[valid, "anomaly_active"].to_numpy()
    if len(np.unique(y_oof)) < 2:
        raise ValueError("Grouped out-of-fold predictions do not contain both anomaly states.")
    calibrator = LogisticRegression(C=1e3, max_iter=1000).fit(oof[valid, None], y_oof)
    positive = list(calibrator.classes_).index(1)
    raw_test = np.mean([
        part["model"].predict_proba(test[part["features"]].fillna(part["medians"]))[:, list(part["model"].classes_).index(1)]
        for part in ensemble
    ], axis=0)
    raw_auc = float(roc_auc_score(test.anomaly_active, raw_test))
    probability = calibrator.predict_proba(raw_test[:, None])[:, positive]
    calibrated_oof = calibrator.predict_proba(oof[valid, None])[:, positive]
    threshold = float(max(
        np.unique(calibrated_oof),
        key=lambda t: fbeta_score(y_oof, calibrated_oof >= t, beta=2, zero_division=0),
    ))
    predicted = probability >= threshold
    y_test = test.anomaly_active.to_numpy()
    test = test.assign(anomaly_probability=probability, predicted_anomaly=predicted.astype(int))
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, probability)),
        "raw_roc_auc": raw_auc,
        "average_precision": float(average_precision_score(y_test, probability)),
        "brier": float(brier_score_loss(y_test, probability)),
        "recall": float(recall_score(y_test, predicted, zero_division=0)),
        "precision": float(precision_score(y_test, predicted, zero_division=0)),
        "false_alarm_rate": float(np.mean(predicted[y_test == 0])) if np.any(y_test == 0) else float("nan"),
        "threshold": threshold,
        "threshold_method": "maximum F2 on grouped out-of-fold training probabilities",
        "test_rows": len(test),
        "train_runs": len(train_ids),
        "test_runs": len(test_ids),
        "prevalence": float(y_test.mean()),
    }
    return {"features": features, "selected_features": selected, "importance": importance,
            "ensemble": ensemble, "calibrator": calibrator, "positive_index": positive,
            "threshold": threshold, "metrics": metrics, "test_rows": test,
            "train_runs": len(train_ids), "test_runs": len(test_ids)}


def summarize(rows: pd.DataFrame) -> dict:
    return {
        "trajectories": int(rows.trajectory_id.nunique()),
        "samples": int(len(rows)),
        "channels": int(len([c for c in rows if c not in {"trajectory_id", "fault_mode", "time_s", "anomaly_active"}])),
        "fault_modes": rows[["trajectory_id", "fault_mode"]].drop_duplicates().fault_mode.value_counts().to_dict(),
        "anomaly_fraction": float(rows.anomaly_active.mean()),
    }
