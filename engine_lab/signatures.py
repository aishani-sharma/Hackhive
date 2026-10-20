"""Window-level DSP features and recording-held-out fault classification."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import kurtosis, skew
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def extract_window_signatures(
    signals: dict[str, np.ndarray],
    sample_rate: float,
    window_seconds: float,
    *,
    overlap: float = 0.5,
    recording_id: str = "recording",
) -> pd.DataFrame:
    """Compute aligned descriptive features for each window and channel."""
    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("Sample rate must be a positive finite number.")
    if not np.isfinite(window_seconds) or window_seconds <= 0:
        raise ValueError("Window duration must be a positive finite number.")
    if not 0 <= overlap < 1:
        raise ValueError("Overlap must be in [0, 1).")
    usable: dict[str, np.ndarray] = {}
    for name, values in signals.items():
        signal = np.asarray(values, dtype=float).reshape(-1)
        finite = np.isfinite(signal)
        if finite.sum() < 32:
            raise ValueError(f"Channel {name} needs at least 32 finite samples.")
        indexes = np.arange(len(signal))
        signal = np.interp(indexes, indexes[finite], signal[finite])
        usable[str(name)] = signal
    if not usable:
        raise ValueError("Each selected channel needs at least 32 finite samples.")

    # A multichannel recording is synchronized by sample index. Truncate small
    # channel-length differences to the shared duration rather than padding data.
    shared_length = min(map(len, usable.values()))
    window_samples = int(round(sample_rate * window_seconds))
    if window_samples < 32:
        raise ValueError("The selected window must contain at least 32 samples.")
    if shared_length < window_samples:
        raise ValueError("The recording is shorter than one selected analysis window.")
    step = max(1, int(round(window_samples * (1 - overlap))))
    starts = range(0, shared_length - window_samples + 1, step)
    rows: list[dict[str, float | int | str]] = []
    for window_id, start in enumerate(starts):
        row: dict[str, float | int | str] = {
            "recording_id": str(recording_id),
            "window": window_id + 1,
            "start_s": start / sample_rate,
            "end_s": (start + window_samples) / sample_rate,
        }
        for channel, values in usable.items():
            x = values[start:start + window_samples]
            x = x - np.mean(x)
            rms = float(np.sqrt(np.mean(x * x)))
            freq, power = welch(x, fs=sample_rate, nperseg=min(2048, len(x)))
            power_sum = float(np.sum(power))
            prefix = f"{channel}__"
            row.update({
                prefix + "mean": float(np.mean(values[start:start + window_samples])),
                prefix + "std": float(np.std(x)),
                prefix + "rms": rms,
                prefix + "peak_to_peak": float(np.ptp(x)),
                prefix + "dominant_hz": float(freq[1 + np.argmax(power[1:])]) if len(power) > 1 else 0.0,
                prefix + "centroid_hz": float(np.sum(freq * power) / max(power_sum, 1e-12)),
                prefix + "spectral_flatness": float(np.exp(np.mean(np.log(power + 1e-12))) / max(np.mean(power), 1e-12)),
                prefix + "skewness": float(skew(x, bias=False)) if len(x) > 2 else 0.0,
                prefix + "kurtosis_excess": float(kurtosis(x, fisher=True, bias=False)) if len(x) > 3 else 0.0,
                prefix + "crest_factor": float(np.max(np.abs(x)) / max(rms, 1e-12)),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def fit_recording_heldout_svm(rows: pd.DataFrame, label_col: str = "fault_label") -> dict:
    """Evaluate an RBF SVM while keeping every window from a recording together."""
    if label_col not in rows or "recording_id" not in rows:
        raise ValueError("Training rows must contain fault_label and recording_id columns.")
    feature_cols = [c for c in rows if "__" in c]
    clean = rows.dropna(subset=[label_col]).copy()
    clean[label_col] = clean[label_col].astype(str)
    classes = sorted(clean[label_col].unique())
    if len(classes) < 2:
        raise ValueError("At least two fault classes are required.")
    per_class_recordings = clean[["recording_id", label_col]].drop_duplicates().groupby(label_col)["recording_id"].nunique()
    if len(per_class_recordings) != len(classes) or int(per_class_recordings.min()) < 2:
        raise ValueError("Each class needs at least two separate recordings for a recording-held-out evaluation.")
    if not feature_cols:
        raise ValueError("No window features were found.")

    recording_labels = clean[["recording_id", label_col]].drop_duplicates()
    train_ids, test_ids = train_test_split(
        recording_labels["recording_id"].to_numpy(),
        test_size=max(len(classes), int(np.ceil(len(recording_labels) * 0.25))),
        random_state=42,
        stratify=recording_labels[label_col].to_numpy(),
    )
    train = clean[clean["recording_id"].isin(train_ids)]
    test = clean[clean["recording_id"].isin(test_ids)]
    if set(train[label_col]) != set(classes) or set(test[label_col]) != set(classes):
        raise ValueError("The recording split must include every class in both training and held-out recordings.")

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        SVC(C=2.0, kernel="rbf", class_weight="balanced", probability=True, random_state=42),
    )
    model.fit(train[feature_cols], train[label_col])
    predicted = model.predict(test[feature_cols])
    probabilities = model.predict_proba(test[feature_cols])
    labels = model.named_steps["svc"].classes_
    report = classification_report(test[label_col], predicted, labels=labels, output_dict=True, zero_division=0)
    scored_test = test[["recording_id", "window", "start_s", "end_s", label_col]].copy()
    scored_test["predicted_fault"] = predicted
    for i, label in enumerate(labels):
        scored_test[f"score__{label}"] = probabilities[:, i]
    full_model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        SVC(C=2.0, kernel="rbf", class_weight="balanced", probability=True, random_state=42),
    )
    full_model.fit(clean[feature_cols], clean[label_col])
    return {
        "model": full_model,
        "feature_columns": feature_cols,
        "classes": list(labels),
        "recordings": int(clean["recording_id"].nunique()),
        "train_recordings": int(train["recording_id"].nunique()),
        "test_recordings": int(test["recording_id"].nunique()),
        "accuracy": float(accuracy_score(test[label_col], predicted)),
        "macro_f1": float(f1_score(test[label_col], predicted, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(test[label_col], predicted, labels=labels),
        "classification_report": report,
        "test_predictions": scored_test,
    }
