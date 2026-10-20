"""Gather the saved model results into the tables the dashboard and README report."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

SPLITS = ("train", "validation", "test_same", "test_unseen_rpm", "test_noise_15db", "test_heavy_noise", "test_noise_0db")


def _models(cache_dir) -> dict:
    """model name -> {split: result cell}, from whichever results files exist."""
    cache = Path(cache_dir)
    models = {}
    path = cache / "results_classical.json"
    if path.exists():
        data = json.loads(path.read_text())
        models.update({name: data[name] for name in ("svm", "softmax") if name in data})
    for name, filename in (("obd_ann", "results_obd_ann.json"), ("cnn", "results_cnn.json")):
        path = cache / filename
        if path.exists():
            models[name] = json.loads(path.read_text())
    return {m: {s: cell for s, cell in cells.items() if s in SPLITS} for m, cells in models.items()}


def build_results_table(cache_dir) -> pd.DataFrame:
    rows = [
        {"model": m, "split": s, "accuracy": cell["accuracy"], "macro_f1": cell["macro_f1"]}
        for m, cells in _models(cache_dir).items()
        for s, cell in cells.items()
    ]
    return pd.DataFrame(rows, columns=["model", "split", "accuracy", "macro_f1"])


def load_confusions(cache_dir) -> dict:
    """(model, split) -> confusion matrix (rows true, columns predicted, in schema.FAULTS order)."""
    return {(m, s): np.array(cell["confusion"]) for m, cells in _models(cache_dir).items() for s, cell in cells.items()}


def latency_table(cache_dir) -> pd.DataFrame:
    path = Path(cache_dir) / "results_latency.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    rows = [{"model": m, "fault": f, **v} for m, faults in data.items() for f, v in faults.items()]
    return pd.DataFrame(rows)


def autoencoder_table(cache_dir) -> pd.DataFrame:
    """One row per split and fault: AUROC and flagged share, plus the healthy false-alarm rate of that split."""
    path = Path(cache_dir) / "results_health_autoencoder.json"
    data = json.loads(path.read_text())["autoencoder"] if path.exists() else {}
    rows = [
        {"split": split, "fault": fault, "false_alarm": cell["false_alarm"], **values}
        for split, cell in data.items() if isinstance(cell, dict)
        for fault, values in cell["per_fault"].items()
    ]
    return pd.DataFrame(rows)
