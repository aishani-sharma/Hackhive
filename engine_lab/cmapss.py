"""NASA C-MAPSS FD001 run-to-failure benchmark loader."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "nasa_cmapss"

BASE_COLUMNS = ["engine_id", "cycle", "setting_1", "setting_2", "setting_3"]
SENSOR_COLUMNS = [f"sensor_{i:02d}" for i in range(1, 22)]
COLUMNS = BASE_COLUMNS + SENSOR_COLUMNS


def load_fd001(data_dir: str | Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return complete training runs and labeled truncated test runs."""
    directory = Path(data_dir)
    train_path = directory / "train_FD001.txt"
    test_path = directory / "test_FD001.txt"
    rul_path = directory / "RUL_FD001.txt"
    missing = [path.name for path in (train_path, test_path, rul_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing NASA FD001 files: {', '.join(missing)}. Run `python prepare_cmapss.py`.")

    train = pd.read_csv(train_path, sep=r"\s+", header=None, names=COLUMNS)
    test = pd.read_csv(test_path, sep=r"\s+", header=None, names=COLUMNS)
    test_remaining = pd.read_csv(rul_path, sep=r"\s+", header=None).iloc[:, 0].to_numpy(dtype=float)
    train_end = train.groupby("engine_id")["cycle"].transform("max")
    train["rul"] = train_end - train["cycle"]
    test_end = test.groupby("engine_id")["cycle"].max()
    if len(test_remaining) != len(test_end):
        raise ValueError("NASA FD001 test labels do not match the number of test engines.")
    remaining_by_engine = pd.Series(test_remaining, index=test_end.index)
    test["rul"] = test["engine_id"].map(test_end + remaining_by_engine) - test["cycle"]
    return train, test
