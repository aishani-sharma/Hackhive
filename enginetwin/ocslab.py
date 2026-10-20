import re

import pandas as pd

# Identity columns stay even when constant (e.g. a file holding a single driver).
_KEEP = {"driver", "time_s", "pathorder"}

# Near-duplicates flagged in the dataset paper (arXiv 2207.10807): redundant, though not byte-identical.
_REDUNDANT = ("engine_torque_after_correction", "engine_coolant_temperature_1")


def _clean_name(name: str) -> str:
    return re.sub(r"[^0-9a-z]+", "_", name.lower()).strip("_")


def _identical_copies(numeric: pd.DataFrame) -> list[str]:
    """Names of columns that repeat an earlier column exactly (identity columns are never dropped)."""
    seen = set()
    copies = []
    for name in numeric.columns:
        key = numeric[name].to_numpy().tobytes()
        if key in seen and name not in _KEEP:
            copies.append(name)
        seen.add(key)
    return copies


def load_ocslab(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [_clean_name(c) for c in df.columns]
    df = df.rename(columns={"class": "driver"})

    constant = [c for c in df.columns if c not in _KEEP and df[c].nunique() <= 1]
    df = df.drop(columns=constant)

    df = df.drop(columns=_identical_copies(df.select_dtypes("number")))
    df = df.drop(columns=[c for c in _REDUNDANT if c in df.columns])

    new_trip = (df["driver"] != df["driver"].shift()) | (df["time_s"] <= df["time_s"].shift())
    trip_no = new_trip.groupby(df["driver"]).cumsum() - 1
    df["trip_id"] = df["driver"] + "-" + trip_no.astype(str)
    return df
