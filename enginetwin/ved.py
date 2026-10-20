import re
from pathlib import Path

import pandas as pd

from enginetwin.efficiency import PETROL_DENSITY_G_PER_ML

# Assumption: closed-loop petrol running at lambda = 1 (14.7 g air per g fuel), corrected by the ECU fuel trims.
STOICH_AFR = 14.7


def _clean_name(name: str) -> str:
    name = re.sub(r"[\[(].*?[\])]", "", name)  # drop the unit in [..] or (..)
    name = re.sub(r"[^0-9a-z]+", "_", name.lower()).strip("_")
    return name.replace("bank_", "bank")


def _with_unit(clean: str, raw: str) -> str:
    unit = re.search(r"[\[(](.*?)[\])]", raw)
    return f"{clean}_{_clean_name(unit.group(1))}" if unit else clean


def load_ved(paths, static_path=None) -> pd.DataFrame:
    """Load weekly VED CSVs with the same column names the Ocslab loader uses.

    With `static_path` (VED's ICE&HEV vehicle table) only combustion-engine cars are kept: hybrids and EVs
    have no usable air-flow/fuel signals for the efficiency work.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    if static_path is not None:
        static = pd.read_excel(static_path)
        df = df[df["VehId"].isin(static.loc[static["Vehicle Type"] == "ICE", "VehId"])]
    base = [_clean_name(c) for c in df.columns]
    clashing = {n for n in base if base.count(n) > 1}  # e.g. A/C power given in both kW and Watts
    df.columns = [_with_unit(n, c) if n in clashing else n for n, c in zip(base, df.columns)]
    df = df.rename(columns={"engine_rpm": "engine_speed"})
    trim = df["short_term_fuel_trim_bank1"].fillna(0) + df["long_term_fuel_trim_bank1"].fillna(0)
    fuel_g_per_s = df["maf"] / STOICH_AFR * (1 + trim / 100)
    estimated = fuel_g_per_s / PETROL_DENSITY_G_PER_ML * 1000  # uL/s, same unit as Ocslab
    measured = df["fuel_rate"] * 1e6 / 3600  # L/h -> uL/s
    df["fuel_consumption"] = measured.fillna(estimated)
    df["trip_id"] = df["vehid"].astype(str) + "-" + df["trip"].astype(str)
    df["time_s"] = (df["timestamp"] - df.groupby("trip_id")["timestamp"].transform("min")) / 1000
    return df
