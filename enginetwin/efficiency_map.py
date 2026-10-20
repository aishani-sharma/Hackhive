import numpy as np
import pandas as pd

from enginetwin.efficiency import brake_thermal_efficiency


def sweet_spot(cells: pd.DataFrame) -> pd.Series:
    """The most efficient cell of an efficiency map."""
    return cells.loc[cells["efficiency"].idxmax()]


def relative_sweet_spot(cells: pd.DataFrame) -> tuple[float, float]:
    """Sweet spot as fractions (0-1) of the map's own rpm and torque range. This lets different engines be compared."""
    best = sweet_spot(cells)
    rpm_fraction = (best["rpm"] - cells["rpm"].min()) / (cells["rpm"].max() - cells["rpm"].min())
    torque_fraction = (best["torque"] - cells["torque"].min()) / (cells["torque"].max() - cells["torque"].min())
    return float(rpm_fraction), float(torque_fraction)


def efficiency_map(df: pd.DataFrame, rpm_step=250, torque_step=5, min_samples=10) -> pd.DataFrame:
    """Median brake thermal efficiency per rpm x torque cell, one row per cell (cell-centre coordinates)."""
    eff = brake_thermal_efficiency(df["engine_torque"], df["engine_speed"], df["fuel_consumption"])
    cells = pd.DataFrame(
        {
            "rpm": np.floor(df["engine_speed"] / rpm_step) * rpm_step + rpm_step / 2,
            "torque": np.floor(df["engine_torque"] / torque_step) * torque_step + torque_step / 2,
            "efficiency": eff,
        }
    ).dropna()
    cells = cells[cells["efficiency"] > 0]
    out = cells.groupby(["rpm", "torque"], as_index=False).agg(
        efficiency=("efficiency", "median"), samples=("efficiency", "size")
    )
    return out[out["samples"] >= min_samples].reset_index(drop=True)
