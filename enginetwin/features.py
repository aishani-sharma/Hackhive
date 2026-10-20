import numpy as np
import pandas as pd

from enginetwin.efficiency import brake_power_w, brake_thermal_efficiency, bsfc_g_per_kwh


def _ratio(numerator, denominator):
    return (numerator / denominator).where(denominator > 0, np.nan)


# feature name -> (input columns, how to compute it). A feature is skipped when an input column is absent.
_INSTANT = {
    "torque_utilisation": (
        ("engine_torque", "maximum_indicated_engine_torque"),
        lambda d: _ratio(d["engine_torque"], d["maximum_indicated_engine_torque"]),
    ),
    "mechanical_efficiency": (
        ("engine_torque", "torque_of_friction"),
        lambda d: _ratio(d["engine_torque"], d["engine_torque"] + d["torque_of_friction"]),
    ),
    "brake_power_w": (
        ("engine_torque", "engine_speed"),
        lambda d: pd.Series(brake_power_w(d["engine_torque"], d["engine_speed"]), index=d.index),
    ),
    "thermal_efficiency": (
        ("engine_torque", "engine_speed", "fuel_consumption"),
        lambda d: pd.Series(
            brake_thermal_efficiency(d["engine_torque"], d["engine_speed"], d["fuel_consumption"]), index=d.index
        ),
    ),
    "bsfc_g_per_kwh": (
        ("engine_torque", "engine_speed", "fuel_consumption"),
        lambda d: pd.Series(
            bsfc_g_per_kwh(d["engine_torque"], d["engine_speed"], d["fuel_consumption"]), index=d.index
        ),
    ),
    "converter_slip": (
        ("engine_speed", "torque_converter_speed"),
        lambda d: 1 - _ratio(d["torque_converter_speed"], d["engine_speed"]),
    ),
    "gear_match": (
        ("engine_speed", "vehicle_speed"),
        lambda d: _ratio(d["engine_speed"], d["vehicle_speed"]),
    ),
    "total_fuel_trim": (
        ("short_term_fuel_trim_bank1", "long_term_fuel_trim_bank1"),
        lambda d: d["short_term_fuel_trim_bank1"] + d["long_term_fuel_trim_bank1"],
    ),
}


# Above this a petrol engine is physically impossible. The row is a torque/fuel timing mismatch at 1 Hz
# (seen in ~0.7% of Ocslab rows, mostly hard deceleration). Masked, not corrected.
MAX_PLAUSIBLE_BTE = 0.45
WARM_C = 85.0  # coolant temperature treated as "warm" (normal operating temperature)


def _seconds_covered(trip: pd.DataFrame) -> pd.Series:
    """Seconds each sample stands for: the gap since the previous one (1 s when the trip has no clock)."""
    if "time_s" not in trip:
        return pd.Series(1.0, index=trip.index)
    gaps = trip["time_s"].diff()
    return gaps.fillna(gaps.median()).fillna(1.0)


def _trip_economy(trip: pd.DataFrame) -> float:
    """km/L over the seconds where fuel flow is known (distance and fuel must cover the same seconds)."""
    known = trip["fuel_consumption"].notna() & trip["vehicle_speed"].notna()
    seconds = _seconds_covered(trip)[known]
    litres = (trip.loc[known, "fuel_consumption"] * seconds).sum() * 1e-6
    km = (trip.loc[known, "vehicle_speed"] * seconds).sum() / 3600
    return km / litres if litres > 0 else np.nan


def _warmup_seconds(trip: pd.DataFrame) -> float:
    warm = trip.loc[trip["engine_coolant_temperature"] >= WARM_C, "time_s"]
    return warm.iloc[0] - trip["time_s"].iloc[0] if len(warm) else np.nan


# feature name -> (input columns, summary of one trip's rows). Samples are 1 Hz. One row is one second.
_PER_TRIP = {
    "idle_fuel_ml": (
        ("vehicle_speed", "fuel_consumption"),
        lambda t: (t["fuel_consumption"] * _seconds_covered(t))[t["vehicle_speed"] == 0].sum() / 1000,
    ),
    "fuel_economy_km_per_l": (("vehicle_speed", "fuel_consumption"), lambda t: _trip_economy(t)),
    "high_load_share": (
        ("calculated_load_value",),
        lambda t: (t["calculated_load_value"] > 80).mean(),
    ),
    "warmup_time_s": (
        ("time_s", "engine_coolant_temperature"),
        lambda t: _warmup_seconds(t),
    ),
    "warm_share": (
        ("engine_coolant_temperature",),
        lambda t: (t["engine_coolant_temperature"] >= WARM_C).mean(),
    ),
    "idle_rpm_std": (
        ("vehicle_speed", "engine_speed"),
        lambda t: t.loc[(t["vehicle_speed"] == 0) & (t["engine_speed"] > 0), "engine_speed"].std(),
    ),
    "aggressiveness": (
        ("accelerator_pedal_value",),
        lambda t: t["accelerator_pedal_value"].diff().abs().iloc[1:].mean(),
    ),
}


def trip_features(df: pd.DataFrame) -> pd.DataFrame:
    """One row per trip summarising how the engine was used."""
    rows = []
    for trip_id, trip in df.groupby("trip_id", sort=False):
        row = {"trip_id": trip_id}
        for name, (inputs, summarise) in _PER_TRIP.items():
            if all(c in df.columns for c in inputs):
                row[name] = summarise(trip)
        rows.append(row)
    return pd.DataFrame(rows)


def add_instant_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for name, (inputs, compute) in _INSTANT.items():
        if all(c in df.columns for c in inputs):
            out[name] = compute(df)
    if "thermal_efficiency" in out:
        artefact = out["thermal_efficiency"] > MAX_PLAUSIBLE_BTE
        out.loc[artefact, ["thermal_efficiency", "bsfc_g_per_kwh"]] = np.nan
    return out
