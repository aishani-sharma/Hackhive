"""Replay a recorded trip: what the engine actually did against what the optimiser would have chosen."""
import numpy as np
import pandas as pd

from enginetwin.losses import degraded_efficiency, efficiency_loss
from enginetwin.optimiser import PRICE_RS_PER_LITRE, learned_efficiency, optimise
from enginetwin.vehicle import VehicleSpec, natural_rpm, wheel_power_w

UL_PER_S_TO_L_PER_H = 3.6e-3  # 1e-6 L per uL x 3600 s per hour
SOURCES = ("ocslab", "ved")
COLUMNS = [
    "time_s", "speed_kmh", "actual_gear", "opt_gear", "actual_rpm", "opt_rpm", "actual_eff", "opt_eff",
    "actual_fuel_l_h", "opt_fuel_l_h",
]


def gear_from_rpm(rpm, speed_kmh, spec: VehicleSpec) -> int:
    """The gear whose engine speed at this road speed is closest to the measured `rpm` (first gear when standing)."""
    if speed_kmh < 1.0:
        return 1
    speeds = [natural_rpm(speed_kmh, g, spec) for g in range(1, len(spec.gear_ratios) + 1)]
    return int(np.argmin(np.abs(np.log(np.array(speeds) / max(rpm, 1.0)))) + 1)


def _prepare(trip, source, spec) -> pd.DataFrame:
    """Rows with complete readings, the seconds each covers and the engine power each needs (W, at the crankshaft)."""
    needed = ["time_s", "vehicle_speed", "engine_speed", "fuel_consumption"] + (["engine_torque"] if source == "ocslab" else [])
    if not set(needed) <= set(trip.columns):
        raise ValueError(f"a {source} trip needs the columns {needed}")
    t = trip[needed].copy()
    gaps = t["time_s"].diff()
    t["dt"] = gaps.fillna(gaps.median()).fillna(1.0)  # the gap since the previous sample, taken before dropping rows
    if source == "ocslab":
        t["engine_power_w"] = t["engine_torque"].clip(lower=0) * t["engine_speed"] * 2 * np.pi / 60
    else:  # VED has no torque: the vehicle model turns speed and acceleration into the power the engine had to make
        accel = np.gradient(t["vehicle_speed"].to_numpy() / 3.6, t["time_s"].to_numpy()) if len(t) > 1 else np.zeros(len(t))
        t["engine_power_w"] = [wheel_power_w(v, a, spec) / spec.drivetrain_efficiency for v, a in zip(t["vehicle_speed"], accel)]
    return t.dropna().reset_index(drop=True)


def replay_trip(trip, model, spec: VehicleSpec, source, fault="healthy", severity=0.0, loss_rates=None, price=PRICE_RS_PER_LITRE):
    """(table, summary) for one trip. `source` is "ocslab" (power from measured torque) or "ved" (power from the vehicle
    model). Actual fuel is the recorded fuel; optimised fuel is that fuel scaled by actual / optimised efficiency, so the
    saving stays tied to real fuel data. With a fault, both are divided by (1 - assumed loss): the same work costs more."""
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}, got {source!r}")
    t = _prepare(trip, source, spec)
    loss = efficiency_loss(fault, severity, loss_rates)
    rows = []
    for r in t.itertuples():
        fuel = r.fuel_consumption * UL_PER_S_TO_L_PER_H / (1 - loss)
        actual_gear = gear_from_rpm(r.engine_speed, r.vehicle_speed, spec)
        row = {"time_s": r.time_s, "speed_kmh": r.vehicle_speed, "actual_gear": actual_gear, "actual_rpm": r.engine_speed,
               "actual_fuel_l_h": fuel, "dt": r.dt}
        if r.engine_power_w > 0 and r.engine_speed > 0:
            torque = r.engine_power_w / (r.engine_speed * 2 * np.pi / 60)
            actual_eff = degraded_efficiency(learned_efficiency(model, r.engine_speed, torque, r.vehicle_speed, spec), fault, severity, loss_rates)
            best = optimise(r.vehicle_speed, r.engine_power_w * spec.drivetrain_efficiency, model, spec, fault, severity, loss_rates, actual_gear)
            row.update(opt_gear=best["gear"], opt_rpm=best["rpm"], actual_eff=actual_eff, opt_eff=best["efficiency"],
                       opt_fuel_l_h=fuel * actual_eff / best["efficiency"])
        else:  # no brake power: nothing for the optimiser to improve
            row.update(opt_gear=actual_gear, opt_rpm=r.engine_speed, actual_eff=np.nan, opt_eff=np.nan, opt_fuel_l_h=fuel)
        rows.append(row)
    table = pd.DataFrame(rows, columns=COLUMNS + ["dt"])
    actual_l = float((table.actual_fuel_l_h * table.dt).sum() / 3600)
    opt_l = float((table.opt_fuel_l_h * table.dt).sum() / 3600)
    worked = table.actual_eff.notna()
    summary = {
        "total_actual_l": actual_l,
        "total_opt_l": opt_l,
        "saved_l": actual_l - opt_l,
        "saved_pct": 100 * (actual_l - opt_l) / actual_l if actual_l else 0.0,
        "rupees_saved": (actual_l - opt_l) * price,
        "mean_actual_eff": float(table.actual_eff[worked].mean()) if worked.any() else float("nan"),
        "mean_opt_eff": float(table.opt_eff[worked].mean()) if worked.any() else float("nan"),
        "fault_extra_l": actual_l * loss,
    }
    return table[COLUMNS], summary
