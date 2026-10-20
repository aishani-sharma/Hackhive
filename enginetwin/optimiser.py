"""Pick the gear (and so the engine point) that burns the least fuel for the power the driver needs right now.

The gearbox is discrete: road speed fixes the engine speed in each gear, so searching gear and rpm is a search over
the five gears. Efficiency comes from the learned model; fault losses are the assumed rates in enginetwin.losses."""
import numpy as np

from enginetwin.efficiency import PETROL_DENSITY_G_PER_ML, PETROL_LHV_J_PER_KG
from enginetwin.efficiency_model import KIA_SOUL_MAX_TORQUE_NM, MIN_SPEED_KMH
from enginetwin.losses import degraded_efficiency, efficiency_loss
from enginetwin.vehicle import VehicleSpec, choose_gear, engine_point, natural_rpm

TORQUE_LIMIT = 0.95  # never ask for more than this share of the full-load torque
KNOCK_LOAD = 0.85  # share of full load above which low-rpm running risks knock (an assumed rule)
KNOCK_RPM = 2500.0
KNOCK_FAULT_TIGHTENING = 0.3  # with knock present the limit falls by this share at full severity (an assumed rule)
PRICE_RS_PER_LITRE = 105.0
LITRES_PER_KG = 1.0 / (PETROL_DENSITY_G_PER_ML)  # g/mL and kg/L are numerically equal


def learned_efficiency(model, rpm, torque_nm, speed_kmh, spec: VehicleSpec) -> float:
    """Efficiency the learned model predicts at an engine point (features in the order it was trained on)."""
    load = torque_nm / KIA_SOUL_MAX_TORQUE_NM * 100.0
    row = [[rpm, load, speed_kmh, rpm / max(speed_kmh, MIN_SPEED_KMH), 0.0]]
    return float(model.predict(np.array(row, dtype=float))[0])


def knock_risk(rpm, torque_nm, spec: VehicleSpec, limit=KNOCK_LOAD) -> bool:
    """High torque at low engine speed invites knock."""
    return bool(rpm < KNOCK_RPM and torque_nm > limit * spec.full_load_torque(rpm))


def candidates(speed_kmh, power_w, spec: VehicleSpec, knock_limit=KNOCK_LOAD) -> list:
    """Gears (highest first) that keep the engine between idle and the redline, within 95% of full-load torque, with low
    knock risk. Each is a dict of gear, rpm, torque_nm, load_pct and full_load_nm."""
    found = []
    for gear in range(len(spec.gear_ratios), 0, -1):
        if not spec.idle_rpm <= natural_rpm(speed_kmh, gear, spec) <= spec.redline_rpm:
            continue
        rpm, torque = engine_point(speed_kmh, gear, power_w, spec)
        full = float(spec.full_load_torque(rpm))
        if torque <= TORQUE_LIMIT * full and not knock_risk(rpm, torque, spec, knock_limit):
            found.append({"gear": gear, "rpm": rpm, "torque_nm": torque, "load_pct": torque / full * 100.0, "full_load_nm": full})
    return found


def _fuel_l_per_h(engine_power_w, efficiency) -> float:
    fuel_w = engine_power_w / efficiency
    return float(fuel_w / PETROL_LHV_J_PER_KG * LITRES_PER_KG * 3600.0)


def _point(model, speed_kmh, gear, power_w, spec, fault, severity, loss_rates) -> dict:
    rpm, torque = engine_point(speed_kmh, gear, power_w, spec)
    healthy = learned_efficiency(model, rpm, torque, speed_kmh, spec)
    efficiency = degraded_efficiency(healthy, fault, severity, loss_rates)
    engine_power = torque * rpm * 2 * np.pi / 60
    return {
        "gear": gear, "rpm": rpm, "torque_nm": torque, "efficiency_healthy": healthy, "efficiency": efficiency,
        "fuel_l_per_h": _fuel_l_per_h(engine_power, efficiency), "fuel_l_per_h_healthy": _fuel_l_per_h(engine_power, healthy),
    }


def optimise(
    speed_kmh, power_w, model, spec: VehicleSpec, fault="healthy", severity=0.0, loss_rates=None, actual_gear=None,
    price_per_litre=PRICE_RS_PER_LITRE,
) -> dict:
    """Best gear for the wheel power `power_w` at `speed_kmh`, compared with the gear actually in use.

    Efficiencies include the assumed fault loss. With a fault the optimiser recalculates: knock tightens the knock
    limit, and the result reports the efficiency and fuel the fault costs against a healthy engine at the same point."""
    actual_gear = actual_gear or choose_gear(speed_kmh, power_w, spec)
    knock_limit = KNOCK_LOAD * (1 - KNOCK_FAULT_TIGHTENING * severity) if "knock" in fault.split("+") else KNOCK_LOAD
    loss = efficiency_loss(fault, severity, loss_rates)
    if power_w <= 0:  # overrun or idle: no brake power, so no efficiency to improve
        rpm, torque = engine_point(speed_kmh, actual_gear, 0.0, spec)
        nothing = {"gear": actual_gear, "rpm": rpm, "torque_nm": 0.0, "efficiency": None, "efficiency_healthy": None, "fuel_l_per_h": 0.0}
        return {**nothing, "forced": False, "actual": dict(nothing), "efficiency_gain": 0.0, "fuel_saved_l_per_h": 0.0,
                "rupees_saved_per_h": 0.0, "fault_loss_fraction": loss, "efficiency_lost_to_fault": 0.0, "fault_extra_fuel_l_per_h": 0.0}
    options = candidates(speed_kmh, power_w, spec, knock_limit)
    forced = not options
    gears = [1] if forced else [c["gear"] for c in options]
    points = [_point(model, speed_kmh, g, power_w, spec, fault, severity, loss_rates) for g in gears]
    best = max(points, key=lambda p: p["efficiency_healthy"])
    actual = _point(model, speed_kmh, actual_gear, power_w, spec, fault, severity, loss_rates)
    saved = actual["fuel_l_per_h"] - best["fuel_l_per_h"]
    return {
        **best,
        "forced": forced,
        "actual": actual,
        "efficiency_gain": best["efficiency"] - actual["efficiency"],
        "fuel_saved_l_per_h": saved,
        "rupees_saved_per_h": saved * price_per_litre,
        "fault_loss_fraction": loss,
        "efficiency_lost_to_fault": best["efficiency_healthy"] - best["efficiency"],
        "fault_extra_fuel_l_per_h": best["fuel_l_per_h"] - best["fuel_l_per_h_healthy"],
    }
