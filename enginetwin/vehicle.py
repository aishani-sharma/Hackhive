"""A simple vehicle model: road speed and acceleration in, the rpm and torque the engine must deliver out.

Every parameter is an assumption for a 1.6 L Kia Soul with a 5-speed gearbox, not a measured value."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

G = 9.81
MIN_CRUISE_RPM = 1500.0  # gear choice avoids lugging the engine below this
TORQUE_MARGIN = 0.9  # a gear must need no more than this share of the full-load torque


@dataclass(frozen=True)
class VehicleSpec:
    mass_kg: float = 1300.0
    gear_ratios: tuple = (3.62, 1.96, 1.30, 0.97, 0.80)
    final_drive: float = 4.1
    wheel_radius_m: float = 0.31
    crr: float = 0.012
    cd_a: float = 0.84
    drivetrain_efficiency: float = 0.9
    air_density: float = 1.2
    idle_rpm: float = 800.0
    redline_rpm: float = 6000.0
    # assumed full-load torque curve (rpm, Nm): peak 157 Nm near 4850 rpm
    torque_curve: tuple = ((800.0, 100.0), (1500.0, 125.0), (2500.0, 145.0), (4850.0, 157.0), (6000.0, 140.0))

    def full_load_torque(self, rpm):
        rpms, torques = zip(*self.torque_curve)
        return np.interp(rpm, rpms, torques)


def wheel_power_w(speed_kmh, accel_ms2, spec: VehicleSpec):
    """Power at the wheels for inertia, rolling resistance and air drag (flat road, rotating inertia ignored)."""
    v = np.asarray(speed_kmh, dtype=float) / 3.6
    force = spec.mass_kg * accel_ms2 + spec.crr * spec.mass_kg * G + 0.5 * spec.air_density * spec.cd_a * v**2
    return np.where(v > 0, force * v, 0.0)[()]


def natural_rpm(speed_kmh, gear, spec) -> float:
    """Engine speed the road speed forces in `gear`, with no idle floor (below idle the clutch would have to slip)."""
    omega_wheel = speed_kmh / 3.6 / spec.wheel_radius_m
    return omega_wheel * spec.gear_ratios[gear - 1] * spec.final_drive * 60 / (2 * np.pi)


def _rpm(speed_kmh, gear, spec):
    return max(natural_rpm(speed_kmh, gear, spec), spec.idle_rpm)


def engine_point(speed_kmh, gear, power_w, spec: VehicleSpec):
    """(rpm, torque in Nm) in `gear` at `speed_kmh` when the wheels need `power_w`. Negative power (overrun) gives 0 Nm,
    and the engine never turns below idle."""
    rpm = _rpm(speed_kmh, gear, spec)
    engine_power = max(power_w, 0.0) / spec.drivetrain_efficiency
    return float(rpm), float(engine_power / (rpm * 2 * np.pi / 60))


def choose_gear(speed_kmh, power_w, spec: VehicleSpec) -> int:
    """Highest gear that keeps the engine between 1500 rpm and the redline with torque inside the full-load margin.
    If no gear fits, the one with the most torque headroom below the redline."""
    fallback, best_ratio = 1, np.inf
    for gear in range(len(spec.gear_ratios), 0, -1):
        rpm, torque = engine_point(speed_kmh, gear, power_w, spec)
        if rpm > spec.redline_rpm:
            continue
        used = torque / spec.full_load_torque(rpm)
        if rpm >= MIN_CRUISE_RPM and used <= TORQUE_MARGIN:
            return gear
        if used < best_ratio:
            fallback, best_ratio = gear, used
    return fallback


def demand_from_trace(speed_kmh, dt, spec: VehicleSpec) -> pd.DataFrame:
    """What the engine must deliver along a speed trace sampled every `dt` seconds."""
    speed = np.asarray(speed_kmh, dtype=float)
    accel = np.gradient(speed / 3.6, dt)
    power = np.array([wheel_power_w(v, a, spec) for v, a in zip(speed, accel)])
    gears = [choose_gear(v, p, spec) for v, p in zip(speed, power)]
    points = [engine_point(v, g, p, spec) for v, g, p in zip(speed, gears, power)]
    rpm, torque = np.array(points).T
    return pd.DataFrame(
        {
            "speed_kmh": speed, "accel_ms2": accel, "wheel_power_w": power, "gear": gears,
            "rpm": rpm, "torque_nm": torque, "engine_power_w": torque * rpm * 2 * np.pi / 60,
        }
    )
