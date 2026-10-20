import numpy as np

from enginetwin.schema import Window

FS = 16000  # one rate for sound and vibration: knock energy sits at 6-7 kHz, which needs fs > 14 kHz
FIRING_ORDER = (1, 3, 4, 2)  # inline-4; cylinder 1 fires at t = 0, then one firing every 180 degrees of the cycle
NOISE_SIGMA = 0.01
# OBD baselines and spreads below are calibrated to real Ocslab Kia Soul rows at ~2000 rpm / 40% load
# (see tests/test_simulator_calibration.py). The wide timing spread is real: it is why OBD alone detects knock poorly.
ENGINE_LITRES = 1.6  # matches the Kia Soul in the Ocslab data (about a 1.6L four-cylinder)
AIR_G_PER_L = 1.2

# (resonance Hz, decay seconds) of one combustion pulse as heard by the microphone / felt by the accelerometer
_SOUND_PULSE = (300.0, 0.004)
_VIBRATION_PULSE = (1200.0, 0.002)
_KNOCK_BURST = (6500.0, 0.0005)  # detonation rings the block at 6-7 kHz and dies out in about a millisecond
_KNOCK_AMPLITUDE = 0.5  # relative to a normal combustion pulse of 1.0
_VALVE_TICK = (3000.0, 0.0005)  # a worn valve clatters at ~3 kHz, well below knock's 6-7 kHz
_VALVE_AMPLITUDE = 1.2  # one tick per cycle now, so twice the old per-firing amplitude keeps the same tick energy
_VALVE_DELAY_DEG = 90.0  # crank degrees after each firing at which the valve event lands


def _pulse(resonance_hz: float, decay_s: float) -> np.ndarray:
    t = np.arange(int(5 * decay_s * FS)) / FS
    return np.exp(-t / decay_s) * np.sin(2 * np.pi * resonance_hz * t)


def _healthy_rms(rpm: float, pulse) -> float:
    """RMS of a healthy channel: one pulse exp(-t/tau) sin(2 pi f t) carries energy tau/4 and one firing arrives every 30/rpm s."""
    return float(np.sqrt(rpm / 30.0 * pulse[1] / 4.0))


def _firing_times(rpm: float, duration_s: float) -> np.ndarray:
    interval = 30.0 / rpm  # a 720-degree cycle takes 120/rpm s and holds 4 firings
    return np.arange(0.0, duration_s, interval)


def _add_pulses(signal, times, amplitudes, shape):
    for t, a in zip(times, amplitudes):
        start = int(round(t * FS))
        if start >= len(signal):
            continue  # the event happens after the window has ended
        end = min(start + len(shape), len(signal))
        signal[start:end] += a * shape[: end - start]


def _obd_readings(rpm, load, fault, severity, rng) -> dict:
    """ECU values with the names the real-data loaders use. Each fault has its own signature in the readings."""
    faults = fault.split("+")
    rpm_dip = 90.0 * severity if "misfire" in faults else 0.0  # lost firings slow the crank
    trim_rise = 8.0 * severity if "misfire" in faults else 0.0  # unburnt air reads lean; the ECU adds fuel
    timing_retard = 6.0 * severity if "knock" in faults else 0.0  # the ECU pulls timing back to stop detonation
    cycles_per_s = rpm / 120.0
    air_g_per_s = cycles_per_s * ENGINE_LITRES * AIR_G_PER_L * (0.3 + 0.6 * load)  # volumetric efficiency rises with load
    return {
        "engine_speed": rpm - rpm_dip + rng.normal(0, 8),
        "calculated_load_value": 100 * load + rng.normal(0, 1),
        "engine_coolant_temperature": 85.0 + rng.normal(0, 2.7),
        "maf": air_g_per_s * (1 + rng.normal(0, 0.02)),
        "timing_advance": 36.3 - 14.0 * load + 0.001 * (rpm - 2000) - timing_retard + rng.normal(0, 6.6),
        "short_term_fuel_trim_bank1": trim_rise + rng.normal(0, 2.9),
        "long_term_fuel_trim_bank1": 2.5 + rng.normal(0, 1.0),
    }


def simulate_window(
    rpm: float,
    fault: str = "healthy",
    severity: float = 0.0,
    seed: int = 0,
    duration_s: float = 2.0,
    cylinder: int = 2,
    load: float = 0.4,
    noise_sigma: float = NOISE_SIGMA,
    snr_db: float | None = None,
    worn_valves: tuple = (1,),
) -> Window:
    """One window of engine sound, vibration and OBD data. Cylinder 1 fires at t = 0, then 3, 4, 2 (180 degrees apart).

    `snr_db` sets each channel's noise from the RMS of a healthy engine at this rpm, so every fault gets the same
    noise level. When given it replaces `noise_sigma`.

    Valve wear ticks once per two revolutions for each worn valve; `worn_valves` names the one or two cylinders whose
    valve is worn."""
    if not 1 <= len(worn_valves) <= 2 or not set(worn_valves) <= {1, 2, 3, 4}:
        raise ValueError(f"worn_valves must be one or two cylinders from 1 to 4, got {worn_valves!r}")
    faults = fault.split("+")  # combined faults are written "misfire+knock"; each applies at the same severity
    rng = np.random.default_rng(seed)
    n = int(duration_s * FS)
    times = _firing_times(rpm, duration_s)
    firing_cylinder = np.array(FIRING_ORDER)[np.arange(len(times)) % 4]
    cylinder_gain = 1 + 0.03 * rng.standard_normal(4)  # slight imbalance between cylinders
    amplitudes = cylinder_gain[firing_cylinder - 1] * (1 + 0.05 * rng.standard_normal(len(times)))
    if "misfire" in faults:
        amplitudes = np.where(firing_cylinder == cylinder, amplitudes * (1 - severity), amplitudes)

    sound_sigma, vibration_sigma = noise_sigma, noise_sigma
    if snr_db is not None:
        sound_sigma = _healthy_rms(rpm, _SOUND_PULSE) / 10 ** (snr_db / 20)
        vibration_sigma = _healthy_rms(rpm, _VIBRATION_PULSE) / 10 ** (snr_db / 20)
    sound = rng.normal(0.0, sound_sigma, n)
    vibration = rng.normal(0.0, vibration_sigma, n)
    _add_pulses(sound, times, amplitudes, _pulse(*_SOUND_PULSE))
    _add_pulses(vibration, times, amplitudes, _pulse(*_VIBRATION_PULSE))

    if "knock" in faults:
        # Severity is the share of firings that detonate. Each burst rings at ~6.5 kHz, starting 10 deg after firing.
        knocking = rng.random(len(times)) < severity
        burst_times = times[knocking] + 10.0 / (6.0 * rpm)
        burst_amplitudes = _KNOCK_AMPLITUDE * rng.uniform(0.7, 1.3, knocking.sum())
        for signal in (sound, vibration):
            _add_pulses(signal, burst_times, burst_amplitudes, _pulse(*_KNOCK_BURST))

    if "valve_wear" in faults:
        # A worn valve opens once per cycle (two revolutions): one tick after its cylinder fires. Severity scales the slap.
        worn = np.isin(firing_cylinder, worn_valves)
        tick_times = times[worn] + _VALVE_DELAY_DEG / (6.0 * rpm)
        tick_amplitudes = _VALVE_AMPLITUDE * severity * rng.uniform(0.7, 1.3, worn.sum())
        for signal in (sound, vibration):
            _add_pulses(signal, tick_times, tick_amplitudes, _pulse(*_VALVE_TICK))

    return Window(
        sound=sound,
        sound_fs=FS,
        vibration=vibration,
        vibration_fs=FS,
        obd=_obd_readings(rpm, load, fault, severity, np.random.default_rng([seed, 1])),
        fault=fault,
        severity=severity,
        source="simulated",
    )
