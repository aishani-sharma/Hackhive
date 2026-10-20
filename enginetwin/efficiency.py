import numpy as np

# Assumptions (stated on the dashboard): Ocslab's fuel column is inferred to be microlitres per second,
# cross-checked against idle burn and km/L. Petrol density and heating value are typical textbook figures.
PETROL_DENSITY_G_PER_ML = 0.745
PETROL_LHV_J_PER_KG = 43.4e6


def brake_power_w(torque_nm, rpm):
    return np.asarray(torque_nm) * np.asarray(rpm) * 2 * np.pi / 60


def _fuel_energy_rate_w(fuel_ul_per_s):
    kg_per_s = np.asarray(fuel_ul_per_s, dtype=float) * 1e-3 * PETROL_DENSITY_G_PER_ML * 1e-3
    return kg_per_s * PETROL_LHV_J_PER_KG


def bsfc_g_per_kwh(torque_nm, rpm, fuel_ul_per_s):
    grams_per_hour = np.asarray(fuel_ul_per_s, dtype=float) * 1e-3 * PETROL_DENSITY_G_PER_ML * 3600
    power_kw = brake_power_w(torque_nm, rpm) / 1000
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(power_kw > 0, grams_per_hour / power_kw, np.nan)[()]


def fuel_economy_km_per_l(speed_kmh, fuel_ul_per_s):
    litres_per_hour = np.asarray(fuel_ul_per_s, dtype=float) * 1e-6 * 3600
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(litres_per_hour > 0, np.asarray(speed_kmh) / litres_per_hour, np.nan)[()]


def brake_thermal_efficiency(torque_nm, rpm, fuel_ul_per_s):
    fuel_w = _fuel_energy_rate_w(fuel_ul_per_s)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(fuel_w > 0, brake_power_w(torque_nm, rpm) / fuel_w, np.nan)[()]
