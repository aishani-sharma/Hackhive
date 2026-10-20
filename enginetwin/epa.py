import pandas as pd

_COLUMNS = ["rpm", "torque", "fuel_g_per_s", "bsfc_g_per_kwh", "efficiency"]
_HEADER_ROWS = 6  # title, citation (2 rows), blank, column names, units


def load_epa_map(path) -> pd.DataFrame:
    """EPA ALPHA 'Fuel Map Data' sheet as one row per measured/modelled operating point (efficiency as a fraction)."""
    raw = pd.read_excel(path, header=None, skiprows=_HEADER_ROWS, usecols=range(len(_COLUMNS)))
    raw.columns = _COLUMNS
    raw = raw.dropna().astype(float).reset_index(drop=True)
    raw["efficiency"] = raw["efficiency"] / 100
    return raw
