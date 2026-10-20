"""Read-only generic OBD-II telemetry capture through python-OBD."""

from __future__ import annotations

import time
from datetime import datetime, timezone

import pandas as pd


PID_NAMES = (
    "RPM", "SPEED", "THROTTLE_POS", "RELATIVE_THROTTLE_POS", "COOLANT_TEMP",
    "INTAKE_TEMP", "INTAKE_PRESSURE", "ENGINE_LOAD", "MAF",
    "SHORT_FUEL_TRIM_1", "LONG_FUEL_TRIM_1", "TIMING_ADVANCE",
    "BAROMETRIC_PRESSURE", "CONTROL_MODULE_VOLTAGE",
)


def capture_obd(
    port: str | None,
    duration_seconds: int,
    sample_interval_seconds: float,
    progress=None,
) -> tuple[pd.DataFrame, list[str]]:
    """Capture only supported standard PIDs; never issue write/control commands."""
    try:
        import obd
    except ImportError as exc:
        raise RuntimeError("Install the optional OBD-II dependency with `python -m pip install obd`.") from exc

    connection = obd.OBD(portstr=port or None, fast=False, timeout=1)
    if not connection.is_connected():
        connection.close()
        raise RuntimeError("No OBD-II adapter connection was established. Check the adapter, port, ignition, and drivers.")
    commands = []
    for name in PID_NAMES:
        command = getattr(obd.commands, name, None)
        if command is not None and connection.supports(command):
            commands.append((name.lower(), command))
    if not commands:
        connection.close()
        raise RuntimeError("The vehicle did not report any supported standard sensor PIDs.")

    rows = []
    start = time.monotonic()
    try:
        while time.monotonic() - start < duration_seconds:
            sample_start = time.monotonic()
            row = {"timestamp_utc": datetime.now(timezone.utc).isoformat()}
            for name, command in commands:
                response = connection.query(command)
                value = response.value
                if value is None:
                    row[name] = float("nan")
                elif hasattr(value, "magnitude"):
                    row[name] = float(value.magnitude)
                else:
                    try:
                        row[name] = float(value)
                    except (TypeError, ValueError):
                        row[name] = str(value)
            rows.append(row)
            elapsed = time.monotonic() - start
            if progress:
                progress(elapsed, duration_seconds, len(rows))
            remaining = sample_interval_seconds - (time.monotonic() - sample_start)
            remaining_run = duration_seconds - elapsed
            if remaining > 0 and remaining_run > 0:
                time.sleep(min(remaining, remaining_run))
    finally:
        connection.close()
    return pd.DataFrame(rows), [name for name, _ in commands]
