from dataclasses import dataclass

import numpy as np

FAULTS = ("healthy", "misfire", "knock", "valve_wear", "misfire+knock", "misfire+valve_wear", "knock+valve_wear")
SOURCES = ("real", "simulated", "mixed")


@dataclass
class Window:
    sound: np.ndarray
    sound_fs: int
    vibration: np.ndarray
    vibration_fs: int
    obd: dict
    fault: str
    severity: float
    source: str

    def __post_init__(self):
        if self.fault not in FAULTS:
            raise ValueError(f"unknown fault label: {self.fault!r}")
        if self.source not in SOURCES:
            raise ValueError(f"unknown source tag: {self.source!r}")
        if not 0.0 <= self.severity <= 1.0:
            raise ValueError(f"severity must be between 0 and 1, got {self.severity}")

    @property
    def duration_s(self) -> float:
        return len(self.sound) / self.sound_fs
