"""Speed unit handling.

All internal computation is done in metres per second (m/s). Inputs may be in
mph, kph, or m/s; this module converts to and from the canonical m/s.
"""
from __future__ import annotations

from enum import Enum

# Exact conversion factors (1 mile = 1609.344 m, 1 km = 1000 m).
_MPH_TO_MPS = 1609.344 / 3600.0
_KPH_TO_MPS = 1000.0 / 3600.0


class SpeedUnit(str, Enum):
    """Supported speed units for input/output."""

    MPH = "mph"
    KPH = "kph"
    MPS = "mps"

    @classmethod
    def parse(cls, value) -> "SpeedUnit":
        if isinstance(value, cls):
            return value
        v = str(value).strip().lower()
        aliases = {
            "mph": cls.MPH, "mi/h": cls.MPH, "miles_per_hour": cls.MPH,
            "kph": cls.KPH, "km/h": cls.KPH, "kmh": cls.KPH, "kmph": cls.KPH,
            "kilometers_per_hour": cls.KPH,
            "mps": cls.MPS, "m/s": cls.MPS, "ms": cls.MPS,
            "meters_per_second": cls.MPS, "metres_per_second": cls.MPS,
        }
        if v not in aliases:
            raise ValueError(
                f"Unknown speed unit {value!r}. Use one of: mph, kph, mps."
            )
        return aliases[v]


def to_mps(value, unit: SpeedUnit):
    """Convert a speed (scalar or array-like) from ``unit`` to m/s."""
    unit = SpeedUnit.parse(unit)
    if unit is SpeedUnit.MPH:
        return value * _MPH_TO_MPS
    if unit is SpeedUnit.KPH:
        return value * _KPH_TO_MPS
    return value  # already m/s


def from_mps(value, unit: SpeedUnit):
    """Convert a speed (scalar or array-like) from m/s to ``unit``."""
    unit = SpeedUnit.parse(unit)
    if unit is SpeedUnit.MPH:
        return value / _MPH_TO_MPS
    if unit is SpeedUnit.KPH:
        return value / _KPH_TO_MPS
    return value
