"""Speed unit handling.

All internal computation is done in metres per second (m/s). Inputs may be in
mph, kph, or m/s; this module converts to and from the canonical m/s.
"""
from __future__ import annotations

import math
from enum import Enum

# Exact conversion factors (1 mile = 1609.344 m, 1 km = 1000 m).
_MPH_TO_MPS = 1609.344 / 3600.0
_KPH_TO_MPS = 1000.0 / 3600.0


# Every string alias accepted for each canonical unit, mapped to the canonical
# short code. Shared with points.py's column-name unit inference so both places
# recognize the exact same set of aliases (see _infer_speed_unit).
_SPEED_UNIT_ALIASES: dict[str, str] = {
    "mph": "mph", "mi/h": "mph", "miles_per_hour": "mph",
    "kph": "kph", "km/h": "kph", "kmh": "kph", "kmph": "kph",
    "kilometers_per_hour": "kph", "kilometres_per_hour": "kph",
    "mps": "mps", "m/s": "mps", "ms": "mps",
    "meters_per_second": "mps", "metres_per_second": "mps",
}


class SpeedUnit(str, Enum):
    """Supported speed units for input/output.

    A ``str`` subclass, so a member compares equal to its own value
    (``SpeedUnit.MPH == "mph"``) and can be used anywhere a plain unit string
    is expected without an explicit ``.value`` access.

    Members
    -------
    MPH : miles per hour.
    KPH : kilometres per hour.
    MPS : metres per second -- the canonical internal unit every computation
        in this package is done in; input/output in the other two units is
        converted to/from this one at the boundary (:func:`to_mps` /
        :func:`from_mps`).
    """

    MPH = "mph"
    KPH = "kph"
    MPS = "mps"

    @classmethod
    def parse(cls, value) -> SpeedUnit:
        """Resolve a unit string (or an existing ``SpeedUnit``) to a member.

        Parameters
        ----------
        value : str or SpeedUnit
            A canonical code (``"mph"``/``"kph"``/``"mps"``) or any alias in
            ``_SPEED_UNIT_ALIASES`` (e.g. ``"km/h"``, ``"kmph"``, ``"m/s"``),
            case-insensitive with surrounding whitespace stripped. Passing an
            existing ``SpeedUnit`` returns it unchanged (idempotent, so
            callers never need to check the type before calling this).

        Returns
        -------
        SpeedUnit

        Raises
        ------
        ValueError
            If ``value`` doesn't match any known unit or alias.
        """
        if isinstance(value, cls):
            return value
        v = str(value).strip().lower()
        if v not in _SPEED_UNIT_ALIASES:
            raise ValueError(
                f"Unknown speed unit {value!r}. Use one of: mph, kph, mps."
            )
        return cls(_SPEED_UNIT_ALIASES[v])


def to_mps(value, unit: SpeedUnit):
    """Convert a speed from ``unit`` to m/s, the package's canonical unit.

    Parameters
    ----------
    value : float or array-like
        Speed(s) expressed in ``unit``. Array-like inputs (numpy arrays,
        pandas Series) work unchanged since the conversion is a single
        multiply/divide, which numpy/pandas broadcast elementwise.
    unit : str or SpeedUnit
        The unit ``value`` is currently in; resolved via
        :meth:`SpeedUnit.parse`, so any accepted alias works here too.

    Returns
    -------
    float or array-like
        ``value`` converted to m/s, same shape/type as the input.
    """
    unit = SpeedUnit.parse(unit)
    if unit is SpeedUnit.MPH:
        return value * _MPH_TO_MPS
    if unit is SpeedUnit.KPH:
        return value * _KPH_TO_MPS
    return value  # already m/s


def from_mps(value, unit: SpeedUnit):
    """Convert a speed from m/s to ``unit`` (the inverse of :func:`to_mps`).

    Parameters
    ----------
    value : float or array-like
        Speed(s) in m/s.
    unit : str or SpeedUnit
        The unit to convert into; resolved via :meth:`SpeedUnit.parse`.

    Returns
    -------
    float or array-like
        ``value`` converted to ``unit``, same shape/type as the input.
    """
    unit = SpeedUnit.parse(unit)
    if unit is SpeedUnit.MPH:
        return value / _MPH_TO_MPS
    if unit is SpeedUnit.KPH:
        return value / _KPH_TO_MPS
    return value


def _usable_speed(value) -> float | None:
    """``value`` as a positive finite speed in m/s, or ``None`` if unusable.

    Screens speeds that arrive from outside this package -- a caller's own
    GeoJSON/Shapefile column, or a hand-set edge attribute -- before they are
    divided by or divided into. A plain truthiness test is not enough:
    **NaN is truthy**, so it would pass and then silently turn every number
    derived from it into NaN; infinity would instead make a travel time zero
    or a congestion ratio zero.

    Uses ``float()`` rather than ``isinstance(value, float)`` because numpy
    scalars (``numpy.float32``) are not Python float subclasses but are
    perfectly good speeds -- the same trap that let a float32 NaN past
    ``analysis._invalid_weight``.
    """
    try:
        speed = float(value)
    except (TypeError, ValueError):
        return None
    return speed if math.isfinite(speed) and speed > 0 else None
