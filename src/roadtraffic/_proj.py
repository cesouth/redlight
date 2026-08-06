"""Pure-numpy map projections.

Replaces PROJ for the three coordinate reference systems this package needs:
WGS84 geographic (EPSG:4326), the 120 WGS84 UTM zones (EPSG:326xx/327xx), and
Web Mercator (EPSG:3857). Anything else is left to the optional ``crs`` extra.

The UTM implementation is the Krueger series as given in Karney (2011),
"Transverse Mercator with an accuracy of a few nanometers". Truncated at n^5
it agrees with PROJ to under ten nanometres anywhere inside a zone -- some
nine orders of magnitude finer than the GPS noise this package absorbs.
"""
from __future__ import annotations

import math
import re

import numpy as np

EPSG_WGS84 = 4326
EPSG_WEB_MERCATOR = 3857

_A = 6378137.0                    # WGS84 semi-major axis (metres)
_F = 1 / 298.257223563            # WGS84 flattening
_E = math.sqrt(_F * (2 - _F))     # first eccentricity
_K0 = 0.9996                      # UTM scale factor on the central meridian
_FALSE_EASTING = 500000.0
_FALSE_NORTHING = 10000000.0      # southern hemisphere only

_EPSG_RE = re.compile(r"^\s*epsg\s*:\s*(\d+)\s*$", re.IGNORECASE)


def _kruger_series():
    """Krueger series coefficients for the WGS84 ellipsoid, to order n^5."""
    n = _F / (2 - _F)
    n2, n3, n4, n5, n6 = n**2, n**3, n**4, n**5, n**6
    a_bar = _A / (1 + n) * (1 + n2 / 4 + n4 / 64 + n6 / 256)
    alpha = (
        n / 2 - 2 * n2 / 3 + 5 * n3 / 16 + 41 * n4 / 180 - 127 * n5 / 288,
        13 * n2 / 48 - 3 * n3 / 5 + 557 * n4 / 1440 + 281 * n5 / 630,
        61 * n3 / 240 - 103 * n4 / 140 + 15061 * n5 / 26880,
        49561 * n4 / 161280 - 179 * n5 / 168,
        34729 * n5 / 80640,
    )
    beta = (
        n / 2 - 2 * n2 / 3 + 37 * n3 / 96 - n4 / 360 - 81 * n5 / 512,
        n2 / 48 + n3 / 15 - 437 * n4 / 1440 + 46 * n5 / 105,
        17 * n3 / 480 - 37 * n4 / 840 - 209 * n5 / 4480,
        4397 * n4 / 161280 - 11 * n5 / 504,
        4583 * n5 / 161280,
    )
    delta = (
        2 * n - 2 * n2 / 3 - 2 * n3 + 116 * n4 / 45,
        7 * n2 / 3 - 8 * n3 / 5 - 227 * n4 / 45,
        56 * n3 / 15 - 136 * n4 / 35,
        4279 * n4 / 630,
    )
    return a_bar, alpha, beta, delta


_A_BAR, _ALPHA, _BETA, _DELTA = _kruger_series()


def parse_epsg(crs) -> int | None:
    """Extract an EPSG code from a CRS string, or None if it has no code.

    pyogrio reports ``"EPSG:32633"`` when the file carries an authority code
    and raw WKT when it does not. Only the former can be handled natively;
    returning None is the caller's signal to fall back to pyproj.
    """
    if not crs:
        return None
    match = _EPSG_RE.match(str(crs))
    return int(match.group(1)) if match else None


def utm_epsg_to_zone(epsg: int) -> tuple[int, bool]:
    """Split a WGS84 UTM EPSG code into ``(zone, is_northern)``.

    Raises
    ------
    ValueError
        If ``epsg`` is not one of the 120 WGS84 UTM codes. Callers use this to
        decide whether pyproj is needed.
    """
    epsg = int(epsg)
    if 32601 <= epsg <= 32660:
        return epsg - 32600, True
    if 32701 <= epsg <= 32760:
        return epsg - 32700, False
    raise ValueError(
        f"EPSG:{epsg} is not a WGS84 UTM zone (expected 32601-32660 for the "
        f"northern hemisphere or 32701-32760 for the southern)."
    )


def utm_forward(lon, lat, zone: int, north: bool):
    """WGS84 lon/lat in degrees -> UTM easting/northing in metres."""
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    lon0 = math.radians(zone * 6 - 183)
    phi = np.radians(lat)
    lam = np.radians(lon) - lon0
    sin_phi = np.sin(phi)
    t = np.sinh(np.arctanh(sin_phi) - _E * np.arctanh(_E * sin_phi))
    xi_p = np.arctan2(t, np.cos(lam))
    # arctanh, not arcsinh: this is the conformal-latitude term, and the two
    # agree near the central meridian but diverge to hundreds of metres at the
    # zone edge, where only the easting is affected.
    eta_p = np.arctanh(np.sin(lam) / np.hypot(1.0, t))
    xi = xi_p + sum(
        a * np.sin(2 * (j + 1) * xi_p) * np.cosh(2 * (j + 1) * eta_p)
        for j, a in enumerate(_ALPHA)
    )
    eta = eta_p + sum(
        a * np.cos(2 * (j + 1) * xi_p) * np.sinh(2 * (j + 1) * eta_p)
        for j, a in enumerate(_ALPHA)
    )
    x = _K0 * _A_BAR * eta + _FALSE_EASTING
    y = _K0 * _A_BAR * xi + (0.0 if north else _FALSE_NORTHING)
    return x, y


def utm_inverse(x, y, zone: int, north: bool):
    """UTM easting/northing in metres -> WGS84 lon/lat in degrees."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    lon0 = math.radians(zone * 6 - 183)
    xi = (y - (0.0 if north else _FALSE_NORTHING)) / (_K0 * _A_BAR)
    eta = (x - _FALSE_EASTING) / (_K0 * _A_BAR)
    xi_p = xi - sum(
        b * np.sin(2 * (j + 1) * xi) * np.cosh(2 * (j + 1) * eta)
        for j, b in enumerate(_BETA)
    )
    eta_p = eta - sum(
        b * np.cos(2 * (j + 1) * xi) * np.sinh(2 * (j + 1) * eta)
        for j, b in enumerate(_BETA)
    )
    chi = np.arcsin(np.clip(np.sin(xi_p) / np.cosh(eta_p), -1.0, 1.0))
    phi = chi + sum(d * np.sin(2 * (j + 1) * chi) for j, d in enumerate(_DELTA))
    lam = np.arctan2(np.sinh(eta_p), np.cos(xi_p))
    return np.degrees(lam + lon0), np.degrees(phi)


def web_mercator_inverse(x, y):
    """EPSG:3857 easting/northing in metres -> WGS84 lon/lat in degrees."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    lon = np.degrees(x / _A)
    lat = np.degrees(2.0 * np.arctan(np.exp(y / _A)) - math.pi / 2.0)
    return lon, lat


class UtmCrs:
    """Minimal stand-in for ``pyproj.CRS``, covering what ``Network`` exposes.

    ``Network.crs_metric`` is public API and callers use ``.to_epsg()`` on it,
    so the attribute cannot simply become an int.
    """

    __slots__ = ("_epsg", "zone", "north")

    def __init__(self, epsg: int):
        self.zone, self.north = utm_epsg_to_zone(epsg)
        self._epsg = int(epsg)

    def to_epsg(self) -> int:
        """The EPSG code, e.g. ``32633``."""
        return self._epsg

    @property
    def name(self) -> str:
        """Human-readable CRS name, e.g. ``"WGS 84 / UTM zone 33N"``."""
        return f"WGS 84 / UTM zone {self.zone}{'N' if self.north else 'S'}"

    def __repr__(self) -> str:
        return f"<UtmCrs EPSG:{self._epsg} {self.name}>"

    def __eq__(self, other) -> bool:
        return isinstance(other, UtmCrs) and other.to_epsg() == self._epsg

    def __hash__(self) -> int:
        return hash(("UtmCrs", self._epsg))


class UtmTransformer:
    """Minimal stand-in for ``pyproj.Transformer`` in one fixed direction."""

    __slots__ = ("zone", "north", "_inverse")

    def __init__(self, zone: int, north: bool, *, inverse: bool):
        self.zone = zone
        self.north = north
        self._inverse = inverse

    def transform(self, xx, yy):
        """``(lon, lat) -> (x, y)``, or ``(x, y) -> (lon, lat)`` if inverse."""
        if self._inverse:
            return utm_inverse(xx, yy, self.zone, self.north)
        return utm_forward(xx, yy, self.zone, self.north)


def utm_crs_and_transformers(epsg: int):
    """Return ``(UtmCrs, forward, inverse)`` for a WGS84 UTM EPSG code."""
    zone, north = utm_epsg_to_zone(epsg)
    return (UtmCrs(epsg),
            UtmTransformer(zone, north, inverse=False),
            UtmTransformer(zone, north, inverse=True))
