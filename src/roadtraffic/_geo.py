"""Shared geodesic machinery.

One WGS84 ellipsoid definition for the whole package, so every module measures
ground distance with the same model.
"""
from __future__ import annotations

from pyproj import Geod

GEOD_WGS84 = Geod(ellps="WGS84")
