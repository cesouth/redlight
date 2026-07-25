"""Fetch OSM road networks via the Overpass API.

Stdlib only (urllib): no new dependencies. Intended for small/medium study
areas -- for anything larger, download an extract and load it with
:meth:`roadtraffic.network.Network.from_geojson` / ``from_file`` instead
(Overpass is a shared public service with usage limits).

The entry point most users want is
:meth:`roadtraffic.network.Network.from_overpass`.
"""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request

from shapely.geometry import LineString

from .units import SpeedUnit, to_mps

DEFAULT_OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Unit suffixes accepted on an OSM ``maxspeed`` value, mapped to this package's
# canonical unit name. The empty suffix is the important one: OSM specifies a
# bare ``maxspeed=50`` as **km/h**, so reading it as mph (natural in a package
# whose reports default to mph) would overstate the limit by ~61%.
_MAXSPEED_UNITS = {"": "kph", "mph": "mph",
                   "km/h": "kph", "kmh": "kph", "kph": "kph"}

# A number, optional whitespace, optional alphabetic/slash unit suffix. Anything
# else (country codes like "RU:urban", multi-values like "50;30", "none",
# "walk", "signals") deliberately fails to match -- see parse_maxspeed.
_MAXSPEED_RE = re.compile(r"(\d+(?:\.\d+)?)\s*([a-z/]*)")

#: Drivable road classes (motorized traffic). Override via ``highway_regex``
#: to include/exclude classes (e.g. add ``track`` for off-road studies).
DEFAULT_HIGHWAY_REGEX = (
    "motorway|motorway_link|trunk|trunk_link|primary|primary_link|"
    "secondary|secondary_link|tertiary|tertiary_link|unclassified|"
    "residential|living_street|service"
)


def build_query(bbox, highway_regex: str | None = None,
                timeout: float = 90.0) -> str:
    """Build the Overpass QL query for drivable ways in a bounding box.

    Parameters
    ----------
    bbox : tuple of float
        ``(min_lon, min_lat, max_lon, max_lat)`` in WGS84 degrees. Note
        Overpass QL itself wants ``(south, west, north, east)`` order for a
        bounding box filter -- the reordering into that convention happens
        inside this function, so callers always use the same
        ``(min_lon, min_lat, max_lon, max_lat)`` convention as the rest of
        the package.
    highway_regex : str, optional
        Regex of OSM ``highway`` tag values to include (matched with
        ``^(...)$``). Defaults to :data:`DEFAULT_HIGHWAY_REGEX` (drivable
        road classes).
    timeout : float
        Value embedded in the query's own ``[timeout:N]`` setting (seconds),
        telling the Overpass server how long it may spend evaluating the
        query server-side. This is independent of (but should not exceed)
        the client-side HTTP timeout passed to :func:`fetch_network_records`.

    Returns
    -------
    str
        The Overpass QL query text.
    """
    min_lon, min_lat, max_lon, max_lat = (float(x) for x in bbox)
    if not (min_lon < max_lon and min_lat < max_lat):
        raise ValueError(
            "bbox must be (min_lon, min_lat, max_lon, max_lat) with "
            f"min < max; got {bbox!r}."
        )
    regex = highway_regex or DEFAULT_HIGHWAY_REGEX
    return (
        f"[out:json][timeout:{int(timeout)}];"
        f'way["highway"~"^({regex})$"]'
        f"({min_lat},{min_lon},{max_lat},{max_lon});"
        "out geom;"
    )


def parse_maxspeed(value) -> float | None:
    """Parse an OSM ``maxspeed`` tag value to metres per second.

    The posted **legal limit** -- not an observed speed. It is useful as a
    per-edge fallback for roads your GPS data never covered (see
    :class:`roadtraffic.routing.Router`), but it must never be mistaken for
    measured traffic: congestion is precisely the gap between the two.

    Parameters
    ----------
    value : str or float or None
        The raw tag value, e.g. ``"50"``, ``"35 mph"``, ``"50 km/h"``. Parsing
        is case- and whitespace-insensitive. Per the OSM specification a bare
        number means **km/h**; an explicit ``mph`` suffix overrides that.

    Returns
    -------
    float or None
        Speed in m/s, or ``None`` when the value carries no unambiguous
        number. Returning ``None`` rather than guessing is deliberate: OSM
        also uses ``"none"`` (no limit), ``"walk"`` (walking pace),
        ``"signals"``/``"variable"`` (dynamic), country defaults
        (``"RU:urban"``), and multi-values (``"50;30"``) -- each of which would
        need an invented constant or a country table to turn into a number.
        Non-positive values are rejected too, since a zero limit would divide
        by zero in any travel-time computation.
    """
    if value is None:
        return None
    match = _MAXSPEED_RE.fullmatch(str(value).strip().lower())
    if match is None:
        return None
    unit = _MAXSPEED_UNITS.get(match.group(2))
    if unit is None:                       # a real unit we don't convert (knots)
        return None
    speed = float(match.group(1))
    if speed <= 0:
        return None
    return float(to_mps(speed, SpeedUnit.parse(unit)))


def ways_to_records(overpass_json: dict) -> list:
    """Convert an Overpass ``out geom`` JSON response to network records.

    Parameters
    ----------
    overpass_json : dict
        Parsed JSON response from an Overpass ``out geom;`` query (as built by
        :func:`build_query`) -- a dict with an ``"elements"`` list, each
        element either a ``"way"`` (with an inline ``"geometry"`` list of
        ``{"lat": ..., "lon": ...}`` points and a ``"tags"`` dict) or another
        element type (nodes, relations), which is skipped.

    Returns
    -------
    list of (shapely.LineString, dict)
        One entry per way, geometry in WGS84 lon/lat -- the input format
        :meth:`roadtraffic.network.Network._build` expects. Way tags become
        the dict's properties, plus an ``osm_id`` key holding the way's OSM
        id. Ways with fewer than two geometry points (degenerate/malformed)
        are skipped. This function does no network I/O itself, so it is
        unit-testable offline against a fixture JSON response.
    """
    records = []
    for el in overpass_json.get("elements", []):
        if el.get("type") != "way":
            continue
        geom = el.get("geometry") or []
        coords = [(pt["lon"], pt["lat"]) for pt in geom
                  if pt and "lon" in pt and "lat" in pt]
        if len(coords) < 2:
            continue
        props = dict(el.get("tags") or {})
        props["osm_id"] = el.get("id")
        records.append((LineString(coords), props))
    return records


def fetch_network_records(bbox, *, highway_regex: str | None = None,
                          url: str | None = None,
                          timeout: float = 90.0) -> list:
    """Query Overpass and return network records for the bounding box.

    Parameters
    ----------
    bbox : tuple of float
        ``(min_lon, min_lat, max_lon, max_lat)`` in WGS84 degrees; see
        :func:`build_query`.
    highway_regex : str, optional
        Passed straight through to :func:`build_query`.
    url : str, optional
        Overpass endpoint to query. Default: the public
        ``overpass-api.de`` instance (:data:`DEFAULT_OVERPASS_URL`) -- pass
        your own mirror/self-hosted instance to avoid its shared usage limits
        for repeated or large queries.
    timeout : float
        Both the client-side HTTP request timeout (seconds, via
        ``urllib.request.urlopen``) and the value embedded in the query's
        own server-side ``[timeout:N]`` setting (see :func:`build_query`).

    Returns
    -------
    list of (shapely.LineString, dict)
        See :func:`ways_to_records`.
    """
    query = build_query(bbox, highway_regex, timeout)
    endpoint = url or DEFAULT_OVERPASS_URL
    payload = urllib.parse.urlencode({"data": query}).encode("utf-8")
    req = urllib.request.Request(
        endpoint, data=payload,
        headers={"User-Agent": "roadtraffic (+https://github.com/cesouth/roadtraffic)"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return ways_to_records(data)
