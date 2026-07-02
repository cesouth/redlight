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
import urllib.parse
import urllib.request

from shapely.geometry import LineString

DEFAULT_OVERPASS_URL = "https://overpass-api.de/api/interpreter"

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

    ``bbox`` is ``(min_lon, min_lat, max_lon, max_lat)`` (WGS84); note Overpass
    itself wants ``(south, west, north, east)`` -- the reordering happens here.
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


def ways_to_records(overpass_json: dict) -> list:
    """Convert an Overpass ``out geom`` JSON response to network records.

    Returns a list of ``(shapely.LineString in WGS84, properties dict)`` --
    the input format of the internal network builder. Way tags become
    properties, plus ``osm_id``. Ways with fewer than two geometry points are
    skipped. This function is pure (no network access), so it is unit-testable
    offline.
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

    See :func:`build_query` for the bbox convention and
    :func:`ways_to_records` for the return format.
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
