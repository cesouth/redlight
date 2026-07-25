"""Offline tests of the Overpass loader (no network access)."""
import pytest

import roadtraffic as rt
from roadtraffic import osm

CANNED = {
    "version": 0.6,
    "elements": [
        {"type": "way", "id": 101,
         "tags": {"highway": "residential", "name": "A St", "oneway": "yes"},
         "geometry": [{"lat": 0.0, "lon": 0.0}, {"lat": 0.0, "lon": 0.001}]},
        {"type": "way", "id": 102,
         "tags": {"highway": "service"},
         "geometry": [{"lat": 0.0, "lon": 0.001}, {"lat": 0.001, "lon": 0.001}]},
        {"type": "way", "id": 103, "tags": {"highway": "residential"},
         "geometry": [{"lat": 5.0, "lon": 5.0}]},          # too short: skipped
        {"type": "node", "id": 1, "lat": 0.0, "lon": 0.0},  # not a way: skipped
    ],
}


def test_ways_to_records():
    records = osm.ways_to_records(CANNED)
    assert len(records) == 2
    geom, props = records[0]
    assert props["osm_id"] == 101
    assert props["name"] == "A St"
    assert list(geom.coords) == [(0.0, 0.0), (0.001, 0.0)]


def test_records_build_network():
    records = osm.ways_to_records(CANNED)
    net = rt.Network._build(records, None, True, "oneway", None)
    # way 101 is oneway (1 edge), way 102 two-way (2 edges)
    assert net.number_of_edges() == 3
    d = net.edge_data(int(net.edge_ids[0]))
    assert d["highway"] in {"residential", "service"}


def test_build_query_contents_and_validation():
    q = osm.build_query((-77.3, 38.6, -77.2, 38.7), timeout=25)
    assert "[out:json]" in q and "out geom;" in q
    assert "(38.6,-77.3,38.7,-77.2)" in q  # Overpass wants south,west,north,east
    assert '["highway"~' in q
    with pytest.raises(ValueError, match="bbox"):
        osm.build_query((1, 1, 0, 0))


def test_build_query_custom_regex():
    q = osm.build_query((0, 0, 1, 1), highway_regex="track")
    assert "^(track)$" in q


# --------------------------------------------------------------- maxspeed
# OSM's maxspeed tag: a bare number means km/h; an explicit unit suffix wins.

def test_parse_maxspeed_bare_number_is_kph():
    assert osm.parse_maxspeed("50") == pytest.approx(50 * 1000 / 3600)


def test_parse_maxspeed_mph_suffix():
    assert osm.parse_maxspeed("35 mph") == pytest.approx(35 * 1609.344 / 3600)


def test_parse_maxspeed_kph_spellings_agree():
    kph = osm.parse_maxspeed("50")
    for text in ("50 km/h", "50 kmh", "50 kph", "50km/h", " 50 KM/H "):
        assert osm.parse_maxspeed(text) == pytest.approx(kph)


def test_parse_maxspeed_accepts_numeric_input():
    assert osm.parse_maxspeed(50) == pytest.approx(50 * 1000 / 3600)


@pytest.mark.parametrize("value", [
    None, "", "   ",
    "none",           # autobahn: no limit, not a number
    "walk",           # 'walking pace' -- no defensible number
    "signals", "variable",
    "RU:urban",       # implicit country default, needs a country table
    "50;30",          # multi-value, ambiguous
    "fast",
    "0", "-20",       # non-positive: would divide by zero / reverse time
])
def test_parse_maxspeed_unparseable_returns_none(value):
    assert osm.parse_maxspeed(value) is None
