"""UTM and Web Mercator, without PROJ.

Reference values are pinned rather than computed against pyproj, so this suite
proves the projections in an environment where pyproj is not installed.
"""
from __future__ import annotations

import numpy as np
import pytest

from roadtraffic import _proj

# (lon, lat, epsg, easting, northing). Generated from PROJ 9.5.1.
UTM_REFERENCE = [
    (15.0, 50.0, 32633, 500000.000000, 5538630.702867),        # zone 33N
    (-122.4194, 37.7749, 32610, 551130.768481, 4180998.881499),  # zone 10N
    (151.2093, -33.8688, 32756, 334368.633648, 6250948.345385),  # zone 56S
    (0.0, 0.0, 32631, 166021.443081, 0.000000),                # zone 31N
    (-74.006, 40.7128, 32618, 583959.372324, 4507350.998243),  # zone 18N
]


@pytest.mark.parametrize("lon,lat,epsg,east,north", UTM_REFERENCE)
def test_utm_forward_matches_reference(lon, lat, epsg, east, north):
    """Sub-millimetre agreement with PROJ, north and south, across zones."""
    zone, is_north = _proj.utm_epsg_to_zone(epsg)
    x, y = _proj.utm_forward(lon, lat, zone, is_north)
    assert float(x) == pytest.approx(east, abs=1e-3)
    assert float(y) == pytest.approx(north, abs=1e-3)


@pytest.mark.parametrize("lon,lat,epsg,east,north", UTM_REFERENCE)
def test_utm_inverse_matches_reference(lon, lat, epsg, east, north):
    """The inverse recovers the original degrees from PROJ's own eastings."""
    zone, is_north = _proj.utm_epsg_to_zone(epsg)
    got_lon, got_lat = _proj.utm_inverse(east, north, zone, is_north)
    # 1e-10 deg is ~11 um. The floor here is the reference itself: the eastings
    # above are pinned to a micrometre, i.e. ~5e-12 deg of rounding. Observed
    # residual across these five points is at most 4.2e-12 deg.
    assert float(got_lon) == pytest.approx(lon, abs=1e-10)
    assert float(got_lat) == pytest.approx(lat, abs=1e-10)


def test_utm_roundtrip_across_a_full_zone_width():
    """Round-trip error stays sub-millimetre over the whole legal zone, not
    just near the central meridian where the series is most accurate."""
    rng = np.random.default_rng(0)
    lon = 15.0 + rng.uniform(-3, 3, 2000)
    lat = rng.uniform(0.5, 84.0, 2000)
    x, y = _proj.utm_forward(lon, lat, 33, True)
    got_lon, got_lat = _proj.utm_inverse(x, y, 33, True)
    # 1e-11 deg is ~1.1 um at the equator. Observed worst case over these 2000
    # points is 1.3e-13 deg, so this bound holds with ~80x of margin; it is set
    # this tight deliberately, because a truncated series is exactly what a
    # loose round-trip tolerance would hide.
    assert np.max(np.abs(got_lon - lon)) < 1e-11
    assert np.max(np.abs(got_lat - lat)) < 1e-11


def test_zone_60_inverse_wraps_across_the_antimeridian():
    """Zone 60 straddles 180 deg, so its western half comes out of the series
    above +180. Longitudes must be wrapped back into [-180, 180): node keys
    have to match the source file's own coordinates, and GeoJSON output above
    180 is invalid per RFC 7946."""
    x, y = _proj.utm_forward(-179.0, 60.0, 60, True)
    lon, lat = _proj.utm_inverse(x, y, 60, True)
    assert float(lon) == pytest.approx(-179.0, abs=1e-10)
    assert float(lat) == pytest.approx(60.0, abs=1e-10)

    # Vectorised, and on both sides of the seam.
    lons = np.array([-179.5, -177.0, 177.0, 179.5])
    xs, ys = _proj.utm_forward(lons, np.full(4, -40.0), 60, False)
    got_lon, got_lat = _proj.utm_inverse(xs, ys, 60, False)
    assert np.all(got_lon >= -180.0) and np.all(got_lon < 180.0)
    assert np.allclose(got_lon, lons, atol=1e-10)
    assert np.allclose(got_lat, -40.0, atol=1e-10)


def test_southern_hemisphere_applies_the_false_northing():
    """Southern zones offset by 10,000 km so northings stay positive. Getting
    this backwards is the classic UTM bug and is invisible near the equator."""
    x, y = _proj.utm_forward(151.2093, -33.8688, 56, False)
    assert 6_000_000 < float(y) < 7_000_000


def test_utm_epsg_to_zone_maps_both_hemispheres():
    assert _proj.utm_epsg_to_zone(32633) == (33, True)
    assert _proj.utm_epsg_to_zone(32756) == (56, False)
    assert _proj.utm_epsg_to_zone(32601) == (1, True)
    assert _proj.utm_epsg_to_zone(32760) == (60, False)


@pytest.mark.parametrize("epsg", [4326, 3857, 27700, 32600, 32661, 32700, 32761])
def test_utm_epsg_to_zone_rejects_non_utm(epsg):
    """Codes outside the two UTM blocks must raise, so callers can fall back
    to pyproj rather than silently projecting into the wrong zone."""
    with pytest.raises(ValueError, match="not a WGS84 UTM zone"):
        _proj.utm_epsg_to_zone(epsg)


def test_web_mercator_inverse_matches_reference():
    """EPSG:3857 for 15E 50N, from PROJ 9.5.1."""
    lon, lat = _proj.web_mercator_inverse(1669792.3618991035, 6446275.841017158)
    assert float(lon) == pytest.approx(15.0, abs=1e-9)
    assert float(lat) == pytest.approx(50.0, abs=1e-9)


@pytest.mark.parametrize("raw,expected", [
    ("EPSG:4326", 4326),
    ("EPSG:32633", 32633),
    ("epsg:3857", 3857),
    ("  EPSG:27700  ", 27700),
    (None, None),
    ("", None),
    ('PROJCS["OSGB 1936 / British National Grid",GEOGCS[...]]', None),
])
def test_parse_epsg(raw, expected):
    """pyogrio hands back 'EPSG:NNNN' when the file carries an authority code,
    and raw WKT when it does not. Only the former can be handled natively."""
    assert _proj.parse_epsg(raw) == expected


# What GDAL/QGIS/ogr2ogr stamp on a CRS84 file, as pyogrio reports it back.
CRS84_WKT = (
    'GEOGCS["WGS 84 (CRS84)",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,'
    '298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],'
    'PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",'
    '0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Longitude",EAST],'
    'AXIS["Latitude",NORTH],AUTHORITY["OGC","CRS84"]]'
)

# A legal WKT2 PROJCRS whose BASEGEOGCRS happens to carry the CRS84 id. The
# string as a whole denotes a projected CRS in metres, not lon/lat degrees --
# the nested tag describes what the projection was built on, not what this
# CRS is. Regression case: an unanchored regex matches the nested node and
# misclassifies the whole thing as WGS84, so downstream code applies no
# transform to coordinates that are actually eastings/northings.
BAD_PROJCRS_WKT = (
    'PROJCRS["Fictitious Grid built on a CRS84 base",'
    'BASEGEOGCRS["WGS 84",ID["OGC","CRS84"]],'
    'CONVERSION["UTM zone 33N",METHOD["Transverse Mercator"]],'
    'CS[Cartesian,2],AXIS["easting",east],AXIS["northing",north],'
    'LENGTHUNIT["metre",1],ID["EPSG",99999]]'
)

# Same class of bug, different wrapper: a BOUNDCRS's SOURCECRS can also carry
# a nested CRS84 id without the BOUNDCRS itself being plain WGS84 lon/lat.
BOUNDCRS_WKT = (
    'BOUNDCRS[SOURCECRS[GEOGCRS["WGS 84",ID["OGC","CRS84"]]],'
    'TARGETCRS[GEOGCRS["WGS 84",ID["EPSG",4326]]],'
    'ABRIDGEDTRANSFORMATION["Null geographic offset",'
    'METHOD["Geographic2D offsets"],'
    'PARAMETER["Longitude offset",0],PARAMETER["Latitude offset",0]]]'
)


@pytest.mark.parametrize("raw,expected", [
    ("EPSG:4326", True),
    ("EPSG:4979", True),      # WGS84 3D; identical horizontally
    ("OGC:CRS84", True),
    (CRS84_WKT, True),
    ('GEOGCRS["WGS 84 (CRS84)",ID["OGC","CRS84"]]', True),  # WKT2 spelling
    ("EPSG:32633", False),
    ("EPSG:3857", False),
    ("EPSG:27700", False),
    ('PROJCS["OSGB 1936 / British National Grid",GEOGCS[...]]', False),
    (BAD_PROJCRS_WKT, False),   # nested CRS84 id inside a PROJCRS's base
    (BOUNDCRS_WKT, False),      # nested CRS84 id inside a BOUNDCRS's source
    (None, False),
    ("", False),
])
def test_is_wgs84(raw, expected):
    """CRS84 carries no EPSG code, so parse_epsg cannot see it -- yet it is the
    single most common spelling in exported GeoJSON and needs no transform."""
    assert _proj.is_wgs84(raw) is expected


def test_is_wgs84_real_gdal_crs84_wkt(tmp_path):
    """The regression fix keys is_wgs84 off the outermost WKT keyword -- make
    sure that still recognises the WKT GDAL itself writes for CRS84, not just
    a hand-pasted approximation of it."""
    pytest.importorskip("pyogrio")
    import numpy as np
    import pyogrio
    import pyogrio.raw
    import shapely

    path = tmp_path / "crs84.gpkg"
    geoms = np.array([shapely.LineString([[15.0, 50.0], [15.01, 50.0]])], dtype=object)
    wkb = shapely.to_wkb(geoms)
    pyogrio.raw.write(str(path), wkb, (np.array(["x"]),), ["name"],
                      geometry_type="LineString", crs="OGC:CRS84")
    crs = pyogrio.read_info(str(path))["crs"]
    assert _proj.is_wgs84(crs) is True


def test_utm_crs_shim_exposes_to_epsg():
    """Network.crs_metric.to_epsg() is public API, used in docs, examples and
    tests. The shim must satisfy it without pyproj."""
    crs, fwd, inv = _proj.utm_crs_and_transformers(32633)
    assert crs.to_epsg() == 32633
    assert crs.name == "WGS 84 / UTM zone 33N"
    assert crs == _proj.UtmCrs(32633)


def test_transformer_shim_roundtrips_arrays():
    """net._transformer_fwd/_inv .transform(x, y) is used outside network.py."""
    _crs, fwd, inv = _proj.utm_crs_and_transformers(32633)
    x, y = fwd.transform(np.array([15.0, 15.5]), np.array([50.0, 50.5]))
    lon, lat = inv.transform(x, y)
    assert np.allclose(lon, [15.0, 15.5], atol=1e-9)
    assert np.allclose(lat, [50.0, 50.5], atol=1e-9)


def test_transformer_shim_accepts_scalars():
    """tests/test_derive_math.py calls float() on the result of a scalar
    transform, which pyproj supported."""
    _crs, fwd, inv = _proj.utm_crs_and_transformers(32633)
    x, y = fwd.transform(15.0, 50.0)
    assert float(x) == pytest.approx(500000.0, abs=1e-3)
    lon, lat = inv.transform(float(x), float(y))
    assert float(lon) == pytest.approx(15.0, abs=1e-9)
