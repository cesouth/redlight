"""Ellipsoidal distance, without PROJ.

Reference values are pinned rather than computed against pyproj, so this suite
proves the geodesy in an environment where pyproj is not installed -- which is
the entire point of the module under test.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from redlight._geo import geodesic_distance

# (lon1, lat1, lon2, lat2, metres). Generated from PROJ 9.5.1's geodesic
# (Karney), which is the reference implementation these values must match.
REFERENCE = [
    (0.0, 0.0, 1.0, 0.0, 111319.490793),          # 1 deg along the equator
    (15.0, 50.0, 15.01, 50.0, 716.957536),        # ~715 m east, mid-latitude
    (-74.006, 40.7128, -0.1276, 51.5072, 5585253.769594),   # New York -> London
    (0.0, 0.0, 0.0, 1.0, 110574.388558),          # 1 deg along a meridian
    (151.2093, -33.8688, 151.21, -33.869, 68.462342),       # ~65 m, southern
]


@pytest.mark.parametrize("lon1,lat1,lon2,lat2,expected", REFERENCE)
def test_matches_reference_values(lon1, lat1, lon2, lat2, expected):
    """Agreement with PROJ to well under a millimetre."""
    got = float(geodesic_distance(lon1, lat1, lon2, lat2))
    assert got == pytest.approx(expected, abs=1e-3)


def test_matches_published_vincenty_line():
    """Geoscience Australia's published Vincenty test line, Flinders Peak to
    Buninyong: 54972.271 m. An independent check that does not trace back to
    PROJ, so a shared misreading of the ellipsoid cannot hide here."""
    got = float(geodesic_distance(144.42486781, -37.95103342,
                                  143.92649552, -37.65282114))
    assert got == pytest.approx(54972.271, abs=0.01)


def test_vectorises_elementwise():
    """Array inputs give one distance per row, matching the scalar answer."""
    lon1 = np.array([0.0, 15.0])
    lat1 = np.array([0.0, 50.0])
    lon2 = np.array([1.0, 15.01])
    lat2 = np.array([0.0, 50.0])
    got = geodesic_distance(lon1, lat1, lon2, lat2)
    assert got.shape == (2,)
    assert got[0] == pytest.approx(111319.490793, abs=1e-3)
    assert got[1] == pytest.approx(716.957536, abs=1e-3)


def test_identical_points_are_zero():
    """Coincident points must give exactly 0.0, not NaN. The formula divides
    by sin(sigma), which is 0 here; the guard for that is load-bearing."""
    assert float(geodesic_distance(5.0, 50.0, 5.0, 50.0)) == 0.0


def test_symmetric():
    """Distance does not depend on argument order."""
    a = float(geodesic_distance(15.0, 50.0, 16.0, 51.0))
    b = float(geodesic_distance(16.0, 51.0, 15.0, 50.0))
    assert a == pytest.approx(b, abs=1e-6)


def test_antipodal_raises_rather_than_returning_a_wrong_number():
    """Vincenty does not converge for near-antipodal pairs. The unconverged
    value is wrong by kilometres and looks entirely plausible, so it must
    never be returned silently."""
    with pytest.raises(ValueError, match="antipodal"):
        geodesic_distance(0.0, 0.0, 180.0, 0.0)


def test_realistic_gps_tick_scale():
    """The dominant real workload: consecutive pings a few metres apart."""
    got = float(geodesic_distance(15.0, 50.0, 15.00005, 50.00005))
    assert 3.0 < got < 10.0
    assert math.isfinite(got)


def test_non_finite_coordinates_give_nan_not_antipodal_error():
    """NaN in, NaN out -- and no claim about antipodal points.

    Regression for F-3.5: a non-finite coordinate never satisfies the
    convergence test, so it used to raise the near-antipodal ValueError, which
    names the wrong cause. One bad pair also aborted every good pair in the
    same vectorised call.
    """
    assert np.isnan(geodesic_distance(0.0, np.nan, 1.0, 1.0))
    assert np.isnan(geodesic_distance(np.nan, 0.0, 1.0, 1.0))
    assert np.isnan(geodesic_distance(0.0, np.inf, 1.0, 1.0))

    # a single bad pair must not poison its neighbours
    lon1 = np.array([0.0, 0.0, 0.0])
    lat1 = np.array([50.0, 50.0, np.nan])
    lon2 = np.array([0.001, 0.002, 0.003])
    lat2 = np.array([50.0, 50.0, 50.0])
    d = geodesic_distance(lon1, lat1, lon2, lat2)
    assert np.isfinite(d[:2]).all()
    assert np.isnan(d[2])
    np.testing.assert_allclose(d[:2], [71.6957, 143.3915], rtol=1e-4)


def test_genuine_antipodal_pair_still_raises():
    """The real non-convergence case keeps its error -- it is not a NaN case."""
    with pytest.raises(ValueError, match="near-antipodal"):
        geodesic_distance(0.0, 0.0, 179.9, 0.1)
