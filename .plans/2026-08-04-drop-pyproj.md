# Drop pyproj as a Core Dependency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove `pyproj` from `roadtraffic`'s core dependencies by reimplementing the three things it is used for in pure numpy, keeping it as an optional `crs` extra for source coordinate systems that genuinely need a projection database.

**Architecture:** Two leaf modules carry all the geodesy. `_geo.py` is rewritten to compute ellipsoidal distance with Vincenty's inverse formula instead of `pyproj.Geod`. A new `_proj.py` implements the Krüger series for WGS84↔UTM plus Web Mercator inverse, and exposes `UtmCrs` / `UtmTransformer` shims whose surface (`.to_epsg()`, `.transform(x, y)`) matches the `pyproj` objects `Network` already stores — so `network.py` changes only where those objects are *constructed*, not where they are used. Anything outside the natively supported CRS set (EPSG:4326, the 120 WGS84 UTM zones, EPSG:3857) falls back to `pyproj` if installed, and otherwise raises an error naming the `crs` extra.

**Tech Stack:** Python 3.9+, numpy, shapely, networkx, pytest, ruff.

**Baseline:** 289 tests passing at `0efc222`. Exact counts below are a guide; the binding requirement is that nothing regresses.

## Global Constraints

- `requires-python = ">=3.9"`. **Every new module must start with `from __future__ import annotations`** — the codebase uses `X | None` annotations, which are syntax errors on 3.9 without it.
- ruff `line-length = 95`, lint rules `["E", "F", "W", "I", "B", "UP"]`. Run `ruff check src tests scripts examples` before every commit.
- Test command: `python -m pytest -q`. Full suite is currently **289 passed**; it must stay green (the count will grow as tasks add tests, but nothing may regress).
- **No new runtime dependencies.** numpy is already core. `pyproj` moves from `dependencies` to `[project.optional-dependencies]` under a new `crs` key.
- After Task 5, `grep -rn "pyproj" src/` must return hits **only** inside deferred (function-local) imports in `network.py`. No module-level `import pyproj` may remain anywhere in `src/`.
- Public API is frozen. `Network.crs_metric.to_epsg()`, `Network.project_points()`, `net._transformer_fwd.transform()` and `net._transformer_inv.transform()` are all used outside `network.py` (in `tests/test_derive_math.py`, `tests/test_pipeline_e2e.py`, `scripts/paper_experiments.py`, `examples/`) and must keep working unchanged.
- Numpy-style docstrings matching the density and tone of `cleaning.py`. Comments explain *why*, not *what*.

## Background: what pyproj is actually used for

Six call sites, three distinct features:

| Feature | Call sites | Replacement |
|---|---|---|
| `Geod(ellps="WGS84").inv()` — distance only, azimuths discarded | `points.py:488`, `cleaning.py:200`, `analysis.py:310` | Vincenty inverse (Task 1) |
| `CRS.from_epsg` + `Transformer`, WGS84↔auto-UTM | `network.py:357-359`, `:505`, `:662` | Krüger series (Tasks 2-3) |
| `CRS.from_user_input(meta["crs"])` — arbitrary source CRS | `network.py:265-268` | Native set + optional pyproj (Task 4) |

**Verified accuracy of the replacements** (measured against PROJ 9.5.1 over 30k random points per zone, 8 zones, full ±3° zone width, latitudes to 84°):

- UTM forward: **9 nanometres** worst case
- UTM round-trip: **0.13 mm** worst case
- Web Mercator inverse: 2.8e-14 degrees
- Geodesic distance: sub-micrometre at road scales, 6.8 µm at 1000 km
- Independent published check: Geoscience Australia's Vincenty test line (Flinders Peak → Buninyong, published **54972.271 m**) computes as **54972.2659 m**

**Do not substitute haversine for the geodesic.** It was measured at 0.56% max error on 10 km hops, which is a ~0.28 km/h bias on a 50 km/h speed and lands directly in the package's speed estimates.

---

### Task 1: Pure-numpy geodesic distance

Replaces `pyproj.Geod` in `_geo.py` and rewires its three consumers. `_geo.py` has no other callers, so `GEOD_WGS84` can be retired outright rather than shimmed.

**Files:**
- Modify: `src/roadtraffic/_geo.py` (full rewrite — currently 10 lines)
- Modify: `src/roadtraffic/points.py:62` (import), `:488-490` (call)
- Modify: `src/roadtraffic/cleaning.py:26` (import), `:200` (call)
- Modify: `src/roadtraffic/analysis.py:34` (import), `:310` (call)
- Create: `tests/test_geo.py`

**Interfaces:**
- Consumes: nothing (leaf module)
- Produces: `roadtraffic._geo.geodesic_distance(lon1, lat1, lon2, lat2) -> np.ndarray` — vectorised, metres, raises `ValueError` on non-convergence. Replaces `GEOD_WGS84.inv(...)`, which returned `(az12, az21, dist)`; the azimuths were discarded at every call site, so the new function returns distance alone.

- [ ] **Step 1: Write the failing test**

Create `tests/test_geo.py`:

```python
"""Ellipsoidal distance, without PROJ.

Reference values are pinned rather than computed against pyproj, so this suite
proves the geodesy in an environment where pyproj is not installed -- which is
the entire point of the module under test.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from roadtraffic._geo import geodesic_distance

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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_geo.py -q`
Expected: FAIL — `ImportError: cannot import name 'geodesic_distance' from 'roadtraffic._geo'`

- [ ] **Step 3: Rewrite `_geo.py`**

Replace the entire contents of `src/roadtraffic/_geo.py`:

```python
"""Shared geodesic machinery.

One WGS84 ellipsoid definition for the whole package, so every module measures
ground distance with the same model. Implemented with Vincenty's inverse
formula in numpy rather than PROJ, so the package carries no native projection
library and no coordinate database.
"""
from __future__ import annotations

import numpy as np

_A = 6378137.0             # WGS84 semi-major axis (metres)
_F = 1 / 298.257223563     # WGS84 flattening
_B = _A * (1 - _F)         # semi-minor axis
_MAX_ITER = 200
_TOL = 1e-12


def geodesic_distance(lon1, lat1, lon2, lat2):
    """Ellipsoidal ground distance in metres between two WGS84 positions.

    Vectorised over numpy arrays and broadcast elementwise. Scalar inputs come
    back as a 0-d array, so wrap in ``float()`` when a Python scalar is wanted.

    Parameters
    ----------
    lon1, lat1, lon2, lat2 : array_like
        WGS84 coordinates in decimal degrees.

    Returns
    -------
    numpy.ndarray
        Distance in metres. Agrees with PROJ's geodesic to a few micrometres
        at every separation this package encounters.

    Raises
    ------
    ValueError
        If the iteration does not converge. This happens only for
        near-antipodal inputs -- points on opposite sides of the globe, where
        every direction is an equally short path and the formula has nothing
        to converge on. Returning the unconverged value would be a silently
        wrong distance, plausible-looking and off by kilometres. No road
        network can legitimately produce such a pair, so this signals corrupt
        input coordinates rather than an unsupported case.
    """
    lon1, lat1, lon2, lat2 = (np.asarray(v, dtype=float)
                              for v in (lon1, lat1, lon2, lat2))
    lam_l = np.radians(lon2 - lon1)
    u1 = np.arctan((1 - _F) * np.tan(np.radians(lat1)))
    u2 = np.arctan((1 - _F) * np.tan(np.radians(lat2)))
    sin_u1, cos_u1 = np.sin(u1), np.cos(u1)
    sin_u2, cos_u2 = np.sin(u2), np.cos(u2)

    lam = np.array(lam_l, dtype=float, copy=True)
    converged = False
    for _ in range(_MAX_ITER):
        sin_lam, cos_lam = np.sin(lam), np.cos(lam)
        sin_sigma = np.hypot(cos_u2 * sin_lam,
                             cos_u1 * sin_u2 - sin_u1 * cos_u2 * cos_lam)
        cos_sigma = sin_u1 * sin_u2 + cos_u1 * cos_u2 * cos_lam
        sigma = np.arctan2(sin_sigma, cos_sigma)
        # Coincident points leave sin_sigma at 0, and equatorial lines leave
        # cos_sq_alpha at 0. Both make the next terms indeterminate, but both
        # cancel out of the final sum, so pinning them to 0 is exact rather
        # than an approximation.
        with np.errstate(invalid="ignore", divide="ignore"):
            sin_alpha = np.where(sin_sigma == 0, 0.0,
                                 cos_u1 * cos_u2 * sin_lam / sin_sigma)
        cos_sq_alpha = 1 - sin_alpha**2
        with np.errstate(invalid="ignore", divide="ignore"):
            cos_2sigma_m = np.where(cos_sq_alpha == 0, 0.0,
                                    cos_sigma - 2 * sin_u1 * sin_u2 / cos_sq_alpha)
        c = _F / 16 * cos_sq_alpha * (4 + _F * (4 - 3 * cos_sq_alpha))
        lam_prev = lam
        lam = lam_l + (1 - c) * _F * sin_alpha * (
            sigma + c * sin_sigma
            * (cos_2sigma_m + c * cos_sigma * (-1 + 2 * cos_2sigma_m**2))
        )
        if np.all(np.abs(lam - lam_prev) < _TOL):
            converged = True
            break

    if not converged:
        raise ValueError(
            "geodesic_distance failed to converge: the input contains a "
            "near-antipodal coordinate pair (points on opposite sides of the "
            "globe). Check the input coordinates -- a road network should "
            "never contain one."
        )

    u_sq = cos_sq_alpha * (_A**2 - _B**2) / _B**2
    big_a = 1 + u_sq / 16384 * (4096 + u_sq * (-768 + u_sq * (320 - 175 * u_sq)))
    big_b = u_sq / 1024 * (256 + u_sq * (-128 + u_sq * (74 - 47 * u_sq)))
    delta_sigma = big_b * sin_sigma * (
        cos_2sigma_m + big_b / 4 * (
            cos_sigma * (-1 + 2 * cos_2sigma_m**2)
            - big_b / 6 * cos_2sigma_m * (-3 + 4 * sin_sigma**2)
            * (-3 + 4 * cos_2sigma_m**2)
        )
    )
    return _B * big_a * (sigma - delta_sigma)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_geo.py -q`
Expected: PASS, 11 tests.

- [ ] **Step 5: Rewire the three call sites**

In `src/roadtraffic/points.py`, change line 62 from `from ._geo import GEOD_WGS84` to:

```python
from ._geo import geodesic_distance
```

and replace lines 488-490:

```python
        dist = geodesic_distance(
            lon[sidx[:-1]], lat[sidx[:-1]], lon[sidx[1:]], lat[sidx[1:]]
        )
```

In `src/roadtraffic/cleaning.py`, change line 26 to `from ._geo import geodesic_distance`, and replace line 200:

```python
            dist = geodesic_distance(lon[i], lat[i], lon[j + 1], lat[j + 1])
```

In `src/roadtraffic/analysis.py`, change line 34 to `from ._geo import geodesic_distance`, and replace line 310:

```python
        gc = float(geodesic_distance(u[0], u[1], v[0], v[1]))
```

- [ ] **Step 6: Verify the full suite and that pyproj is gone from these modules**

```bash
python -m pytest -q
grep -n "pyproj\|GEOD_WGS84" src/roadtraffic/points.py src/roadtraffic/cleaning.py \
    src/roadtraffic/analysis.py src/roadtraffic/_geo.py
ruff check src tests scripts examples
```

Expected: 289 existing tests still pass plus 11 new ones (**300 passed**); the grep returns **nothing**; ruff is clean.

- [ ] **Step 7: Commit**

```bash
git add src/roadtraffic/_geo.py src/roadtraffic/points.py \
    src/roadtraffic/cleaning.py src/roadtraffic/analysis.py tests/test_geo.py
git commit -m "refactor: compute geodesic distance in numpy instead of pyproj.Geod"
```

---

### Task 2: `_proj.py` — UTM and Web Mercator in numpy

A standalone leaf module, fully tested before anything imports it. Nothing in the package changes in this task.

**Files:**
- Create: `src/roadtraffic/_proj.py`
- Create: `tests/test_proj.py`

**Interfaces:**
- Consumes: nothing (leaf module)
- Produces:
  - `EPSG_WGS84 = 4326`, `EPSG_WEB_MERCATOR = 3857`
  - `utm_epsg_to_zone(epsg: int) -> tuple[int, bool]` — `(zone, is_northern)`, raises `ValueError` for non-UTM codes
  - `utm_forward(lon, lat, zone, north) -> (x, y)`
  - `utm_inverse(x, y, zone, north) -> (lon, lat)`
  - `web_mercator_inverse(x, y) -> (lon, lat)`
  - `parse_epsg(crs) -> int | None`
  - `class UtmCrs` with `.to_epsg()`, `.name`, `.zone`, `.north`
  - `class UtmTransformer` with `.transform(xx, yy)`
  - `utm_crs_and_transformers(epsg) -> (UtmCrs, UtmTransformer, UtmTransformer)`

- [ ] **Step 1: Write the failing test**

Create `tests/test_proj.py`:

```python
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
    assert float(got_lon) == pytest.approx(lon, abs=1e-9)
    assert float(got_lat) == pytest.approx(lat, abs=1e-9)


def test_utm_roundtrip_across_a_full_zone_width():
    """Round-trip error stays sub-millimetre over the whole legal zone, not
    just near the central meridian where the series is most accurate."""
    rng = np.random.default_rng(0)
    lon = 15.0 + rng.uniform(-3, 3, 2000)
    lat = rng.uniform(0.5, 84.0, 2000)
    x, y = _proj.utm_forward(lon, lat, 33, True)
    got_lon, got_lat = _proj.utm_inverse(x, y, 33, True)
    # 1e-8 deg is ~1.1 mm at the equator.
    assert np.max(np.abs(got_lon - lon)) < 1e-8
    assert np.max(np.abs(got_lat - lat)) < 1e-8


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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_proj.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'roadtraffic._proj'`

- [ ] **Step 3: Create `_proj.py`**

Create `src/roadtraffic/_proj.py`:

```python
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
        2 * n - 2 * n2 / 3 - 2 * n3,
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_proj.py -q`
Expected: PASS, 31 tests.

- [ ] **Step 5: Commit**

```bash
ruff check src tests
git add src/roadtraffic/_proj.py tests/test_proj.py
git commit -m "feat: add pure-numpy UTM and Web Mercator projections"
```

---

### Task 3: Wire `Network` onto `_proj`

Removes the module-level pyproj import from `network.py`. The metric CRS is UTM by default; a user-supplied non-UTM `metric_epsg` still works, via a deferred pyproj import.

**Files:**
- Modify: `src/roadtraffic/network.py:29` (drop module-level import), `:357-359` (`_build`), and add two helpers near `_auto_utm_epsg`
- Modify: `tests/test_network.py` (add coverage for the fallback error)

**Interfaces:**
- Consumes: `roadtraffic._proj.utm_crs_and_transformers`, `_proj.parse_epsg`, `_proj.EPSG_WGS84`
- Produces: `network._metric_crs_and_transformers(epsg: int) -> (crs, fwd, inv)` and `network._require_pyproj(what: str) -> module` — both used again in Task 4.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_network.py`:

```python
def test_metric_crs_is_native_utm_without_pyproj(straight_net):
    """The default path must not construct a pyproj object at all."""
    from roadtraffic import _proj
    assert isinstance(straight_net.crs_metric, _proj.UtmCrs)
    assert straight_net.crs_metric.to_epsg() == 32631


def test_non_utm_metric_epsg_errors_clearly_without_pyproj(tmp_path, monkeypatch):
    """A user-supplied projected CRS outside the native set still works when
    pyproj is installed, but must fail with an actionable message when it is
    not -- never with a bare ModuleNotFoundError."""
    import builtins

    from roadtraffic import network as net_mod

    path = write_geojson(tmp_path / "n.json", [
        ([[15.0, 50.0], [15.01, 50.0]], {"highway": "residential"}),
    ])
    real_import = builtins.__import__

    def no_pyproj(name, *args, **kwargs):
        if name == "pyproj":
            raise ImportError("no pyproj")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_pyproj)
    with pytest.raises(ImportError, match=r"roadtraffic\[crs\]"):
        net_mod.Network.from_geojson(path, metric_epsg=27700)
```

Note: `write_geojson` is already imported at the top of `tests/test_network.py` from `conftest`; do not re-import it.

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_network.py -q -k "native_utm or non_utm_metric"`
Expected: FAIL — `AttributeError: module 'roadtraffic._proj' has no attribute 'UtmCrs'` is already satisfied by Task 2, so the real failure is on the second test: `pyproj.exceptions.CRSError` or a bare `ImportError` without the `roadtraffic[crs]` text.

- [ ] **Step 3: Replace the module-level pyproj import**

In `src/roadtraffic/network.py`, delete line 29 (`from pyproj import CRS, Transformer`) and add `from . import _proj` to the local-import block so the imports read:

```python
import numpy as np
import shapely
from shapely.geometry import LineString, shape
from shapely.ops import transform as shapely_transform

from . import _proj
```

- [ ] **Step 4: Add the two helpers**

Insert immediately after `_auto_utm_epsg` (currently ending at line 42):

```python
def _require_pyproj(what: str):
    """Import pyproj, or raise an error that names the extra and the way out.

    pyproj is not a core dependency: it ships PROJ's native library and its
    coordinate database, which is both the bulk of the install and the usual
    source of ``proj.db`` conflicts. The natively supported systems cover
    essentially all road data, so this is reached only for the long tail.
    """
    try:
        import pyproj
    except ImportError as exc:
        raise ImportError(
            f"{what} needs pyproj, which roadtraffic does not install by "
            f"default. Either install the extra:\n"
            f"    pip install 'roadtraffic[crs]'\n"
            f"or use a natively supported CRS: EPSG:4326 (WGS84), EPSG:3857 "
            f"(Web Mercator), or a WGS84 UTM zone (EPSG:32601-32660 and "
            f"32701-32760)."
        ) from exc
    return pyproj


def _metric_crs_and_transformers(epsg: int):
    """Return ``(crs, forward, inverse)`` for the metric CRS.

    UTM -- the default and the overwhelmingly common case -- is handled in
    numpy. Any other projected CRS the caller asks for via ``metric_epsg``
    falls back to pyproj.
    """
    try:
        return _proj.utm_crs_and_transformers(epsg)
    except ValueError:
        pass
    pyproj = _require_pyproj(f"metric_epsg=EPSG:{epsg}")
    crs = pyproj.CRS.from_epsg(epsg)
    wgs84 = pyproj.CRS.from_epsg(_proj.EPSG_WGS84)
    return (crs,
            pyproj.Transformer.from_crs(wgs84, crs, always_xy=True),
            pyproj.Transformer.from_crs(crs, wgs84, always_xy=True))
```

- [ ] **Step 5: Rewire `_build`**

In `src/roadtraffic/network.py`, replace lines 357-359:

```python
        crs_metric, fwd, inv = _metric_crs_and_transformers(metric_epsg)
```

Nothing else in `_build` changes — `fwd.transform(...)` at line 396 and the `Network(...)` construction at line 450 already use only the shimmed surface.

- [ ] **Step 6: Run the tests**

```bash
python -m pytest -q
ruff check src tests
```

Expected: **333 passed** (300 after Task 1 + 31 from Task 2 + 2 new). In particular `tests/test_derive_math.py` and `tests/test_pipeline_e2e.py`, which use `_transformer_inv.transform()` and `crs_metric.to_epsg()`, must pass untouched — they are the proof that the shim surface is right.

- [ ] **Step 7: Commit**

```bash
git add src/roadtraffic/network.py tests/test_network.py
git commit -m "refactor: project the metric CRS in numpy, deferring pyproj to non-UTM overrides"
```

---

### Task 4: Source-CRS handling in `from_file`

The only place a genuinely arbitrary CRS can arrive. `from_geojson` and `from_overpass` are unaffected — both are WGS84 by specification and read no CRS member.

**Files:**
- Modify: `src/roadtraffic/network.py:265-268` and `:279-282` (the reprojection branch), plus one helper
- Modify: `tests/test_network.py:151-169` (drop the pyproj import from `test_from_file_reprojects_non_wgs84_crs`)

**Interfaces:**
- Consumes: `network._require_pyproj`, `_proj.parse_epsg`, `_proj.utm_epsg_to_zone`, `_proj.utm_inverse`, `_proj.web_mercator_inverse`, `_proj.EPSG_WGS84`, `_proj.EPSG_WEB_MERCATOR`
- Produces: `network._source_to_wgs84(crs) -> callable | None` — returns `None` when the source is already WGS84.

- [ ] **Step 1: Write the failing test**

Replace `test_from_file_reprojects_non_wgs84_crs` in `tests/test_network.py:151-169` with the following, and add the three tests after it:

```python
def test_from_file_reprojects_utm_without_pyproj(tmp_path):
    """A UTM GeoPackage reprojects natively. Pinned eastings from PROJ 9.5.1
    for 15.0E/50.0N and 15.01E/50.0N in EPSG:32633, so this test does not
    import pyproj at all."""
    pytest.importorskip("pyogrio")
    path = write_ogr(tmp_path / "utm.gpkg", [
        ([[500000.000000, 5538630.702867],
          [500716.670753, 5538630.750777]], {"highway": "residential"}),
    ], crs="EPSG:32633")
    net = rt.Network.from_file(path)
    lons = sorted(n[0] for n in net.graph.nodes())
    assert lons[0] == pytest.approx(15.0, abs=1e-6)
    assert lons[1] == pytest.approx(15.01, abs=1e-6)


def test_from_file_reprojects_web_mercator(tmp_path):
    """EPSG:3857 is the other CRS handled natively."""
    pytest.importorskip("pyogrio")
    path = write_ogr(tmp_path / "wm.gpkg", [
        ([[1669792.3618991035, 6446275.841017159],
          [1670905.5568070365, 6446275.841017159]], {"highway": "residential"}),
    ], crs="EPSG:3857")
    net = rt.Network.from_file(path)
    lons = sorted(n[0] for n in net.graph.nodes())
    assert lons[0] == pytest.approx(15.0, abs=1e-6)


def test_from_file_wgs84_needs_no_transform(tmp_path):
    """The common case must not touch the projection code at all."""
    pytest.importorskip("pyogrio")
    from roadtraffic import network as net_mod
    assert net_mod._source_to_wgs84("EPSG:4326") is None
    assert net_mod._source_to_wgs84(None) is None


def test_from_file_exotic_crs_errors_clearly_without_pyproj(tmp_path, monkeypatch):
    """British National Grid has no closed form here. Without pyproj the
    failure must name the extra, not surface as ModuleNotFoundError."""
    pytest.importorskip("pyogrio")
    import builtins

    path = write_ogr(tmp_path / "bng.gpkg", [
        ([[529000.0, 181000.0], [529100.0, 181100.0]], {"highway": "residential"}),
    ], crs="EPSG:27700")
    real_import = builtins.__import__

    def no_pyproj(name, *args, **kwargs):
        if name == "pyproj":
            raise ImportError("no pyproj")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_pyproj)
    with pytest.raises(ImportError, match=r"roadtraffic\[crs\]"):
        rt.Network.from_file(path)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_network.py -q -k "from_file"`
Expected: FAIL — `AttributeError: module 'roadtraffic.network' has no attribute '_source_to_wgs84'`

- [ ] **Step 3: Add the `_source_to_wgs84` helper**

Insert in `src/roadtraffic/network.py` immediately after `_metric_crs_and_transformers` (added in Task 3):

```python
def _source_to_wgs84(crs):
    """Return an ``(x, y) -> (lon, lat)`` callable for a file's source CRS.

    Returns None when the source is already WGS84 and no transform is needed.
    UTM and Web Mercator are handled in numpy; anything else -- national
    grids, non-WGS84 datums, raw WKT with no authority code -- needs PROJ's
    database and falls back to pyproj.
    """
    epsg = _proj.parse_epsg(crs)
    if epsg is None:
        # No CRS recorded at all means WGS84 by convention; a CRS that is
        # recorded but unparseable (raw WKT) is a real one we cannot read.
        if not crs:
            return None
        pyproj = _require_pyproj(f"Reading a file in {crs!s:.60}")
        return pyproj.Transformer.from_crs(
            pyproj.CRS.from_user_input(crs),
            pyproj.CRS.from_epsg(_proj.EPSG_WGS84),
            always_xy=True,
        ).transform
    if epsg == _proj.EPSG_WGS84:
        return None
    if epsg == _proj.EPSG_WEB_MERCATOR:
        return _proj.web_mercator_inverse
    try:
        zone, north = _proj.utm_epsg_to_zone(epsg)
    except ValueError:
        pyproj = _require_pyproj(f"Reading a file in EPSG:{epsg}")
        return pyproj.Transformer.from_crs(
            pyproj.CRS.from_epsg(epsg),
            pyproj.CRS.from_epsg(_proj.EPSG_WGS84),
            always_xy=True,
        ).transform
    return lambda x, y: _proj.utm_inverse(x, y, zone, north)
```

- [ ] **Step 4: Rewire `from_file`**

In `src/roadtraffic/network.py`, replace lines 265-268:

```python
        to_wgs84 = _source_to_wgs84(meta.get("crs"))
```

and replace the reprojection branch at lines 279-282:

```python
                if to_wgs84 is not None:
                    part = shapely_transform(to_wgs84, part)
```

- [ ] **Step 5: Run the tests**

```bash
python -m pytest -q
ruff check src tests
```

Expected: **336 passed** (333 after Task 3, minus the 1 replaced test, plus 4 new).

- [ ] **Step 6: Commit**

```bash
git add src/roadtraffic/network.py tests/test_network.py
git commit -m "feat: read UTM and Web Mercator source files without pyproj"
```

---

### Task 5: Packaging, documentation, and a no-pyproj proof

Moves the dependency, corrects every claim that names it, and adds CI coverage proving the package works with pyproj absent — the only test that can keep this from silently regressing.

**Files:**
- Modify: `pyproject.toml:29-36` (dependencies), `:39-48` (extras)
- Modify: `README.md:12`, `docs/index.md:9`, `docs/methodology.md:46`, `docs/statistics.md:23`, `docs/api.md:21`
- Modify: `.github/workflows/ci.yml:25-27` (comment) and add a no-pyproj job
- Modify: `src/roadtraffic/network.py:1-20` (module docstring)
- Create: `tests/test_no_pyproj.py`

**Interfaces:**
- Consumes: everything from Tasks 1-4
- Produces: no new code surface.

- [ ] **Step 1: Write the failing test**

Create `tests/test_no_pyproj.py`:

```python
"""Proof that the core package never imports pyproj.

The point of the numpy geodesy is that a default install has no PROJ in it.
That property is invisible in a dev environment where pyproj happens to be
installed, so it needs a test that fails loudly when someone reintroduces a
module-level import.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap

CORE_WORKLOAD = textwrap.dedent("""
    import sys

    class Blocker:
        # find_spec, not the legacy find_module, which was removed in 3.12.
        def find_spec(self, name, path=None, target=None):
            if name == "pyproj" or name.startswith("pyproj."):
                raise ImportError("pyproj is blocked for this test")
            return None

    sys.meta_path.insert(0, Blocker())

    import json, os, tempfile
    import roadtraffic as rt

    tmp = tempfile.mkdtemp()
    net_path = os.path.join(tmp, "net.json")
    with open(net_path, "w") as fh:
        json.dump({"type": "FeatureCollection", "features": [{
            "type": "Feature",
            "properties": {"highway": "residential"},
            "geometry": {"type": "LineString",
                         "coordinates": [[0.0, 0.0], [0.01, 0.0]]},
        }]}, fh)

    net = rt.Network.from_geojson(net_path)
    assert net.crs_metric.to_epsg() == 32631, net.crs_metric.to_epsg()
    px, py = net.project_points([0.005], [0.0])
    assert abs(float(px[0])) > 0

    pts_path = os.path.join(tmp, "pts.csv")
    with open(pts_path, "w") as fh:
        fh.write("id,lon,lat,time\\n")
        for i in range(6):
            fh.write(f"a,{i * 0.0015:.6f},0.00002,2026-06-01T08:00:{i * 10:02d}\\n")

    pts = rt.load_points(pts_path, id_col="id")
    matched = rt.NearestMatcher(net, max_dist=60).match(pts)
    speeds = rt.derive_speeds(net, matched, pts)
    assert len(speeds["intervals"]) > 0, speeds["intervals"]

    stats = rt.network_stats(net)
    assert stats["n_edges"] == 2, stats["n_edges"]

    assert "pyproj" not in sys.modules, "pyproj was imported by the core path"
    print("OK")
""")


def test_core_workload_runs_with_pyproj_blocked():
    """Load, match, derive speeds, and compute stats with pyproj unimportable.

    Runs in a subprocess because the import blocker has to be installed before
    roadtraffic is first imported, and pytest has already imported it here.
    """
    result = subprocess.run(
        [sys.executable, "-c", CORE_WORKLOAD],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        f"core workload failed with pyproj blocked:\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    assert "OK" in result.stdout
```

This exact script has been run against the current API: `load_points`, `NearestMatcher.match`, `derive_speeds(net, matched, pts)["intervals"]` and `network_stats(net)["n_edges"]` are all correct as written, and the workload prints `OK` when pyproj is importable. It fails today only because `_geo.py` still imports `pyproj.Geod` at module level.

- [ ] **Step 2: Run the test to verify it fails or passes for the right reason**

Run: `python -m pytest tests/test_no_pyproj.py -q`
Expected: PASS if Tasks 1-4 are complete. If it FAILS, the assertion message names the module that still imports pyproj — fix that before continuing. This test is the gate for the whole plan.

- [ ] **Step 3: Move the dependency in `pyproject.toml`**

Replace the `dependencies` list (lines 29-36):

```toml
# Core dependencies: all lightweight wheels. No GDAL, no PROJ.
dependencies = [
    "numpy>=1.21",
    "pandas>=1.3",
    "scipy>=1.7",
    "shapely>=2.0",
    "networkx>=2.6",
]
```

and add the new extra to `[project.optional-dependencies]`, immediately before the `shapefile` entry:

```toml
# Coordinate systems outside the natively supported set. roadtraffic projects
# WGS84, the WGS84 UTM zones and Web Mercator in numpy; this extra adds PROJ
# for everything else -- national grids, non-WGS84 datums, raw-WKT CRS.
crs = ["pyproj>=3.2"]
```

- [ ] **Step 4: Correct the documentation**

`README.md:12` — replace the dependency sentence with:

```markdown
`shapely`, `networkx`. **No GDAL and no PROJ** required for the core: WGS84,
the UTM zones and Web Mercator are projected in numpy. GeoJSON
```

`docs/index.md:9-10`:

```markdown
Built on `numpy`, `pandas`, `scipy`, `shapely` and `networkx`.
**No GDAL and no PROJ required** for the core.
```

`docs/methodology.md:46` — drop `pyproj` from the dependency list so it reads `scipy, shapely, networkx`.

`docs/statistics.md:23` — extend the `metric_epsg` note:

```markdown
- You may override this with `metric_epsg=` if your study area spans zones or
  you need a specific projected CRS. UTM zones are projected in numpy; any
  other EPSG code requires the `crs` extra (`pip install 'roadtraffic[crs]'`).
```

`docs/api.md:21` — extend the `metric_epsg` row:

```markdown
| `metric_epsg` | EPSG code of the projected CRS for distance math. Default: auto UTM zone of the first vertex. UTM zones need no extra dependency; other codes require the `crs` extra. |
```

`src/roadtraffic/network.py:17-19` — replace the CRS paragraph of the module docstring:

```python
Coordinates are stored in WGS84 (EPSG:4326) and also projected to a local
metric CRS (auto UTM, or user-specified) for distance-correct snapping and
length computation. Snapping/length math must never be done in degrees. UTM
and Web Mercator are projected in numpy (see ``_proj``); any other CRS needs
the optional ``crs`` extra, which pulls in pyproj.
```

- [ ] **Step 5: Correct the CI comment and add a no-pyproj job**

In `.github/workflows/ci.yml`, replace the misleading comment at lines 25-27:

```yaml
          # The floor and the ceiling are the versions most likely to break;
          # check them on macOS too, since shapely and scipy ship different
          # wheels there.
```

("pure-Python only by intent" was never accurate — numpy, scipy and shapely all ship C extensions. Removing pyproj does not change that, and the comment should not imply otherwise.)

Then add this job at the end of the file, at the same indentation as `test`:

```yaml
  no-pyproj:
    name: core install has no PROJ
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install without any extras
        run: |
          python -m pip install --upgrade pip
          pip install . pytest
      - name: Assert pyproj is genuinely absent
        run: |
          ! pip show pyproj
          python -c "import roadtraffic; print(roadtraffic.__version__)"
      - name: Run the suite
        run: pytest -q
```

- [ ] **Step 6: Verify everything**

```bash
python -m pytest -q
ruff check src tests scripts examples
grep -rn "pyproj" src/ | grep -v "_require_pyproj\|import pyproj\|pyproj\."
python -m pytest tests/test_docs.py -q
```

Expected: **337 passed**; ruff clean; the grep returns only comment/docstring mentions and no module-level import; `test_docs.py` green, confirming no documentation code block broke.

Then confirm the dependency is really gone from a fresh build:

```bash
python -m pip install --dry-run . 2>&1 | grep -i pyproj || echo "pyproj not in the dependency set"
```

- [ ] **Step 7: Update the changelog and commit**

Add to the top of `CHANGELOG.md` under a new Unreleased heading:

```markdown
### Changed
- `pyproj` is no longer a core dependency. WGS84, the 120 WGS84 UTM zones and
  Web Mercator are now projected in numpy, and geodesic distance uses
  Vincenty's inverse formula. Verified against PROJ 9.5.1 to under 10 nm for
  the UTM forward projection and a few micrometres for geodesic distance.
  This removes 21 MB from an install and the `proj.db` class of environment
  conflicts along with it.
- Reading a file in any other CRS -- national grids, non-WGS84 datums, raw-WKT
  CRS -- now requires the new `crs` extra: `pip install 'roadtraffic[crs]'`.
  The same applies to a `metric_epsg=` override outside the UTM zones. Both
  raise an `ImportError` naming the extra rather than failing obscurely.
```

```bash
git add pyproject.toml README.md docs/ .github/workflows/ci.yml CHANGELOG.md \
    src/roadtraffic/network.py tests/test_no_pyproj.py
git commit -m "build: drop pyproj from core dependencies, add the crs extra"
```

---

## Verification Summary

After Task 5, all of the following must hold:

| Check | Command | Expected |
|---|---|---|
| Suite green | `python -m pytest -q` | 337 passed |
| Lint clean | `ruff check src tests scripts examples` | no findings |
| No core pyproj import | `python -m pytest tests/test_no_pyproj.py -q` | passed |
| Docs still execute | `python -m pytest tests/test_docs.py -q` | passed |
| Dependency removed | `grep -n pyproj pyproject.toml` | only under `[project.optional-dependencies]` |
| Public API intact | `python -m pytest tests/test_derive_math.py tests/test_pipeline_e2e.py -q` | passed |

## Out of Scope

- **Azimuths.** `pyproj.Geod.inv` returned `(az12, az21, dist)`; every call site discarded the azimuths. `geodesic_distance` returns distance only. If a bearing is ever needed, extend `_geo.py` then — do not add it speculatively.
- **Forward geodesic / destination-point.** Not used anywhere in the package.
- **Datum transformation.** Any non-WGS84 datum needs PROJ's shift grids and stays behind the `crs` extra permanently. This is a deliberate boundary, not a gap to close later.
- **`pyogrio`.** Unchanged, still behind the `shapefile` extra, and it does not depend on pyproj — so that extra does not reintroduce PROJ.
