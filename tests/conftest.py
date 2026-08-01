"""Shared fixtures: tiny synthetic networks and GPS point files.

Everything is offline and fast; geographic scale is chosen so degrees are
easy to reason about (~0.001 deg lon at the equator ~ 111 m).
"""
import json

import pandas as pd
import pytest

import roadtraffic as rt


def write_geojson(path, features):
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    return str(path)


def line_feature(coords, **props):
    return {
        "type": "Feature",
        "properties": props,
        "geometry": {"type": "LineString", "coordinates": coords},
    }


def write_ogr(path, features, *, crs="EPSG:4326", layer=None):
    """Write LineString features to a Shapefile/GPKG via pyogrio (no geopandas).

    ``features``: list of ``(coords, props)`` pairs -- same shape as
    ``line_feature``'s inputs, but for pyogrio-written formats (``.gpkg``,
    ``.shp``; the extension on ``path`` picks the driver). Imports pyogrio
    and shapely locally so collecting this module never requires the optional
    'shapefile' extra; callers must ``pytest.importorskip("pyogrio")`` first.
    """
    import numpy as np
    import pyogrio
    import shapely

    geoms = np.array([shapely.LineString(coords) for coords, _ in features],
                     dtype=object)
    wkb = shapely.to_wkb(geoms)
    field_names = sorted({k for _, props in features for k in props})
    field_data = tuple(
        np.array([props.get(name) for _, props in features], dtype=object)
        for name in field_names
    )
    kwargs = {} if layer is None else {"layer": layer}
    pyogrio.raw.write(str(path), wkb, field_data, field_names,
                      geometry_type="LineString", crs=crs, **kwargs)
    return str(path)


@pytest.fixture
def straight_net(tmp_path):
    """One two-way 1.1 km road along the equator from (0,0) to (0.01,0)."""
    path = write_geojson(tmp_path / "straight.json", [
        line_feature([[0, 0], [0.01, 0]], highway="residential", name="main"),
    ])
    return rt.Network.from_geojson(path)


@pytest.fixture
def grid_net(tmp_path):
    """A 3x3 grid of two-way streets, spacing 0.001 deg (~111 m)."""
    feats = []
    n, sp = 3, 0.001
    for i in range(n):
        for j in range(n):
            lon, lat = j * sp, i * sp
            if j < n - 1:
                feats.append(line_feature([[lon, lat], [lon + sp, lat]],
                                          highway="residential"))
            if i < n - 1:
                feats.append(line_feature([[lon, lat], [lon, lat + sp]],
                                          highway="residential"))
    return rt.Network.from_geojson(write_geojson(tmp_path / "grid.json", feats))


@pytest.fixture
def make_points_csv(tmp_path):
    """Factory: write a list of row dicts to CSV, return the path."""
    def _make(rows, name="points.csv"):
        p = tmp_path / name
        pd.DataFrame(rows).to_csv(p, index=False)
        return str(p)
    return _make


def drive_along_road(n=6, *, start_lon=0.0005, dlon=0.00015, lat=1e-5,
                     t0="2026-06-01 08:00:00", dt_s=10, traj=None):
    """Rows simulating a mover driving east along the equator road.

    Default step 0.00015 deg / 10 s ~ 1.67 m/s.
    """
    t0 = pd.Timestamp(t0)
    rows = []
    for k in range(n):
        row = {"lon": start_lon + dlon * k, "lat": lat,
               "time": (t0 + pd.Timedelta(seconds=dt_s * k)).isoformat()}
        if traj is not None:
            row["id"] = traj
        rows.append(row)
    return rows


def track_along_road(n=10, *, speed_mps, start_lon=0.0002, lat=1e-5,
                     t0="2026-06-01 08:00:00", dt_s=10, traj=None):
    """Rows for a mover travelling east along the equator road at a set speed.

    Spacing is computed from the speed so the planted value is exact: each fix
    sits ``speed_mps * dt_s`` metres further along, and ~111319.5 m is one
    degree of longitude at the equator.
    """
    t0 = pd.Timestamp(t0)
    dlon = speed_mps * dt_s / 111319.5
    rows = []
    for k in range(n):
        row = {"lon": start_lon + dlon * k, "lat": lat,
               "time": (t0 + pd.Timedelta(seconds=dt_s * k)).isoformat()}
        if traj is not None:
            row["id"] = traj
        rows.append(row)
    return rows


def walk_along_road(n=25, *, speed_mps=1.4, dt_s=15, **kw):
    """A mover on foot: ~1.4 m/s is an ordinary walking pace."""
    return track_along_road(n, speed_mps=speed_mps, dt_s=dt_s, **kw)
