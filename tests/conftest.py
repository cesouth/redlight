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
