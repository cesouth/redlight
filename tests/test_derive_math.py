"""Tests for on-road speed derivation (roadtraffic.speeds.derive_speeds).

Two layers, matching the two scripts these were adapted from:

- Distance/speed math against a synthetic network with known ground truth
  (validates the undirected on-road distance calculation directly).
- A full end-to-end smoke test: no-speed CSV -> HMM match -> derive_speeds ->
  filter_by_speed -> aggregate_speeds -> assign_speeds -> Router, on a
  simulated noisy trip.
"""
import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString

import roadtraffic as rt
from roadtraffic.points import PointSet


# --------------------------------------------------------------------------- #
# A. on-road distance/speed math against known ground truth
# --------------------------------------------------------------------------- #
def _l_shaped_network(tmp):
    """Two-way L-shaped road: east leg then north leg -> 4 directed edges."""
    path = os.path.join(tmp, "lnet.geojson")
    gj = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {"name": "east_leg"},
             "geometry": {"type": "LineString",
                          "coordinates": [[0.0, 0.0], [0.001, 0.0]]}},
            {"type": "Feature", "properties": {"name": "north_leg"},
             "geometry": {"type": "LineString",
                          "coordinates": [[0.001, 0.0], [0.001, 0.001]]}},
        ],
    }
    with open(path, "w") as fh:
        json.dump(gj, fh)
    return rt.Network.from_geojson(path)


def _find_edge(net, u, v):
    for eid in net.edge_ids:
        if net.edge_endpoints(int(eid)) == (u, v):
            return int(eid)
    raise KeyError((u, v))


def _place(net, eid, frac, perp_axis, perp_m=3.0):
    """lon/lat at arc-fraction `frac` along edge `eid`, offset perpendicular so
    the snap is non-trivial."""
    geom = net._edge_geoms_proj[eid]
    p = geom.interpolate(frac * geom.length)
    x, y = p.x, p.y
    if perp_axis == "y":
        y += perp_m
    else:
        x += perp_m
    lon, lat = net._transformer_inv.transform(x, y)
    return float(lon), float(lat)


def _run_case(net, eid1, f1, ax1, eid2, f2, ax2, true_v, true_dist):
    lon1, lat1 = _place(net, eid1, f1, ax1)
    lon2, lat2 = _place(net, eid2, f2, ax2)
    dt = true_dist / true_v
    base = pd.Timestamp("2024-01-01T08:00:00")
    times = [base, base + pd.to_timedelta(dt, unit="s")]

    pdf = pd.DataFrame({
        "point_id": [0, 1],
        "traj_id": ["T", "T"],
        "lon": [lon1, lon2],
        "lat": [lat1, lat2],
        "time": times,
    })
    ps = PointSet(pdf, has_traj=True)

    matched = pd.DataFrame({
        "point_id": [0, 1],
        "edge_id": [eid1, eid2],
        "snap_dist_m": [3.0, 3.0],
        "time": times,
        "traj_id": ["T", "T"],
    })

    res = rt.derive_speeds(net, matched, ps, default_pos_sigma_m=5.0)
    iv = res["intervals"].iloc[0]
    return iv, res["edge_observations"]


def test_derive_speeds_turn_across_shared_node():
    tmp = tempfile.mkdtemp()
    net = _l_shaped_network(tmp)
    n00, n10, n11 = (0.0, 0.0), (0.001, 0.0), (0.001, 0.001)
    east_fwd = _find_edge(net, n00, n10)
    north_fwd = _find_edge(net, n10, n11)
    L_east = net.edge_length(east_fwd)
    L_north = net.edge_length(north_fwd)

    # true on-road distance = remaining half of east leg + full turn + half of north leg
    true_dist = (1 - 0.5) * L_east + 0.5 * L_north
    iv, eo = _run_case(net, east_fwd, 0.5, "y", north_fwd, 0.5, "x",
                        true_v=10.0, true_dist=true_dist)

    assert iv["distance_m"] == pytest.approx(true_dist, rel=1e-6)
    assert iv["speed_mps"] == pytest.approx(10.0, rel=1e-6)
    # both directions of both traversed roads are populated for routing
    assert eo["edge_id"].nunique() == 4


def test_derive_speeds_same_edge_forward():
    tmp = tempfile.mkdtemp()
    net = _l_shaped_network(tmp)
    n00, n10 = (0.0, 0.0), (0.001, 0.0)
    east_fwd = _find_edge(net, n00, n10)
    L_east = net.edge_length(east_fwd)

    true_dist = (0.75 - 0.25) * L_east
    iv, eo = _run_case(net, east_fwd, 0.25, "y", east_fwd, 0.75, "y",
                        true_v=8.0, true_dist=true_dist)

    assert iv["distance_m"] == pytest.approx(true_dist, rel=1e-6)
    assert iv["speed_mps"] == pytest.approx(8.0, rel=1e-6)
    assert eo["edge_id"].nunique() == 2  # both directions of the one road


# --------------------------------------------------------------------------- #
# B. end-to-end: no-speed CSV -> HMM -> derive_speeds -> aggregate/assign/route
# --------------------------------------------------------------------------- #
def test_speeds_pipeline_end_to_end():
    rng = np.random.default_rng(0)
    tmp = tempfile.mkdtemp()

    # a gentle eastward polyline then a northward leg
    coords_e = [[0.0 + 0.0005 * k, 0.0] for k in range(6)]          # ~278 m east
    coords_n = [[coords_e[-1][0], 0.0005 * k] for k in range(5)]    # ~221 m north
    gj = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"name": "E"},
         "geometry": {"type": "LineString", "coordinates": coords_e}},
        {"type": "Feature", "properties": {"name": "N"},
         "geometry": {"type": "LineString", "coordinates": coords_n}},
    ]}
    net_path = os.path.join(tmp, "road2.geojson")
    with open(net_path, "w") as fh:
        json.dump(gj, fh)
    net = rt.Network.from_geojson(net_path)

    # simulate a trip along E then N at ~10 m/s, 1 Hz, ~5 m GPS noise, NO speed col
    fwd, inv = net._transformer_fwd, net._transformer_inv
    all_ll = coords_e + coords_n[1:]
    xs, ys = fwd.transform(np.array([c[0] for c in all_ll]),
                            np.array([c[1] for c in all_ll]))
    line = LineString(np.column_stack([xs, ys]))
    v_true = 10.0
    times = np.arange(0, line.length / v_true, 1.0)
    rows = []
    for t in times:
        p = line.interpolate(v_true * t)
        nx_, ny_ = p.x + rng.normal(0, 5), p.y + rng.normal(0, 5)
        lon, lat = inv.transform(nx_, ny_)
        rows.append({"track_id": "trip1", "lon": float(lon), "lat": float(lat),
                     "timestamp": pd.Timestamp("2024-06-01T08:00:00")
                     + pd.to_timedelta(t, "s")})
    csv_path = os.path.join(tmp, "trip.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    # load WITHOUT a speed column -- position+time only must be valid input
    ps = rt.load_points(csv_path, id_col="track_id")
    assert "speed_mps" not in ps.df.columns

    matched = rt.HMMMatcher(net, sigma_z=6.0, beta=30.0, max_dist=80.0).match(ps)
    assert "speed_mps" not in matched.columns
    assert (matched["edge_id"] != -1).mean() > 0.8  # most fixes should match

    res = rt.derive_speeds(net, matched, ps, default_pos_sigma_m=5.0, min_baseline_m=75.0)
    iv, eo = res["intervals"], res["edge_observations"]
    good = iv[iv["quality"]]
    assert len(good) > 0
    # recovered median speed should be in the right ballpark of the true 10 m/s
    assert 7.0 < good["speed_mps"].median() < 13.0

    # feed straight into the existing cleaning/aggregation/routing pipeline
    clean = rt.filter_by_speed(eo, min_speed=1, max_speed=80, unit="mph",
                                drop_unmatched=True, mad_outliers=True, per_edge=True)
    assert len(clean) > 0
    agg = rt.aggregate_speeds(clean, statistic="both", output_unit="mph", by_edge=True)
    assert len(agg) > 0

    info = rt.assign_speeds(net, clean, statistic="median")
    assert info["n_edges_observed"] > 0
    router = rt.Router(net)
    route = router.route(tuple(coords_e[0]), tuple(coords_n[-1]), mode="time")
    assert route["travel_time_s"] > 0
    implied_speed = route["distance_m"] / route["travel_time_s"]
    assert 5.0 < implied_speed < 15.0


# --------------------------------------------------------------------------- #
# C. mapping export on top of an assigned network
# --------------------------------------------------------------------------- #
def _assigned_grid_network(tmp):
    net_path = os.path.join(tmp, "grid.geojson")
    feats = []
    for i in range(3):
        for j in range(3):
            lon, lat = -77.30 + j * 0.005, 38.68 + i * 0.005
            if j < 2:
                feats.append({"type": "Feature",
                              "properties": {"highway": "residential", "oneway": "no"},
                              "geometry": {"type": "LineString",
                                           "coordinates": [[lon, lat], [lon + 0.005, lat]]}})
            if i < 2:
                feats.append({"type": "Feature",
                              "properties": {"highway": "residential", "oneway": "no"},
                              "geometry": {"type": "LineString",
                                           "coordinates": [[lon, lat], [lon, lat + 0.005]]}})
    with open(net_path, "w") as fh:
        json.dump({"type": "FeatureCollection", "features": feats}, fh)
    net = rt.Network.from_geojson(net_path)
    rows = []
    for _u, _v, d in net.graph.edges(data=True):
        rows.append({"edge_id": d["edge_id"], "speed_mps": 10.0,
                     "time": "2024-06-01T08:00:00"})
    matched = pd.DataFrame(rows)
    rt.assign_speeds(net, matched, statistic="median")
    return net


def test_to_geojson_export():
    tmp = tempfile.mkdtemp()
    net = _assigned_grid_network(tmp)
    fc = rt.to_geojson(net, speed_unit="mph")
    assert fc["type"] == "FeatureCollection"
    assert len(fc["features"]) > 0
    assert all(f["properties"]["speed"] is not None for f in fc["features"])


def test_plot_speed_map():
    pytest.importorskip("matplotlib")
    tmp = tempfile.mkdtemp()
    net = _assigned_grid_network(tmp)
    out = os.path.join(tmp, "speeds.png")
    fig = rt.plot_speed_map(net, out, speed_unit="mph")
    assert os.path.exists(out)
    import matplotlib.pyplot as plt
    plt.close(fig)
