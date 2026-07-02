"""Tests for the speed-derivation, dwell-aware cleaning, segment-speed and
routing-error features added on top of the base pipeline."""
import json
import os
import tempfile
import warnings

import numpy as np
import pandas as pd
import pytest

import roadtraffic as rt


# --------------------------------------------------------------------------- #
# fixtures / helpers
# --------------------------------------------------------------------------- #
def _grid_geojson(path, n=4, spacing=0.005, lon0=-77.30, lat0=38.68):
    feats = []
    for i in range(n):
        for j in range(n):
            lon, lat = lon0 + j * spacing, lat0 + i * spacing
            if j < n - 1:
                feats.append({"type": "Feature",
                              "properties": {"highway": "residential", "oneway": "no"},
                              "geometry": {"type": "LineString",
                                           "coordinates": [[lon, lat], [lon + spacing, lat]]}})
            if i < n - 1:
                feats.append({"type": "Feature",
                              "properties": {"highway": "residential", "oneway": "no"},
                              "geometry": {"type": "LineString",
                                           "coordinates": [[lon, lat], [lon, lat + spacing]]}})
    with open(path, "w") as fh:
        json.dump({"type": "FeatureCollection", "features": feats}, fh)


def _write_points_no_speed(path, lon0=-77.30, lat0=38.68, spacing=0.005, n=4):
    """Two vehicles driving east along the bottom row, hour-dependent timing,
    NO speed column. Peak hours (8, 17) slow; nights fast."""
    rows = []
    for hour in range(24):
        base_mph = 10.0 if hour in (7, 8, 9, 16, 17, 18) else (36.0 if hour < 6 else 24.0)
        base_mps = base_mph * 0.44704
        for trip in range(2):
            tid = f"veh_{hour:02d}_{trip}"
            t0 = pd.Timestamp("2024-06-01") + pd.Timedelta(hours=hour, minutes=trip * 5)
            sec = 0
            for j in range(n - 1):
                for frac in (0.25, 0.5, 0.75):
                    lon = lon0 + (j + frac) * spacing
                    lat = lat0
                    ts = (t0 + pd.Timedelta(seconds=int(sec))).isoformat()
                    rows.append({"uid": tid, "lon": lon, "lat": lat,
                                 "timestamp": ts})
                    seg_m = 0.25 * spacing * 86900.0
                    sec += max(1, round(seg_m / base_mps))
    pd.DataFrame(rows).to_csv(path, index=False)


# --------------------------------------------------------------------------- #
# A. speed derivation
# --------------------------------------------------------------------------- #
def test_derive_speed_computes_expected_value():
    tmp = tempfile.mkdtemp()
    p = os.path.join(tmp, "pts.csv")
    # one vehicle, 0.001 deg lon steps (~86.9 m) every 20 s -> ~4.34 m/s
    rows = [{"uid": "A", "lon": -77.30 + 0.001 * k, "lat": 38.68,
             "timestamp": (pd.Timestamp("2024-06-01 08:00:00")
                           + pd.Timedelta(seconds=20 * k)).isoformat()}
            for k in range(6)]
    pd.DataFrame(rows).to_csv(p, index=False)
    pts = rt.load_points(p, derive_speed=True, id_col="uid")
    assert len(pts) == 6
    speeds = pts.df["speed_mps"].to_numpy()
    assert np.allclose(speeds, speeds[0])           # constant speed
    assert 4.0 < speeds[0] < 4.7                     # ~4.34 m/s


def test_missing_speed_without_derive_loads_position_only():
    # No speed column and derive_speed=False is valid: position+time-only data
    # for a matcher + roadtraffic.speeds.derive_speeds pipeline.
    tmp = tempfile.mkdtemp()
    p = os.path.join(tmp, "pts.csv")
    pd.DataFrame([{"uid": "A", "lon": -77.3, "lat": 38.68,
                   "timestamp": "2024-06-01T08:00:00"}]).to_csv(p, index=False)
    pts = rt.load_points(p)
    assert "speed_mps" not in pts.df.columns
    assert len(pts) == 1


def test_missing_lonlat_still_raises():
    tmp = tempfile.mkdtemp()
    p = os.path.join(tmp, "pts.csv")
    pd.DataFrame([{"uid": "A", "timestamp": "2024-06-01T08:00:00"}]).to_csv(p, index=False)
    with pytest.raises(ValueError, match="longitude/latitude"):
        rt.load_points(p)


def test_derive_speed_without_id_raises():
    tmp = tempfile.mkdtemp()
    p = os.path.join(tmp, "pts.csv")
    pd.DataFrame([{"lon": -77.3, "lat": 38.68, "timestamp": "2024-06-01T08:00:00"},
                  {"lon": -77.2, "lat": 38.68, "timestamp": "2024-06-01T08:00:20"}]
                 ).to_csv(p, index=False)
    with pytest.raises(ValueError, match="unique-id"):
        rt.load_points(p, derive_speed=True)


# --------------------------------------------------------------------------- #
# B. save round-trip
# --------------------------------------------------------------------------- #
def test_save_points_roundtrip_csv_and_geojson():
    tmp = tempfile.mkdtemp()
    src = os.path.join(tmp, "src.csv")
    _write_points_no_speed(src)
    pts = rt.load_points(src, derive_speed=True, id_col="uid")

    csv_out = os.path.join(tmp, "out.csv")
    rt.save_points(pts, csv_out, speed_unit="mph")
    back = pd.read_csv(csv_out)
    assert {"point_id", "traj_id", "lon", "lat", "time",
            "speed_mps", "speed_mph"}.issubset(back.columns)
    assert len(back) == len(pts)

    gj_out = os.path.join(tmp, "out.geojson")
    rt.save_points(pts, gj_out)
    gj = json.load(open(gj_out))
    assert gj["type"] == "FeatureCollection"
    assert gj["features"][0]["geometry"]["type"] == "Point"
    assert "speed_mps" in gj["features"][0]["properties"]


# --------------------------------------------------------------------------- #
# E. dwell-aware trajectory cleaning
# --------------------------------------------------------------------------- #
def test_filter_trajectory_speed_drops_dwell_keeps_slow_moving():
    rows = []
    t0 = pd.Timestamp("2024-06-01 08:00:00")
    # 10 genuinely-moving points: ~43 m every 10 s (~2 mph) -- slow but each step
    # clears the 25 m dwell radius, so they must be kept as congestion.
    for k in range(10):
        rows.append({"traj_id": "A", "edge_id": 0,
                     "lon": -77.30 + 5e-4 * k, "lat": 38.68,
                     "time": (t0 + pd.Timedelta(seconds=10 * k)).isoformat(),
                     "speed_mps": 1.0})
    # then a parked dwell well away: 8 points at one spot, 30 s apart (210 s)
    dwell_t = t0 + pd.Timedelta(seconds=200)
    for k in range(8):
        rows.append({"traj_id": "A", "edge_id": 0,
                     "lon": -77.20, "lat": 38.68,
                     "time": (dwell_t + pd.Timedelta(seconds=30 * k)).isoformat(),
                     "speed_mps": 0.0})
    matched = pd.DataFrame(rows)
    out = rt.filter_trajectory_speed(matched, dwell_radius_m=25, dwell_min_s=120)
    # the 8 stationary dwell points are removed; every slow-but-moving point kept
    assert len(out) == 10
    assert (out["speed_mps"] == 1.0).all()


def test_filter_trajectory_speed_drops_missing_speed():
    rows = [{"traj_id": "A", "edge_id": 0, "lon": -77.30 + 5e-5 * k, "lat": 38.68,
             "time": (pd.Timestamp("2024-06-01 08:00:00")
                      + pd.Timedelta(seconds=5 * k)).isoformat(),
             "speed_mps": (np.nan if k == 2 else 5.0)} for k in range(6)]
    out = rt.filter_trajectory_speed(pd.DataFrame(rows))
    assert out["speed_mps"].notna().all()
    assert len(out) == 5


def test_filter_trajectory_speed_requires_columns():
    with pytest.raises(ValueError, match="traj_id|columns"):
        rt.filter_trajectory_speed(pd.DataFrame({"speed_mps": [1.0]}))


# --------------------------------------------------------------------------- #
# C. classify_hours + three segment speeds + period routing
# --------------------------------------------------------------------------- #
def test_classify_hours_override_and_auto():
    tmp = tempfile.mkdtemp()
    src = os.path.join(tmp, "src.csv")
    _write_points_no_speed(src)
    pts = rt.load_points(src, derive_speed=True, id_col="uid")
    matched = rt.NearestMatcher(_load_grid(tmp), max_dist=80).match(pts)
    clean = rt.filter_by_speed(matched, min_speed=1, max_speed=80, unit="mph")

    over = rt.classify_hours(clean, peak_hours=[7, 8, 9], offpeak_hours=[0, 1, 2])
    assert over["source"] == "override"
    assert over["peak_hours"] == [7, 8, 9]

    auto = rt.classify_hours(clean, statistic="median")
    assert auto["source"] == "auto"
    # rush hours (slow) should land in the peak block
    assert 8 in auto["peak_hours"] and 17 in auto["peak_hours"]


def test_segment_speeds_and_period_routing():
    tmp = tempfile.mkdtemp()
    src = os.path.join(tmp, "src.csv")
    _write_points_no_speed(src)
    net = _load_grid(tmp)
    pts = rt.load_points(src, derive_speed=True, id_col="uid")
    matched = rt.NearestMatcher(net, max_dist=80).match(pts)
    clean = rt.filter_by_speed(matched, min_speed=1, max_speed=80, unit="mph")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        info = rt.assign_segment_speeds(net, clean, statistic="median")
    assert info["coverage"]["overall"] > 0
    for _u, _v, d in net.graph.edges(data=True):
        if d.get("obs_speed_mps_peak") and d.get("obs_speed_mps_offpeak"):
            break

    router = rt.Router(net, default_speed_mps=11.0)
    o = (-77.30, 38.68)
    d = (-77.30 + 0.015, 38.68)
    r_peak = router.route(o, d, mode="time", period="peak")
    r_off = router.route(o, d, mode="time", period="offpeak")
    # peak congestion -> at least as slow as off-peak on the same corridor
    assert r_peak["travel_time_s"] >= r_off["travel_time_s"] - 1e-6
    assert "n_edges_default" in r_peak


# --------------------------------------------------------------------------- #
# D. routing error handling
# --------------------------------------------------------------------------- #
def test_routing_no_path_raises_actionable():
    tmp = tempfile.mkdtemp()
    net = _load_grid(tmp)
    # add an isolated 2-node component
    net.graph.add_edge((-80.0, 40.0), (-80.001, 40.0),
                       edge_id=10 ** 9, length_m=90.0, geometry=None)
    router = rt.Router(net)
    with pytest.raises(ValueError, match="No .* route exists"):
        router.route((-80.0, 40.0), (-77.30, 38.68), mode="distance")


def test_routing_bad_period_raises():
    tmp = tempfile.mkdtemp()
    net = _load_grid(tmp)
    router = rt.Router(net)
    with pytest.raises(ValueError, match="period"):
        router.route((-77.30, 38.68), (-77.285, 38.68), mode="time", period="rush")


# --------------------------------------------------------------------------- #
def _load_grid(tmp):
    net_path = os.path.join(tmp, "grid.geojson")
    _grid_geojson(net_path)
    return rt.Network.from_geojson(net_path)
