"""End-to-end exercise of the full pipeline on synthetic data.

Builds a small grid road network as GeoJSON, simulates GPS points traversing it
at time-of-day-dependent speeds, then runs: load -> match (both matchers) ->
filter -> aggregate -> peak analysis -> assign speeds -> route.
"""
import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

import roadtraffic as rt


def build_grid_geojson(path, n=6, spacing_deg=0.005, lon0=-77.30, lat0=38.68):
    """A simple n x n lon/lat grid of two-way streets."""
    feats = []
    for i in range(n):
        for j in range(n):
            lon = lon0 + j * spacing_deg
            lat = lat0 + i * spacing_deg
            if j < n - 1:  # horizontal edge
                feats.append({
                    "type": "Feature",
                    "properties": {"highway": "residential", "oneway": "no"},
                    "geometry": {"type": "LineString",
                                 "coordinates": [[lon, lat],
                                                 [lon + spacing_deg, lat]]},
                })
            if i < n - 1:  # vertical edge
                feats.append({
                    "type": "Feature",
                    "properties": {"highway": "residential", "oneway": "no"},
                    "geometry": {"type": "LineString",
                                 "coordinates": [[lon, lat],
                                                 [lon, lat + spacing_deg]]},
                })
    with open(path, "w") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f)


def simulate_points(path, lon0=-77.30, lat0=38.68, spacing_deg=0.005,
                    n_rows=6, seed=0):
    """Simulate vehicles driving east along the bottom row at hour-dependent speed.

    Peak (slow) at hour 8 and 17; off-peak (fast) overnight.
    """
    rng = np.random.default_rng(seed)
    rows = []
    pid = 0
    for hour in range(24):
        # base speed mph: slow at rush hours, fast overnight
        if hour in (7, 8, 9, 16, 17, 18):
            base = 12.0
        elif hour in (0, 1, 2, 3, 4, 5):
            base = 38.0
        else:
            base = 28.0
        for trip in range(3):  # a few trips per hour
            tid = f"h{hour}_t{trip}"
            # drive east along bottom row, sampling points between nodes
            for j in range(n_rows - 1):
                for frac in (0.25, 0.5, 0.75):
                    lon = lon0 + (j + frac) * spacing_deg
                    lat = lat0 + rng.normal(0, 0.00002)  # tiny GPS noise
                    spd = max(1.0, base + rng.normal(0, 2.0))
                    ts = pd.Timestamp("2024-06-01") + pd.Timedelta(
                        hours=hour, minutes=trip * 5, seconds=j * 10
                    )
                    rows.append({"uid": tid, "lon": lon, "lat": lat,
                                 "timestamp": ts.isoformat(), "speed": spd})
                    pid += 1
    pd.DataFrame(rows).to_csv(path, index=False)


def main():
    tmp = tempfile.mkdtemp()
    net_path = os.path.join(tmp, "grid.geojson")
    pts_path = os.path.join(tmp, "points.csv")
    build_grid_geojson(net_path)
    simulate_points(pts_path)

    print("== Load network ==")
    net = rt.Network.from_geojson(net_path)
    print(f"nodes={net.number_of_nodes()} edges={net.number_of_edges()} "
          f"crs={net.crs_metric.to_epsg()}")

    print("\n== Load points ==")
    pts = rt.load_points(pts_path, speed_unit="mph")
    print(f"loaded {len(pts)} points, has_traj={pts.has_traj}")

    print("\n== Nearest matcher ==")
    nm = rt.NearestMatcher(net, max_dist=60)
    m_near = nm.match(pts)
    print(f"matched edges (nearest): "
          f"{(m_near['edge_id'] != -1).sum()}/{len(m_near)}")

    print("\n== HMM matcher ==")
    hm = rt.HMMMatcher(net, sigma_z=6, beta=30, max_dist=60)
    m_hmm = hm.match(pts)
    print(f"matched edges (hmm): {(m_hmm['edge_id'] != -1).sum()}/{len(m_hmm)}")

    print("\n== Filter ==")
    clean = rt.filter_by_speed(m_near, min_speed=2, max_speed=80, unit="mph",
                               mad_outliers=True, per_edge=True)
    print(f"after filter: {len(clean)} (from {len(m_near)})")

    print("\n== Aggregate hourly (both stats) ==")
    agg = rt.aggregate_speeds(clean, block_hours=1, statistic="both",
                              output_unit="mph")
    print(agg[["block_label", "n", "mean_speed", "sem_speed",
               "median_speed", "iqr_speed"]].to_string(index=False))

    print("\n== Aggregate 6-hour blocks ==")
    agg6 = rt.aggregate_speeds(clean, block_hours=6, statistic="mean",
                               output_unit="mph")
    print(agg6[["block_label", "n", "mean_speed", "ci95_low",
                "ci95_high"]].to_string(index=False))

    print("\n== Peak analysis ==")
    pk = rt.peak_analysis(agg, statistic="mean", n_peak=2, n_offpeak=2)
    print("peak (slowest):", [(r["block_label"], round(r["mean_speed"], 1))
                              for r in pk["peak"]])
    print("off-peak (fastest):", [(r["block_label"], round(r["mean_speed"], 1))
                                  for r in pk["off_peak"]])

    print("\n== Assign speeds + route ==")
    info = rt.assign_speeds(net, clean, statistic="median")
    print(f"edges observed: {info['n_edges_observed']}/{info['n_edges_total']}")
    router = rt.Router(net)
    o = (-77.30, 38.68)
    d = (-77.30 + 0.025, 38.68 + 0.025)
    r_time = router.route(o, d, mode="time")
    r_dist = router.route(o, d, mode="distance")
    print(f"time route:  {r_dist['distance_m']:.0f} m via "
          f"{r_time['n_edges']} edges, {r_time['travel_time_s']:.0f} s")
    print(f"dist route:  {r_dist['distance_m']:.0f} m, "
          f"{r_dist['travel_time_s']:.0f} s")
    print("\nALL STAGES OK")
    return net, m_near, m_hmm, clean, agg, pk, r_time, r_dist


def test_pipeline_runs():
    net, m_near, m_hmm, clean, agg, pk, r_time, r_dist = main()
    # matchers attribute every point on this fully-covered grid
    assert (m_near["edge_id"] != -1).all()
    assert (m_hmm["edge_id"] != -1).all()
    # cleaning keeps a sensible amount of data
    assert len(clean) > 0
    # all 24 hourly bins present
    assert agg["block_label"].nunique() == 24
    # peak (slowest) speed is well below off-peak (fastest) speed
    peak_speed = pk["peak"][0]["mean_speed"]
    offpeak_speed = pk["off_peak"][-1]["mean_speed"]
    assert peak_speed < offpeak_speed
    # time-optimal route is no slower than the distance-optimal route's time
    assert r_time["travel_time_s"] <= r_dist["travel_time_s"] + 1e-6


def test_mode_screening_recovers_vehicle_only_speeds(straight_net, make_points_csv):
    """Walkers on the same road drag the network speed down; screening must
    move the answer back toward the vehicles-only truth."""
    from conftest import track_along_road, walk_along_road

    veh_rows, ped_rows = [], []
    for i in range(6):
        veh_rows += track_along_road(8, speed_mps=12.0, traj=f"veh{i}",
                                     t0=f"2026-06-01 0{i}:00:00")
    for i in range(6):
        ped_rows += walk_along_road(25, traj=f"ped{i}",
                                    t0=f"2026-06-01 1{i}:00:00")

    def intervals(rows, name):
        path = make_points_csv(rows, name=name)
        pts = rt.load_points(path, id_col="id")
        matched = rt.HMMMatcher(straight_net, max_dist=60).match(pts)
        return rt.derive_speeds(straight_net, matched, pts)["intervals"]

    truth = intervals(veh_rows, "veh.csv")["speed_mps"].median()
    mixed = intervals(veh_rows + ped_rows, "mixed.csv")
    contaminated = mixed["speed_mps"].median()

    movers = rt.classify_movers(mixed, threshold=3.0, unit="mps")
    screened = rt.filter_by_mode(mixed, movers)["speed_mps"].median()

    assert contaminated < truth * 0.9          # contamination is real
    assert abs(screened - truth) < abs(contaminated - truth)
    assert screened == pytest.approx(truth, rel=0.10)


if __name__ == "__main__":
    main()
