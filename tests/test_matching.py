import numpy as np
import pytest

import redlight as rl
from conftest import drive_along_road
from redlight.matching import _CSRDistCache


def _load(make_points_csv, rows):
    return rl.load_points(make_points_csv(rows))


def test_nearest_matches_on_road(straight_net, make_points_csv):
    pts = _load(make_points_csv, drive_along_road(4, traj="a"))
    m = rl.NearestMatcher(straight_net, max_dist=50).match(pts)
    assert (m["edge_id"] != -1).all()
    assert np.isfinite(m["snap_dist_m"]).all()
    assert len(m) == len(pts)


def test_nearest_far_point_is_unmatched(straight_net, make_points_csv):
    rows = drive_along_road(2, traj="a")
    rows.append({"id": "a", "lon": 0.005, "lat": 0.01,  # ~1.1 km north
                 "time": "2026-06-01T08:10:00"})
    pts = _load(make_points_csv, rows)
    m = rl.NearestMatcher(straight_net, max_dist=50).match(pts)
    assert m["edge_id"].tolist()[-1] == -1
    assert np.isnan(m["snap_dist_m"].iloc[-1])


def test_hmm_requires_traj_id(straight_net, make_points_csv):
    pts = _load(make_points_csv, drive_along_road(3))  # no id column
    with pytest.raises(ValueError, match="trajectory id"):
        rl.HMMMatcher(straight_net).match(pts)


def test_hmm_off_network_first_fix_does_not_crash(straight_net, make_points_csv):
    """Regression: used to raise ValueError (max() on an empty dict)."""
    rows = [{"id": "a", "lon": 0.0, "lat": 0.01, "time": "2026-06-01T07:59:50"}]
    rows += drive_along_road(4, traj="a")
    pts = _load(make_points_csv, rows)
    m = rl.HMMMatcher(straight_net, max_dist=50).match(pts)
    assert m["edge_id"].iloc[0] == -1          # off-network fix stays unmatched
    assert (m["edge_id"].iloc[1:] != -1).all()  # chain restarts and matches


def test_hmm_mid_trajectory_gap_reports_unmatched(straight_net, make_points_csv):
    """Regression: carry-forward used to fabricate a match (edge id of a
    previous state with snap NaN) for candidate-less fixes."""
    rows = drive_along_road(2, traj="a")
    rows.append({"id": "a", "lon": 0.0009, "lat": 0.002,  # ~220 m off
                 "time": "2026-06-01T08:00:25"})
    rows += drive_along_road(2, traj="a", start_lon=0.0011,
                             t0="2026-06-01 08:00:30")
    pts = _load(make_points_csv, rows)
    m = rl.HMMMatcher(straight_net, max_dist=50).match(pts).sort_values("point_id")
    assert m["edge_id"].iloc[2] == -1
    assert np.isnan(m["snap_dist_m"].iloc[2])
    assert (m["edge_id"].drop(m.index[2]) != -1).all()


def test_hmm_gap_transition_uses_correct_anchor(grid_net, make_points_csv):
    """Regression: after a GPS dropout (candidate-less fix), the transition
    distance used to compare against the raw distance from the immediately-
    prior (off-network, noisy) fix instead of the last fix that actually
    anchored a Viterbi state -- corrupting the transition score right where
    an intersection makes the wrong candidate a real possibility."""
    rows = drive_along_road(5, traj="a", start_lon=0.0002, dlon=0.00022, lat=3e-5)
    # a wild, far off-network fix mid-trajectory simulates a GPS dropout
    rows.insert(3, {"id": "a", "lon": rows[2]["lon"], "lat": 0.03,
                    "time": "2026-06-01T08:00:35"})
    pts = rl.load_points(make_points_csv(rows))
    m = rl.HMMMatcher(grid_net, max_dist=60).match(pts).sort_values("point_id")
    assert m["edge_id"].iloc[3] == -1  # the dropout fix itself stays unmatched
    matched_others = m["edge_id"].drop(m.index[3])
    assert (matched_others != -1).all()
    for eid in matched_others:
        (u, v) = grid_net.edge_endpoints(int(eid))
        assert u[1] == 0.0 and v[1] == 0.0  # stayed on the bottom row


def test_hmm_all_off_network(straight_net, make_points_csv):
    rows = [{"id": "a", "lon": 0.002 * k, "lat": 0.02, "time": f"2026-06-01T08:00:{k:02d}"}
            for k in range(3)]
    pts = _load(make_points_csv, rows)
    m = rl.HMMMatcher(straight_net, max_dist=50).match(pts)
    assert (m["edge_id"] == -1).all()


def test_csr_dist_cache_refresh_bumps_recency(grid_net):
    """Regression: refreshing a cached entry (larger cutoff) didn't move it
    to the MRU end, so a just-recomputed, actively-used source could be
    evicted right after being recomputed instead of a genuinely cold entry."""
    csr, node_int, int_node = grid_net.csgraph()
    cache = _CSRDistCache(csr, node_int, maxsize=3)
    a, b, c, d = int_node[0], int_node[1], int_node[2], int_node[3]
    cache.lookup(a, b, 50.0)    # caches a
    cache.lookup(b, a, 50.0)    # caches b
    cache.lookup(c, a, 50.0)    # caches c; at maxsize == 3: {a, b, c}
    cache.lookup(a, b, 500.0)   # refresh a (larger cutoff) -- must bump to MRU
    cache.lookup(d, a, 50.0)    # caches d; over maxsize -> evicts the LRU entry
    assert node_int[a] in cache._cache      # just refreshed -> must survive
    assert node_int[b] not in cache._cache  # now the oldest untouched entry


def test_matchers_share_output_schema(straight_net, make_points_csv):
    pts = _load(make_points_csv, drive_along_road(3, traj="a"))
    a = rl.NearestMatcher(straight_net).match(pts)
    b = rl.HMMMatcher(straight_net).match(pts)
    assert set(a.columns) == set(b.columns)
    assert len(a) == len(b) == len(pts)


def test_hmm_prefers_trajectory_consistency(grid_net, make_points_csv):
    """Points along the bottom row match the bottom-row edges, not the
    verticals, even when a fix drifts toward an intersection."""
    rows = drive_along_road(8, traj="a", start_lon=0.0002, dlon=0.00022,
                            lat=3e-5)
    pts = _load(make_points_csv, rows)
    m = rl.HMMMatcher(grid_net, max_dist=60).match(pts)
    assert (m["edge_id"] != -1).all()
    # all matched edges lie on the bottom row (lat 0 for both endpoints)
    for eid in m["edge_id"]:
        (u, v) = grid_net.edge_endpoints(int(eid))
        assert u[1] == 0.0 and v[1] == 0.0
