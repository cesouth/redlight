import networkx as nx
import numpy as np

import redlight as rl
from conftest import drive_along_road
from redlight.speeds import _SourceDistCache


def _match(net, make_points_csv, rows):
    pts = rl.load_points(make_points_csv(rows))
    return pts, rl.NearestMatcher(net, max_dist=60).match(pts)


def test_constant_speed_recovered(straight_net, make_points_csv):
    # 0.00015 deg / 10 s at the equator ~ 1.6698 m/s
    pts, m = _match(straight_net, make_points_csv, drive_along_road(6, traj="a"))
    res = rl.derive_speeds(straight_net, m, pts)
    assert len(res["intervals"]) == 5
    np.testing.assert_allclose(res["intervals"]["speed_mps"], 1.6698, rtol=1e-3)


def test_no_traj_id_input_produces_intervals(straight_net, make_points_csv):
    """Regression: traj-id-less input silently produced empty output."""
    pts, m = _match(straight_net, make_points_csv, drive_along_road(6))
    assert "traj_id" not in m.columns
    res = rl.derive_speeds(straight_net, m, pts)
    assert len(res["intervals"]) == 5
    assert len(res["edge_observations"]) > 0


def test_interval_id_present_and_shared(straight_net, make_points_csv):
    pts, m = _match(straight_net, make_points_csv, drive_along_road(6, traj="a"))
    res = rl.derive_speeds(straight_net, m, pts)
    assert "interval_id" in res["intervals"].columns
    assert "interval_id" in res["edge_observations"].columns
    assert set(res["edge_observations"]["interval_id"]) == \
        set(res["intervals"]["interval_id"])


def test_edge_observations_cover_both_directions(straight_net, make_points_csv):
    """Attribution to both directions of a two-way road is deliberate."""
    pts, m = _match(straight_net, make_points_csv, drive_along_road(3, traj="a"))
    res = rl.derive_speeds(straight_net, m, pts)
    eids = {int(e) for e in straight_net.edge_ids}
    assert set(res["edge_observations"]["edge_id"]) == eids


def test_source_dist_cache_reapplies_cutoff():
    """Regression: a warm larger-cutoff cache bypassed smaller cutoffs."""
    g = nx.Graph()
    g.add_edge("a", "b", length_m=400.0)
    g.add_edge("b", "c", length_m=400.0)
    cache = _SourceDistCache(g)
    d, path = cache.query("a", "c", cutoff=1200.0)
    assert d == 800.0 and path == ["a", "b", "c"]
    d2, path2 = cache.query("a", "c", cutoff=300.0)  # same warm cache
    assert d2 is None and path2 is None


def test_nan_snap_fails_quality(straight_net, make_points_csv):
    """Regression: non-finite snap distances used to PASS the quality gate."""
    pts, m = _match(straight_net, make_points_csv, drive_along_road(3, traj="a"))
    m = m.copy()
    m["snap_dist_m"] = np.nan  # simulate unquantified match quality
    res = rl.derive_speeds(straight_net, m, pts)
    assert len(res["intervals"]) == 2
    assert not res["intervals"]["quality"].any()


def test_one_sided_nan_snap_fails_quality(straight_net, make_points_csv):
    """Regression: only a NaN at BOTH endpoints failed quality; one finite +
    one NaN snap silently passed even though one endpoint's match quality
    was completely unknown."""
    pts, m = _match(straight_net, make_points_csv, drive_along_road(3, traj="a"))
    m = m.copy()
    m.loc[0, "snap_dist_m"] = np.nan  # only the first fix is unquantified
    res = rl.derive_speeds(straight_net, m, pts)
    assert len(res["intervals"]) == 2
    assert not res["intervals"]["quality"].iloc[0]


def test_no_snap_column_keeps_benefit_of_doubt(straight_net, make_points_csv):
    pts, m = _match(straight_net, make_points_csv,
                    drive_along_road(3, traj="a", dlon=0.0006, dt_s=30))
    m = m.drop(columns=["snap_dist_m"])
    res = rl.derive_speeds(straight_net, m, pts)
    assert res["intervals"]["quality"].all()


def test_quality_flags_low_snr(straight_net, make_points_csv):
    # 1.67 m/s * 10 s = 16.7 m displacement < 3 * sqrt(2)*15 m sigma -> low SNR
    pts, m = _match(straight_net, make_points_csv, drive_along_road(3, traj="a"))
    res = rl.derive_speeds(straight_net, m, pts, default_pos_sigma_m=15.0)
    assert not res["intervals"]["quality"].any()
    # with honest 1 m sigma the same intervals are fine
    res2 = rl.derive_speeds(straight_net, m, pts, default_pos_sigma_m=1.0)
    assert res2["intervals"]["quality"].all()


def test_min_baseline_merges_intervals(straight_net, make_points_csv):
    pts, m = _match(straight_net, make_points_csv, drive_along_road(7, traj="a"))
    res = rl.derive_speeds(straight_net, m, pts, min_baseline_m=40.0)
    # each hop ~16.7 m -> merged in groups of >=3 hops
    assert 0 < len(res["intervals"]) <= 2
    assert (res["intervals"]["distance_m"] >= 40.0).all()
    np.testing.assert_allclose(res["intervals"]["speed_mps"], 1.6698, rtol=1e-3)


def test_min_baseline_shortfall_flags_quality_false(straight_net, make_points_csv):
    """Regression: a trajectory that ran out of points before reaching
    min_baseline_m still had its final (short) interval silently pass
    quality, violating the documented SNR-boosting guarantee."""
    pts, m = _match(straight_net, make_points_csv, drive_along_road(3, traj="a"))
    res = rl.derive_speeds(straight_net, m, pts, min_baseline_m=1000.0,
                           default_pos_sigma_m=1.0)
    assert len(res["intervals"]) == 1
    assert res["intervals"]["distance_m"].iloc[0] < 1000.0
    assert not res["intervals"]["quality"].iloc[0]


def test_unmatched_fix_breaks_interval(straight_net, make_points_csv):
    pts, m = _match(straight_net, make_points_csv, drive_along_road(5, traj="a"))
    m.loc[2, "edge_id"] = -1  # simulate an unmatchable middle fix
    res = rl.derive_speeds(straight_net, m, pts)
    # intervals never silently bridge the unmatched fix
    pairs = list(zip(res["intervals"]["point_id_from"],
                     res["intervals"]["point_id_to"]))
    assert (0, 1) in pairs and (3, 4) in pairs
    assert (1, 3) not in pairs


def test_parallel_roads_not_conflated(tmp_path, make_points_csv):
    """Fixes on the straight road must not take the 'same road' shortcut
    against a parallel road sharing both endpoints."""
    from conftest import line_feature, write_geojson
    path = write_geojson(tmp_path / "par.json", [
        line_feature([[0, 0], [0.001, 0]], name="straight"),
        line_feature([[0, 0], [0.0005, 0.0004], [0.001, 0]], name="detour"),
    ])
    net = rl.Network.from_geojson(path)
    rows = drive_along_road(3, traj="a", start_lon=0.0002, dlon=0.0003)
    pts = rl.load_points(make_points_csv(rows))
    m = rl.NearestMatcher(net, max_dist=30).match(pts)
    res = rl.derive_speeds(net, m, pts)
    # distance per 10 s hop ~33.4 m along the straight road
    np.testing.assert_allclose(res["intervals"]["speed_mps"], 3.34, rtol=0.02)


def test_interval_id_start_lets_two_runs_be_combined(straight_net,
                                                     make_points_csv):
    """Each derive_speeds call numbers intervals from 0, so two runs collide
    on interval_id; offsetting the second makes the concatenation safe to
    aggregate instead of silently losing half the data."""
    import pandas as pd
    import pytest

    pts_a, m_a = _match(straight_net, make_points_csv,
                        drive_along_road(6, traj="a"))
    res_a = rl.derive_speeds(straight_net, m_a, pts_a)
    obs_a = res_a["edge_observations"]

    # same trajectory shape, processed as a separate run
    pts_b, m_b = _match(straight_net, make_points_csv,
                        drive_along_road(6, traj="b"))
    colliding = rl.derive_speeds(straight_net, m_b, pts_b)["edge_observations"]
    assert colliding["interval_id"].min() == 0          # restarts from 0

    with pytest.raises(ValueError, match="interval_id"):
        rl.aggregate_speeds(pd.concat([obs_a, colliding], ignore_index=True))

    offset = int(res_a["intervals"]["interval_id"].max()) + 1
    obs_b = rl.derive_speeds(straight_net, m_b, pts_b,
                             interval_id_start=offset)["edge_observations"]
    assert obs_b["interval_id"].min() == offset
    combined = pd.concat([obs_a, obs_b], ignore_index=True)
    out = rl.aggregate_speeds(combined, output_unit="mps")
    assert out["n"].sum() == len(obs_a["interval_id"].unique()) + \
        len(obs_b["interval_id"].unique())


def _tied_rows(n_tied=150):
    """``n_tied`` fixes sharing one timestamp, then one fix 10 s later."""
    rows = [{"id": "a", "lon": 0.0005 + 1e-5 * k, "lat": 1e-5,
             "time": "2026-06-01T08:00:00"} for k in range(n_tied)]
    rows.append({"id": "a", "lon": 0.0005 + 1e-5 * n_tied, "lat": 1e-5,
                 "time": "2026-06-01T08:00:10"})
    return rows


def test_duplicate_timestamps_resolve_in_file_order(straight_net,
                                                    make_points_csv):
    """Tied timestamps must resolve by a stated rule, not by pandas' sort.

    Regression for F-3.3: ``sort_values`` defaults to quicksort, which is not
    stable, so with enough tied rows the fix that survived to anchor the next
    interval was chosen by the sort implementation rather than by the data.
    The documented rule is that the last tied fix in file order anchors it.
    """
    n_tied = 150
    pts, m = _match(straight_net, make_points_csv, _tied_rows(n_tied))
    res = rl.derive_speeds(straight_net, m, pts)
    assert len(res["intervals"]) == 1
    assert int(res["intervals"]["point_id_from"].iloc[0]) == n_tied - 1


def test_duplicate_timestamps_warn(straight_net, make_points_csv):
    """Contradictory fixes at one instant are reported, not silently resolved."""
    import pytest
    pts, m = _match(straight_net, make_points_csv, _tied_rows(4))
    with pytest.warns(UserWarning, match="duplicate timestamp"):
        rl.derive_speeds(straight_net, m, pts)
