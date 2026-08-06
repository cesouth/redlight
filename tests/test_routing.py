import pandas as pd
import pytest

import redlight as rl
from conftest import line_feature, write_geojson


def _hourly_obs(edge_id, hour, speed_mps, n=3):
    return [{"edge_id": edge_id, "speed_mps": speed_mps,
             "time": pd.Timestamp(f"2026-06-01 {hour:02d}:10:{k:02d}")}
            for k in range(n)]


def test_distance_route_on_grid(grid_net):
    r = rl.Router(grid_net)
    res = r.route((0, 0), (0.002, 0.002), mode="distance")
    assert res["n_edges"] == 4
    assert res["distance_m"] == pytest.approx(4 * 111.32, rel=0.01)


def test_route_same_node(grid_net):
    r = rl.Router(grid_net)
    res = r.route((0, 0), (0, 0))
    assert res["n_edges"] == 0 and res["distance_m"] == 0.0


def test_time_route_uses_regime_speeds(straight_net):
    df = pd.DataFrame(_hourly_obs(0, 8, 4.0) + _hourly_obs(0, 22, 16.0))
    rl.assign_segment_speeds(straight_net, df, peak_hours=[8],
                             offpeak_hours=[22])
    r = rl.Router(straight_net)
    tt_peak = r.route((0, 0), (0.01, 0), mode="time", period="peak")["travel_time_s"]
    tt_off = r.route((0, 0), (0.01, 0), mode="time", period="offpeak")["travel_time_s"]
    assert tt_peak == pytest.approx(4 * tt_off, rel=1e-6)


def test_invalid_period_raises(straight_net):
    with pytest.raises(ValueError, match="period"):
        rl.Router(straight_net).route((0, 0), (0.01, 0), mode="time",
                                      period="rush")


def test_default_speed_mps_must_be_positive(straight_net):
    """Regression: default_speed_mps=0 raised a raw ZeroDivisionError, and a
    negative value was silently accepted and produced negative travel times."""
    with pytest.raises(ValueError, match="default_speed_mps"):
        rl.Router(straight_net, default_speed_mps=0.0)
    with pytest.raises(ValueError, match="default_speed_mps"):
        rl.Router(straight_net, default_speed_mps=-5.0)


def test_default_speed_fallback_counted(straight_net):
    r = rl.Router(straight_net, default_speed_mps=10.0)
    res = r.route((0, 0), (0.01, 0), mode="time")
    assert res["n_edges_default"] == res["n_edges"] == 1
    assert res["travel_time_s"] == pytest.approx(res["distance_m"] / 10.0)


def test_parallel_edge_distance_pick(tmp_path):
    """The route must use (and report) the shorter of two parallel roads."""
    path = write_geojson(tmp_path / "par.json", [
        line_feature([[0, 0], [0.001, 0]], name="straight"),
        line_feature([[0, 0], [0.0005, 0.0006], [0.001, 0]], name="detour"),
    ])
    net = rl.Network.from_geojson(path)
    r = rl.Router(net)
    res = r.route((0, 0), (0.001, 0), mode="distance")
    assert res["n_edges"] == 1
    assert res["distance_m"] == pytest.approx(111.32, rel=0.01)
    assert net.edge_data(res["edge_ids"][0]).get("name") == "straight"


def test_oneway_blocks_illegal_direction(tmp_path):
    path = write_geojson(tmp_path / "ow.json", [
        line_feature([[0, 0], [0.001, 0]], oneway="yes"),
    ])
    net = rl.Network.from_geojson(path)
    r = rl.Router(net)
    assert r.route((0, 0), (0.001, 0), mode="distance")["n_edges"] == 1
    with pytest.raises(ValueError, match="one-way"):
        r.route((0.001, 0), (0, 0), mode="distance")


def test_disconnected_components_error(tmp_path):
    path = write_geojson(tmp_path / "disc.json", [
        line_feature([[0, 0], [0.001, 0]]),
        line_feature([[0.05, 0], [0.051, 0]]),
    ])
    net = rl.Network.from_geojson(path)
    with pytest.raises(ValueError, match="disconnected"):
        rl.Router(net).route((0, 0), (0.051, 0), mode="distance")


def test_cost_mode_receives_parallel_edge_dict(grid_net):
    seen = {}

    def cost(u, v, edges):
        seen["is_dict_of_dicts"] = all(isinstance(a, dict) for a in edges.values())
        return min(a.get("length_m", 1.0) for a in edges.values())

    res = rl.Router(grid_net).route((0, 0), (0.002, 0), mode="cost",
                                    cost_func=cost)
    assert res["n_edges"] == 2
    assert seen["is_dict_of_dicts"]


def test_cost_mode_reports_the_edge_it_actually_costed(tmp_path):
    """Regression: for parallel edges, mode='cost' reported edge_ids/distance
    picked by lowest travel_time_s, ignoring what cost_func actually costed
    the path over -- reported metrics didn't match the path Dijkstra chose."""
    path = write_geojson(tmp_path / "par.json", [
        line_feature([[0, 0], [0.001, 0]], name="straight"),
        line_feature([[0, 0], [0.0005, 0.0006], [0.001, 0]], name="detour"),
    ])
    net = rl.Network.from_geojson(path)
    # Make travel-time and length disagree on which parallel edge is
    # "cheapest": this is what exposes picking-by-time instead of by cost.
    for _u, _v, data in net.graph.edges(data=True):
        data["travel_time_s"] = 10.0 if data.get("name") == "detour" else 1000.0

    def cost_by_length(u, v, edges):
        return min(a["length_m"] for a in edges.values())

    res = rl.Router(net).route((0, 0), (0.001, 0), mode="cost",
                               cost_func=cost_by_length)
    assert net.edge_data(res["edge_ids"][0]).get("name") == "straight"
    assert res["distance_m"] == pytest.approx(111.32, rel=0.01)


def test_cost_mode_requires_callable(grid_net):
    with pytest.raises(ValueError, match="cost_func"):
        rl.Router(grid_net).route((0, 0), (0.002, 0), mode="cost")


def test_route_geometry(grid_net):
    r = rl.Router(grid_net)
    res = r.route((0, 0), (0.002, 0), mode="distance")
    coords = r.route_geometry_lonlat(res)
    assert coords[0] == pytest.approx((0.0, 0.0), abs=1e-9)
    assert coords[-1][0] == pytest.approx(0.002, abs=1e-8)


# ------------------------------------------ posted-limit fallback (maxspeed)
# An edge with no observations is estimated. Without a posted limit that
# estimate is one global constant for every road; with one it is per-edge.

def _road(tmp_path, **props):
    path = write_geojson(tmp_path / "road.json", [
        line_feature([[0, 0], [0.01, 0]], highway="primary", **props),
    ])
    return rl.Network.from_geojson(path)


def test_unobserved_edge_routes_at_posted_limit(tmp_path):
    net = _road(tmp_path, maxspeed="35 mph")
    res = rl.Router(net).route((0, 0), (0.01, 0), mode="time")
    expected = res["distance_m"] / (35 * 1609.344 / 3600)
    assert res["travel_time_s"] == pytest.approx(expected, rel=1e-9)


def test_unobserved_edge_without_limit_uses_global_default(tmp_path):
    net = _road(tmp_path)
    r = rl.Router(net, default_speed_mps=11.176)
    res = r.route((0, 0), (0.01, 0), mode="time")
    assert res["travel_time_s"] == pytest.approx(res["distance_m"] / 11.176,
                                                 rel=1e-9)


def test_measured_speed_beats_posted_limit(tmp_path):
    """Observed congestion must win: the limit is only a fallback."""
    net = _road(tmp_path, maxspeed="60 mph")
    rl.assign_speeds(net, pd.DataFrame(_hourly_obs(0, 8, 5.0)))
    res = rl.Router(net).route((0, 0), (0.01, 0), mode="time")
    assert res["travel_time_s"] == pytest.approx(res["distance_m"] / 5.0,
                                                 rel=1e-6)
    assert res["n_edges_default"] == 0


def test_posted_limit_edge_still_counts_as_estimated(tmp_path):
    """maxspeed refines the fallback speed; it is not measured data, so the
    honesty counter must keep reporting the edge as non-observed."""
    net = _road(tmp_path, maxspeed="35 mph")
    res = rl.Router(net).route((0, 0), (0.01, 0), mode="time")
    assert res["n_edges_default"] == res["n_edges"] == 1


@pytest.mark.parametrize("bad", [
    float("nan"),    # truthy! would poison the whole route total silently
    float("inf"),    # would make the edge free to cross
    0.0, -5.0, "fast", None,
])
def test_unusable_maxspeed_mps_falls_back_to_default(tmp_path, bad):
    """A maxspeed_mps that arrives from the caller's own source data (rather
    than from parse_maxspeed) is not trustworthy -- anything non-finite or
    non-positive must fall through to the global default, not propagate."""
    net = _road(tmp_path)
    for eid in net.edge_ids:
        net.edge_data(int(eid))["maxspeed_mps"] = bad
    res = rl.Router(net, default_speed_mps=11.176).route(
        (0, 0), (0.01, 0), mode="time")
    assert res["travel_time_s"] == pytest.approx(res["distance_m"] / 11.176,
                                                 rel=1e-9)


def test_numpy_float_maxspeed_mps_is_usable(tmp_path):
    """numpy scalars are not Python float subclasses; they must still count."""
    import numpy as np
    net = _road(tmp_path)
    for eid in net.edge_ids:
        net.edge_data(int(eid))["maxspeed_mps"] = np.float32(20.0)
    res = rl.Router(net).route((0, 0), (0.01, 0), mode="time")
    assert res["travel_time_s"] == pytest.approx(res["distance_m"] / 20.0,
                                                 rel=1e-5)


def test_posted_limit_fallback_can_be_disabled(tmp_path):
    net = _road(tmp_path, maxspeed="35 mph")
    r = rl.Router(net, default_speed_mps=11.176, use_maxspeed=False)
    res = r.route((0, 0), (0.01, 0), mode="time")
    assert res["travel_time_s"] == pytest.approx(res["distance_m"] / 11.176,
                                                 rel=1e-9)
