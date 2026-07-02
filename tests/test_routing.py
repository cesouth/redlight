import pandas as pd
import pytest

import roadtraffic as rt
from conftest import line_feature, write_geojson


def _hourly_obs(edge_id, hour, speed_mps, n=3):
    return [{"edge_id": edge_id, "speed_mps": speed_mps,
             "time": pd.Timestamp(f"2026-06-01 {hour:02d}:10:{k:02d}")}
            for k in range(n)]


def test_distance_route_on_grid(grid_net):
    r = rt.Router(grid_net)
    res = r.route((0, 0), (0.002, 0.002), mode="distance")
    assert res["n_edges"] == 4
    assert res["distance_m"] == pytest.approx(4 * 111.32, rel=0.01)


def test_route_same_node(grid_net):
    r = rt.Router(grid_net)
    res = r.route((0, 0), (0, 0))
    assert res["n_edges"] == 0 and res["distance_m"] == 0.0


def test_time_route_uses_regime_speeds(straight_net):
    df = pd.DataFrame(_hourly_obs(0, 8, 4.0) + _hourly_obs(0, 22, 16.0))
    rt.assign_segment_speeds(straight_net, df, peak_hours=[8],
                             offpeak_hours=[22])
    r = rt.Router(straight_net)
    tt_peak = r.route((0, 0), (0.01, 0), mode="time", period="peak")["travel_time_s"]
    tt_off = r.route((0, 0), (0.01, 0), mode="time", period="offpeak")["travel_time_s"]
    assert tt_peak == pytest.approx(4 * tt_off, rel=1e-6)


def test_invalid_period_raises(straight_net):
    with pytest.raises(ValueError, match="period"):
        rt.Router(straight_net).route((0, 0), (0.01, 0), mode="time",
                                      period="rush")


def test_default_speed_fallback_counted(straight_net):
    r = rt.Router(straight_net, default_speed_mps=10.0)
    res = r.route((0, 0), (0.01, 0), mode="time")
    assert res["n_edges_default"] == res["n_edges"] == 1
    assert res["travel_time_s"] == pytest.approx(res["distance_m"] / 10.0)


def test_parallel_edge_distance_pick(tmp_path):
    """The route must use (and report) the shorter of two parallel roads."""
    path = write_geojson(tmp_path / "par.json", [
        line_feature([[0, 0], [0.001, 0]], name="straight"),
        line_feature([[0, 0], [0.0005, 0.0006], [0.001, 0]], name="detour"),
    ])
    net = rt.Network.from_geojson(path)
    r = rt.Router(net)
    res = r.route((0, 0), (0.001, 0), mode="distance")
    assert res["n_edges"] == 1
    assert res["distance_m"] == pytest.approx(111.32, rel=0.01)
    assert net.edge_data(res["edge_ids"][0]).get("name") == "straight"


def test_oneway_blocks_illegal_direction(tmp_path):
    path = write_geojson(tmp_path / "ow.json", [
        line_feature([[0, 0], [0.001, 0]], oneway="yes"),
    ])
    net = rt.Network.from_geojson(path)
    r = rt.Router(net)
    assert r.route((0, 0), (0.001, 0), mode="distance")["n_edges"] == 1
    with pytest.raises(ValueError, match="one-way"):
        r.route((0.001, 0), (0, 0), mode="distance")


def test_disconnected_components_error(tmp_path):
    path = write_geojson(tmp_path / "disc.json", [
        line_feature([[0, 0], [0.001, 0]]),
        line_feature([[0.05, 0], [0.051, 0]]),
    ])
    net = rt.Network.from_geojson(path)
    with pytest.raises(ValueError, match="disconnected"):
        rt.Router(net).route((0, 0), (0.051, 0), mode="distance")


def test_cost_mode_receives_parallel_edge_dict(grid_net):
    seen = {}

    def cost(u, v, edges):
        seen["is_dict_of_dicts"] = all(isinstance(a, dict) for a in edges.values())
        return min(a.get("length_m", 1.0) for a in edges.values())

    res = rt.Router(grid_net).route((0, 0), (0.002, 0), mode="cost",
                                    cost_func=cost)
    assert res["n_edges"] == 2
    assert seen["is_dict_of_dicts"]


def test_cost_mode_requires_callable(grid_net):
    with pytest.raises(ValueError, match="cost_func"):
        rt.Router(grid_net).route((0, 0), (0.002, 0), mode="cost")


def test_route_geometry(grid_net):
    r = rt.Router(grid_net)
    res = r.route((0, 0), (0.002, 0), mode="distance")
    coords = r.route_geometry_lonlat(res)
    assert coords[0] == pytest.approx((0.0, 0.0), abs=1e-9)
    assert coords[-1][0] == pytest.approx(0.002, abs=1e-8)
