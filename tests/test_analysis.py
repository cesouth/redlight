import pandas as pd
import pytest

import roadtraffic as rt
from conftest import line_feature, write_geojson


# --------------------------------------------------------------------------- #
# edge_betweenness_centrality
# --------------------------------------------------------------------------- #
def test_missing_weight_attribute_raises(straight_net):
    """A fresh network with no speed pipeline run has no travel_time_s at all."""
    with pytest.raises(ValueError, match="travel_time_s"):
        rt.edge_betweenness_centrality(straight_net, weight="travel_time_s")


def test_partial_missing_weight_still_raises(straight_net):
    """Regression: only fully-missing used to be caught, not partially-missing."""
    eid0 = int(straight_net.edge_ids[0])
    df = pd.DataFrame([{"edge_id": eid0, "speed_mps": 10.0,
                        "time": pd.Timestamp("2026-01-01 08:00:00")}])
    rt.assign_speeds(straight_net, df)  # no default_speed_mps: only eid0 gets it
    with pytest.raises(ValueError, match="travel_time_s"):
        rt.edge_betweenness_centrality(straight_net, weight="travel_time_s")


def test_weight_length_m_works_without_any_pipeline(grid_net):
    bc = rt.edge_betweenness_centrality(grid_net, weight="length_m")
    assert set(bc.keys()) == {int(e) for e in grid_net.edge_ids}
    assert all(0.0 <= v <= 1.0 for v in bc.values())


def test_weight_none_is_unweighted(straight_net):
    bc = rt.edge_betweenness_centrality(straight_net, weight=None)
    assert set(bc.keys()) == {int(e) for e in straight_net.edge_ids}


def test_parallel_edges_favor_the_cheaper_one(tmp_path):
    """The direct road should carry more shortest-path traffic than the detour."""
    path = write_geojson(tmp_path / "par.json", [
        line_feature([[0, 0], [0.001, 0]], name="straight"),
        line_feature([[0, 0], [0.0005, 0.0006], [0.001, 0]], name="detour"),
    ])
    net = rt.Network.from_geojson(path)
    bc = rt.edge_betweenness_centrality(net, weight="length_m")
    straight_ids = [int(e) for e in net.edge_ids
                    if net.edge_data(int(e)).get("name") == "straight"]
    detour_ids = [int(e) for e in net.edge_ids
                  if net.edge_data(int(e)).get("name") == "detour"]
    assert max(bc[e] for e in straight_ids) > max(bc[e] for e in detour_ids)


def test_bridge_edge_has_highest_betweenness(tmp_path):
    """A single link road joining two clusters should be the network's chokepoint."""
    path = write_geojson(tmp_path / "bridge.json", [
        line_feature([[0, 0], [0.001, 0]]),
        line_feature([[0.001, 0], [0.001, 0.001]]),
        line_feature([[0.001, 0.001], [0.001, 0.002]]),  # the bridge
        line_feature([[0.001, 0.002], [0.002, 0.002]]),
        line_feature([[0.002, 0.002], [0.002, 0.003]]),
    ])
    net = rt.Network.from_geojson(path)
    bc = rt.edge_betweenness_centrality(net, weight="length_m")
    bridge_node_pair = {(0.001, 0.001), (0.001, 0.002)}
    bridge_eid = next(eid for eid in bc
                      if set(net.edge_endpoints(eid)) == bridge_node_pair)
    bridge_edges = net.road_edge_ids(bridge_eid)
    top_score = max(bc.values())
    assert all(bc[e] == pytest.approx(top_score) for e in bridge_edges)
    assert all(bc[e] <= top_score for e in bc)


def test_k_zero_raises_value_error_not_zero_division(straight_net):
    """Regression-guard: networkx itself raises a raw ZeroDivisionError for k=0."""
    with pytest.raises(ValueError, match="k"):
        rt.edge_betweenness_centrality(straight_net, weight="length_m", k=0)


def test_k_sampling_reproducible_with_seed(grid_net):
    a = rt.edge_betweenness_centrality(grid_net, weight="length_m", k=5, seed=1)
    b = rt.edge_betweenness_centrality(grid_net, weight="length_m", k=5, seed=1)
    assert a == b


def test_write_attr_writes_onto_graph_only_when_given(straight_net):
    bc = rt.edge_betweenness_centrality(straight_net, weight="length_m")
    for eid in bc:
        assert "bc_test" not in straight_net.edge_data(eid)

    bc2 = rt.edge_betweenness_centrality(straight_net, weight="length_m",
                                         write_attr="bc_test")
    for eid, score in bc2.items():
        assert straight_net.edge_data(eid)["bc_test"] == pytest.approx(score)


def test_write_attr_collision_raises(straight_net):
    with pytest.raises(ValueError, match="length_m"):
        rt.edge_betweenness_centrality(straight_net, weight="length_m",
                                       write_attr="length_m")
    with pytest.raises(ValueError, match="travel_time_s"):
        rt.edge_betweenness_centrality(straight_net, weight="length_m",
                                       write_attr="travel_time_s")


# --------------------------------------------------------------------------- #
# network_stats
# --------------------------------------------------------------------------- #
def test_grid_streets_per_node_and_intersections(grid_net):
    stats = rt.network_stats(grid_net)
    assert stats["n_nodes"] == 9
    assert stats["n_edges"] == 24
    assert stats["n_physical_roads"] == 12
    assert stats["n_intersections"] == 5
    assert stats["n_dead_ends"] == 0
    assert stats["streets_per_node_avg"] == pytest.approx(24 / 9)
    assert stats["streets_per_node_counts"] == {2: 4, 3: 4, 4: 1}


def test_intersection_definition_collinear_vs_branching(tmp_path):
    """Regression-guard for the intersection definition: physical-road-degree,
    not raw directed degree (which double-counts every two-way road)."""
    collinear = write_geojson(tmp_path / "collinear.json", [
        line_feature([[0, 0], [0.001, 0]]),
        line_feature([[0.001, 0], [0.002, 0]]),
    ])
    net = rt.Network.from_geojson(collinear)
    stats = rt.network_stats(net)
    assert stats["n_intersections"] == 0  # pure through-point, not a branch
    assert stats["n_dead_ends"] == 2

    branch = write_geojson(tmp_path / "branch.json", [
        line_feature([[0, 0], [0.001, 0]]),
        line_feature([[0.001, 0], [0.002, 0]]),
        line_feature([[0.001, 0], [0.001, 0.001]]),
    ])
    net2 = rt.Network.from_geojson(branch)
    stats2 = rt.network_stats(net2)
    assert stats2["n_intersections"] == 1
    assert stats2["n_dead_ends"] == 3


def test_circuity_straight_road_near_one(straight_net):
    stats = rt.network_stats(straight_net)
    assert stats["circuity_avg"] == pytest.approx(1.0, rel=0.01)


def test_circuity_bent_road_greater_than_one(tmp_path):
    path = write_geojson(tmp_path / "detour.json", [
        line_feature([[0, 0], [0.0005, 0.0006], [0.001, 0]], name="detour"),
    ])
    net = rt.Network.from_geojson(path)
    stats = rt.network_stats(net)
    assert stats["circuity_avg"] == pytest.approx(1.557, rel=0.01)
    assert stats["circuity_avg"] > 1.2


def test_area_metrics_none_without_area_km2(grid_net):
    stats = rt.network_stats(grid_net)
    assert stats["area_km2"] is None
    assert stats["intersection_density_km2"] is None
    assert stats["edge_density_km2"] is None


def test_area_metrics_computed_with_area_km2(grid_net):
    stats = rt.network_stats(grid_net, area_km2=2.0)
    assert stats["area_km2"] == 2.0
    assert stats["intersection_density_km2"] == pytest.approx(
        stats["n_intersections"] / 2.0)
    # physical-road length, not directed-edge length (would double-count)
    expected_physical_m = sum(
        grid_net.edge_length(int(e))
        for e in grid_net.edge_ids
        if int(e) == min(grid_net.road_edge_ids(int(e)))
    )
    assert stats["edge_density_km2"] == pytest.approx(expected_physical_m / 2.0)


# --------------------------------------------------------------------------- #
# connectivity_report
# --------------------------------------------------------------------------- #
def test_grid_is_strongly_connected(grid_net):
    cr = rt.connectivity_report(grid_net)
    assert cr["is_strongly_connected"] is True
    assert cr["n_strongly_connected_components"] == 1
    assert cr["largest_component_fraction_nodes"] == pytest.approx(1.0)
    assert cr["stranded_nodes"] == []
    assert cr["stranded_edge_ids"] == []


def test_one_way_trap_weak_not_strong(tmp_path):
    path = write_geojson(tmp_path / "trap.json", [
        line_feature([[0, 0], [0.001, 0]], oneway="yes"),
        line_feature([[0.001, 0], [0.002, 0]]),
    ])
    net = rt.Network.from_geojson(path)
    cr = rt.connectivity_report(net)
    assert cr["is_strongly_connected"] is False
    assert cr["is_weakly_connected"] is True
    assert cr["strongly_connected_component_sizes"] == [2, 1]


def test_genuinely_disconnected_weak_and_strong_false(tmp_path):
    path = write_geojson(tmp_path / "disc.json", [
        line_feature([[0, 0], [0.001, 0]]),                  # cluster: 2 nodes
        line_feature([[1, 1], [1.001, 1]]),
        line_feature([[1.001, 1], [1.002, 1]]),               # cluster: 3 nodes
    ])
    net = rt.Network.from_geojson(path)
    cr = rt.connectivity_report(net)
    assert cr["is_strongly_connected"] is False
    assert cr["is_weakly_connected"] is False
    assert cr["strongly_connected_component_sizes"] == [3, 2]
    assert cr["largest_component_fraction_nodes"] == pytest.approx(3 / 5)


def test_edge_partition_covers_every_edge_exactly_once(tmp_path):
    path = write_geojson(tmp_path / "disc2.json", [
        line_feature([[0, 0], [0.001, 0]], oneway="yes"),
        line_feature([[0.001, 0], [0.002, 0]]),
        line_feature([[1, 1], [1.001, 1]]),
    ])
    net = rt.Network.from_geojson(path)
    cr = rt.connectivity_report(net)
    all_ids = sorted(int(e) for e in net.edge_ids)
    partition = sorted(cr["largest_component_edge_ids"] + cr["stranded_edge_ids"])
    assert partition == all_ids
    assert len(set(partition)) == len(partition)  # no duplicates
