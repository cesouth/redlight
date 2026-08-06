"""Equivalence tests for the vectorised/batched matching paths.

The Tier-1 performance work must be *exact*: batch snapping, the scipy
Dijkstra cache and process parallelism have to produce the same answers as
the straightforward per-point / networkx / serial paths they replace.
"""
import networkx as nx
import numpy as np
import pandas as pd
import pytest

import redlight as rl
from conftest import drive_along_road
from redlight.matching import _CSRDistCache


def _random_metric_points(net, n=300, spread_deg=0.0035, seed=7):
    rng = np.random.default_rng(seed)
    lon = rng.uniform(-0.0005, spread_deg, n)
    lat = rng.uniform(-0.0005, spread_deg, n)
    return net.project_points(lon, lat)


def test_nearest_edges_matches_per_point_reference(grid_net):
    px, py = _random_metric_points(grid_net)
    eids, snap = grid_net.nearest_edges(px, py, k=10, max_dist=50.0)
    for i in range(len(px)):
        cands = grid_net.candidate_edges(px[i], py[i], k=10, max_dist=50.0)
        if not cands:
            assert eids[i] == -1 and np.isnan(snap[i])
        else:
            ref_eid, ref_dist, _ = min(cands, key=lambda c: c[1])
            assert snap[i] == pytest.approx(ref_dist, abs=1e-9)
            # ties between the two directions of one road are legitimate;
            # accept any edge at the same minimal distance
            same_dist = [e for e, d, _t in cands
                         if d == pytest.approx(ref_dist, abs=1e-9)]
            assert eids[i] in same_dist


def test_nearest_edges_chunking_is_invisible(grid_net):
    px, py = _random_metric_points(grid_net, n=100)
    a = grid_net.nearest_edges(px, py, k=8, max_dist=40.0)
    b = grid_net.nearest_edges(px, py, k=8, max_dist=40.0, chunk_size=7)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_allclose(a[1], b[1], equal_nan=True)


def test_candidate_edges_batch_matches_scalar(grid_net):
    # k large enough that the KDTree shortlist covers everything within
    # max_dist: then both paths must return the SAME complete candidate set.
    # (At small k, exact-tie segment pairs straddling the k-th neighbour can
    # legitimately differ between equally valid shortlists.)
    px, py = _random_metric_points(grid_net, n=200)
    k = 40
    batched = grid_net.candidate_edges_batch(px, py, k=k, max_dist=50.0)
    for i in range(len(px)):
        scalar = grid_net.candidate_edges(px[i], py[i], k=k, max_dist=50.0)
        got = {e: (d, t) for e, d, t in batched[i]}
        ref = {e: (d, t) for e, d, t in scalar}
        assert got.keys() == ref.keys()
        for e in ref:
            assert got[e][0] == pytest.approx(ref[e][0], abs=1e-9)
            assert got[e][1] == pytest.approx(ref[e][1], abs=1e-9)
        # both lists are ordered by increasing snap distance
        dists = [d for _e, d, _t in batched[i]]
        assert dists == sorted(dists)


def test_csgraph_min_over_parallel_edges(tmp_path):
    from conftest import line_feature, write_geojson
    path = write_geojson(tmp_path / "par.json", [
        line_feature([[0, 0], [0.001, 0]], name="straight"),
        line_feature([[0, 0], [0.0005, 0.0006], [0.001, 0]], name="detour"),
    ])
    net = rl.Network.from_geojson(path)
    csr, node_int, _ = net.csgraph()
    u, v = node_int[(0.0, 0.0)], node_int[(0.001, 0.0)]
    # CSR keeps the min length over parallel edges, matching Dijkstra's choice
    assert csr[u, v] == pytest.approx(111.32, rel=0.01)


def test_csr_dist_cache_matches_networkx(grid_net):
    csr, node_int, _ = grid_net.csgraph()
    cache = _CSRDistCache(csr, node_int)
    src = (0.0, 0.0)
    cutoff = 500.0
    ref = nx.single_source_dijkstra_path_length(
        grid_net.graph, src, cutoff=cutoff, weight="length_m")
    for dst in grid_net.graph.nodes():
        got = cache.lookup(src, dst, cutoff)
        if dst in ref:
            assert got == pytest.approx(ref[dst], rel=1e-9)
        else:
            assert got is None


def test_csr_dist_cache_reapplies_cutoff():
    """The S2 regression, for the new cache: a warm larger-cutoff entry must
    not leak distances beyond a smaller requested cutoff."""
    from scipy.sparse import csr_matrix
    csr = csr_matrix(np.array([[0.0, 400.0, 0.0],
                               [400.0, 0.0, 400.0],
                               [0.0, 400.0, 0.0]]))
    node_int = {"a": 0, "b": 1, "c": 2}
    cache = _CSRDistCache(csr, node_int)
    assert cache.lookup("a", "c", 1200.0) == pytest.approx(800.0)
    assert cache.lookup("a", "c", 300.0) is None  # same warm cache
    assert cache.lookup("a", "zzz", 300.0) is None  # unknown node


def test_csr_dist_cache_lru_bound(grid_net):
    csr, node_int, _ = grid_net.csgraph()
    cache = _CSRDistCache(csr, node_int, maxsize=3)
    nodes = list(grid_net.graph.nodes())[:6]
    for src in nodes:
        cache.lookup(src, nodes[0], 400.0)
    assert len(cache._cache) <= 3


def _hmm_scenario_points(make_points_csv):
    rows = []
    for t, start in (("a", 0.0002), ("b", 0.0004), ("c", 0.0007)):
        rows += drive_along_road(8, traj=t, start_lon=start, dlon=0.00022,
                                 lat=3e-5)
    # one trajectory with an off-network gap in the middle
    rows += drive_along_road(3, traj="d", start_lon=0.0002)
    rows.append({"id": "d", "lon": 0.001, "lat": 0.003,
                 "time": "2026-06-01T08:00:40"})
    rows += drive_along_road(3, traj="d", start_lon=0.0012,
                             t0="2026-06-01 08:00:50")
    return rl.load_points(make_points_csv(rows))


def test_hmm_parallel_equals_serial(grid_net, make_points_csv):
    pts = _hmm_scenario_points(make_points_csv)
    serial = rl.HMMMatcher(grid_net, max_dist=60, n_jobs=1).match(pts)
    parallel = rl.HMMMatcher(grid_net, max_dist=60, n_jobs=2).match(pts)
    a = serial.sort_values("point_id").reset_index(drop=True)
    b = parallel.sort_values("point_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_hmm_n_jobs_minus_one_runs(grid_net, make_points_csv):
    pts = _hmm_scenario_points(make_points_csv)
    out = rl.HMMMatcher(grid_net, max_dist=60, n_jobs=-1).match(pts)
    assert len(out) == len(pts)


def test_hmm_matcher_pickle_drops_cache(grid_net, make_points_csv):
    import pickle
    pts = _hmm_scenario_points(make_points_csv)
    m = rl.HMMMatcher(grid_net, max_dist=60)
    m.match(pts)  # warm the cache
    clone = pickle.loads(pickle.dumps(m))
    assert clone._dist_cache is None
    out = clone.match(pts)  # and the clone still works
    assert len(out) == len(pts)
