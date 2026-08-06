import networkx as nx
import pytest

import roadtraffic as rt
from conftest import line_feature, write_geojson, write_ogr


def test_graph_is_multidigraph(straight_net):
    assert isinstance(straight_net.graph, nx.MultiDiGraph)
    assert straight_net.number_of_edges() == 2  # two-way => both directions


def test_parallel_roads_coexist(tmp_path):
    """Regression: a second road between the same endpoints used to overwrite
    the first one's graph attributes while side tables kept both ids."""
    path = write_geojson(tmp_path / "par.json", [
        line_feature([[0, 0], [0.001, 0]], name="straight"),
        line_feature([[0, 0], [0.0005, 0.0004], [0.001, 0]], name="detour"),
    ])
    net = rt.Network.from_geojson(path)
    assert net.number_of_edges() == 4  # 2 roads x 2 directions
    u, v = (0.0, 0.0), (0.001, 0.0)
    assert len(net.edges_between(u, v)) == 4
    # every edge id resolves to ITS OWN length: the straight pair ~111 m,
    # the detour pair noticeably longer
    lengths = sorted(net.edge_length(int(e)) for e in net.edge_ids)
    assert lengths[0] == pytest.approx(lengths[1])
    assert lengths[2] == pytest.approx(lengths[3])
    assert lengths[2] > lengths[0] * 1.2
    # graph attrs agree with the side tables for every id
    for eid in net.edge_ids:
        assert net.edge_data(int(eid))["edge_id"] == int(eid)
        assert net.edge_data(int(eid))["length_m"] == pytest.approx(
            net.edge_length(int(eid)))


def test_oneway_forward(tmp_path):
    path = write_geojson(tmp_path / "ow.json", [
        line_feature([[0, 0], [0.001, 0]], oneway="yes"),
    ])
    net = rt.Network.from_geojson(path)
    assert net.graph.has_edge((0.0, 0.0), (0.001, 0.0))
    assert not net.graph.has_edge((0.001, 0.0), (0.0, 0.0))


def test_oneway_minus_one_is_reverse(tmp_path):
    """Regression: oneway=-1 (OSM: against digitized direction) used to
    create the edge in exactly the wrong direction."""
    path = write_geojson(tmp_path / "owr.json", [
        line_feature([[0, 0], [0.001, 0]], oneway="-1"),
    ])
    net = rt.Network.from_geojson(path)
    assert not net.graph.has_edge((0.0, 0.0), (0.001, 0.0))
    assert net.graph.has_edge((0.001, 0.0), (0.0, 0.0))


def test_closed_loop_warns(tmp_path):
    path = write_geojson(tmp_path / "loop.json", [
        line_feature([[0, 0], [0.001, 0]]),
        line_feature([[0.002, 0], [0.0025, 0.0005], [0.002, 0]], name="loop"),
    ])
    with pytest.warns(UserWarning, match="closed-loop"):
        net = rt.Network.from_geojson(path)
    assert net.number_of_edges() == 2  # only the open road


def test_reserved_property_keys_preserved(tmp_path):
    """Regression: properties named length_m/edge_id crashed add_edge."""
    path = write_geojson(tmp_path / "res.json", [
        line_feature([[0, 0], [0.001, 0]], length_m=999.0, edge_id=7),
    ])
    with pytest.warns(UserWarning, match="_src"):
        net = rt.Network.from_geojson(path)
    d = net.edge_data(int(net.edge_ids[0]))
    assert d["length_m_src"] == 999.0
    assert d["edge_id_src"] == 7
    assert d["length_m"] == pytest.approx(111.3, rel=0.01)  # real geometry length


def test_all_degenerate_raises_clear_error(tmp_path):
    path = write_geojson(tmp_path / "deg.json", [
        line_feature([[0, 0], [0.0005, 0.0005], [0, 0]]),
    ])
    with pytest.raises(ValueError, match="no usable edges"), \
            pytest.warns(UserWarning, match="closed-loop"):
        rt.Network.from_geojson(path)


def test_road_edge_ids_pairs_two_way(straight_net):
    eids = [int(e) for e in straight_net.edge_ids]
    assert straight_net.road_edge_ids(eids[0]) == eids
    assert straight_net.road_edge_ids(eids[1]) == eids


def test_road_edge_ids_oneway_is_single(tmp_path):
    path = write_geojson(tmp_path / "ow2.json", [
        line_feature([[0, 0], [0.001, 0]], oneway="yes"),
    ])
    net = rt.Network.from_geojson(path)
    eid = int(net.edge_ids[0])
    assert net.road_edge_ids(eid) == [eid]


def test_edge_coords_lonlat_round_trip(straight_net):
    coords = straight_net.edge_coords_lonlat(int(straight_net.edge_ids[0]))
    assert coords[0][0] == pytest.approx(0.0, abs=1e-9)
    assert coords[-1][0] == pytest.approx(0.01, abs=1e-8)
    assert all(abs(lat) < 1e-9 for _lon, lat in coords)


def test_candidate_edges_tolerance(straight_net):
    px, py = straight_net.project_points([0.005], [0.0002])  # ~22 m north
    cands = straight_net.candidate_edges(px[0], py[0], max_dist=50.0)
    assert cands and cands[0][1] == pytest.approx(22.1, rel=0.05)
    assert straight_net.candidate_edges(px[0], py[0], max_dist=10.0) == []


def test_length_attr_override(tmp_path):
    path = write_geojson(tmp_path / "la.json", [
        line_feature([[0, 0], [0.001, 0]], seg_len=500.0),
    ])
    net = rt.Network.from_geojson(path, length_attr="seg_len")
    assert net.edge_length(int(net.edge_ids[0])) == 500.0


# ------------------------------------------------------------- from_file (pyogrio)
def test_from_file_gpkg_basic(tmp_path):
    """A GeoPackage loads to the same graph structure as equivalent GeoJSON."""
    pytest.importorskip("pyogrio")
    path = write_ogr(tmp_path / "basic.gpkg", [
        ([[0, 0], [0.01, 0]], {"highway": "residential"}),
    ])
    net = rt.Network.from_file(path)
    gj_path = write_geojson(tmp_path / "basic.geojson", [
        line_feature([[0, 0], [0.01, 0]], highway="residential"),
    ])
    net_gj = rt.Network.from_geojson(gj_path)
    assert net.number_of_edges() == net_gj.number_of_edges() == 2
    assert set(net.graph.nodes()) == set(net_gj.graph.nodes())


def test_from_file_shp_basic(tmp_path):
    """A Shapefile loads correctly too (not just GeoPackage)."""
    pytest.importorskip("pyogrio")
    path = write_ogr(tmp_path / "basic.shp", [
        ([[0, 0], [0.01, 0]], {"highway": "residential"}),
    ])
    net = rt.Network.from_file(path)
    assert net.number_of_edges() == 2  # two-way by default
    assert net.number_of_nodes() == 2


def test_from_file_reprojects_utm_without_pyproj(tmp_path):
    """A UTM GeoPackage reprojects natively. Pinned eastings from PROJ 9.5.1
    for 15.0E/50.0N and 15.01E/50.0N in EPSG:32633, so this test does not
    import pyproj at all."""
    pytest.importorskip("pyogrio")
    path = write_ogr(tmp_path / "utm.gpkg", [
        ([[500000.000000, 5538630.702867],
          [500716.670753, 5538630.750777]], {"highway": "residential"}),
    ], crs="EPSG:32633")
    net = rt.Network.from_file(path)
    lons = sorted(n[0] for n in net.graph.nodes())
    assert lons[0] == pytest.approx(15.0, abs=1e-6)
    assert lons[1] == pytest.approx(15.01, abs=1e-6)


def test_from_file_reprojects_web_mercator(tmp_path):
    """EPSG:3857 is the other CRS handled natively."""
    pytest.importorskip("pyogrio")
    path = write_ogr(tmp_path / "wm.gpkg", [
        ([[1669792.3618991035, 6446275.841017159],
          [1670905.5568070365, 6446275.841017159]], {"highway": "residential"}),
    ], crs="EPSG:3857")
    net = rt.Network.from_file(path)
    lons = sorted(n[0] for n in net.graph.nodes())
    assert lons[0] == pytest.approx(15.0, abs=1e-6)


def test_from_file_wgs84_needs_no_transform(tmp_path):
    """The common case must not touch the projection code at all."""
    pytest.importorskip("pyogrio")
    from roadtraffic import network as net_mod
    assert net_mod._source_to_wgs84("EPSG:4326") is None
    assert net_mod._source_to_wgs84(None) is None


def test_from_file_exotic_crs_errors_clearly_without_pyproj(tmp_path, monkeypatch):
    """British National Grid has no closed form here. Without pyproj the
    failure must name the extra, not surface as ModuleNotFoundError."""
    pytest.importorskip("pyogrio")
    import builtins

    path = write_ogr(tmp_path / "bng.gpkg", [
        ([[529000.0, 181000.0], [529100.0, 181100.0]], {"highway": "residential"}),
    ], crs="EPSG:27700")
    real_import = builtins.__import__

    def no_pyproj(name, *args, **kwargs):
        if name == "pyproj":
            raise ImportError("no pyproj")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_pyproj)
    with pytest.raises(ImportError, match=r"roadtraffic\[crs\]"):
        rt.Network.from_file(path)


def test_from_file_layer_param(tmp_path):
    """layer= selects a specific layer of a multi-layer GeoPackage."""
    pytest.importorskip("pyogrio")
    path = tmp_path / "multi.gpkg"
    write_ogr(path, [([[0, 0], [0.01, 0]], {"name": "a"})], layer="layer_a")
    write_ogr(path, [([[1, 1], [1.01, 1]], {"name": "b"})], layer="layer_b")
    net_b = rt.Network.from_file(str(path), layer="layer_b")
    names = {d.get("name") for _u, _v, d in net_b.graph.edges(data=True)}
    assert names == {"b"}


def test_from_file_missing_pyogrio_message(tmp_path, monkeypatch):
    """Regression-guard: a clear, actionable error, not a bare ImportError
    from deep inside pyogrio, when the 'shapefile' extra isn't installed."""
    import sys
    monkeypatch.setitem(sys.modules, "pyogrio", None)
    with pytest.raises(ImportError, match="shapefile"):
        rt.Network.from_file(str(tmp_path / "whatever.gpkg"))


# --------------------------------------------------------------- maxspeed
def _net_with(tmp_path, **props):
    path = write_geojson(tmp_path / "ms.json", [
        line_feature([[0, 0], [0.01, 0]], highway="residential", **props),
    ])
    return rt.Network.from_geojson(path)


def test_maxspeed_tag_becomes_numeric_edge_attr(tmp_path):
    net = _net_with(tmp_path, maxspeed="35 mph")
    for _u, _v, d in net.graph.edges(data=True):
        assert d["maxspeed_mps"] == pytest.approx(35 * 1609.344 / 3600)
        assert d["maxspeed"] == "35 mph"      # raw tag preserved alongside


def test_edge_without_maxspeed_has_no_maxspeed_mps(tmp_path):
    net = _net_with(tmp_path)
    for _u, _v, d in net.graph.edges(data=True):
        assert "maxspeed_mps" not in d


def test_unparseable_maxspeed_sets_no_numeric_attr(tmp_path):
    net = _net_with(tmp_path, maxspeed="none")
    for _u, _v, d in net.graph.edges(data=True):
        assert "maxspeed_mps" not in d
        assert d["maxspeed"] == "none"        # raw tag still preserved


def test_metric_crs_is_native_utm_without_pyproj(straight_net):
    """The default path must not construct a pyproj object at all."""
    from roadtraffic import _proj
    assert isinstance(straight_net.crs_metric, _proj.UtmCrs)
    assert straight_net.crs_metric.to_epsg() == 32631


def test_non_utm_metric_epsg_errors_clearly_without_pyproj(tmp_path, monkeypatch):
    """A user-supplied projected CRS outside the native set still works when
    pyproj is installed, but must fail with an actionable message when it is
    not -- never with a bare ModuleNotFoundError."""
    import builtins

    from roadtraffic import network as net_mod

    path = write_geojson(tmp_path / "n.json", [
        line_feature([[15.0, 50.0], [15.01, 50.0]], highway="residential"),
    ])
    real_import = builtins.__import__

    def no_pyproj(name, *args, **kwargs):
        if name == "pyproj":
            raise ImportError("no pyproj")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_pyproj)
    with pytest.raises(ImportError, match=r"roadtraffic\[crs\]"):
        net_mod.Network.from_geojson(path, metric_epsg=27700)
