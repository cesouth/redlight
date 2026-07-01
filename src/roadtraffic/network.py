"""Road network container.

Loads a road network from GeoJSON (native, no extra deps), or Shapefile/GPKG
(optional, requires the ``shapefile`` extra -> fiona). Builds a NetworkX graph
whose nodes are endpoint coordinates and whose edges carry geometry, length,
and any source attributes (e.g. OSM ``highway``, ``maxspeed``, ``oneway``).

Coordinates are stored in WGS84 (EPSG:4326) and also projected to a local
metric CRS (auto UTM, or user-specified) for distance-correct snapping and
length computation. Snapping/length math must never be done in degrees.
"""
from __future__ import annotations

import json
import math
from typing import Iterable, Optional

import numpy as np
from pyproj import CRS, Transformer
from shapely.geometry import LineString, shape
from shapely.ops import transform as shapely_transform

try:  # networkx is a core dependency
    import networkx as nx
except ImportError as exc:  # pragma: no cover
    raise ImportError("roadtraffic requires networkx. pip install networkx") from exc


def _auto_utm_epsg(lon: float, lat: float) -> int:
    """Return the EPSG code of the UTM zone containing (lon, lat)."""
    zone = int(math.floor((lon + 180.0) / 6.0) % 60) + 1
    return (32600 if lat >= 0 else 32700) + zone


def _round_node(x: float, y: float, ndigits: int = 7):
    """Quantise a coordinate so shared endpoints hash to the same node.

    7 decimal degrees ~ 1.1 cm at the equator: enough to merge endpoints that
    are meant to be identical without collapsing genuinely distinct nodes.
    """
    return (round(x, ndigits), round(y, ndigits))


class Network:
    """A road network as a graph plus projected geometry.

    Attributes
    ----------
    graph : networkx.DiGraph
        Directed graph. Each edge has attrs: ``geometry`` (projected
        LineString, metres), ``length_m`` (float), ``edge_id`` (int), plus
        source attributes. Two-way roads get edges in both directions.
    crs_metric : pyproj.CRS
        Projected CRS used for all distance math.
    """

    def __init__(self, graph, crs_metric, edge_index, edge_geoms_proj,
                 edge_ids, transformer_fwd, transformer_inv):
        self.graph = graph
        self.crs_metric = crs_metric
        self._edge_index = edge_index          # edge_id -> (u, v)
        self._edge_geoms_proj = edge_geoms_proj  # edge_id -> projected LineString
        self._edge_ids = edge_ids              # ordered array of edge_ids
        self._transformer_fwd = transformer_fwd  # wgs84 -> metric
        self._transformer_inv = transformer_inv  # metric -> wgs84
        self._kdtree = None
        self._seg_table = None  # (edge_id, x0, y0, x1, y1) per segment for snap

    # ------------------------------------------------------------------ loaders
    @classmethod
    def from_geojson(cls, path: str, *, metric_epsg: Optional[int] = None,
                     directed: bool = True, oneway_attr: str = "oneway",
                     length_attr: Optional[str] = None) -> "Network":
        """Load a network from a GeoJSON LineString FeatureCollection."""
        with open(path, "r", encoding="utf-8") as fh:
            gj = json.load(fh)
        feats = gj.get("features", []) if isinstance(gj, dict) else []
        records = []
        for feat in feats:
            geom = (feat or {}).get("geometry") or {}
            gtype = geom.get("type")
            props = dict(feat.get("properties") or {})
            if gtype == "LineString":
                records.append((shape(geom), props))
            elif gtype == "MultiLineString":
                for part in shape(geom).geoms:
                    records.append((part, dict(props)))
        if not records:
            raise ValueError("GeoJSON contained no LineString features.")
        return cls._build(records, metric_epsg, directed, oneway_attr, length_attr)

    @classmethod
    def from_file(cls, path: str, *, metric_epsg: Optional[int] = None,
                  directed: bool = True, oneway_attr: str = "oneway",
                  length_attr: Optional[str] = None) -> "Network":
        """Load a network from Shapefile or GPKG (requires the 'shapefile' extra).

        GeoJSON files are dispatched to :meth:`from_geojson`.
        """
        low = path.lower()
        if low.endswith((".geojson", ".json")):
            return cls.from_geojson(
                path, metric_epsg=metric_epsg, directed=directed,
                oneway_attr=oneway_attr, length_attr=length_attr,
            )
        try:
            import fiona
        except ImportError as exc:
            raise ImportError(
                "Reading Shapefile/GPKG requires the optional 'shapefile' extra. "
                "Install with: pip install roadtraffic[shapefile]"
            ) from exc
        records = []
        with fiona.open(path) as src:
            src_crs = CRS.from_user_input(src.crs) if src.crs else CRS.from_epsg(4326)
            to_wgs = Transformer.from_crs(src_crs, CRS.from_epsg(4326), always_xy=True)
            for feat in src:
                geom = feat["geometry"]
                if geom is None:
                    continue
                g = shape(geom)
                props = dict(feat["properties"] or {})
                parts = g.geoms if g.geom_type == "MultiLineString" else [g]
                for part in parts:
                    if part.geom_type != "LineString":
                        continue
                    if src_crs.to_epsg() != 4326:
                        part = shapely_transform(
                            lambda x, y, z=None: to_wgs.transform(x, y), part
                        )
                    records.append((part, dict(props)))
        if not records:
            raise ValueError("File contained no LineString features.")
        return cls._build(records, metric_epsg, directed, oneway_attr, length_attr)

    # ------------------------------------------------------------------ builder
    @classmethod
    def _build(cls, records, metric_epsg, directed, oneway_attr, length_attr):
        # Pick a metric CRS from the centroid of the first geometry.
        first = records[0][0]
        c = first.coords[0]
        if metric_epsg is None:
            metric_epsg = _auto_utm_epsg(c[0], c[1])
        crs_metric = CRS.from_epsg(metric_epsg)
        fwd = Transformer.from_crs(CRS.from_epsg(4326), crs_metric, always_xy=True)
        inv = Transformer.from_crs(crs_metric, CRS.from_epsg(4326), always_xy=True)

        graph = nx.DiGraph()
        edge_index, edge_geoms_proj = {}, {}
        edge_id = 0
        for geom_wgs, props in records:
            xs, ys = zip(*[(pt[0], pt[1]) for pt in geom_wgs.coords])
            px, py = fwd.transform(np.asarray(xs), np.asarray(ys))
            proj_line = LineString(np.column_stack([px, py]))
            length_m = (float(props[length_attr]) if length_attr and length_attr in props
                        else proj_line.length)
            u = _round_node(xs[0], ys[0])
            v = _round_node(xs[-1], ys[-1])
            if u == v:
                continue  # skip degenerate zero-length edges

            oneway = str(props.get(oneway_attr, "")).strip().lower() in {
                "yes", "true", "1", "-1", "t"
            }
            base_attrs = dict(props)

            def _add(a, b, line, eid):
                graph.add_node(a, lon=a[0], lat=a[1])
                graph.add_node(b, lon=b[0], lat=b[1])
                graph.add_edge(a, b, edge_id=eid, length_m=length_m,
                               geometry=line, **base_attrs)
                edge_index[eid] = (a, b)
                edge_geoms_proj[eid] = line

            _add(u, v, proj_line, edge_id)
            edge_id += 1
            if directed and not oneway:
                rev = LineString(list(proj_line.coords)[::-1])
                _add(v, u, rev, edge_id)
                edge_id += 1
            elif not directed:
                rev = LineString(list(proj_line.coords)[::-1])
                _add(v, u, rev, edge_id)
                edge_id += 1

        edge_ids = np.array(sorted(edge_index.keys()), dtype=np.int64)
        net = cls(graph, crs_metric, edge_index, edge_geoms_proj, edge_ids, fwd, inv)
        net._build_segment_table()
        return net

    # ------------------------------------------------------ spatial index/snap
    # Long segments are split so a query point near a segment end is never
    # crowded out of the midpoint KDTree shortlist by nearer-midpoint segments.
    # The exact foot-of-perpendicular distance is still computed against the
    # original segment endpoints, so the matched edge is identical -- this only
    # makes candidate retrieval reliable. 25 m is well below typical snap
    # tolerances yet keeps the table small.
    _SNAP_DENSIFY_M = 25.0

    def _build_segment_table(self):
        """Flatten every edge into its constituent 2-point segments for snapping.

        A KDTree over segment midpoints gives fast candidate retrieval; exact
        point-to-segment distance is then computed for the shortlist. Segments
        longer than ``_SNAP_DENSIFY_M`` are subdivided so every midpoint stays
        close to the geometry it represents.
        """
        d = self._SNAP_DENSIFY_M
        rows = []
        for eid in self._edge_ids:
            coords = np.asarray(self._edge_geoms_proj[eid].coords)
            for i in range(len(coords) - 1):
                x0, y0 = coords[i]
                x1, y1 = coords[i + 1]
                seg_len = math.hypot(x1 - x0, y1 - y0)
                n_sub = max(1, int(math.ceil(seg_len / d))) if d and d > 0 else 1
                if n_sub == 1:
                    rows.append((eid, x0, y0, x1, y1))
                else:
                    ts = np.linspace(0.0, 1.0, n_sub + 1)
                    xs = x0 + ts * (x1 - x0)
                    ys = y0 + ts * (y1 - y0)
                    for j in range(n_sub):
                        rows.append((eid, xs[j], ys[j], xs[j + 1], ys[j + 1]))
        self._seg_table = np.array(rows, dtype=float)  # cols: eid,x0,y0,x1,y1
        mids = np.column_stack([
            (self._seg_table[:, 1] + self._seg_table[:, 3]) / 2.0,
            (self._seg_table[:, 2] + self._seg_table[:, 4]) / 2.0,
        ])
        from scipy.spatial import cKDTree
        self._kdtree = cKDTree(mids)

    def project_points(self, lon, lat):
        """Project WGS84 lon/lat arrays to the metric CRS (metres)."""
        return self._transformer_fwd.transform(np.asarray(lon), np.asarray(lat))

    def candidate_edges(self, px, py, *, k: int = 10, max_dist: float = 50.0):
        """For metric point (px, py), return candidate (edge_id, perp_dist, t).

        ``t`` is the fractional position (0..1) of the foot of the perpendicular
        along the matched segment. ``max_dist`` is the snap tolerance in metres.
        Results are unique per edge_id, keeping the closest segment.
        """
        k = min(k, len(self._seg_table))
        dists, idxs = self._kdtree.query([px, py], k=k)
        idxs = np.atleast_1d(idxs)
        seg = self._seg_table[idxs]
        eid = seg[:, 0]
        x0, y0, x1, y1 = seg[:, 1], seg[:, 2], seg[:, 3], seg[:, 4]
        dx, dy = x1 - x0, y1 - y0
        seg_len2 = dx * dx + dy * dy
        seg_len2[seg_len2 == 0] = 1e-12
        t = ((px - x0) * dx + (py - y0) * dy) / seg_len2
        t = np.clip(t, 0.0, 1.0)
        fx, fy = x0 + t * dx, y0 + t * dy
        perp = np.hypot(px - fx, py - fy)
        order = np.argsort(perp)
        best = {}
        for j in order:
            e = int(eid[j])
            if perp[j] > max_dist:
                continue
            if e not in best:
                best[e] = (perp[j], float(t[j]))
        return [(e, d, tt) for e, (d, tt) in best.items()]

    # ------------------------------------------------------------- convenience
    def edge_endpoints(self, edge_id: int):
        return self._edge_index[edge_id]

    def number_of_edges(self) -> int:
        return self.graph.number_of_edges()

    def number_of_nodes(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def edge_ids(self):
        return self._edge_ids
