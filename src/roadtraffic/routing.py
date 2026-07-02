"""Routing over the network graph.

Supports shortest path by:
  - ``"distance"``  -> minimises summed ``length_m``.
  - ``"time"``      -> minimises summed ``travel_time_s`` (requires
                       assign_speeds to have run; falls back to length/
                       default speed where missing).
  - ``"cost"``      -> minimises a user-supplied per-edge weight function.

All use Dijkstra (networkx), which is correct for non-negative weights -- always
true for distance and time. Nodes are coordinate tuples; helper methods locate
the nearest graph node to an arbitrary lon/lat.

The network graph is a :class:`networkx.MultiDiGraph`, so parallel roads
between the same node pair are distinct edges; routing picks the cheapest
parallel edge under the active weight. A custom ``cost_func`` therefore
receives ``(u, v, edges)`` where ``edges`` is a dict keyed by edge key
(= ``edge_id``) mapping to each parallel edge's attribute dict -- return the
cost of the cheapest acceptable parallel edge.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

try:
    import networkx as nx
except ImportError as exc:  # pragma: no cover
    raise ImportError("roadtraffic requires networkx.") from exc


class Router:
    """Shortest-path routing on a :class:`~roadtraffic.network.Network`."""

    def __init__(self, network, *, default_speed_mps: float = 11.176):
        # 11.176 m/s ~ 25 mph: a sane urban fallback when an edge lacks a
        # travel time and the user requests time routing.
        self.network = network
        self.default_speed_mps = default_speed_mps
        self._node_array = None
        self._node_list = None

    def _ensure_node_index(self):
        if self._node_array is None:
            nodes = list(self.network.graph.nodes())
            self._node_list = nodes
            lon = np.array([n[0] for n in nodes])
            lat = np.array([n[1] for n in nodes])
            px, py = self.network.project_points(lon, lat)
            self._node_array = np.column_stack([px, py])
            from scipy.spatial import cKDTree
            self._node_tree = cKDTree(self._node_array)

    def nearest_node(self, lon: float, lat: float):
        """Return the graph node (coordinate tuple) nearest to (lon, lat)."""
        self._ensure_node_index()
        px, py = self.network.project_points([lon], [lat])
        _, idx = self._node_tree.query([px[0], py[0]])
        return self._node_list[int(idx)]

    _PERIODS = ("overall", "peak", "offpeak")

    def _edge_travel_time(self, d, period: str):
        """Period-aware travel time for one edge's attrs; returns (seconds, used_default).

        Prefers ``travel_time_s_<period>``, falls back to the overall
        ``travel_time_s``, then to ``length_m / default_speed_mps``.
        """
        attr = ("travel_time_s" if period == "overall"
                else f"travel_time_s_{period}")
        tt = d.get(attr)
        if tt is None:                      # this regime had no data on this edge
            tt = d.get("travel_time_s")     # fall back to overall
        if tt is None:                      # no speed data at all
            return d.get("length_m", 0.0) / self.default_speed_mps, True
        return tt, False

    # ------------------------------------------------- multigraph edge selection
    # networkx passes a *keyed* dict (edge key -> attrs) to weight callables on
    # multigraphs; the cheapest parallel edge wins.
    def _min_length_attrs(self, edges: dict) -> dict:
        return min(edges.values(), key=lambda a: a.get("length_m", 1.0))

    def _min_time_attrs(self, edges: dict, period: str) -> dict:
        return min(edges.values(),
                   key=lambda a: self._edge_travel_time(a, period)[0])

    def _weight_func(self, mode: str, cost_func: Callable | None, period: str):
        if mode == "distance":
            return lambda u, v, d: self._min_length_attrs(d).get("length_m", 1.0)
        if mode == "time":
            if period not in self._PERIODS:
                raise ValueError(
                    f"period must be one of {self._PERIODS}, got {period!r}."
                )
            return lambda u, v, d: self._edge_travel_time(
                self._min_time_attrs(d, period), period)[0]
        if mode == "cost":
            if cost_func is None:
                raise ValueError(
                    "mode='cost' requires cost_func=callable(u, v, edges) where "
                    "edges is a dict of edge_id -> attribute dict."
                )
            return cost_func
        raise ValueError("mode must be 'distance', 'time' or 'cost'.")

    def route(
        self,
        origin,
        destination,
        *,
        mode: str = "time",
        cost_func: Callable | None = None,
        snap: bool = True,
        period: str = "overall",
    ) -> dict:
        """Compute a shortest path.

        Parameters
        ----------
        origin, destination : (lon, lat) tuples, or graph nodes.
        mode : {"distance", "time", "cost"}
        cost_func : callable(u, v, edges) -> float, required if mode='cost'.
            ``edges`` is a dict keyed by edge key (= edge_id) of parallel-edge
            attribute dicts (the graph is a MultiDiGraph).
        snap : bool
            If True, treat origin/destination as lon/lat and snap to the nearest
            node. If False, they must already be graph nodes.
        period : {"overall", "peak", "offpeak"}
            For ``mode="time"``, which per-edge regime to route on (written by
            :func:`~roadtraffic.aggregate.assign_segment_speeds`). Edges lacking
            that regime fall back to the overall speed, then the default speed.

        Returns
        -------
        dict
            ``path`` (list of nodes), ``edge_ids`` (list), ``distance_m``,
            ``travel_time_s``, ``n_edges``, and ``n_edges_default`` (how many
            edges used the fallback default speed -- a coverage signal).

        Raises
        ------
        ValueError
            With an actionable message if the network is empty, an endpoint is not
            a graph node (``snap=False``), or no path exists between the endpoints.
        """
        graph = self.network.graph
        if graph.number_of_edges() == 0:
            raise ValueError(
                "Cannot route: the network has no edges. Did the network load "
                "correctly (non-empty GeoJSON/Shapefile of LineStrings)?"
            )

        if snap:
            src = self.nearest_node(origin[0], origin[1])
            dst = self.nearest_node(destination[0], destination[1])
        else:
            src, dst = origin, destination
            if src not in graph:
                raise ValueError(
                    f"origin {src!r} is not a graph node. Pass (lon, lat) with "
                    "snap=True, or an exact node coordinate tuple."
                )
            if dst not in graph:
                raise ValueError(
                    f"destination {dst!r} is not a graph node. Pass (lon, lat) "
                    "with snap=True, or an exact node coordinate tuple."
                )

        weight = self._weight_func(mode, cost_func, period)

        if src == dst:
            return {"path": [src], "edge_ids": [], "distance_m": 0.0,
                    "travel_time_s": 0.0, "n_edges": 0, "n_edges_default": 0}

        try:
            path = nx.shortest_path(graph, src, dst, weight=weight)
        except nx.NetworkXNoPath:
            reachable_undirected = nx.has_path(graph.to_undirected(as_view=True),
                                               src, dst)
            hint = ("one-way directions block the route (the nodes connect only "
                    "against the legal direction of travel)"
                    if reachable_undirected else
                    "the endpoints are in different disconnected components of "
                    "the network (e.g. a clipped/incomplete extract)")
            raise ValueError(
                f"No {mode} route exists between the origin and destination: "
                f"{hint}. Try mode='distance' to confirm, supply a fuller/"
                "better-connected network, or relax the snap by moving the "
                "endpoints onto the main network."
            ) from None

        time_period = period if mode == "time" else "overall"
        edge_ids, dist_m, time_s, n_default = [], 0.0, 0.0, 0
        for u, v in zip(path[:-1], path[1:]):
            edges = graph.get_edge_data(u, v)
            # pick the same parallel edge the active weight would have chosen
            if mode == "distance":
                d = self._min_length_attrs(edges)
            else:
                d = self._min_time_attrs(edges, time_period)
            edge_ids.append(d.get("edge_id"))
            dist_m += d.get("length_m", 0.0)
            tt, used_default = self._edge_travel_time(d, time_period)
            time_s += tt
            if used_default:
                n_default += 1
        return {
            "path": path,
            "edge_ids": edge_ids,
            "distance_m": dist_m,
            "travel_time_s": time_s,
            "n_edges": len(edge_ids),
            "n_edges_default": n_default,
        }

    def route_geometry_lonlat(self, route_result):
        """Return the route as a list of (lon, lat) coordinates for plotting."""
        coords = []
        for eid in route_result["edge_ids"]:
            if eid is None:
                continue
            coords.extend(self.network.edge_coords_lonlat(int(eid)))
        return coords
