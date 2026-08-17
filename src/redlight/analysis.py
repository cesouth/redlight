"""Road-network structure measures: chokepoints, basic stats, connectivity.

Scoped to what a trafficability / speed-routing tool actually needs -- not
general urban-form analysis (contrast with ``osmnx``'s much larger stats
menu, most of which describes urban form rather than route-ability or
congestion risk):

* :func:`edge_betweenness_centrality` -- which roads carry a disproportionate
  share of shortest paths. Weighted by travel time, this finds real
  trafficability chokepoints, not just topologically central roads; weighted
  by length, it's the purely geometric equivalent; unweighted, it's pure
  topology.
* :func:`network_stats` -- circuity, streets-per-node, intersection/dead-end
  counts, and intersection/edge density. The study area behind those two
  densities is measured from the network's own projected geometry by default
  (see :func:`_detect_area_km2`), and can be overridden with a hand-measured
  ``area_km2`` whenever you know the real study boundary.
* :func:`connectivity_report` -- largest strongly-connected-component size
  and the actual node/edge partition (not just a headline number), plus a
  weakly- vs. strongly-connected distinction that separates a one-way trap
  from a genuinely disconnected extract.

All three take a :class:`~redlight.network.Network` as their first
argument and read ``network.graph`` directly -- this package's convention of
free functions over graph methods (see :mod:`redlight.aggregate`,
:mod:`redlight.routing`). No new methods are added to ``Network`` itself.
"""
from __future__ import annotations

import math

import networkx as nx
import numpy as np

from ._geo import geodesic_distance
from .network import _RESERVED_EDGE_ATTRS

# Attribute names redlight's own pipeline writes for speed/time data (see
# aggregate.py). Colliding a write_attr= with one of these would silently
# corrupt real pipeline data, so it's guarded against in
# edge_betweenness_centrality below.
_SPEED_TIME_ATTRS = frozenset({
    "obs_speed_mps", "travel_time_s",
    "obs_speed_mps_overall", "travel_time_s_overall",
    "obs_speed_mps_peak", "travel_time_s_peak",
    "obs_speed_mps_offpeak", "travel_time_s_offpeak",
})


def _invalid_weight(value) -> bool:
    """True if ``value`` isn't a usable numeric edge weight.

    Covers both a fully-absent attribute and one present but explicitly
    ``None`` -- the latter is a real, reachable case, not just theoretical:
    ``mapping.to_geojson`` writes ``"travel_time_s": None`` into its exported
    properties for any edge with no observed speed, and re-loading that
    GeoJSON via ``Network.from_geojson`` copies it straight onto the new
    graph's edges as a literal ``None`` attribute value -- present, but
    exactly as useless as being absent. A NaN is guarded too, since it would
    otherwise propagate through networkx's internal distance sums silently.

    The NaN test goes through ``math.isnan`` rather than
    ``isinstance(value, float)``, which would only catch Python floats and
    ``numpy.float64`` (a float subclass) while missing ``numpy.float32`` and
    ``Decimal`` NaNs -- those reach networkx and surface as a
    ``ZeroDivisionError`` from inside its accumulation loop, far from the
    cause. Non-numeric values raise ``TypeError`` here and are reported as
    valid, leaving type complaints to networkx itself.
    """
    if value is None:
        return True
    try:
        return math.isnan(value)
    except (TypeError, ValueError):
        return False


def _require_edge_weight(graph, weight: str, fn_name: str) -> None:
    """Raise a clear, actionable error if any edge lacks a usable ``weight``.

    networkx silently substitutes a default weight of 1.0 for a missing edge
    attribute rather than raising (verified empirically: a graph with the
    attribute missing on an edge produces byte-identical shortest-path/
    centrality output to the same graph with that edge's attribute explicitly
    set to 1.0). Left unguarded, an edge with no ``travel_time_s`` would look
    artificially cheap -- like it takes about a second to cross -- and pull
    shortest-path traffic through it, corrupting the centrality ranking. An
    edge with the attribute present but ``None``/``NaN`` is just as unusable,
    but fails differently and worse: not a wrong-but-plausible answer, a raw
    ``TypeError``/``NaN`` propagation from deep inside networkx's own
    Dijkstra implementation. Both cases are guarded here identically.
    """
    total = graph.number_of_edges()
    missing = sum(1 for *_, d in graph.edges(data=True)
                 if weight not in d or _invalid_weight(d[weight]))
    if missing:
        raise ValueError(
            f"{fn_name}: {missing} of {total} edges have no usable {weight!r} "
            "value (missing entirely, or present as None/NaN). networkx "
            "silently substitutes a default weight of 1.0 for a missing "
            "attribute (verified: identical output to the attribute being "
            "explicitly 1.0), which would make those edges "
            "look artificially cheap and corrupt the ranking. Fix by one of: "
            "(1) run assign_speeds(...) or assign_segment_speeds(..., "
            f"default_speed_mps=...) first so every edge carries {weight!r} "
            "-- note a genuinely observed 0 m/s (gridlock) edge ALSO has no "
            "travel_time_s by design (undefined/infinite travel time), so it "
            "will still trip this guard even after default_speed_mps is set; "
            "write travel_time_s onto it explicitly (e.g. float('inf')) if "
            "you want it included as 'never useful' rather than fixing this "
            "error; (2) call with weight='length_m' for distance-weighted "
            "(not time-weighted) centrality, which is always present; "
            "(3) pass weight=None for unweighted (hop-count) centrality; or "
            "(4) supply a custom numeric edge attribute you've computed for "
            "every edge."
        )


def edge_betweenness_centrality(
    network,
    *,
    weight: str | None,
    normalized: bool = True,
    k: int | None = None,
    seed: int | None = None,
    write_attr: str | None = None,
) -> dict[int, float]:
    """Edge betweenness centrality: which roads carry the most shortest paths.

    Betweenness centrality of an edge is the fraction of all-pairs shortest
    paths that pass through it. Weighted by ``"travel_time_s"``, this finds
    real trafficability chokepoints -- roads a disproportionate share of fast
    routes are forced through, so losing them (an incident, closure) would
    force the most detours -- rather than roads that merely look important
    topologically. Weighted by ``"length_m"`` it answers the same question
    for distance instead of time; unweighted (``weight=None``) it is a purely
    topological measure of structural importance, independent of any
    geometry or speed data at all.

    Parameters
    ----------
    network : redlight.network.Network
    weight : str or None
        Edge attribute to weight shortest paths by. Required -- there is no
        default, deliberately: defaulting to ``"travel_time_s"`` would break
        on any freshly-loaded network (before a speed pipeline has run), and
        defaulting to ``"length_m"`` would silently answer a different
        question than "chokepoints by real travel time" with no error at
        all. Pass ``"travel_time_s"`` for the trafficability-chokepoint
        reading (see :func:`~redlight.aggregate.assign_speeds` /
        :func:`~redlight.aggregate.assign_segment_speeds`, which write
        it), ``"length_m"`` for the purely geometric reading (always present,
        no pipeline needed), ``None`` for unweighted/topological betweenness,
        or any other numeric edge attribute you've computed yourself.
    normalized : bool
        If True (default), scores are normalised to ``[0, 1]`` (comparable
        across networks of different sizes); if False, raw path counts.
    k : int, optional
        If given, estimate betweenness from a random sample of ``k`` source
        nodes instead of every node -- Brandes' algorithm is
        :math:`O(VE)` unweighted (worse weighted), so exact computation on a
        large real-world extract can be slow; sampling trades exactness for
        speed. Must be >= 1 (``k=0`` would otherwise reach networkx's own
        internal division by the sample size and raise a raw
        ``ZeroDivisionError``; this is guarded here with a clear
        ``ValueError`` instead).
    seed : int, optional
        Random seed for the ``k`` sampling, for reproducible results.
        Ignored when ``k`` is None (exact computation has no randomness).
    write_attr : str, optional
        If given, also write each edge's score onto ``network.graph`` under
        this attribute name (default: off -- nothing is written unless you
        ask). This lets the result flow straight into
        :func:`~redlight.mapping.to_geojson`'s ``keep_tags=`` or
        :func:`~redlight.mapping.plot_speed_map` for a chokepoint map,
        without a separate join step. Must not collide with a reserved or
        pipeline-owned attribute name (``edge_id``, ``length_m``,
        ``geometry``, or any of the ``obs_speed_mps``/``travel_time_s``
        family) -- this raises rather than silently renaming, since (unlike
        e.g. ``to_geojson``'s ``keep_tags`` collision handling) this is a
        first-party naming choice for this function's own output, not
        incidental external data landing on a reserved name.

    Returns
    -------
    dict of int -> float
        ``edge_id -> centrality score``, one entry per directed edge.

    Raises
    ------
    ValueError
        If ``weight`` is a string and any edge lacks that attribute (see
        :func:`_require_edge_weight`); if ``k`` is given and less than 1; or
        if ``write_attr`` collides with a reserved/pipeline-owned name.

    Notes
    -----
    ``network.graph`` is a :class:`networkx.MultiDiGraph` (parallel roads
    between the same node pair are distinct edges); networkx's Dijkstra-based
    betweenness handles this correctly out of the box -- verified directly
    against a hand-built graph with two parallel edges of different weights,
    betweenness is attributed only to the cheaper one, exactly the same
    "cheapest parallel edge wins" semantics :class:`~redlight.routing.Router`
    already uses for routing. No separate simple-graph collapse is needed.
    """
    if k is not None and k < 1:
        raise ValueError(f"k must be >= 1 (or None for exact), got {k!r}.")

    graph = network.graph
    if weight is not None:
        _require_edge_weight(graph, weight, "edge_betweenness_centrality")
    if write_attr is not None:
        reserved = _RESERVED_EDGE_ATTRS | _SPEED_TIME_ATTRS
        if write_attr in reserved:
            raise ValueError(
                f"write_attr={write_attr!r} collides with a reserved or "
                "pipeline-owned edge attribute; pick a different name."
            )

    raw = nx.edge_betweenness_centrality(
        graph, weight=weight, normalized=normalized, k=k, seed=seed,
    )
    # MultiDiGraph keys from networkx are (u, v, key) tuples, and `key` is
    # already this package's edge_id (Network._build calls
    # graph.add_edge(a, b, key=eid, ...)).
    scores = {int(key[2]): float(v) for key, v in raw.items()}

    if write_attr is not None:
        for eid, score in scores.items():
            network.edge_data(eid)[write_attr] = score

    return scores


_AREA_METHODS = ("convex_hull", "bbox")


def _network_vertices(network):
    """Every vertex of every edge, as an ``(N, 2)`` array of **metres**.

    The projected geometry is used, not lon/lat: ``Network`` already stores
    each edge in a metric CRS (UTM, or Web Mercator/a supplied ``metric_epsg``
    -- see :meth:`redlight.network.Network._build`), so an area in those
    coordinates is already in m^2 and needs no reprojection, no equal-area
    trickery, and no PROJ. Full vertices rather than just graph nodes, so a
    road that bulges outside the hull of its own endpoints still counts.
    """
    coords = [np.asarray(network.edge_geometry(int(eid)).coords, dtype=float)
              for eid in network.edge_ids]
    return np.vstack(coords) if coords else np.empty((0, 2), dtype=float)


def _detect_area_km2(network, method: str) -> float | None:
    """Measure the study area from the network's own footprint, or decline.

    ``"convex_hull"`` is the tightest convex polygon containing every vertex;
    ``"bbox"`` is the axis-aligned bounding rectangle, which is never smaller
    and is usually a good deal larger.

    Returns ``None`` -- rather than ``0.0`` -- when the network encloses no
    area at all: an empty network, a single road, or any set of exactly
    collinear roads. Zero would be arithmetically fatal downstream, turning
    every density into a division by zero, so declining is the honest answer.

    The hull is computed by ``scipy.spatial.ConvexHull`` (Qhull), which is
    roughly 15x faster than building a shapely ``MultiPoint`` for the same
    result -- ~4 ms versus ~62 ms over 28k vertices. Qhull raises on
    degenerate input instead of returning a zero-area hull, so that case
    falls back to shapely, which handles collinear points by returning a
    ``LineString`` whose ``.area`` is a plain ``0.0``.
    """
    pts = _network_vertices(network)
    if len(pts) < 3:
        return None

    if method == "bbox":
        area_m2 = float(np.ptp(pts[:, 0]) * np.ptp(pts[:, 1]))
    else:
        try:
            from scipy.spatial import QhullError
        except ImportError:                     # scipy < 1.8
            from scipy.spatial.qhull import QhullError
        from scipy.spatial import ConvexHull
        try:
            # In 2-D, Qhull's "volume" is the enclosed area.
            area_m2 = float(ConvexHull(pts).volume)
        except QhullError:
            # Degenerate (collinear, or near enough that Qhull's initial
            # simplex is flat). shapely is slower but never raises here.
            from shapely.geometry import MultiPoint
            area_m2 = float(MultiPoint(pts).convex_hull.area)

    return (area_m2 / 1e6) if area_m2 > 0 else None


def network_stats(network, *, area_km2: float | None = None,
                  area_method: str | None = "convex_hull") -> dict:
    """Basic descriptive statistics about the network's structure.

    Parameters
    ----------
    network : redlight.network.Network
    area_km2 : float, optional
        Study-area size in square kilometres. Supplying one is a measurement
        and always wins over the detected value below, so pass it whenever
        you know your real study boundary -- a jurisdiction, a delivery
        polygon, a clipped AOI. Omit it and the area is measured from the
        network's own footprint instead.
    area_method : {'convex_hull', 'bbox', None}, default 'convex_hull'
        How to measure the area when ``area_km2`` is not supplied.
        ``'convex_hull'`` is the tightest convex polygon around every road
        vertex; ``'bbox'`` is the axis-aligned bounding rectangle. ``None``
        disables detection entirely, leaving the densities ``None`` unless
        ``area_km2`` is given.

        The hull is exact for a convex extract -- it recovers a rectangular
        lattice's true extent to well under 1%. Where it over-states is a
        **non-convex** extract, because it fills in the empty space the roads
        bend around: measured on synthetic cases, an L-shaped arterial's hull
        is 22.5 km^2 against a corridor that occupies a fraction of it, and a
        ring road's hull includes the entire doughnut hole. If your extract is
        L-shaped, ring-shaped or radial, treat a detected area as an upper
        bound -- the densities are then correspondingly a lower bound -- and
        pass ``area_km2`` when the distinction matters. No cheap automatic
        test can tell an empty wedge from an unmapped one, so this is
        reported (via ``area_method``) rather than guessed at.

    Returns
    -------
    dict
        ``n_nodes``, ``n_edges`` : int
            Totals. ``n_edges`` counts directed edges (a two-way road counts
            twice, once per direction).
        ``n_physical_roads`` : int
            Distinct physical roads, deduplicated via
            :meth:`~redlight.network.Network.road_edge_ids` so a two-way
            road counts once.
        ``n_intersections``, ``n_dead_ends`` : int
            Nodes with physical-road-degree >= 3 / == 1 respectively.
            "Physical-road-degree" counts distinct roads touching a node
            (via ``road_edge_ids``), not raw directed graph degree -- raw
            degree is misleading here since a two-way road alone already
            contributes 2 to it, making even a plain through-point (two
            collinear-ish roads meeting with nothing else joining) look like
            degree 4. Because this package's nodes are always true feature
            endpoints (never intermediate shape-points along a digitized
            road -- see :meth:`Network._build`), a physical-road-degree of 2
            reliably means a non-branching through-point, not a simplification
            artifact the way an un-simplified raw-OSM-node graph would have.
        ``streets_per_node_avg`` : float
            Mean physical-road-degree over all nodes.
        ``streets_per_node_counts`` : dict of int -> int
            ``{degree: node_count}`` distribution.
        ``circuity_avg`` : float
            ``sum(length_m) / sum(geodesic distance between edge endpoints)``
            over every directed edge. A ratio of sums, not a mean of
            per-edge ratios -- a per-edge ratio is vulnerable to blowing up
            on a short edge with a near-zero geodesic denominator; the sum/
            sum form is immune to that by construction. Whether directed
            edges are deduplicated by physical road doesn't matter for this
            formula: a two-way road's forward and reverse edges have
            identical length and identical geodesic distance (by symmetry),
            so including both scales numerator and denominator equally and
            leaves the ratio unchanged. 1.0 means every road is perfectly
            straight; higher means more circuitous.
        ``area_km2`` : float or None
            The area the densities were divided by, whether supplied or
            detected. ``None`` only when detection was disabled or declined.
        ``area_method`` : str or None
            Provenance for the line above: ``'supplied'`` when it came from
            the caller, otherwise the detection method used
            (``'convex_hull'``/``'bbox'``). ``None`` when there is no area --
            either ``area_method=None`` was passed, or the network encloses
            no area to measure (empty, a single road, or exactly collinear
            roads). Check this before quoting a density.
        ``intersection_density_km2``, ``edge_density_km2`` : float or None
            ``n_intersections / area_km2``, and metres of *physical* road
            length (not directed-edge length -- a two-way road is not
            double-counted) per km². Both ``None`` when ``area_km2`` is.

    Raises
    ------
    ValueError
        If ``area_method`` is neither ``None`` nor one of
        ``'convex_hull'``/``'bbox'``.
    """
    if area_method is not None and area_method not in _AREA_METHODS:
        raise ValueError(
            f"area_method must be None or one of {list(_AREA_METHODS)}, "
            f"got {area_method!r}"
        )
    graph = network.graph
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()

    # One entry per physical road (canonical id = min of its 1-2 directed
    # edge ids), so a two-way road is never double-counted in length/degree.
    road_length_m: dict[int, float] = {}
    road_nodes: dict[int, tuple] = {}
    total_length = 0.0
    # Endpoint pairs are collected and measured in a single vectorised call:
    # the geodesic is iterative, so calling it per edge inside this loop costs
    # ~90 us each and dominates the whole function on a city-sized graph.
    endpoints: list[tuple] = []
    for u, v, d in graph.edges(data=True):
        eid = int(d["edge_id"])
        length = float(d["length_m"])
        total_length += length
        endpoints.append((u[0], u[1], v[0], v[1]))
        rid = min(network.road_edge_ids(eid))
        if rid not in road_length_m:
            road_length_m[rid] = length
            road_nodes[rid] = (u, v)

    if endpoints:
        ep = np.asarray(endpoints, dtype=float)
        total_gc = float(np.sum(geodesic_distance(ep[:, 0], ep[:, 1],
                                                  ep[:, 2], ep[:, 3])))
    else:
        total_gc = 0.0

    node_roads: dict = {n: set() for n in graph.nodes()}
    for rid, (u, v) in road_nodes.items():
        node_roads[u].add(rid)
        node_roads[v].add(rid)

    degrees = [len(roads) for roads in node_roads.values()]
    n_intersections = sum(1 for deg in degrees if deg >= 3)
    n_dead_ends = sum(1 for deg in degrees if deg == 1)
    streets_per_node_avg = (sum(degrees) / n_nodes) if n_nodes else float("nan")
    streets_per_node_counts: dict[int, int] = {}
    for deg in degrees:
        streets_per_node_counts[deg] = streets_per_node_counts.get(deg, 0) + 1

    circuity_avg = (total_length / total_gc) if total_gc > 0 else float("nan")

    # A supplied area is a measurement and outranks a detected one; detection
    # only runs when the caller left it out.
    if area_km2 is not None:
        area_source: str | None = "supplied"
    elif area_method is not None:
        area_km2 = _detect_area_km2(network, area_method)
        area_source = area_method if area_km2 is not None else None
    else:
        area_source = None

    intersection_density_km2 = None
    edge_density_km2 = None
    if area_km2 is not None:
        intersection_density_km2 = n_intersections / area_km2
        edge_density_km2 = sum(road_length_m.values()) / area_km2

    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "n_physical_roads": len(road_length_m),
        "n_intersections": n_intersections,
        "n_dead_ends": n_dead_ends,
        "streets_per_node_avg": streets_per_node_avg,
        "streets_per_node_counts": streets_per_node_counts,
        "circuity_avg": circuity_avg,
        "area_km2": area_km2,
        "area_method": area_source,
        "intersection_density_km2": intersection_density_km2,
        "edge_density_km2": edge_density_km2,
    }


def connectivity_report(network) -> dict:
    """Diagnose the network's connectivity: is it all one routable piece?

    Pure topology -- no speed data or ``weight`` involved. Useful to run
    before routing on an unfamiliar or clipped network, since it distinguishes
    two very different failure modes a caller would otherwise only discover
    one failed :meth:`~redlight.routing.Router.route` call at a time: a
    *one-way trap* (every node is reachable if you ignore direction, but some
    are unreachable respecting it -- e.g. a neighbourhood only exited via one
    oneway street) versus a *genuinely disconnected* extract (e.g. a clipped
    OSM download that split the road network into unconnected pieces).

    Parameters
    ----------
    network : redlight.network.Network

    Returns
    -------
    dict
        ``n_nodes``, ``n_edges`` : int
            Totals.
        ``is_strongly_connected`` : bool
            True iff every node can reach every other node *respecting*
            one-way restrictions.
        ``n_strongly_connected_components`` : int
        ``strongly_connected_component_sizes`` : list of int
            Node counts per strongly-connected component, largest first.
        ``largest_component_fraction_nodes`` : float
            Largest component's node count / ``n_nodes`` -- how much of the
            network is actually mutually reachable.
        ``largest_component_nodes`` : list of (lon, lat)
            The node keys in the largest strongly-connected component.
        ``largest_component_edge_ids`` : list of int
            Directed edge ids with both endpoints in the largest component
            (i.e. the "routable core" of the network).
        ``stranded_nodes`` : list of (lon, lat)
            Every node not in the largest component.
        ``stranded_edge_ids`` : list of int
            Every edge not in ``largest_component_edge_ids`` (inside a
            smaller component, or bridging between components in a direction
            that doesn't count as strongly connected).
        ``is_weakly_connected``, ``n_weakly_connected_components`` : bool, int
            Same questions ignoring edge direction entirely -- the
            "one-way trap" diagnostic: ``is_strongly_connected=False`` with
            ``is_weakly_connected=True`` means the network is one connected
            piece geometrically, but one-way restrictions prevent some nodes
            reaching others; ``is_weakly_connected=False`` means there is no
            way to get between some parts of the network at all, one-way
            restrictions or not. This mirrors the same diagnosis
            :meth:`~redlight.routing.Router.route` already makes
            internally when a route fails
            (``nx.has_path(graph.to_undirected(...), ...)``), exposed here so
            it can be checked proactively instead of one failed query at a
            time.

    Notes
    -----
    To build an actual ``networkx`` subgraph of just the routable core from
    ``largest_component_edge_ids``:
    ``[(*network.edge_endpoints(eid), eid) for eid in
    result["largest_component_edge_ids"]]``. Reconstructing a full, cleaned
    :class:`~redlight.network.Network` from that (rebuilding the spatial
    index, segment table, and other internal structures
    :meth:`Network._build` maintains) is out of scope here -- this function
    only diagnoses, it doesn't repair.
    """
    graph = network.graph
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()

    # Sorted largest-first so sccs[0] is always the largest component; this
    # also gives is_strongly_connected "for free" (true iff there's only one)
    # without a second full-graph traversal via nx.is_strongly_connected.
    sccs = sorted(nx.strongly_connected_components(graph), key=len, reverse=True)
    sizes = [len(c) for c in sccs]
    largest = sccs[0] if sccs else set()

    largest_edge_ids = []
    stranded_edge_ids = []
    for u, v, d in graph.edges(data=True):
        eid = int(d["edge_id"])
        if u in largest and v in largest:
            largest_edge_ids.append(eid)
        else:
            stranded_edge_ids.append(eid)

    n_wcc = nx.number_weakly_connected_components(graph)

    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "is_strongly_connected": len(sccs) <= 1,
        "n_strongly_connected_components": len(sccs),
        "strongly_connected_component_sizes": sizes,
        "largest_component_fraction_nodes": (
            len(largest) / n_nodes if n_nodes else float("nan")
        ),
        "largest_component_nodes": sorted(largest),
        "largest_component_edge_ids": sorted(largest_edge_ids),
        "stranded_nodes": sorted(n for n in graph.nodes() if n not in largest),
        "stranded_edge_ids": sorted(stranded_edge_ids),
        "is_weakly_connected": n_wcc <= 1,
        "n_weakly_connected_components": n_wcc,
    }
