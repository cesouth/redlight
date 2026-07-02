"""Derive speed from matched GPS trajectories via on-road displacement.

The matchers in :mod:`roadtraffic.matching` answer *which edge* each GPS fix sits
on. This module answers *how fast the vehicle was going*, by differencing
consecutive fixes within a trajectory:

    average speed over [t_i, t_{i+1}]  =  on-road distance(i -> i+1) / (t_{i+1} - t_i)

Two facts drive every design choice here:

1. **Speed is a property of an interval, not a fix.** The number computed above is
   the *mean* speed over the road actually traversed between two fixes. It is not
   instantaneous speed at either fix (recovering that from positions alone means
   differentiating a noisy trajectory, which amplifies GPS error). For
   trafficability we want the interval mean anyway, so we attribute each interval's
   speed to *every edge it traversed*.

2. **On-road distance is measured along the graph, never as the crow flies.** With
   the matched edge and the arc-length position of the snap on that edge, the
   distance from fix i to fix i+1 is a three-piece sum:

       (remaining length of edge A after the snap)
     + (graph shortest-path length from A's downstream node to B's upstream node)
     + (length of edge B up to the snap)

   When both fixes land on the same physical road it collapses to the difference of
   their arc-length positions.

Output
------
``derive_speeds`` returns a dict with two DataFrames:

``intervals``
    One row per consecutive fix pair (the per-point speed record). Columns:
    ``interval_id, traj_id, point_id_from, point_id_to, time, t_from, t_to,
    dt_s, distance_m, speed_mps, edge_from, edge_to, n_edges, snap_dist_m,
    speed_sigma_mps, speed_var, quality``. ``time`` is the interval *midpoint*
    (when the speed physically applies). **Use this frame for network-wide
    statistics** -- each row is one independent measurement.

``edge_observations``
    Long format, one row per (interval, traversed edge) -- deliberately
    duplicated so per-edge statistics see every interval that crossed the
    edge. Columns: ``interval_id, edge_id, speed_mps, time, traj_id, dt_s,
    distance_m, snap_dist_m, speed_sigma_mps, speed_var, quality``. This is
    schema-compatible with :func:`roadtraffic.cleaning.filter_by_speed`,
    :func:`roadtraffic.aggregate.aggregate_speeds` and
    :func:`roadtraffic.aggregate.assign_speeds`. Network-wide aggregations
    deduplicate on ``interval_id`` automatically (see
    :func:`roadtraffic.aggregate.aggregate_speeds`) so the duplication does
    not inflate sample sizes.

Quality
-------
The estimate's noise is dominated by GPS position error over the time gap:
``sigma_v ~ sqrt(sigma_i^2 + sigma_j^2) / dt``. An interval is flagged
``quality=False`` (but still returned) when the displacement is not clearly above
the noise floor (``distance_m < min_snr * sigma_combined``), when ``dt`` is outside
``[min_dt_s, max_dt_s]``, when either snap distance exceeds ``max_snap_dist_m``
(a non-finite snap distance also fails, unless the input carries no
``snap_dist_m`` column at all), or when the implied speed is outside
``[0, max_speed_mps]``. Keep ``quality`` rows for aggregation; the ``speed_var``
column also supports inverse-variance weighting.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from shapely.geometry import Point

try:
    import networkx as nx
except ImportError as exc:  # pragma: no cover
    raise ImportError("roadtraffic requires networkx.") from exc


# --------------------------------------------------------------------------- util
def _arc_position(network, edge_id: int, px: float, py: float) -> float:
    """Arc-length (m) of the projection of metric point (px, py) onto ``edge_id``,
    measured from that *directed* edge's start node. Uses shapely's exact
    line-projection rather than the segment-local ``t`` from candidate_edges."""
    geom = network.edge_geometry(edge_id)
    return float(geom.project(Point(px, py)))


class _SourceDistCache:
    """Lazily cached bounded single-source Dijkstra (lengths *and* paths) on an
    **undirected** view of the network.

    Speed is a magnitude, so on-road distance is measured undirected: this makes
    the estimate robust to the matcher flip-flopping between the two directed
    edges of a two-way road (a common HMM ambiguity that would otherwise inflate
    distances by a full edge length per fix). Direction still selects *which*
    road each fix sits on; it just no longer dictates the distance.

    A cached result computed with a larger cutoff may be reused for a smaller
    one, but the *requested* cutoff is always re-applied to the answer, so
    results never depend on query order.
    """

    def __init__(self, undirected_graph, weight: str = "length_m"):
        self.graph = undirected_graph
        self.weight = weight
        self._cache: dict = {}

    def query(self, src, dst, cutoff: float):
        if src == dst:
            return 0.0, [src]
        entry = self._cache.get(src)
        if entry is None or entry[0] < cutoff:
            dist, paths = nx.single_source_dijkstra(
                self.graph, src, cutoff=cutoff, weight=self.weight
            )
            entry = (cutoff, dist, paths)
            self._cache[src] = entry
        _, dist, paths = entry
        d = dist.get(dst)
        if d is None or d > cutoff:  # re-apply the caller's cutoff on cache hits
            return None, None
        return d, paths[dst]


def _build_undirected(graph):
    """Collapse the directed graph to an undirected one (min length per node pair)."""
    ug = nx.Graph()
    for u, v, d in graph.edges(data=True):
        L = float(d["length_m"])
        if ug.has_edge(u, v):
            if L < ug[u][v]["length_m"]:
                ug[u][v]["length_m"] = L
        else:
            ug.add_edge(u, v, length_m=L)
    return ug


def _canonical_arc(network, edge_id: int, s_dir: float):
    """Return (P, Q, L, arc_from_P): the road's endpoints in a canonical order and
    the snap's arc-length from P, so two opposite directed edges of one road share
    a frame. ``s_dir`` is arc-length from the *directed* edge's start node."""
    u, v = network.edge_endpoints(edge_id)
    L = network.edge_length(edge_id)
    P, Q = (u, v) if u <= v else (v, u)
    arc_from_P = s_dir if u == P else (L - s_dir)
    return P, Q, L, arc_from_P


# ------------------------------------------------------------- distance + edges
def _hop_distance(network, ucache, edge_a, s_a, edge_b, s_b, cutoff):
    """Undirected on-road distance and traversed edge set from one fix to the next.

    The vehicle can leave road A via either endpoint and enter road B via either
    endpoint; the true travelled path is the minimal such option (it didn't take a
    detour between two consecutive fixes). Returns ``(distance_m, edge_ids, ok)``.
    """
    Pa, Qa, La, aP = _canonical_arc(network, edge_a, s_a)
    Pb, Qb, Lb, bP = _canonical_arc(network, edge_b, s_b)

    exits_a = [(Pa, aP), (Qa, La - aP)]   # (exit node, cost to reach it on road A)
    enters_b = [(Pb, bP), (Qb, Lb - bP)]  # (enter node, cost from it on road B)

    best = None
    best_path = None
    for xn, xc in exits_a:
        for yn, yc in enters_b:
            spl, node_path = ucache.query(xn, yn, cutoff)
            if spl is None:
                continue
            d = xc + spl + yc
            if best is None or d < best:
                best, best_path = d, node_path

    # direct same-road traversal (no need to reach a node). Restricted to the
    # SAME physical road (the edge or its reverse), not merely edges sharing
    # endpoints -- parallel roads have different geometries and lengths.
    if edge_b in network.road_edge_ids(edge_a):
        bP_in_a = bP  # same road => same canonical frame (P, Q, L)
        d_direct = abs(aP - bP_in_a)
        if best is None or d_direct < best:
            best, best_path = d_direct, "same"

    if best is None:
        return None, [edge_a, edge_b], False

    # attribute to both directions of every traversed road (one-way roads have one)
    eids = set(network.road_edge_ids(edge_a))
    eids |= set(network.road_edge_ids(edge_b))
    if isinstance(best_path, list) and len(best_path) > 1:
        for x, y in zip(best_path[:-1], best_path[1:]):
            cands = network.edges_between(x, y)
            if cands:
                # the undirected view kept the min-length road between the pair
                eid_min = min(cands, key=lambda c: c[1])[0]
                eids |= set(network.road_edge_ids(eid_min))
    return best, sorted(eids), True


# ------------------------------------------------------------------- public entry
def derive_speeds(
    network,
    matched: pd.DataFrame,
    points,
    *,
    pos_accuracy_col: Optional[str] = None,
    default_pos_sigma_m: float = 15.0,
    min_dt_s: float = 0.5,
    max_dt_s: float = 120.0,
    max_snap_dist_m: float = 60.0,
    max_speed_mps: float = 60.0,
    min_snr: float = 3.0,
    max_route_dist_factor: float = 6.0,
    route_cutoff_floor_m: float = 300.0,
    min_baseline_m: Optional[float] = None,
) -> dict:
    """Compute per-interval and per-edge speeds from matched trajectories.

    Parameters
    ----------
    network : roadtraffic.network.Network
    matched : DataFrame
        Output of a matcher (needs ``point_id``, ``edge_id``; ``traj_id`` if
        present). Lon/lat are recovered by joining to ``points`` on ``point_id``.
    points : roadtraffic.points.PointSet
        The points that were matched (supplies lon/lat/time, and optionally a
        per-fix accuracy column).
    pos_accuracy_col : str, optional
        Column in ``points.df`` giving per-fix horizontal accuracy in metres
        (e.g. a reported accuracy or HDOP-derived value). If absent,
        ``default_pos_sigma_m`` is used for every fix.
    default_pos_sigma_m : float
        Fallback per-fix position sigma (m) when no accuracy column is given.
    min_dt_s, max_dt_s : float
        Plausible spacing between consecutive fixes; outside -> quality=False.
    max_snap_dist_m : float
        Snap distance above which a fix's match is considered unreliable.
    max_speed_mps : float
        Implausible-speed ceiling for the quality flag (does not drop rows).
    min_snr : float
        Minimum (distance / combined-sigma) for quality=True. With min_snr=3 a
        kept interval has roughly <= 33% relative speed error.
    max_route_dist_factor, route_cutoff_floor_m : float
        The middle shortest-path search is bounded at
        ``max(max_route_dist_factor * straight_step, route_cutoff_floor_m)``;
        beyond that the hop is treated as unreachable (guards detours / gaps).
    min_baseline_m : float, optional
        If set, consecutive hops are merged until their summed on-road distance
        reaches this baseline before a speed is emitted. Trades temporal/edge
        resolution for a higher signal-to-noise ratio -- recommended when fixes
        are dense and GPS is noisy. Default None (one interval per fix pair).

    Returns
    -------
    dict with keys ``"intervals"`` and ``"edge_observations"`` (see module docstring).
    """
    pdf = points.df
    keep_cols = ["point_id", "lon", "lat", "time"]
    if pos_accuracy_col is not None and pos_accuracy_col in pdf.columns:
        keep_cols.append(pos_accuracy_col)
    df = matched.merge(pdf[keep_cols], on="point_id", how="left", suffixes=("", "_pt"))

    # resolve trajectory grouping; dropna=False keeps a missing/None traj_id as
    # one group (a point set with no id column is a single trajectory)
    if "traj_id" not in df.columns:
        df["traj_id"] = None
    group_iter = df.groupby("traj_id", sort=False, dropna=False)

    has_snap_col = "snap_dist_m" in matched.columns
    cache = _SourceDistCache(_build_undirected(network.graph), weight="length_m")
    interval_rows: list[dict] = []
    edge_rows: list[dict] = []

    for tid, sub in group_iter:
        sub = sub.sort_values("time").reset_index(drop=True)
        n = len(sub)
        if n < 2:
            continue

        lon = sub["lon"].to_numpy(dtype=float)
        lat = sub["lat"].to_numpy(dtype=float)
        px, py = network.project_points(lon, lat)
        eid = sub["edge_id"].to_numpy()
        pid = sub["point_id"].to_numpy()
        tstamp = pd.to_datetime(sub["time"]).to_numpy()
        snapd = (sub["snap_dist_m"].to_numpy(dtype=float)
                 if has_snap_col else np.full(n, np.nan))
        if pos_accuracy_col is not None and pos_accuracy_col in sub.columns:
            sigma = pd.to_numeric(sub[pos_accuracy_col], errors="coerce").to_numpy(dtype=float)
            sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, default_pos_sigma_m)
        else:
            sigma = np.full(n, float(default_pos_sigma_m))

        # arc-length position of each fix on its matched edge (skip unmatched)
        s = np.full(n, np.nan)
        for i in range(n):
            if eid[i] != -1:
                s[i] = _arc_position(network, int(eid[i]), float(px[i]), float(py[i]))

        # ---- build the list of hops to emit (consecutive, or merged to a baseline)
        i = 0
        while i < n - 1:
            if eid[i] == -1:
                i += 1
                continue

            # accumulate one interval: i -> j
            j = i + 1
            acc_dist = 0.0
            acc_edges: list[int] = []
            ok = True
            last = i
            while j < n:
                if eid[j] == -1:
                    ok = False
                    break
                step = float(np.hypot(px[j] - px[last], py[j] - py[last]))
                cutoff = max(max_route_dist_factor * step, route_cutoff_floor_m)
                d_hop, edges_hop, hop_ok = _hop_distance(
                    network, cache, int(eid[last]), float(s[last]),
                    int(eid[j]), float(s[j]), cutoff,
                )
                if not hop_ok:
                    ok = False
                    break
                acc_dist += d_hop
                acc_edges.extend(edges_hop)
                last = j
                if min_baseline_m is None or acc_dist >= min_baseline_m:
                    break
                j += 1  # keep merging toward the baseline

            if not ok:
                # advance past the offending fix; do not bridge a gap silently
                i = last if last > i else i + 1
                continue

            a, b = i, last
            dt_s = (tstamp[b] - tstamp[a]) / np.timedelta64(1, "s")
            if dt_s <= 0:
                i = b
                continue

            distance_m = acc_dist
            speed_mps = distance_m / dt_s
            sigma_comb = float(np.hypot(sigma[a], sigma[b]))
            speed_sigma = sigma_comb / dt_s
            speed_var = speed_sigma ** 2
            finite_snaps = [x for x in (snapd[a], snapd[b]) if np.isfinite(x)]
            snap_pair = float(max(finite_snaps)) if finite_snaps else np.nan
            # A non-finite snap means the match quality is unknown at best; only
            # inputs with no snap column at all get the benefit of the doubt.
            snap_ok = (snap_pair <= max_snap_dist_m if np.isfinite(snap_pair)
                       else not has_snap_col)

            quality = bool(
                (distance_m >= min_snr * sigma_comb)
                and (min_dt_s <= dt_s <= max_dt_s)
                and (0.0 <= speed_mps <= max_speed_mps)
                and snap_ok
            )

            # interval midpoint time = when this average speed applies
            mid_time = tstamp[a] + (tstamp[b] - tstamp[a]) / 2
            uniq_edges = list(dict.fromkeys(int(e) for e in acc_edges))
            interval_id = len(interval_rows)

            interval_rows.append({
                "interval_id": interval_id,
                "traj_id": tid,
                "point_id_from": int(pid[a]),
                "point_id_to": int(pid[b]),
                "time": mid_time,
                "t_from": tstamp[a],
                "t_to": tstamp[b],
                "dt_s": float(dt_s),
                "distance_m": float(distance_m),
                "speed_mps": float(speed_mps),
                "edge_from": int(eid[a]),
                "edge_to": int(eid[b]),
                "n_edges": len(uniq_edges),
                "snap_dist_m": snap_pair,
                "speed_sigma_mps": float(speed_sigma),
                "speed_var": float(speed_var),
                "quality": quality,
            })
            for e in uniq_edges:
                edge_rows.append({
                    "interval_id": interval_id,
                    "edge_id": e,
                    "speed_mps": float(speed_mps),
                    "time": mid_time,
                    "traj_id": tid,
                    "dt_s": float(dt_s),
                    "distance_m": float(distance_m),
                    "snap_dist_m": snap_pair,
                    "speed_sigma_mps": float(speed_sigma),
                    "speed_var": float(speed_var),
                    "quality": quality,
                })

            i = b  # next interval starts where this one ended

    intervals = pd.DataFrame(interval_rows)
    edge_observations = pd.DataFrame(edge_rows)
    return {"intervals": intervals, "edge_observations": edge_observations}
