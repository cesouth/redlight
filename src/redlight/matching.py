"""Match GPS points to network edges.

Two matchers, one output schema. Downstream code is identical regardless of
which matcher is used.

Output: a pandas DataFrame indexed like the input points, with columns
``point_id``, ``edge_id`` (matched edge, or -1 if unmatched), ``snap_dist_m``
(perpendicular distance to the matched edge, metres; NaN when unmatched),
plus the carried-through ``lon``, ``lat`` and ``time`` columns (``lon``/``lat``
let trajectory-aware cleaning measure displacement without the original
PointSet). ``speed_mps`` is carried through too, but only if the input
``PointSet`` has one -- points loaded position+time-only (no speed column,
``derive_speed=False``) match the same way and feed straight into
:func:`redlight.speeds.derive_speeds`.

A fix with no candidate edge within ``max_dist`` is always reported as
``edge_id == -1`` -- by both matchers. The HMM still carries its state past
such fixes so the rest of the trajectory stays consistently matched, but it
never fabricates a match for them.

NearestMatcher
    Snaps each point independently to its closest edge within a tolerance.
    Fully vectorised (one batch KDTree query for the whole point set); fast,
    no sequence assumption. Noisy at intersections / parallel roads.

HMMMatcher
    Hidden Markov Model with Viterbi decoding over each trajectory
    (Newson & Krumm, 2009). Emission ~ Gaussian in snap distance; transition
    compares great-circle point step to network shortest-path distance between
    candidates. Requires a trajectory id. Transition distances run on scipy's
    C Dijkstra with cross-step caching, and independent trajectories can be
    decoded in parallel processes (``n_jobs``). No extra dependencies.
"""
from __future__ import annotations

import math
import os
import pickle
from collections import OrderedDict

import numpy as np
import pandas as pd
from scipy.sparse import csgraph as _csgraph


def _match_frame(sub: pd.DataFrame, edge_ids, snap, *, traj_id=None) -> pd.DataFrame:
    """Assemble the shared matcher output schema (single source of truth)."""
    out = pd.DataFrame({
        "point_id": sub["point_id"].values,
        "edge_id": edge_ids,
        "snap_dist_m": snap,
        "lon": sub["lon"].values,
        "lat": sub["lat"].values,
        "time": sub["time"].values,
    })
    if "speed_mps" in sub.columns:
        out["speed_mps"] = sub["speed_mps"].values
    if traj_id is not None:
        out["traj_id"] = traj_id
    elif "traj_id" in sub.columns:
        out["traj_id"] = sub["traj_id"].values
    return out


class NearestMatcher:
    """Independent nearest-edge snapping (vectorised)."""

    def __init__(self, network, *, max_dist: float = 50.0, k: int = 10):
        self.network = network
        self.max_dist = max_dist
        self.k = k

    def match(self, points) -> pd.DataFrame:
        df = points.df
        px, py = self.network.project_points(df["lon"].values, df["lat"].values)
        edge_ids, snap = self.network.nearest_edges(
            px, py, k=self.k, max_dist=self.max_dist
        )
        return _match_frame(df, edge_ids, snap)


class _CSRDistCache:
    """Bounded single-source shortest-path distances, cached per source node.

    Runs scipy's C Dijkstra on the network's CSR adjacency (see
    :meth:`redlight.network.Network.csgraph`) instead of pure-Python
    networkx searches. A result computed with a larger cutoff is reused for
    smaller ones, and the *requested* cutoff is always re-applied to the
    answer, so results never depend on query order. LRU-bounded: each cached
    source holds only the nodes reachable within its cutoff.
    """

    def __init__(self, csr, node_int, maxsize: int = 10_000):
        self._csr = csr
        self._node_int = node_int
        self._maxsize = maxsize
        self._cache: OrderedDict = OrderedDict()

    def lookup(self, src_node, dst_node, cutoff: float):
        """Distance src -> dst if reachable within ``cutoff``, else None."""
        si = self._node_int.get(src_node)
        di = self._node_int.get(dst_node)
        if si is None or di is None:
            return None
        entry = self._cache.get(si)
        if entry is None or entry[0] < cutoff:
            row = np.asarray(_csgraph.dijkstra(
                self._csr, directed=True, indices=si, limit=cutoff
            )).ravel()
            finite = np.flatnonzero(np.isfinite(row))
            entry = (cutoff, dict(zip(finite.tolist(), row[finite].tolist())))
            self._cache[si] = entry
        # Always bump recency, including on a refresh: reassigning an existing
        # OrderedDict key does NOT move it, so a just-recomputed, actively-used
        # source could otherwise be evicted next -- ahead of genuinely cold ones.
        self._cache.move_to_end(si)
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)
        d = entry[1].get(di)
        if d is None or d > cutoff:  # re-apply the caller's cutoff on cache hits
            return None
        return d


class HMMMatcher:
    """HMM / Viterbi trajectory map matching (Newson & Krumm, 2009).

    Parameters
    ----------
    sigma_z : float
        GPS measurement noise std-dev in metres (emission spread). Typical
        consumer GPS: 4-10 m. Default 6.0; raise it toward your data's real
        error (this package targets 1-100 m error regimes).
    beta : float
        Transition decay parameter in metres, controlling tolerance to the
        difference between straight-line and on-network travel distance.
        Default 30.0 (a robust value from the source paper's calibration).
    max_dist : float
        Candidate snap tolerance in metres.
    k : int
        Max candidate edges per point.
    max_route_dist_factor : float
        Bounds the transition shortest-path search. The actual cutoff used at
        each step is ``max(max_route_dist_factor * gc_step, max_dist * 4)``,
        where ``gc_step`` is the straight-line distance from this fix back to
        the last fix that actually anchored a state (the immediately-previous
        fix normally, but further back across a candidate-less gap) -- the
        ``max_dist * 4`` floor keeps slow-moving (small-step) trajectories
        from pruning every transition. Same-edge transitions are always
        allowed. When no predecessor is reachable within the cutoff, the chain
        connects to the best prior state with a saturating penalty (see the
        inline note); when there is no prior state at all (leading fixes had
        no candidates), the chain restarts fresh at this fix.
    n_jobs : int
        Number of worker processes for decoding independent trajectories in
        parallel. 1 (default) = serial; -1 = all CPU cores. Results are
        identical to serial matching -- trajectories are independent; only
        the schedule changes. **Measure before enabling**: the serial path
        already sustains tens of thousands of points per second, so worker
        start-up (fresh interpreter + network deserialisation per process
        under the 'spawn' start method) and DataFrame transfer costs dominate
        until jobs run for minutes -- on the development machine serial
        matching stayed faster up to at least two million points. Workers
        also each start with a cold shortest-path cache that the serial run
        shares. As with any Python
        multiprocessing, call ``match`` from inside an
        ``if __name__ == "__main__":`` guard in scripts.
    dist_cache_size : int
        Max number of source nodes whose bounded shortest-path distances are
        kept cached across steps (per process). Larger = faster on data that
        revisits the same roads, at more memory (~tens of KB per entry).

    Notes
    -----
    Fixes with no candidate edge within ``max_dist`` are returned with
    ``edge_id == -1`` and ``snap_dist_m = NaN``; the Viterbi state is carried
    across them so the surrounding fixes still decode as one chain.
    """

    def __init__(self, network, *, sigma_z: float = 6.0, beta: float = 30.0,
                 max_dist: float = 50.0, k: int = 8,
                 max_route_dist_factor: float = 8.0,
                 n_jobs: int = 1, dist_cache_size: int = 10_000):
        self.network = network
        self.sigma_z = sigma_z
        self.beta = beta
        self.max_dist = max_dist
        self.k = k
        self.max_route_dist_factor = max_route_dist_factor
        self.n_jobs = n_jobs
        self.dist_cache_size = dist_cache_size
        self._dist_cache: _CSRDistCache | None = None

    def __getstate__(self):
        # the distance cache is a per-process working set, not matcher state;
        # dropping it keeps worker payloads small
        state = self.__dict__.copy()
        state["_dist_cache"] = None
        return state

    # emission log-prob: Gaussian in perpendicular snap distance
    def _emission_logp(self, dist):
        return -0.5 * (dist / self.sigma_z) ** 2 - math.log(
            math.sqrt(2 * math.pi) * self.sigma_z
        )

    # transition log-prob: exponential in |gc_dist - route_dist|
    def _transition_logp(self, delta):
        return -delta / self.beta - math.log(self.beta)

    def _get_dist_cache(self) -> _CSRDistCache:
        if self._dist_cache is None:
            csr, node_int, _ = self.network.csgraph()
            self._dist_cache = _CSRDistCache(csr, node_int,
                                             maxsize=self.dist_cache_size)
        return self._dist_cache

    def _resolve_n_jobs(self) -> int:
        if self.n_jobs == -1:
            return os.cpu_count() or 1
        return max(1, int(self.n_jobs))

    def match(self, points) -> pd.DataFrame:
        if not points.has_traj:
            raise ValueError(
                "HMMMatcher requires a trajectory id column. Either provide one "
                "via load_points(id_col=...) or use NearestMatcher."
            )
        trajs = list(points.trajectories())
        n_jobs = self._resolve_n_jobs()
        if n_jobs > 1 and len(trajs) > 1:
            frames = _match_parallel(self, trajs, n_jobs)
        else:
            frames = [self._match_one(tid, sub) for tid, sub in trajs]
        return pd.concat(frames, ignore_index=True)

    def _match_one(self, tid, sub) -> pd.DataFrame:
        n = len(sub)
        lon = sub["lon"].values
        lat = sub["lat"].values
        px, py = self.network.project_points(lon, lat)
        dcache = self._get_dist_cache()

        # candidate sets for the whole trajectory in one vectorised pass
        cand_list = self.network.candidate_edges_batch(
            px, py, k=self.k, max_dist=self.max_dist
        )
        has_cand = [bool(c) for c in cand_list]

        # Viterbi
        V = [dict() for _ in range(n)]      # state -> best log-prob
        back = [dict() for _ in range(n)]   # state -> previous edge_id (or None)
        snap_t = [dict() for _ in range(n)]  # state -> fractional position (0..1)
                                              # along that edge, from candidate_edges
        # anchor[i]: index whose raw GPS position (px, py) is "the" observed
        # position for whatever state V[i] currently holds. Equal to i for a row
        # with its own candidates; propagated from anchor[i-1] across a
        # candidate-less gap, since a carried-forward state's last real fix is
        # still that earlier row, not the noisy/off-network row(s) in between.
        anchor = list(range(n))
        for i in range(n):
            if not has_cand[i]:
                # No candidates: carry the state set forward unchanged so the
                # chain survives the gap. This fix itself is reported as -1.
                if i > 0:
                    V[i] = dict(V[i - 1])
                    back[i] = {e: e for e in V[i]}
                    snap_t[i] = dict(snap_t[i - 1])
                    anchor[i] = anchor[i - 1]
                continue

            if i == 0 or not V[i - 1]:
                # Start (or restart after a candidate-less prefix): emission only.
                for eid, dist, t_cur in cand_list[i]:
                    V[i][eid] = self._emission_logp(dist)
                    back[i][eid] = None
                    snap_t[i][eid] = t_cur
                continue

            # Straight-line distance from the last row that actually anchored a
            # state (which may be several rows back, across a gap) to this row --
            # NOT necessarily the immediately-previous row.
            j = anchor[i - 1]
            gc_step = float(np.hypot(px[i] - px[j], py[i] - py[j]))
            cutoff = max(self.max_route_dist_factor * gc_step, self.max_dist * 4)
            # End node per distinct predecessor, resolved once per step; the
            # bounded Dijkstra behind dcache.lookup is computed at most once
            # per (source node, cutoff regime) and reused across steps.
            prev_end = {peid: self.network.edge_endpoints(peid)[1]
                        for peid in V[i - 1]}
            for eid, dist, t_cur in cand_list[i]:
                em = self._emission_logp(dist)
                u_cur, _ = self.network.edge_endpoints(eid)
                leadin_cur = t_cur * self.network.edge_length(eid)
                best_lp, best_prev = -np.inf, None
                for peid, plp in V[i - 1].items():
                    if peid == eid:
                        # Staying on the same edge is always consistent: the
                        # on-network move equals the GPS step (delta 0). Without
                        # this, dense sampling -- many consecutive points on one
                        # long edge -- would need an end-to-start loop distance
                        # and be pruned, shattering the chain.
                        rd = gc_step
                    else:
                        node_dist = dcache.lookup(prev_end[peid], u_cur, cutoff)
                        if node_dist is None:
                            continue  # unreachable within cutoff
                        # Full on-road distance, matching the three-piece sum
                        # redlight.speeds._hop_distance uses for the same
                        # quantity: remaining length of the previous edge past
                        # its snap, the node-to-node middle, then the current
                        # edge's length up to its own snap.
                        remaining_prev = ((1.0 - snap_t[i - 1][peid])
                                          * self.network.edge_length(peid))
                        rd = remaining_prev + node_dist + leadin_cur
                    delta = abs(gc_step - rd)
                    lp = plp + self._transition_logp(delta) + em
                    if lp > best_lp:
                        best_lp, best_prev = lp, peid
                if best_prev is None:
                    # No predecessor reachable within the cutoff. Connect to the
                    # best prior state with a *saturating* penalty rather than
                    # restarting for free: a free restart (lp = emission only)
                    # would outscore every long continuous chain -- which always
                    # accumulates per-step cost -- and the backtrack would then
                    # truncate to the last restart. Staying connected keeps the
                    # Viterbi path intact; emission still picks the edge.
                    # V[i-1] is guaranteed non-empty here (handled above).
                    pbest = max(V[i - 1], key=V[i - 1].get)
                    best_lp = (V[i - 1][pbest]
                               + self._transition_logp(cutoff) + em)
                    best_prev = pbest
                V[i][eid] = best_lp
                back[i][eid] = best_prev
                snap_t[i][eid] = t_cur

        # backtrack from the last step that has any state; fixes without
        # candidates stay -1 (never fabricate a match for them)
        matched = np.full(n, -1, dtype=np.int64)
        state = None
        start_i = -1
        for i in range(n - 1, -1, -1):
            if V[i]:
                state = max(V[i], key=V[i].get)
                start_i = i
                break
        i = start_i
        while i >= 0 and state is not None:
            if has_cand[i]:
                matched[i] = state
            state = back[i].get(state)
            i -= 1

        # snap distances for the chosen edges
        snap = np.full(n, np.nan)
        for i in range(n):
            if matched[i] == -1:
                continue
            for eid, dist, _t in cand_list[i]:
                if eid == matched[i]:
                    snap[i] = dist
                    break

        return _match_frame(sub, matched, snap, traj_id=tid)


# ------------------------------------------------------------ parallel workers
# Module-level so they pickle under the 'spawn' start method (macOS/Windows).
# The matcher (with its Network) is deserialised ONCE per worker process by
# the pool initializer, not once per trajectory.
_WORKER_STATE: dict = {}


def _pool_init(matcher_bytes: bytes) -> None:
    _WORKER_STATE["matcher"] = pickle.loads(matcher_bytes)


def _pool_match_one(task):
    tid, sub = task
    return _WORKER_STATE["matcher"]._match_one(tid, sub)


def _match_parallel(matcher: HMMMatcher, trajs: list, n_jobs: int) -> list:
    from concurrent.futures import ProcessPoolExecutor

    payload = pickle.dumps(matcher)
    n_jobs = min(n_jobs, len(trajs))
    chunksize = max(1, len(trajs) // (n_jobs * 4))
    with ProcessPoolExecutor(max_workers=n_jobs, initializer=_pool_init,
                             initargs=(payload,)) as pool:
        # map preserves input order, so output matches serial ordering
        return list(pool.map(_pool_match_one, trajs, chunksize=chunksize))
