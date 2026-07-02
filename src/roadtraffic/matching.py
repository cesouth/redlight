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
:func:`roadtraffic.speeds.derive_speeds`.

A fix with no candidate edge within ``max_dist`` is always reported as
``edge_id == -1`` -- by both matchers. The HMM still carries its state past
such fixes so the rest of the trajectory stays consistently matched, but it
never fabricates a match for them.

NearestMatcher
    Snaps each point independently to its closest edge within a tolerance.
    Fast, no sequence assumption. Noisy at intersections / parallel roads.

HMMMatcher
    Hidden Markov Model with Viterbi decoding over each trajectory
    (Newson & Krumm, 2009). Emission ~ Gaussian in snap distance; transition
    compares great-circle point step to network shortest-path distance between
    candidates. Requires a trajectory id. No extra dependencies; costs compute.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

try:
    import networkx as nx
except ImportError as exc:  # pragma: no cover
    raise ImportError("roadtraffic requires networkx.") from exc


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
    """Independent nearest-edge snapping."""

    def __init__(self, network, *, max_dist: float = 50.0, k: int = 10):
        self.network = network
        self.max_dist = max_dist
        self.k = k

    def match(self, points) -> pd.DataFrame:
        df = points.df
        px, py = self.network.project_points(df["lon"].values, df["lat"].values)
        edge_ids = np.full(len(df), -1, dtype=np.int64)
        snap = np.full(len(df), np.nan)
        for i in range(len(df)):
            cands = self.network.candidate_edges(
                px[i], py[i], k=self.k, max_dist=self.max_dist
            )
            if cands:
                # candidate_edges returns candidates ordered by distance
                edge_ids[i] = cands[0][0]
                snap[i] = cands[0][1]
        return _match_frame(df, edge_ids, snap)


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
        each step is ``max(max_route_dist_factor * step, max_dist * 4)`` --
        the ``max_dist * 4`` floor keeps slow-moving (small-step) trajectories
        from pruning every transition. Same-edge transitions are always
        allowed. When no predecessor is reachable within the cutoff, the chain
        connects to the best prior state with a saturating penalty (see the
        inline note); when there is no prior state at all (leading fixes had
        no candidates), the chain restarts fresh at this fix.

    Notes
    -----
    Fixes with no candidate edge within ``max_dist`` are returned with
    ``edge_id == -1`` and ``snap_dist_m = NaN``; the Viterbi state is carried
    across them so the surrounding fixes still decode as one chain.
    """

    def __init__(self, network, *, sigma_z: float = 6.0, beta: float = 30.0,
                 max_dist: float = 50.0, k: int = 8,
                 max_route_dist_factor: float = 8.0):
        self.network = network
        self.sigma_z = sigma_z
        self.beta = beta
        self.max_dist = max_dist
        self.k = k
        self.max_route_dist_factor = max_route_dist_factor

    # emission log-prob: Gaussian in perpendicular snap distance
    def _emission_logp(self, dist):
        return -0.5 * (dist / self.sigma_z) ** 2 - math.log(
            math.sqrt(2 * math.pi) * self.sigma_z
        )

    # transition log-prob: exponential in |gc_dist - route_dist|
    def _transition_logp(self, delta):
        return -delta / self.beta - math.log(self.beta)

    def match(self, points) -> pd.DataFrame:
        if not points.has_traj:
            raise ValueError(
                "HMMMatcher requires a trajectory id column. Either provide one "
                "via load_points(id_col=...) or use NearestMatcher."
            )
        frames = []
        for tid, sub in points.trajectories():
            frames.append(self._match_one(tid, sub))
        out = pd.concat(frames, ignore_index=True)
        return out

    def _match_one(self, tid, sub) -> pd.DataFrame:
        n = len(sub)
        lon = sub["lon"].values
        lat = sub["lat"].values
        px, py = self.network.project_points(lon, lat)

        # candidate sets per observation
        cand_list = []
        for i in range(n):
            cands = self.network.candidate_edges(
                px[i], py[i], k=self.k, max_dist=self.max_dist
            )
            cand_list.append(cands)
        has_cand = [bool(c) for c in cand_list]

        # great-circle metric step between consecutive points (projected dist)
        step = np.zeros(n)
        if n > 1:
            step[1:] = np.hypot(np.diff(px), np.diff(py))

        # Viterbi
        V = [dict() for _ in range(n)]    # state -> best log-prob
        back = [dict() for _ in range(n)]  # state -> previous edge_id (or None)
        for i in range(n):
            if not has_cand[i]:
                # No candidates: carry the state set forward unchanged so the
                # chain survives the gap. This fix itself is reported as -1.
                if i > 0:
                    V[i] = dict(V[i - 1])
                    back[i] = {e: e for e in V[i]}
                continue

            if i == 0 or not V[i - 1]:
                # Start (or restart after a candidate-less prefix): emission only.
                for eid, dist, _t in cand_list[i]:
                    V[i][eid] = self._emission_logp(dist)
                    back[i][eid] = None
                continue

            cutoff = max(self.max_route_dist_factor * step[i], self.max_dist * 4)
            # One bounded Dijkstra per distinct predecessor end-node, reused
            # across every current candidate this step. Replaces O(k^2) full
            # shortest-path calls and applies the distance cutoff
            # (transitions beyond it are pruned, guarding absurd detours).
            dist_from = {}
            for peid in V[i - 1]:
                _, pv = self.network.edge_endpoints(peid)
                if pv not in dist_from:
                    dist_from[pv] = nx.single_source_dijkstra_path_length(
                        self.network.graph, pv, cutoff=cutoff, weight="length_m"
                    )
            for eid, dist, _t in cand_list[i]:
                em = self._emission_logp(dist)
                u_cur, _ = self.network.edge_endpoints(eid)
                best_lp, best_prev = -np.inf, None
                for peid, plp in V[i - 1].items():
                    if peid == eid:
                        # Staying on the same edge is always consistent: the
                        # on-network move equals the GPS step (delta 0). Without
                        # this, dense sampling -- many consecutive points on one
                        # long edge -- would need an end-to-start loop distance
                        # and be pruned, shattering the chain.
                        rd = step[i]
                    else:
                        _, pv = self.network.edge_endpoints(peid)
                        rd = dist_from[pv].get(u_cur)
                        if rd is None:
                            continue  # unreachable within cutoff
                    delta = abs(step[i] - rd)
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
