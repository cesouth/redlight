"""Where derive_speeds actually spends its time.

Task 7 measured that ``derive_speeds`` costs more than ``HMMMatcher.match`` on
the same data -- 60 % of pipeline wall time against the matcher's 39 % -- but
attributed that cost by reading the code rather than measuring it. This
profiles it properly: a cProfile run plus direct instrumentation of the two
suspects, ``_arc_position``'s per-fix shapely projection and
``_hop_distance``'s pure-Python networkx Dijkstra. Run:

    python benchmarks/profile_speeds.py --points 40000 --trajectories 100

Reuses the on-network generator from profile_hmm.py, so every fix matches and
the profile reflects real work rather than points rejected for having no
candidate edge.

Not part of the test suite (pytest ignores benchmarks/); numbers depend on
hardware, so treat them as relative, not absolute.
"""
from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import tempfile
import time
from pathlib import Path

from profile_hmm import build_grid, simulate_on_network

import redlight as rl
from redlight import speeds as _speeds


def instrument(net, matched, points):
    """One derive_speeds run with the two suspects counted and timed."""
    counters = {"arc_calls": 0, "arc_s": 0.0,
                "hop_calls": 0, "hop_s": 0.0,
                "query_calls": 0, "query_s": 0.0, "dijkstra_calls": 0}

    real_arc = _speeds._arc_position
    real_hop = _speeds._hop_distance
    real_query = _speeds._SourceDistCache.query

    def arc(*a, **kw):
        counters["arc_calls"] += 1
        t0 = time.perf_counter()
        try:
            return real_arc(*a, **kw)
        finally:
            counters["arc_s"] += time.perf_counter() - t0

    def hop(*a, **kw):
        counters["hop_calls"] += 1
        t0 = time.perf_counter()
        try:
            return real_hop(*a, **kw)
        finally:
            counters["hop_s"] += time.perf_counter() - t0

    def query(self, src, dst, cutoff):
        counters["query_calls"] += 1
        cached = self._cache.get(src)
        if cached is None or cached[0] < cutoff:
            counters["dijkstra_calls"] += 1
        t0 = time.perf_counter()
        try:
            return real_query(self, src, dst, cutoff)
        finally:
            counters["query_s"] += time.perf_counter() - t0

    _speeds._arc_position = arc
    _speeds._hop_distance = hop
    _speeds._SourceDistCache.query = query
    try:
        t0 = time.perf_counter()
        out = rl.derive_speeds(net, matched, points)
        counters["total_s"] = time.perf_counter() - t0
    finally:
        _speeds._arc_position = real_arc
        _speeds._hop_distance = real_hop
        _speeds._SourceDistCache.query = real_query

    counters["intervals"] = len(out["intervals"])
    counters["edge_obs"] = len(out["edge_observations"])
    return counters


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--points", type=int, default=40_000)
    ap.add_argument("--trajectories", type=int, default=100)
    ap.add_argument("--grid", type=int, default=30)
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    tmp = Path(tempfile.mkdtemp())
    net = rl.Network.from_geojson(build_grid(str(tmp / "grid.geojson"), args.grid))
    csv = simulate_on_network(str(tmp / "pts.csv"), args.points,
                              args.trajectories, args.grid)
    pts = rl.load_points(csv, id_col="id")
    matched = rl.HMMMatcher(net, max_dist=50.0).match(pts)
    n = len(pts.df)
    print(f"grid {args.grid}x{args.grid}, {n:,} points, {args.trajectories} "
          f"trajectories, {len(net.edge_ids):,} directed edges")
    print(f"matched {int((matched['edge_id'] != -1).sum()):,}/{n:,}\n")

    print("=" * 78)
    print("1. cProfile of derive_speeds")
    print("=" * 78)
    prof = cProfile.Profile()
    prof.enable()
    rl.derive_speeds(net, matched, pts)
    prof.disable()
    buf = io.StringIO()
    pstats.Stats(prof, stream=buf).sort_stats("cumulative").print_stats(args.top)
    print("\n".join(buf.getvalue().splitlines()[4:]))
    buf = io.StringIO()
    pstats.Stats(prof, stream=buf).sort_stats("tottime").print_stats(args.top)
    print("--- by tottime (self time) ---")
    print("\n".join(buf.getvalue().splitlines()[6:]))

    print("=" * 78)
    print("2. Instrumented run — are the two suspects the cost?")
    print("=" * 78)
    c = instrument(net, matched, pts)
    tot = c["total_s"]
    other = tot - c["arc_s"] - c["hop_s"]
    print(f"  total derive_speeds        {tot:7.2f} s   ({n / tot:,.0f} pts/s)")
    print(f"  _arc_position (shapely)    {c['arc_s']:7.2f} s   "
          f"{100 * c['arc_s'] / tot:5.1f}%   {c['arc_calls']:,} calls "
          f"({c['arc_calls'] / n:.2f}/point)")
    print(f"  _hop_distance (total)      {c['hop_s']:7.2f} s   "
          f"{100 * c['hop_s'] / tot:5.1f}%   {c['hop_calls']:,} calls")
    print(f"    of which _SourceDistCache {c['query_s']:6.2f} s   "
          f"{100 * c['query_s'] / tot:5.1f}%   {c['query_calls']:,} queries, "
          f"{c['dijkstra_calls']:,} real Dijkstras")
    print(f"    cache hit rate           "
          f"{100 * (1 - c['dijkstra_calls'] / max(c['query_calls'], 1)):5.1f}%")
    print(f"  everything else            {other:7.2f} s   "
          f"{100 * other / tot:5.1f}%")
    print(f"  produced                   {c['intervals']:,} intervals, "
          f"{c['edge_obs']:,} edge observations")

    print("\n=== scaling ===")
    for frac in (0.25, 0.5, 1.0):
        k = max(2, int(n * frac))
        sub = rl.load_points(csv, id_col="id")
        sub.df = sub.df.iloc[:k].reset_index(drop=True)
        m2 = rl.HMMMatcher(net, max_dist=50.0).match(sub)
        t0 = time.perf_counter()
        rl.derive_speeds(net, m2, sub)
        dt = time.perf_counter() - t0
        print(f"  {k:>8,} points: {dt:6.2f} s   {k / dt:>9,.0f} pts/s")


if __name__ == "__main__":
    main()
