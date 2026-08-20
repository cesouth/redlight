"""Where HMMMatcher.match actually spends its time.

Complements bench_matching.py, which measures throughput. This one answers
*why*: a cProfile run over a realistic workload, plus direct instrumentation of
the four things the profile alone cannot show -- the bounded-Dijkstra cache hit
rate, how often the per-(candidate, predecessor) network lookups fire, how wide
the Viterbi frontier gets, and how much of the wall clock is candidate
retrieval rather than decoding. It also times derive_speeds and
aggregate_speeds on the same data, so matcher effort can be judged against the
pipeline it sits in. Run:

    python benchmarks/profile_hmm.py --points 40000 --trajectories 100

Unlike bench_matching.py's generator, movers here stay *on* the network (they
turn at the grid edge instead of driving off it), so the profile reflects real
decoding work rather than points rejected for having no candidate at all.

Not part of the test suite (pytest ignores benchmarks/); numbers depend on
hardware, so treat them as relative, not absolute.
"""
from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import resource
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

import numpy as np

import redlight as rl
from redlight import matching as _matching

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
from _synth import build_grid, simulate_on_network  # noqa: E402


class _CountingCache(_matching._CSRDistCache):
    """The real cache, counting hits, misses and evictions."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.hits = self.misses = self.evictions = 0

    def lookup(self, src_node, dst_node, cutoff: float):
        si = self._node_int.get(src_node)
        entry = self._cache.get(si) if si is not None else None
        if entry is None or entry[0] < cutoff:
            self.misses += 1
            before = len(self._cache)
            out = super().lookup(src_node, dst_node, cutoff)
            if len(self._cache) <= before and before:
                self.evictions += 1
            return out
        self.hits += 1
        return super().lookup(src_node, dst_node, cutoff)


def instrument(net, points, *, k, max_dist, cache_size):
    """Run one match with everything counted. Returns a dict of measurements."""
    m = rl.HMMMatcher(net, k=k, max_dist=max_dist, dist_cache_size=cache_size)
    csr, node_int, _ = net.csgraph()
    cache = _CountingCache(csr, node_int, maxsize=cache_size)
    m._dist_cache = cache

    calls = {"edge_endpoints": 0, "edge_length": 0}
    real_ep, real_el = net.edge_endpoints, net.edge_length

    def ep(eid):
        calls["edge_endpoints"] += 1
        return real_ep(eid)

    def el(eid):
        calls["edge_length"] += 1
        return real_el(eid)

    widths: list[int] = []
    real_batch = net.candidate_edges_batch
    real_arcs = net._candidate_arcs_batch
    spent = {"candidates": 0.0}

    def timed_arcs(*a, **kw):
        t0 = time.perf_counter()
        out = real_arcs(*a, **kw)
        spent["candidates"] += time.perf_counter() - t0
        widths.extend(len(c) for c in out)
        return out

    def timed_batch(*a, **kw):
        t0 = time.perf_counter()
        out = real_batch(*a, **kw)
        spent["candidates"] += time.perf_counter() - t0
        return out

    net.edge_endpoints, net.edge_length = ep, el
    net._candidate_arcs_batch, net.candidate_edges_batch = timed_arcs, timed_batch
    try:
        t0 = time.perf_counter()
        out = m.match(points)
        total = time.perf_counter() - t0
    finally:
        net.edge_endpoints, net.edge_length = real_ep, real_el
        net._candidate_arcs_batch = real_arcs
        net.candidate_edges_batch = real_batch

    n = len(points.df)
    return {
        "total_s": total,
        "candidates_s": spent["candidates"],
        "matched": int((out["edge_id"] != -1).sum()),
        "n_points": n,
        "cache_hits": cache.hits,
        "cache_misses": cache.misses,
        "cache_entries": len(cache._cache),
        "cache_maxsize": cache_size,
        "edge_endpoints_calls": calls["edge_endpoints"],
        "edge_length_calls": calls["edge_length"],
        "candidate_width_mean": float(np.mean(widths)) if widths else 0.0,
        "candidate_width_max": int(np.max(widths)) if widths else 0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--points", type=int, default=40_000)
    ap.add_argument("--trajectories", type=int, default=100)
    ap.add_argument("--grid", type=int, default=30)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-dist", type=float, default=50.0)
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    tmp = Path(tempfile.mkdtemp())
    net = rl.Network.from_geojson(build_grid(str(tmp / "grid.geojson"), args.grid))
    csv = simulate_on_network(str(tmp / "pts.csv"), args.points,
                              args.trajectories, args.grid)
    pts = rl.load_points(csv, id_col="id")
    print(f"grid {args.grid}x{args.grid}, {len(pts.df):,} points, "
          f"{args.trajectories} trajectories, {len(net.edge_ids):,} directed edges")
    print(f"k={args.k}  max_dist={args.max_dist}\n")

    print("=" * 78)
    print("1. cProfile of HMMMatcher.match")
    print("=" * 78)
    m = rl.HMMMatcher(net, k=args.k, max_dist=args.max_dist)
    prof = cProfile.Profile()
    tracemalloc.start()
    prof.enable()
    m.match(pts)
    prof.disable()
    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    buf = io.StringIO()
    pstats.Stats(prof, stream=buf).sort_stats("cumulative").print_stats(args.top)
    print("\n".join(buf.getvalue().splitlines()[4:]))
    buf = io.StringIO()
    pstats.Stats(prof, stream=buf).sort_stats("tottime").print_stats(args.top)
    print("--- by tottime (self time) ---")
    print("\n".join(buf.getvalue().splitlines()[6:]))

    print("=" * 78)
    print("2. Instrumented run")
    print("=" * 78)
    r = instrument(net, pts, k=args.k, max_dist=args.max_dist, cache_size=10_000)
    n, tot = r["n_points"], r["total_s"]
    looks = r["cache_hits"] + r["cache_misses"]
    print(f"  matched                     {r['matched']:,}/{n:,} "
          f"({100 * r['matched'] / n:.1f}%)")
    print(f"  total match time            {tot:.2f} s "
          f"({n / tot:,.0f} pts/s)")
    print(f"  candidate retrieval         {r['candidates_s']:.2f} s "
          f"({100 * r['candidates_s'] / tot:.1f}% of total)")
    print(f"  Viterbi + everything else   {tot - r['candidates_s']:.2f} s "
          f"({100 * (1 - r['candidates_s'] / tot):.1f}%)")
    print(f"  dist-cache lookups          {looks:,} "
          f"({looks / n:.2f} per point)")
    print(f"  dist-cache hit rate         "
          f"{100 * r['cache_hits'] / max(looks, 1):.1f}%  "
          f"({r['cache_hits']:,} hits / {r['cache_misses']:,} misses)")
    print(f"  cache entries held          {r['cache_entries']:,} of "
          f"maxsize {r['cache_maxsize']:,}  -> "
          f"{'BINDS' if r['cache_entries'] >= r['cache_maxsize'] else 'does not bind'}")
    print(f"  edge_endpoints() calls      {r['edge_endpoints_calls']:,} "
          f"({r['edge_endpoints_calls'] / n:.1f} per point)")
    print(f"  edge_length() calls         {r['edge_length_calls']:,} "
          f"({r['edge_length_calls'] / n:.1f} per point)")
    print(f"  candidates per fix          mean {r['candidate_width_mean']:.2f}, "
          f"max {r['candidate_width_max']} (k={args.k})")
    print(f"  peak traced memory          {peak / 1e6:.1f} MB")
    print(f"  peak RSS                    "
          f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} MB")

    print("\n=== does a bigger dist_cache_size help? ===")
    for size in (1_000, 10_000, 100_000):
        rr = instrument(net, pts, k=args.k, max_dist=args.max_dist, cache_size=size)
        looks = rr["cache_hits"] + rr["cache_misses"]
        print(f"  maxsize {size:>7,}: {rr['total_s']:6.2f} s   hit rate "
              f"{100 * rr['cache_hits'] / max(looks, 1):5.1f}%   "
              f"entries held {rr['cache_entries']:,}")

    print("\n=== how does the frontier widen with k and max_dist? ===")
    for k in (4, 8, 16):
        for md in (30.0, 50.0, 80.0):
            rr = instrument(net, pts, k=k, max_dist=md, cache_size=10_000)
            print(f"  k={k:<3} max_dist={md:<5.0f} candidates/fix mean "
                  f"{rr['candidate_width_mean']:.2f} max {rr['candidate_width_max']}"
                  f"   {rr['total_s']:6.2f} s")

    print("\n" + "=" * 78)
    print("3. The rest of the pipeline, same data")
    print("=" * 78)
    matched = rl.HMMMatcher(net, k=args.k, max_dist=args.max_dist).match(pts)
    t0 = time.perf_counter()
    derived = rl.derive_speeds(net, matched, pts)
    t_derive = time.perf_counter() - t0
    eo = derived["edge_observations"]
    t0 = time.perf_counter()
    rl.aggregate_speeds(eo, output_unit="mph")
    t_agg = time.perf_counter() - t0
    t0 = time.perf_counter()
    rl.NearestMatcher(net, max_dist=args.max_dist).match(pts)
    t_near = time.perf_counter() - t0
    total = tot + t_derive + t_agg
    print(f"  NearestMatcher.match      {t_near:7.2f} s")
    print(f"  HMMMatcher.match          {tot:7.2f} s   "
          f"{100 * tot / total:5.1f}% of the HMM pipeline")
    print(f"  derive_speeds             {t_derive:7.2f} s   "
          f"{100 * t_derive / total:5.1f}%   ({len(eo):,} edge observations)")
    print(f"  aggregate_speeds          {t_agg:7.2f} s   "
          f"{100 * t_agg / total:5.1f}%")


if __name__ == "__main__":
    main()
