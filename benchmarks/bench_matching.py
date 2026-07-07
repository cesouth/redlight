"""Matching throughput benchmark on synthetic data.

Builds a grid road network, simulates noisy GPS trajectories driving along
it, and times NearestMatcher and HMMMatcher (serial and parallel). Run:

    python benchmarks/bench_matching.py --points 200000 --trajectories 200

Not part of the test suite (pytest ignores benchmarks/); numbers depend on
hardware, so treat them as relative, not absolute.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time

import numpy as np

import roadtraffic as rt

GRID_SPACING_DEG = 0.001  # ~111 m


def build_grid(path: str, n: int) -> str:
    feats = []
    for i in range(n):
        for j in range(n):
            lon, lat = j * GRID_SPACING_DEG, i * GRID_SPACING_DEG
            for coords in ([[lon, lat], [lon + GRID_SPACING_DEG, lat]] if j < n - 1 else [],
                           [[lon, lat], [lon, lat + GRID_SPACING_DEG]] if i < n - 1 else []):
                if coords:
                    feats.append({"type": "Feature",
                                  "properties": {"highway": "residential"},
                                  "geometry": {"type": "LineString",
                                               "coordinates": coords}})
    with open(path, "w") as fh:
        json.dump({"type": "FeatureCollection", "features": feats}, fh)
    return path


def simulate_points(path: str, n_points: int, n_traj: int, grid_n: int,
                    noise_deg: float = 6e-5, seed: int = 0) -> str:
    """Movers drive east along random grid rows at ~8 m/s, 1 fix / 3 s."""
    rng = np.random.default_rng(seed)
    per_traj = max(2, n_points // n_traj)
    rows = []
    t0 = np.datetime64("2026-06-01T06:00:00")
    for t in range(n_traj):
        lat_row = rng.integers(0, grid_n) * GRID_SPACING_DEG
        lon0 = rng.uniform(0, GRID_SPACING_DEG)
        start = t0 + np.timedelta64(int(rng.integers(0, 43200)), "s")
        for k in range(per_traj):
            lon = lon0 + k * 2.2e-4  # ~24 m / 3 s
            rows.append({
                "id": f"veh{t:05d}",
                "lon": lon + rng.normal(0, noise_deg),
                "lat": lat_row + rng.normal(0, noise_deg),
                "time": str(start + np.timedelta64(3 * k, "s")),
            })
    import pandas as pd
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def bench(label: str, fn, n_points: int):
    t0 = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t0
    matched = int((out["edge_id"] != -1).sum())
    print(f"{label:<34} {dt:8.2f} s   {n_points / dt:>12,.0f} pts/s   "
          f"matched {matched}/{n_points}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", type=int, default=200_000)
    ap.add_argument("--trajectories", type=int, default=200)
    ap.add_argument("--grid", type=int, default=30)
    ap.add_argument("--n-jobs", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--skip-hmm", action="store_true")
    args = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix="rt_bench_")
    print(f"grid {args.grid}x{args.grid}, {args.points:,} points, "
          f"{args.trajectories} trajectories")
    net = rt.Network.from_geojson(build_grid(os.path.join(tmp, "net.json"),
                                             args.grid))
    print(f"network: {net.number_of_edges():,} directed edges")
    pts = rt.load_points(simulate_points(os.path.join(tmp, "pts.csv"),
                                         args.points, args.trajectories,
                                         args.grid))
    n = len(pts)

    bench("NearestMatcher", lambda: rt.NearestMatcher(net, max_dist=60).match(pts), n)
    if not args.skip_hmm:
        bench("HMMMatcher (serial)",
              lambda: rt.HMMMatcher(net, max_dist=60, n_jobs=1).match(pts), n)
        if args.n_jobs > 1:
            bench(f"HMMMatcher (n_jobs={args.n_jobs})",
                  lambda: rt.HMMMatcher(net, max_dist=60,
                                        n_jobs=args.n_jobs).match(pts), n)


if __name__ == "__main__":
    main()
