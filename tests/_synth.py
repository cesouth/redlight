"""Synthetic networks and trajectories shared by tests and benchmarks.

These generators started life in ``benchmarks/profile_hmm.py``. They moved here
when the matcher and speeds invariance tests began using them: ``MANIFEST.in``
prunes ``benchmarks/`` from the sdist as development scaffolding, so importing
them from there made the shipped test suite fail at collection -- the exact
failure ``MANIFEST.in`` exists to prevent. ``recursive-include tests *.py``
ships this file, and the benchmarks import it from here.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

GRID_SPACING_DEG = 0.001  # ~111 m


def build_grid(path: str, n: int) -> str:
    """An n x n grid of two-way streets, written as GeoJSON."""

    feats = []
    for i in range(n):
        for j in range(n - 1):
            for coords in (
                [[j * GRID_SPACING_DEG, i * GRID_SPACING_DEG],
                 [(j + 1) * GRID_SPACING_DEG, i * GRID_SPACING_DEG]],
                [[i * GRID_SPACING_DEG, j * GRID_SPACING_DEG],
                 [i * GRID_SPACING_DEG, (j + 1) * GRID_SPACING_DEG]],
            ):
                feats.append({
                    "type": "Feature",
                    "properties": {"highway": "residential"},
                    "geometry": {"type": "LineString", "coordinates": coords},
                })
    Path(path).write_text(json.dumps({"type": "FeatureCollection",
                                      "features": feats}))
    return path


def simulate_on_network(path: str, n_points: int, n_traj: int, grid_n: int,
                        noise_deg: float = 6e-5, seed: int = 0) -> str:
    """Movers that stay on the grid: drive to the edge, then turn.

    bench_matching.py's generator walks movers off the network after ~136
    fixes, so at large sizes most points have no candidate edge and are
    rejected before any Viterbi work happens. Turning instead keeps every fix
    matchable, which is what a profile of the *decoder* needs.
    """
    import pandas as pd

    rng = np.random.default_rng(seed)
    per_traj = max(2, n_points // n_traj)
    span = (grid_n - 1) * GRID_SPACING_DEG
    step = 2.2e-4  # ~24 m per fix
    t0 = np.datetime64("2026-06-01T06:00:00")
    rows = []
    for t in range(n_traj):
        lon = rng.uniform(0, span)
        lat = rng.integers(0, grid_n) * GRID_SPACING_DEG
        along_lon, direction = True, 1.0
        start = t0 + np.timedelta64(int(rng.integers(0, 43200)), "s")
        for k in range(per_traj):
            if along_lon:
                lon += direction * step
                if not 0.0 <= lon <= span:          # bounce and turn a corner
                    lon = min(max(lon, 0.0), span)
                    along_lon, direction = False, -direction
            else:
                lat += direction * step
                if not 0.0 <= lat <= span:
                    lat = min(max(lat, 0.0), span)
                    along_lon, direction = True, -direction
            rows.append({
                "id": f"veh{t:05d}",
                "lon": lon + rng.normal(0, noise_deg),
                "lat": lat + rng.normal(0, noise_deg),
                "time": str(start + np.timedelta64(3 * k, "s")),
            })
    pd.DataFrame(rows).to_csv(path, index=False)
    return path
