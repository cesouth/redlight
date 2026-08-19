"""Byte-for-byte invariance of matcher output.

Task 8 optimizes ``HMMMatcher`` under a hard constraint: the decoded path must
not move. Performance work on a Viterbi decoder is exactly the kind of change
that silently reorders a tie or drops a candidate, and the existing suite pins
behaviour case by case rather than end to end.

This module pins it end to end. Several seeded networks and trajectories are
decoded and compared against expected values generated from the code as it
stood *before* any Task 8 change (commit 189a92e). Edge id sequences must be
identical; snap distances must agree to 1e-12.

The generators deliberately reuse ``benchmarks/profile_hmm.py`` so the fixtures
match the workload the profile was taken on, rather than inventing a third
synthetic world.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

import redlight as rl

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "benchmarks"))
from profile_hmm import build_grid, simulate_on_network  # noqa: E402

EXPECTED = Path(__file__).resolve().parent / "data" / "matching_invariance.json"

# (label, grid, points, trajectories, seed, k, max_dist)
CASES = [
    ("grid8_dense", 8, 600, 6, 0, 8, 50.0),
    ("grid8_sparse", 8, 200, 10, 1, 4, 30.0),
    ("grid12_wide", 12, 900, 9, 2, 16, 80.0),
]


def _scenario(tmp_path, grid, points, trajectories, seed):
    net = rl.Network.from_geojson(
        build_grid(str(tmp_path / f"g{grid}.geojson"), grid))
    csv = simulate_on_network(str(tmp_path / f"p{seed}.csv"), points,
                              trajectories, grid, seed=seed)
    return net, rl.load_points(csv, id_col="id")


def _fingerprint(frame):
    return {
        "edge_id": [int(v) for v in frame["edge_id"]],
        "snap_dist_m": [None if not np.isfinite(v) else float(v)
                        for v in frame["snap_dist_m"].to_numpy(dtype=float)],
        "point_id": [int(v) for v in frame["point_id"]],
        "traj_id": [str(v) for v in frame["traj_id"]] if "traj_id" in frame else [],
    }


def _run_case(tmp_path, label, grid, points, trajectories, seed, k, max_dist):
    net, pts = _scenario(tmp_path, grid, points, trajectories, seed)
    hmm = rl.HMMMatcher(net, k=k, max_dist=max_dist).match(pts)
    near = rl.NearestMatcher(net, k=k, max_dist=max_dist).match(pts)
    return {"hmm": _fingerprint(hmm), "nearest": _fingerprint(near)}


def _compare(label, got, want):
    for matcher in ("hmm", "nearest"):
        g, w = got[matcher], want[matcher]
        assert g["edge_id"] == w["edge_id"], (
            f"{label}/{matcher}: decoded edge sequence changed at index "
            f"{next(i for i, (a, b) in enumerate(zip(g['edge_id'], w['edge_id'])) if a != b)}"
        )
        assert g["point_id"] == w["point_id"], f"{label}/{matcher}: point order changed"
        assert g["traj_id"] == w["traj_id"], f"{label}/{matcher}: traj order changed"
        assert len(g["snap_dist_m"]) == len(w["snap_dist_m"]), f"{label}/{matcher}: length"
        for i, (a, b) in enumerate(zip(g["snap_dist_m"], w["snap_dist_m"])):
            if a is None or b is None:
                assert a is b, f"{label}/{matcher}: NaN-ness changed at {i}"
            else:
                assert a == pytest.approx(b, abs=1e-12), (
                    f"{label}/{matcher}: snap distance moved at {i}: {a} != {b}")


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_matcher_output_is_unchanged(case, tmp_path):
    """Decoded paths and snap distances must not move."""
    assert EXPECTED.exists(), (
        f"{EXPECTED} is missing; regenerate with "
        "`python tests/test_matching_invariance.py --write`"
    )
    want = json.loads(EXPECTED.read_text())
    label = case[0]
    assert label in want, f"no stored expectation for {label}"
    _compare(label, _run_case(tmp_path, *case), want[label])


def _write() -> None:
    """Regenerate the stored expectations from the current code."""
    import tempfile
    out = {}
    for case in CASES:
        out[case[0]] = _run_case(Path(tempfile.mkdtemp()), *case)
    EXPECTED.parent.mkdir(parents=True, exist_ok=True)
    EXPECTED.write_text(json.dumps(out, indent=1))
    print(f"wrote {EXPECTED} ({EXPECTED.stat().st_size:,} bytes)")


if __name__ == "__main__":
    if "--write" in sys.argv:
        _write()
    else:
        print(__doc__)
