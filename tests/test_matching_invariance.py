"""Byte-for-byte invariance of matcher output.

Task 8 optimizes ``HMMMatcher`` under a hard constraint: the decoded path must
not move. Performance work on a Viterbi decoder is exactly the kind of change
that silently reorders a tie or drops a candidate, and the existing suite pins
behaviour case by case rather than end to end.

This module pins it end to end. Several seeded networks and trajectories are
decoded and compared against expected values generated from the code as it
stood *before* any Task 8 change (commit 189a92e). Edge id sequences must be
identical *as physical roads* -- see below -- and snap distances
must agree to 1e-9 *relative*.

The tolerance is deliberately relative rather than absolute. An earlier version
pinned 1e-12 absolute, which encoded the bit pattern of the machine that
generated the expectations: CI's Linux runners differ from a macOS Intel box by
~3e-12 relative on the same arithmetic (different libm, different GEOS build),
and the test failed there while nothing was wrong. 1e-9 is still three orders of
magnitude tighter than any real change measured against it -- the mutations
below move values by percent-scale or change the road sequence outright.

Edges are compared as **physical roads**, not directed edge ids. A two-way road
is two directed edges over the same geometry at the same snap distance, and
nothing in either matcher breaks that tie on a property of the data: Task 3
established the choice is arbitrary (03-numerical-accuracy.md, behaviour (a)),
and tests/test_matching_batch.py already accepts either direction for the same
reason. Pinning the directed id pinned a coin flip, and it landed differently on
CI's Linux runners than on the machine that generated the expectations. The road
is the real output; the direction is not.

The generators live in ``tests/_synth.py`` and are shared with
``benchmarks/profile_hmm.py``, so the fixtures match the workload the Task 7
profile was taken on rather than inventing a third synthetic world.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

import redlight as rl
from _synth import build_grid, simulate_on_network

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


def _road_of(net, eid):
    """Canonical id of the physical road an edge belongs to (direction-free)."""
    return -1 if int(eid) == -1 else min(int(e) for e in net.road_edge_ids(int(eid)))


def _fingerprint(net, frame):
    return {
        "road_id": [_road_of(net, v) for v in frame["edge_id"]],
        "snap_dist_m": [None if not np.isfinite(v) else float(v)
                        for v in frame["snap_dist_m"].to_numpy(dtype=float)],
        "point_id": [int(v) for v in frame["point_id"]],
        "traj_id": [str(v) for v in frame["traj_id"]] if "traj_id" in frame else [],
    }


def _run_case(tmp_path, label, grid, points, trajectories, seed, k, max_dist):
    net, pts = _scenario(tmp_path, grid, points, trajectories, seed)
    hmm = rl.HMMMatcher(net, k=k, max_dist=max_dist).match(pts)
    near = rl.NearestMatcher(net, k=k, max_dist=max_dist).match(pts)
    return {"hmm": _fingerprint(net, hmm), "nearest": _fingerprint(net, near)}


def _compare(label, got, want):
    for matcher in ("hmm", "nearest"):
        g, w = got[matcher], want[matcher]
        assert g["road_id"] == w["road_id"], (
            f"{label}/{matcher}: decoded road sequence changed at index "
            f"{next(i for i, (a, b) in enumerate(zip(g['road_id'], w['road_id'])) if a != b)}"
        )
        assert g["point_id"] == w["point_id"], f"{label}/{matcher}: point order changed"
        assert g["traj_id"] == w["traj_id"], f"{label}/{matcher}: traj order changed"
        assert len(g["snap_dist_m"]) == len(w["snap_dist_m"]), f"{label}/{matcher}: length"
        for i, (a, b) in enumerate(zip(g["snap_dist_m"], w["snap_dist_m"])):
            if a is None or b is None:
                assert a is b, f"{label}/{matcher}: NaN-ness changed at {i}"
            else:
                assert a == pytest.approx(b, rel=1e-9, abs=1e-9), (
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
