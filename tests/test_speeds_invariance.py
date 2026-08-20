"""Byte-for-byte invariance of derive_speeds output.

The matcher has such a pin (tests/test_matching_invariance.py); the speed
derivation is the larger share of pipeline wall time (Task 7: 60 % against the
matcher's 39 %) and had none. Optimizing it means touching the loop that
assembles every interval, which is exactly the kind of change that silently
drops a row or shifts a dtype.

Expectations were generated from the code as it stood at commit caebd20,
before any speeds optimization. Ids, dtypes, columns and row counts must match
exactly; numeric columns must agree to 1e-9 *relative*. Dtypes are compared by
*kind* (float stays float, bool stays bool) rather than by exact string, since
the supported pandas range spans ``datetime64[ns]`` and ``datetime64[us]``.

The tolerance is relative, not absolute, and deliberately so: an earlier version
pinned 1e-12 absolute and encoded the bit pattern of the machine that generated
the expectations, so it failed on CI's Linux runners (~3e-12 relative apart on
the same arithmetic) while nothing was wrong. 1e-9 still catches a 1e-7
perturbation, which is what this test is for.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

import redlight as rl
from _synth import build_grid, simulate_on_network

EXPECTED = Path(__file__).resolve().parent / "data" / "speeds_invariance.json"

# (label, grid, points, trajectories, seed, max_dist, min_baseline_m)
CASES = [
    ("grid8_plain", 8, 600, 6, 0, 50.0, None),
    ("grid8_merged", 8, 600, 6, 0, 50.0, 150.0),
    ("grid12_wide", 12, 900, 9, 2, 80.0, None),
]

NUMERIC = ["dt_s", "distance_m", "speed_mps", "snap_dist_m",
           "speed_sigma_mps", "speed_var"]


def _fingerprint(net, res):
    out = {}
    for key in ("intervals", "edge_observations"):
        f = res[key]
        rec = {"n_rows": int(len(f)),
               "columns": list(f.columns),
               # dtype *kind*, not the exact string: the supported pandas
               # range spans datetime64[ns] and datetime64[us], and int width
               # varies by platform. The invariant worth holding is that a
               # float column stays float and a bool column stays bool -- the
               # ways a column-wise rewrite actually goes wrong.
               "dtype_kinds": [getattr(d, "kind", str(d)) for d in f.dtypes]}
        for col in NUMERIC:
            if col in f.columns:
                rec[col] = [None if not np.isfinite(v) else float(v)
                            for v in f[col].to_numpy(dtype=float)]
        # edge columns are canonicalised to the physical road: a two-way road
        # is two directed edges over one geometry and nothing breaks that tie
        # on the data, so the direction is not a portable property (see
        # tests/test_matching_invariance.py).
        for col in ("interval_id", "point_id_from", "point_id_to", "n_edges"):
            if col in f.columns:
                rec[col] = [int(v) for v in f[col]]
        for col in ("edge_id", "edge_from", "edge_to"):
            if col in f.columns:
                rec[col] = sorted(
                    -1 if int(v) == -1
                    else min(int(e) for e in net.road_edge_ids(int(v)))
                    for v in f[col]) if col == "edge_id" else [
                    -1 if int(v) == -1
                    else min(int(e) for e in net.road_edge_ids(int(v)))
                    for v in f[col]]
        if "quality" in f.columns:
            rec["quality"] = [bool(v) for v in f["quality"]]
        if "time" in f.columns:
            rec["time"] = [str(v) for v in f["time"]]
        out[key] = rec
    return out


def _run_case(tmp_path, label, grid, points, trajectories, seed, max_dist, baseline):
    net = rl.Network.from_geojson(build_grid(str(tmp_path / f"g{grid}.geojson"), grid))
    csv = simulate_on_network(str(tmp_path / f"p{seed}.csv"), points,
                              trajectories, grid, seed=seed)
    pts = rl.load_points(csv, id_col="id")
    matched = rl.HMMMatcher(net, max_dist=max_dist).match(pts)
    return _fingerprint(net, rl.derive_speeds(net, matched, pts,
                                              min_baseline_m=baseline))


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_derive_speeds_output_is_unchanged(case, tmp_path):
    assert EXPECTED.exists(), (
        f"{EXPECTED} is missing; regenerate with "
        "`python tests/test_speeds_invariance.py --write`")
    want = json.loads(EXPECTED.read_text())
    label = case[0]
    assert label in want, f"no stored expectation for {label}"
    got = _run_case(tmp_path, *case)
    want = want[label]
    for key in ("intervals", "edge_observations"):
        g, w = got[key], want[key]
        assert g["n_rows"] == w["n_rows"], f"{label}/{key}: row count"
        assert g["columns"] == w["columns"], f"{label}/{key}: columns"
        assert g["dtype_kinds"] == w["dtype_kinds"], f"{label}/{key}: dtype kinds"
        for col, expect in w.items():
            if col in ("n_rows", "columns", "dtype_kinds"):
                continue
            actual = g[col]
            assert len(actual) == len(expect), f"{label}/{key}/{col}: length"
            for i, (a, b) in enumerate(zip(actual, expect)):
                if isinstance(b, float):
                    assert a is not None and \
                        a == pytest.approx(b, rel=1e-9, abs=1e-9), \
                        f"{label}/{key}/{col}[{i}]: {a} != {b}"
                else:
                    assert a == b, f"{label}/{key}/{col}[{i}]: {a} != {b}"


def _write() -> None:
    out = {c[0]: _run_case(Path(tempfile.mkdtemp()), *c) for c in CASES}
    EXPECTED.parent.mkdir(parents=True, exist_ok=True)
    EXPECTED.write_text(json.dumps(out, indent=1))
    print(f"wrote {EXPECTED} ({EXPECTED.stat().st_size:,} bytes)")


if __name__ == "__main__":
    if "--write" in sys.argv:
        _write()
    else:
        print(__doc__)
