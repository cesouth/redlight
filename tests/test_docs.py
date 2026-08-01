"""Execute the Python examples embedded in the documentation.

Documentation drifts silently: a parameter is renamed, a return key changes,
and the prose keeps describing an API that no longer exists. Nothing catches
it, because docs are not run. This module runs them.

Every fenced ``python`` block in README.md and docs/*.md is executed against a
shared namespace pre-populated with a small road network and a matched GPS
sample, so snippets can be short and realistic instead of repeating four lines
of setup. Blocks that cannot run in a test -- shell commands, network access,
deliberately-failing illustrations -- opt out with an HTML comment on the line
before the fence::

    <!-- skip-test -->
    ```python
    net = rt.Network.from_overpass(bbox)   # needs the internet
    ```

Add ``<!-- skip-test: reason -->`` to record why.

The namespace is rebuilt for each block, so blocks cannot depend on each other
and can be reordered or edited freely. That rebuild is not optional: a shallow
copy would share the ``Network`` object, and a block calling ``assign_speeds``
without a default silently strips ``travel_time_s`` from edges a later block
needs -- making the suite pass or fail on block order.
"""
from __future__ import annotations

import re
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import roadtraffic as rt

REPO = Path(__file__).resolve().parent.parent
DOC_FILES = [REPO / "README.md"] + sorted((REPO / "docs").glob("*.md"))

# ```python ... ``` with an optional <!-- skip-test --> on the preceding line.
BLOCK = re.compile(
    r"(?P<skip><!--\s*skip-test.*?-->\s*\n)?```python\n(?P<code>.*?)```",
    re.S,
)


# --------------------------------------------------------------- fixtures
def _build_namespace(workdir: Path):
    """A small but complete analysis state for snippets to use.

    ``workdir`` becomes the process CWD for the duration of the session and
    holds ``network.geojson`` and ``points.csv``, so documentation can use
    those natural filenames instead of contorting itself for the test.
    """
    lon0, lat0 = -77.30, 38.80
    dlon, dlat = 0.009, 0.008
    feats = []
    for r in range(3):
        for c in range(4):
            lon, lat = lon0 + c * dlon, lat0 + r * dlat
            props = {"name": f"Road {r}", "highway": "residential",
                     "maxspeed": "30 mph", "oneway": "no"}
            if c < 3:
                feats.append({"type": "Feature", "properties": dict(props),
                              "geometry": {"type": "LineString", "coordinates":
                                           [[lon, lat], [lon + dlon, lat]]}})
            if r < 2:
                feats.append({"type": "Feature", "properties": dict(props),
                              "geometry": {"type": "LineString", "coordinates":
                                           [[lon, lat], [lon, lat + dlat]]}})

    import json
    (workdir / "network.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": feats}))

    # The sample carries BOTH a logged speed and a per-fix accuracy, because
    # the docs legitimately describe both paths: reading a logged speed, and
    # reconstructing one from positions with derive_speeds.
    rng = np.random.default_rng(0)
    rows = []
    for trip in range(14):
        speed = 12.0 if trip % 3 else 3.0        # a few genuinely slow movers
        step = speed * 15.0 / 111319.5
        t0 = pd.Timestamp("2026-06-01 06:00:00") + pd.Timedelta(hours=trip)
        for i in range(10):
            rows.append({
                # Both spellings, because different docs use different names
                # for the mover id and neither is wrong.
                "vehicle_id": f"m{trip}",
                "device_id": f"m{trip}",
                "timestamp": (t0 + pd.Timedelta(seconds=15 * i)).isoformat(),
                "longitude": lon0 + 0.0005 + i * step,
                "latitude": lat0 + float(rng.normal(0, 3e-5)),
                "speed": round(speed * 2.23694 + float(rng.normal(0, 1.5)), 2),
                "accuracy_m": round(float(rng.uniform(4, 12)), 1),
            })
    pd.DataFrame(rows).to_csv(workdir / "points.csv", index=False)

    net = rt.Network.from_geojson("network.geojson")
    points = rt.load_points("points.csv", id_col="device_id",
                            time_col="timestamp", lon_col="longitude",
                            lat_col="latitude", speed_col="speed",
                            speed_unit="mph")
    matched = rt.HMMMatcher(net, max_dist=60).match(points)
    derived = rt.derive_speeds(net, matched, points,
                               pos_accuracy_col="accuracy_m")
    obs = derived["edge_observations"]
    clean = rt.filter_by_speed(obs, max_speed=80, unit="mph")
    hourly = rt.aggregate_speeds(clean, block_hours=1, statistic="both",
                                 output_unit="mph")
    # default_speed_mps fills edges the sample never traversed, so snippets
    # that route or compute betweenness have a travel_time_s on every edge.
    seg = rt.assign_segment_speeds(net, clean, statistic="median",
                                   default_speed_mps=11.176)
    movers = rt.classify_movers(derived["intervals"], threshold=6.0, unit="mph")
    router = rt.Router(net)
    nodes = sorted(net.graph.nodes)
    origin, destination = nodes[0], nodes[-1]

    return {
        "rt": rt, "np": np, "pd": pd,
        "net": net, "network": net,
        "points": points, "pts": points,
        "matched": matched,
        "derived": derived,
        "intervals": derived["intervals"],
        "edge_observations": obs, "obs": obs,
        "clean": clean,
        "hourly": hourly, "aggregated": hourly,
        "seg": seg,
        "movers": movers,
        "router": router,
        # Route endpoints, under the short names the routing docs use when
        # continuing a worked example across several blocks.
        "origin": origin, "destination": destination,
        "o": origin, "d": destination,
    }


@pytest.fixture(scope="session")
def _doc_workdir(tmp_path_factory):
    """One scratch directory for the session, holding the sample input files.

    Snippets may write files (GeoJSON exports, PNGs); doing that here keeps the
    repository clean whatever the docs demonstrate. Only the *files* are shared
    across blocks -- every live object is rebuilt per block.
    """
    import os
    workdir = tmp_path_factory.mktemp("docs")
    prev = os.getcwd()
    os.chdir(workdir)
    try:
        yield workdir
    finally:
        os.chdir(prev)


@pytest.fixture
def doc_namespace(_doc_workdir):
    return _build_namespace(_doc_workdir)


def _blocks():
    """(doc-relative-path, 1-based block index, line number, code) per block."""
    out = []
    for path in DOC_FILES:
        if not path.exists():
            continue
        text = path.read_text()
        for n, m in enumerate(BLOCK.finditer(text), start=1):
            if m.group("skip"):
                continue
            line = text[:m.start()].count("\n") + 1
            # A fence nested inside a list item is indented; dedent it or
            # every such block is an IndentationError rather than a real test.
            code = textwrap.dedent(m.group("code"))
            out.append((path.relative_to(REPO).as_posix(), n, line, code))
    return out


BLOCKS = _blocks()


def test_documentation_contains_python_examples():
    """Guard the guard: if the extractor silently matches nothing, every
    other test in this module passes vacuously."""
    assert len(BLOCKS) >= 10, (
        f"only {len(BLOCKS)} runnable python blocks found across "
        f"{len(DOC_FILES)} docs -- has the fence format changed?"
    )


@pytest.mark.parametrize(
    "doc,index,line,code",
    BLOCKS,
    ids=[f"{d}:block{i}" for d, i, _, _ in BLOCKS],
)
def test_doc_example_runs(doc, index, line, code, doc_namespace):
    try:
        exec(compile(code, f"{doc}:{line}", "exec"), doc_namespace)
    except Exception as exc:  # noqa: BLE001 - we re-raise with context
        raise AssertionError(
            f"\n{doc} (block {index}, starting line {line}) failed:\n"
            f"  {type(exc).__name__}: {exc}\n\n"
            f"--- code ---\n{code}\n"
            f"------------\n"
            f"If this block cannot run in a test, mark it with an HTML "
            f"comment on the line before the fence:\n"
            f"    <!-- skip-test: reason -->"
        ) from exc
