"""Proof that the core package never imports pyproj.

The point of the numpy geodesy is that a default install has no PROJ in it.
That property is invisible in a dev environment where pyproj happens to be
installed, so it needs a test that fails loudly when someone reintroduces a
module-level import.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap

CORE_WORKLOAD = textwrap.dedent("""
    import sys

    class Blocker:
        # find_spec, not the legacy find_module, which was removed in 3.12.
        def find_spec(self, name, path=None, target=None):
            if name == "pyproj" or name.startswith("pyproj."):
                raise ImportError("pyproj is blocked for this test")
            return None

    sys.meta_path.insert(0, Blocker())

    import json, os, tempfile
    import redlight as rl

    tmp = tempfile.mkdtemp()
    net_path = os.path.join(tmp, "net.json")
    with open(net_path, "w") as fh:
        json.dump({"type": "FeatureCollection", "features": [{
            "type": "Feature",
            "properties": {"highway": "residential"},
            "geometry": {"type": "LineString",
                         "coordinates": [[0.0, 0.0], [0.01, 0.0]]},
        }]}, fh)

    net = rl.Network.from_geojson(net_path)
    assert net.crs_metric.to_epsg() == 32631, net.crs_metric.to_epsg()
    px, py = net.project_points([0.005], [0.0])
    assert abs(float(px[0])) > 0

    pts_path = os.path.join(tmp, "pts.csv")
    with open(pts_path, "w") as fh:
        fh.write("id,lon,lat,time\\n")
        for i in range(6):
            fh.write(f"a,{i * 0.0015:.6f},0.00002,2026-06-01T08:00:{i * 10:02d}\\n")

    pts = rl.load_points(pts_path, id_col="id")
    matched = rl.NearestMatcher(net, max_dist=60).match(pts)
    speeds = rl.derive_speeds(net, matched, pts)
    assert len(speeds["intervals"]) > 0, speeds["intervals"]

    stats = rl.network_stats(net)
    assert stats["n_edges"] == 2, stats["n_edges"]

    assert "pyproj" not in sys.modules, "pyproj was imported by the core path"
    print("OK")
""")


def test_core_workload_runs_with_pyproj_blocked():
    """Load, match, derive speeds, and compute stats with pyproj unimportable.

    Runs in a subprocess because the import blocker has to be installed before
    redlight is first imported, and pytest has already imported it here.
    """
    result = subprocess.run(
        [sys.executable, "-c", CORE_WORKLOAD],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        f"core workload failed with pyproj blocked:\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    assert "OK" in result.stdout
