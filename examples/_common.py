"""Shared setup for the examples.

``01_basics/load_match_derive.py`` spells the pipeline out longhand and is the
one to read first. Every later example imports :func:`prepare` from here
instead, so it can stay focused on its own topic rather than repeating four
steps of boilerplate.
"""
from __future__ import annotations

import os
import sys

import redlight as rl

EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(EXAMPLES_DIR, "sample_data")


def add_examples_to_path() -> None:
    """Let a script in a topic folder ``import _common``."""
    if EXAMPLES_DIR not in sys.path:
        sys.path.insert(0, EXAMPLES_DIR)


def require_sample_data() -> None:
    """Fail with an actionable message rather than a FileNotFoundError."""
    net = os.path.join(DATA, "network.geojson")
    if not os.path.exists(net):
        raise SystemExit(
            "Sample data not found. Generate it first:\n"
            "    python examples/00_setup/generate_sample_data.py")


def load_network():
    require_sample_data()
    return rl.Network.from_geojson(os.path.join(DATA, "network.geojson"))


def load_gps():
    require_sample_data()
    return rl.load_points(
        os.path.join(DATA, "points.csv"),
        id_col="device_id", time_col="timestamp",
        lon_col="longitude", lat_col="latitude",
        tz="America/New_York",
    )


def prepare(*, min_baseline_m: float = 150.0, quiet: bool = False):
    """Load, match and derive speeds. Returns ``(net, points, derived)``.

    ``derived`` is the dict from :func:`redlight.derive_speeds`, with keys
    ``"intervals"`` (one row per independent measurement) and
    ``"edge_observations"`` (one row per interval-and-edge-crossed).
    """
    net = load_network()
    pts = load_gps()
    matched = rl.HMMMatcher(net, max_dist=50).match(pts)
    derived = rl.derive_speeds(net, matched, pts,
                               pos_accuracy_col="accuracy_m",
                               min_baseline_m=min_baseline_m)
    if not quiet:
        print(f"[setup] {len(pts):,} fixes -> "
              f"{len(derived['intervals']):,} speed intervals "
              f"across {net.number_of_edges()} edges")
    return net, pts, derived


def rule(title: str) -> None:
    """A section heading, so example output is readable when piped."""
    print(f"\n{title}\n{'-' * len(title)}")
