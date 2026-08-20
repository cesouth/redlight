"""The core pipeline: load GPS -> match to roads -> derive speed from positions.

    python examples/00_setup/generate_sample_data.py   # once
    python examples/01_basics/load_match_derive.py

Every other example starts here, so this one is worth reading closely.

The input carries position, time, a mover id and a per-fix accuracy -- but no
speed. Speed is reconstructed from how far each mover travelled *along the
road* between consecutive fixes, which is more trustworthy than a receiver's
instantaneous reading and is the only option when there is no reading at all.
"""
import os

import redlight as rl

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "sample_data")

# This example keeps its own explicit paths rather than importing _common, so
# the core pipeline reads end to end without indirection. It still owes the
# reader the same courtesy every other example gives: sample_data/ is
# gitignored and generated, so say which command makes it rather than dying on
# a FileNotFoundError three lines later.
if not os.path.exists(os.path.join(DATA, "network.geojson")):
    raise SystemExit(
        "Sample data not found. Generate it first:\n"
        "    python examples/00_setup/generate_sample_data.py")


def main() -> None:
    # ---------------------------------------------------------------- network
    net = rl.Network.from_geojson(os.path.join(DATA, "network.geojson"))
    print(f"Network: {net.number_of_nodes()} nodes, {net.number_of_edges()} "
          f"directed edges, metric CRS EPSG:{net.crs_metric.to_epsg()}")

    # ----------------------------------------------------------------- points
    # tz= converts the timestamps to local clock time. Getting this wrong
    # shifts every hour-of-day statistic by the UTC offset, which quietly
    # relabels the rush hours.
    pts = rl.load_points(
        os.path.join(DATA, "points.csv"),
        id_col="device_id", time_col="timestamp",
        lon_col="longitude", lat_col="latitude",
        tz="America/New_York",
    )
    print(f"Loaded {len(pts)} fixes from {pts.df['traj_id'].nunique()} movers "
          f"(has_traj={pts.has_traj})")
    print(f"  speed column present: {'speed_mps' in pts.df.columns}")
    print(f"  extra source columns kept: "
          f"{[c for c in pts.df.columns if c in ('accuracy_m', 'mode')]}")

    # ---------------------------------------------------------------- matching
    # HMMMatcher decodes the whole trajectory with Viterbi, so a fix that is
    # closer to the wrong road still lands on the road the mover was actually
    # travelling. NearestMatcher snaps each fix independently -- faster, but it
    # has no way to know that.
    matched = rl.HMMMatcher(net, max_dist=50).match(pts)
    n_ok = int((matched["edge_id"] != -1).sum())
    print(f"Matched {n_ok}/{len(matched)} fixes "
          f"({100 * n_ok / len(matched):.1f}%)")

    # ------------------------------------------------------------ derive speed
    # pos_accuracy_col makes the error model use each fix's own reported
    # accuracy instead of one assumed sigma for every point.
    # min_baseline_m merges consecutive hops until they cover that much road,
    # which lifts short displacements clear of GPS noise -- worth setting
    # whenever fixes are dense relative to the noise.
    derived = rl.derive_speeds(
        net, matched, pts,
        pos_accuracy_col="accuracy_m",
        min_baseline_m=150.0,
    )
    intervals = derived["intervals"]
    edge_obs = derived["edge_observations"]

    print(f"\nDerived {len(intervals):,} speed intervals "
          f"-> {len(edge_obs):,} edge observations")
    print("  'intervals' is one row per independent measurement -- use it for")
    print("  network-wide statistics. 'edge_observations' repeats an interval")
    print("  once per edge it crossed -- use it for per-edge statistics.")

    mph = rl.from_mps(intervals["speed_mps"], "mph")
    print(f"\nSpeed: median {mph.median():.1f} mph, "
          f"{mph.quantile(0.05):.1f}-{mph.quantile(0.95):.1f} mph (5-95%)")

    # Every interval carries its own uncertainty and a quality flag.
    good = int(intervals["quality"].sum())
    print(f"Quality: {good:,}/{len(intervals):,} intervals passed the screen "
          f"({100 * good / len(intervals):.0f}%)")
    print(f"  median 1-sigma speed uncertainty: "
          f"{rl.from_mps(intervals['speed_sigma_mps'], 'mph').median():.2f} mph")
    print("\nNote: quality=False rows are RETURNED, not dropped. Excluding them")
    print("      biases speeds upward, because slow traffic covers the least")
    print("      ground per fix and fails the screen most often.")


if __name__ == "__main__":
    main()
