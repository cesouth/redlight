"""Example 2 — HMM map matching and N-hour block aggregation.

Compares the nearest-edge and HMM matchers, then aggregates into 6-hour blocks.

    python examples/02_hmm_and_blocks.py
"""
import os

import roadtraffic as rt

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "sample_data")


def main():
    net = rt.Network.from_geojson(os.path.join(DATA, "network.geojson"))
    pts = rt.load_points(os.path.join(DATA, "points.csv"),
                         speed_unit="mph", id_col="vehicle_id")

    near = rt.NearestMatcher(net, max_dist=60).match(pts)
    hmm = rt.HMMMatcher(net, sigma_z=6, beta=30, max_dist=60).match(pts)

    agree = (near.sort_values("point_id")["edge_id"].values ==
             hmm.sort_values("point_id")["edge_id"].values).mean()
    print(f"Nearest vs HMM edge agreement: {agree:.1%}")

    clean = rt.filter_by_speed(hmm, min_speed=2, max_speed=80, unit="mph",
                               mad_outliers=True, per_edge=True)

    for bh in (2, 4, 6):
        agg = rt.aggregate_speeds(clean, block_hours=bh, statistic="mean",
                                  output_unit="mph")
        print(f"\n{bh}-hour blocks (mean mph, 95% CI):")
        print(agg[["block_label", "n", "mean_speed",
                   "ci95_low", "ci95_high"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
