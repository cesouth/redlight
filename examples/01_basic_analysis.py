"""Example 1 — basic trafficability analysis.

Load -> match (nearest) -> clean -> aggregate hourly -> peak/off-peak.

    python examples/00_generate_sample_data.py   # run once first
    python examples/01_basic_analysis.py
"""
import os

import roadtraffic as rt

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "sample_data")


def main():
    net = rt.Network.from_geojson(os.path.join(DATA, "network.geojson"))
    print(f"Network: {net.number_of_nodes()} nodes, "
          f"{net.number_of_edges()} edges, UTM EPSG:{net.crs_metric.to_epsg()}")

    pts = rt.load_points(os.path.join(DATA, "points.csv"),
                         speed_unit="mph", id_col="vehicle_id")
    print(f"Loaded {len(pts)} GPS points (has_traj={pts.has_traj})")

    matched = rt.NearestMatcher(net, max_dist=60).match(pts)
    n_ok = (matched["edge_id"] != -1).sum()
    print(f"Matched {n_ok}/{len(matched)} points to edges")

    clean = rt.filter_by_speed(matched, min_speed=2, max_speed=80, unit="mph",
                               mad_outliers=True, per_edge=True)
    print(f"After cleaning: {len(clean)} observations")

    hourly = rt.aggregate_speeds(clean, block_hours=1, statistic="both",
                                 output_unit="mph")
    print("\nHourly speeds (mph):")
    print(hourly[["block_label", "n", "mean_speed", "sem_speed",
                  "median_speed", "iqr_speed"]].round(2).to_string(index=False))

    peaks = rt.peak_analysis(hourly, statistic="median", n_peak=3, n_offpeak=3)
    print("\nPeak hours (slowest / busiest):")
    for r in peaks["peak"]:
        print(f"  {r['block_label']}  median={r['median_speed']:.1f} mph  n={r['n']}")
    print("Off-peak hours (fastest):")
    for r in peaks["off_peak"]:
        print(f"  {r['block_label']}  median={r['median_speed']:.1f} mph  n={r['n']}")


if __name__ == "__main__":
    main()
