"""Cleaning derived speeds, then aggregating them by time of day.

    python examples/02_speed_analysis/clean_and_aggregate.py

The important lesson here is which filter to reach for. ``filter_by_speed``
takes a ``min_speed``, and using it on a trafficability study is almost always
a mistake: a vehicle crawling through a chokepoint and a mover that is not a
vehicle at all look identical at the level of one observation, so a speed floor
deletes the congestion the study exists to measure. Filter the top end for
GPS-jump artefacts, and leave the bottom end alone.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _common import prepare, rule  # noqa: E402

import roadtraffic as rt  # noqa: E402


def main() -> None:
    net, pts, derived = prepare()
    obs = derived["edge_observations"]

    # ------------------------------------------------------------ cleaning
    rule("Cleaning")
    before = len(obs)
    # max_speed only. mad_outliers with per_edge=True removes robust outliers
    # within each edge, so a fast arterial is not judged against a slow street.
    clean = rt.filter_by_speed(obs, max_speed=80, unit="mph",
                               mad_outliers=True, per_edge=True)
    print(f"{before:,} -> {len(clean):,} observations "
          f"({before - len(clean):,} removed as >80 mph or per-edge outliers)")

    floored = rt.filter_by_speed(obs, min_speed=10, max_speed=80, unit="mph")
    kept = rt.from_mps(clean["speed_mps"], "mph").median()
    lost = rt.from_mps(floored["speed_mps"], "mph").median()
    print("\nWhat a 10 mph floor would have done instead:")
    print(f"  observations {len(obs):,} -> {len(floored):,}")
    print(f"  median speed {kept:.1f} mph -> {lost:.1f} mph "
          f"({lost - kept:+.1f} mph)")
    print("  That shift is not a better measurement. It is the congestion")
    print("  being deleted. Do not filter the bottom end.")

    # --------------------------------------------------------- aggregation
    rule("Speed by hour of day")
    hourly = rt.aggregate_speeds(clean, block_hours=1, statistic="both",
                                 output_unit="mph", min_samples=3)
    print(hourly[["block_label", "n", "mean_speed", "sem_speed",
                  "median_speed"]].round(2).to_string(index=False))
    print("\nNetwork-wide aggregates deduplicate on interval_id automatically,")
    print("so an interval crossing five edges still counts once.")

    # ------------------------------------------------------------- blocks
    rule("Coarser blocks trade resolution for tighter intervals")
    for bh in (1, 4, 8):
        agg = rt.aggregate_speeds(clean, block_hours=bh, statistic="mean",
                                  output_unit="mph")
        width = (agg["ci95_high"] - agg["ci95_low"]).mean()
        print(f"  {bh:>2}-hour blocks: {len(agg):>2} periods, "
              f"mean 95% CI width {width:.2f} mph")


if __name__ == "__main__":
    main()
