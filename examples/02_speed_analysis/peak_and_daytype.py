"""Finding the peak windows, and comparing weekdays against weekends.

    python examples/02_speed_analysis/peak_and_daytype.py

Peak hours are discovered from the data rather than assumed, and weekday and
weekend traffic are treated as two populations rather than averaged into one.
Pooling them hides both: a Tuesday 09:00 and a Saturday 09:00 describe
different worlds.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _common import prepare, rule  # noqa: E402

import redlight as rl  # noqa: E402


def main() -> None:
    net, pts, derived = prepare()
    clean = rl.filter_by_speed(derived["edge_observations"], max_speed=80,
                               unit="mph", mad_outliers=True, per_edge=True)

    # -------------------------------------------------------------- peaks
    rule("Peak and off-peak, discovered from the data")
    hourly = rl.aggregate_speeds(clean, block_hours=1, statistic="median",
                                 output_unit="mph", min_samples=3)
    peaks = rl.peak_analysis(hourly, statistic="median", n_peak=3, n_offpeak=3)
    print("slowest hours (peak):")
    for r in peaks["peak"]:
        print(f"  {r['block_label']}  {r['median_speed']:5.1f} mph  n={r['n']}")
    print("fastest hours (off-peak):")
    for r in peaks["off_peak"]:
        print(f"  {r['block_label']}  {r['median_speed']:5.1f} mph  n={r['n']}")

    # ------------------------------------------------------- classify hours
    rule("A different question: contiguous windows, as reusable hour sets")
    hours = rl.classify_hours(clean, statistic="median", n_peak=3, n_offpeak=3)
    print(f"peak hours    : {hours['peak_hours']}")
    print(f"off-peak hours: {hours['offpeak_hours']}")
    print("\nExpect these NOT to match the three hours listed above, and")
    print("do not read them as 'the same answer in another shape'. The two")
    print("functions rank differently under the same n_peak= argument:")
    print("  peak_analysis  returns the n individually slowest bins, which")
    print("                 may be scattered across the day;")
    print("  classify_hours returns the slowest CONTIGUOUS n-hour window.")
    print("A regime has to be a period you can route on, which is why")
    print("assign_segment_speeds wants the window rather than the ranking.")
    print("Being contiguous, a window can also span an hour that carries no")
    print("observations at all -- check n before quoting any single hour.")

    # ---------------------------------------------------------- day types
    rule("Weekday versus weekend")
    report = rl.day_type_report(clean, statistic="median", output_unit="mph")
    for label, grp in report["groups"].items():
        print(f"  {label:<8} n={grp['n']:>5,}  "
              f"median {grp['overall_speed']:.1f} mph")
    delta = report["overall"].get("delta_pct")
    if delta is not None:
        print(f"\n  weekend vs weekday: {delta:+.1f}% "
              f"(positive = weekends faster)")
    print("\nIf one day type has few observations the comparison is partial --")
    print("check the n values before quoting the delta.")

    # ------------------------------------------------- write onto the graph
    rule("Writing per-regime speeds onto the network")
    seg = rl.assign_segment_speeds(net, clean, statistic="median",
                                   n_peak=3, n_offpeak=3)
    cov = seg["coverage"]
    print(f"edges with an observed speed: overall {cov['overall']}, "
          f"peak {cov['peak']}, off-peak {cov['offpeak']} "
          f"of {net.number_of_edges()}")
    print("Edges never traversed keep no observed speed -- they are not")
    print("silently assigned the network average.")


if __name__ == "__main__":
    main()
