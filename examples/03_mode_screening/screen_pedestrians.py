"""Removing non-vehicle movers from a mixed GPS feed, without deleting congestion.

    python examples/03_mode_screening/screen_pedestrians.py

The problem: a feed that also carries people on foot drags every road's score
down. The tempting fix -- a minimum-speed filter -- is the one that breaks the
study, because a pedestrian at 3 mph and a vehicle crawling through a
chokepoint at 3 mph are indistinguishable in a single observation.

The resolution is that **mode is a property of the mover, not the fix**. A
pedestrian is slow for their whole track; a congested vehicle is slow here and
free-flowing elsewhere in the same trip. So classify whole trajectories on a
high percentile of their speed, and apply each verdict to *all* of that mover's
observations. A mover kept as a vehicle keeps its slow rows.

The sample data carries a ``mode`` column recording the planted truth, so this
example can score its own accuracy. Real feeds will not have that.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _common import prepare, rule  # noqa: E402

import roadtraffic as rt  # noqa: E402


def main() -> None:
    net, pts, derived = prepare()
    intervals, obs = derived["intervals"], derived["edge_observations"]

    # ------------------------------------------------------------- diagnose
    rule("1. Look at the distribution before choosing anything")
    feat = rt.mover_features(intervals, unit="mph")
    pct = "speed_p85_mph"
    print(f"{len(feat)} movers, each summarised by its 85th-percentile speed.")
    print("The 85th percentile is high enough to catch a vehicle's")
    print("free-flowing stretch, and low enough to ignore one GPS jump.")

    lo, hi = feat[pct].min(), feat[pct].max()
    edges = [0, 4, 6, 8, 12, 20, 30, 100]
    print(f"\nper-mover p85 speed ({lo:.1f}-{hi:.1f} mph):")
    for a, b in zip(edges, edges[1:]):
        n = int(((feat[pct] >= a) & (feat[pct] < b)).sum())
        print(f"  {a:>3}-{b:<3} mph |{'#' * n} {n}")

    # ------------------------------------------------------------ threshold
    rule("2. Let the data suggest a cut, then sanity-check it")
    suggested = rt.suggest_mode_threshold(feat[pct], unit="mph")
    if suggested is None:
        print("No walking-speed population found -- nothing to screen.")
        print("That is the honest answer for an all-vehicle feed, and the")
        print("function returns None rather than inventing a threshold.")
        return
    print(f"Suggested threshold: {suggested:.1f} mph")
    print("It belongs in the gap BETWEEN the two humps above. If there is")
    print("only one hump there is no gap, and the function returns None.")

    # -------------------------------------------------------------- classify
    rule("3. Classify movers, then filter observations by the verdict")
    movers = rt.classify_movers(intervals, threshold=suggested, unit="mph")
    print(movers["mode"].value_counts().to_string())

    kept = rt.filter_by_mode(obs, movers)                  # vehicles only
    print(f"\nobservations {len(obs):,} -> {len(kept):,}")

    # ----------------------------------------------------------- did it work
    rule("4. Score it against the planted truth")
    truth = pts.df.groupby("traj_id")["mode"].first()
    tab = movers.join(truth.rename("truth"))
    correct = int((((tab["mode"] == "vehicle") & (tab["truth"] == "vehicle")) |
                   ((tab["mode"] == "pedestrian") &
                    (tab["truth"] == "pedestrian"))).sum())
    print(f"{correct}/{len(tab)} movers classified correctly")
    print()
    print(tab.groupby(["truth", "mode"]).size().rename("movers").to_string())

    # ------------------------------------------------------------ the effect
    rule("5. What it did to the answer")
    all_mph = rt.from_mps(obs.drop_duplicates("interval_id")["speed_mps"], "mph")
    veh_mph = rt.from_mps(kept.drop_duplicates("interval_id")["speed_mps"], "mph")
    print(f"median speed, unscreened: {all_mph.median():5.1f} mph")
    print(f"median speed, screened  : {veh_mph.median():5.1f} mph "
          f"({veh_mph.median() - all_mph.median():+.1f})")

    rule("The limitation, stated plainly")
    print("A vehicle gridlocked for its ENTIRE track never shows a fast")
    print("stretch, so it is excluded along with the walkers. That biases")
    print("speeds UPWARD -- the same direction as require_quality. Run the")
    print("study both ways, compare peak speeds, and report the gap as")
    print("uncertainty. If your feed carries a device type or fleet id, use")
    print("that instead: real metadata beats inference every time.")


if __name__ == "__main__":
    main()
