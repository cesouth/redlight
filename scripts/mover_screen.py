"""Separate vehicles from pedestrians (and other slow movers) in mixed GPS data.

Why this exists
---------------
A trafficability study reads congestion off *slow* observations, so the one thing
you must not do is drop observations for being slow -- that deletes the finding.
But if the feed also carries people on foot, their fixes land on the same roads
and drag every score down.

The resolution is that **mode is a property of the mover, not of the fix**. A
pedestrian is slow for their entire track; a congested vehicle is slow *here* and
free-flowing somewhere else in the same trip. So this classifies whole
trajectories on a high percentile of their derived speed (default 85th) and drops
the ones that never move like a vehicle. Every surviving mover keeps all of its
observations, congestion included.

Usage
-----
1. Diagnose -- look at the distribution and let it suggest a threshold:

       python scripts/mover_screen.py --network net.geojson --points gps.csv \
           --id-col device_id

2. Apply -- write a filtered points file, then run the report on that:

       python scripts/mover_screen.py --network net.geojson --points gps.csv \
           --id-col device_id --threshold 6 --out-points vehicles.csv
       python scripts/customer_report.py --network net.geojson \
           --points vehicles.csv --id-col device_id --out report.pdf

   Or let the screen pick the threshold itself with ``--threshold auto``.

``--out-movers`` writes one row per mover with a ``mode`` column: ``vehicle``,
``pedestrian``, or ``unknown``. ``unknown`` means there was too little data to
judge the mover (see ``--min-intervals``), never that its speed was
ambiguous -- a congested vehicle is classified ``vehicle``, not ``unknown``.

Threshold choice
----------------
Put it in the *valley* between the two humps the histogram shows, not at a round
number. Too low leaves walkers in and understates speeds; too high starts
deleting genuinely gridlocked vehicles, which biases congestion upward -- the
same failure direction as --require-quality. On a synthetic mixed set with known
ground truth, a threshold at the valley (6 mph) recovered per-edge speeds to
within 0.6 mph of the vehicles-only truth, while 8 mph overshot peak speeds by
1.8 mph by deleting 23 of 231 gridlocked vehicles.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

import roadtraffic as rt
from roadtraffic.units import SpeedUnit, from_mps


def histogram(x: np.ndarray, unit_label: str, focus_hi: float,
              width: int = 46) -> None:
    """Text histogram -- the point is to SEE whether there are two humps.

    Binned over the decision region rather than the full range: the valley to
    look for is a few units wide near walking pace, and bins sized to span
    motorway speeds are far too coarse to show it. Everything above the focus
    range goes in one overflow row, since its only job is to show that the fast
    hump exists.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    hi = max(min(float(np.percentile(x, 98)), focus_hi), 1.0)
    edges = np.linspace(0, hi, 25)
    counts, _ = np.histogram(x, bins=edges)
    over = int((x > hi).sum())
    scale = width / max(max(counts.max(), over), 1)
    print(f"\n  per-mover {unit_label} speed distribution ({len(x)} movers):")
    for i, c in enumerate(counts):
        print(f"  {edges[i]:6.1f}-{edges[i + 1]:<6.1f} |"
              f"{'#' * int(round(c * scale))} {c}")
    if over:
        print(f"  {hi:6.1f}+        |{'#' * int(round(over * scale))} {over}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Classify GPS movers as vehicles vs pedestrians and "
                    "optionally write a vehicles-only points file.")
    p.add_argument("--network", required=True, help="road network GeoJSON")
    p.add_argument("--points", required=True, help="GPS points CSV/TSV/GeoJSON")
    p.add_argument("--id-col", default=None, help="mover/trajectory id column")
    p.add_argument("--time-col", default=None)
    p.add_argument("--lon-col", default=None)
    p.add_argument("--lat-col", default=None)
    p.add_argument("--tz", default=None, help="IANA timezone, e.g. America/New_York")
    p.add_argument("--accuracy-col", default="accuracy",
                   help="per-point horizontal accuracy in METRES")
    p.add_argument("--unit", default="mph", choices=["mph", "kph", "mps"])
    p.add_argument("--max-dist", type=float, default=50.0,
                   help="max snap distance, metres (default 50)")
    p.add_argument("--min-baseline", type=float, default=150.0,
                   help="merge hops to this on-road distance before deriving a "
                        "speed; raises SNR so slow movers are measured, not "
                        "merely noisy (default 150 m)")
    p.add_argument("--percentile", type=float, default=85.0,
                   help="per-mover speed percentile to classify on (default 85). "
                        "High enough to catch a vehicle's free-flowing stretch, "
                        "low enough to ignore a single GPS jump.")
    p.add_argument("--threshold", type=str, default=None,
                   help="keep movers whose percentile speed is at or above this, "
                        "in --unit, or 'auto' to use the suggested threshold "
                        "(and fail loudly if none can be found). Omit to "
                        "diagnose only.")
    p.add_argument("--min-intervals", type=int, default=3,
                   help="movers with fewer derived intervals are too short to "
                        "classify; see --keep-unclassified (default 3)")
    p.add_argument("--keep-unclassified", action="store_true",
                   help="keep movers with too few intervals to judge, rather "
                        "than dropping them")
    p.add_argument("--out-points", default=None,
                   help="write the surviving movers' original rows here")
    p.add_argument("--out-movers", default=None,
                   help="write the per-mover feature table here, including "
                        "the classification in a 'mode' column (vehicle / "
                        "pedestrian / unknown)")
    args = p.parse_args(argv)

    threshold: float | str | None = args.threshold
    if threshold is not None and threshold != "auto":
        try:
            threshold = float(threshold)
        except ValueError as exc:
            raise SystemExit(
                f"--threshold must be a number or 'auto', got "
                f"{args.threshold!r}.") from exc

    unit = SpeedUnit.parse(args.unit)
    net = rt.Network.from_geojson(args.network)
    pts = rt.load_points(args.points, tz=args.tz, id_col=args.id_col,
                         time_col=args.time_col, lon_col=args.lon_col,
                         lat_col=args.lat_col)
    if not pts.has_traj:
        raise SystemExit(
            "Mode separation is per-mover, so it needs a trajectory id. "
            "Pass --id-col.")

    print("[1/3] matching (HMM/Viterbi)", file=sys.stderr)
    matched = rt.HMMMatcher(net, max_dist=args.max_dist).match(pts)

    print("[2/3] deriving speeds from on-road displacement", file=sys.stderr)
    acc = args.accuracy_col if args.accuracy_col in pts.df.columns else None
    if args.accuracy_col and acc is None:
        print(f"      accuracy column {args.accuracy_col!r} not found; "
              f"using the default sigma", file=sys.stderr)
    derived = rt.derive_speeds(net, matched, pts, pos_accuracy_col=acc,
                               min_baseline_m=args.min_baseline)
    iv = derived["intervals"]
    if not len(iv):
        raise SystemExit("No speed intervals could be derived; check --id-col.")

    print("[3/3] reducing to per-mover features", file=sys.stderr)
    feat = rt.mover_features(iv, percentile=args.percentile, unit=unit)
    pct_col = f"speed_p{args.percentile:g}_{unit.value}"

    # 6 m/s (~13 mph) is comfortably above any walking or jogging pace and
    # below urban free flow, so the split -- if there is one -- lies under it.
    walk_hi = float(from_mps(6.0, unit))
    histogram(feat[pct_col].to_numpy(float), f"p{args.percentile:g}",
              focus_hi=2.5 * walk_hi)
    sugg = rt.suggest_mode_threshold(feat[pct_col], unit=unit)
    print()
    if sugg is None:
        print("  No walking-speed population found. Either the feed is all "
              "vehicles (nothing to exclude), or the two populations overlap "
              "too much to split on speed alone -- check the histogram for a "
              "hump at walking pace before overriding with --threshold.")
    else:
        print(f"  Suggested threshold: {sugg:.1f} {unit.value} "
              f"(deepest density valley below {walk_hi:.0f} {unit.value})")
    print("  Confirm it against the histogram above before trusting it: the "
          "threshold belongs in the gap BETWEEN two humps, and if there is only "
          "one hump there is no gap to find.")

    if threshold is None:
        print("\n  Diagnostic only -- pass --threshold to apply.", file=sys.stderr)
        if args.out_movers:
            feat.to_csv(args.out_movers)
            print(f"  wrote {args.out_movers}", file=sys.stderr)
        return 0

    movers = rt.classify_movers(iv, threshold=threshold,
                                percentile=args.percentile,
                                min_intervals=args.min_intervals, unit=unit)
    keep = ("vehicle", "unknown") if args.keep_unclassified else ("vehicle",)
    kept_ids = set(movers.index[movers["mode"].isin(keep)])

    thresh_label = "auto" if threshold == "auto" else f"{threshold:g}"
    n_drop = len(movers) - len(kept_ids)
    print(f"\n  threshold {thresh_label} {unit.value}: keeping {len(kept_ids)} of "
          f"{len(movers)} movers, dropping {n_drop} "
          f"({100.0 * n_drop / max(len(movers), 1):.0f}%)")
    short = movers["n_intervals"] < args.min_intervals
    if short.any():
        verb = "kept" if args.keep_unclassified else "dropped"
        print(f"  {int(short.sum())} movers had fewer than "
              f"{args.min_intervals} intervals and were {verb} "
              f"(--keep-unclassified flips this)")

    dropped_frac_fixes = float(pts.df["traj_id"].isin(kept_ids).mean())
    print(f"  that retains {100.0 * dropped_frac_fixes:.0f}% of GPS fixes")
    if not kept_ids:
        raise SystemExit("The threshold removed every mover; lower it.")

    if args.out_movers:
        movers.to_csv(args.out_movers)
        print(f"  wrote {args.out_movers}")

    if args.out_points:
        src = pd.read_csv(args.points) if not args.points.endswith(
            (".geojson", ".json")) else None
        if src is None:
            raise SystemExit(
                "--out-points currently writes CSV only; the input is GeoJSON. "
                "Use --out-movers and filter the source yourself.")
        id_col = args.id_col
        if id_col is None or id_col not in src.columns:
            raise SystemExit(
                "--out-points needs the source id column; pass --id-col "
                "with a name present in the input file.")
        out = src[src[id_col].isin(kept_ids)]
        out.to_csv(args.out_points, index=False)
        print(f"  wrote {args.out_points}  ({len(out):,} of {len(src):,} rows)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
