"""Measured speed as a fraction of the posted limit.

    python examples/04_congestion/congestion_vs_limits.py

A raw speed cannot tell you whether a road is performing badly: 25 mph is free
flow on a residential street and gridlock on an arterial. The ratio of measured
speed to posted limit is the comparison raw speeds cannot make, and it needs a
network whose edges carry a ``maxspeed`` tag -- which OSM extracts usually do.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _common import prepare, rule  # noqa: E402

import roadtraffic as rt  # noqa: E402


def main() -> None:
    net, pts, derived = prepare()
    clean = rt.filter_by_speed(derived["edge_observations"], max_speed=80,
                               unit="mph", mad_outliers=True, per_edge=True)

    # ----------------------------------------------------- posted limits
    rule("Posted limits come off the network's maxspeed tags")
    tagged = sum(1 for e in net.edge_ids
                 if net.edge_data(int(e)).get("maxspeed_mps"))
    print(f"{tagged}/{net.number_of_edges()} edges carry a usable maxspeed.")
    print("OSM's maxspeed is parsed at load time: a bare number means km/h,")
    print("'55 mph' is honoured, and unparseable values ('none', 'signals',")
    print("'RU:urban') are left unset rather than guessed at.")

    # -------------------------------------------------------- the report
    rule("Congestion report")
    report = rt.congestion_report(net, clean, statistic="median",
                                  output_unit="mph")
    summary = report["summary"]
    print(f"  rows                {summary['n_rows']:,}")
    print(f"  edges observed      {summary['n_edges_observed']}")
    print(f"  edges rated         {summary['n_edges_rated']}")
    print(f"  median ratio        {summary['median_ratio']:.2f}")
    print(f"  mean ratio          {summary['mean_ratio']:.2f}")
    print("\n'observed' counts edges with a measured speed; 'rated' counts the")
    print("subset that also had a usable posted limit to compare against.")
    print("\nA ratio of 1.00 means traffic moved at the posted limit;")
    print("0.45 means it crawled at 45% of it. The ratio is NOT clipped at")
    print("1.0 -- roads where traffic runs above the limit are real and")
    print("clipping them would hide a genuine finding.")

    # --------------------------------------------------------- worst roads
    rule("Worst-performing roads")
    edges = report["edges"]
    rated = edges[edges["ratio"].notna()].sort_values("ratio")
    seen, shown = set(), 0
    for r in rated.itertuples():
        d = net.edge_data(int(r.edge_id))
        name = d.get("name") or d.get("highway") or f"edge {r.edge_id}"
        if name in seen:          # a two-way street is two directed edges;
            continue              # show each physical road once
        seen.add(name)
        print(f"  {name:<16} {r.observed_speed:5.1f} / {r.speed_limit:5.1f} mph"
              f"   ratio {r.ratio:.2f}   n={int(r.n)}")
        shown += 1
        if shown >= 6:
            break


if __name__ == "__main__":
    main()
