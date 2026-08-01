"""What the road network is like, independent of the GPS survey.

    python examples/05_network/structure_and_chokepoints.py

Two questions this answers that speed data cannot:

* Is the extract usable at all? A network split into disconnected fragments,
  or trapped by one-way tags, produces routes that silently fail.
* Which roads are structurally load-bearing? Betweenness weighted by measured
  travel time finds the segments a disproportionate share of fastest routes
  must cross -- chokepoints, not merely roads that look central on a map.
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

    # ----------------------------------------------------------- structure
    rule("Network structure")
    # The sample grid spans about 2.67 km N-S by 3.12 km E-W. Densities are
    # only as good as the area you supply, so measure your study area rather
    # than eyeballing it -- the library cannot check this number for you.
    stats = rt.network_stats(net, area_km2=8.3)
    print(f"  nodes                  {stats['n_nodes']:,}")
    print(f"  directed edges         {stats['n_edges']:,}")
    print(f"  physical roads         {stats['n_physical_roads']:,}")
    print(f"  intersections          {stats['n_intersections']:,}")
    print(f"  dead ends              {stats['n_dead_ends']:,}")
    print(f"  streets per node (avg) {stats['streets_per_node_avg']:.2f}")
    print(f"  circuity (avg)         {stats['circuity_avg']:.3f}")
    print(f"  road length m/km2      {stats['edge_density_km2']:,.0f}")
    print(f"  intersections /km2     {stats['intersection_density_km2']:.1f}")
    print("\ncircuity is on-road distance divided by straight-line distance:")
    print("1.0 is a perfect grid, higher means detours. 'physical roads'")
    print("counts a two-way street once, where 'directed edges' counts it")
    print("twice. The two densities need area_km2 to be passed in, and are")
    print("None without it -- they are not guessed from the bounding box.")

    # -------------------------------------------------------- connectivity
    rule("Connectivity")
    conn = rt.connectivity_report(net)
    print(f"  strongly connected     {conn['is_strongly_connected']}")
    print(f"  weakly connected       {conn['is_weakly_connected']}")
    print(f"  strong components      {conn['n_strongly_connected_components']}")
    print(f"  largest component      "
          f"{100 * conn['largest_component_fraction_nodes']:.1f}% of nodes")
    print(f"  stranded nodes         {len(conn['stranded_nodes'])}")
    print("\nWeakly but not strongly connected means a one-way trap: the")
    print("geometry joins up but the directed graph does not, so routes into")
    print("a district exist and routes out do not. That is a different fault")
    print("from a clipped extract, and needs a different fix -- but both look")
    print("identical from a failed route.")

    # -------------------------------------------------------- chokepoints
    rule("Chokepoints, weighted by measured travel time")
    rt.assign_segment_speeds(net, clean, statistic="median")
    try:
        bc = rt.edge_betweenness_centrality(net, weight="travel_time_s")
    except ValueError as exc:
        print(f"skipped: {exc}")
        return

    seen, shown = set(), 0
    for edge_id, score in sorted(bc.items(), key=lambda kv: -kv[1]):
        d = net.edge_data(int(edge_id))
        name = d.get("name") or d.get("highway") or f"edge {edge_id}"
        if name in seen:      # collapse the two directions of one road
            continue
        seen.add(name)
        print(f"  {name:<16} betweenness {score:.3f}")
        shown += 1
        if shown >= 6:
            break
    print("\nWeighting by travel time rather than distance means a slow road")
    print("is less attractive to route through, so the ranking reflects how")
    print("the network actually performs, not just how it is drawn.")


if __name__ == "__main__":
    main()
