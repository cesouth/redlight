"""Routing on measured speeds, and how peak traffic changes the answer.

    python examples/05_network/routing.py

Once measured speeds are written onto the graph, the fastest route is a
question about the network as it actually performs -- and the answer differs
between peak and off-peak. Where a segment was never observed, the router falls
back to the posted limit before it falls back to a global default.
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
    rt.assign_segment_speeds(net, clean, statistic="median",
                             n_peak=3, n_offpeak=3)

    router = rt.Router(net)
    # Route between opposite corners of the grid. nearest_node snaps a
    # lon/lat to the closest network node, which is what you want when the
    # endpoints come from an address or a click rather than the graph.
    origin = router.nearest_node(-77.300, 38.800)
    dest = router.nearest_node(-77.300 + 4 * 0.0090, 38.800 + 3 * 0.0080)

    rule("Fastest route, by regime")
    for period in ("overall", "peak", "offpeak"):
        route = router.route(origin, dest, mode="time", period=period)
        print(f"  {period:<8} {route['distance_m']:7.0f} m  "
              f"{route['travel_time_s'] / 60:5.1f} min  "
              f"{route['n_edges']} edges "
              f"({route['n_edges_default']} on a fallback speed)")
    print("\nSame endpoints, different regimes. The peak route may be longer")
    print("in distance and still faster in time -- which is the point of")
    print("routing on measured speeds rather than on geometry.")
    print("\nn_edges_default is the honesty column: it counts edges where no")
    print("measured speed existed for that regime, so the router used the")
    print("posted limit or the global default. A route that is mostly")
    print("fallback is a statement about your coverage, not about traffic.")

    rule("Shortest by distance, for comparison")
    route = router.route(origin, dest, mode="distance")
    print(f"  distance-optimal: {route['distance_m']:.0f} m, "
          f"{route['travel_time_s'] / 60:.1f} min, {route['n_edges']} edges")
    print("  Optimising distance can pick a slower path; optimising time can")
    print("  pick a longer one. Choose the one that matches your question.")

    rule("The route as geometry")
    geom = router.route_geometry_lonlat(route)
    print(f"  {len(geom)} lon/lat vertices, "
          f"first={geom[0][0]:.4f},{geom[0][1]:.4f} "
          f"last={geom[-1][0]:.4f},{geom[-1][1]:.4f}")
    print("  Feed this straight to a map layer or a GeoJSON LineString.")

    rule("Falling back when a road was never observed")
    print("Router(use_maxspeed=True) is the default. For an unobserved edge it")
    print("uses the posted limit from the network's maxspeed tag, and only")
    print("falls back to default_speed_mps when there is no usable tag either.")
    print("Without that, an unobserved arterial and an unobserved side street")
    print("would be treated as equally fast, and time-based routing would")
    print("quietly degenerate into distance-based routing.")


if __name__ == "__main__":
    main()
