"""Example 3 — routing by time, distance, and custom cost.

Assigns edge speeds for a chosen hour, then compares routes.

    python examples/03_routing.py
"""
import os

import roadtraffic as rt

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "sample_data")

ORIGIN = (-77.30, 38.68)
DEST = (-77.30 + 0.025, 38.68 + 0.025)


def main():
    net = rt.Network.from_geojson(os.path.join(DATA, "network.geojson"))
    pts = rt.load_points(os.path.join(DATA, "points.csv"),
                         speed_unit="mph", id_col="vehicle_id")
    matched = rt.NearestMatcher(net, max_dist=60).match(pts)
    clean = rt.filter_by_speed(matched, min_speed=2, max_speed=80, unit="mph",
                               mad_outliers=True, per_edge=True)

    router = rt.Router(net, default_speed_mps=11.0)

    # Distance-optimal route (independent of speeds)
    r_dist = router.route(ORIGIN, DEST, mode="distance")
    print(f"Shortest distance: {r_dist['distance_m']:.0f} m, "
          f"{r_dist['n_edges']} edges")

    # Time-optimal route at a peak hour (8 AM) vs an off-peak hour (3 AM)
    for hour in (8, 3):
        info = rt.assign_speeds(net, clean, statistic="median",
                                target_hour=hour, block_hours=1,
                                default_speed_mps=11.0)
        r = router.route(ORIGIN, DEST, mode="time")
        print(f"Time route @ {hour:02d}:00 — {r['travel_time_s']:.0f} s "
              f"({r['distance_m']:.0f} m); edges with data: "
              f"{info['n_edges_observed']}/{info['n_edges_total']}")

    # Custom cost: penalise longer edges quadratically (toy example)
    def cost(u, v, data):
        return data["length_m"] ** 1.5
    r_cost = router.route(ORIGIN, DEST, mode="cost", cost_func=cost)
    print(f"Custom-cost route: {r_cost['distance_m']:.0f} m, "
          f"{r_cost['n_edges']} edges")

    # Export route geometry as lon/lat for mapping
    coords = router.route_geometry_lonlat(r_dist)
    print(f"Route geometry has {len(coords)} lon/lat vertices; "
          f"first={coords[0]}, last={coords[-1]}")


if __name__ == "__main__":
    main()
