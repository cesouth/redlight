"""Validate roadtraffic.speeds against a synthetic network with known ground truth."""
import json, sys
import numpy as np
import pandas as pd
from shapely.geometry import Point

sys.path.insert(0, "/home/claude")
from roadtraffic.network import Network
from roadtraffic.points import PointSet, load_points
from roadtraffic.matching import HMMMatcher
from roadtraffic.speeds import derive_speeds

# ---- build an L-shaped two-way road: east leg then north leg ------------------
gj = {
    "type": "FeatureCollection",
    "features": [
        {"type": "Feature", "properties": {"name": "east_leg"},
         "geometry": {"type": "LineString",
                      "coordinates": [[0.0, 0.0], [0.001, 0.0]]}},
        {"type": "Feature", "properties": {"name": "north_leg"},
         "geometry": {"type": "LineString",
                      "coordinates": [[0.001, 0.0], [0.001, 0.001]]}},
    ],
}
with open("/home/claude/lnet.geojson", "w") as fh:
    json.dump(gj, fh)

net = Network.from_geojson("/home/claude/lnet.geojson")  # two-way -> 4 directed edges

# locate forward directed edges by their endpoints
def find_edge(u, v):
    for eid in net.edge_ids:
        if net.edge_endpoints(int(eid)) == (u, v):
            return int(eid)
    raise KeyError((u, v))

n00, n10, n11 = (0.0, 0.0), (0.001, 0.0), (0.001, 0.001)
east_fwd = find_edge(n00, n10)
north_fwd = find_edge(n10, n11)
L_east = net.graph[n00][n10]["length_m"]
L_north = net.graph[n10][n11]["length_m"]
print(f"edge lengths: east={L_east:.3f} m  north={L_north:.3f} m")

def place(eid, frac, perp_axis, perp_m=3.0):
    """lon/lat of a point at arc-fraction `frac` along edge `eid`, offset
    `perp_m` metres perpendicular (axis 'x' or 'y') so the snap is non-trivial."""
    geom = net._edge_geoms_proj[eid]
    p = geom.interpolate(frac * geom.length)
    x, y = p.x, p.y
    if perp_axis == "y":
        y += perp_m
    else:
        x += perp_m
    lon, lat = net._transformer_inv.transform(x, y)
    return float(lon), float(lat)


def run_case(name, eid1, f1, ax1, eid2, f2, ax2, true_v, true_dist):
    lon1, lat1 = place(eid1, f1, ax1)
    lon2, lat2 = place(eid2, f2, ax2)
    dt = true_dist / true_v
    pdf = pd.DataFrame({
        "point_id": [0, 1],
        "traj_id": ["T", "T"],
        "lon": [lon1, lon2],
        "lat": [lat1, lat2],
        "speed_mps": [np.nan, np.nan],
    })
    # exact timestamps: base and base + dt seconds
    base = pd.Timestamp("2024-01-01T08:00:00")
    pdf["time"] = [base, base + pd.to_timedelta(dt, unit="s")]
    ps = PointSet(pdf, has_traj=True)

    matched = pd.DataFrame({
        "point_id": [0, 1],
        "edge_id": [eid1, eid2],
        "snap_dist_m": [3.0, 3.0],
        "time": pdf["time"].values,
        "speed_mps": [np.nan, np.nan],
        "traj_id": ["T", "T"],
    })

    res = derive_speeds(net, matched, ps, default_pos_sigma_m=5.0)
    iv = res["intervals"].iloc[0]
    eo = res["edge_observations"]
    print(f"\n[{name}]")
    print(f"  true distance = {true_dist:.3f} m   derived = {iv['distance_m']:.3f} m   "
          f"err = {abs(iv['distance_m']-true_dist)*100/true_dist:.3f}%")
    print(f"  true speed    = {true_v:.3f} m/s  derived = {iv['speed_mps']:.3f} m/s  "
          f"err = {abs(iv['speed_mps']-true_v)*100/true_v:.3f}%")
    print(f"  edges attributed = {sorted(eo['edge_id'].tolist())}  (n_edges={iv['n_edges']})")
    return iv, eo


# Case 1: different edges, route turns the corner at the shared node n10.
# true on-road distance = (1-0.5)*L_east + 0 + 0.5*L_north
d1 = (1 - 0.5) * L_east + 0.0 + 0.5 * L_north
run_case("turn across shared node", east_fwd, 0.5, "y", north_fwd, 0.5, "x",
         true_v=10.0, true_dist=d1)

# Case 2: same directed edge, forward from 0.25 to 0.75
d2 = (0.75 - 0.25) * L_east
run_case("same edge forward", east_fwd, 0.25, "y", east_fwd, 0.75, "y",
         true_v=8.0, true_dist=d2)

print("\nExpected attributed roads for case 1: both directions of east+north edges")
