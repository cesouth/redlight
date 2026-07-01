"""End-to-end smoke test: no-speed CSV -> HMM -> derive_speeds -> aggregate/assign/route."""
import json, sys
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/claude")
from roadtraffic.network import Network
from roadtraffic.points import load_points
from roadtraffic.matching import HMMMatcher
from roadtraffic.speeds import derive_speeds
from roadtraffic.cleaning import filter_by_speed
from roadtraffic.aggregate import aggregate_speeds, assign_speeds
from roadtraffic.routing import Router

rng = np.random.default_rng(0)

# longer road: a gentle eastward polyline then a northward leg
coords_e = [[0.0 + 0.0005 * k, 0.0] for k in range(6)]          # ~278 m east
coords_n = [[coords_e[-1][0], 0.0005 * k] for k in range(5)]    # ~221 m north
gj = {"type": "FeatureCollection", "features": [
    {"type": "Feature", "properties": {"name": "E"},
     "geometry": {"type": "LineString", "coordinates": coords_e}},
    {"type": "Feature", "properties": {"name": "N"},
     "geometry": {"type": "LineString", "coordinates": coords_n}},
]}
with open("/home/claude/road2.geojson", "w") as fh:
    json.dump(gj, fh)
net = Network.from_geojson("/home/claude/road2.geojson")
print(f"network: {net.number_of_nodes()} nodes, {net.number_of_edges()} directed edges")

# simulate a trip along E then N at ~10 m/s, 1 Hz, with ~5 m GPS noise, NO speed col
fwd = net._transformer_fwd
inv = net._transformer_inv
# dense ground-truth path in projected metres
all_ll = coords_e + coords_n[1:]
xs, ys = fwd.transform(np.array([c[0] for c in all_ll]), np.array([c[1] for c in all_ll]))
# walk along the polyline at 10 m/s sampling every 1 s
from shapely.geometry import LineString
line = LineString(np.column_stack([xs, ys]))
v_true = 10.0
total = line.length
times = np.arange(0, total / v_true, 1.0)
rows = []
for k, t in enumerate(times):
    p = line.interpolate(v_true * t)
    nx_, ny_ = p.x + rng.normal(0, 5), p.y + rng.normal(0, 5)
    lon, lat = inv.transform(nx_, ny_)
    rows.append({"track_id": "trip1", "lon": float(lon), "lat": float(lat),
                 "timestamp": pd.Timestamp("2024-06-01T08:00:00") + pd.to_timedelta(t, "s")})
df = pd.DataFrame(rows)
df.to_csv("/home/claude/trip.csv", index=False)
print(f"simulated {len(df)} fixes, true speed {v_true} m/s ({v_true/0.44704:.1f} mph)")

# ---- load WITHOUT a speed column (the patched loader must allow this) ---------
ps = load_points("/home/claude/trip.csv", id_col="track_id")
assert "speed_mps" not in ps.df.columns, "loader should not invent a speed column"
print(f"loaded PointSet: {len(ps)} rows, has_traj={ps.has_traj}, "
      f"columns={list(ps.df.columns)}")

# ---- HMM match ----------------------------------------------------------------
matched = HMMMatcher(net, sigma_z=6.0, beta=30.0, max_dist=80.0).match(ps)
matched_frac = (matched["edge_id"] != -1).mean()
print(f"HMM matched {matched_frac*100:.0f}% of fixes to an edge")

# ---- derive speeds ------------------------------------------------------------
res = derive_speeds(net, matched, ps, default_pos_sigma_m=5.0, min_baseline_m=75.0)
iv = res["intervals"]
eo = res["edge_observations"]
good = iv[iv["quality"]]
print(f"\nintervals: {len(iv)} total, {len(good)} quality")
print(f"  median derived speed (quality) = {good['speed_mps'].median():.2f} m/s "
      f"({good['speed_mps'].median()/0.44704:.1f} mph)  [true 10.0 m/s]")
print(f"  edge_observations rows = {len(eo)}")

# ---- feed straight into existing cleaning/aggregation -------------------------
clean = filter_by_speed(eo, min_speed=1, max_speed=80, unit="mph",
                        drop_unmatched=True, mad_outliers=True, per_edge=True)
agg = aggregate_speeds(clean, statistic="both", output_unit="mph", by_edge=True)
print(f"\nfilter_by_speed kept {len(clean)}/{len(eo)} rows")
print(f"per-edge aggregation rows: {len(agg)}")
print(agg[["edge_id", "block_label", "n", "median_speed", "mean_speed"]].to_string(index=False))

# ---- assign back to the graph and route by time -------------------------------
info = assign_speeds(net, clean, statistic="median", output_unit="mps")
print(f"\nassign_speeds: {info}")
r = Router(net)
origin = (coords_e[0][0], coords_e[0][1])
dest = (coords_n[-1][0], coords_n[-1][1])
route = r.route(origin, dest, mode="time")
print(f"route by time: {route['n_edges']} edges, {route['distance_m']:.0f} m, "
      f"{route['travel_time_s']:.1f} s  -> implied {route['distance_m']/route['travel_time_s']:.2f} m/s")
print("\nALL STAGES OK")
