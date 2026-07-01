# Quickstart & concepts

## The pipeline

`roadtraffic` follows a fixed five-stage flow. Every stage hands a clean object
to the next:

```
network + points  →  match  →  clean  →  aggregate  →  peaks / route
```

1. **Load** a road network and a GPS point file.
2. **Match** each GPS point to a network edge.
3. **Clean** the matched observations (speed bounds + robust outlier removal).
4. **Aggregate** speeds into time bins (hour or N-hour block).
5. **Analyse** — rank peak/off-peak periods, and/or assign edge speeds and
   route.

---

## 1. Input data expectations

### Road network

Any of:

- **GeoJSON** of `LineString` / `MultiLineString` features (works with the core
  install).
- **Shapefile** or **GeoPackage** (`pip install roadtraffic[shapefile]`).

Optional feature properties that the package understands: a one-way flag
(default property name `oneway`, values like `yes`/`true`/`1`/`-1`) and any
attributes you want preserved on edges (e.g. OSM `highway`, `maxspeed`).

```python
import roadtraffic as rt
net = rt.Network.from_geojson("network.geojson")          # GeoJSON
# net = rt.Network.from_file("network.gpkg")              # needs [shapefile]
```

### GPS points

A CSV/TSV or GeoJSON Point file with, per row: longitude, latitude, a
timestamp, a speed, and optionally a trajectory/unit id. Column names are
auto-detected from common spellings; override any of them explicitly.

```python
pts = rt.load_points(
    "points.csv",
    speed_unit="kph",        # mph | kph | mps
    id_col="vehicle_id",     # needed if you want HMM matching
)
```

If timestamps are numeric epochs, pass `timestamp_unit="s"` (or `ms`/`us`/`ns`).

**No speed column?** Set `derive_speed=True` to compute speed per unit from
successive GPS positions (requires `id_col`). Persist the result with
`rt.save_points(pts, "with_speed.csv")`. Method: [statistics §7](statistics.md).

```python
pts = rt.load_points("points.csv", derive_speed=True, id_col="vehicle_id")
rt.save_points(pts, "points_with_speed.csv", speed_unit="mph")
```

---

## 2. Choosing a matcher

| | `NearestMatcher` | `HMMMatcher` |
|---|---|---|
| Speed | Fast | Slower (shortest-path calls) |
| Needs trajectory id | No | **Yes** |
| Accuracy at intersections | Lower | Higher |
| Best for | dense data, simple networks, quick looks | ordered trajectories, dense urban grids |

```python
matched = rt.NearestMatcher(net, max_dist=50).match(pts)
# or
matched = rt.HMMMatcher(net, sigma_z=6, beta=30).match(pts)
```

Both return the same columns, so everything after this point is identical.

---

## 3. Cleaning

```python
clean = rt.filter_by_speed(
    matched,
    min_speed=2, max_speed=80, unit="mph",  # hard physical bounds
    mad_outliers=True, per_edge=True,        # robust outlier removal per edge
)
```

`min_speed` typically removes parked/idling pings; `max_speed` removes GPS
jumps. MAD-based removal is robust to the very errors you are trying to drop —
see [statistics](statistics.md#3-speed-cleaning).

For **HMM-matched trajectories**, prefer `filter_trajectory_speed`: it drops
parked *dwells* but keeps slow-but-moving congestion, so you don't bias segment
speeds upward by deleting the traffic you're studying ([statistics §8](statistics.md)):

```python
clean = rt.filter_trajectory_speed(matched, dwell_radius_m=25, dwell_min_s=120)
```

---

## 4. Aggregation

```python
# Hourly, both mean and median, reported in mph
hourly = rt.aggregate_speeds(clean, block_hours=1, statistic="both",
                             output_unit="mph")

# 6-hour blocks (00–06, 06–12, 12–18, 18–24), mean only
blocks = rt.aggregate_speeds(clean, block_hours=6, statistic="mean")
```

The mean path reports `sem_speed` and a 95% CI; the median path reports the IQR.
`by_edge=True` gives one row per edge per time bin.

---

## 5. Peaks and routing

```python
peaks = rt.peak_analysis(hourly, statistic="median", n_peak=3, n_offpeak=3)
for r in peaks["peak"]:        # slowest = busiest
    print(r["block_label"], r["median_speed"])

# Three per-segment speeds (overall / peak / off-peak), blocks auto-detected,
# then route on a chosen regime. Pooling into two blocks keeps far more data per
# segment than 24 hourly slices — which is what makes time routing stable.
info = rt.assign_segment_speeds(net, clean, statistic="median")
print("peak hours:", info["peak_hours"], "coverage:", info["coverage"])
router = rt.Router(net)
res = router.route((-77.30, 38.68), (-77.27, 38.71), mode="time", period="peak")
print(res["travel_time_s"], "s; edges on default speed:", res["n_edges_default"])

# Compare against the shortest-distance route
res_d = router.route((-77.30, 38.68), (-77.27, 38.71), mode="distance")
```

For a custom objective, pass `mode="cost"` with a weight function:

```python
def risk_weight(u, v, data):
    base = data.get("travel_time_s") or data["length_m"] / 11.0
    return base * (2.0 if data.get("highway") == "motorway" else 1.0)

res = router.route(o, d, mode="cost", cost_func=risk_weight)
```

---

## Next steps

- Worked, runnable scripts: [`examples/`](../examples).
- Why each number is what it is: [statistics](statistics.md).
- Full signatures: [API reference](api.md).
