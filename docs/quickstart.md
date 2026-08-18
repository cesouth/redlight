# Quickstart & concepts

## The pipeline

`redlight` follows a fixed flow. Every stage hands a clean object to the
next:

```
network + points  →  match  →  (derive speed)  →  clean  →  aggregate  →  peaks / route / map
```

1. **Load** a road network and a GPS point file.
2. **Match** each GPS point to a network edge.
3. **Derive speed** from the match, if the points had none (§2a below).
4. **Clean** the matched observations (speed bounds + robust outlier removal).
5. **Aggregate** speeds into time bins (hour or N-hour block).
6. **Analyse** — rank peak/off-peak periods, assign edge speeds and route,
   and/or export a speed map.

---

## 1. Input data expectations

### Road network

Any of:

- **GeoJSON** of `LineString` / `MultiLineString` features (works with the core
  install).
- **Shapefile** or **GeoPackage** (`pip install redlight[shapefile]`,
  needs Python 3.10+).

Optional feature properties that the package understands: a one-way flag
(default property name `oneway`, values like `yes`/`true`/`1`/`-1`) and any
attributes you want preserved on edges (e.g. OSM `highway`, `maxspeed`).

```python
import redlight as rl
net = rl.Network.from_geojson("network.geojson")          # GeoJSON
# net = rl.Network.from_file("network.gpkg")              # needs [shapefile]
```

### GPS points

A CSV/TSV or GeoJSON Point file with, per row: longitude, latitude, a
timestamp, a speed, and optionally a trajectory/unit id. Column names are
auto-detected from common spellings; override any of them explicitly.

```python
pts = rl.load_points(
    "points.csv",
    speed_unit="kph",        # mph | kph | mps
    id_col="vehicle_id",     # needed if you want HMM matching
)
```

If timestamps are numeric epochs, pass `timestamp_unit="s"` (or `ms`/`us`/`ns`).

**No speed column?** You have two options, in order of preference:

1. **Match first, then derive speed on-road** (recommended for noisy/sparse
   GPS — see step 2 and [statistics §10](statistics.md#10-on-road-speed-derivation-from-matched-trajectories)).
   Just omit `speed_col` — `load_points` doesn't require it:
   ```python
   pts = rl.load_points("points.csv", id_col="vehicle_id")  # no speed_col needed
   ```
2. **`derive_speed=True`** computes a per-point speed directly from successive
   straight-line GPS positions (requires `id_col`), without needing a network
   or a matching step first. Simpler, but biased low on curves and by noise
   over short gaps — see [statistics §7](statistics.md). Persist the result
   with `rl.save_points(pts, "with_speed.csv")`.
   ```python
   pts = rl.load_points("points.csv", derive_speed=True, id_col="vehicle_id")
   rl.save_points(pts, "points_with_speed.csv", speed_unit="mph")
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
matched = rl.NearestMatcher(net, max_dist=50).match(pts)
# or
matched = rl.HMMMatcher(net, sigma_z=6, beta=30).match(pts)
```

Both return the same columns, so everything after this point is identical.
`matched` always carries `lon`/`lat`; it carries `speed_mps` too, but only if
`pts` had one. For noisy GPS (1–100 m accuracy), raise `HMMMatcher`'s
`sigma_z` (try 15–25 m) and `max_dist` (try 100–150 m) so poor fixes still
match.

---

## 2a. No speed yet? Derive it from the match

If `pts` had no speed column, `matched` doesn't either — derive it from
on-road displacement between consecutive matched fixes
([statistics §10](statistics.md#10-on-road-speed-derivation-from-matched-trajectories)):

```python
res = rl.derive_speeds(
    net, matched, pts,
    pos_accuracy_col="accuracy",   # per-fix metres, if you have it
    default_pos_sigma_m=20.0,      # fallback if you don't
    min_baseline_m=100.0,          # important for noisy/dense fixes -- see below
    max_speed_mps=60.0,
)
intervals = res["intervals"]           # per-point speed record
edge_obs  = res["edge_observations"]   # long: one row per (interval, edge) --
                                        # feed this into filter_by_speed / aggregate_speeds
```

If fixes are close together in time and GPS is noisy, single-interval
displacement can be dominated by noise, which biases naive per-fix speed
*high*, not just scattered. `min_baseline_m` merges consecutive hops until
their summed on-road distance clears a baseline (roughly 5–10× your GPS
sigma) before computing a speed — set it whenever fixes are dense and noisy.
`edge_obs` is schema-compatible with the cleaning/aggregation steps below;
use it in place of `matched` from here on when you derived speed this way.

---

## 3. Cleaning

```python
clean = rl.filter_by_speed(
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
clean = rl.filter_trajectory_speed(matched, dwell_radius_m=25, dwell_min_s=120)
```

---

## 4. Aggregation

```python
# Hourly, both mean and median, reported in mph
hourly = rl.aggregate_speeds(clean, block_hours=1, statistic="both",
                             output_unit="mph")

# 6-hour blocks (00–06, 06–12, 12–18, 18–24), mean only
blocks = rl.aggregate_speeds(clean, block_hours=6, statistic="mean")
```

The mean path reports `sem_speed` and a 95% CI; the median path reports the IQR.
`by_edge=True` gives one row per edge per time bin.

---

## 5. Peaks and routing

```python
peaks = rl.peak_analysis(hourly, statistic="median", n_peak=3, n_offpeak=3)
for r in peaks["peak"]:        # slowest = busiest
    print(r["block_label"], r["median_speed"])

# Three per-segment speeds (overall / peak / off-peak), blocks auto-detected,
# then route on a chosen regime. Pooling into two blocks keeps far more data per
# segment than 24 hourly slices — which is what makes time routing stable.
info = rl.assign_segment_speeds(net, clean, statistic="median")
print("peak hours:", info["peak_hours"], "coverage:", info["coverage"])
router = rl.Router(net)
res = router.route((-77.30, 38.80), (-77.27, 38.81), mode="time", period="peak")
print(res["travel_time_s"], "s; edges on default speed:", res["n_edges_default"])

# Compare against the shortest-distance route
res_d = router.route((-77.30, 38.80), (-77.27, 38.81), mode="distance")
```

For a custom objective, pass `mode="cost"` with a weight function. The network
is a `MultiDiGraph` (parallel roads are distinct edges), so `cost_func(u, v,
edges)` receives `edges` as a dict of `edge_id -> attrs` for every parallel
road between `u` and `v` -- reduce over it yourself (typically `min`):

```python
def risk_weight(u, v, edges):
    def cost(d):
        base = d.get("travel_time_s") or d["length_m"] / 11.0
        return base * (2.0 if d.get("highway") == "motorway" else 1.0)
    return min(cost(d) for d in edges.values())

res = router.route(o, d, mode="cost", cost_func=risk_weight)
```

---

## 6. Mapping

Once `assign_speeds` (or `assign_segment_speeds`) has written `obs_speed_mps`
onto the graph, export it as a map:

```python
# GeoJSON for QGIS / Kepler / Leaflet / Mapbox -- no extra dependency
rl.to_geojson(net, "speeds.geojson", speed_unit="mph")
```

For a quick static PNG coloured by speed, install the mapping extra
(`pip install redlight[mapping]`):

<!-- needs: matplotlib -->
```python
rl.plot_speed_map(net, "speeds.png", speed_unit="mph")
```

---

## 7. Network analysis

`redlight.analysis` answers structural questions about the network
itself — independent of the speed pipeline above, except where noted:

```python
# Which roads are chokepoints? Needs assign_speeds/assign_segment_speeds to
# have run first (every edge needs a travel_time_s); write it onto the graph
# so it shows up in to_geojson()'s keep_tags / plot_speed_map for free.
bc = rl.edge_betweenness_centrality(net, weight="travel_time_s",
                                    write_attr="betweenness")

# Basic descriptive stats -- no speed pipeline needed.
stats = rl.network_stats(net)
print(stats["circuity_avg"], stats["n_intersections"], stats["streets_per_node_avg"])

# Is the network one routable piece? Also no speed pipeline needed -- run
# this before routing on an unfamiliar or clipped network.
report = rl.connectivity_report(net)
if not report["is_strongly_connected"]:
    print("stranded edges:", report["stranded_edge_ids"])
```

See [statistics §11](statistics.md) for what each measure means and the
correctness guards behind them (in particular, `edge_betweenness_centrality`
requires an explicit `weight=` — there's no default, since silently getting
it wrong is worse than an error here).

---

## Next steps

- Worked, runnable scripts:
  [`examples/`](https://github.com/cesouth/redlight/tree/main/examples).
- Why each number is what it is: [statistics](statistics.md).
- Full signatures: [API reference](api.md).
