# roadtraffic

**Lightweight, bring-your-own-data trafficability analysis for road networks.**

`roadtraffic` turns your own road network and GPS point observations into
statistically defensible trafficability studies: average speed by time of day,
peak / off-peak detection, and shortest-path routing by time, distance, or a
custom cost.

It is deliberately built on a small, well-understood dependency set
(`numpy`, `pandas`, `scipy`, `shapely`, `pyproj`, `networkx`) — **no GDAL
required** for the core. GeoJSON works out of the box; Shapefile/GeoPackage
support is an opt-in extra.

---

## Features

- **Bring your own data.** Road network as GeoJSON (native) or Shapefile/GPKG
  (optional extra). GPS points as CSV/TSV or GeoJSON, with auto-detected
  columns.
- **No speed column? Two ways to get one.** Opt-in `derive_speed=True`
  reconstructs a per-point speed from successive GPS positions (geodesic
  distance / time); `save_points` writes the result back out. Or, for
  noisy/sparse GPS, match first and call `derive_speeds` to reconstruct speed
  from **on-road displacement** after map matching — more accurate, and
  robust to the matcher flip-flopping between the two directions of a
  two-way road.
- **Units handled for you.** Input speed in mph, kph, or m/s; everything is
  computed internally in m/s and reported in the unit you choose.
- **Two GPS-to-road matchers, one interface.**
  - `NearestMatcher` — fast independent nearest-edge snapping.
  - `HMMMatcher` — trajectory-aware HMM / Viterbi map matching
    (Newson & Krumm, 2009). No extra dependencies.
- **Sound speed cleaning.** Hard physical bounds plus robust MAD-based outlier
  removal — and a trajectory-aware `filter_trajectory_speed` that drops parked
  *dwells* while keeping slow-but-moving congestion (no upward speed bias).
- **Temporal aggregation.** Average speed by hour or by an N-hour block, with
  your choice of **mean** (with standard error and 95% CI) or **median**
  (with IQR).
- **Peak / off-peak detection.** Rank time bins by congestion, and assign three
  speeds per segment — overall, peak block, and off-peak block.
- **Routing.** Shortest path by time (per overall/peak/off-peak regime),
  distance, or a user-supplied cost function, using Dijkstra on a NetworkX graph,
  with actionable errors when no route exists.
- **Mapping.** Export a speed-annotated network as GeoJSON for QGIS / Kepler /
  Leaflet / Mapbox, or render a quick static PNG trafficability map
  (`pip install roadtraffic[mapping]`).

See [`docs/statistics.md`](docs/statistics.md) for the full statistical
methodology behind every number this package reports.

---

## Installation

```bash
# Core (GeoJSON networks, CSV/GeoJSON points)
pip install roadtraffic

# With Shapefile / GeoPackage network support (pulls fiona / GDAL)
pip install roadtraffic[shapefile]

# With static trafficability map rendering (pulls matplotlib)
pip install roadtraffic[mapping]
```

From source:

```bash
git clone https://gitlab.com/your-namespace/roadtraffic.git
cd roadtraffic
pip install -e .
```

---

## Quickstart

```python
import roadtraffic as rt

# 1. Load a road network (GeoJSON here; .shp/.gpkg via Network.from_file)
net = rt.Network.from_geojson("network.geojson")

# 2. Load GPS points (speed in mph, columns auto-detected)
pts = rt.load_points("points.csv", speed_unit="mph")

# 3. Match points to edges (fast) — or HMMMatcher for accuracy
matched = rt.NearestMatcher(net, max_dist=50).match(pts)

# 4. Clean: drop parked/implausible speeds + robust outliers
clean = rt.filter_by_speed(matched, min_speed=2, max_speed=80, unit="mph",
                           mad_outliers=True, per_edge=True)

# 5. Aggregate hourly, mean and median, reported in mph
agg = rt.aggregate_speeds(clean, block_hours=1, statistic="both",
                          output_unit="mph")

# 6. Find peak (slowest) and off-peak (fastest) hours
peaks = rt.peak_analysis(agg, statistic="median", n_peak=3, n_offpeak=3)

# 7. Route by time of day
rt.assign_speeds(net, clean, statistic="median", target_hour=8, block_hours=1)
router = rt.Router(net)
result = router.route((-77.30, 38.68), (-77.27, 38.71), mode="time")
print(result["travel_time_s"], "seconds over", result["distance_m"], "m")
```

No speed column in your GPS data? `load_points` doesn't require one — see
[Quickstart §2a](docs/quickstart.md#2a-no-speed-yet-derive-it-from-the-match)
for the match-then-`derive_speeds` pipeline. More worked examples are in
[`examples/`](examples/).

---

## Documentation

- [Quickstart & concepts](docs/quickstart.md)
- [Statistical methodology](docs/statistics.md)
- [API reference](docs/api.md)
- Build the docs site: `pip install roadtraffic[docs] && mkdocs serve`

---

## License

MIT. See `LICENSE`.
