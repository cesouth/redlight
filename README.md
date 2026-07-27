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

- **Bring your own data — or fetch it.** Road network as GeoJSON (native),
  Shapefile/GPKG (optional extra), or straight from OpenStreetMap via
  `Network.from_overpass(bbox)` (stdlib only). GPS points as CSV/TSV or
  GeoJSON, with auto-detected columns, unit-aware speed parsing
  (`speed_kph` columns convert as kph), and timezone handling (`tz=`) so
  peak hours are computed on the local clock.
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
  (with IQR). An optional `days=` filter (`"weekday"`, `"weekend"`, day
  names/numbers, or a custom list) keeps weekday and weekend traffic apart
  instead of pooling them into the same hour-of-day bin — and
  `day_type_report` turns that into a ready-made weekday-vs-weekend comparison
  (overall/hourly/peak speeds plus the per-block delta). Optional
  `weight_by_variance=True` weights each observation by the precision
  `derive_speeds` measured it to, so a noisy 3-second hop stops counting as
  much as a clean 90-second baseline.
- **Peak / off-peak detection, your way.** Rank time bins by congestion
  (peak = slowest), pick contiguous peak/off-peak windows of user-selected
  width (`n_peak=` / `n_offpeak=`, wrapping midnight), pass explicit hour
  lists, or let the automatic median split decide — then assign three speeds
  per segment: overall, peak block, and off-peak block.
- **Routing.** Shortest path by time (per overall/peak/off-peak regime),
  distance, or a user-supplied cost function, using Dijkstra on a NetworkX
  multigraph (parallel roads and OSM one-way semantics — including
  `oneway=-1` — handled correctly), with actionable errors when no route
  exists. Roads your GPS data never covered are estimated at their **posted
  OSM speed limit** (parsed from `maxspeed`, km/h vs mph handled per the OSM
  spec) rather than at one blanket speed for every road — so an uncovered
  motorway doesn't get routed at the same 25 mph as a side street. Measured
  speeds always win over the limit, and estimated edges stay flagged as
  estimated.
- **Mapping.** Export a speed-annotated network as GeoJSON for QGIS / Kepler /
  Leaflet / Mapbox, or render a quick static PNG trafficability map
  (`pip install roadtraffic[mapping]`) — for any of the three time regimes
  (`period="peak"`).
- **Congestion vs. the posted limit.** `congestion_report` divides observed
  speed by each road's OSM speed limit — the standard level-of-service view,
  and the one raw speeds can't give you: a motorway at 18 mph and a side
  street at 20 mph look alike until you see they're at 0.28 and 0.81 of their
  respective limits. Optional `require_quality=True` across the aggregation
  functions keeps only the intervals `derive_speeds` actually vouched for.
- **Network structure diagnostics.** Travel-time-weighted edge betweenness
  centrality to find chokepoints (not just topologically central roads),
  basic network stats (circuity, streets-per-node, intersection/dead-end
  counts, optional area-based densities), and connectivity diagnostics
  (largest strongly-connected component, one-way-trap vs. genuinely
  disconnected detection) — `roadtraffic.analysis`.

See [`docs/statistics.md`](docs/statistics.md) for the full statistical
methodology behind every number this package reports.

---

## Installation

```bash
# Core (GeoJSON networks, CSV/GeoJSON points)
pip install roadtraffic

# With Shapefile / GeoPackage network support (pulls pyogrio / GDAL;
# needs Python 3.10+)
pip install roadtraffic[shapefile]

# With static trafficability map rendering (pulls matplotlib)
pip install roadtraffic[mapping]
```

From source:

```bash
git clone https://github.com/cesouth/roadtraffic.git
cd roadtraffic
pip install -e .[dev]
pytest  # 223 offline tests, ~7 s
```

---

## Quickstart

```python
import roadtraffic as rt

# 1. Load a road network (GeoJSON here; .shp/.gpkg via Network.from_file;
#    or fetch OSM directly: rt.Network.from_overpass((-77.31, 38.67, -77.26, 38.72)))
net = rt.Network.from_geojson("network.geojson")

# 2. Load GPS points (columns auto-detected; tz makes peak hours local-clock)
pts = rt.load_points("points.csv", tz="America/New_York")

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

# 7. Assign per-segment speeds: overall + a 3-hour peak window + a 4-hour
#    off-peak window (contiguous, chosen from the data), then map and route
rt.assign_segment_speeds(net, clean, n_peak=3, n_offpeak=4)
rt.to_geojson(net, "trafficability_peak.geojson", period="peak")
router = rt.Router(net)
result = router.route((-77.30, 38.68), (-77.27, 38.71), mode="time", period="peak")
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
- [Methodology paper & empirical defense](docs/methodology.md) — the full
  argument for the HMM matcher and the on-road speed estimator, with
  reproducible ground-truth experiments (`scripts/paper_experiments.py`)
- [API reference](docs/api.md)
- [Changelog](CHANGELOG.md)
- Build the docs site: `pip install roadtraffic[docs] && mkdocs serve`

---

## License

MIT. See `LICENSE`.
