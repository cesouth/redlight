# API reference

All public objects are importable directly from `roadtraffic`.

```python
import roadtraffic as rt
```

---

## Networks

### `Network.from_geojson(path, *, metric_epsg=None, directed=True, oneway_attr="oneway", length_attr=None)`

Load a road network from a GeoJSON `LineString` / `MultiLineString`
FeatureCollection.

| Parameter | Meaning |
|-----------|---------|
| `path` | Path to the `.geojson` / `.json` file. |
| `metric_epsg` | EPSG code of the projected CRS for distance math. Default: auto UTM zone of the first vertex. |
| `directed` | If `True`, build a directed graph; two-way roads get both directions. |
| `oneway_attr` | Feature property naming one-way status (`yes`/`true`/`1`/`-1`). |
| `length_attr` | Property holding precomputed edge length in metres; if absent, computed from projected geometry. |

### `Network.from_file(path, ...)`

Same signature as `from_geojson`. Dispatches GeoJSON to `from_geojson`;
Shapefile (`.shp`) and GeoPackage (`.gpkg`) require the `shapefile` extra
(`pip install roadtraffic[shapefile]`). Source CRS is auto-reprojected to WGS84.

**Useful attributes / methods:** `.graph` (NetworkX `DiGraph`),
`.crs_metric`, `.number_of_nodes()`, `.number_of_edges()`, `.edge_ids`,
`.candidate_edges(px, py, k, max_dist)`, `.project_points(lon, lat)`.

Each edge carries: `edge_id`, `length_m`, `geometry` (projected `LineString`),
plus all source properties (e.g. `highway`, `maxspeed`).

---

## Loading points

### `load_points(path, *, speed_unit="mph", lon_col=None, lat_col=None, time_col=None, speed_col=None, id_col=None, timestamp_unit=None, sep=None, derive_speed=False) -> PointSet`

Load GPS observations from CSV/TSV or GeoJSON Point features.

| Parameter | Meaning |
|-----------|---------|
| `speed_unit` | Unit of the source speed column: `"mph"`, `"kph"`, `"mps"`. Ignored when `derive_speed=True`. |
| `lon_col`, `lat_col` | Coordinate columns; auto-detected if omitted (GeoJSON uses geometry). |
| `time_col` | Timestamp column; auto-detected from common names. |
| `speed_col` | Speed column; auto-detected. Optional when `derive_speed=True`. |
| `id_col` | Trajectory id column (required for `HMMMatcher` and for `derive_speed`); auto-detected. |
| `timestamp_unit` | If timestamps are numeric epochs: `"s"`, `"ms"`, `"us"`, `"ns"`. |
| `sep` | Delimiter for text files; inferred from extension otherwise. |
| `derive_speed` | If `True`, compute speed per unique id from successive GPS positions instead of reading a speed column (needs `id_col`). See [statistics §7](statistics.md). |

Returns a **`PointSet`** with `.df` (canonical columns `point_id`, optional
`traj_id`, `lon`, `lat`, `time`, `speed_mps`), `.has_traj`, and
`.trajectories()` (yields time-sorted sub-frames per trajectory). Also:
`.to_frame(speed_unit=...)`, `.to_csv(path, speed_unit=...)`,
`.to_geojson(path, speed_unit=...)`.

### `save_points(points, path, *, speed_unit="mph") -> str`

Write a `PointSet` to disk (GeoJSON for `.geojson`/`.json`, else CSV). Speed is
written in both m/s (`speed_mps`) and `speed_unit` (`speed_<unit>`) — handy for
persisting a set loaded with `derive_speed=True`.

---

## Matching

### `NearestMatcher(network, *, max_dist=50.0, k=10)`

`.match(points) -> DataFrame`. Independent nearest-edge snapping within
`max_dist` metres; `k` is the max candidate segments examined.

### `HMMMatcher(network, *, sigma_z=6.0, beta=30.0, max_dist=50.0, k=8, max_route_dist_factor=8.0)`

`.match(points) -> DataFrame`. HMM/Viterbi trajectory matching. Requires
`points.has_traj`. `sigma_z` = GPS noise std (m); `beta` = transition decay (m).

**Output columns (both matchers):** `point_id`, `edge_id` (`-1` if unmatched),
`snap_dist_m`, `time`, `speed_mps`, and `traj_id` if present.

---

## Cleaning

### `filter_by_speed(matched, *, min_speed=None, max_speed=None, unit="mph", drop_unmatched=True, mad_outliers=False, mad_threshold=3.5, per_edge=False) -> DataFrame`

Hard speed bounds (in `unit`), optional MAD-based robust outlier removal, and
dropping of unmatched points. `per_edge=True` applies the MAD screen within each
edge group. Best for independent `NearestMatcher` points. See
[statistics](statistics.md#3-speed-cleaning).

### `filter_trajectory_speed(matched, *, drop_unmatched=True, drop_missing_speed=True, dwell_radius_m=25.0, dwell_min_s=120.0, max_speed=None, unit="mph") -> DataFrame`

Trajectory-aware cleaning for HMM-matched data (needs `traj_id`, `lon`, `lat`,
`time`, `speed_mps`). Drops missing-speed points and **stationary dwells**
(parked/idling) but **keeps slow-but-moving congestion**, so trafficability signal
is preserved. A dwell is a run of points within `dwell_radius_m` for ≥
`dwell_min_s`. See [statistics §8](statistics.md).

---

## Aggregation & peaks

### `aggregate_speeds(matched, *, block_hours=1, statistic="mean", output_unit="mph", by_edge=False, min_samples=1) -> DataFrame`

Bin by hour (`block_hours=1`) or N-hour block. `statistic` ∈
`{"mean", "median", "both"}`.

| Always present | `block_start_hour`, `block_label`, `n`, `unit` |
|----------------|------------------------------------------------|
| If mean | `mean_speed`, `std_speed`, `sem_speed`, `ci95_low`, `ci95_high` |
| If median | `median_speed`, `q1_speed`, `q3_speed`, `iqr_speed` |

`by_edge=True` aggregates per `(edge_id, bin)`. `min_samples` suppresses small
bins.

### `peak_analysis(aggregated, *, statistic="mean", n_peak=1, n_offpeak=1) -> dict`

Ranks network-wide bins; **slowest = peak**. Returns
`{"speed_column", "peak", "off_peak", "ranked"}`.

---

## Speed assignment & routing

### `assign_speeds(network, matched, *, statistic="median", output_unit="mps", default_speed_mps=None, block_hours=24, target_hour=None) -> dict`

Computes a per-edge representative speed (optionally restricted to a time-of-day
block via `target_hour`) and writes `obs_speed_mps` and `travel_time_s` onto the
graph. Returns coverage counts.

### `classify_hours(matched, *, statistic="median", peak_hours=None, offpeak_hours=None, min_samples=1) -> dict`

Split the 24 hours into a peak and an off-peak block. Auto by default (hours at or
below the median network-wide hourly speed are peak); override with `peak_hours` /
`offpeak_hours`. Returns `peak_hours`, `offpeak_hours`, `threshold_speed_mps`,
`source`. See [statistics §9](statistics.md).

### `assign_segment_speeds(network, matched, *, statistic="median", peak_hours=None, offpeak_hours=None, default_speed_mps=None, min_samples=1) -> dict`

Write three representative speeds per edge — `obs_speed_mps_{overall,peak,offpeak}`
with matching `travel_time_s_{…}` (and the plain `obs_speed_mps`/`travel_time_s` =
overall for back-compat). Returns the hour blocks used and per-regime `coverage`.

### `Router(network, *, default_speed_mps=11.176)`

- `.route(origin, destination, *, mode="time", cost_func=None, snap=True, period="overall") -> dict`
  — `mode` ∈ `{"distance", "time", "cost"}`; `period` ∈
  `{"overall", "peak", "offpeak"}` selects the time regime (from
  `assign_segment_speeds`). With `snap=True`, `origin`/`destination` are
  `(lon, lat)` snapped to the nearest node. Returns `path`, `edge_ids`,
  `distance_m`, `travel_time_s`, `n_edges`, `n_edges_default`. Raises a clear
  `ValueError` on an empty network, a non-node endpoint (`snap=False`), or no
  path (distinguishing a one-way trap from a disconnected network).
- `.nearest_node(lon, lat)` — nearest graph node to a coordinate.
- `.route_geometry_lonlat(result)` — route as a list of `(lon, lat)` for
  plotting/export.

---

## Units

`SpeedUnit` (`MPH`/`KPH`/`MPS`), `to_mps(value, unit)`, `from_mps(value, unit)`.
