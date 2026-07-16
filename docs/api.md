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
| `oneway_attr` | Feature property naming one-way status, with OSM semantics: `yes`/`true`/`1` = one-way along the digitized direction; `-1`/`reverse` = one-way **against** it. |
| `length_attr` | Property holding precomputed edge length in metres; if absent, computed from projected geometry. |

Closed-loop features (endpoints coincide, e.g. a roundabout digitized as one
closed way) cannot be modelled and are skipped with a warning — split them into
open segments first. Source properties that collide with the reserved edge
attributes (`edge_id`, `length_m`, `geometry`) are preserved under an `_src`
suffix.

### `Network.from_file(path, ..., layer=None)`

Same signature as `from_geojson`, plus `layer` (name or index of a layer to
read from a multi-layer file, e.g. one table in a GeoPackage that holds
several; default: the file's own default layer). Dispatches GeoJSON to
`from_geojson`; Shapefile (`.shp`) and GeoPackage (`.gpkg`) require the
`shapefile` extra (`pip install roadtraffic[shapefile]`, needs Python 3.10+).
Source CRS is auto-reprojected to WGS84.

### `Network.from_overpass(bbox, *, metric_epsg=None, directed=True, oneway_attr="oneway", highway_regex=None, url=None, timeout=90.0)`

Fetch an OSM road network for `bbox = (min_lon, min_lat, max_lon, max_lat)`
via the Overpass API (stdlib only — no new dependencies; needs internet).
`highway_regex` filters OSM highway classes (default: drivable roads). Use a
downloaded extract + `from_geojson` for large study areas.

**Useful attributes / methods:** `.graph` (NetworkX `MultiDiGraph`, edge key =
`edge_id` — parallel roads between the same nodes coexist),
`.crs_metric`, `.number_of_nodes()`, `.number_of_edges()`, `.edge_ids`,
`.candidate_edges(px, py, k, max_dist)`, `.project_points(lon, lat)`,
`.edge_length(eid)`, `.edge_geometry(eid)`, `.edge_data(eid)`,
`.edge_coords_lonlat(eid)`, `.road_edge_ids(eid)` (the directed edge pair of a
two-way road), `.edges_between(u, v)`.

**Batch / vectorised lookups** (used internally by `NearestMatcher` and
`HMMMatcher`; useful directly if you're writing your own matching or
snapping logic over many points at once):

- `.nearest_edges(px, py, *, k=10, max_dist=50.0, chunk_size=200_000)` —
  nearest edge per point for many points in one vectorised pass. Returns
  `(edge_ids, snap_dists)` arrays; `edge_ids[i] == -1` and `snap_dists[i]`
  is `NaN` where no edge lies within `max_dist`. Identical results to
  calling `.candidate_edges` per point and taking the closest candidate,
  10-30x faster for large point sets; `chunk_size` only bounds peak memory.
- `.candidate_edges_batch(px, py, *, k=10, max_dist=50.0)` — per-point
  candidate lists for many points in one vectorised pass. Returns a list
  (one entry per point) of `(edge_id, perp_dist, t)` lists, same content
  and ordering as calling `.candidate_edges` per point.
- `.csgraph()` — the directed graph as a `scipy.sparse` CSR matrix, plus
  node<->int index maps: `(csr, node_to_int, int_to_node)`. `csr[i, j]` is
  the minimum `length_m` over parallel edges from node `i` to node `j` (the
  same distance Dijkstra would use on the multigraph). Built lazily and
  cached; pairs with `scipy.sparse.csgraph.dijkstra` for fast bounded
  shortest-path queries without pure-Python networkx search.

Each edge carries: `edge_id`, `length_m`, `geometry` (projected `LineString`),
plus all source properties (e.g. `highway`, `maxspeed`).

---

## Loading points

### `load_points(path, *, speed_unit=None, lon_col=None, lat_col=None, time_col=None, speed_col=None, id_col=None, timestamp_unit=None, tz=None, sep=None, derive_speed=False) -> PointSet`

Load GPS observations from CSV/TSV or GeoJSON Point features.

| Parameter | Meaning |
|-----------|---------|
| `speed_unit` | Unit of the source speed column: `"mph"`, `"kph"`, `"mps"`. Default `None`: inferred from a unit-bearing column name (`speed_kph` → kph), else mph. An explicit value wins (warning on contradiction). Ignored when `derive_speed=True` or no speed column is found. |
| `lon_col`, `lat_col` | Coordinate columns; auto-detected if omitted (GeoJSON uses geometry). |
| `time_col` | Timestamp column; auto-detected from common names. |
| `speed_col` | Speed column; auto-detected. Fully optional — see below. |
| `id_col` | Trajectory id column (required for `HMMMatcher` and for `derive_speed`); auto-detected. Rows with a missing id are dropped with a warning. |
| `timestamp_unit` | If timestamps are numeric epochs: `"s"`, `"ms"`, `"us"`, `"ns"` (epochs are UTC — pair with `tz`). |
| `tz` | IANA timezone of the study area (e.g. `"America/New_York"`). Aware timestamps/epochs are converted to it; naive input is treated as UTC when `tz` is given. Stored times are always naive **local clock** — what hour-of-day peak statistics need. Aware input without `tz` warns. |
| `sep` | Delimiter for text files; inferred from extension otherwise. |
| `derive_speed` | If `True`, compute speed per unique id from successive GPS positions instead of reading a speed column (needs `id_col`). See [statistics §7](statistics.md). |

Three ways to end up with speed (or not): (1) a speed column is found/given —
read and converted to m/s; (2) `derive_speed=True` — a per-point speed is
reconstructed from consecutive straight-line positions; (3) neither — the
returned `PointSet` simply has **no** `speed_mps` column, valid for a
matcher + `derive_speeds` pipeline ([statistics §10](statistics.md)). Only
`lon`/`lat`/`time` are ever required.

Returns a **`PointSet`** with `.df` (canonical columns `point_id`, optional
`traj_id`, `lon`, `lat`, `time`, and `speed_mps` if available), `.has_traj`, and
`.trajectories()` (yields time-sorted sub-frames per trajectory). Also:
`.to_frame(speed_unit=...)`, `.to_csv(path, speed_unit=...)`,
`.to_geojson(path, speed_unit=...)` (speed columns omitted from all three if
`speed_mps` isn't present).

### `save_points(points, path, *, speed_unit="mph") -> str`

Write a `PointSet` to disk (GeoJSON for `.geojson`/`.json`, else CSV). Speed is
written in both m/s (`speed_mps`) and `speed_unit` (`speed_<unit>`) if present —
handy for persisting a set loaded with `derive_speed=True`.

---

## Matching

### `NearestMatcher(network, *, max_dist=50.0, k=10)`

`.match(points) -> DataFrame`. Independent nearest-edge snapping within
`max_dist` metres; `k` is the max candidate segments examined. Fully
vectorised — one batch KDTree query for the whole point set (hundreds of
thousands of points per second).

### `HMMMatcher(network, *, sigma_z=6.0, beta=30.0, max_dist=50.0, k=8, max_route_dist_factor=8.0, n_jobs=1, dist_cache_size=10_000)`

`.match(points) -> DataFrame`. HMM/Viterbi trajectory matching. Requires
`points.has_traj`. `sigma_z` = GPS noise std (m) — raise it toward your data's
real error; `beta` = transition decay (m). The per-step transition search is
bounded at `max(max_route_dist_factor × step, max_dist × 4)` so cost stays
bounded on large networks. Fixes with no candidate edge within `max_dist` are
reported as `edge_id = -1` (never fabricated); the chain restarts after a
candidate-less prefix. See [statistics §2.2](statistics.md).

Transition distances run on scipy's C Dijkstra over the network's CSR
adjacency with an LRU cache (`dist_cache_size` sources) reused across steps
and trajectories — tens of thousands of points per second serial. `n_jobs`
decodes independent trajectories in parallel processes with identical
results; measure before enabling (worker start-up costs dominate until jobs
run for minutes — see the class docstring).

**Output columns (both matchers):** `point_id`, `edge_id` (`-1` if unmatched),
`snap_dist_m`, `lon`, `lat`, `time`, `traj_id` if present, and `speed_mps` **if**
the input `PointSet` had one (position+time-only input matches the same way).
`lon`/`lat` let trajectory-aware cleaning and `derive_speeds` work from the
matched frame alone, without rejoining the original `PointSet`.

---

## Speed derivation from positions

### `derive_speeds(network, matched, points, *, pos_accuracy_col=None, default_pos_sigma_m=15.0, min_dt_s=0.5, max_dt_s=120.0, max_snap_dist_m=60.0, max_speed_mps=60.0, min_snr=3.0, max_route_dist_factor=6.0, route_cutoff_floor_m=300.0, min_baseline_m=None) -> dict`

Reconstruct speed from **on-road displacement** between consecutive matched
fixes in a trajectory — the recommended way to get speed when `load_points`
found no speed column. See [statistics §10](statistics.md) for the full
methodology (undirected-graph distance, the quality flag, and why
`min_baseline_m` matters for noisy/sparse fixes).

| Parameter | Meaning |
|-----------|---------|
| `matched` | Output of a matcher (`point_id`, `edge_id`; `traj_id` if present). |
| `points` | The `PointSet` that was matched — lon/lat/time (and optionally `pos_accuracy_col`) are joined back in on `point_id`. |
| `pos_accuracy_col` | Column in `points.df` giving per-fix horizontal accuracy (m); if omitted, `default_pos_sigma_m` is used for every fix. |
| `min_dt_s`, `max_dt_s` | Plausible spacing between consecutive fixes; outside this range -> `quality=False`. |
| `max_snap_dist_m` | Snap distance above which a fix's match is considered unreliable. |
| `max_speed_mps` | Implausible-speed ceiling for the quality flag (does not drop rows). |
| `min_snr` | Minimum distance / combined-sigma ratio for `quality=True`. |
| `max_route_dist_factor`, `route_cutoff_floor_m` | Bound the mid-hop shortest-path search; beyond it the hop is unreachable. |
| `min_baseline_m` | If set, merge consecutive hops until summed on-road distance reaches this baseline before emitting a speed. Recommended for dense, noisy fixes (try 5–10× your GPS sigma). |

Returns `{"intervals": DataFrame, "edge_observations": DataFrame}`:

- `intervals` — one row per consecutive fix pair (the per-point speed record):
  `interval_id, traj_id, point_id_from, point_id_to, time, t_from, t_to, dt_s,
  distance_m, speed_mps, edge_from, edge_to, n_edges, snap_dist_m,
  speed_sigma_mps, speed_var, quality`. `time` is the interval midpoint. Each
  row is one independent measurement — the right frame for network-wide
  statistics.
- `edge_observations` — long format, one row per (interval, traversed edge):
  `interval_id, edge_id, speed_mps, time, traj_id, dt_s, distance_m,
  snap_dist_m, speed_sigma_mps, speed_var, quality`. Schema-compatible with
  `filter_by_speed`, `aggregate_speeds` and `assign_speeds` — feed it straight
  in. Network-wide aggregations deduplicate on `interval_id` automatically, so
  the deliberate per-edge replication never inflates sample sizes.

---

## Cleaning

### `filter_by_speed(matched, *, min_speed=None, max_speed=None, unit="mph", drop_unmatched=True, mad_outliers=False, mad_threshold=3.5, per_edge=False) -> DataFrame`

Hard speed bounds (in `unit`), optional MAD-based robust outlier removal, and
dropping of unmatched points. `per_edge=True` applies the MAD screen within each
edge group. Best for independent `NearestMatcher` points. Requires a
`speed_mps` column — run `derive_speeds` first if you loaded points without a
speed column. See [statistics](statistics.md#3-speed-cleaning).

### `filter_trajectory_speed(matched, *, drop_unmatched=True, drop_missing_speed=True, dwell_radius_m=25.0, dwell_min_s=120.0, max_speed=None, unit="mph") -> DataFrame`

Trajectory-aware cleaning for HMM-matched data (needs `traj_id`, `lon`, `lat`,
`time`, `speed_mps`). Drops missing-speed points and **stationary dwells**
(parked/idling) but **keeps slow-but-moving congestion**, so trafficability signal
is preserved. A dwell is a run of points within `dwell_radius_m` for ≥
`dwell_min_s`. See [statistics §8](statistics.md).

---

## Aggregation & peaks

### `aggregate_speeds(matched, *, block_hours=1, statistic="mean", output_unit="mph", by_edge=False, min_samples=1, dedup_intervals=True) -> DataFrame`

Bin by hour (`block_hours=1`) or N-hour block. `statistic` ∈
`{"mean", "median", "both"}`.

| Always present | `block_start_hour`, `block_label`, `n`, `unit` |
|----------------|------------------------------------------------|
| If mean | `mean_speed`, `std_speed`, `sem_speed`, `ci95_low`, `ci95_high` |
| If median | `median_speed`, `q1_speed`, `q3_speed`, `iqr_speed` |

`by_edge=True` aggregates per `(edge_id, bin)`. `min_samples` suppresses small
bins. Rows with `edge_id == -1` (unmatched) are excluded; network-wide runs
deduplicate `derive_speeds` rows on `interval_id` (disable with
`dedup_intervals=False`). Bins with a single observation report NaN
std/SEM/CI — one observation carries no spread information.

### `peak_analysis(aggregated, *, statistic="mean", n_peak=1, n_offpeak=1) -> dict`

Ranks network-wide bins; **slowest = peak**. Returns
`{"speed_column", "peak", "off_peak", "ranked"}`. Rejects per-edge or empty
aggregations with a clear error; warns when `n_peak + n_offpeak` exceeds the
number of bins (the lists would overlap).

---

## Speed assignment & routing

### `assign_speeds(network, matched, *, statistic="median", default_speed_mps=None, block_hours=24, target_hour=None) -> dict`

Computes a per-edge representative speed (optionally restricted to a time-of-day
block via `target_hour`) and writes `obs_speed_mps` (always m/s -- unlike
`aggregate_speeds`, there is no `output_unit`, since this output feeds the
router, not display) and `travel_time_s` onto the graph. Returns coverage counts.

### `classify_hours(matched, *, statistic="median", peak_hours=None, offpeak_hours=None, n_peak=None, n_offpeak=None, min_samples=1) -> dict`

Split the 24 hours into a peak and an off-peak block. Three modes:

1. **Explicit** — `peak_hours` / `offpeak_hours` (validated: hours 0–23,
   disjoint).
2. **Contiguous windows** — `n_peak` and `n_offpeak` (both required): the peak
   block is the contiguous `n_peak`-hour window (wrapping midnight) with the
   *lowest* network-wide speed, the off-peak block the contiguous
   `n_offpeak`-hour window with the *highest* speed disjoint from it. This is
   the user-selectable "N peak / N off-peak hours" mode.
3. **Auto median split** (default) — hours at or below the median hourly speed
   are peak.

Returns `peak_hours`, `offpeak_hours`, `threshold_speed_mps`, `source`
(`"override"`/`"window"`/`"auto"`), plus `peak_speed_mps`/`offpeak_speed_mps`
window scores in window mode. See [statistics §9](statistics.md).

### `assign_segment_speeds(network, matched, *, statistic="median", peak_hours=None, offpeak_hours=None, n_peak=None, n_offpeak=None, default_speed_mps=None, min_samples=1) -> dict`

Write three representative speeds per edge — `obs_speed_mps_{overall,peak,offpeak}`
with matching `travel_time_s_{…}` (and the plain `obs_speed_mps`/`travel_time_s` =
overall for back-compat). Hour blocks come from `classify_hours` (all three
modes, including `n_peak`/`n_offpeak` windows). Returns the hour blocks used,
`source`, and per-regime `coverage`.

### `Router(network, *, default_speed_mps=11.176)`

- `.route(origin, destination, *, mode="time", cost_func=None, snap=True, period="overall") -> dict`
  — `mode` ∈ `{"distance", "time", "cost"}`; `period` ∈
  `{"overall", "peak", "offpeak"}` selects the time regime (from
  `assign_segment_speeds`). With `snap=True`, `origin`/`destination` are
  `(lon, lat)` snapped to the nearest node. Returns `path`, `edge_ids`,
  `distance_m`, `travel_time_s`, `n_edges`, `n_edges_default`. Raises a clear
  `ValueError` on an empty network, a non-node endpoint (`snap=False`), or no
  path (distinguishing a one-way trap from a disconnected network).
  Routing picks the cheapest parallel edge under the active weight; a custom
  `cost_func(u, v, edges)` receives a dict keyed by edge key (= `edge_id`) of
  parallel-edge attribute dicts (the graph is a `MultiDiGraph`).
- `.nearest_node(lon, lat)` — nearest graph node to a coordinate.
- `.route_geometry_lonlat(result)` — route as a list of `(lon, lat)` for
  plotting/export.

---

## Network analysis

`roadtraffic.analysis` — road-network structure measures scoped to what a
trafficability/routing tool needs, not general urban-form analysis. All three
take a `Network` first, matching the rest of the package's free-function
convention.

### `edge_betweenness_centrality(network, *, weight, normalized=True, k=None, seed=None, write_attr=None) -> dict[int, float]`

Which roads carry a disproportionate share of shortest paths — a real
trafficability chokepoint, not just a topologically central one.

| Parameter | Meaning |
|-----------|---------|
| `weight` | Edge attribute to weight shortest paths by. **Required, no default** — pass `"travel_time_s"` for the chokepoint-by-real-travel-time reading (needs `assign_speeds`/`assign_segment_speeds` to have run on every edge), `"length_m"` for the purely geometric reading (always present, no pipeline needed), `None` for unweighted/topological betweenness, or any other numeric edge attribute you've computed. |
| `normalized` | If `True` (default), scores are in `[0, 1]`; if `False`, raw path counts. |
| `k` | Sample `k` source nodes instead of every node, for faster approximate results on large networks (exact computation is `O(VE)` or worse). Must be `>= 1`. |
| `seed` | Random seed for `k` sampling, for reproducible results. |
| `write_attr` | If given, also write each edge's score onto the graph under this name, so it flows into `to_geojson(keep_tags=[...])`/`plot_speed_map` for a chokepoint map. Off by default. Raises if it collides with a reserved or pipeline-owned attribute name. |

Raises `ValueError` if `weight` is a string and any edge lacks that
attribute (networkx would otherwise silently treat a missing weight as
`1.0`, corrupting the ranking — see [statistics §11](statistics.md)), if
`k < 1`, or on a `write_attr` collision.

### `network_stats(network, *, area_km2=None) -> dict`

Basic descriptive stats: `n_nodes`, `n_edges`, `n_physical_roads`,
`n_intersections`/`n_dead_ends` (by physical-road-degree, not raw directed
degree), `streets_per_node_avg`/`streets_per_node_counts`, `circuity_avg`
(always computed), plus `intersection_density_km2`/`edge_density_km2` (only
when `area_km2` is supplied — `Network` has no stored query-boundary polygon
to compute an area from automatically). See
[statistics §11](statistics.md) for the exact formulas and why.

### `connectivity_report(network) -> dict`

Diagnoses whether the network is one routable piece before you ever call
`Router.route`. Returns strongly-connected-component sizes and the actual
largest-component node/edge partition (not just a headline number), plus a
weakly- vs. strongly-connected distinction that separates a one-way trap
(`is_strongly_connected=False`, `is_weakly_connected=True`) from a genuinely
disconnected extract (`is_weakly_connected=False`) — the same diagnosis
`Router.route` already makes internally on a failed query, exposed here
proactively.

---

## Mapping / visualization

Requires `network.graph` edges to already carry `obs_speed_mps` (written by
`assign_speeds` / `assign_segment_speeds`).

### `to_geojson(network, path=None, *, directional=False, period="overall", speed_unit="mph", keep_tags=("name", "highway", "maxspeed", "oneway", "ref")) -> dict`

Export the speed-annotated network as a GeoJSON `FeatureCollection`, ready to
style by speed in QGIS / Kepler / Leaflet / Mapbox. `period` selects the
overall / peak / off-peak regime (from `assign_segment_speeds`).
`directional=False` (default) merges the two directed edges of a two-way road
into one feature (speed = mean of whichever directions were observed; travel
time recomputed from that merged speed so the two always agree);
`directional=True` keeps one feature per directed edge. Any unit alias
`SpeedUnit.parse` accepts works (`"km/h"`, `"m/s"`, ...); unknown units raise.
Writes to `path` if given, in addition to returning the dict. No extra
dependency.

### `plot_speed_map(network, path=None, *, period="overall", speed_unit="mph", cmap="RdYlGn", vmin=None, vmax=None, no_data_color="#cccccc", linewidth=2.0, figsize=(10, 10), dpi=150)`

Render a quick static PNG trafficability map coloured by per-edge speed (green
= fast, red = slow by default); unobserved edges are drawn in
`no_data_color`. Returns the `matplotlib` Figure, and saves to `path` if
given. Requires the `mapping` extra: `pip install roadtraffic[mapping]`.

---

## Units

`SpeedUnit` (`MPH`/`KPH`/`MPS`), `to_mps(value, unit)`, `from_mps(value, unit)`.
