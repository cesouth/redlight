# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
(0.x: minor versions may contain breaking changes, noted below).

## [0.6.0] - 2026-08-06

### Added
- `network_stats` now measures the study area itself, so the per-km2 densities
  no longer need a hand-measured `area_km2`. The area comes from the network's
  own projected geometry -- the convex hull of every road vertex, already in
  metres -- so it needs no reprojection, no equal-area maths and no new
  dependency. The earlier docstring claimed no automatic area was possible
  because a `Network` stores no query boundary; that overlooked the projected
  geometry it does store. Verified against lattices of known extent: the hull
  recovers a 5x5 grid's 11.05 km2 as 11.11 and a 2x12 corridor's 7.60 km2 as
  7.64, both inside 0.5%.
  - New `area_method` argument: `"convex_hull"` (default), `"bbox"`, or `None`
    to disable detection and keep the previous opt-in behaviour.
  - New `area_method` key in the returned dict, recording where the area came
    from -- `"supplied"`, the detection method, or `None`. A supplied
    `area_km2` always wins over detection.
  - Detection declines (returns `None`, not `0.0`) when the network encloses
    no area -- empty, a single road, or exactly collinear roads -- since zero
    would make every density a division by zero.
  - The hull over-states a **non-convex** extract, filling the empty space the
    roads bend around: the inside of an L, a ring road's doughnut hole, the
    wedges of a radial network. For those, read the area as an upper bound and
    the densities as a lower bound, or pass `area_km2`. This is documented
    rather than corrected for, because no cheap test tells an empty wedge from
    an unmapped one.

### Changed
- **Behaviour change:** `intersection_density_km2` and `edge_density_km2` are
  now populated by default. They were `None` unless `area_km2` was passed.
  Pass `area_method=None` to restore the old behaviour.
- `scripts/customer_report.py` relabels its `Edges / km2` tile to
  `Road metres / km2`, which is what the figure has always been, and captions
  it with the area used and where it came from.
- **The package is renamed from `roadtraffic` to `redlight`.** Update imports:
  `import roadtraffic as rt` becomes `import redlight as rl`. The optional
  extras move with it -- `redlight[crs]`, `redlight[shapefile]`,
  `redlight[mapping]`, `redlight[docs]`. No API beyond the top-level name has
  changed, so a rename of the import is the whole migration. The old name was
  also already taken on PyPI by an unrelated project, which this resolves.
- `pyproj` is no longer a core dependency. WGS84, the 120 WGS84 UTM zones and
  Web Mercator are now projected in numpy, and geodesic distance uses
  Vincenty's inverse formula. Verified against PROJ 9.5.1 over a full UTM zone
  to 7.5 nm forward and 14 nm inverse, and to a few micrometres for geodesic
  distance. This removes 21 MB from an install and the `proj.db` class of
  environment conflicts along with it.
- `OGC:CRS84` source files are read natively. It is plain WGS84 lon/lat and is
  what GDAL/QGIS/`ogr2ogr` stamp on exported GeoJSON, but it carries no EPSG
  code, so it needs recognising by name rather than by code. `EPSG:4979`
  (WGS84 3D) likewise: the Z is dropped on read, leaving EPSG:4326.
- Reading a file in any other CRS -- national grids, non-WGS84 datums, raw-WKT
  CRS -- now requires the new `crs` extra: `pip install 'redlight[crs]'`.
  The same applies to a `metric_epsg=` override outside the UTM zones. Both
  raise an `ImportError` naming the extra rather than failing obscurely, and
  each names only the CRS *it* can handle natively (the metric path is UTM
  only -- Web Mercator is excluded there because it inflates ground distance
  by sec(latitude), some 55% at 50 deg N).

### Fixed
- `filter_trajectory_speed` no longer slows to a crawl on long stationary
  clusters -- the exact case it exists to detect. The dwell scan measures
  candidate points in blocks instead of one geodesic call at a time
  (a 4,000-point idle: minutes -> 3 ms), and a missing coordinate now ends the
  run it lands in rather than raising.
- `network_stats` computes all its great-circle distances in one vectorised
  call (19,320 edges: 1.31 s -> 0.05 s).
- UTM inverse longitudes in zone 60 are wrapped into [-180, 180) instead of
  coming back above +180, which is invalid GeoJSON per RFC 7946.

## [0.5.0] - 2026-08-01

### Added

- **Mode screening: `redlight.modes`.** A mixed GPS feed that also carries
  people on foot drags every road's score down, and the obvious remedy is the
  one that breaks the study. A pedestrian at 3 mph and a vehicle crawling
  through a chokepoint at 3 mph are indistinguishable in a single observation,
  so a `min_speed` filter deletes both -- removing exactly the congestion a
  trafficability study exists to measure.

  Mode is a property of the **mover**, not the fix: a pedestrian is slow for
  their whole track, while a congested vehicle is slow on one segment and
  free-flowing elsewhere in the same trip. The new functions classify whole
  trajectories and apply each verdict to *all* of that mover's observations, so
  a mover kept as a vehicle keeps its slow rows.

  - `mover_features(obs, *, percentile=85.0, unit="mph")` -- one evidence row
    per `traj_id`. Deduplicates on `interval_id` when present, so a long-format
    `edge_observations` frame does not weight each mover by how many edges its
    hops crossed.
  - `suggest_mode_threshold(mover_speeds, *, unit="mph")` -- the density valley
    separating walkers from drivers, or `None` when there is no walking
    population to split off. It never substitutes a default: a silently chosen
    wrong threshold produces a study that looks correct.
  - `classify_movers(obs, *, threshold, ...)` -- labels `pedestrian`,
    `vehicle` or `unknown`, where `unknown` means *insufficient evidence*
    (`min_intervals`, `min_distance_m`) and never speed ambiguity. A congested
    vehicle is a vehicle. `threshold="auto"` delegates to the suggester and
    raises rather than guessing.
  - `filter_by_mode(obs, movers, *, keep=("vehicle",))` -- applies the verdict.
    Warns and returns an empty frame when nothing survives; a library does not
    exit the process.

  Measured against vehicles-only ground truth on a mixed synthetic set (231
  vehicles, 88 pedestrians): per-edge error fell from 6.0 mph unscreened to
  0.6 mph, where a 12 mph observation floor overshot peak speeds by +5.0 mph
  and `require_quality` alone recovered under a third of the error.

  Known limitation, documented in the module and in `docs/methodology.md`: a
  vehicle gridlocked for its entire track never shows a fast stretch and is
  excluded with the walkers, biasing speeds **upward**. Run the study screened
  and unscreened, compare, and report the gap. Automatic threshold detection
  needs roughly an 8% pedestrian share before a walking hump is detectable.

- **`scripts/mover_screen.py`** -- a CLI to diagnose a feed's mover-speed
  distribution, pick a threshold from it, and write a screened points file.
- **`--mode-threshold` in `scripts/customer_report.py`**, off by default, with
  a "what the feed is made of" deck section and the upward-bias caveat recorded
  in the report's data notes.
- **Documentation examples are now executed by the test suite**
  (`tests/test_docs.py`). Every fenced Python block in `README.md` and
  `docs/*.md` runs against a real network and GPS sample, so a renamed
  parameter or changed return key fails the build instead of rotting quietly.

### Changed

- `examples/` restructured into topic folders, covering the current API:
  speed derivation from positions, cleaning, peak and day-type analysis, mode
  screening, congestion against posted limits, network structure, routing and
  mapping. The previous examples ran, but described the 0.2-era package and
  demonstrated `filter_by_speed(min_speed=...)` -- the anti-pattern this
  release's documentation argues against.
- Internal design documents moved out of `docs/` to `.plans/`, so they are no
  longer published with the user documentation.

### Fixed

- **`load_points` no longer discards the source's extra columns**, which made
  `derive_speeds(pos_accuracy_col=...)` unreachable in practice: the loader
  built a canonical frame (`point_id`, `traj_id`, `lon`, `lat`, `time`,
  `speed_mps`) and dropped everything else, so a per-point horizontal-accuracy
  column was gone by the time `derive_speeds` looked for it and every point
  silently fell back to the assumed `default_pos_sigma_m`. The per-point error
  model was therefore inert for anyone loading data the documented way -- on a
  sample dataset, restoring it tightens the median speed uncertainty by 36%
  (0.707 -> 0.451 m/s). Extras are now attached *before* the row-dropping
  passes, so they are filtered along with their own rows; re-attaching them
  afterwards is not sound, because `point_id` is renumbered after the drop and
  so cannot serve as a key back into the source. New `keep_cols=` selects a
  subset (`[]` restores the old lean frame), and a source column colliding with
  a canonical name is preserved as `<name>_src`.

## [0.4.0] - 2026-07-27

### Added

- **`congestion_report`** -- observed speed as a fraction of the **posted**
  limit, per edge and optionally per time block. The standard level-of-service
  framing, and the thing raw speeds cannot tell you: in one worked case a
  motorway at 17.9 mph and a side street at 20.1 mph look comparable until you
  divide by their limits, at which point the motorway is running at 0.28 of
  what it should and the side street at 0.81. Ratios are **not clipped at
  1.0** (traffic above the limit is a real finding) and edges with no usable
  limit report NaN rather than being dropped, so coverage gaps stay visible.
- **`require_quality=True`** on `aggregate_speeds`, `classify_hours`,
  `assign_speeds`, `assign_segment_speeds` and `day_type_report`. `derive_speeds`
  has always flagged intervals it does not trust -- below the SNR floor, bad
  `dt`, bad snap distance -- and `speeds.py` has always said "keep `quality`
  rows for aggregation", but **nothing in the package consumed the flag**:
  filtering it was left as something every caller had to remember by hand, and
  forgetting it silently pulled known-bad intervals into headline numbers.
- **Inverse-variance weighting for the mean** — `aggregate_speeds(...,
  weight_by_variance=True)`. `derive_speeds` has always reported a
  per-interval `speed_var`, and the docs have always said it "supports
  inverse-variance weighting downstream", but nothing consumed it: every
  observation counted equally, so a noisy 3-second hop weighed as much as a
  clean 90-second baseline. Now `mean_speed` is `sum(x/var) / sum(1/var)` and
  `sem_speed` is `sqrt(1 / sum(1/var))` — a propagated measurement uncertainty
  rather than a spread-over-`sqrt(n)`, and therefore defined even for a single
  observation. `std_speed` deliberately stays the unweighted sample spread: the
  spread is a property of the traffic, the weights of the instrument. Rows with
  a non-finite or non-positive `speed_var` are dropped with a warning. See
  `docs/statistics.md` for what the weighted SEM does and does not claim.
- **`derive_speeds(..., interval_id_start=)`** so the output of separate runs
  can be given non-overlapping interval ids and safely concatenated.

### Changed

- The cleaning prologue shared by `aggregate_speeds`, `assign_speeds` and
  `assign_segment_speeds` (drop unmatched, drop missing speeds, apply `days=`)
  is now one `_prepare` helper rather than three copies that had begun to
  diverge, so those entry points cannot disagree about what counts as a usable
  observation. `_usable_speed` likewise moved to `units.py`, shared by the
  router's posted-limit fallback and `congestion_report`.

### Fixed

- **Colliding `interval_id`s are now detected instead of silently halving the
  data.** `derive_speeds` numbers intervals from 0 on every call, and
  network-wide aggregation deduplicates on `interval_id` — so concatenating
  two runs made distinct intervals collide, and the dedup kept one row per
  colliding group and dropped the rest. Two equal-sized runs lost ~50% of
  their observations with no error and no warning. `aggregate_speeds` now
  cross-checks each id against the measurement it names (`traj_id`, `time`,
  `speed_mps`) and raises with the available fixes. Legitimate duplication —
  one interval attributed to several edges — is unaffected, as is
  `dedup_intervals=False`.

- **A non-finite `maxspeed_mps` no longer poisons route travel times.** The
  posted-limit fallback screened the attribute with a plain truthiness test,
  but **NaN is truthy** -- so a `maxspeed_mps` arriving from a caller's own
  GeoJSON/Shapefile (an empty numeric field reads as NaN, and networks that
  already carry the attribute skip parsing entirely) made every travel time
  derived from it, and the route total summing them, silently NaN. Infinity
  would have made the edge free to cross. Non-finite, non-positive and
  non-numeric limits now fall through to `default_speed_mps`.
- **`days=` accepts numpy integers.** `isinstance(days, int)` is False for
  `numpy.int64`, so a value taken straight from pandas
  (`df["time"].dt.dayofweek.unique()[0]`) was treated as an iterable and
  failed with `'numpy.int64' object is not iterable` -- an error naming
  neither `days` nor the cause.
- Corrected the advertised test counts in `README.md` and `docs/methodology.md`
  (long stale at 136 and 122), and the import path documented for
  `parse_maxspeed`, which is not re-exported on the top-level namespace.

- **OSM posted speed limits are now parsed and used as a per-edge routing
  fallback.** `redlight.osm.parse_maxspeed` converts an OSM `maxspeed` tag
  to m/s --
  correctly reading a bare `maxspeed=50` as **km/h** per the OSM spec (reading
  it as mph would overstate the limit by ~61%) while honouring an explicit
  `mph`/`km/h` suffix. Values that carry no unambiguous number (`"none"`,
  `"walk"`, `"signals"`, country defaults like `"RU:urban"`, multi-values like
  `"50;30"`) parse to `None` rather than an invented constant. Any network
  whose source data carries a `maxspeed` property -- Overpass, or a GeoJSON
  exported from OSM -- now gets a numeric `maxspeed_mps` edge attribute
  alongside the untouched raw tag.
  `Router` uses it (new `use_maxspeed=True`) to estimate **unobserved** edges
  at their posted limit instead of at one global `default_speed_mps` for every
  road. Previously a motorway with no GPS coverage was routed at the same
  25 mph as a side street, which made time-mode routing on unobserved networks
  degenerate into distance-mode. Measured travel times always take precedence,
  and a limit-estimated edge is still counted as non-observed in
  `n_edges_default` -- the posted limit is an assumption, never a measurement.
  Pass `use_maxspeed=False` for the previous uniform-fallback behaviour.
- **Day-of-week support across temporal aggregation.** `aggregate_speeds`,
  `classify_hours`, `assign_speeds`, and `assign_segment_speeds` all take a new
  optional `days=` argument to restrict analysis to particular weekdays before
  binning -- the fix for hour-of-day bins otherwise silently pooling, say,
  Tuesday 09:00 with Saturday 09:00. Accepts a preset (`"weekday"` = Mon-Fri,
  `"weekend"` = Sat-Sun, `"all"`), a day name or number (`"Mon"`/`0` ...
  `"Sun"`/`6`), or any iterable of those; `days=None` (the default) preserves
  the previous every-day behaviour exactly, so this is fully back-compatible.
  Weekdays are read on the stored **local clock** (load with `tz=`).
- **`day_type_report`** -- a ready-made weekday-vs-weekend (or any custom
  day-grouping) comparison built on `days=`. For each group it reports the
  network-wide overall speed, the full per-hour/per-block profile, and the
  peak/off-peak blocks, then lines the groups up block-by-block into a single
  `comparison` DataFrame carrying the per-block `delta_speed`/`delta_pct` -- so
  the change between weekday and weekend traffic is one printable/plottable
  table. A day-type with no observations (e.g. a dataset that never sampled a
  weekend) is reported honestly with `n=0`/NaN and a warning, not an error.
- **`redlight.analysis`**: a new module of road-network structure
  measures scoped to what a trafficability/speed-routing tool actually
  needs (not general urban-form analysis):
  - **`edge_betweenness_centrality`** -- travel-time-weighted (or any other
    edge attribute) betweenness centrality, identifying the roads carrying
    a disproportionate share of shortest paths ("chokepoints"), not just
    topologically central ones. Guards a networkx landmine: a missing *or*
    explicitly `None`/`NaN` weight attribute is silently treated as `1.0` by
    networkx itself (a missing attribute verified against an explicit `1.0`
    weight -- identical output), which would make unobserved edges look
    artificially cheap and corrupt the ranking; this now raises a clear
    `ValueError` instead of computing a silently wrong answer (or, for the
    `None`/`NaN` case, crashing with a raw `TypeError` deep inside
    networkx's own Dijkstra -- caught reviewing real OSM data reloaded via
    `to_geojson` -> `from_geojson`, which round-trips an unobserved edge's
    `travel_time_s: null` property as a literal `None` attribute, present
    but exactly as useless as absent). Optional `write_attr=` writes scores
    onto the graph so they flow into `to_geojson`/`plot_speed_map` for a
    chokepoint map.
  - **`network_stats`** -- circuity, streets-per-node, and
    intersection/dead-end counts, always available; intersection/edge
    density additionally when an `area_km2` is supplied (there's no
    query-boundary polygon retained by `Network` to compute one from
    automatically, unlike `osmnx`).
  - **`connectivity_report`** -- largest strongly-connected-component size
    and fraction, plus the actual node/edge partition (so a caller can
    isolate the routable core, not just read a number), and a
    weakly-vs-strongly-connected distinction that separates a genuinely
    disconnected extract from a network that's merely a "one-way trap" --
    the same diagnosis `Router.route()` already makes internally on a
    failed query, exposed proactively.

## [0.3.0] - 2026-07-15

This release follows a second full-package review (12 correctness defects, all
reproduced before being fixed), an expanded parameter/attribute documentation
pass, and the fiona -> pyogrio migration for Shapefile/GeoPackage reading.

### Fixed

- **A single NaN speed could silently empty an entire `filter_by_speed` call.**
  `_mad_mask` computed the median/MAD over the raw array; `np.median`
  propagates NaN, so one missing speed poisoned every row's modified Z-score
  and the whole batch was dropped as "outliers". NaN entries are now excluded
  from the median/MAD computation and never pass the mask themselves.
- **Speed-unit inference only recognized the exact `mph`/`kph`/`mps` tokens.**
  A column named e.g. `speed_kmph` or `speed_km/h` silently fell back to the
  mph conversion (a ~61% overstatement). `_infer_speed_unit` now recognizes
  every alias `SpeedUnit.parse` accepts.
- **Numeric-epoch timestamps never warned about an assumed clock**, even
  though they're UTC-aware by construction and an equivalent ISO `Z` string
  did warn. `load_points(timestamp_unit=..., tz=None)` now warns too.
- **`speed_unit=` contradiction warnings fired even under `derive_speed=True`**,
  where the speed column (and therefore its unit) is never read — a
  misleading warning about a setting with no effect.
- **A genuine 0 m/s (gridlock) was treated identically to "no data"** in
  `assign_speeds`, `assign_segment_speeds`, `to_geojson`, and
  `plot_speed_map` — the exact condition a trafficability tool exists to
  surface was silently hidden or fell back to the default speed. All four now
  treat only a missing observation as "no data"; travel time at 0 m/s is left
  undefined rather than raising `ZeroDivisionError`.
- **`to_geojson(keep_tags=...)` could silently overwrite computed properties**
  (`speed`, `length_m`, etc.) when a source tag shared that name. Colliding
  tags are now copied under `"<tag>_src"` instead (matching how network
  loading already handles the analogous collision).
- **`Router(mode="cost")` could report the wrong parallel edge.** Path
  reconstruction picked the parallel edge with the lowest overall travel
  time, not the one `cost_func` actually costed the path over — so
  `distance_m`/`travel_time_s`/geometry could describe a different road than
  the one Dijkstra used. It now re-evaluates `cost_func` per parallel edge to
  find the one it actually preferred.
- **`Router(default_speed_mps=0)` raised a raw `ZeroDivisionError`**, and a
  negative value was silently accepted and produced negative travel times.
  Both now raise a `ValueError` from the constructor.
- **`derive_speeds`'s quality gate let a one-sided NaN `snap_dist_m` pass.**
  Only a NaN at *both* endpoints of an interval failed quality; one finite +
  one NaN snap passed even though one endpoint's match quality was entirely
  unknown. Both endpoints must now be finite (and within tolerance) whenever
  the column is present.
- **`min_baseline_m` wasn't actually guaranteed.** A trajectory that ran out
  of points before its merged interval reached the requested baseline still
  emitted that interval as if the guarantee held. It's now flagged
  `quality=False` when the baseline wasn't reached.
- **HMM transitions across a GPS dropout compared the wrong points.** After
  one or more candidate-less fixes, the transition distance used the raw
  step from the *immediately preceding* (off-network, noisy) fix instead of
  the last fix that actually anchored a Viterbi state — corrupting the
  transition score exactly where a nearby intersection makes the wrong
  candidate plausible. The straight-line step is now measured from the true
  last-anchored fix.
- **HMM transition route distance omitted the partial-edge terms** that
  `redlight.speeds`'s on-road distance measurement already accounts for
  (remaining length past the previous snap; length up to the current snap) —
  a systematic bias that could favor a geometrically wrong candidate. The
  transition distance is now the same three-piece sum used elsewhere.
- **The bounded shortest-path cache's LRU eviction was broken for refreshed
  entries.** Recomputing a cached source at a larger cutoff didn't move it to
  the recently-used end (`OrderedDict` reassignment doesn't reorder), so an
  actively-used source could be evicted right after being recomputed.
  Refreshes now bump recency like cache hits do.

### Changed

- **Breaking:** `assign_speeds`'s `output_unit` parameter is removed — it was
  never applied (the function always wrote `obs_speed_mps` in m/s, as its
  docstring already said, for router compatibility). `aggregate_speeds`'s
  `output_unit` is unaffected and still works.
- **Shapefile/GeoPackage reading now uses `pyogrio` instead of `fiona`.**
  `fiona` is the legacy I/O path in the geopandas ecosystem now that
  geopandas itself defaults to `pyogrio`; it also has materially weaker
  Windows wheel coverage. `Network.from_file()`'s behavior and output are
  unchanged (still splits `MultiLineString` into per-part edges, still
  reprojects to WGS84 before building the graph, still raises the same
  errors) -- only the backend and the `shapefile` extra's declared
  dependency changed. **Breaking (scoped):** the `shapefile` extra now
  requires Python >= 3.10 (pyogrio dropped 3.9 support in its 0.12 release);
  the core package's Python 3.9+ floor is unaffected -- GeoJSON and
  `from_overpass` still work on 3.9.
- CI moved to GitLab CI (`.gitlab-ci.yml`, same ruff + pytest matrix on
  Python 3.9/3.11/3.13) ahead of the repository's migration to GitLab; the
  GitHub Actions workflow was removed. The `shapefile` extra is now installed
  on the 3.11/3.13 legs (previously not installed in CI at all); the 3.9 leg
  skips it and the pyogrio-dependent tests self-skip there.

- **Matching is 10-30x faster, with identical results.** NearestMatcher is
  fully vectorised (one batch KDTree query + one vectorised
  foot-of-perpendicular pass for the whole point set): ~13k -> ~400k points/s
  on the benchmark grid. HMMMatcher's transition distances now run on scipy's
  C Dijkstra over a CSR adjacency (`Network.csgraph()`) with an LRU-bounded
  per-source cache reused across steps and trajectories: ~7.5k -> ~90k
  points/s. Regression tests pin the new paths to the old per-point /
  networkx results.
- New `Network` batch methods: `nearest_edges`, `candidate_edges_batch`,
  `csgraph()` (lazy CSR + node index maps). The scalar `candidate_edges`
  is unchanged.

### Added

- **`Network.from_file(..., layer=...)`**: select a layer by name or index
  when reading a multi-layer GeoPackage (previously the loader had no way to
  address anything but the file's default layer).
- Regression tests for `Network.from_file()` reading `.shp`/`.gpkg`,
  including CRS reprojection and multi-layer selection -- this loader had no
  dedicated test coverage before.
- `HMMMatcher(n_jobs=...)`: optional process-parallel decoding of independent
  trajectories (identical output to serial). Off by default -- measured on
  macOS, worker start-up and data-transfer overhead means serial stays faster
  up to at least ~2M points; see the docstring before enabling.
- `HMMMatcher(dist_cache_size=...)` to bound the shortest-path cache memory.
- `benchmarks/bench_matching.py`: reproducible synthetic-data throughput
  benchmark for both matchers.
- `docs/methodology.md`: a methodology paper defending the package's
  methods (HMM/Viterbi vs nearest-edge matching, on-road interval speeds,
  robust aggregation, peak/off-peak detection) with citations, limitations,
  and ground-truth experiments; figures and numbers are regenerated by
  `scripts/paper_experiments.py` (seeded, ~2-3 min).

## [0.2.0] - 2026-07-02

This release follows a full-package code review (multi-angle, with every
finding empirically reproduced before being fixed) and a professionalization
pass: test suite, CI, linting, packaging modernization, and this changelog.

### Fixed

- **`HMMMatcher` crash on off-network leading fixes.** A trajectory whose
  first fix had no candidate edge within `max_dist` crashed the whole
  `match()` call (`ValueError: max() arg is an empty sequence`). The chain now
  restarts at the first fix that has candidates.
- **`derive_speeds` silently returned empty results** for input without a
  trajectory id (the documented position+time-only pipeline): `groupby` was
  dropping the null trajectory key. Id-less input is now treated as a single
  trajectory.
- **Parallel roads no longer overwrite each other.** The road graph is now a
  `networkx.MultiDiGraph` keyed by `edge_id`; previously a second road between
  the same two endpoints silently replaced the first one's attributes while
  the spatial index kept both ids — producing wrong lengths in on-road
  distance math, losing the shadowed road's observations in speed assignment,
  and omitting its geometry from exports.
- **`oneway=-1` is now handled with OSM semantics** (one-way *against* the
  digitized direction). It was previously treated as a forward one-way,
  creating the edge in exactly the illegal direction of travel.
- **Speed-unit mis-scaling on unit-named columns.** A `speed_kph` /
  `speed_mps` column was auto-detected by name but converted with the default
  mph factor (a silent ×1.609 error). The unit embedded in the column name is
  now used; an explicit `speed_unit=` wins, with a warning on contradiction.
- **Timezone handling.** UTC-stamped GPS data produced peak/off-peak hours on
  the UTC clock, and mixed UTC offsets (DST transitions) crashed
  `load_points` on modern pandas. New `tz=` parameter converts aware
  timestamps and numeric epochs to the study area's local clock; aware input
  without `tz=` warns instead of silently mis-binning.
- **Order-dependent on-road distances.** The bounded-Dijkstra cache in
  `derive_speeds` reused results computed with a larger cutoff without
  re-applying the smaller requested cutoff, silently bypassing the
  `max_route_dist_factor` detour guard depending on query order.
- **Fabricated HMM matches.** Fixes with no candidate edge were emitted with
  the previous state's `edge_id` and `snap_dist_m = NaN`; they are now
  reported as `edge_id = -1` (the documented contract) while the Viterbi
  state still carries across the gap.
- **Quality gate accepted the worst matches.** A non-finite snap distance now
  *fails* the `derive_speeds` quality check (inputs with no `snap_dist_m`
  column at all keep the benefit of the doubt); the all-NaN `RuntimeWarning`
  is gone.
- **Inflated network-wide sample sizes.** `derive_speeds` deliberately
  replicates each interval once per traversed edge (and direction) for
  per-edge statistics; network-wide `aggregate_speeds`/`classify_hours` now
  deduplicate on the new `interval_id` column so one interval counts once
  (SEM/CI are no longer shrunk by fabricated n).
- **Unmatched points polluted statistics.** `aggregate_speeds` and
  `classify_hours` now exclude `edge_id == -1` sentinel rows.
- **Zero-width confidence intervals.** Bins with one observation reported
  std = SEM = 0 and a zero-width 95% CI; they now report NaN.
- **Silently inconsistent matcher outputs.** Rows with a missing trajectory
  id vanished from `HMMMatcher` output (but not `NearestMatcher`'s), so the
  two matchers returned different row counts for identical input;
  `load_points` now drops such rows with a specific warning.
- **Loader crashes on legal input.** Source properties named `edge_id`,
  `length_m` or `geometry` crashed network loading (duplicate keyword); they
  are preserved under an `_src` suffix with a warning. A network whose
  features are all degenerate now raises a clear error instead of an
  `IndexError`.
- **Silently dropped roads.** Closed-loop features (roundabouts digitized as
  one closed way) are still unmodellable but are now counted and warned
  about instead of vanishing.
- **Mapping export inconsistencies.** Merged two-way features reported the
  mean speed of both directions but the travel time of an arbitrary one;
  travel time is now recomputed from the merged speed. Unknown speed-unit
  strings raised nowhere and exported m/s values under a wrong label; units
  now go through `SpeedUnit.parse` (aliases like `"km/h"` work, unknown units
  raise). A genuine 0 m/s no longer renders as "no data".
- **Misleading coverage counts.** `assign_speeds` counted edges written with
  the fallback `default_speed_mps` as "observed" when their statistic was
  non-positive.
- **Misleading errors.** `peak_analysis` rejects per-edge and empty
  aggregations with actionable messages (previously it silently ranked
  (edge, hour) cells, or blamed the wrong cause), and warns when
  `n_peak + n_offpeak` exceeds the number of bins. Overlapping
  `peak_hours`/`offpeak_hours` overrides now raise instead of double-counting
  hours in both regimes. The degenerate median split (all hours tied) warns.
- HMM documentation now describes the real transition-cutoff formula
  (`max(factor × step, max_dist × 4)`), the same-edge exemption, the
  saturating no-predecessor penalty, and the restart semantics.

### Added

- **User-selectable contiguous peak / off-peak windows**: `classify_hours`
  and `assign_segment_speeds` accept `n_peak=` / `n_offpeak=` — the peak
  block is the contiguous `n_peak`-hour window (wrapping midnight) with the
  lowest network-wide speed; the off-peak block is the fastest disjoint
  `n_offpeak`-hour window.
- **`Network.from_overpass(bbox)`** and the `redlight.osm` module: fetch
  an OSM road network for a bounding box via the Overpass API using only the
  standard library (no new dependencies).
- **Per-regime map export**: `to_geojson` and `plot_speed_map` accept
  `period={"overall", "peak", "offpeak"}`, so peak-hour trafficability maps
  are available through the public API.
- **`interval_id`** column in both `derive_speeds` outputs, enabling
  deduplication of the per-edge row replication.
- **`tz=` parameter** on `load_points` (see Fixed).
- **Public `Network` helpers**: `edge_length`, `edge_geometry`, `edge_data`,
  `edge_coords_lonlat`, `road_edge_ids`, `edges_between` — downstream modules
  and user code no longer need private attributes.
- **Offline test suite** (94 pytest tests, ~2 s) covering the full pipeline,
  including regression tests for every fixed defect.
- **GitHub Actions CI** (lint + tests on Python 3.9/3.11/3.13) and **ruff**
  linting configuration.
- This changelog.

### Changed

- **Breaking:** `Network.graph` is a `networkx.MultiDiGraph` (was `DiGraph`).
  Code using the documented API is unaffected; code reaching into the graph
  directly must handle edge keys, and a custom `Router` `cost_func` now
  receives `(u, v, edges)` where `edges` is a dict of parallel-edge attribute
  dicts.
- **Breaking (edge case):** `load_points` drops rows whose trajectory-id cell
  is empty (with a warning) instead of keeping them inconsistently.
- `load_points(speed_unit=...)` defaults to `None` (infer from the column
  name, else mph) instead of `"mph"`. Behavior is unchanged unless your speed
  column name embeds a different unit — which was precisely the silent-error
  case.
- Packaging modernized: PEP 639 SPDX license metadata, single-sourced version
  (`redlight.__version__`), project URLs point at the actual repository,
  Python 3.9–3.13 classifiers, `pytest`/`ruff` dev extras.
- One shared WGS84 `Geod` (`redlight._geo`) and shared input validators
  replace duplicated definitions; the redundant unit-conversion table in
  `mapping` was removed.

### Known limitations

- Closed-loop ways are skipped (split them before loading).
- Node identity uses 7-decimal coordinate rounding; features meant to connect
  must share exact endpoint coordinates in the source data.
- The matchers and `derive_speeds` use per-point Python loops; they are fine
  for the tens-of-thousands-of-points scale and will be vectorised in a
  future performance-focused release.

## [0.1.0]

Initial release: GeoJSON/Shapefile network loading, CSV/GeoJSON GPS point
ingestion with unit handling, nearest-edge and HMM/Viterbi map matching,
on-road speed derivation, speed cleaning (bounds, MAD, dwell-aware),
hourly/block aggregation with mean/median statistics, peak/off-peak
detection, per-edge speed assignment, time/distance/cost routing, and
GeoJSON/PNG trafficability maps.
