# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
(0.x: minor versions may contain breaking changes, noted below).

## [Unreleased]

### Added

- **`roadtraffic.analysis`**: a new module of road-network structure
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
  `roadtraffic.speeds`'s on-road distance measurement already accounts for
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
- **`Network.from_overpass(bbox)`** and the `roadtraffic.osm` module: fetch
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
  (`roadtraffic.__version__`), project URLs point at the actual repository,
  Python 3.9–3.13 classifiers, `pytest`/`ruff` dev extras.
- One shared WGS84 `Geod` (`roadtraffic._geo`) and shared input validators
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
