# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
(0.x: minor versions may contain breaking changes, noted below).

## [Unreleased]

### Changed

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

- `HMMMatcher(n_jobs=...)`: optional process-parallel decoding of independent
  trajectories (identical output to serial). Off by default -- measured on
  macOS, worker start-up and data-transfer overhead means serial stays faster
  up to at least ~2M points; see the docstring before enabling.
- `HMMMatcher(dist_cache_size=...)` to bound the shortest-path cache memory.
- `benchmarks/bench_matching.py`: reproducible synthetic-data throughput
  benchmark for both matchers.

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
