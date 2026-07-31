# Mover mode screening

**Date:** 2026-07-31
**Status:** approved, not yet implemented
**Target version:** 0.5.0

## Problem

Customer GPS feeds are not always all vehicles. When a feed also carries people on
foot, their fixes match to the same roads and drag every road's speed score down.

The obvious remedy — a minimum-speed filter on observations — is the one that
breaks the study. A pedestrian at 3 mph and a vehicle crawling through a
chokepoint at 3 mph are indistinguishable at the level of a single measurement,
so any speed floor deletes both, removing exactly the congestion a trafficability
study exists to measure.

Mode is a property of the **mover**, not of the fix. A pedestrian is slow for
their entire track; a congested vehicle is slow on one segment and free-flowing
elsewhere in the same trip. Classification must therefore happen per trajectory,
and the verdict must apply to all of that trajectory's observations — including
its slowest.

## Prior evidence

Measured on a synthetic mixed dataset with known ground truth: 231 vehicles and
88 pedestrians on the same network, with the same pipeline run on the vehicle
rows alone as the baseline.

| strategy | Δ peak speed | per-edge MAE |
|---|---|---|
| no filter (contaminated) | −1.2 mph | 6.0 mph |
| drop observations < 12 mph | +5.0 mph | 3.1 mph |
| `require_quality` only | −2.1 mph | 3.9 mph |
| **mover p85 ≥ 6 mph** | **+0.2 mph** | **0.6 mph** |

At the chosen threshold the screen removed 87 of 88 pedestrians and kept 206 of
206 normal vehicles.

Three findings shape this design:

- **`require_quality` is not a mode filter.** It recovered under a third of the
  error. Walkers pass the quality screen once `min_baseline_m` merges their hops.
- **Contamination is worse per-road than in aggregate.** The network median fell
  17.1 → 10.7 mph, but per-edge error was 6.0 mph, because walkers concentrate on
  a few streets. A headline that looks merely low hides badly wrong individual
  roads.
- **A density valley alone does not identify a mode boundary.** A vehicle-only
  feed is itself multi-modal — residential free-flow, arterial free-flow and
  rush-hour crawl form separate humps. Valley depth alone nominated 13.2 mph on
  clean vehicle-only data, which would have deleted every congested vehicle in the
  study, invisibly.

## Scope

A new module `roadtraffic/modes.py` providing a per-mover mode classifier, plus
documentation, a rewired CLI, and integration into the customer report.

Explicitly out of scope: classifying cyclists as their own mode. Cyclists occupy
9–16 mph, which is where urban vehicles in traffic also sit. Speed alone cannot
separate them honestly, and a band wide enough to try would mark most congested
vehicles as ambiguous. This is documented as a limitation, not modelled.

## API

```python
mover_features(obs, *, percentile=85.0, unit="mph") -> DataFrame
suggest_mode_threshold(mover_speeds, *, unit="mph") -> float | None
classify_movers(obs, *, threshold, percentile=85.0, min_intervals=3,
                min_distance_m=0.0, unit="mph") -> DataFrame
filter_by_mode(obs, movers, *, keep=("vehicle",)) -> DataFrame
```

Mode labels are the string constants `"pedestrian"`, `"vehicle"`, `"unknown"`.

### `mover_features`

One row per `traj_id`, indexed by it. Columns: `n_intervals`,
`speed_p<pct>_<unit>`, `speed_median_<unit>`, `distance_m`, and `snap_dist_m`
when the input carries it.

`unit` does double duty: it names the speed columns and is the unit the
`threshold` argument of `classify_movers` is interpreted in. One unit for the
whole call, so a threshold can never be read against a differently-scaled column.
The percentile is formatted without a decimal point when integral and with the
fraction otherwise — `percentile=85.0` gives `speed_p85_mph`, `percentile=87.5`
gives `speed_p87.5_mph`.

Accepts any frame with `traj_id` and `speed_mps` — `derive_speeds` intervals, or
a matched frame carrying a logged speed column. **Deduplicates on `interval_id`
when that column is present.** `edge_observations` repeats each interval once per
edge traversed, so without this each mover's statistics would be weighted by how
many edges its hops crossed, biasing every mover toward its longest intervals.

Empty input returns an empty frame with the correct columns rather than raising.

### `suggest_mode_threshold`

Takes the per-mover percentile speeds and returns the speed at the density valley
separating a walking hump from a driving hump, or `None` when no such split
exists. The input is a 1-D array or Series of one speed per mover, expressed in
`unit` — typically the `speed_p85_<unit>` column of `mover_features`. Any index is
ignored; only the values are used. Non-finite and non-positive values are
dropped before estimation.

Estimated on **log** speed. A fixed relative bandwidth on raw speed is set by the
spread of the whole sample, and the driving hump runs out to motorway speeds —
wide enough to smooth away a valley only a couple of mph across. Log speed makes
the bandwidth scale-free.

Two guards, both required, both derived from observed failures:

1. **Prominence, not depth.** Candidate minima are ranked by
   `min(left_peak, right_peak) / valley`, and the density is evaluated well
   beyond the selection window in both directions. Ranking by absolute depth
   selects the lowest point of a monotone tail at the window edge, which is not a
   valley at all.
2. **The lower hump must be walkers.** The density peak below the candidate must
   fall between 0.6 and 2.5 m/s. Without this, the valley between two kinds of
   *driving* is accepted as a mode boundary — the 13.2 mph failure above.

Returns `None` when fewer than 20 movers are supplied, when scipy is unavailable,
or when no candidate satisfies both guards.

### `classify_movers`

Returns `mover_features`' table plus a `mode` column.

- `threshold`: a number in `unit`, or the string `"auto"`. `"auto"` delegates to
  `suggest_mode_threshold` and **raises `ValueError`** when it returns `None`.
  It never falls back to a default, because a silently chosen threshold that is
  wrong produces a study that looks correct.
- A mover is `unknown` when `n_intervals < min_intervals` or
  `distance_m < min_distance_m`; `vehicle` when its percentile speed is `>=
  threshold`; `pedestrian` otherwise. Evidence sufficiency is checked first.
- `unknown` never arises from speed ambiguity. A congested vehicle is a
  `vehicle`.

**No `require_quality` parameter, deliberately.** Classification uses every
interval including `quality=False`. The quality screen rejects intervals whose
displacement is small relative to GPS noise, which is what a walking mover
produces; filtering on it would delete the evidence identifying slow movers and
push them into `unknown`.

### `filter_by_mode`

Takes `obs` and a classification table, returns the rows belonging to movers
whose mode is in `keep`, index reset.

Defaults to `("vehicle",)`, so `unknown` is excluded unless requested. When the
result is empty, emits a `UserWarning` and returns the empty frame — a library
must not exit the process the way the current script does.

## Data flow

```
load_points → HMMMatcher → derive_speeds ─┬→ intervals ──→ classify_movers
                                          │                      ↓
                                          └→ edge_observations ─→ filter_by_mode
                                                                  ↓
                                                        cleaning → aggregation
```

Classify on `intervals` — one row per independent measurement. Filter
`edge_observations` — what the aggregators consume. Both carry `traj_id`, so the
verdict transfers.

## Errors

| condition | behaviour |
|---|---|
| `traj_id` or `speed_mps` missing | `ValueError` naming the fix, in the style of `filter_trajectory_speed` |
| `threshold="auto"`, no walking hump | `ValueError` — never a silent default |
| `filter_by_mode` removes every mover | `UserWarning`, empty frame returned |
| empty input to `mover_features` | empty frame with correct columns |

## Integration

### `scripts/mover_screen.py`

Keeps the CLI, the text histogram, and the `--out-points` / `--out-movers`
exports; reduces to argument parsing plus calls into the package. The KDE moves
into `suggest_mode_threshold`. ASCII rendering is presentation and stays in the
script.

### `scripts/customer_report.py`

- New `--mode-threshold <float|auto>`, **off by default**, and `--keep-unknown`.
- Screening runs between derive and clean; the progress banner becomes 8 steps.
- Data notes record the threshold, whether it was auto or explicit, how many
  movers and fixes were excluded, and the upward-bias caveat when screening is
  on — mirroring the existing `--require-quality` warning.
- **New deck section, "What the feed is made of":** movers and intervals by mode,
  and the per-mover percentile-speed histogram with the threshold marked. The
  histogram is the most useful artifact for explaining the method to a customer,
  so it belongs in the deliverable.

The `dataviz` skill must be loaded before writing that chart: three named
categories require a categorical palette, a different rule from the sequential
and diverging scales already in the file.

### Docs

- `docs/api.md`: a `## Mode screening` section after `## Cleaning`.
- `docs/methodology.md`: why per-mover rather than per-observation, how to choose
  a threshold, the comparison table above, and the gridlock limitation.
- `src/roadtraffic/__init__.py`: exports and module-docstring entries.

## Testing

`tests/test_modes.py`, written test-first. Two tests carry most of the weight.

1. **Single-mode regression.** A vehicle-only distribution must make
   `suggest_mode_threshold` return `None`. The version that nominated 13.2 mph
   would have deleted every congested vehicle, and nothing in the output would
   have looked wrong.
2. **The core invariant.** Plant a mover with a free-flow stretch *and* a 3 mph
   crawl; classify, filter, then assert the 3 mph rows survive. This is the
   property separating this from `filter_by_speed`, and the claim the method
   rests on.

Also:

- Bimodal input suggests a value inside the gap between the humps.
- `intervals` and `edge_observations` from one `derive_speeds` call yield
  identical verdicts (the dedup path).
- `min_intervals` and `min_distance_m` produce `unknown`; excluded by default,
  recoverable via `keep=("vehicle", "unknown")`.
- A threshold in mph and its kph equivalent classify identically.
- Missing columns, and `"auto"` with no walking population, raise.
- Emptying the frame warns rather than raises.

One end-to-end test in `tests/test_pipeline_e2e.py`: on planted mixed data, the
screened result lands closer to the vehicles-only truth than the unscreened one.
This validates the premise rather than the plumbing.

`tests/conftest.py` gains a `walk_along_road` helper beside the existing
`drive_along_road`, so fixtures stay compact.

## Known limitations

Stated in the module docstring and in `methodology.md`, not buried:

- **Gridlocked vehicles.** A vehicle whose entire track is congested never shows
  a fast stretch and is classified as a pedestrian. In testing, 12 of 25
  deliberately extreme gridlock cases were dropped with the walkers. The
  resulting bias is **upward** — the same direction as `require_quality` — because
  the vehicles wrongly dropped are the slowest. Mitigation is procedural: run the
  study screened and unscreened and compare, and report the gap as uncertainty.
- **Cyclists** are not separable from congested vehicles on speed and will
  generally be labelled `vehicle`.
- **Snap distance was tested as a second axis and rejected.** Alone it dropped 60
  real vehicles; combined with speed it readmitted walkers and made per-edge
  error worse (1.7 vs 0.6 mph). It is retained in `mover_features` as a
  diagnostic column but takes no part in the verdict.
- Where the feed carries device type, fleet ID or source app, that metadata
  beats inference and should be used instead.
