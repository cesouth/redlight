# Spec drift — approved specs vs shipped code — findings

**Pass:** Task 1
**Date:** 2026-08-17
**Commit reviewed:** `2aa8720`
**Scope:** `.plans/2026-07-31-mover-mode-screening-design.md`,
`.plans/2026-07-31-mover-mode-screening.md`, `.plans/2026-08-04-drop-pyproj.md`
read as the authority; verified against `src/redlight/modes.py`,
`src/redlight/_geo.py`, `src/redlight/_proj.py`, `src/redlight/network.py`,
`src/redlight/__init__.py`, plus the integration surfaces the specs name
(`scripts/mover_screen.py`, `scripts/customer_report.py`, `tests/test_modes.py`,
`tests/test_network.py`, `tests/test_geo.py`, `tests/test_proj.py`,
`tests/test_no_pyproj.py`, `tests/conftest.py`, `tests/test_pipeline_e2e.py`,
`pyproject.toml`, `.github/workflows/ci.yml`, `docs/api.md`,
`docs/methodology.md`, `docs/statistics.md`, `docs/index.md`, `README.md`,
`CHANGELOG.md`).
**Method:** `git log --oneline`, `git show --stat` on the two post-spec fix
commits, plus seven scratch scripts run under `.venv/bin/python` in the
scratchpad (`t1_pyproj_api.py`, `t2_invariant.py`, `t3_proj_accuracy.py`,
`t4_hump_share.py`, `t5_misc.py`, `t6_empty_cols.py`, `t7_dataflow.py`). Full
suite re-run to confirm the baseline: **375 passed**, `git status --porcelain`
empty. No source file changed; nothing committed.

## Summary

Both specs are implemented essentially in full — every function, constant, test,
doc section, script rewire and packaging change they call for is present, and
all four binding constraints hold under execution, including the central
methodological claim, which I proved on real pipeline output rather than by
reading. The drift is almost entirely in one direction: the code was
**hardened after** the specs were written (`c9a92bb`, `a03733f`, `367e683`) and
the spec documents were never updated, so eight requirements now describe
something narrower or less accurate than what ships. The single worst item is
not drift at all but a defect the drift exposed: `mover_features` emits a
different column set for an empty input than for a non-empty one, so a caller
reading the documented `snap_dist_m` column gets a `KeyError` the moment a feed
filters down to nothing.

---

## Binding constraints — all four hold

| Constraint | Result | Evidence |
|---|---|---|
| `grep -rn "pyproj" src/` hits only function-local deferred imports in `network.py` | **HOLDS** | The only `import pyproj` in `src/` is `network.py:81`, inside `_require_pyproj`. `grep -rn "^import pyproj\|^from pyproj" src/` returns nothing. Remaining hits are docstrings/comments in `_proj.py` and `network.py` — which is what the spec's own Task 5 Step 6 permits ("only comment/docstring mentions and no module-level import") — plus `src/redlight.egg-info/`, a gitignored editable-install artifact (`.gitignore:3`), not source. |
| `crs_metric.to_epsg()`, `project_points()`, `_transformer_fwd/_inv.transform()` all still work unchanged | **HOLDS** | `t1_pyproj_api.py`, below. All four work on scalars and arrays, and `pyproj` is never imported. |
| Classification is per-trajectory; the verdict applies to ALL of a mover's observations including its slowest | **HOLDS** | `t2_invariant.py` and `t7_dataflow.py`, below. Proved on `derive_speeds` output: a vehicle and a pedestrian sharing an *identical* 1.401 m/s observation get opposite verdicts, and all 18 of the vehicle's walking-pace rows survive the filter. |
| `modes.py` depends only on `units` + scipy | **HOLDS** | Its entire import list is `warnings`, `numpy`, `pandas`, `from .units import SpeedUnit, from_mps, to_mps`, and a deferred `from scipy.stats import gaussian_kde` at `modes.py:187`. No other intra-package import. |

```
$ .venv/bin/python t1_pyproj_api.py
crs_metric           : <UtmCrs EPSG:32633 WGS 84 / UTM zone 33N>
crs_metric.to_epsg() : 32633
crs_metric.name      : WGS 84 / UTM zone 33N
project_points       : [500000.  500716.67075252] [5538630.70286748 5538630.75077691]
_transformer_fwd     : 500000.0 5538630.702867475
_transformer_inv     : 15.0 50.00000000000011
fwd array            : [500000.  535460.44581219] [5538630.70286748 5594344.78612974]
pyproj in sys.modules: False
```

(500000.000000 / 5538630.702867 is the PROJ 9.5.1 value the drop-pyproj spec
pinned at line 355; it is reproduced exactly.)

```
$ .venv/bin/python t7_dataflow.py          # real derive_speeds output
intervals cols        : ['traj_id', 'interval_id', 'speed_mps']
edge_observations cols: ['traj_id', 'interval_id', 'speed_mps']
intervals rows 110, edge_observations rows 220

          n_intervals  speed_p85_mps        mode
traj_id
ped0              24       1.401372  pedestrian
ped1              24       1.401372  pedestrian
ped2              24       1.401372  pedestrian
veh0               7      12.011753     vehicle
veh1               7      12.011753     vehicle
veh2               7      12.011753     vehicle
veh_jam           17      13.012710     vehicle

edge_observations kept traj_ids: ['veh0', 'veh1', 'veh2', 'veh_jam']
veh_jam rows kept: 34, min speed kept: 1.401 m/s
veh_jam rows below 2 m/s kept: 18  (of 18 in the input)

VERDICT transfers intervals -> edge_observations, and the congested vehicle's
walking-pace rows survive: True
```

---

## Requirement matrix — `2026-08-04-drop-pyproj.md`

| # | Requirement | Status |
|---|---|---|
| T1 | `_geo.geodesic_distance` replaces `pyproj.Geod`, Vincenty inverse, raises on non-convergence | IMPLEMENTED AS SPECIFIED — `_geo.py` matches the spec's code block line for line |
| T1 | Three call sites rewired (`points.py`, `cleaning.py`, `analysis.py`); `GEOD_WGS84` retired | IMPLEMENTED AS SPECIFIED |
| T1 | `tests/test_geo.py` with pinned PROJ references + the Geoscience Australia line | IMPLEMENTED AS SPECIFIED |
| T2 | `_proj.py`: `utm_epsg_to_zone`, `utm_forward`, `utm_inverse`, `web_mercator_inverse`, `parse_epsg`, `EPSG_WGS84`, `EPSG_WEB_MERCATOR` | IMPLEMENTED AS SPECIFIED |
| T2 | `UtmCrs` (`.to_epsg()`, `.name`, `.zone`, `.north`), `UtmTransformer.transform`, `utm_crs_and_transformers` | IMPLEMENTED AS SPECIFIED |
| T2 | Krüger series; spec's code truncates `delta` at n^4 | IMPLEMENTED BUT LATER CHANGED — `delta` now runs to n^5 (`c9a92bb`). **F-1.5** |
| T2 | Accuracy: 9 nm forward, 0.13 mm round-trip | IMPLEMENTED DIFFERENTLY — forward still 7.5 nm; round-trip now 0.000016 mm. **F-1.5** |
| T2 | `utm_inverse` returns lon/lat | IMPLEMENTED DIFFERENTLY — now wraps into `[-180, 180)`. **F-1.7** |
| T2 | `tests/test_proj.py` | IMPLEMENTED AS SPECIFIED (extended) |
| T3 | Module-level `from pyproj import CRS, Transformer` deleted from `network.py` | IMPLEMENTED AS SPECIFIED |
| T3 | `_metric_crs_and_transformers(epsg) -> (crs, fwd, inv)` | IMPLEMENTED AS SPECIFIED |
| T3 | `_require_pyproj(what: str) -> module` | IMPLEMENTED DIFFERENTLY — now `_require_pyproj(what, supported)`, second arg required. **F-1.6** |
| T3 | `_build` rewired; `Network(...)` construction untouched | IMPLEMENTED AS SPECIFIED |
| T3 | Both new `tests/test_network.py` tests | IMPLEMENTED AS SPECIFIED (`:374`, `:381`) |
| T4 | `_source_to_wgs84(crs) -> callable | None` | IMPLEMENTED DIFFERENTLY — now short-circuits on `_proj.is_wgs84`, covering CRS84 and EPSG:4979. **F-1.7** |
| T4 | `from_file` rewired; four new tests | IMPLEMENTED AS SPECIFIED (extended: `:219`, `:222`) |
| T5 | `tests/test_no_pyproj.py` subprocess proof | IMPLEMENTED AS SPECIFIED — passes |
| T5 | `pyproject.toml`: pyproj out of `dependencies`, into `crs` extra | IMPLEMENTED AS SPECIFIED — `pyproject.toml:41` is the only mention |
| T5 | Doc edits: `README.md`, `docs/index.md`, `methodology.md`, `statistics.md`, `api.md` | IMPLEMENTED AS SPECIFIED (all five, with `roadtraffic` → `redlight`) |
| T5 | `network.py` module docstring CRS paragraph | IMPLEMENTED AS SPECIFIED (`network.py:17-21`) |
| T5 | `ci.yml` comment corrected | IMPLEMENTED AS SPECIFIED (`ci.yml:24-26`) |
| T5 | `ci.yml` no-pyproj job, "Install without any extras" | IMPLEMENTED DIFFERENTLY — installs `.[shapefile]`. **F-1.8** |
| T5 | `CHANGELOG.md` entry | IMPLEMENTED AS SPECIFIED (`CHANGELOG.md:49`) |

Nothing in this spec is NOT IMPLEMENTED.

## Requirement matrix — `2026-07-31-mover-mode-screening{-design}.md`

| # | Requirement | Status |
|---|---|---|
| API | Four public functions with the design's exact signatures | IMPLEMENTED AS SPECIFIED — verified via `inspect.signature`, all four match character for character |
| API | `MODE_PEDESTRIAN` / `MODE_VEHICLE` / `MODE_UNKNOWN` string constants, exported | IMPLEMENTED AS SPECIFIED |
| `mover_features` | One row per `traj_id`, indexed by it; documented column set | IMPLEMENTED AS SPECIFIED |
| `mover_features` | `unit` does double duty (names columns, reads `threshold`) | IMPLEMENTED AS SPECIFIED |
| `mover_features` | Percentile label `:g`-formatted (`p85`, `p87.5`) | IMPLEMENTED AS SPECIFIED — verified both |
| `mover_features` | Dedup on `interval_id` when present | IMPLEMENTED AS SPECIFIED — verified verdicts identical across `intervals` / `edge_observations` |
| `mover_features` | Empty input → empty frame **with the correct columns** | IMPLEMENTED DIFFERENTLY — `snap_dist_m` is dropped. **F-1.1** |
| `suggest_mode_threshold` | Log-speed KDE, prominence ranking, evaluated beyond the window | IMPLEMENTED AS SPECIFIED |
| `suggest_mode_threshold` | "Two guards, both required" | IMPLEMENTED BUT LATER CHANGED — three conditions now, and the hump test was redefined (`367e683`). **F-1.2** |
| `suggest_mode_threshold` | `None` for <20 movers, no scipy, or no qualifying candidate | IMPLEMENTED AS SPECIFIED (plus a fourth `None` path, **F-1.4**) |
| `suggest_mode_threshold` | Plan's code computes in the caller's `unit` | IMPLEMENTED DIFFERENTLY — computes in m/s (`c7744c0`). **F-1.3** |
| `classify_movers` | `"auto"` raises rather than defaulting; evidence floors checked first; no `require_quality` | IMPLEMENTED AS SPECIFIED |
| `classify_movers` | `unknown` never from speed ambiguity; a congested vehicle is a vehicle | IMPLEMENTED AS SPECIFIED — proved in `t2`/`t7` |
| `filter_by_mode` | `keep=("vehicle",)` default, index reset, `UserWarning` on empty | IMPLEMENTED AS SPECIFIED |
| `filter_by_mode` | Null `traj_id` handling | IMPLEMENTED DIFFERENTLY — explicit null-mover matching added, absent from spec. **F-1.4** |
| Data flow | Classify `intervals`, filter `edge_observations`; both carry `traj_id` | IMPLEMENTED AS SPECIFIED — proved in `t7` |
| Testing | All 11 named tests + the e2e premise test + `walk_along_road` in conftest | IMPLEMENTED AS SPECIFIED — 29 tests in `test_modes.py`, `walk_along_road` at `conftest.py:127`, e2e at `test_pipeline_e2e.py:163` |
| Integration | `mover_screen.py` reduced to argparse + package calls; `histogram` kept; `--threshold` accepts `auto` | IMPLEMENTED AS SPECIFIED — `per_mover_features` and `suggest_threshold` are gone |
| Integration | `customer_report.py`: `--mode-threshold`, `--keep-unknown`, screening between derive and clean, 8-step banner, two data notes | IMPLEMENTED AS SPECIFIED — banners `[1/8]`–`[8/8]`, notes at `:525`–`:537` |
| Integration | "What the feed is made of" deck section, categorical palette, figure/table/tiles | IMPLEMENTED AS SPECIFIED — `SECTIONS` entry `:863`, `fig_modes` `:726`, `CAT_1/2/3` `:108` |
| Docs | `docs/api.md` `## Mode screening` after `## Cleaning`; `methodology.md` §4.5; `__init__.py` docstring entries; 0.5.0 bump | IMPLEMENTED AS SPECIFIED |

Nothing in this spec is NOT IMPLEMENTED.

---

### F-1.1 — `mover_features` returns a different column set for an empty feed, so reading the documented `snap_dist_m` raises `KeyError`

- **Severity:** S2
- **Location:** `src/redlight/modes.py:103-108`
- **Claim:** The empty-input early return hardcodes four columns and omits
  `snap_dist_m`, even when the input frame carries it — so the same call on the
  same schema emits five columns with rows and four without, and a caller that
  reads the documented diagnostic column crashes exactly when the feed is empty.
- **Evidence:**
  ```
  $ .venv/bin/python t6_empty_cols.py
    non-empty -> snap_dist_m present (1 rows)
    empty     -> KeyError 'snap_dist_m'

  Same call, same input schema, different columns out. A caller that reads
  feat['snap_dist_m'] works on data and raises KeyError on an empty feed.
  ```
  Both calls pass a frame whose columns are exactly
  `["interval_id", "traj_id", "speed_mps", "distance_m", "snap_dist_m"]`; the
  second is `full.iloc[:0]`.
- **Expected vs actual:** The design (`-design.md:95`) requires "Empty input
  returns an empty frame with the correct columns rather than raising", and the
  shipped docstring (`modes.py:88-90`) promises `snap_dist_m` "when the input
  carries it" — the empty input *does* carry it. Actual: the column is absent,
  and indexing it raises. This is reachable from a real feed that filters down
  to nothing (all-`NaN` `speed_mps`, or an upstream screen that removed
  everything), which is exactly the case the empty-frame branch exists to serve.
- **Suggested fix:** Build the empty frame's column list from the same rules the
  populated path uses — append `snap_dist_m` when `"snap_dist_m" in df.columns`
  — so the two branches agree by construction rather than by duplication.
- **Verdict:**
- **Outcome:**

---

### F-1.2 — `suggest_mode_threshold` has three guards, not the design's two, and the extra one silently narrows what `None` means

- **Severity:** S3
- **Location:** `src/redlight/modes.py:134-137`, `:221-252`
- **Claim:** Commit `367e683` added a third condition (`_MIN_HUMP_SHARE = 0.15`)
  and redefined the "hump below the candidate" from `argmax(dens[:i])` to *an
  interior local maximum*. The design still says "Two guards, both required"
  and describes guard 2 purely as a location test. The observable consequence
  is a detection floor the design never states: a genuine walking population
  below roughly 8% of movers is no longer found, and `threshold="auto"` raises
  instead of screening.
- **Evidence:**
  ```
  $ .venv/bin/python t4_hump_share.py
  Design doc's own prior-evidence mix (88 pedestrians, 231 vehicles):
    seed 0: 3.010 m/s      seed 1: 3.010 m/s      seed 2: 3.101 m/s
    seed 3: 3.147 m/s      seed 4: 3.070 m/s

  Walker-share sweep, 300 movers total, 5 seeds each:
     walkers   share   suggested threshold (m/s) per seed
         150  50.0%    3.19   3.21   3.23   3.36   3.26
          90  30.0%    3.05   3.02   3.12   3.16   3.09
          60  20.0%    2.94   2.94   3.01   3.09   2.98
          45  15.0%    2.88   2.92   2.94   3.04   2.94
          30  10.0%    2.82   2.82   2.91   3.02   2.91
          24   8.0%    2.81   2.79   2.86   None   None
          18   6.0%    None   None   None   None   None
          15   5.0%    None   None   None   None   None
           9   3.0%    None   None   None   None   None

  Negative control -- vehicle-only, must stay None:
    seed 0..4: None None None None None
  ```
- **Expected vs actual:** Per the design (`-design.md:112-123`), two guards, and
  `None` means "no such split exists". Actual: three conditions, and `None` also
  means "the walking hump is real but carries under 15% of the peak density".
  The design's own headline dataset (88 walkers / 231 vehicles, 27.6%) still
  works and still lands near its chosen 6 mph cut — 3.01-3.15 m/s is 6.7-7.0 mph
  — and the negative control still holds, so the guard did not break the
  spec's evidence. **The difference is an improvement and the spec is the thing
  that is now wrong**: the plan's literal `lower_hump` returned the grid point
  immediately below the candidate on any monotone tail, which made the location
  test vacuous, and the code's own docstring (`modes.py:224-229`) says so. But
  the ~8% floor is undocumented in the design, in `docs/api.md:301-312` and in
  the docstring, all of which describe `None` only as "no walking population
  exists".
- **Suggested fix:** Update the design's "Two guards" section to three, and add
  one sentence to the `suggest_mode_threshold` docstring and `docs/api.md`
  stating that a walking minority below roughly a tenth of movers will not be
  detected and needs an explicit threshold. No code change.
- **Verdict:**
- **Outcome:**

---

### F-1.3 — the plan's `suggest_mode_threshold` computes in the caller's unit, violating the plan's own m/s constraint; the code computes in m/s

- **Severity:** S3
- **Location:** `src/redlight/modes.py:185`, `:191-194`, `:258`
- **Claim:** The plan's Task 2 code block converts the search bounds *out* of
  m/s and runs the KDE on `mover_speeds` as supplied, contradicting the plan's
  own Global Constraint that "All internal computation is in m/s. Convert at the
  boundary". Commit `c7744c0` inverted this: the code now calls
  `to_mps(x, unit)` on entry, works entirely in m/s, and converts the answer
  back with `from_mps` on return.
- **Evidence:**
  ```
  $ .venv/bin/python t5_misc.py
  == unit consistency of suggest_mode_threshold (mph vs kph vs mps) ==
    mps 3.054859 | kph 10.997492 (=3.054859 mps) | mph 6.833525 (=3.054859 mps)
  ```
  The same physical sample expressed in three units yields the same physical
  threshold to six decimal places.
- **Expected vs actual:** The two formulations are mathematically equivalent —
  the estimate is built on `log(x)`, so a unit change is a constant shift of the
  grid and the bandwidth is scale-free — so **the outcome is neutral**. What is
  not neutral is which document is right: **the plan is wrong** and the code is
  right, because the plan's version put a hardcoded unit conversion inside the
  computation instead of at the boundary. The plan's text is still what a reader
  reconstructing the design would follow.
- **Suggested fix:** Annotate the plan's Task 2 code block as superseded by
  `c7744c0`, or leave it and rely on this findings file as the record. No code
  change.
- **Verdict:**
- **Outcome:**

---

### F-1.4 — `modes.py` carries null-id and degenerate-distribution hardening that neither spec's Errors table mentions

- **Severity:** S3
- **Location:** `src/redlight/modes.py:369-377`, `:197-204`
- **Claim:** Two behaviours exist that no spec describes. (a) `filter_by_mode`
  matches a null-`traj_id` mover explicitly, because `isin` never matches NaN
  against itself and a "keep" verdict was otherwise deleting all of that mover's
  rows — dtype-dependently, per the code comment. (b) `suggest_mode_threshold`
  catches `numpy.linalg.LinAlgError` from `gaussian_kde` and returns `None`
  rather than propagating, for the all-movers-at-one-speed case. The design's
  Errors table (`-design.md:170-175`) lists four conditions and neither of these
  is among them.
- **Evidence:** Both paths are pinned by tests that the specs do not call for,
  and which therefore came later:
  ```
  $ grep -n "^def test" tests/test_modes.py | tail -4
  269:def test_suggest_threshold_survives_a_degenerate_distribution():
  276:def test_a_null_trajectory_id_keeps_its_rows_through_classify_and_filter():
  292:def test_null_and_nan_trajectory_ids_filter_identically():
  ```
  ```
  $ .venv/bin/pytest -q tests/test_modes.py
  29 passed
  ```
  The design's Testing section (`-design.md:209-238`) names 11 tests; the file
  has 29.
- **Expected vs allowed:** Both are **improvements** — (a) fixes a silent
  data-loss bug and (b) turns a crash into the module's documented "no split
  found" answer. **The spec is the thing that is wrong**, in that its Errors
  table now under-describes the contract. Neither changes a documented
  behaviour, so nothing downstream is misled.
- **Suggested fix:** Add both rows to the design's Errors table. No code change.
- **Verdict:**
- **Outcome:**

---

### F-1.5 — the drop-pyproj spec's accuracy table is stale: the Krüger `delta` series now runs to n^5

- **Severity:** S3
- **Location:** `src/redlight/_proj.py:81-87`, spec `.plans/2026-08-04-drop-pyproj.md:33-39`
- **Claim:** The spec's Task 2 code block truncates `delta` (conformal → geodetic
  latitude) at n^4 and its headline table claims "UTM round-trip: 0.13 mm worst
  case". Commit `c9a92bb` extended `delta` to n^5, which improves round-trip
  accuracy by roughly four orders of magnitude. Anyone reading the spec as the
  record of what shipped would understate the package's own geodesy by 8000x.
- **Evidence:**
  ```
  $ .venv/bin/python t3_proj_accuracy.py
  UTM forward   vs PROJ, worst :      7.48 nm   (spec claimed 9 nm)
  UTM inverse   vs PROJ, worst :     15.03 nm   (module claims 14 nm)
  UTM round-trip,        worst :     15.82 nm = 0.000016 mm   (spec claimed 0.13 mm)
  ```
  200,000 points across 10 zones (32601/10/18/33/56/60 north, 32701/33/56/60
  south), full ±3° zone width, latitudes 0.1-84°N and 0.1-80°S, oracle is the
  installed pyproj 3.7.2.
- **Expected vs actual:** Spec: 9 nm forward, 0.13 mm round-trip, `delta` at n^4.
  Actual: 7.48 nm forward, 0.000016 mm round-trip, `delta` at n^5. The
  difference is an unambiguous **improvement** and **the spec is what is now
  wrong**. One sub-item runs the other way: `_proj.py:10-11` claims "14 nm
  inverse" and the wider sample above measures 15.03 nm — the module's leading
  "a few tens of nanometres" still covers it, but the parenthetical figure is a
  point measurement stated as a bound.
- **Suggested fix:** Nothing in the code. If the spec is retained as history,
  note the supersession; if `_proj.py:10-11` is to keep a number, restate it as
  "under 20 nm" or name the sample it was measured on.
- **Verdict:**
- **Outcome:**

---

### F-1.6 — `_require_pyproj` gained a second required argument the spec's interface does not list

- **Severity:** S3
- **Location:** `src/redlight/network.py:62`
- **Claim:** The spec (Task 3, Interfaces) produces
  `network._require_pyproj(what: str) -> module` with a single hardcoded list of
  natively supported CRS in the message. The shipped signature is
  `_require_pyproj(what: str, supported: str)`, and the two call sites pass
  different constants — `_SOURCE_CRS_NATIVE` and `_METRIC_CRS_NATIVE` — because
  the sets genuinely differ: Web Mercator is a legal *source* CRS but not a legal
  *metric* one, since it inflates ground distance by `sec(latitude)`.
- **Evidence:**
  ```
  $ grep -n "_require_pyproj\|_NATIVE = " src/redlight/network.py
  48:# What each caller of _require_pyproj can natively handle. These differ: a
  52:_SOURCE_CRS_NATIVE = (
  56:_METRIC_CRS_NATIVE = (
  62:def _require_pyproj(what: str, supported: str):
  118:    pyproj = _require_pyproj(f"metric_epsg=EPSG:{epsg}", _METRIC_CRS_NATIVE)
  141:    pyproj = _require_pyproj(f"Reading a file in {_crs_excerpt(crs)}",
  153:    pyproj = _require_pyproj(f"Reading a file in EPSG:{epsg}",
  ```
  ```
  $ .venv/bin/pytest -q tests/test_network.py
  ... 375 passed (full suite); the two spec'd fallback-error tests are at
  tests/test_network.py:262 and :381 and both assert the redlight[crs] text
  ```
- **Expected vs actual:** Spec: one parameter, one supported-CRS list. Actual:
  two parameters, two lists. The difference is an **improvement** — the spec's
  single list would have told a user whose `metric_epsg=3857` was rejected that
  EPSG:3857 is supported, sending them back to the CRS that just failed. The
  helper is private (leading underscore, not in `__all__`), so nothing outside
  the module is affected. **The spec is the thing that is wrong.**
- **Suggested fix:** None to the code. Correct the spec's Interfaces line if it
  is retained as a reference.
- **Verdict:**
- **Outcome:**

---

### F-1.7 — CRS84 / EPSG:4979 recognition and antimeridian wrapping are absent from the drop-pyproj spec's interface

- **Severity:** S3
- **Location:** `src/redlight/_proj.py:107-153`, `:222-225`; `src/redlight/network.py:137`
- **Claim:** Three behaviours added by `c9a92bb` and `a03733f` are not in the
  spec's Task 2/Task 4 Interfaces lists: `_proj.is_wgs84()`,
  `_proj.EPSG_WGS84_3D = 4979`, and the `[-180, 180)` wrap at the end of
  `utm_inverse`. `_source_to_wgs84` was restructured to short-circuit on
  `is_wgs84(crs)` before `parse_epsg`, replacing the spec's explicit
  `epsg == EPSG_WGS84` branch.
- **Evidence:**
  ```
  $ grep -n "is_wgs84\|EPSG_WGS84_3D\|% 360" src/redlight/_proj.py
   22:EPSG_WGS84_3D = 4979          # WGS84 with an ellipsoidal height; 2D it is 4326
  107:def is_wgs84(crs) -> bool:
  146:        return epsg in (EPSG_WGS84, EPSG_WGS84_3D)
  224:    lon = (np.degrees(lam + lon0) + 180.0) % 360.0 - 180.0

  $ grep -n "is_wgs84" src/redlight/network.py
  137:    if not crs or _proj.is_wgs84(crs):
  ```
  Tests that no spec calls for and that therefore postdate them:
  ```
  tests/test_network.py:219  assert net_mod._source_to_wgs84("EPSG:4979") is None
  tests/test_network.py:222  def test_source_to_wgs84_rejects_projcrs_with_nested_crs84_base()
  ```
- **Expected vs actual:** Spec: `_source_to_wgs84` recognises only `EPSG:4326`
  natively and sends everything unparseable to pyproj. Actual: it also
  recognises `OGC:CRS84` (what GDAL/QGIS stamp on GeoJSON, and which carries no
  EPSG code) and `EPSG:4979`, and it refuses to be fooled by a CRS84 identifier
  nested inside a projected CRS's base — the bug `a03733f` fixed. All three are
  **improvements**; the CRS84 one in particular keeps the most common exported
  file format off the pyproj path for what is an identity transform, which is
  the whole point of the spec. **The spec is the thing that is wrong**, being
  merely incomplete.
- **Suggested fix:** None to the code. Extend the spec's Task 2 and Task 4
  Interfaces lists if the documents are kept as reference.
- **Verdict:**
- **Outcome:**

---

### F-1.8 — the no-pyproj CI job installs the `shapefile` extra, where the spec said "Install without any extras"

- **Severity:** S5
- **Location:** `.github/workflows/ci.yml:118-122`
- **Claim:** The spec's Task 5 Step 5 defines the job as `pip install . pytest`
  under the step name "Install without any extras". The shipped job runs
  `pip install '.[shapefile]' pytest` under "Install with the shapefile extra
  only", with an in-file comment giving the reason.
- **Evidence:**
  ```
  $ sed -n '107,130p' .github/workflows/ci.yml
    no-pyproj:
      name: core install has no PROJ
      ...
        # pyogrio, not the full shapefile stack: it needs only certifi, numpy and
        # packaging, so PROJ still never arrives. Without it every from_file test
        # importorskips out and this job never exercises the source-CRS path,
        # which is precisely the code that has to work with no pyproj.
        - name: Install with the shapefile extra only
          run: |
            python -m pip install --upgrade pip
            pip install '.[shapefile]' pytest
        - name: Assert pyproj is genuinely absent
          run: |
            ! pip show pyproj
            python -c "import redlight; print(redlight.__version__)"
  ```
- **Expected vs actual:** The job's guarantee — `! pip show pyproj` — is
  unchanged, and pyogrio does not depend on pyproj (the spec says so itself in
  its own Out of Scope section). The change **strengthens** the job: without
  pyogrio the `from_file` tests `importorskip` out and the no-pyproj run never
  touches `_source_to_wgs84`, which is the code most at risk. This is an
  improvement and **the spec is what is out of date**. Recorded only so the
  deviation is not mistaken later for an accident.
- **Suggested fix:** None. Note the supersession in the spec if it is retained.
- **Verdict:**
- **Outcome:**

---

## Repo state on exit

```
$ .venv/bin/pytest -q
375 passed, 6 warnings in 13.62s

$ git status --porcelain
(empty)
```

Baseline test count matches `00-baseline.md`. No source file was modified and
nothing was committed; all scratch scripts live in the session scratchpad.
