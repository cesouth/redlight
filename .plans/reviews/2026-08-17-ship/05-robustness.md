# Robustness — bad input, edge cases and error messages — findings
**Pass:** Task 5
**Date:** 2026-08-18
**Commit reviewed:** `bae2e19`
**Scope:** `src/redlight/points.py`, `src/redlight/network.py`, `src/redlight/osm.py`,
`src/redlight/routing.py`, `src/redlight/analysis.py`, `src/redlight/mapping.py`.
**Method:** every case below was run, not reasoned about, except `from_overpass`
(read-only by instruction — no network access). Optional extras were hidden by
an `__import__` hook in a subprocess to test the "extra not installed" paths.
Scripts in `…/c8494443-…/scratchpad/t5/`: `p1_points.py`, `p2_network.py`, plus
inline batteries for CRS, routing, analysis and mapping. No source file changed;
`git status --porcelain` empty (verified below).

Outcomes classified as the pass brief requires: **GOOD** (correct, or an error
naming problem and fix), **LOUD** (raises, but unhelpfully — S2), **SILENT**
(plausible-looking wrong answer — S1).

## Summary

The error messages in this package are unusually good — `_require_pyproj`,
the `HMMMatcher` trajectory-id error, the `_INTERVAL_IDENTITY` guard and the
`edge_betweenness` weight check all name the problem *and* the way out, and
degenerate geometry (zero-length edges, closed loops, empty feature sets) is
caught and explained. The boundary has one serious hole: **`Network.from_geojson`
ignores a declared `crs` member entirely**, so a file in a projected CRS is read
as lon/lat degrees and produces a 1,000 m road of length 16,585,698 m and an
aggregate speed of 4.5 million mph, with no error anywhere. Two further silent
classes: coordinates outside valid lat/lon range are accepted (closing deferred
F-3.4), and speed columns are never sanity-checked, so negative and supersonic
speeds pass through untouched.

### What is GOOD (recorded so later passes do not re-test it)

| Case | Behaviour |
|---|---|
| empty file / header-only / one row | `EmptyDataError`; `n=0`; `n=1` — all clean |
| missing lon/lat or time column | `ValueError` naming `lon_col=`/`lat_col=`/`time_col=` |
| `derive_speed=True` without `id_col` | `ValueError` explaining speeds must not cross movers |
| tz-aware timestamps | parsed, and **warned** that hour-of-day will use that offset |
| mixed UTC offsets in one file | parsed via UTC, warned |
| NaN / empty-string coordinate | dropped with a warning and a count |
| `traj_id` with NaN | dropped, warning names the column |
| `traj_id` mixed `1` and `"1"` | coerced to one trajectory (not silently split) |
| unsorted timestamps | sorted internally |
| GeoJSON with zero features / only Points | `ValueError: GeoJSON contained no LineString features.` |
| zero-length edges, closed loops | skipped with a warning; raises only if nothing usable is left |
| `MultiLineString`, duplicate parallel edges | handled; parts become separate edges |
| network spanning two UTM zones | works; length error +152 ppm (21 m in 139 km) — immaterial |
| network spanning the antimeridian | works; edge length within 0.04 % of the geodesic |
| `HMMMatcher` without a trajectory column | `ValueError` naming both `load_points(id_col=…)` and `NearestMatcher` — the `matching.py:224` claim is verified |
| trajectory entirely off-network | all `edge_id = -1`, `derive_speeds` emits 0 intervals |
| trajectory jumping between disconnected components | matched per-component; `derive_speeds` emits **0** intervals rather than bridging |
| unsupported `metric_epsg` without pyproj | names the extra, the pip command *and* the supported alternatives |
| `Router.route` origin == destination | `distance_m = 0.0`, empty `edge_ids` |
| `Router.route` across components | `ValueError` saying the endpoints are in different components |
| `edge_betweenness_centrality` all-NaN `time` | `ValueError` reporting "10 of 10 edges have no usable 'time' value" |
| `network_stats` / `connectivity_report` on single-edge, disconnected, antimeridian | all return sane structures, no crash |
| `to_geojson` with no speeds, NaN speeds, `Timestamp`/`set`/`ndarray` attrs | valid JSON in every case — `_jsonable` works |
| `from_overpass` empty result | `ValueError: Overpass returned no ways for that bounding box / highway filter.` |
| `from_overpass` timeout | present on both sides — client-side `urlopen(..., timeout=)` and the query's own `[timeout:N]`, default 90 s |

---

### F-5.1 — `Network.from_geojson` ignores a declared `crs` member and reads projected coordinates as lon/lat
- **Severity:** S1 (SILENT)
- **Location:** `src/redlight/network.py:273-330` (`from_geojson`), `:472-475`
  (`_auto_utm_epsg` on the first coordinate)
- **Claim:** A GeoJSON carrying `"crs": {"properties": {"name":
  "urn:ogc:def:crs:EPSG::27700"}}` is read as though its eastings/northings were
  degrees. Nothing warns. `_proj.parse_epsg` and `_proj.is_wgs84` already exist
  for exactly this and are consulted on the Shapefile/GPKG path — the GeoJSON
  loader simply never asks.
- **Evidence:**
  ```
  A. A GeoJSON in British National Grid (EPSG:27700) with a declared crs member
     The road is 1000 m long in its own CRS (530000,180000)->(531000,180000).
     metric CRS chosen : EPSG:32644
     edge_length_m     : 16,585,698.5 m     <-- true answer is 1000.0 m
     node coords (lonlat as the package believes): [(80.0000000001, 7.6e-12), (0.00156, 7.6e-12)]
     where the road REALLY is: (-0.1284, 51.5040) -> (-0.1140, 51.5038)

  B. does it help if the crs member is absent (the RFC 7946 default)?
     metric CRS EPSG:32644  edge_length=16,585,698.5 m
  ```
  End-to-end, with points in the same CRS (the realistic case — one export, one
  projection):
  ```
  A. END-TO-END with a projected-CRS (EPSG:27700) network AND points
     matched edge_ids : [1, 1, -1, -1, -1, -1]
     intervals emitted: 1
     speed_mps        : [2010003.4]   <-- truth is 20.0 m/s
     quality          : False
     aggregate        : [{'n': 1, 'mean_speed': 4496249.579192796, 'unit': 'mph'}]
  ```
- **Expected vs actual:** RFC 7946 says GeoJSON is WGS84, so *ignoring* the
  member is defensible in the abstract — but the member is still widely emitted
  by QGIS and `ogr2ogr`, and the failure mode is a four-orders-of-magnitude
  wrong answer rather than a refusal. Expected: read the member, and either
  transform (natively for UTM/3857, via the `crs` extra otherwise) or raise the
  same excellent error `_require_pyproj` already produces. Note the mitigations
  that exist — most fixes go unmatched and `quality=False` — do **not** save the
  user, because `aggregate_speeds` does not filter on quality by default and
  reported 4.5 million mph.
- **Suggested fix:** In `from_geojson`, parse any `crs` member with
  `_proj.parse_epsg` / `_proj.is_wgs84`. If it is not WGS84, either project the
  input (the machinery exists) or raise naming the code found and the `crs`
  extra. A cheap belt-and-braces addition: warn when input coordinates fall
  outside ±180 / ±90, which is a certain sign of a projected file.
- **Verdict:** ACCEPT
- **Outcome:** fixed (c4d40ab)

---

### F-5.2 — Coordinates outside valid lat/lon range are accepted without comment
- **Severity:** S1 (SILENT)
- **Location:** `src/redlight/points.py:330-420` (`load_points` validation)
- **Claim:** `lat = 91`, `lon = 181` load cleanly. This is the loader-boundary
  half of **F-3.4**, which Task 3 deferred to this pass; Task 3 established that
  `geodesic_distance` then returns a confident, plausible, wrong number for such
  a point (lat 91 and lat 89 give the *identical* distance, where `pyproj`
  returns NaN).
- **Evidence:**
  ```
  POINTS — coordinates
    lat/lon SWAPPED (51.5,-0.13 -> -0.13,51.5)     OK  n=2
    lat 91 / lon 181 out of range                  OK  n=2
  ```
  and from Task 3 (`03-numerical-accuracy.md`, F-3.4):
  ```
    lat 91 vs lat 0  (lat/lon swap symptom)   redlight 9890271.864398   pyproj nan
    lat=  91.0 -> distance to equator =   9890271.8644 m
    lat=  89.0 -> distance to equator =   9890271.8644 m
  ```
- **Expected vs actual:** Expected a warning or an error naming the offending
  rows — `|lat| > 90` is not a position. Actual: silent acceptance, then a
  plausible number downstream. The *swapped* case (valid ranges, wrong columns)
  is genuinely undetectable from ranges alone and is not part of this finding.
- **Suggested fix:** Range-check in `load_points` where the drop-with-warning
  machinery already exists, so it costs nothing on the hot path — this is why
  F-3.4 was routed here rather than into `geodesic_distance`.
- **Verdict:** ACCEPT
- **Outcome:** fixed (ecbe530) -- also closes the deferred F-3.4

---

### F-5.3 — Speed columns are never sanity-checked: negative, supersonic and mis-declared units all pass
- **Severity:** S1 (SILENT)
- **Location:** `src/redlight/points.py` speed ingestion (`speed_col` /
  `speed_unit`)
- **Claim:** Three separate wrong inputs are accepted without a word: a negative
  speed, a physically impossible speed, and a column in mph declared as m/s.
- **Evidence:**
  ```
  POINTS — speed column units
    speed col in mph but speed_unit='mps'   OK  speed[0]=60.0     <- 134 mph
    same data declared mph                  OK  speed[0]=26.8224  <- correct
    negative speeds                         OK  speed[0]=-5.0

  D. negative and absurd speeds through the whole pipeline
     negative             -> speed_mps=[-5.0, -5.0]    warns=[]
     400 m/s (Mach 1.2)   -> speed_mps=[400.0, 400.0]  warns=[]
  ```
- **Expected vs actual:** A speed magnitude cannot be negative, and 400 m/s is
  not road traffic. `units._usable_speed` already encodes exactly this
  judgement — "positive and finite" — but is applied to network speed limits,
  not to ingested point speeds. `derive_speeds` has `max_speed_mps` for the
  speeds it computes; nothing guards the ones handed to it.
- **Suggested fix:** Warn (do not drop — the package's convention is advisory)
  on non-positive speeds and on speeds above a generous ceiling, reporting the
  count and the observed maximum. The mis-declared-unit case is only detectable
  this way, and it is the most likely of the three in practice.
- **Verdict:** ACCEPT
- **Outcome:** fixed (257e819)

---

### F-5.4 — A single-point trajectory with `derive_speed=True` discards the whole dataset and blames "unparseable" data
- **Severity:** S2 (LOUD)
- **Location:** `src/redlight/points.py:410-415`
- **Claim:** One fix in, zero rows out, with a warning saying the row was
  "missing/unparseable". The row parsed perfectly; it simply has no second fix
  to difference against. A user with many single-fix movers loses them all and
  is pointed at their file format.
- **Evidence:**
  ```
  B. single-point trajectory + derive_speed: what exactly is dropped?
     rows in -> out: 1 -> 0
     warning: ['Dropped 1 row(s) with missing/unparseable lon/lat/time/speed.']

  C. two-point trajectory, does it survive?
     rows in -> out: 2 -> 2  speeds=[1.1132, 1.1132]
  ```
- **Expected vs actual:** Expected a message distinguishing "could not parse
  this row" from "this mover has too few fixes to derive a speed", with the
  count of movers affected. Actual: one message covering both, naming only the
  first.
- **Suggested fix:** Count the two causes separately and warn separately. The
  drop itself is correct.
- **Verdict:** ACCEPT
- **Outcome:** fixed (257e819) -- committed with F-5.3; both touch the same block

---

### F-5.5 — Mixing tz-naive and tz-aware timestamps silently drops the naive rows
- **Severity:** S2 (LOUD)
- **Location:** `src/redlight/points.py:191-198` (`_parse_times`)
- **Claim:** A file with one `2026-06-01T08:00:00` and one
  `2026-06-01T08:00:10+01:00` keeps 1 of 2 rows, warning "missing/unparseable".
  Both values are parseable; they merely disagree about whether they carry an
  offset. Same misleading diagnosis as F-3.6, which fixed the *format* half of
  this in `708ed94`.
- **Evidence:**
  ```
  POINTS — timestamps
    tz-naive + tz-aware mixed       OK  n=1
      warns: ['Dropped 1 row(s) with missing/unparseable lon/lat/time.']
  ```
- **Expected vs actual:** Expected either a decision (assume the naive rows are
  in the same zone as the aware ones, and say so) or a refusal naming the real
  conflict. Actual: half the data vanishes behind the wrong reason.
- **Suggested fix:** Detect the mixed-awareness case explicitly and warn naming
  it. Whether to coerce or refuse is a design decision worth making
  deliberately — a silent 50 % data loss is the one option that should be off
  the table.
- **Verdict:** ACCEPT -- the misleading message only. Whether to coerce naive rows into the
  aware rows' offset or refuse the file outright is a real design decision and
  stays open; fixing the diagnosis does not prejudge it.
- **Outcome:** fixed (e58b0ca) -- diagnosis only. The rows are still dropped; coerce-vs-refuse
  remains an open design decision, flagged rather than guessed at.

---

### F-5.6 — A one-coordinate LineString raises a raw GEOS exception
- **Severity:** S2 (LOUD)
- **Location:** `src/redlight/network.py` `_build` geometry handling
- **Claim:** A `LineString` with a single coordinate — which real exports do
  produce from truncated ways — escapes as a shapely internal error naming
  neither the file, the feature, nor the fix.
- **Evidence:**
  ```
  NETWORK — geometry
    LineString with 1 coordinate   GEOSException: IllegalArgumentException:
                                   point array must contain 0 or >1 elements
  ```
  Contrast the adjacent degenerate cases, which are handled well:
  ```
    zero-length edge (identical endpoints)  ValueError: Network has no usable edges: …
      warns: ['Skipped 1 closed-loop feature(s) whose endpoints coincide …']
  ```
- **Expected vs actual:** Expected the same treatment the other degenerate
  geometries get — skip with a counted warning, raise only if nothing usable
  remains. Actual: an uncaught third-party exception.
- **Suggested fix:** Screen `len(coords) < 2` alongside the existing
  zero-length/closed-loop checks.
- **Verdict:** ACCEPT
- **Outcome:** fixed (7075b9c)

---

### F-5.7 — `plot_speed_map` without matplotlib raises a bare ImportError that never names the `mapping` extra
- **Severity:** S2 (LOUD)
- **Location:** `src/redlight/mapping.py:237`
- **Claim:** `import matplotlib` at `mapping.py:237` is unguarded, so a user
  without the extra gets the bare module-not-found error. The docstring at
  `mapping.py:190` documents `pip install redlight[mapping]`, but the error the
  user actually sees does not.
- **Evidence:**
  ```
  plot_speed_map w/o matplotlib:
    ImportError: No module named 'matplotlib'
  ```
  (matplotlib hidden by an `__import__` hook in a subprocess.) The package
  already has the pattern this should follow — `network._require_pyproj`
  produces:
  ```
  ImportError: metric_epsg=EPSG:27700 needs pyproj, which redlight does not
  install by default. Either install the extra:
      pip install 'redlight[crs]'
  or use a natively supported CRS: a WGS84 UTM zone …
  ```
- **Expected vs actual:** Expected `_require_pyproj`'s quality for the
  `mapping` extra. Actual: the raw import error.
- **Suggested fix:** Wrap the import and raise naming the extra, mirroring
  `_require_pyproj`. Worth checking `pyogrio` (the `shapefile` extra) for the
  same gap while in there.
- **Verdict:** ACCEPT
- **Outcome:** fixed (527621d)

---

### F-5.8 — `Router.route` with a node id that does not exist fails inside a float conversion
- **Severity:** S2 (LOUD)
- **Location:** `src/redlight/routing.py` (`Router.route`)
- **Claim:** Passing an id that is not a node produces
  `ValueError: could not convert string to float`, because the argument is
  being treated as a coordinate. Nothing tells the user the node was not found
  or what a valid node looks like.
- **Evidence:**
  ```
  ROUTING
    route to nonexistent node    ValueError: could not convert string to float: np.str_('nope')
  ```
  Compare the sibling case, which is handled well:
  ```
    route across components      ValueError: No time route exists between the origin and
                                 destination: the endpoints are in different components
  ```
- **Expected vs actual:** Expected "node X is not in the network; use
  `Router.nearest_node(lon, lat)` to find one". Actual: an error about float
  conversion.
- **Suggested fix:** Validate membership in `network.graph` before use and
  raise naming `nearest_node`.
- **Verdict:** ACCEPT
- **Outcome:** fixed (806cc3d)

---

### F-5.9 — `from_overpass` propagates raw HTTP and JSON errors
- **Severity:** S5
- **Location:** `src/redlight/osm.py:199-200`
- **Claim:** `urlopen` and `json.loads` are called without handling, so an HTTP
  5xx from Overpass surfaces as `urllib.error.HTTPError` and a truncated or
  HTML response as `json.JSONDecodeError`. Overpass returning an error page
  under load is routine.
- **Evidence:** read-only, as the pass brief requires (no network access):
  ```
  199:    with urllib.request.urlopen(req, timeout=timeout) as resp:
  200:        data = json.loads(resp.read().decode("utf-8"))
  ```
  No `try`/`except` on either line. **Timeouts are handled well** — both
  client-side (`urlopen(..., timeout=)`) and server-side (`[timeout:N]` in the
  query), default 90 s — and an empty-but-valid response is caught with a clear
  `ValueError` at `network.py:440`.
- **Expected vs actual:** Expected the two failure modes a public endpoint
  actually produces to be named. Actual: raw exceptions. Low severity because
  the caller is opting into a network call and the exception type is at least
  accurate. **Not runtime-verified** — this is the one item in the pass
  established by reading rather than running.
- **Suggested fix:** Wrap both in a message naming the endpoint, the status,
  and the suggestion to retry or narrow the bbox.
- **Verdict:** ACCEPT
- **Outcome:** fixed (e201f50)

---

## Repo state on exit

```
$ git status --porcelain
(empty)
```

Nothing was changed or committed. All scratch data and networks were written
under the scratchpad.

## Unverified suspicions

1. **A non-WGS84 *geographic* CRS would be the dangerous form of F-5.1.** The
   projected case demonstrated above is wrong by 16,000× and hard to miss. A
   file in, say, OSGB36 lat/lon (EPSG:4277) would carry coordinates that look
   entirely normal and be wrong by ~100 m — plausible, and silent. Not
   constructed here; worth one test when F-5.1 is fixed.
2. **`pyogrio` (the `shapefile` extra) may have the same unguarded-import gap as
   F-5.7.** Only `matplotlib` was hidden and tested; the shapefile path was not.
3. **`Router` falls back to `default_speed_mps` (11.176) when every edge speed
   is NaN or None**, routing successfully and silently by a constant. That is
   the documented purpose of the parameter, so it is recorded as correct — but
   whether a user who assigned speeds and got all-NaN would want silence rather
   than a warning was not put to the test.
