# Accuracy — the numerical core — findings
**Pass:** Task 3
**Date:** 2026-08-17
**Commit reviewed:** `83880be`
**Scope:** `src/redlight/_geo.py`, `src/redlight/_proj.py`, `src/redlight/units.py`,
`src/redlight/speeds.py`, `src/redlight/matching.py`. Read for context only:
`src/redlight/network.py` (`_build_segment_table`, `candidate_edges*`,
`_auto_utm_epsg`), `src/redlight/aggregate.py` (`_INTERVAL_IDENTITY`),
`src/redlight/points.py` (`_parse_times`, `_derive_speed_mps`).
**Method:** external-oracle comparison (`pyproj.Geod`, `pyproj.Transformer`,
`scipy.stats`, exact rational arithmetic) plus hand-constructed synthetic
networks with analytically known on-road distances, and a toggleable copy of
`HMMMatcher._match_one` used to decode the same trajectory with and without
each non-paper behaviour. All scripts in
`/private/tmp/claude-501/…/c8494443-225d-45a5-b2bf-a37253d05b4e/scratchpad/`:
`g1`–`g3` (geodesy), `p1`–`p2` (projections), `u1` (units), `s1`–`s5` (speeds),
`m1`–`m6` + `hmm_variant.py` (matching). No source file was modified;
`git status --porcelain` is empty and `.venv/bin/pytest -q` reports
**380 passed** at this commit.

## Summary

The pure-numpy geodesy and projections are excellent and need no defence: the
Vincenty inverse agrees with `pyproj.Geod` to **77 µm worst-case globally and
6 µm at road scale**, the Krüger series agrees with `pyproj.Transformer` to
**4.4 nm forward / 9.5 nm inverse across all 120 UTM zones**, the unit
constants are **bit-identical** to their exact definitions, and
`speeds._hop_distance` reproduces an analytically-known three-piece on-road
distance to **1e-9 m** in every configuration tested. The single worst thing is
in `matching.py`: it reads the `t` value from `candidate_edges` as a fraction
of the **edge** when it is in fact a fraction of a ≤25 m **sub-segment**, so
every non-same-edge transition route distance is wrong — by 891 m on a 1114 m
edge in the worked example — and **~10% of fixes decode onto a different edge**
than they would with the correct arc position. Two further silent-wrong-number
findings follow: the same-edge shortcut's blanket `delta = 0` misprices curved
edges by up to 70 nats (though see the correction under F-3.2 -- the decode
evidence for it did not survive re-testing), and `derive_speeds` resolves
duplicate timestamps by input row order, changing the aggregated mean speed
from 72.28 mph to 77.27 mph on the same four fixes.

### What is confirmed correct (recorded so no later pass re-derives it)

| Claim | Verdict | Measured |
|---|---|---|
| `_geo.geodesic_distance` vs `pyproj.Geod` | **correct** | 7.7e-05 m max over 20,000 random global pairs; 6.0e-06 m max over 20,000 road-scale pairs |
| Vincenty non-convergence behaviour | **safe** | raises `ValueError`, never returns an unconverged value |
| `_proj` UTM vs `pyproj.Transformer`, 120 zones × 7 lats × 7 lons | **correct** | 4.42e-09 m forward, 9.46e-09 m inverse, 8.56e-09 m round-trip |
| `_proj.web_mercator_inverse` vs pyproj | **correct** | 1.97e-10 m |
| `UtmCrs.to_epsg()` for southern zones (327xx vs 326xx) | **correct** | 32733 → 32733, `"WGS 84 / UTM zone 33S"`, `north=False`; forward matches pyproj exactly |
| `units.to_mps` / `from_mps` constants | **exact** | `_MPH_TO_MPS` and `_KPH_TO_MPS` are bit-identical to `1609.344/3600` and `1000/3600`; round-trip max relative error 2.0e-16 (1 ulp) |
| `speeds._hop_distance` three-piece sum | **correct** | ≤1.3e-09 m vs the analytic value across shared-node, real-middle, same-road, reverse-direction, zero-distance and Y-junction cases |
| `speeds` interval-midpoint `time` convention | **as documented** | `time == t_from + (t_to - t_from)/2` exactly |
| `speed_sigma_mps` / `speed_var` error model | **as documented** | `hypot(σ_i, σ_j)/dt` and its square, to 1e-12 |
| `quality` predicate and advisory (non-filtering) behaviour | **as documented** | all five terms fire independently; failing rows are still returned |
| `interval_id` numbering and `interval_id_start` | **correct** | contiguous, unique, continuous across trajectories, honoured by both frames |
| `aggregate._INTERVAL_IDENTITY` collision guard | **fires** | raises on the colliding concat; does not fire on a legitimate single run; correctly scoped to the only path that dedups |
| `matching._emission_logp` vs Newson & Krumm eq. (1) | **exact** | max 1.8e-15 vs `scipy.stats.norm.logpdf`, normalising constant present and tracking `sigma_z` |
| `matching._transition_logp` vs Newson & Krumm eq. (2) | **exact** | 0.0e+00 vs `scipy.stats.expon.logpdf`, normalising constant present and tracking `beta` |

**Do the projection error magnitudes matter for a 1–100 m GPS regime?** No, by
eight orders of magnitude. The worst projection disagreement anywhere in the
120 zones is 9.5 nanometres; the worst geodesic disagreement at road scale is
6 micrometres. Neither is observable in a package whose smallest meaningful
quantity is a metre. The module docstring's claim of "7.5 nm forward, 14 nm
inverse" is corroborated (this pass measured 4.4 nm / 9.5 nm on a different
sampling grid).

**Norway/Svalbard zone-width exceptions:** the code does not claim to handle
them and does not (`_auto_utm_epsg` is pure arithmetic, so Bergen gets zone 31
where the official grid widens zone 32). This is **not a defect**: the metric
CRS is only a local working frame, and the cost is scale distortion of
−199 ppm instead of +103 ppm — 0.2 m over a 1 km edge, far inside the noise.
Straying outside a zone is also safe: the series still round-trips to 2e-06 m
at 90° from the central meridian.

---

### F-3.1 — `matching.py` treats a segment-local fraction as an edge fraction, corrupting every transition route distance
- **Severity:** S1
- **Location:** `src/redlight/matching.py:295` (`leadin_cur`) and
  `src/redlight/matching.py:313-314` (`remaining_prev`)
- **Claim:** The `t` returned by `Network.candidate_edges` / `candidate_edges_batch`
  is the fractional position along one **≤25 m sub-segment** of the snap table
  (`Network._SNAP_DENSIFY_M = 25.0`, `network.py:579`), not along the edge.
  `matching.py` multiplies it by `network.edge_length(eid)` as if it were an
  edge fraction, so the lead-in and lead-out terms of the three-piece route
  distance are wrong by up to a full edge length. `speeds._arc_position`
  (`speeds.py:80-85`) avoids exactly this trap and says so in its docstring —
  "Uses shapely's exact line-projection rather than the segment-local ``t``
  from candidate_edges" — so the two modules disagree about the same quantity
  and `matching.py` is the one that is wrong.
- **Evidence:**
  ```
  $ .venv/bin/python m5_snap_t_scale.py
  one straight edge, length 1114.28 m
  _SNAP_DENSIFY_M = 25.0 m  ->  45 sub-segments in the snap table

  A point placed at a known fraction along the edge; what does t say?
   true frac  true arc (m)  t from cand      t*L (m)    error (m)
        0.05         55.71       0.2500       278.57       222.86
        0.10        111.43       0.5000       557.14       445.71
        0.60        668.57       0.0000         0.00       668.57
        0.90       1002.85       0.5000       557.14       445.71
        0.99       1103.14       0.5500       612.86       490.28
    worst |t*L - true arc| = 668.57 m  on a 1114 m edge

  Same question via candidate_edges_batch (the path HMMMatcher uses):
   true frac   true arc        t        t*L      error
        0.02      22.29   0.9000    1002.85     980.57
        0.98    1092.00   0.1000     111.43     980.57

  ==========================================================================
  Consequence for the HMM transition route distance (matching.py:308-315)
  ==========================================================================
    edge A id=0 len=1114.3   edge B id=2 len=1114.3 (in series)
    fix 1 at 90% of A (arc  1002.9 m)  -> t = 0.5000
    fix 2 at 10% of B (arc   111.4 m)  -> t = 0.5000
    straight-line step gc_step   =    222.86 m
    TRUE  on-road route distance =    222.86 m   (delta     0.00 -> logp    -3.401)
    CODE  on-road route distance =   1114.28 m   (delta   891.42 -> logp   -33.115)
    route-distance error         =    891.42 m
    transition log-prob error    =    29.714 nats

    Cross-check: speeds._hop_distance for the SAME pair uses shapely and
    gets the true value:
      speeds._arc_position: 1002.85 m (true 1002.85), 111.43 m (true 111.43)
      speeds._hop_distance: 222.86 m   vs matching's 1114.28 m
  ```
  Decoded-path impact, measured against an otherwise byte-identical decoder
  that substitutes the true arc position (`m6_decode_impact.py` +
  `ArcExactHMM`), on a properly noded 5×5 grid of 200 m blocks with a
  staircase route (many turns), 10 seeded trajectories per noise level:
  ```
  80 directed edges, 25 nodes, block length ~201 m
    [noise 5 m, trial 0] 6/48 fixes differ, first at index 6
       shipped   : [0,0,0,0,0,0,0,18,18,18,18,18,18,20,20,20,20,20,20,38,...]
       arc-exact : [0,0,0,0,0,0,18,18,18,18,18,18,20,20,20,20,20,20,20,38,...]
    noise    5 m: 47/480 fixes decoded differently (9.8%) over 10 staircase trajectories
    noise   10 m: 48/480 fixes decoded differently (10.0%) over 10 staircase trajectories
    noise   15 m: 43/480 fixes decoded differently (9.0%) over 10 staircase trajectories
  ```
- **Expected vs actual:** Per Newson & Krumm (2009) the transition argument is
  the on-road distance between consecutive snapped positions, which
  `matching.py:308-312`'s own comment defines correctly as "remaining length of
  the previous edge past its snap, the node-to-node middle, then the current
  edge's length up to its own snap". Expected `remaining_prev = L_prev − s_prev`
  and `leadin_cur = s_cur` with `s` a true arc length in metres; actual is
  `(1 − t)·L_prev` and `t·L_cur` with `t` a fraction of a ≤25 m sub-segment. On
  the worked pair the route distance is 1114.28 m instead of 222.86 m.
- **Suggested fix:** Carry the true arc length rather than `t` — compute it once
  per (fix, candidate) with the same `shapely` projection `speeds._arc_position`
  already uses, store metres in `snap_t`, and set `leadin_cur = s_cur`,
  `remaining_prev = edge_length(peid) − s_prev`. Note this is a hot-path change;
  Task 7/8 should measure it, and Task 8's invariance test must be regenerated
  because output legitimately changes.
- **Verdict:** ACCEPT
- **Outcome:** fixed (5d2f1d8)

---

### F-3.2 — The same-edge shortcut forces `delta = 0` on curved edges, overriding the emission term
- **Severity:** S1
- **Location:** `src/redlight/matching.py:296-303`
- **Claim:** The shortcut sets route distance = `gc_step` whenever the candidate
  edge equals the predecessor edge, making `delta` identically 0 — the maximum
  possible transition score — regardless of how much arc the vehicle would
  actually have had to drive. On a straight edge that is right. On a curved
  edge the along-edge distance can be many times the straight-line step, and
  the blanket `delta = 0` then outweighs a large emission preference for a
  different, closer edge. **Verdict on the behaviour: a principled extension
  with an over-reaching implementation** — it is genuinely load-bearing (see
  below), but the constant it uses is wrong.
- **Evidence:** The shortcut is necessary — without it, six fixes driving
  straight down one two-way road decode as an impossible direction alternation:
  ```
  $ .venv/bin/python m2_behaviours.py
  (a) SAME-EDGE TRANSITION SHORTCUT  (matching.py:296-303)
    network: one two-way road, edges [0, 1] (0 = W->E, 1 = E->W), length 2228.6 m
    trajectory: 6 fixes driving straight west->east along it
      AS SHIPPED (shortcut on)           edge_id=[0, 0, 0, 0, 0, 0]
      shortcut OFF                       edge_id=[1, 0, 1, 0, 1, 0]
  ```
  But the quantity it asserts is wrong on curved edges, by up to 70 nats:
  ```
  $ .venv/bin/python m4_sameedge_curved.py
  hairpin road: one edge, arc length 2280.5 m, arms 67 m apart

  == The quantity the shortcut asserts vs the truth, along the hairpin ==
  fix pair                          gc_step   true arc  true delta  logp(code)  logp(true)  gap (nats)
  same arm, 111 m apart               110.7      110.7         0.0      -3.401      -3.401       0.000
  opposite arms, same lat              66.9     1395.0      1328.2      -3.401     -47.674      44.273
  west arm low -> east arm high       667.5     1173.7       506.2      -3.401     -20.276      16.874
  opposite arms, near the open end     66.9     2169.8      2103.0      -3.401     -73.500      70.099
  ```
  And it produces a wrong decode. A hairpin road plus a cross street joining
  its two arms; the vehicle drives up the west arm, across the cross street,
  down the east arm. Fix 2 lies **exactly on** the cross street (snap 0.00 m)
  and 33.43 m from the hairpin:
  ```
  hairpin(0/1, 2280 m) + cross street(2/3, 67 m); fix 2 sits ON the cross street
    shipped HMMMatcher: edge_id=[0, 0, 0, 0, 0]  snap=[0.0, 0.0, 33.43, 0.0, 0.0]
    arc-exact leadin  : edge_id=[0, 0, 0, 0, 0]  snap=[0.0, 0.0, 33.43, 0.0, 0.0]

  Emission-only preference for fix 2 (what the paper's model alone says):
     edge 2: snap   0.00 m -> emission logp    -2.711
     edge 3: snap   0.00 m -> emission logp    -2.711
     edge 1: snap  33.43 m -> emission logp   -18.230
     edge 0: snap  33.43 m -> emission logp   -18.230
     -> emission prefers the cross street by 15.5 nats;
        the same-edge shortcut's blanket delta=0 for the hairpin outweighs it.
  ```
  The decoded path claims the vehicle stayed on the hairpin, i.e. drove ~1.4 km
  in 20 s (70 m/s ≈ 157 mph), rather than 67 m across the cross street. Fixing
  F-3.1 alone does **not** repair this (the `arc-exact leadin` row above is
  identical), confirming the two defects are independent.
- **Expected vs actual:** Newson & Krumm's transition argument is
  `|gc_step − route_distance|`. For a same-edge transition the route distance
  is the along-edge arc between the two snaps, `|s_cur − s_prev|`, which equals
  `gc_step` only when the edge is straight between them. Expected `delta =
  ||s_cur − s_prev| − gc_step|`; actual `delta = 0` unconditionally.
- **Suggested fix:** Once F-3.1 gives true arc positions, replace `rd = gc_step`
  with `rd = abs(s_cur − s_prev)`. That keeps the property the shortcut exists
  for (a straight two-way road still scores `delta ≈ 0`, so no flip-flop) while
  charging curved edges honestly. The two fixes should be designed together.
- **CORRECTION (added during the fix cycle, 2026-08-18):** **the decode
  evidence above does not support the claim, and was mis-attributed.** The
  hairpin/cross-street network used to demonstrate the wrong decode is
  topologically *disconnected* -- the cross street shares no node with the
  hairpin, because the loader does not split ways at intersections -- so the
  cross street was never a reachable alternative and the decode was driven by
  the saturating penalty, not by the same-edge shortcut. Re-tested after F-3.1
  landed:
  ```
  UNNODED cross street (drawn crossing, no shared node -- common in raw OSM)
    gc_step rule (shipped+F3.1): [1, 1, 1, 1, 1]  snaps=[0.0, 0.0, 33.43, 0.0, 0.0]
    arc rule  (F-3.2 applied):   [1, 1, 1, 1, 1]  snaps=[0.0, 0.0, 33.43, 0.0, 0.0]

  NODED closing road between the hairpin's two end nodes
    gc_step rule (shipped+F3.1): [1, 2, 2, 2, 1]  snaps=[0.0, 0.0, 0.0, 0.0, 0.0]
    arc rule  (F-3.2 applied):   [1, 2, 2, 2, 1]  snaps=[0.0, 0.0, 0.0, 0.0, 0.0]
  ```
  A randomised search over 300 networks with curved multi-vertex edges and
  noisy trajectories along them found **no decode difference at all**:
  ```
  $ .venv/bin/python f2_search.py
  networks tried: 300, fixes compared: 3600
  fixes decoded differently by the F-3.2 change: 0 (0.00%)
  ```
  What survives is the modelling gap alone: `rd = gc_step` is not the arc, and
  the 70-nat table above is real. But for consecutive fixes the emission term
  keeps successive snaps close together on the edge, where arc and chord agree,
  so the gap has no reachable consequence that this pass could construct. The
  change was written, tested and reverted rather than shipped unverified.
- **Verdict:** ACCEPT (revisit -- see correction)
- **Outcome:** **STOPPED -- no change made.** See the correction note above; the claim
  this finding rests on did not survive re-testing, and the Verdict needs
  revisiting before anything is changed.

---

### F-3.3 — `derive_speeds` resolves duplicate timestamps by input row order, silently changing reported speeds
- **Severity:** S1
- **Location:** `src/redlight/speeds.py:286` (`sub.sort_values("time")`) and
  `src/redlight/speeds.py:356-358` (the `dt_s <= 0` skip)
- **Claim:** When two fixes in one trajectory share a timestamp, the zero-length
  interval is skipped and `i` advances past it, so exactly one of the tied
  fixes survives to anchor the neighbouring intervals — and *which* one is
  decided by the order the rows happen to sit in. Reordering the two tied rows
  of an otherwise identical file changes `distance_m`, `speed_mps` and the
  aggregated mean, with `quality=True` throughout and no warning anywhere.
  `pandas.sort_values` also defaults to `kind="quicksort"`, which is not
  stable, so for ≥100 tied rows the outcome additionally depends on pandas'
  internal sort.
- **Evidence:** Fully public API — `load_points` → `NearestMatcher` →
  `derive_speeds` → `aggregate_speeds`. Four GPS fixes, two of which share one
  second (an entirely ordinary logger file); the only difference between the
  two runs is which of those two rows is written first:
  ```
  $ .venv/bin/python s5_tie_public_api.py
  Two orderings of the SAME four fixes (only the two tied rows swap):

    file order A (…0130, 0132…)
     point_id_from  point_id_to  dt_s  distance_m  speed_mps  quality
                 0            1  10.0  334.282573  33.428257     True
                 2            3  10.0  311.996181  31.199618     True
     aggregate_speeds -> [{'block_start_hour': 8, 'n': 2, ..., 'mean_speed': 72.28422000825398, ..., 'unit': 'mph'}]

    file order B (…0132, 0130…)
     point_id_from  point_id_to  dt_s  distance_m  speed_mps  quality
                 0            1  10.0  356.568045  35.656805     True
                 2            3  10.0  334.281653  33.428165     True
     aggregate_speeds -> [{'block_start_hour': 8, 'n': 2, ..., 'mean_speed': 77.2693381543401, ..., 'unit': 'mph'}]

  ==========================================================================
  Does load_points itself flag or deduplicate the tied timestamps?
    rows in file: 4   rows kept: 4
    warnings emitted: []
    rows sharing (traj_id, time): 2
  ```
  With a larger tied block the spread is much wider — 8 fixes sharing one
  timestamp, four input permutations, four different answers:
  ```
  $ .venv/bin/python s4_tie_determinism.py
    as given          : n=1  distance_m=(222.853925,)   from=(7,) to=(8,)
    reversed tied block: n=1  distance_m=(1782.842849,) from=(0,) to=(8,)
    shuffled tied block: n=1  distance_m=(891.418152,)  from=(4,) to=(8,)
    shuffled again     : n=1  distance_m=(1114.273712,) from=(3,) to=(8,)
    distinct outcomes across 4 input orderings: 4
    NOT DETERMINISTIC -- output depends on input row order
  ```
  And `sort_values` is not stable at scale:
  ```
    n=   10 tied rows: sort_values preserved input order = True
    n=  100 tied rows: sort_values preserved input order = False
    n= 1000 tied rows: sort_values preserved input order = False
    n= 5000 tied rows: sort_values preserved input order = False
  ```
- **Expected vs actual:** Two positions stamped at the same instant are
  contradictory data and cannot both be true; the package should say so or
  resolve them by a stated rule. Expected: a deterministic, documented policy
  (e.g. stable sort plus "the first fix at a timestamp wins", or a warning
  naming the affected trajectory). Actual: a silently order-dependent answer —
  72.28 mph vs 77.27 mph from the same four fixes.
- **Suggested fix:** Pass `kind="stable"` to the sort so the result is at least
  reproducible, and detect duplicate `(traj_id, time)` pairs at the top of
  `derive_speeds`, warning with the count and the trajectory ids. Document the
  tie-breaking rule in the module docstring's Quality section.
- **Verdict:** ACCEPT
- **Outcome:** fixed (07d5bc4)

---

### F-3.4 — `geodesic_distance` returns a plausible finite distance for out-of-range latitudes where the oracle returns NaN
- **Severity:** S1
- **Location:** `src/redlight/_geo.py:47-53`; the same class of behaviour at
  `src/redlight/_proj.py:176-200` (`utm_forward`)
- **Claim:** For `|lat| > 90` the formula silently reflects the point across the
  pole and returns a confident number. `pyproj.Geod` returns NaN for the same
  input. Latitude 91 and latitude 89 return the *identical* distance, so the
  out-of-range value is indistinguishable from a valid one in the output. This
  is the signature of a lat/lon swap or a corrupt coordinate column, and the
  package reports a number for it rather than a NaN or an error.
- **Evidence:**
  ```
  $ .venv/bin/python g3_geo_reachable.py
  == B. Out-of-range latitude: silent plausible number vs pyproj NaN ==
  case                                               redlight (m)         pyproj (m)
  lat 91 vs lat 0  (lat/lon swap symptom)          9890271.864398                nan
  lat 100 vs lat 0                                 8885139.871936                nan

  == C. lat 91 round trip: what latitude is it silently treating it as? ==
    lat=  91.0 -> distance to equator =   9890271.8644 m
    lat=  89.0 -> distance to equator =   9890271.8644 m
    lat=  95.0 -> distance to equator =   9443510.1407 m
    lat=  85.0 -> distance to equator =   9443510.1407 m
    lat= 100.0 -> distance to equator =   8885139.8719 m
    lat=  80.0 -> distance to equator =   8885139.8719 m
  ```
  The reflection is not even self-consistent: a point at (lon 0, lat 91) is
  physically (lon 180, lat 89), whose distance to (0, 0) is ~10,113 km, not the
  9,890 km returned. `utm_forward` behaves the same way — at lat 90.5 it
  returns a finite easting/northing where pyproj returns `inf`, and the round
  trip is off by 19,900 km:
  ```
  $ .venv/bin/python p2_utm_edges.py
  == D. Outside the valid UTM latitude band (>84N / <80S) ==
    lat=  90.0 -> x=    500000.000 y=     9997964.943  roundtrip err=0.000e+00 m  vs pyproj=0.000e+00
    lat=  90.5 -> x=    500000.000 y=     9942140.306  roundtrip err=1.989e+07 m  vs pyproj=inf
  ```
- **Expected vs actual:** Per `pyproj.Geod` (the oracle this pass is instructed
  to use), `|lat| > 90` is not a position and the answer is NaN. Expected NaN
  or a `ValueError` naming the offending coordinate; actual a plausible
  kilometre-scale distance that no downstream check can distinguish from a real
  one. Note `_geo` is reached from `analysis.network_stats`
  (`analysis.py:417`), `points._derive_speed_mps` (`points.py:489`) and
  `cleaning` (`cleaning.py:217-234`).
- **Suggested fix:** Range-check latitude in `geodesic_distance` and return NaN
  (matching the oracle, and composable with the existing NaN handling in
  `cleaning._dwell_mask`) for `|lat| > 90`. **Triage note:** reaching this
  requires already-invalid input, and Task 5 owns input validation at the
  loader boundary; the severity here reflects the *silence* of the failure, and
  may reasonably be downgraded if Task 5 adds a loader-level range check
  instead.
- **Verdict:** DEFER -- to Task 5. The failure is genuinely silent, but it needs already-invalid
  coordinates, and the right place to reject `|lat| > 90` is once at the loader
  boundary, not inside `geodesic_distance`, which is called per point pair from
  `cleaning` and `points`.
- **Outcome:** deferred to Task 5, which owns validation at the loader boundary

---

### F-3.5 — A single non-finite coordinate raises a "near-antipodal" error and kills the whole vectorised batch
- **Severity:** S2
- **Location:** `src/redlight/_geo.py:80-90`
- **Claim:** The convergence test is `np.all(np.abs(lam - lam_prev) < _TOL)`.
  `NaN < tol` is False, so any non-finite coordinate never converges and the
  function raises the near-antipodal `ValueError` — a message that names the
  wrong cause and points the user at the wrong fix. Because the test is
  `np.all`, one bad pair anywhere in an array also aborts every good pair in
  the same call, where `pyproj` returns per-element NaN.
- **Evidence:** Reachable from the public API with one missing longitude:
  ```
  $ .venv/bin/python -c "import redlight; redlight.load_points('nan_pts.csv', id_col='traj_id', derive_speed=True)"
    File ".../redlight/points.py", line 398, in load_points
      out["speed_mps"] = _derive_speed_mps(out)
    File ".../redlight/points.py", line 489, in _derive_speed_mps
      dist = geodesic_distance(
    File ".../redlight/_geo.py", line 85, in geodesic_distance
      raise ValueError(
  ValueError: geodesic_distance failed to converge: the input contains a
  near-antipodal coordinate pair (points on opposite sides of the globe).
  Check the input coordinates -- a road network should never contain one.
  ```
  Batch poisoning:
  ```
  $ .venv/bin/python g2_geo_degenerate.py
  == 5. Vectorised: one NaN pair poisons the whole batch ==
  two good + one NaN, as one array   -> ValueError: geodesic_distance failed to converge: ...
  pyproj on the same array: [ 71.69575362 143.39150723          nan]
  ```
  Genuine near-antipodal pairs *are* handled correctly and safely — they raise
  rather than returning an unconverged value, exactly as the docstring
  promises, and normal road-scale pairs converge in 3 iterations.
- **Expected vs actual:** Expected an error that names the real problem ("input
  contains non-finite coordinates at rows …") or per-element NaN as pyproj
  gives. Actual: a confident, specific and wrong diagnosis that sends the user
  looking for antipodal coordinates that are not there.
- **Suggested fix:** Screen non-finite inputs before the iteration and either
  return NaN for those elements (leaving the rest to converge) or raise naming
  non-finite coordinates. `cleaning._dwell_mask` (`cleaning.py:257-268`) already
  works around this by substituting (0, 0) and masking afterwards, which shows
  the sharp edge is known; `points._derive_speed_mps` does not.
- **Verdict:** ACCEPT
- **Outcome:** fixed (f1f618c)

---

### F-3.6 — Mixed ISO-8601 timestamp spellings in one file silently drop rows
- **Severity:** S2
- **Location:** `src/redlight/points.py:192-198` (`_parse_times`)
- **Claim:** `pd.to_datetime(values, errors="coerce")` infers one format from
  the first element and coerces every non-matching value to `NaT`; those rows
  are then dropped with a generic "missing/unparseable" warning. But
  `2024-03-05T08:00:00` and `2024-03-05 08:00:10` are both perfectly valid and
  both parse fine on their own — the warning's diagnosis is wrong, and the
  silent thinning of a trajectory changes every speed derived from it.
  Whole-second and fractional-second stamps in the same column collide the same
  way, which is the common case for loggers that only print `.%f` when non-zero.
- **Evidence:**
  ```
  $ .venv/bin/python -c "..."   # three-row CSV, one column of timestamps
  all T-form         kept 3/3   warnings=[]
  all space-form     kept 3/3   warnings=[]
  T then space       kept 1/3   warnings=['Dropped 2 row(s) with missing/unparseable lon/lat/time.']
  space then T       kept 1/3   warnings=['Dropped 2 row(s) with missing/unparseable lon/lat/time.']
  T + fractional     kept 2/3   warnings=['Dropped 1 row(s) with missing/unparseable lon/lat/time.']

  raw pandas for the mixed case:
  0   2024-03-05 08:00:00
  1                   NaT
  dtype: datetime64[us]
  pandas version: 3.0.5
  ```
- **Expected vs actual:** Expected all three rows parsed — every value is
  unambiguous ISO 8601. Actual: two thirds of the trajectory discarded behind a
  warning that blames missing data.
- **Suggested fix:** Pass `format="mixed"` (pandas ≥2.0) in `_parse_times`, or
  retry with it when the first pass produces `NaT` for values that are
  individually parseable. **Scope note:** `points.py` is Task 5's file, not this
  pass's; recorded here because it was demonstrated while building the
  `derive_speeds` harness, and it corrupts derived speeds. Route to Task 5 if
  you would rather keep the files aligned with the passes.
- **Verdict:** ACCEPT
- **Outcome:** fixed (708ed94)

---

### F-3.7 — A `NaT` timestamp reaching `derive_speeds` yields `dt_s = NaN` and `speed_mps = NaN` rows
- **Severity:** S5
- **Location:** `src/redlight/speeds.py:355-358`
- **Claim:** The guard is `if dt_s <= 0`, and `NaN <= 0` is False, so a `NaT`
  timestamp produces an interval with `dt_s = NaN`, `speed_mps = NaN` and
  `quality = False` in both output frames rather than being skipped.
- **Evidence:**
  ```
  $ .venv/bin/python s3_edge_cases.py
  == E. NaT timestamp reaching the dt guard ==
    intervals emitted: 2
   interval_id  dt_s  distance_m  speed_mps  quality
             0  20.0  401.138977  20.056949     True
             1   NaN  200.569323        NaN    False
    any NaN speed? True
  ```
- **Expected vs actual:** Expected the interval to be skipped exactly as a
  zero or negative `dt` is. Actual: a NaN row is emitted. Impact is contained —
  `load_points` drops `NaT` rows before this, and `aggregate._usable_rows`
  filters `speed_mps.isna()` — so this is hygiene, not a wrong number. Reaching
  it required bypassing the loader with a hand-built frame.
- **Suggested fix:** Change the guard to `if not (dt_s > 0)`, which rejects NaN
  and zero and negative alike in one expression.
- **Verdict:** ACCEPT
- **Outcome:** fixed (5fc7017)

---

## Non-paper HMM behaviours — verdicts

The pass was asked to decide, for each, principled extension or bug. Decoded
paths with and without each behaviour are in `m2_behaviours.py` and
`m3_anchor_flip.py`.

**(a) Same-edge transition shortcut (`matching.py:296-303`) — principled in
intent, defective in implementation.** It is genuinely load-bearing: without
it, six fixes driving straight along one two-way road decode as
`[1, 0, 1, 0, 1, 0]`, alternating direction every fix. But it over-reaches on
curved edges — see **F-3.2**.

**(b) Saturating penalty when no predecessor is reachable (`matching.py:320-332`)
— principled extension, keep.** The penalty is `_transition_logp(cutoff)`, and
since any within-cutoff transition has `delta ≤ cutoff`, it is a strict lower
bound on the score any legal transition could have earned — a defensible
"worst legal transition" rather than an arbitrary constant. It is applied
uniformly to every candidate at that step and connects them all to the same
best predecessor, so it does not distort the ranking among candidates; the
emission alone decides, exactly as the inline comment claims. Measured effect:
```
  network: two DISCONNECTED roads ~33 km apart
  trajectory: 3 fixes on road 1, teleport, 3 fixes on road 2
    AS SHIPPED (penalty on)            edge_id=[1, 1, 1, 3, 3, 3]
    penalty OFF (free restart)         edge_id=[-1, -1, -1, 3, 3, 3]
  -> with the penalty every fix is matched: 6/6
  -> with a free restart only              : 3/6
```
On a connected network with a routable gap the two agree, so the behaviour only
engages where it is needed. It does assert continuity across a genuinely
disconnected jump, but that does not leak a wrong speed:
`speeds._hop_distance` independently returns `ok=False` for such a pair and the
interval is never emitted (`s1_hop_distance.py`, case 9).

**(c) Carrying state across candidate-less fixes with `anchor`
(`matching.py:256-270`) — principled extension, keep, and load-bearing.**
Measuring `gc_step` from the last fix that actually anchored a state, rather
than from the raw off-network position of the previous row, changes the
transition score by 52 nats and the search cutoff by 3.3× on a 2 km excursion:
```
    gc_step from anchor (fix 1, last on-road)  =     668.57 m
    gc_step from fix 3  (2 km off-network)     =    2224.83 m
    TRUE on-road distance fix 1 -> fix 4       =     668.57 m
    |gc-rd| using anchor  =       0.00 m  -> logp   -3.401
    |gc-rd| without       =    1556.26 m  -> logp  -55.277
```
When the vehicle stays on one edge across the gap the decode is unchanged —
behaviour (a) shields it, since the same-edge branch never consults `gc_step`
for the route distance. When the vehicle *changes* edge across the gap, the
anchor changes the decoded path outright (`m3` follow-up, network = road A then
road C in series plus a parallel road D reachable only by a ~9 km detour):
```
  gap at (0.03, 0.02): gc(anchor)=  1226.4  gc(i-1)=   2883.3
     anchor ON  -> [0, 0, -1, -1, 2, 2]      (east on A, then east on C — correct)
     anchor OFF -> [1, 1, -1, -1, 2, 2]      (westbound A — wrong direction)
  gap at (0.08, 0.06): gc(anchor)=  1226.4  gc(i-1)=   9965.1
     anchor ON  -> [0, 0, -1, -1, 2, 2]
     anchor OFF -> [1, 1, -1, -1, 5, 5]      (wrong direction AND wrong road)
```

---

## Unverified suspicions

Recorded but **not demonstrated**; do not act on these without evidence.

1. **`beta = 30.0` may not be "a robust value from the source paper's
   calibration"** (`matching.py:135-138`). The emission and transition *forms*
   match Newson & Krumm exactly, and `sigma_z = 6.0` sits inside the docstring's
   own stated 4–10 m range against the paper's measured 4.07 m. But the paper's
   calibrated `beta` is, to the best of this pass's recollection, roughly an
   order of magnitude smaller than 30 m, and this pass had no access to the
   paper to check. Someone with the PDF should confirm the number and the
   sampling interval it was calibrated at; if it does not support "from the
   source paper's calibration", that is an S3 against the docstring.

2. **Whether the ~10% boundary shift from F-3.1 materially moves published
   per-edge speeds.** F-3.1 measures the fix-level decode difference on a
   synthetic grid; the decoded *route* was identical in every trial, only the
   fix-to-edge assignment near turns moved. Whether that changes
   `aggregate_speeds` output enough to matter — and whether the effect is larger
   on ambiguous real geometry (dual carriageways, parallel service roads) than
   on a clean grid — was not measured. Task 4 is better placed to answer it on
   the real networks in `examples/sample_data`.

3. **`0.0 <= speed_mps` in the quality predicate (`speeds.py:380`) looks
   unreachable.** `distance_m` is a non-negative shortest-path length and `dt_s`
   is guaranteed positive by the preceding guard, so the lower bound can never
   fail. Harmless if so, but it was not proved unreachable and may be defending
   against a case this pass did not construct.

4. **`_SourceDistCache` and `_CSRDistCache` reuse a cached result whenever the
   stored cutoff is ≥ the requested one and re-apply the caller's cutoff on
   hits**, which reads as order-independent and was spot-checked as such by the
   `_hop_distance` cutoff sweep. A dedicated order-independence test across many
   interleaved cutoffs was not written; Task 6 should consider one.
