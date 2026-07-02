# Statistical methodology

This document specifies exactly how `roadtraffic` produces every quantity it
reports, the assumptions behind each method, and when to prefer one option over
another. The goal is that a user can defend any number this package outputs.

All internal computation is performed in **metres per second (m/s)**. Input and
output units (mph / kph / m/s) are exact conversions
(`1 mile = 1609.344 m`, `1 km = 1000 m`) and introduce no rounding beyond
floating point.

---

## 1. Coordinate handling and distance

GPS coordinates are stored in WGS84 (EPSG:4326, degrees) but **all distance and
geometry math is performed in a projected metric CRS**, never in degrees.

- By default the package selects the **UTM zone** containing the network's
  first vertex (`32600 + zone` in the northern hemisphere, `32700 + zone` in the
  southern). UTM is conformal and accurate to well under 0.1% scale distortion
  within a zone — appropriate for city- to regional-scale studies.
- You may override this with `metric_epsg=` if your study area spans zones or
  you have a preferred local projection.

**Implication.** Snap distances, edge lengths, and route distances are true
ground metres, not degree approximations. Studies spanning more than a few UTM
zones should be split or use an equal-distance projection suited to the extent.

---

## 2. Matching GPS points to road edges

A speed observation only becomes useful once it is attributed to a specific road
segment (edge). Two methods are provided; both emit identical output so all
downstream statistics are method-agnostic.

### 2.1 Nearest-edge snapping (`NearestMatcher`)

Each point is matched **independently** to the edge minimising the
perpendicular point-to-segment distance, subject to a tolerance `max_dist`
(metres). Candidate retrieval uses a `scipy.spatial.cKDTree` over segment
midpoints for speed; the exact foot-of-perpendicular distance is then computed
for the shortlist.

- **Assumption.** The nearest edge is the correct edge.
- **Strengths.** Fast, O(N log M) for N points and M segments; no trajectory
  assumption, so it works on unordered point clouds.
- **Weaknesses.** Ambiguous at intersections, divided highways, and parallel
  service roads, where the geometrically nearest edge may not be the travelled
  one.

### 2.2 HMM / Viterbi map matching (`HMMMatcher`)

Implements the hidden Markov model of **Newson & Krumm (2009)**, the method
underlying production matchers such as OSRM and Valhalla. It uses the *sequence*
of points in a trajectory (hence a trajectory id is required) to find the most
probable sequence of edges.

For each observation `i`, candidate edges are the states. Two probabilities are
combined:

**Emission probability** — how likely observation `i` arose from candidate edge
`c`, modelled as a zero-mean Gaussian in the perpendicular snap distance
`d(i,c)`:

```
p_emit(i, c)  ∝  exp( -0.5 * (d(i,c) / σ_z)^2 ) / (sqrt(2π) · σ_z)
```

`σ_z` is the GPS measurement noise standard deviation in metres (`sigma_z`,
default 6 m; consumer GPS is typically 4–10 m).

**Transition probability** — how plausible it is to move from candidate `c` at
point `i` to candidate `c'` at point `i+1`, comparing the straight-line distance
between the two GPS points to the on-network shortest-path distance between the
candidates:

```
Δ = | dist_gc(i, i+1) − dist_route(c, c') |
p_trans(c, c')  ∝  exp( −Δ / β ) / β
```

`β` (`beta`, default 30 m) controls tolerance to the difference; the default is
the robustly calibrated value from the source paper. The route distance term is
the expensive part — it requires shortest-path queries on the graph — but adds
**no new dependency** beyond `networkx`.

The most probable edge sequence is recovered by **Viterbi decoding** in log
space (sums of log-probabilities, avoiding underflow). The transition
shortest-path search is bounded at `max(max_route_dist_factor × straight-line
step, max_dist × 4)` — the floor keeps slow-moving (small-step) trajectories
from pruning every transition — and staying on the same edge is always allowed.
When no predecessor is reachable within that bound, the chain connects to the
best prior state with a *saturating* penalty (a free restart would outscore any
long continuous chain and truncate the decode); when there is no prior state at
all (the leading fixes had no candidate edges), the chain restarts fresh at the
first fix that has candidates. Fixes with no candidate edge within `max_dist`
are always reported as `edge_id = -1` — the HMM never fabricates a match for
them.

- **Assumption.** Points within a trajectory are time-ordered and reasonably
  frequent relative to the network's edge lengths.
- **Strengths.** Resolves intersection/parallel-road ambiguity; far fewer
  mismatches.
- **Weaknesses.** Higher compute cost (shortest-path calls); needs trajectory
  ids; very sparse pings (e.g. > 1 km apart) weaken the transition signal.

**Choosing.** Use `NearestMatcher` for quick looks, dense data, or sparse simple
networks. Use `HMMMatcher` when accuracy matters and you have ordered
trajectories — especially in dense urban grids.

---

## 3. Speed cleaning

### 3.1 Hard physical bounds

`min_speed` / `max_speed` remove observations outside a plausible band for the
study (e.g. drop 0 mph parked pings; drop > 80 mph GPS-jump artefacts on city
streets). These are **domain decisions**, not statistical ones, and should be
justified by the road class and vehicle type under study.

### 3.2 Robust outlier removal — modified Z-score (MAD)

Optional (`mad_outliers=True`). An observation is flagged an outlier when its
**modified Z-score** exceeds a threshold (default 3.5; Iglewicz & Hoaglin,
1993):

```
M_i  = 0.6745 · (x_i − median(x)) / MAD
MAD  = median( | x_i − median(x) | )
```

- The **median** and **MAD** are used instead of the mean and standard
  deviation because they have a ~50% breakdown point: up to half the data can be
  contaminated before the estimate is corrupted. A handful of wild GPS speed
  errors will not inflate the threshold and mask themselves, which is the
  failure mode of mean/SD-based screening.
- The constant **0.6745** scales MAD to estimate the standard deviation
  consistently for normally distributed data, so the threshold 3.5 is
  interpretable as "≈ 3.5 robust standard deviations".
- If `MAD = 0` (≥ half the values identical), no robust scale exists and **no**
  points are dropped, rather than dropping everything.

Set `per_edge=True` to screen within each edge group. This is recommended when
edges have very different free-flow speeds (a motorway and a residential street
should not share one outlier threshold).

`filter_by_speed` treats every observation independently, which is the right
model for `NearestMatcher` output. For HMM-matched trajectories there is a better,
trajectory-aware option that does **not** silently delete congestion — see
§8.

---

---

## 4. Temporal aggregation

Observations are binned by **hour of day** (`block_hours=1`, 24 bins) or by an
**N-hour block** (`block_hours=2,3,4,6,8,12`). Values that do not divide 24
evenly are allowed but produce a narrower final block and emit a warning. For
each bin the package reports one or both summaries.

### 4.1 Mean (`statistic="mean"` or `"both"`)

For `n` observations with sample mean `x̄`:

- **Sample standard deviation** `s` with Bessel's correction (`ddof=1`):
  the unbiased estimator of the population SD; reported as `std_speed`.
- **Standard error of the mean** `SEM = s / √n` (`sem_speed`): the uncertainty
  in the *bin's mean speed*, distinct from the spread of individual speeds.
- **95% confidence interval** `x̄ ± 1.96 · SEM` (`ci95_low`, `ci95_high`):
  a normal-approximation interval for the true mean speed of that bin.

**Assumptions.** The mean/SD/SEM summary is most appropriate when per-bin speeds
are roughly symmetric and not heavily outlier-laden — pair it with MAD cleaning.
The 1.96 multiplier is the large-sample normal approximation; for small `n` the
interval is mildly optimistic (a t-multiplier would be wider). `n` is always
reported so you can judge.

### 4.2 Median (`statistic="median"` or `"both"`)

- **Median** (`median_speed`): the 50th percentile, robust to skew and outliers.
- **Quartiles** `Q1`, `Q3` (`q1_speed`, `q3_speed`) and **interquartile range**
  `IQR = Q3 − Q1` (`iqr_speed`): a robust measure of spread.

**When to prefer the median.** Congested-flow speed distributions are frequently
right- or left-skewed and contain residual outliers; the median and IQR describe
such distributions more faithfully than the mean and SD. Use `"both"` to compare.

### 4.3 Sample-size suppression

`min_samples` drops bins with too few observations to be meaningful. There is no
universal minimum; for stable mean estimates, bins with `n < 30` should be read
with caution (the normal approximation for the CI weakens), and very small bins
(`n < 5–10`) are often better suppressed entirely.

---

## 5. Peak / off-peak detection

`peak_analysis` ranks network-wide time bins by the chosen speed statistic.
**Lower speed = heavier traffic = peak.** The slowest bins are returned as
`peak`, the fastest as `off_peak`, with the full ranking in `ranked`. This is a
descriptive ranking of the aggregated speeds, not a hypothesis test; to claim
two periods differ significantly, compare their confidence intervals or run an
appropriate test (e.g. Mann–Whitney U for medians) on the underlying
observations.

---

## 6. Edge speed assignment and routing

`assign_speeds` computes a representative speed per edge — **median**
(default, robust) or **mean** — optionally restricted to a time-of-day block via
`target_hour`/`block_hours`, enabling time-dependent routing. It writes
`obs_speed_mps` and `travel_time_s = length_m / obs_speed_mps` onto each edge.
Edges without observations receive `default_speed_mps` if supplied, else carry
no travel time.

`Router` runs **Dijkstra's algorithm** (`networkx.shortest_path`) which is exact
for non-negative edge weights — always satisfied by distance and time:

- `mode="distance"` minimises summed `length_m`.
- `mode="time"` minimises summed `travel_time_s`, falling back to
  `length_m / default_speed_mps` on unobserved edges.
- `mode="cost"` minimises a user-supplied `cost_func(u, v, data)` — define any
  non-negative weight (e.g. risk, fuel, a blend of time and turns).

**Caveat.** Time routing is only as good as the observed speeds. Edges with no
data fall back to a default; routes through sparsely observed areas should be
interpreted accordingly. Report coverage using the
`n_edges_observed / n_edges_total` returned by `assign_speeds`, or the
per-route `n_edges_default` returned by `Router.route`. For the three-regime
(overall / peak / off-peak) edge speeds and time-of-day routing, see §9.

When no path exists, `Router.route` raises a `ValueError` that distinguishes a
**one-way trap** (the endpoints are connected only against the legal direction of
travel) from a **disconnected network** (the endpoints lie in different
components, typical of a clipped Overpass extract), so the failure is actionable
rather than an opaque graph exception.

---

## 7. Deriving speed from GPS positions

When a feed carries no speed column, speed can be reconstructed from the motion of
each tracked unit (`load_points(derive_speed=True)`, opt-in). A **unique id is
required** so speeds are only ever differenced between successive points of the
*same* movement, never across two different vehicles.

Within one trajectory, sorted by time, the speed at point `i` is

```
v_i = d_geo(p_{i-1}, p_i) / (t_i − t_{i-1})
```

where `d_geo` is the **geodesic (ellipsoidal WGS84) distance** computed directly
from lon/lat with an inverse geodesic — no projection, so it is exact ground
distance and free of UTM-zone edge effects. `v_i` is therefore the *average*
speed over the interval **ending at** `i`; the first point of each trajectory
inherits the speed of its first interval.

**Guards.** A non-positive time gap (duplicate or out-of-order timestamps) or a
missing coordinate/time yields `NaN`, and such points are dropped downstream with
the usual warning. Differencing never crosses a trajectory boundary.

**Interpretation and limits.** Because each value is an interval average, the
result is only as fine-grained as the ping spacing:

- Coarse spacing **smooths** the speed profile — brief stops and short bursts are
  averaged away, so derived peak speeds are biased low and derived congestion is
  understated relative to an instantaneous on-board reading.
- The straight (geodesic) chord between pings is **shorter than the road arc** on
  curves, biasing speed slightly low where the path bends between pings.
- Both effects shrink as ping frequency rises. Derived speed is best treated as a
  defensible estimate for trafficability patterns, not a calibrated speedometer.

If you're matching anyway (§2), `roadtraffic.speeds.derive_speeds` (§10) fixes
both limitations by measuring displacement on the road graph after matching,
and is the recommended choice for noisy or sparsely-sampled GPS.

---

## 8. Trajectory-aware cleaning: dwell vs. congestion

A flat "drop everything below *x* mph" rule is actively harmful to a
trafficability study: the slow observations it deletes are exactly the congestion
the study exists to measure, and removing them biases every segment's speed
**upward** (the road looks more trafficable than it is). For independent
`NearestMatcher` points a speed floor is still acceptable (each point stands
alone), but for HMM-matched **trajectories** `filter_trajectory_speed` uses the
movement itself to separate two genuinely different things:

- **Slow but moving** — a unit crawling through congestion still covers ground
  over time. It is **kept**, low speed and all, because the trajectory confirms it
  is traversing the segment.
- **Stationary dwell** — parked, idling, or GPS jitter while stopped. This is not
  travel on the segment and is **dropped**.

A dwell is detected as a maximal run of consecutive same-trajectory points (time
ordered) all within `dwell_radius_m` (geodesic) of the run's first point and
spanning at least `dwell_min_s`. Equivalently, a unit is treated as stopped only
if its average speed over the window stays below `dwell_radius_m / dwell_min_s`;
with the defaults (25 m, 120 s) that floor is ≈ 0.2 m/s (≈ 0.47 mph), so ordinary
signal stops and stop-and-go survive while sustained parking is removed. Raise
`dwell_min_s` to keep longer operational stops; lower it to be stricter. Points
with no logged speed are also dropped (`drop_missing_speed`), and an optional
`max_speed` ceiling discards GPS-jump artefacts. There is deliberately **no**
`min_speed`.

---

## 9. Peak / off-peak segment speeds and time-of-day routing

Splitting a sparse GPS feed into 24 hourly per-edge speeds usually leaves most
(edge, hour) cells empty or with too few points to be stable — the common cause of
unstable or failing time routing. Instead, hours are pooled into two broad blocks
so each segment's statistics rest on many more observations.

**Classifying the blocks (`classify_hours`).** By default the split is
data-driven: the network-wide representative speed (median or mean) is computed
for each hour of day, and hours **at or below the median** of those hourly speeds
are labelled *peak* (slower ⇒ busier), the rest *off-peak*. The split adapts to
each dataset and naturally captures non-contiguous peaks (a morning and an evening
rush). Pass explicit `peak_hours` / `offpeak_hours` to override.

**Three per-segment speeds (`assign_segment_speeds`).** For every edge the median
(default) or mean speed is computed over (a) the whole timeframe, (b) the peak
block, and (c) the off-peak block, written as
`obs_speed_mps_{overall,peak,offpeak}` with matching travel times. Coverage per
regime is returned, and a warning is emitted if a regime observed no edges.

**Routing on a block (`Router.route(period=…)`).** Time routing uses the chosen
regime's travel time per edge, falling back to the overall speed where a regime
has no data for an edge, and to the default speed where the edge has no data at
all. The fallback count (`n_edges_default`) is reported so a route built largely
on defaults can be flagged rather than trusted blindly.

---

## 10. On-road speed derivation from matched trajectories

Section 7 derives speed from the *straight-line* (geodesic) chord between
consecutive fixes, before any matching. `roadtraffic.speeds.derive_speeds`
instead derives speed **after** map matching, from **on-road displacement**
between consecutive matched fixes:

```
speed over [t_i, t_{i+1}]  =  on-road distance(i -> i+1) / (t_{i+1} - t_i)
```

This is more accurate for noisy, sparsely-sampled GPS (the regime this method
targets), at the cost of needing a network and a matching step first. Prefer
it whenever you have both; fall back to `load_points(derive_speed=True)`
(§7) when you don't want to match first, or have no network at all.

**On-road distance is measured along the graph, not as the crow flies.**
Given the matched edge and arc-length snap position of each fix, the distance
from fix *i* to fix *i+1* is the remaining length of edge A after the snap,
plus the graph shortest-path length from A's downstream node to B's upstream
node, plus the length of edge B up to its snap (or just the difference of
arc-length positions when both fixes land on the same directed edge).

**Distance is measured on the *undirected* graph.** Speed is a magnitude, so
direction is dropped for the distance calculation — this makes the estimate
robust to the matcher flip-flopping between the two directed edges of a
two-way road (a common ambiguity: identical geometry gives tied emission
probabilities), which would otherwise inflate distance by a full edge length
per fix. Direction still decides *which road* a fix is on; it no longer
decides the distance. Each observation is attributed to **both** directed
edges of every road it traverses (one edge for one-way roads), populating
both directions for routing.

**Speed is a property of the interval, not a fix.** The value above is the
*mean* speed over the interval; it is not recovered as an instantaneous speed
at either endpoint (differentiating a noisy trajectory to do that would
amplify GPS error). Each interval's speed is attributed to every edge it
traversed. `derive_speeds` returns two DataFrames:

- `intervals` — one row per consecutive fix pair (per-point speed record),
  with `time` set to the interval **midpoint** (when the speed applies).
- `edge_observations` — long format, one row per (interval, traversed edge).
  Schema-compatible with `filter_by_speed`, `aggregate_speeds` and
  `assign_speeds` — feed it straight in.

### Noise model and the quality flag

The estimate's noise is dominated by GPS position error over the time gap:

```
sigma_v ≈ sqrt(sigma_i² + sigma_j²) / dt
```

where `sigma_i`, `sigma_j` are the two fixes' position sigmas (`pos_accuracy_col`
if you have per-fix accuracy, else `default_pos_sigma_m` for all fixes). An
interval is flagged `quality=False` — but still returned, for transparency —
when any of the following hold:

- the displacement is not clearly above the noise floor:
  `distance_m < min_snr * sigma_combined` (default `min_snr=3`, i.e. roughly
  ≤ 33% relative speed error is required to trust the interval);
- `dt` falls outside `[min_dt_s, max_dt_s]`;
- either fix's snap distance exceeds `max_snap_dist_m`;
- the implied speed falls outside `[0, max_speed_mps]`.

Keep `quality` rows for aggregation; `speed_var` (`= speed_sigma_mps ** 2`)
also supports inverse-variance weighting downstream.

### Adaptive baseline (`min_baseline_m`) — required for noisy, sparse fixes

When fixes are close together in time and GPS is noisy, single-interval
displacement can be comparable to the noise floor. In that regime the quality
filter above *selects for* noise-inflated, too-fast intervals — a systematic
bias, not just added scatter, because noise almost never displaces a fix
*backwards* enough to cancel out.

The fix is to merge consecutive hops until their summed on-road distance
reaches `min_baseline_m`, then compute `speed = Σdistance / Σdt`. Pick the
baseline at roughly **5–10× the GPS sigma**. Illustrative validation on 1 Hz
fixes with 5 m noise (true speed 10.0 m/s):

| baseline | derived median |
|---|---|
| none (per-fix) | 23.0 m/s ← biased high |
| 50 m | 11.6 m/s |
| 75 m | 10.2 m/s |
| 100 m | 10.2 m/s |

The cost of raising the baseline is coarser spatial resolution per speed
sample (a longer span touches more edges) — an acceptable trade for
edge-average trafficability. Leave `min_baseline_m=None` (the default, one
interval per fix pair) only when fixes are already well separated relative to
GPS noise.

### Bounded transition search (`max_route_dist_factor`, `route_cutoff_floor_m`)

The shortest-path search between a hop's exit and entry candidates is bounded
at `max(max_route_dist_factor * straight_step, route_cutoff_floor_m)`; beyond
that the hop is treated as unreachable rather than paying an unbounded
Dijkstra per candidate. This guards both compute cost on large networks and
against silently bridging real gaps (e.g. a missing road segment) with an
implausible detour.

---

## References

- Newson, P., & Krumm, J. (2009). *Hidden Markov Map Matching Through Noise and
  Sparseness.* ACM SIGSPATIAL GIS.
- Iglewicz, B., & Hoaglin, D. C. (1993). *How to Detect and Handle Outliers.*
  ASQC Quality Press. (Modified Z-score, cutoff 3.5.)
- Dijkstra, E. W. (1959). *A Note on Two Problems in Connexion with Graphs.*
  Numerische Mathematik.
