# Deriving speed from position-only GPS — `roadtraffic.speeds`

Your GPS fixes carry a trajectory id and a timestamp but **no speed**. This adds a
`speeds` module that derives speed from on-road displacement between consecutive
fixes, plus three small patches so the existing loader and matchers accept
speed-less data. The output of speed derivation is **schema-compatible with your
existing `cleaning` / `aggregate` / `routing` modules** — it flows straight in.

## What changed

| File | Change | Why |
|---|---|---|
| `speeds.py` | **new module** | derive per-interval and per-edge speed |
| `points.py` | speed column made **optional** in `load_points` | position+time data must load |
| `matching.py` | matchers carry `speed_mps` only **if present**; `_net_dist` now honors its `cutoff` | match speed-less data; make `max_route_dist_factor` actually prune |

Nothing else in your package was modified. The matcher **output schema is
unchanged** — `speeds.derive_speeds` recovers lon/lat by joining the matched frame
back to your `PointSet` on `point_id`, so it needs no new columns from the matcher.

## The recommended pipeline

```python
from roadtraffic.network import Network
from roadtraffic.points import load_points
from roadtraffic.matching import HMMMatcher
from roadtraffic.speeds import derive_speeds
from roadtraffic.cleaning import filter_by_speed
from roadtraffic.aggregate import aggregate_speeds, assign_speeds
from roadtraffic.routing import Router

net = Network.from_geojson("roads.geojson")
pts = load_points("fixes.csv", id_col="track_id")          # no speed column needed

# 1) map-match first (HMM is direction-aware and robust to noise)
matched = HMMMatcher(net, sigma_z=20.0, beta=30.0, max_dist=150.0).match(pts)

# 2) derive speed from on-road displacement
res = derive_speeds(
    net, matched, pts,
    pos_accuracy_col="accuracy",   # per-fix metres, if you have it (see below)
    default_pos_sigma_m=20.0,      # fallback if you don't
    min_baseline_m=100.0,          # IMPORTANT for your data — see below
    max_speed_mps=60.0,
)
intervals = res["intervals"]            # per-point speed record (one row per fix pair)
edge_obs  = res["edge_observations"]    # long: one row per (interval, edge)

# 3) clean + aggregate per edge using your existing tools
clean = filter_by_speed(edge_obs, min_speed=1, max_speed=80, unit="mph",
                        drop_unmatched=True, mad_outliers=True, per_edge=True)
agg   = aggregate_speeds(clean, statistic="both", output_unit="mph", by_edge=True)

# 4) write speeds onto the graph and route by travel time
assign_speeds(net, clean, statistic="median", output_unit="mps")
route = Router(net).route(origin_lonlat, dest_lonlat, mode="time")
```

`intervals` is your **per-point speed** (the average speed over each
`[t_from, t_to]`, with `time` set to the interval midpoint). `edge_observations`
is what you aggregate for **per-edge trafficability**.

## Two things your data forces you to get right

Both were found and fixed by validating against ground truth; both matter for the
1–100 m accuracy / variable-Δt regime you described.

### 1. Distance is measured on the *undirected* graph
Speed is a magnitude, so the distance between two fixes is measured undirected.
This makes it **robust to the matcher flip-flopping between the two directed edges
of a two-way road** — a common ambiguity (identical geometry → tied emission
probabilities) that would otherwise inflate distance by a whole edge length per
fix. Direction still decides *which road* a fix is on; it no longer decides the
distance. Each observation is attributed to **both** directed edges of every
road it traverses (one edge for one-way roads), which populates both directions
for routing.

### 2. Use an adaptive baseline (`min_baseline_m`) — not optional for you
The speed error is dominated by GPS noise over the time gap:
`sigma_v ≈ sqrt(σ_i² + σ_j²) / Δt`. When fixes are close in time and GPS is
noisy, single-interval displacement is comparable to the noise floor, and the
quality filter (which keeps intervals whose displacement clears the noise) then
**selects the noise-inflated, too-fast intervals** — a bias, not just scatter.

The fix is to merge consecutive hops until their summed on-road distance reaches a
baseline, then compute `speed = Σdistance / Σdt`. Pick the baseline at roughly
**5–10× your GPS sigma**. Validation on 1 Hz fixes with 5 m noise (true 10.0 m/s):

| baseline | derived median |
|---|---|
| none (per-fix) | 23.0 m/s  ← biased high |
| 50 m | 11.6 m/s |
| 75 m | 10.2 m/s |
| 100 m | 10.2 m/s |

Given your accuracy spans 1–100 m, start around `min_baseline_m=100` and tune. If
much of your data is good (≤10 m) you can go lower; if you keep the worst fixes,
go higher. The cost is coarser spatial resolution per speed sample (a longer span
touches more edges), which is fine for edge-average trafficability.

## Per-fix accuracy — please check
You said accuracy ranges 1–100 m. **Does each fix carry its own accuracy/HDOP
value?** If so, pass that column as `pos_accuracy_col` — it lets the module weight
and quality-flag each interval by its actual uncertainty (a 1 m fix and a 100 m fix
are treated very differently) and exposes `speed_var` for inverse-variance
weighting downstream. If you only know the global range, leave it unset and set
`default_pos_sigma_m` to a representative value (e.g. 20–30 m for a mixed bag).

## HMM tuning for noisy GPS
The matcher defaults assume good GPS. For 1–100 m data, raise `sigma_z` (the GPS
noise term, try 15–25 m) and `max_dist` (snap search radius, try 100–150 m) so
poor fixes still match. `max_route_dist_factor` now genuinely bounds the
transition search, so larger networks won't pay an unbounded-Dijkstra cost per
candidate.

## Quality column
Every row carries `quality` (bool). It's `False` when displacement is below
`min_snr × σ`, when Δt is implausible, when the snap distance is too large, or when
the implied speed is out of range. Keep `quality` rows for aggregation; the
`mad_outliers=True` path in `filter_by_speed` plus median aggregation then handle
the residual outliers robustly.

## Reproduce the validation
`validate_speeds.py` checks the on-road distance/speed math against an L-shaped
network with fixes at known positions (0.000 % error on both the cornering and
same-edge cases). `integration_test.py` runs the whole chain end-to-end on a
simulated noisy trip and recovers the true speed.
