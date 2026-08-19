# Performance — measured baseline — findings
**Pass:** Task 7
**Date:** 2026-08-19
**Commit reviewed:** `a692d02`
**Scope:** `benchmarks/bench_matching.py`, `src/redlight/matching.py`,
`src/redlight/network.py`. Created: `benchmarks/profile_hmm.py`.
**Method:** `bench_matching.py` run at five sizes; a new `profile_hmm.py`
combining cProfile with direct instrumentation of the distance cache, the
per-(candidate, predecessor) network lookups, the Viterbi frontier width and
candidate-retrieval time; `tracemalloc` + `resource.getrusage` for memory.
**No optimization was written and `src/` was not touched.**

**Machine:** Intel Core i5-7360U @ 2.30 GHz, **2 physical / 4 logical cores**,
macOS (Darwin 22.6.0) — the machine `00-baseline.md` describes. Every number
below is relative to this hardware. Two physical cores is the floor case for
any parallelism claim, which matters for F-7.1.

## Summary

Two docstring claims were handed to this pass as unverified. One holds, one
does not: serial matching does sustain tens of thousands of points per second,
but **the claim that serial stays faster than `n_jobs>1` "up to at least two
million points" is false here — the crossover is between 500k and 1M**, four
times earlier than claimed, and that is on the *fewest* cores this claim will
ever face. The more consequential finding is not about the matcher at all:
**`derive_speeds` costs more than `HMMMatcher.match` on the same data — 60 % of
pipeline wall time against the matcher's 39 %** — so Task 8, which is scoped
entirely to the matcher, is aimed at the smaller half of the problem.

## 1. Throughput (`bench_matching.py`, 30×30 grid, 3,480 directed edges)

| Points | Trajectories | Nearest | HMM serial | HMM `n_jobs=4` | Serial vs parallel |
|---|---|---|---|---|---|
| 20,000 | 50 | 333,498 /s | 31,305 /s | 5,132 /s | serial **6.10×** faster |
| 200,000 | 200 | 346,592 /s | 77,773 /s | 41,194 /s | serial **1.89×** faster |
| 500,000 | 300 | 366,358 /s | 102,836 /s | 84,354 /s | serial **1.22×** faster |
| 1,000,000 | 500 | 414,678 /s | 113,436 /s | 123,826 /s | **parallel 1.09× faster** |
| 1,000,000 (repeat) | 500 | 424,040 /s | 112,536 /s | 120,555 /s | **parallel 1.07× faster** |
| 2,000,000 | 1000 | 433,600 /s | 115,666 /s | 138,299 /s | **parallel 1.20× faster** |

HMM throughput *rises* with size (31k → 116k /s) because longer trajectories
amortise per-process setup and warm the shortest-path cache. `NearestMatcher`
is 3.7× faster than the HMM at 2M and 10× faster at 20k.

## 2. Profile of `HMMMatcher.match`

`profile_hmm.py --points 8000 --trajectories 40`, 100 % of points matched.
cProfile inflates wall time ~14× (9.6 s profiled vs 0.67 s real), so these are
**relative attributions only**. Top entries by cumulative time:

| ncalls | tottime | cumtime | function |
|---|---|---|---|
| 1 | 0.044 | 9.611 | `matching.py:223(match)` |
| 40 | **3.791** | 9.507 | `matching.py:237(_match_one)` |
| 119,312 | 1.795 | **3.462** | `matching.py:99(lookup)` — bounded Dijkstra + cache |
| 487,393 | 0.755 | 0.755 | `dict.get` |
| 40 | 0.621 | 0.705 | `network.py:789(_candidate_arcs_batch)` |
| 131,697 | 0.624 | 0.624 | `builtins.abs` |
| 52,723 | 0.318 | 0.318 | `dict.items` |
| 119,312 | 0.318 | 0.318 | `OrderedDict.move_to_end` — LRU bookkeeping, one per lookup |
| 128,281 | 0.280 | 0.280 | `builtins.len` |
| 131,697 | 0.104 | 0.155 | `matching.py:208(_transition_logp)` |
| 34,042 | 0.063 | 0.106 | `matching.py:202(_emission_logp)` |

## 3. The specific questions, answered with measurements

Instrumented runs (no profiler overhead), 8k and 40k points, all matched:

| Question | 8,000 points | 40,000 points |
|---|---|---|
| **Time in `_CSRDistCache.lookup` vs Viterbi bookkeeping** | 36 % cumulative in `lookup`; 40 % self time in `_match_one` | — |
| **Candidate retrieval vs the Viterbi itself** | 13.0 % / 87.0 % | 16.3 % / 83.7 % |
| **Cache hit rate** | **98.2 %** (117,170 / 2,142) | **99.5 %** (529,033 / 2,871) |
| **Does `dist_cache_size=10,000` bind?** | **No** — 723 entries held | **No** — 790 entries held |
| **Lookups per point** | 14.91 | 13.30 |
| **`edge_endpoints()` calls per point** | 8.5 | 8.2 |
| **`edge_length()` calls per point** | 12.9 | 11.3 |
| **Viterbi frontier (candidates/fix)** | mean 4.26, max 8 (k=8) | mean 4.11, max 8 |
| **Peak traced memory / peak RSS** | 3.0 MB / 119.3 MB | 7.7 MB / 134.1 MB |

**Does raising `dist_cache_size` help?** No, and it cannot: the cache never
fills.

```
  maxsize   1,000:   0.64 s   hit rate  98.2%   entries held 723
  maxsize  10,000:   0.71 s   hit rate  98.2%   entries held 723
  maxsize 100,000:   0.63 s   hit rate  98.2%   entries held 723
```

**How the frontier widens with `k` and `max_dist`** — `k` is the cost knob;
`max_dist` is nearly free because the KDTree shortlist saturates first:

```
  k=4   max_dist=30    candidates/fix mean 3.06 max 4     0.47 s
  k=4   max_dist=80    candidates/fix mean 3.06 max 4     0.39 s
  k=8   max_dist=50    candidates/fix mean 4.26 max 8     0.66 s
  k=16  max_dist=50    candidates/fix mean 5.50 max 8     1.06 s
  k=16  max_dist=80    candidates/fix mean 5.83 max 8     0.90 s
```

## 4. The whole pipeline, same data

| Stage | 8,000 points | 40,000 points |
|---|---|---|
| `NearestMatcher.match` | 0.02 s | 0.12 s |
| `HMMMatcher.match` | 0.62 s — 46.3 % | 2.10 s — **38.9 %** |
| `derive_speeds` | 0.70 s — 52.4 % | 3.25 s — **60.2 %** |
| `aggregate_speeds` | 0.02 s — 1.3 % | 0.05 s — 0.9 % |

`derive_speeds` also scales worse: 4.6× the time for 5× the data, against the
matcher's 3.4×.

---

### F-7.1 — "serial matching stayed faster up to at least two million points" is false
- **Severity:** S3
- **Location:** `src/redlight/matching.py:163-165`
- **Claim:** The `n_jobs` docstring says serial stayed faster than `n_jobs>1`
  "up to at least two million points". Measured here the crossover is **between
  500,000 and 1,000,000 points** — parallel wins at 1M and 2M.
- **Evidence:**
  ```
     20,000 pts: serial  31,305 /s   n_jobs=4   5,132 /s   serial 6.10x faster
    200,000 pts: serial  77,773 /s   n_jobs=4  41,194 /s   serial 1.89x faster
    500,000 pts: serial 102,836 /s   n_jobs=4  84,354 /s   serial 1.22x faster
  1,000,000 pts: serial 113,436 /s   n_jobs=4 123,826 /s   PARALLEL 1.09x faster
  1,000,000 rpt: serial 112,536 /s   n_jobs=4 120,555 /s   PARALLEL 1.07x faster
  2,000,000 pts: serial 115,666 /s   n_jobs=4 138,299 /s   PARALLEL 1.20x faster
  ```
  The 1M measurement was repeated; the 7–9 % parallel win is reproducible.
- **Expected vs actual:** The claim is falsified on the very machine
  `00-baseline.md` names, and this machine has **2 physical cores** — the
  weakest case parallelism will meet. On a many-core machine the crossover will
  come earlier still, so the claim cannot be rescued by hardware.
- **Suggested fix:** Restate with the measured crossover and the caveat that it
  moves with core count: serial wins decisively below ~200k points (1.9–6.1×),
  is roughly a wash by 500k, and loses above ~1M. The surrounding advice —
  "measure before enabling" — is sound and should stay.
- **Verdict:**
- **Outcome:**

---

### F-7.2 — The "tens of thousands of points per second" figure is measured on a workload that is 93 % unmatchable
- **Severity:** S3
- **Location:** `src/redlight/matching.py:159-161`; `benchmarks/bench_matching.py:43-64`
- **Claim:** The claim is *literally* true against `bench_matching.py` — 31k to
  116k points/s — but that benchmark's generator walks each mover off the
  network after ~136 fixes, so at realistic sizes **only 6.6 % of points have
  any candidate edge at all** and the rest are rejected before any Viterbi work
  happens. On a workload where every point matches, serial throughput is
  **12,000–18,000 points/s**.
- **Evidence:** the benchmark's own match counts, and the same matcher on an
  on-network workload:
  ```
  bench_matching.py  1,000,000 points: matched   66,195/1,000,000  (6.6%)  113,436 pts/s
  bench_matching.py  2,000,000 points: matched  132,461/2,000,000  (6.6%)  115,666 pts/s

  profile_hmm.py        8,000 points: matched    8,000/8,000  (100.0%)  12,998 pts/s
  profile_hmm.py       40,000 points: matched   40,000/40,000 (100.0%)  18,226 pts/s
  ```
  The cause is arithmetic: the grid spans `(30-1) x 0.001 = 0.029` degrees and
  `simulate_points` advances `lon` by `2.2e-4` per fix without bound, so a mover
  leaves the network after ~132 fixes while `per_traj` is 2,000.
- **Expected vs actual:** "Tens of thousands of points per second" reads as a
  matching rate; it is a *throughput* rate over a stream that is mostly
  rejections. Both numbers are real, but only one is what a user planning a run
  needs. 12–18k/s is still a defensible headline — it is just 6× smaller.
- **Suggested fix:** Either qualify the docstring ("including points with no
  candidate edge; on fully-matched data expect roughly 12–18k/s on a 2017
  laptop core"), or give `bench_matching.py` a bounded generator so its headline
  measures matching. `profile_hmm.py::simulate_on_network` is such a generator
  and can be reused. **Note this affects F-7.1's numbers too** — the crossover
  was measured on the same mostly-unmatched workload, so it should be re-checked
  against on-network data before the docstring is rewritten.
- **Verdict:**
- **Outcome:**

---

### F-7.3 — `derive_speeds` costs more than the matcher, and no pass has looked at it
- **Severity:** S4
- **Location:** `src/redlight/speeds.py` (whole module); `Task 8` scope
- **Claim:** On identical data, `derive_speeds` takes **60.2 %** of pipeline
  wall time against `HMMMatcher.match`'s **38.9 %**, and scales worse. Task 8 is
  scoped entirely to the matcher, so as written it is aimed at the smaller half
  of the problem: even a *perfect* matcher optimization caps out at a 39 %
  pipeline saving, while `derive_speeds` has never been profiled.
- **Evidence:**
  ```
  8,000 points                      40,000 points
    NearestMatcher   0.02 s           NearestMatcher   0.12 s
    HMMMatcher.match 0.62 s  46.3%    HMMMatcher.match 2.10 s  38.9%
    derive_speeds    0.70 s  52.4%    derive_speeds    3.25 s  60.2%
    aggregate_speeds 0.02 s   1.3%    aggregate_speeds 0.05 s   0.9%
  ```
  Scaling 8k → 40k (5× the data): matcher 3.4×, `derive_speeds` **4.6×**.
- **Expected vs actual:** The plan anticipated this exact possibility — "If
  matching is 5 % of a real pipeline's wall time, that is the single most
  important finding in this pass." It is not 5 %, but it is the minority, and
  the likely culprit is visible without profiling: `speeds._arc_position` calls
  shapely's `project` once per fix, and `_hop_distance` runs up to four
  `_SourceDistCache.query` calls per hop through pure-Python networkx, where the
  matcher uses scipy's C Dijkstra.
- **Suggested fix:** Widen Task 8's remit to cover `derive_speeds`, or add a
  Task 8b for it. The same invariance discipline applies — output must not
  change. The cheapest candidate to investigate first is moving
  `_SourceDistCache` onto `Network.csgraph()` and scipy, matching what
  `matching._CSRDistCache` already does.
- **Verdict:**
- **Outcome:**

---

### F-7.4 — `dist_cache_size` cannot do what its docstring claims; the cache never fills
- **Severity:** S3
- **Location:** `src/redlight/matching.py:168-171`
- **Claim:** The docstring says a larger `dist_cache_size` is "faster on data
  that revisits the same roads, at more memory (~tens of KB per entry)". The
  cache holds **723 entries out of 10,000** at 8k points and **790 at 40k** —
  it is bounded by the number of distinct source nodes in the network (900 for
  a 30×30 grid), not by `maxsize`. The parameter is inert on any network with
  fewer nodes than `maxsize`, which is most of them.
- **Evidence:**
  ```
    maxsize   1,000:   0.64 s   hit rate  98.2%   entries held 723
    maxsize  10,000:   0.71 s   hit rate  98.2%   entries held 723
    maxsize 100,000:   0.63 s   hit rate  98.2%   entries held 723
  ```
  Hit rate is 98.2 % at 8k points and 99.5 % at 40k — there is essentially no
  headroom for a bigger cache to recover.
- **Expected vs actual:** The claim is not wrong in principle — it would hold on
  a network with more than 10,000 nodes — but nothing tells the reader that the
  parameter does nothing below that, and the ~tens of KB per entry figure was
  not verified here.
- **Suggested fix:** Say the cache is bounded by distinct source nodes, so
  `maxsize` only binds on networks with more nodes than it, and that the default
  is already generous for city-scale networks. Task 8 should not spend effort on
  cache tuning: at 98–99.5 % there is nothing to win.
- **Verdict:**
- **Outcome:**

---

### F-7.5 — The inner-loop network lookups fire 8–13 times per point (Task 8's named target, quantified)
- **Severity:** S4
- **Location:** `src/redlight/matching.py:289-315`
- **Claim:** Task 8 is told to consider hoisting `network.edge_endpoints()` and
  `network.edge_length()` out of the inner loop. This pass quantifies the
  target: **8.2 `edge_endpoints` and 11.3 `edge_length` calls per point** at
  40k, plus 13.3 cache lookups. They are cheap individually (dict and list
  indexing) but land in `_match_one`'s self time, which is the single largest
  entry in the profile at 40 %.
- **Evidence:**
  ```
    dist-cache lookups          531,904 (13.30 per point)
    edge_endpoints() calls      327,854 (8.2 per point)
    edge_length() calls         453,672 (11.3 per point)
    candidates per fix          mean 4.11, max 8 (k=8)
  ```
  Supporting profile lines: `dict.get` 487,393 calls (0.755 s tottime),
  `OrderedDict.move_to_end` 119,312 calls (0.318 s) — one per lookup, including
  hits — and `builtins.abs` 131,697 calls (0.624 s).
- **Expected vs actual:** Consistent with a frontier of ~4.1 candidates per fix
  against ~4.1 predecessors, i.e. ~17 (candidate, predecessor) pairs per step.
  Precomputing per-edge length and endpoint arrays once per network and indexing
  them with numpy is the obvious win and provably cannot change output.
- **Suggested fix:** For Task 8, in priority order from this profile: (1) hoist
  the per-pair `edge_endpoints`/`edge_length` calls; (2) look at
  `move_to_end` firing on every cache hit at a 99.5 % hit rate; (3) leave the
  cache size alone (F-7.4). Do **not** start with candidate retrieval — it is
  13–16 % of the total.
- **Verdict:**
- **Outcome:**

---

### F-7.6 — `bench_matching.py`'s generator walks movers off the network
- **Severity:** S4
- **Location:** `benchmarks/bench_matching.py:43-64`
- **Claim:** `simulate_points` advances `lon` by a fixed `2.2e-4` per fix with
  no bound, while the grid spans `(grid_n - 1) * 0.001` degrees. Every mover
  therefore leaves the network after ~132 fixes and every later fix is
  unmatchable. At the plan's own suggested size (200k points, 200 trajectories)
  that is 87 % of the data; at 2M it is 93 %.
- **Evidence:**
  ```
  matched  26,493/200,000    (13.2%)   at --points 200000 --trajectories 200
  matched  66,195/1,000,000  ( 6.6%)   at --points 1000000 --trajectories 500
  matched 132,461/2,000,000  ( 6.6%)   at --points 2000000 --trajectories 1000
  ```
  versus `profile_hmm.py`'s bounded generator, same grid: **100.0 % matched**.
- **Expected vs actual:** The benchmark is not wrong to include unmatched
  points — real data has them — but at 93 % they dominate, and every number it
  reports is mostly a measure of how fast the KDTree can reject. This is the
  root cause of F-7.2 and it also makes F-7.1's crossover suspect.
- **Suggested fix:** Bound the walk (turn at the grid edge, as
  `profile_hmm.py::simulate_on_network` does) or make the unmatched fraction an
  explicit `--offroad-fraction` argument so it is a stated parameter rather than
  an accident of arithmetic.
- **Verdict:**
- **Outcome:**

---

## Repo state on exit

```
$ git status --porcelain
?? benchmarks/profile_hmm.py
$ .venv/bin/pytest -q
405 passed
$ .venv/bin/ruff check src tests scripts examples benchmarks
All checks passed!
```

`src/` was not modified. `benchmarks/profile_hmm.py` is the only new file, and
`testpaths = ["tests"]` keeps it out of collection.

## Unverified suspicions

1. **F-7.1's crossover was measured on the 93 %-unmatched workload.** Unmatched
   points are cheap and identical in both paths, so they dilute the parallel
   overhead and may well move the crossover. The direction of the bias is not
   obvious — re-measure on `simulate_on_network` before rewriting the docstring.
2. **`derive_speeds`' cost was measured, not attributed.** F-7.3 names
   `_arc_position`'s per-fix shapely `project` and `_hop_distance`'s networkx
   Dijkstra as the likely culprits from reading, not from a profile. Whoever
   takes F-7.3 should profile before optimizing, exactly as this pass did for
   the matcher.
3. **All numbers are single-run** apart from the repeated 1M measurement. The
   suspiciously flat `dist_cache_size` timings (0.64 / 0.71 / 0.63 s) show
   run-to-run noise of roughly ±10 % on this machine, which is enough to hide
   any effect smaller than that. Task 8's ~5 % accept/revert threshold will need
   repeated runs to resolve at all.
