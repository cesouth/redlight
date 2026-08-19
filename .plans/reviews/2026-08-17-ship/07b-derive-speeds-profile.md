# Performance — `derive_speeds` profile — findings
**Pass:** Task 7b (added after Task 7; not in the original plan)
**Date:** 2026-08-19
**Commit reviewed:** `53fde7b`
**Scope:** `src/redlight/speeds.py`. Created: `benchmarks/profile_speeds.py`.
**Method:** cProfile plus direct instrumentation of `_arc_position`,
`_hop_distance` and `_SourceDistCache.query`, on the on-network workload from
`profile_hmm.py` (100 % of fixes matched). Run at 8k and 40k points, with a
scaling sweep at each. **No optimization written; `src/` untouched.**

**Machine:** Intel Core i5-7360U @ 2.30 GHz, 2 physical / 4 logical cores.

## Summary

Task 7 established that `derive_speeds` costs more than the matcher, but
attributed that cost by reading the code. Measured, **the attribution in F-7.3
was mostly wrong**: the pure-Python networkx Dijkstra I nominated as a likely
culprit is **4–6 % of the time**, and its cache already runs at a 99.5 % hit
rate, so the scipy swap I proposed would win almost nothing. The shapely
projection is real but secondary at 22–24 %. **The dominant cost — 60–62 % — is
`derive_speeds`' own row-by-row Python body**: the `while` loop, the per-row
dict construction, and the pandas frame built from lists of dicts. Task 7's
claim that `derive_speeds` scales worse than the matcher is also **not
supported** by a cleaner measurement: it is linear.

## Where the time goes

Instrumented runs, no profiler overhead, every fix matched:

| Component | 8,000 points | 40,000 points |
|---|---|---|
| **total `derive_speeds`** | 0.73 s (11,034 pts/s) | 3.48 s (11,501 pts/s) |
| `_arc_position` — shapely `project`, 1.00 call/point | 0.16 s — **22.0 %** | 0.84 s — **24.3 %** |
| `_hop_distance` total | 0.12 s — 16.3 % | 0.53 s — 15.2 % |
| …of which `_SourceDistCache` (networkx Dijkstra) | 0.04 s — **6.1 %** | 0.14 s — **4.1 %** |
| **everything else — the function's own body** | 0.45 s — **61.8 %** | 2.11 s — **60.5 %** |
| distance-cache hit rate | 98.3 % (551 real Dijkstras / 31,840 queries) | 99.5 % (857 / 159,600) |
| output | 7,960 intervals, 18,204 edge obs | 39,900 intervals, 85,652 edge obs |

cProfile corroborates and shows what the 60 % is made of — `derive_speeds`'
self time is the single largest entry, with pandas construction from
lists-of-dicts close behind:

| ncalls | tottime | cumtime | function |
|---|---|---|---|
| 1 | **0.342** | 1.062 | `speeds.py:211(derive_speeds)` — the loop itself |
| 8,000 | 0.017 | 0.285 | `speeds.py:91(_arc_position)` |
| 7,960 | 0.059 | 0.209 | `speeds.py:161(_hop_distance)` |
| 24,000/16,000 | 0.022 | 0.198 | `shapely/decorators.py:171(wrapper)` |
| 8,000 | 0.023 | 0.160 | `shapely/geometry/point.py:54(__new__)` — a `Point` per fix |
| 8,000 | 0.008 | 0.098 | `shapely/geometry/base.py:913(project)` |
| 31,840 | 0.022 | 0.089 | `speeds.py:119(query)` |
| 2 | 0.000 | 0.085 | `pandas/core/frame.py:702(__init__)` — the two output frames |
| 479 | 0.001 | 0.053 | `networkx…single_source_dijkstra` |
| 2 | 0.020 | 0.039 | `pandas…_list_of_dict_to_arrays` |

## Scaling — linear, not superlinear

```
  10,000 points:   0.77 s      12,920 pts/s
  20,000 points:   1.62 s      12,330 pts/s
  40,000 points:   3.05 s      13,110 pts/s
```

`derive_speeds` holds ~11–13k points/s across a 20× range. For comparison,
`HMMMatcher.match` on the same data runs at 13,000–18,226 pts/s, so the two are
within about 1.5× of each other per point.

---

### F-7b.1 — The cost is the per-row Python body, not the geometry or the graph search
- **Severity:** S4
- **Location:** `src/redlight/speeds.py:311-425` (the hop loop and row assembly)
- **Claim:** 60–62 % of `derive_speeds` is neither `_arc_position` nor
  `_hop_distance`. It is the function's own body: a Python `while` loop over
  fixes that appends one 18-key dict per interval and one 11-key dict per
  (interval, edge), then hands two lists of dicts to `pd.DataFrame`. At 40k
  points that is 39,900 + 85,652 = 125,552 dicts built one at a time.
- **Evidence:**
  ```
    total derive_speeds           3.48 s   (11,501 pts/s)
    _arc_position (shapely)       0.84 s    24.3%   40,000 calls (1.00/point)
    _hop_distance (total)         0.53 s    15.2%   39,900 calls
      of which _SourceDistCache   0.14 s     4.1%   159,600 queries, 857 real Dijkstras
    everything else               2.11 s    60.5%
  ```
  and from cProfile, `derive_speeds`' own self time is the largest single entry
  (0.342 s of 1.062 s cumulative), with `pandas…_list_of_dict_to_arrays` at
  0.039 s and `frame.__init__` at 0.085 s cumulative for just two calls.
- **Expected vs actual:** This is the opposite of where F-7.3 pointed. It also
  means the win here is not a clever algorithm — it is the ordinary work of
  accumulating into typed arrays and constructing the frames once, which is
  exactly the kind of change that can be made output-identical and verified as
  such.
- **Suggested fix:** For Task 8b: accumulate interval fields into
  preallocated numpy arrays (or parallel lists) and build each DataFrame from a
  dict of columns rather than a list of row dicts. Nothing about the maths
  changes, so a byte-identical invariance test is achievable. Measure before
  and after with this harness; the ~5 % accept/revert threshold needs repeated
  runs (see Task 7's suspicion 3 on this machine's ±10 % noise).
- **Verdict:**
- **Outcome:**

---

### F-7b.2 — Correction: the networkx Dijkstra is not a bottleneck, and F-7.3's scaling claim is unsupported
- **Severity:** S4
- **Location:** `.plans/reviews/2026-08-17-ship/07-performance-baseline.md`, F-7.3
- **Claim:** Two statements in F-7.3 do not survive measurement.
  1. F-7.3 named "moving `_SourceDistCache` onto `Network.csgraph()` and scipy"
     as "the cheapest candidate to investigate first". It is not: the whole
     `_SourceDistCache` accounts for **4.1 %** at 40k points, and it performs
     only **857 real Dijkstra runs for 159,600 queries** — a 99.5 % hit rate.
     Even a free Dijkstra would save at most ~4 %.
  2. F-7.3 said `derive_speeds` "scales worse — 4.6× the time for 5× the data,
     against the matcher's 3.4×". That comparison varied trajectory count as
     well as point count (40 → 100 trajectories). Holding the workload fixed and
     varying only size, throughput is **flat**.
- **Evidence:**
  ```
  === scaling ===
    10,000 points:   0.77 s      12,920 pts/s
    20,000 points:   1.62 s      12,330 pts/s
    40,000 points:   3.05 s      13,110 pts/s
  ```
  and the cache measurements above (98.3 % at 8k, 99.5 % at 40k).
- **Expected vs actual:** F-7.3's *headline* — that `derive_speeds` costs more
  than the matcher and deserves its own pass — stands, and is what justified
  this pass. Its *attribution* and its *scaling* claim do not. The relevant
  correction has been noted here rather than by editing F-7.3, so the record
  shows what was believed when Task 8's scope was being decided.
- **Suggested fix:** Amend F-7.3's "Suggested fix" line when it is triaged, so
  Task 8b is not sent at the Dijkstra. No code change.
- **Verdict:**
- **Outcome:**

---

### F-7b.3 — `_arc_position` does one shapely round trip per fix, and the network already has the data to do it in a batch
- **Severity:** S4
- **Location:** `src/redlight/speeds.py:91-96`
- **Claim:** `_arc_position` builds a `shapely.Point` and calls
  `geom.project()` once per fix — 22–24 % of `derive_speeds`, and the second
  largest component. Meanwhile `Network._seg_table` already carries every
  edge's sub-segments *with an arc offset per row* (added for F-3.1 in
  `c4d40ab`), which is the same quantity computed vectorised.
- **Evidence:**
  ```
    _arc_position (shapely)       0.84 s    24.3%   40,000 calls (1.00/point)
  ```
  cProfile shows the cost is inside shapely, not in the wrapper: `Point.__new__`
  0.160 s cumulative, `base.project` 0.098 s, `linear.line_locate_point`
  0.034 s tottime — against `_arc_position`'s own 0.017 s tottime.
- **Expected vs actual:** A vectorised `Network.arc_positions(edge_ids, px, py)`
  reading the seg table could replace the per-fix round trip. **This is a
  proposal, not a plan** — two things need settling first: the seg-table lookup
  must be restricted to the sub-segments of the *given* edge rather than the
  nearest one, and its answer must match shapely's `project` to a tolerance
  tight enough not to move any output. F-3.1 is a reminder of what happens when
  a seg-table quantity is assumed rather than checked.
- **Suggested fix:** Task 8b should take F-7b.1 first — it is larger, simpler
  and provably output-preserving. Only if that lands and more is wanted should
  this be attempted, and then with an invariance test that compares
  `arc_positions` against `_arc_position` fix by fix before any switch.
- **Verdict:**
- **Outcome:**

---

## Repo state on exit

```
$ git status --porcelain
?? benchmarks/profile_speeds.py
$ .venv/bin/ruff check src tests scripts examples benchmarks
All checks passed!
```

`src/` was not modified.

## Still outstanding

**F-7.1's crossover and F-7.2's throughput were not re-measured on the
on-network workload.** Task 7 flagged that both were taken on
`bench_matching.py`'s 93 %-unmatched data, and I offered to re-check them here.
I did not: an on-network run at 1M points matches every fix, so at the measured
11–18k points/s it costs roughly 90 s serial plus a comparable parallel run per
size — several minutes per data point, against a fixed context budget I chose
to spend on the profile itself. It remains the right thing to do before either
docstring is rewritten, and `profile_hmm.py::simulate_on_network` is the
generator to do it with.

## Unverified suspicions

1. **The 60 % "everything else" was attributed by subtraction, not by line
   profiling.** cProfile's self-time for `derive_speeds` (0.342 s of 1.062 s)
   and the pandas construction entries corroborate it, but no line-level
   profile was taken, so the split between the `while` loop, the dict building
   and the frame construction is inferred. `line_profiler` would settle it and
   is dev-only.
2. **Whether the two output frames can be built column-wise without changing
   dtypes.** `pd.DataFrame(list_of_dicts)` infers per-column dtypes from the
   data; building from arrays fixes them explicitly. That is usually identical
   but not automatically so — `traj_id` (which can be `None`), `quality` (bool)
   and the datetime columns are the ones to check against the current output.
3. **All numbers are single-run.** Task 7 measured ±10 % run-to-run noise on
   this machine; the 22–24 % and 60–62 % bands above are consistent across two
   sizes, which is weak evidence of stability but not a repeat measurement.
