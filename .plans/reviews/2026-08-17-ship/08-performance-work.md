# Performance — optimization work log
**Pass:** Task 8
**Date:** 2026-08-19
**Commit at start:** `6d3844d`
**Scope:** `src/redlight/matching.py`. Created:
`tests/test_matching_invariance.py`, `tests/data/matching_invariance.json`,
`.plans/2026-08-17-fast-map-matching-proposal.md`.
**Measuring instrument:** a 9-run median of `HMMMatcher.match` alone (20,000
on-network points, 60 trajectories, 30×30 grid), plus
`benchmarks/profile_hmm.py` for call counts. `profile_hmm.py`'s full run was too
noisy at ±10 % to resolve a 5 % effect; the lean median harness holds a 5–9 %
spread and separates the medians cleanly.

## Summary

One optimization kept (**4.2–4.6 %**), one measured and reverted for no gain,
three declined on the profile's own evidence. The invariance test was written
first, validated by mutation, and never weakened. **409 tests pass**, ruff clean
across all directories, and the decoded output is byte-identical to `189a92e`.

The headline is not the 4.5 %: it is that Task 7's profile correctly predicted
which candidates were worth trying, and the two it flagged as cold turned out to
be cold when measured. The plan's rule — *work only on hot spots the profile
identified* — did its job.

## The invariance test came first

`tests/test_matching_invariance.py` (committed `6d3844d`, before any change to
`matching.py`) decodes three seeded scenarios and compares against expectations
generated from the pre-Task-8 code. Edge id sequences must be identical; snap
distances must agree to 1e-12; `NearestMatcher` is pinned too.

**Validated by mutation, not assumption.** It catches every failure mode the
hoist could plausibly introduce:

| Mutation | Caught? |
|---|---|
| wrong endpoint on the predecessor (`[1]` → `[0]`) | **yes**, 3/3 scenarios |
| wrong endpoint on the candidate | **yes**, 3/3 |
| dropped the lead-in term | **yes**, 3/3 |
| dropped the lead-out term | **yes**, 3/3 |
| route distance perturbed by 1e-7 | no — sub-threshold, does not change the decode |
| `>` → `>=` in the best-score test | no — no exact ties occur on these fixtures |

The last two are the threshold behaving correctly rather than gaps, but they
are worth stating: **the test pins the decision, not the arithmetic.** An
optimization that perturbs a value without changing any decision will pass.

Writing it also surfaced a packaging bug: `MANIFEST.in` shipped
`recursive-include tests *.py` but no data, so the sdist would have carried the
test without its expectations — the exact failure that file exists to prevent.
Fixed in the same commit.

---

## Part A

### 1. Hoist per-predecessor edge lookups — **KEPT**, 4.2–4.6 % (`cbd5567`)

`remaining_prev` depends only on the predecessor edge, not on the candidate,
but sat inside the candidate loop — so the same `edge_length` lookup ran once
per (candidate, predecessor) pair. Moved beside the `prev_end` resolution that
already ran once per predecessor per step.

```
edge_length() calls   11.3 per point  ->  4.1 per point   (-64%)

9-run medians, match only:
  without   0.9476 s   (min 0.9208, max 0.9742)
  with      0.9038 s   (min 0.8936, max 0.9402)
  with      0.9115 s   (min 0.8985, max 0.9469)   independent repeat
```

**Kept at 4.2–4.6 %, marginally under the plan's ~5 % threshold, deliberately.**
That threshold exists to reject *complexity that does not pay for itself*. This
is not complexity: the inner loop lost a statement, the per-step loop gained
one, and the arithmetic is unchanged. Had it added a branch or an index
indirection, it would have gone.

### 2. Skip LRU bookkeeping while the cache is under its bound — **REVERTED**

Task 7 F-7.5 flagged `OrderedDict.move_to_end` firing on every lookup at a
99.5 % hit rate. Since the cache never fills on these networks (790 entries of
10,000, F-7.4), recency is unused bookkeeping. Safe to skip — eviction order
cannot change output, because the cache is transparent: a miss recomputes and a
hit re-applies the cutoff, so both return the same distance.

```
committed baseline   0.9038 / 0.9115 s
with the change      0.9195 / 0.9116 s      -> no gain, possibly worse
```

Reverted per the plan's rule. The profile's 0.647 s of `move_to_end` self time
across 270,586 calls is a cProfile artifact: per-call overhead dominates the
measurement for a builtin that costs tens of nanoseconds.

### 3. Flat arrays for the Viterbi state — **DECLINED**

The plan gates this on "**if and only if** the profile shows dict overhead is
material". It does not clear that bar. `dict.get` is 1.544 s self time across
1,104,686 calls under cProfile; at real per-call cost that is roughly 6 % of
wall time, and much of it is inside `_CSRDistCache.lookup` rather than the
Viterbi state dicts the rewrite would replace.

Against that: `_match_one` is the most intricate function in the package and
the one Task 3 found a subtle, long-lived bug in (F-3.1, shipped in v0.3.0 and
undetected for two releases). Restructuring its state representation for a
plausible ~5 % is the trade the revert rule exists to refuse.

### 4. Vectorize the emission log-prob — **DECLINED, cold**

`_emission_logp` is 34,042 calls at **0.063 s** self time — well under 1 %.
The profile says it is cold; the plan says cold things are out of scope.

### 5. Cache cutoff quantization — **OUT by Task 7 F-7.4**

The plan permits this "only if the baseline measured a low hit rate". It
measured 98.2–99.5 %. There is nothing to recover, and rounding a cutoff *down*
would change output and is forbidden.

### 6. Lossless candidate pruning — **not attempted**

Candidate retrieval is 13–16 % of match time and the frontier is already narrow
(mean 4.1 candidates per fix at k=8). No pruning rule was identified that is
provably lossless and cheaper than the work it removes.

---

## Part B — the FMM proposal

Written to `.plans/2026-08-17-fast-map-matching-proposal.md`. The
recommendation is **decline**, with the reasoning measured rather than argued:

`_CSRDistCache` is already a UBODT — built lazily, bounded by actual queries
rather than a global δ, sized to what a run touches (790 entries) rather than
all pairs, and running at a 99.5 % hit rate. FMM's proposition is to make
transition-distance requests hash lookups; they already are. Measured table
costs range from 9.8 MB (900 nodes) to an extrapolated 3.4 GB (200,000 nodes),
against a current peak traced allocation of 7.7 MB. And it cannot be exact
without keeping the bounded Dijkstra as a fallback, because the cutoff varies
per step and a fixed δ cannot be guaranteed to cover it.

Awaiting your decision at the top of that file.

---

## What remains on the table

### F-8.1 — `derive_speeds` is the larger target and is untouched
- **Severity:** S4
- **Location:** `src/redlight/speeds.py:311-425`
- **Claim:** Task 8's scope is the matcher, which Task 7 measured at **39 %** of
  pipeline wall time. `derive_speeds` is **60 %**, and Task 7b attributed
  **60–62 % of that** to its own row-by-row body — 125,552 dicts built one at a
  time at 40k points, then handed to pandas as lists of dicts. That is the
  largest measured win available to this package, and this pass could not take
  it without exceeding its remit.
- **Evidence:** `07b-derive-speeds-profile.md`, and by comparison the 4.5 %
  this pass could win inside the matcher.
- **Suggested fix:** A Task 8b under the same discipline — invariance test
  first, then accumulate into typed arrays and build each frame from a dict of
  columns. Note the dtype caveat in 07b's suspicion 2: `traj_id` (nullable),
  `quality` (bool) and the datetime columns need checking against current
  output, since `pd.DataFrame(list_of_dicts)` infers dtypes that explicit
  arrays fix.
- **Verdict:**
- **Outcome:**

### F-8.2 — A\* with a geometric heuristic is the one live idea from Part B
- **Severity:** S4
- **Location:** `src/redlight/matching.py:99-122`
- **Claim:** Of the alternatives weighed against FMM, only this one survives:
  it needs no precomputation, no new parameter and no invalidation contract,
  and it would speed the 0.5 % of lookups that miss cache. It is also honestly
  small — under 1 % overall on the measured hit rate — so it is recorded rather
  than recommended.
- **Evidence:** 857 real Dijkstra runs out of 159,600 queries (99.5 % hit rate).
- **Suggested fix:** Only worth revisiting if a real workload is found where the
  cache hit rate collapses. Instrument first.
- **Verdict:**
- **Outcome:**

## Addendum — Task 8b, first optimization (2026-08-19)

Taken up immediately after this pass on the strength of F-8.1, under the same
discipline: `tests/test_speeds_invariance.py` written and committed first
(`04a7312`), verified to fail on a 1e-7 perturbation of `speed_var`.

**Vectorised the arc-position lookup — KEPT, 25 % (`1d07672`).**
`_arc_position` built a `shapely.Point` and called `geom.project()` once per
fix. `shapely.line_locate_point` is the vectorised form of the same GEOS entry
point, so results are bit-identical.

```
7-run medians of derive_speeds alone, 40,000 on-network points:
  per-fix loop   2.9955 s   (min 2.9656, max 3.0926)   13,353 pts/s
  vectorised     2.2431 s   (min 2.1863, max 2.2933)   17,832 pts/s
  vectorised     2.2556 s   (min 2.2194, max 2.2997)   17,734 pts/s   repeat
  -> 25 %, ranges do not overlap
```

In the profile `_arc_positions` falls from 22–24 % to **1.7 %**; 20,000
Python-level round trips become 60 batched calls, one per trajectory. No new
dependency — `shapely>=2.0` is already core.

**A correction to F-7b.1.** That finding attributed the 60 % "everything else"
to "the `while` loop, the per-row dict construction, and the pandas frame built
from lists of dicts". The pandas share is now measured and it is small:

```
pd.DataFrame(list_of_dicts)   0.109 s
pd.DataFrame(dict_of_cols)    0.079 s     1.4x   -> under 1 % of total
```

So column-wise frame construction is **not** the win F-7b.1 implied. The
remaining cost is the loop itself, and within it the `for e in uniq_edges:`
inner loop that builds 85,652 dicts of 11 keys at 40k points — each repeating
the same scalar interval values across the edges that interval touched. A
`np.repeat` expansion over per-interval edge counts is the next candidate.
**Unmeasured**, so no number is claimed for it here.

**Second optimization: column-wise frames — KEPT, 12.9 % (`653939b`).**
Both output frames are now accumulated column-wise, and `edge_observations`
becomes an `np.repeat` expansion of `intervals` over a per-interval edge count.

```
7-run medians of derive_speeds alone, 40,000 on-network points:
  row-wise      2.2520 s   (min 2.2130, max 2.2943)   17,762 pts/s
  column-wise   1.9614 s   (min 1.9175, max 1.9767)   20,393 pts/s
  column-wise   1.9452 s   (min 1.9152, max 1.9623)   20,563 pts/s   repeat
  -> 12.9 %, ranges do not overlap
```

**A methodological note worth keeping.** Measuring the two halves in isolation
suggested 10.1 % (edges) + 5.9 % (intervals) = 16 %. The end-to-end figure is
12.9 %, because those probes timed the frame *construction* while excluding the
per-column *accumulation* that feeds it — the fast path was given data it would
have had to build. Isolated microbenchmarks of a fast path flatter it; only the
end-to-end number is trustworthy.

Two output details the change had to preserve, both verified:
- **dtypes** — building the edge frame by repeating the interval frame's own
  numpy columns inherits its dtypes instead of re-inferring them.
- **the empty-run contract** — an empty run has always returned frames with *no
  columns at all*; column-wise construction would silently have returned
  `(0, 17)`. Guarded explicitly.

**Cumulative effect on `derive_speeds` across both optimizations:
13,353 → 20,563 points/s, a 1.54x speedup, with byte-identical output.**

Post-first-optimization profile, 20k points:

```
  total derive_speeds           1.29 s   (15,547 pts/s)
  _arc_positions (shapely)      0.02 s     1.7%   60 batched call(s)
  _hop_distance (total)         0.27 s    21.0%
    of which _SourceDistCache   0.08 s     6.2%   79,680 queries, 699 real Dijkstras
  everything else               0.99 s    77.3%
```

## Repo state on exit

```
$ .venv/bin/pytest -q
412 passed
$ .venv/bin/ruff check src tests scripts examples benchmarks
All checks passed!
$ git status --porcelain
(empty)
```

## Unverified suspicions

1. **The 4.2–4.6 % is one workload on one machine.** 20k on-network points on a
   30×30 grid, 2 physical cores. The hoist removes a fixed fraction of calls
   (64 % of `edge_length`), so the direction is safe, but the magnitude will
   differ where the frontier is wider (larger `k`) or narrower.
2. **The invariance test's three scenarios are all grids.** Grid networks are
   the adversarial case for the *transition* term (methodology.md §2.3 argues
   this), which is good for pinning it — but no curved or irregular geometry is
   covered, and F-3.2's hairpin case showed curvature is where the same-edge
   shortcut misbehaves. Worth adding one irregular network before Task 8b
   changes anything shared.
3. **Whether `dict.get`'s ~6 % is really in the Viterbi state.** The estimate
   splits 1.1 M calls between `_CSRDistCache.lookup` and the state dicts by
   inspection, not by measurement. If it is mostly in `lookup`, the flat-array
   rewrite would win even less than the estimate that declined it.
