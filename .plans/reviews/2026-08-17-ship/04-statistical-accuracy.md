# Accuracy — statistical claims and the published experiments — findings
**Pass:** Task 4
**Date:** 2026-08-18
**Commit reviewed:** `6aef04e`
**Scope:** `scripts/paper_experiments.py`, `docs/methodology.md`,
`docs/statistics.md`, `docs/figures/experiment_results.json`,
`src/redlight/aggregate.py`, `src/redlight/modes.py`, `src/redlight/cleaning.py`.
Read for context: `tests/test_aggregate.py`, `tests/test_modes.py`,
`tests/test_cleaning.py`, and the numeric-claims checklist handed over by
`02-doc-drift.md`.
**Method:** `scripts/paper_experiments.py` run three ways — at `HEAD`, twice for
determinism, and once in a throwaway `git worktree` at `83880be` (the commit
Task 3 reviewed, i.e. *before* this review's fixes) — each output diffed
leaf-by-leaf against the committed `experiment_results.json`. Parts B–D are
hand-constructed cases with analytically known answers. Scripts in
`…/c8494443-…/scratchpad/t4/`: `diff_json.py`, `partB.py`, `partB2.py`,
`partCD.py`. No source file changed; `git status --porcelain` is empty (verified,
output below).

## Summary

The statistical layer itself is in very good shape: the inverse-variance
weighted mean, the block-boundary arithmetic, the day filtering, the
peak-window rule, `congestion_report`, the per-trajectory mode verdict and the
dwell-aware cleaner all do exactly what they document, several of them to
machine precision. **The published experiments, however, no longer reproduce —
and, importantly, they had already stopped reproducing before this review
touched anything.** `NearestMatcher`'s numbers are bit-identical across every
version tested, so all the movement is in the HMM: 48 of 105 recorded values
differ at `83880be` (pre-fix) and 51 of 105 at `HEAD`. Beyond stale digits,
three *qualitative* claims in `docs/methodology.md` no longer hold — the
"HMM wins most clearly at σ ≤ 15 m" claim has inverted, the "surplus spread"
that §3.4 exists to explain has essentially vanished, and Experiment C's
published peak window has moved to the other end of the day.

### Determinism

`scripts/paper_experiments.py` **is properly seeded** (`SEED = 42`, plus
`SEED + 1`, `SEED + 2`, and `range(60)` for the figure sweep). Two consecutive
runs at `HEAD` produced **0 differing leaf values out of 105**. The
reproducibility problem is drift in the code under it, not an unseeded script.

### Part A — three-way reconciliation

`docs/figures/experiment_results.json` was last written at `a9d3189`
(2026-07-08). Experiment A, road-level matching accuracy:

| σ | nearest PUB | nearest PRE | nearest POST | HMM **PUB** | HMM **PRE-FIX** | HMM **POST-FIX** | pub→post |
|---|---|---|---|---|---|---|---|
| 5 m | 91.2 | 91.2 | 91.2 | 96.7 | 87.3 | 94.8 | **−1.9** |
| 15 m | 75.6 | 75.6 | 75.6 | 80.6 | 75.5 | 84.7 | **+4.1** |
| 30 m | 53.4 | 53.4 | 53.4 | 55.6 | 63.3 | 69.8 | **+14.2** |
| 50 m | 34.5 | 34.5 | 34.5 | 36.4 | 39.5 | 42.8 | **+6.4** |

Two independent drifts, cleanly separated by the worktree run:

1. **published → pre-fix (48/105 values):** caused by commits between
   `a9d3189` and `83880be`, i.e. *before* this review. This is the more
   troubling one: the paper was already unreproducible at the commit Task 3
   audited, and nothing had noticed.
2. **pre-fix → post-fix (51/105 values):** caused by `5d2f1d8` (F-3.1). It moves
   every HMM number in the same direction — better.

`NearestMatcher` is identical to the last digit in all three
(`+1.4 / +12.7 / +59.9 / +146.8 %` bias unchanged), which is what isolates the
cause to the HMM.

### What is confirmed correct

| Claim | Verdict | Evidence |
|---|---|---|
| RNG seeding / determinism | **correct** | 0/105 differences across two runs |
| `aggregate_speeds` mean vs median | **correct** | mean 22.0, median 3.0 on `[1,2,3,4,100]` — both by hand |
| Block boundaries, all divisors of 24 | **correct** | 1→24 bins … 24→1 bin; last label always `HH:00-24:00`, start-inclusive/end-exclusive |
| `24 % block_hours != 0` | **warns, then behaves** | `block_hours=5` → 5 bins, last `20:00-24:00` (narrower), warning names the cause |
| Empty groups | **dropped, not NaN rows** | hours with no data simply absent; `min_samples` filters on `n` |
| `weight_by_variance` estimator | **exact** | speeds 10/20, vars 1/4 → weighted mean 12.0 = `Σ(x/v)/Σ(1/v)`; SEM 0.894427191 = `√(1/Σ(1/v))` |
| … reduces to unweighted at equal variance | **yes** | both 15.0000000000 |
| … non-finite / non-positive variance | **warns and drops** | zero, negative, NaN and inf all warn, `n` drops 2→1 |
| … guardrails | **correct** | missing `speed_var` and `statistic="median"` both raise naming the fix |
| `days=` presets, names, numbers, `None` | **correct** | Mon=0…Sun=6 verified against a real Monday; `weekday`→5, `weekend`→2, `all`/`None`→7; `"Funday"` raises |
| Timezone of hour-of-day / day-of-week | **consistent and documented** | the stored wall clock is used; `points._parse_times` converts tz-aware → local and strips tz, and warns for numeric epochs without `tz=`. No silent UTC-as-local path found |
| `classify_hours` window mode | **correct** | contiguous, wraps midnight (slow at 23/0/1 → `[0,1,23]`), ties break to the **earliest** window, deterministic |
| `classify_hours` degenerate cases | **loud** | 1 hour + window mode raises naming the fix; flat speeds warn "every observed hour tied at or below the median" |
| `congestion_report` | **correct** | 15 mph observed on a 30 mph road → ratio 0.4996; missing limit → NaN ratio, excluded from `n_edges_rated`; `maxspeed="0 mph"` → treated as absent, no divide-by-zero |
| `classify_movers` is per-trajectory | **correct — the central claim holds** | a "car" at 1.0 m/s for 12 of 20 intervals is still `vehicle`, and that verdict covers all 20 observations including its slowest |
| `suggest_mode_threshold` returns None with no walking hump | **correct** | vehicles-only → None; pedestrians-only → None; n=4, n=1, empty → None |
| `mover_features` p85 default | **well chosen** | at p50 the congested car's feature (2.24 mph) falls *below* the pedestrian's (2.91); at p75+ it is 44.74. The percentile is load-bearing and 85 is on the right side of the cliff |
| `cleaning` keeps congestion, drops outliers | **matches §4.1 exactly** | 1.5 and 5.0 m/s crawls: 30/30 jammed fixes kept, GPS outlier removed; a 300 s stop inside 25 m is removed as a dwell. The prose's "≲ 0.2 m/s sustained" boundary is precisely where the behaviour changes |

---

### F-4.1 — Every published experimental number in `docs/methodology.md` is stale, and was already stale before this review
- **Severity:** S3
- **Location:** `docs/methodology.md:23`, `:175-178`, `:287-289`, `:296-298`,
  `:302-305`, `:316`, `:319`, `:322`, `:413`, `:421-423`;
  `docs/figures/experiment_results.json`
- **Claim:** `scripts/paper_experiments.py` is supposed to reproduce every
  figure and number in `docs/methodology.md`. It does not. 51 of 105 recorded
  values differ at `HEAD`, and 48 of 105 already differed at `83880be` — so
  this is not solely a consequence of the Task 3 fixes; the paper had drifted
  from the code before this review began.
- **Evidence:**
  ```
  $ .venv/bin/python diff_json.py experiment_results.BASELINE.json experiment_results.PREFIX.json
  48 differing leaf values out of 105
  $ .venv/bin/python diff_json.py experiment_results.BASELINE.json experiment_results.RERUN1.json
  51 differing leaf values out of 105
  ```
  Headline movements (published → re-run at `HEAD`):
  ```
  §2.3 accuracy table  HMM   96.7 -> 94.8 | 80.6 -> 84.7 | 55.6 -> 69.8 | 36.4 -> 42.8
                       nearest         91.2 / 75.6 / 53.4 / 34.5   (UNCHANGED)
  §3.4 Table 3 dt sweep bias  +31.8% -> +20.6% | +8.3% -> +2.2% | -1.7% -> -5.7%
  §3.4 matcher rows HMM   +0.3 -> +0.1 | +4.1 -> +0.9 | +32.9 -> +21.8 | +105.7 -> +92.9
                  nearest +1.4 / +12.7 / +59.9 / +146.8            (UNCHANGED)
  Abstract  "+0.3 % bias" -> +0.1 %;  "15 % relative spread" -> 14.8 % (still holds)
  §4.4  peak_speed 7.410 -> 7.328 ;  offpeak_speed 15.613 -> 15.172 (truth 15.0)
  §4.4  coverage peak 378 -> 376 ; offpeak 488 -> 476
  ```
- **Expected vs actual:** The script is the stated authority for these numbers
  and it is deterministic, so the docs are what is wrong, not the code — and
  the code has moved in the *right* direction on every HMM figure. The
  qualitative direction of the paper's argument survives except where noted in
  F-4.2, F-4.3 and F-4.4.
- **Suggested fix:** Regenerate `docs/figures/` from the current code and update
  every number above, then decide whether regeneration becomes a release step
  so this cannot silently rot again — a test asserting the committed JSON
  matches a fresh run would catch it, at the cost of a ~15 s test.
- **Verdict:** DEFER -- into a single regeneration step after Tasks 5 and 6, the last passes
  that can still change behaviour (Task 8 is invariance-constrained). Doing it
  now means doing it twice.
- **Outcome:** deferred -- bundled with F-4.2/F-4.3/F-4.4 into one regeneration step after
  Tasks 5 and 6; see the Verdict.

---

### F-4.2 — The "HMM wins most clearly at σ ≤ 15 m" claim has inverted
- **Severity:** S3
- **Location:** `docs/methodology.md:184-185`
- **Claim:** The prose reads "the HMM wins at every noise level, most clearly in
  the σ ≤ 15 m regime where most real GPS data lives". With the published
  numbers that was true — the margin shrank monotonically with noise. With the
  current code the margin *peaks at σ = 30 m*, and the sentence now says the
  opposite of what the data shows.
- **Evidence:**
  ```
  --- accuracy margins HMM - nearest ---
    sigma= 5.0: published margin= +5.5pp   re-run margin= +3.6pp
    sigma=15.0: published margin= +5.0pp   re-run margin= +9.1pp
    sigma=30.0: published margin= +2.2pp   re-run margin=+16.4pp
    sigma=50.0: published margin= +1.9pp   re-run margin= +8.3pp
  ```
- **Expected vs actual:** Published margins `5.5, 5.0, 2.2, 1.9` decrease with
  σ, supporting the sentence. Current margins `3.6, 9.1, 16.4, 8.3` peak in the
  middle — the HMM is now *least* distinguishable from nearest-edge at σ = 5 m,
  where both are nearly right, and most valuable at σ = 30 m. The finding is a
  strengthening of the package's case, but the sentence as written is false.
- **Suggested fix:** Rewrite to match: the margin grows with noise up to
  σ ≈ 30 m and narrows at σ = 50 m where both matchers are failing. That is a
  better argument for the HMM than the original claim.
- **Verdict:** DEFER -- with F-4.1; the rewrite needs the final numbers.
- **Outcome:** deferred -- with F-4.1.

---

### F-4.3 — §3.4's "surplus spread" explanation now describes a phenomenon that has essentially vanished
- **Severity:** S3
- **Location:** `docs/methodology.md:322`
- **Claim:** The paper observes that measured spread exceeds the analytic
  prediction — "surplus (0.52 vs 0.42 at σ = 15; 1.07 vs 0.85 at σ = 30)" —
  and attributes the surplus to matching error. F-3.1 removed most of that
  matching error, so the surplus the sentence explains is now roughly a tenth
  of its published size, and the quoted pairs are wrong.
- **Evidence:**
  ```
  --- 'surplus spread' measured vs theory ---
    sigma= 5.0: theory=0.14  measured pub=0.15 post=0.15   surplus +0.006 -> +0.006
    sigma=15.0: theory=0.42  measured pub=0.52 post=0.45   surplus +0.099 -> +0.023
    sigma=30.0: theory=0.85  measured pub=1.07 post=0.87   surplus +0.018 (was +0.224)
    sigma=50.0: theory=1.41  measured pub=1.60 post=1.42   surplus +0.009 (was +0.188)
  ```
- **Expected vs actual:** The explanation was correct when written — the surplus
  *was* matching error, which is why fixing the matcher removed it. But the
  numbers quoted no longer exist, and a reader checking them against a fresh run
  will find the measured spread now tracks theory to within 0.02 at every σ.
  This is the strongest empirical confirmation in the pass that F-3.1 was a real
  bug: the paper's own residual is what it was hiding in.
- **Suggested fix:** Requote the new numbers and reframe: the analytic model now
  predicts observed spread to within ~2 % at every σ tested, and the former
  surplus is documented as having been matcher error since removed. Worth citing
  F-3.1 explicitly — it turns a caveat into a result.
- **Verdict:** DEFER -- with F-4.1; the rewrite needs the final numbers.
- **Outcome:** deferred -- with F-4.1.

---

### F-4.4 — Experiment C's published peak window has moved, and the experiment plants two equally valid peaks while reporting one
- **Severity:** S3
- **Location:** `docs/methodology.md:413`, `scripts/paper_experiments.py:327-414`
- **Claim:** The paper states "The detector returned **peak = {7, 8, 9}** and
  **off-peak = {2, 3, 4}**". It now returns `peak = {16, 17, 18}`. Both are
  genuine — the experiment plants 4 m/s congestion at *both* 07–09 and 16–18 —
  but with `n_peak = 3` the detector can only name one of them, so which one
  appears in the paper is decided by whichever window measures marginally
  slower, not by the ground truth.
- **Evidence:**
  ```
  experiment_c:
    true_peak_hours      pub=[7, 8, 9, 16, 17, 18]  pre=[7, 8, 9, 16, 17, 18]  post=[7, 8, 9, 16, 17, 18]
    detected_peak_hours  pub=[7, 8, 9]              pre=[7, 8, 9]              post=[16, 17, 18]
    detected_offpeak_hours  pub=[2, 3, 4]           pre=[2, 3, 4]              post=[2, 3, 4]
  ```
  The detector itself is sound and deterministic — on perfectly tied windows it
  breaks to the earliest, and a 0.01 m/s difference flips it correctly:
  ```
  B6. classify_hours — window mode with TWO equally-slow windows
    perfectly tied windows -> peak=[7, 8, 9] offpeak=[0, 1, 2] source=window
    morning slower -> peak=[7, 8, 9]
    evening slower -> peak=[16, 17, 18]
  ```
- **Expected vs actual:** The detector is behaving correctly; the *experiment
  design* is what makes the published value fragile. A reader would reasonably
  read "returned peak = {7,8,9}" as the detector having found the morning peak,
  when it found one of two and the choice sits inside measurement noise.
- **Suggested fix:** Two options, both cheap: run Experiment C with `n_peak = 6`
  so the detector can name both planted windows and the result becomes a real
  test of recall; or keep `n_peak = 3` and state in the prose that two equal
  peaks were planted and either is a correct answer. The first is the better
  experiment.
- **Verdict:** DEFER -- with F-4.1, but note the *script* change (n_peak=6) must land BEFORE
  regeneration, not with the prose.
- **Outcome:** deferred -- with F-4.1; the n_peak=6 script change must precede regeneration.

---

### F-4.5 — `suggest_mode_threshold`'s documented 10 % detection floor is about half the measured one
- **Severity:** S3
- **Location:** `src/redlight/modes.py:166-169`
- **Claim:** The docstring says "measured on synthetic mixes, a 10 % minority is
  found and a 5 % one is not". Across 8 seeds a 10 % pedestrian minority was
  found **0 times out of 8**. The real floor is around 15 % (6/8) and is solid
  only at 20 % (8/8).
- **Evidence:**
  ```
  C1b. 10% minority: is 'a 10% minority is found' robust? (modes.py:167-169)
      frac seed0  seed1  seed2  seed3  seed4  seed5  seed6  seed7    found/8
      0.05   no     no     no     no     no     no     no     no     0/8
      0.08   no     no     no     no     no     no     no     no     0/8
      0.10   no     no     no     no     no     no     no     no     0/8
      0.12   no     no     no     no     no     no     no     no     0/8
      0.15   yes    yes    no     yes    no     yes    yes    yes    6/8
      0.20   yes    yes    yes    yes    yes    yes    yes    yes    8/8
      0.30   yes    yes    yes    yes    yes    yes    yes    yes    8/8
      0.50   yes    yes    yes    yes    yes    yes    yes    yes    8/8
  ```
  Mix: vehicles `N(28, 6)` mph, pedestrians `N(3, 0.8)` mph, n = 300.
- **Expected vs actual:** The *guard is principled* — it is not tuned to one
  dataset, it degrades smoothly with minority fraction, and it correctly returns
  `None` for vehicles-only, pedestrians-only and n < 5. Only the quoted
  sensitivity figure is optimistic. Note the measured floor depends on the
  separation between the humps, so the honest statement is a range, not a
  number.
- **Suggested fix:** Restate as "a minority below roughly 15 % is not reliably
  found; 20 % is", and say the floor depends on how well separated the two
  populations are. If the original 10 % figure came from a specific mix, name
  it.
- **Verdict:** ACCEPT
- **Outcome:** fixed (75265a1)

---

### F-4.6 — `classify_hours` window mode returns unobserved hours, undocumented (confirms deferred F-2.3)
- **Severity:** S3
- **Location:** `src/redlight/aggregate.py:544-556`
- **Claim:** Task 2 deferred F-2.3 here. This pass confirms it and the analysis
  in it: mode 2 windows are contiguous by construction, so hours with zero
  observations are returned inside them, while the docstring only rules that out
  for mode 3. **The doc is incomplete; the behaviour is correct** — excluding
  empty hours would break contiguity and defeat the mode.
- **Evidence:** the window rule verified directly (`partB2.py`): contiguous,
  wrapping midnight, scored on the mean of *observed* hourly speeds so an empty
  hour contributes nothing to the score, ties broken to the earliest window.
  ```
    slow at 23,0,1    : peak=[0, 1, 23] wrap_ok=True
    perfectly tied windows -> peak=[7, 8, 9] offpeak=[0, 1, 2] source=window
  ```
  Combined with F-2.3's own evidence that three of six nominated hours carried
  no observations on the shipped sample data.
- **Expected vs actual:** As F-2.3 stated. The consequence is real because those
  hour lists flow into `assign_segment_speeds`, so an edge can be labelled peak
  or off-peak for an hour nothing was measured in.
- **Suggested fix:** As F-2.3 proposed — one sentence in mode 2 saying the window
  is contiguous and may therefore include hours with no observations, which are
  scored as absent rather than as slow. No code change.
- **Verdict:** ACCEPT
- **Outcome:** fixed (582cec2)

---

### F-4.7 — `peak_analysis` reports a peak and an off-peak on perfectly flat data, with no warning
- **Severity:** S5
- **Location:** `src/redlight/aggregate.py:417-520`
- **Claim:** Given 24 hours of identical speeds, `peak_analysis` returns
  `peak = [0, 1, 2]` and `off_peak = [21, 22, 23]` at the same speed, silently.
  `classify_hours` warns on exactly the same input. The two functions disagree
  about whether a degenerate split is worth mentioning.
- **Evidence:**
  ```
  perfectly flat (all hours 10.0 m/s):
    peak:     00:00-01:00, 01:00-02:00, 02:00-03:00  mean_speed 10.0
    off_peak: 21:00-22:00, 22:00-23:00, 23:00-24:00  mean_speed 10.0
  ```
  versus, for the same data through `classify_hours`:
  ```
  UserWarning: classify_hours: every observed hour tied at or below the median
  speed, so the off-peak set is empty. Consider window mode (n_peak=/n_offpeak=)
  or explicit hour lists.
  ```
  Among tied hours the ordering also comes from the aggregation's row order
  rather than the clock (`ranked` lists 03:00 before 00:00 before 06:00), so the
  reported "peak" on tied data is an artifact of sort order.
- **Expected vs actual:** The returned speeds are identical so a careful reader
  can see the split is meaningless, which is why this is S5 rather than higher.
  But a caller reading only `peak` would report peak hours 00–03 on data with no
  peak at all.
- **Suggested fix:** Warn when the peak and off-peak representative speeds are
  equal (or within a tolerance), mirroring the wording `classify_hours` already
  uses.
- **Verdict:** ACCEPT
- **Outcome:** fixed (eed166f)

---

### F-4.8 — `paper_experiments.py` claims a runtime an order of magnitude longer than it has
- **Severity:** S5
- **Location:** `scripts/paper_experiments.py:14`
- **Claim:** The module docstring says "Everything is seeded; run time is a
  couple of minutes". It runs in 15 seconds.
- **Evidence:**
  ```
  $ time .venv/bin/python scripts/paper_experiments.py
  real	0m15.233s
  user	0m14.759s
  sys	0m0.426s
  ```
- **Expected vs actual:** "a couple of minutes" vs 15.2 s on the machine
  described in `00-baseline.md` (2017 dual-core i5, 4 logical CPUs) — which is
  modest hardware, so a faster machine will not close the gap. The claim
  discourages running the script, which is precisely the thing that would have
  caught F-4.1.
- **Suggested fix:** Restate as "well under a minute", or drop the estimate.
- **Verdict:** ACCEPT
- **Outcome:** fixed (829aa9b)

---

## Repo state on exit

```
$ git status --porcelain
(empty)
```

`docs/figures/` was restored with `git checkout -- docs/figures/` after each
run, and the throwaway worktree at `83880be` was removed with
`git worktree remove --force`. Nothing was committed.

## Unverified suspicions

1. **The published → pre-fix drift (48/105) is unattributed.** This pass
   established *that* it happened and bounded it between `a9d3189` (2026-07-08)
   and `83880be`, but did not bisect which commit caused it. `02-doc-drift.md`
   nominates `c9a92bb` and `a03733f` as candidates. A bisect over
   `experiment_a.hmm_acc` would settle it in a few runs and is worth doing
   before regenerating the paper, so the regeneration is not silently blessing
   an unrelated regression.
2. **Whether the σ = 5 m accuracy regression (96.7 → 94.8, and 87.3 → 94.8 from
   pre-fix) is fully explained by F-3.1.** F-3.1 improved every other cell; σ = 5
   is the one place the current code is worse than the published figure. It is
   still better than pre-fix, so nothing regressed *in this review*, but the
   low-noise behaviour of the corrected transition term was not investigated.
3. **`mover_features`' percentile cliff between p50 and p75** is sharp on the
   synthetic case built here (2.24 → 44.74 mph). Whether real congested vehicle
   trajectories sit as close to that cliff as this constructed one was not
   tested, and it governs how safe the `percentile=85.0` default is on customer
   data.
