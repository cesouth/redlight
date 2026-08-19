# Test-suite quality audit — findings
**Pass:** Task 6
**Date:** 2026-08-18
**Commit reviewed:** `199ebd0`
**Scope:** all 21 files under `tests/` (4,710 lines, 313 test functions) plus
`tests/conftest.py`. The source was read only to choose mutation targets.
**Method:** a 13-mutation spot-check driven by `…/scratchpad/t6/mutate.py`,
which applies one mutation, runs `.venv/bin/pytest -q -x`, reverts with
`git checkout -- src/` and asserts the tree is clean before continuing; plus
AST analysis of every test function for assertion quality, a regex sweep for
numeric tolerances, per-file isolation runs, and an `__all__` coverage check.

## Summary

The suite is stronger than a green light usually implies: **11 of 13 plausible
mutations to load-bearing lines were caught**, every test file passes in
isolation, there are no skips, no order dependence and no unseeded RNG. The two
survivors are the finding: **nothing pins the shape of `HMMMatcher`'s emission
log-prob, and nothing pins the speed error model** — both were verified by hand
in Task 3 against external references, and both could now be changed to
something wrong without a single test failing. There is no tolerance inflation
anywhere in the suite.

### Mutation spot-check

Baseline before and after every mutation: **403 passed**, `src/` clean.

| # | File | Mutation | Caught? | First failing test |
|---|---|---|---|---|
| 1 | `_geo.py` | Vincenty `cos_sigma`: `+` → `-` | **YES** | `test_analysis.py::test_circuity_straight_road_near_one` |
| 2 | `_geo.py` | drop the `delta_sigma` correction | **YES** | `test_geo.py::test_matches_reference_values` |
| 3 | `_proj.py` | `utm_forward`: drop the false easting | **YES** | `test_derive_math.py::test_derive_speeds_turn_across…` |
| 4 | `_proj.py` | `utm_epsg_to_zone`: off-by-one, southern base | **YES** | `test_proj.py::test_utm_forward_matches_reference` |
| 5 | `speeds.py` | hop distance: drop the lead-out term | **YES** | `test_derive_math.py::test_derive_speeds_turn_across…` |
| 6 | `speeds.py` | `sigma_comb`: one endpoint instead of both | **no** | — |
| 7 | `matching.py` | emission: drop the square | **no** | — |
| 8 | `matching.py` | transition: flip the sign of the decay | **YES** | `test_matching.py::test_hmm_gap_transition_uses_correct_anchor` |
| 9 | `aggregate.py` | inverse-variance mean: weight by `var`, not `1/var` | **YES** | `test_aggregate.py::test_inverse_variance_weighting_from…` |
| 10 | `aggregate.py` | block boundary: off-by-one on the bin start | **YES** | `test_aggregate.py::test_aggregate_hourly_mean_exact` |
| 11 | `modes.py` | valley predicate: flip the hump-mass comparison | **YES** | `test_modes.py::test_suggest_threshold_finds_the_valley` |
| 12 | `cleaning.py` | MAD z-score: flip the keep/drop comparison | **YES** | `test_cleaning.py::test_mad_outliers_removed` |
| 13 | `analysis.py` | dead ends: `deg == 1` → `deg == 2` | **YES** | `test_analysis.py::test_grid_streets_per_node_and_inter…` |

**11/13 caught, 2 survived.**

### What is clean (recorded so it is not re-tested)

| Check | Result |
|---|---|
| Skips with all extras installed | **none** — `pytest -q -rs` reports 0 |
| Order dependence | `-p no:randomly` → 403 passed; `test_aggregate.py test_geo.py` alone → 59 passed; **every one of the 21 files passes on its own** |
| Shared mutable state in `conftest.py` | none found — every fixture is function-scoped and builds from `tmp_path` |
| RNG determinism | all 5 RNG uses in tests are seeded (`test_cleaning.py:105`, `test_matching_batch.py:18`, `test_pipeline_e2e.py:51`, `test_modes.py:259,332`); the seeds are fixture params or loop indices, so runs are reproducible |
| Tolerance inflation | **none found.** The loosest numeric assertions are `abs=1.0`/`abs=2.0` on ~1000 m lengths, and those are deliberately coarse (see F-6.4); everything else is `rtol=1e-3` or tighter, and `test_proj.py` asserts to `abs=1e-3` m against pinned reference values |
| Public API coverage | 30 of 33 `__all__` symbols appear by name in the suite |

---

### F-6.1 — Nothing pins the shape of the HMM emission log-prob
- **Severity:** S5
- **Location:** `src/redlight/matching.py:202-205`; no covering test
- **Claim:** Changing `-0.5 * (dist / sigma_z) ** 2` to `-0.5 * (dist / sigma_z)`
  — turning the Gaussian into an exponential, i.e. abandoning Newson & Krumm's
  emission model — leaves the entire suite green. This is one of the two
  probabilities the matcher is built on, and Task 3 had to verify it against
  `scipy.stats.norm.logpdf` because the suite does not.
- **Evidence:**
  ```
   7 matching.py   emission: drop the square                                 no
  ```
  (403 passed with the mutation in place; `src/` verified clean before and after.)
- **Expected vs actual:** The transition log-prob's sibling mutation (#8) *is*
  caught, by `test_hmm_gap_transition_uses_correct_anchor` — so the asymmetry is
  accidental, not a decision. Expected a direct unit test of
  `_emission_logp`, the way `test_geo.py` pins the geodesy against pinned
  reference values.
- **Suggested fix:** Add a unit test asserting `_emission_logp` equals
  `log N(dist; 0, sigma_z)` at several distances and two values of `sigma_z`,
  with the normalising constant — pinned constants rather than a `scipy` call,
  matching `test_geo.py`'s house style of not depending on an oracle at test
  time. Task 3 recorded the exact values.
- **Verdict:** ACCEPT
- **Outcome:**

---

### F-6.2 — Nothing pins the speed error model that `weight_by_variance` consumes
- **Severity:** S5
- **Location:** `src/redlight/speeds.py:362-364`; no covering test
- **Claim:** Replacing `sigma_comb = hypot(sigma[a], sigma[b])` with
  `sigma_comb = sigma[a]` — dropping one endpoint from the documented model
  `σ_v = √(σᵢ² + σⱼ²) / dt` — leaves the suite green. `speed_sigma_mps` and
  `speed_var` are not incidental outputs: `speed_var` is what
  `aggregate_speeds(weight_by_variance=True)` weights by, so a wrong model
  silently re-weights every aggregate.
- **Evidence:**
  ```
   6 speeds.py     sigma_comb: use one endpoint instead of both              no
  ```
  Note the estimator that *consumes* this value is well covered — mutation #9
  (inverse-variance mean) is caught by
  `test_aggregate.py::test_inverse_variance_weighting_from…`. It is the input
  to that estimator that is unpinned.
- **Expected vs actual:** Expected a test asserting `speed_sigma_mps` for a
  known pair of per-fix accuracies and a known `dt`; Task 3 verified it by hand
  to 1e-12 and the values are in `03-numerical-accuracy.md`.
- **Suggested fix:** One test with `pos_accuracy_col` set to two different
  values and an exact assertion on `speed_sigma_mps` and `speed_var`.
- **Verdict:** ACCEPT
- **Outcome:**

---

### F-6.3 — The `n_jobs` test asserts the row count, not the property the docstring claims
- **Severity:** S5
- **Location:** `tests/test_matching_batch.py::test_hmm_n_jobs_minus_one_runs`
- **Claim:** `HMMMatcher`'s docstring states "Results are identical to serial
  matching — trajectories are independent; only the schedule changes." The test
  named for that behaviour asserts only `len(out) == len(pts)`.
- **CORRECTION (added during the fix cycle):** **this overstated the gap.** The
  AST sweep looked at the function in isolation; its immediate neighbour
  `test_hmm_parallel_equals_serial` (`tests/test_matching_batch.py:135`) already
  asserts `pd.testing.assert_frame_equal(serial, parallel)` for `n_jobs=2`, so
  the equality property *was* covered. What was genuinely untested is the
  `n_jobs=-1` resolution path specifically — `_resolve_n_jobs` turning `-1` into
  `os.cpu_count()`, and the different chunking that follows from a different
  worker count. That is a narrower gap than the finding as written claims.
- **Evidence:** the test's only assertion, from the AST sweep:
  ```
  test_matching_batch.py   test_hmm_n_jobs_minus_one_runs        len(out) == len(pts)
  ```
- **Expected vs actual:** Expected the serial and parallel frames compared
  directly — the docstring's claim is an equality, so the test should be one.
  Actual: a smoke test with a length check.
- **Suggested fix:** Run the same matcher with `n_jobs=1` and `n_jobs=-1` over
  the same points and assert the two frames are equal. This is also exactly the
  invariance Task 8 will need, so it is worth writing before that pass rather
  than after.
- **Verdict:** ACCEPT
- **Outcome:**

---

### F-6.4 — Two coarse length tolerances of my own, and three untested public constants
- **Severity:** S5
- **Location:** `tests/test_network.py:431,442,451`; `redlight.__all__`
- **Claim:** Two separate small items, both minor and both recorded for
  completeness rather than urgency.
  1. The three tests added for F-5.1 in `c4d40ab` assert edge length to
     `abs=1.0` m and `abs=2.0` m on ~1000 m roads. That is deliberately coarse —
     they exist to separate 1,000 m from 16,585,698 m — but at that tolerance
     they would not notice a 0.1 % projection error. They are the loosest
     numeric assertions in the suite, and they are mine, so they belong in this
     record.
  2. `MODE_PEDESTRIAN`, `MODE_VEHICLE` and `MODE_UNKNOWN` are the only members
     of `redlight.__all__` that never appear by name in the suite. Their
     *values* are exercised indirectly (`test_modes.py` compares against the
     literals `"vehicle"`/`"pedestrian"`), which is precisely the problem:
     renaming a constant's value would break users without failing a test.
- **Evidence:**
  ```
  === 3. TOLERANCE: loosest numeric assertions ===
    2          test_network.py:451  assert net.edge_length(...) == pytest.approx(1113.2, abs=2.
    1          test_network.py:442  assert net.edge_length(...) == pytest.approx(1000.0, abs=1.
    1          test_network.py:431  assert net.edge_length(...) == pytest.approx(1000.0, abs=1.

  === 6. PUBLIC API COVERAGE ===
    33 public symbols; untested by name: ['MODE_PEDESTRIAN', 'MODE_VEHICLE', 'MODE_UNKNOWN']
  ```
- **Expected vs actual:** For (1), a projection round-trip is nanometre-accurate
  (Task 3 measured 9.5 nm across all 120 zones), so `abs=1e-3` would still
  comfortably pass while catching far more. For (2), the literals in
  `test_modes.py` should be the constants.
- **Suggested fix:** Tighten the three tolerances to `abs=1e-3`; replace the
  string literals in `test_modes.py` with the exported constants.
- **Verdict:** ACCEPT
- **Outcome:**

---

## Assertion quality — the full list

30 of 313 test functions (9.6 %) assert only on shape, length, membership or
type, after excluding every test that uses `pytest.raises`, `pytest.warns`,
`approx` or `assert_allclose` — those *do* assert behaviour and a naive count
badly overstates the problem (an unfiltered sweep flags 83).

Most of the 30 are legitimately structural and should stay as they are —
`test_matchers_share_output_schema` is *supposed* to compare column sets,
`test_mover_features_empty_input_keeps_the_same_columns_as_a_full_one` is about
columns by design, and `test_doc_example_runs` exists to catch exceptions.
Three are worth strengthening, and only the first is load-bearing enough to
have earned its own finding:

| Test | Only asserts | Why it matters |
|---|---|---|
| `test_matching_batch.py::test_hmm_n_jobs_minus_one_runs` | `len(out) == len(pts)` | **F-6.3** — the docstring claims equality with serial |
| `test_cleaning.py::test_trajectory_filter_keeps_slow_moving_traffic` | `len(out) == n` | the point is *which* rows survive, not how many |
| `test_matching_batch.py::test_csr_dist_cache_lru_bound` | `len(cache._cache) <= 3` | does not check the *right* entries were evicted |

## Repo state on exit

```
$ git status --porcelain
(empty)
$ .venv/bin/pytest -q
403 passed, 13 warnings
```

Every mutation was reverted immediately after measurement, with a clean-tree
assertion between each; no mutation was left in the tree at any point.

## Unverified suspicions

1. **The mutation sample is 13 lines out of a package of ~6,000.** An 85 % catch
   rate on hand-picked load-bearing lines is a good signal but not a coverage
   measurement, and the two survivors were both found in the modules Task 3
   already flagged as mathematically load-bearing — which suggests the sample
   was well chosen, not that the rest is safe. A real mutation-testing run
   (`mutmut`, `cosmic-ray`) would settle it; both are dev-only dependencies, so
   nothing in the "no new runtime dependencies" constraint blocks that.
2. **Order dependence was tested by file, not by test.** Every file passes
   alone and the whole suite passes with collection order unchanged, but
   `pytest-randomly` is not installed so no shuffled-within-file run was
   possible. Installing it as a dev extra would close this properly.
3. **Whether `test_docs.py` would notice a doc example that runs but prints
   something wrong.** It executes the snippets and asserts nothing about their
   output, which is defensible — but it means the doc-drift class Task 2 hunted
   by hand cannot be caught automatically.
