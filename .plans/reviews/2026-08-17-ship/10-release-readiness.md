# v0.6.0 release readiness — assessment
**Pass:** Task 10
**Date:** 2026-08-19
**Commit assessed:** `f1126ac`
**Scope:** every findings file in this directory, `CHANGELOG.md`,
`src/redlight/__init__.py`, `README.md`, `MANIFEST.in`,
`.github/workflows/ci.yml`, `.gitlab-ci.yml`, `mkdocs.yml`, `pyproject.toml`,
and both built distributions.
**Method:** every check below was run. Clean-environment installs used
throwaway `python3.12` venvs outside the repo, driving a real pipeline
(load → match → derive → aggregate → export) on `examples/sample_data`.

## Recommendation: **GO**

Nothing unfixed blocks a v0.6.0 GitHub release. Three things found in this pass
were fixed here (all mine, all introduced today); two are recorded as open and
argued below; CI is green across all eight jobs.

---

## 1. Unfinished business

47 findings across nine passes. **34 fixed, 6 rejected, 7 open.** Every open one:

| Finding | Sev | Verdict | Blocks? |
|---|---|---|---|
| F-3.2 same-edge `delta = 0` on curved edges | S1 | DEFER | **No — argued below** |
| F-1.5 → F-2.5 `_proj` accuracy claim | S3 | DEFER chain | No — resolved: Task 3 measured 4.4 nm / 9.5 nm across all 120 zones, docstring corroborated |
| F-2.3 `classify_hours` mode-2 doc gap | S3 | DEFER | No — resolved as F-4.6, fixed in `582cec2` |
| F-3.4 out-of-range latitude | S1 | DEFER | No — resolved as F-5.2, fixed in `ecbe530` |
| F-7.2 / F-7.6 benchmark measures mostly-unmatched points | S3/S4 | untriaged | No — benchmark hygiene, no user-facing claim left after `f1126ac` |
| F-7.3 / F-8.1 `derive_speeds` is the larger target | S4 | untriaged | No — partly acted on already (1.54× faster today) |
| F-7.5 / F-8.2 further matcher micro-optimizations | S4 | untriaged | No — performance only |

**The S1 exception, argued explicitly.** F-3.2 is nominally S1 and unfixed. It
should not block, because **its own evidence was withdrawn during the Task 3 fix
cycle**. The finding claimed the same-edge shortcut caused a wrong decode; the
network used to demonstrate that turned out to be topologically disconnected, so
the decode was driven by the saturating penalty, not the shortcut. Re-tested
after F-3.1 landed, toggling the rule changed **no decode at all** — not on the
original network, not on a corrected one, and not across 300 randomised networks
with curved edges (0 of 3,600 fixes). What survives is a real but unreachable
modelling gap: `rd = gc_step` is not the arc. It is recorded, not lost.

**Task 9 (API surface) was never run.** By its own charter its findings are
recorded for a future major version rather than fixed — the API is frozen for
this cycle — so it is v0.7 planning material. Worth running before v0.7; it does
not gate v0.6.0.

## 2. Version consistency — **PASS**

- `src/redlight/__init__.py:78` → `__version__ = "0.6.0"`; `pyproject.toml:67`
  reads it dynamically. Single source of truth, no drift.
- `README.md:3` masthead reads **0.6.0** (fixed earlier as F-2.1; a test now
  pins it).
- No other file hardcodes a version. Existing tags run `v0.1.0`–`v0.5.0`, so
  `0.6.0` is unreleased.
- **0.6.0 is the right number.** The release contains a breaking rename, which
  on a 0.x line semver permits as a minor bump — provided it is stated loudly,
  which item 3 covers.

## 3. CHANGELOG — **FIXED IN THIS PASS**

Two problems, both corrected:

- **The date was 2026-08-06**, before the rename, the study-area work and this
  entire review. Now `2026-08-19`.
- **The rename was buried ~35 lines into the "Changed" section.** The plan calls
  leading with it "the single most important edit in this task", and it was not
  being done. There is now a `### Breaking — read this first` section at the top
  of 0.6.0 stating in the loudest available terms that every existing
  `import roadtraffic` breaks, with a before/after block, the extras rename, and
  the reassurance that no other API changed so find-and-replace is the whole
  migration.

The section also now covers **this review's 18 fix and perf commits**, which it
did not before — grouped as correctness fixes (with the numbers that moved),
error-message fixes, performance, and the regenerated paper.

## 4. Packaging — **PASS** (after a fix)

```
$ .venv/bin/twine check dist/*
Checking dist/redlight-0.6.0-py3-none-any.whl: PASSED
Checking dist/redlight-0.6.0.tar.gz: PASSED
```

| Check | Result |
|---|---|
| sdist carries `tests/conftest.py` | present |
| sdist carries `tests/data/` | present |
| sdist carries `LICENSE`, `MANIFEST.in`, `docs/*.md` | present |
| sdist excludes `.DS_Store`, `site/`, `build/`, `.pytest_cache`, `__pycache__`, `.plans`, `benchmarks/`, `docs/figures` | all absent |
| wheel excludes `tests/`, `examples/`, `docs/`, `benchmarks`, `.DS_Store` | all absent |
| wheel top level | `redlight/` + `redlight-0.6.0.dist-info/` only |

A `.DS_Store` does exist in the repo root and is gitignored; confirmed
`MANIFEST.in` does not pull it into the sdist either.

## 5. Clean-environment install — **PASS** (after a fix)

Both artifacts installed into throwaway venvs and driven from outside the repo:

```
  redlight 0.6.0 from .../venv-whl/lib/python3.12/site-packages/redlight
  network: 62 edges   points: 1682
  matched: 1681/1682
  intervals: 1578   edge obs: 3464
  aggregate rows: 13
  geojson features: 31
  OK
```

Identical from the sdist. Installed dependencies are **core only** — networkx,
numpy, pandas, scipy, shapely (plus transitive python-dateutil, six). No pyproj,
no pyogrio, no matplotlib: the package's headline "no PROJ" promise holds in a
real install, not just in the repo.

The unpacked sdist's own suite: **399 passed, 13 skipped** — the skips are the
optional-extras tests, correctly absent in a core-only environment.

## 6. CI — **PASS** (after three fixes)

All eight jobs green on `50a8e35`:

```
✓ build + verify the distributions      ✓ core install has no PROJ
✓ docs build                            ✓ py3.9  on ubuntu / macos
✓ py3.11 on ubuntu                      ✓ py3.13 on ubuntu / macos
```

**CI was red for three consecutive runs before this pass, entirely because of
tests I added earlier today.** See F-10.1.

- **Matrix vs classifiers:** `pyproject.toml` claims 3.9–3.13; CI tests
  3.9/3.11/3.13 on ubuntu and 3.9/3.13 on macOS. **3.10 and 3.12 are claimed but
  untested.** Sampling the ends and middle of a range is normal practice and the
  package is pure Python, so this is recorded rather than raised — see F-10.3.
- **Lint invocation** is `ruff check src tests scripts examples`, which still
  omits `benchmarks/`. That directory is currently clean (verified), so the gap
  is latent. It is the finding Task 9 was to own; since Task 9 did not run, it is
  recorded here as F-10.4.
- **`.gitlab-ci.yml` is vestigial.** It is *gitignored* (`.gitignore`, last
  block), so it is never published to the GitHub remote, and `.gitignore` itself
  states "The active pipeline is `.github/workflows/ci.yml`". It is a local
  leftover of the pre-GitHub migration. No action needed; it cannot run.

## 7. Docs site — **PASS**

```
$ .venv/bin/mkdocs build --strict
INFO - Documentation built in 1.24 seconds     (exit 0)
```

Every nav entry resolves; no broken internal link. The one line matching
`warning` is a Material-for-MkDocs branding notice about the upcoming mkdocs
2.0, not a build warning — `--strict` exits 0. `site/` is gitignored and not
committed.

## 8. Repo hygiene — **PASS**

- `git status --porcelain` → empty.
- `build/`, `dist/`, `site/`, `.pytest_cache/`, `.ruff_cache/`, `__pycache__/`
  and `.DS_Store` all ignored — verified by creating them, not by pattern
  inspection (`git check-ignore` on a non-existent path is misleading).
- `[project.urls]` all point at `github.com/cesouth/redlight` and match the
  remote.
- `examples/sample_data/` is gitignored *by design* — it is generated by
  `examples/00_setup/generate_sample_data.py`, not shipped.

---

### F-10.1 — The invariance tests I added today broke CI three separate ways
- **Severity:** S2
- **Location:** `tests/test_matching_invariance.py`,
  `tests/test_speeds_invariance.py`
- **Claim:** Both tests encoded properties of the machine that generated their
  expectations rather than properties of the package. CI went red on
  `caebd20` and stayed red for three runs. Each failure was a different
  non-portable assertion:
  1. **Absolute float tolerance.** `abs=1e-12` pinned this machine's bit
     pattern; Linux runners differ by ~3e-12 relative on the same arithmetic
     (different libm and GEOS build). Fixed to `rel=1e-9` in `be6809c`.
  2. **Directed edge ids.** A two-way road is two directed edges over one
     geometry at one snap distance, and *nothing breaks that tie on the data* —
     Task 3 established the choice is arbitrary, and
     `tests/test_matching_batch.py` has always accepted either direction. The
     test was pinning a coin flip. Fixed by canonicalising to
     `min(road_edge_ids(edge))` in `e6f87c3`.
  3. **Exact dtype strings.** The supported pandas range spans
     `datetime64[ns]` and `datetime64[us]`, so the 3.9 job (pandas floor)
     disagreed. Fixed to compare dtype *kind* in `50a8e35`.
- **Evidence:** `gh run list` showed `failure` on 32310491364, 32317565046,
  32318459809 and 32318684608, then `success` on 32318907063 with all eight
  jobs green.
- **Expected vs actual:** An invariance test should pin what the package
  determines, not what the platform happens to produce. All three assertions
  failed that standard.
- **Suggested fix:** Done. Each relaxation was **re-verified against the
  mutations the test exists to catch** before being accepted — wrong endpoint,
  dropped lead-in, dropped lead-out, emission losing its square, a 1e-7
  perturbation, and a bool becoming an int. All still caught.
- **Verdict:**
- **Outcome:** fixed (`be6809c`, `e6f87c3`, `50a8e35`)

---

### F-10.2 — The sdist shipped two tests that could not import their fixtures
- **Severity:** S2
- **Location:** `MANIFEST.in`, `tests/test_matching_invariance.py`,
  `tests/test_speeds_invariance.py`
- **Claim:** Both invariance tests imported their generators from
  `benchmarks/profile_hmm.py`, but `MANIFEST.in` prunes `benchmarks/` as
  development scaffolding. The sdist therefore failed collection with 2 errors —
  the exact failure `MANIFEST.in` was written to prevent, reintroduced by me.
- **Evidence:**
  ```
  ERROR tests/test_matching_invariance.py
  ERROR tests/test_speeds_invariance.py
  !!!!!!!!! Interrupted: 2 errors during collection !!!!!!!!!
  ```
  from an unpacked sdist. **Caught only by the clean-environment install that
  Task 0 explicitly deferred to this pass** — the in-repo suite was green
  throughout.
- **Expected vs actual:** The generators became test fixtures the moment the
  tests used them, so they belong in `tests/`, which `recursive-include` already
  ships.
- **Suggested fix:** Done — moved to `tests/_synth.py`; the benchmarks import
  from there. One copy, and the dependency points from scaffolding to the suite
  rather than the reverse. Unpacked sdist now runs 399 passed, 13 skipped.
- **Verdict:**
- **Outcome:** fixed (`e14db79`)

---

### F-10.3 — Python 3.10 and 3.12 are claimed but never tested
- **Severity:** S5
- **Location:** `pyproject.toml:18-23`, `.github/workflows/ci.yml:22`
- **Claim:** The classifiers advertise 3.9–3.13. CI tests 3.9/3.11/3.13 on
  ubuntu and 3.9/3.13 on macOS. 3.10 and 3.12 are asserted, not verified.
- **Evidence:** `python-version: ["3.9", "3.11", "3.13"]` against five
  `Programming Language :: Python :: 3.x` classifiers.
- **Expected vs actual:** Sampling the ends and middle of a supported range is
  common and defensible for a pure-Python package with no version-gated code
  paths. Recorded so the claim is a decision rather than an oversight.
- **Suggested fix:** Either add 3.10 and 3.12 to the ubuntu matrix (cheap; the
  suite runs in ~40 s) or leave as is deliberately. Not release-blocking.
- **Verdict:**
- **Outcome:**

---

### F-10.4 — CI still does not lint `benchmarks/`
- **Severity:** S5
- **Location:** `.github/workflows/ci.yml:54`
- **Claim:** The lint step is `ruff check src tests scripts examples`, omitting
  `benchmarks/` — which now holds three files, two of them added during this
  review, one of which the test suite imports from.
- **Evidence:** `benchmarks/` is currently clean
  (`ruff check src tests scripts examples benchmarks` → All checks passed), so
  the gap is latent, not an active failure. The Global Constraints flagged it as
  a Task 9 candidate; Task 9 did not run.
- **Expected vs actual:** `tests/_synth.py` is linted, but the benchmark that
  imports it is not.
- **Suggested fix:** Add `benchmarks` to the CI lint invocation. One word, and
  it would not turn CI red today.
- **Verdict:**
- **Outcome:**

---

## Repo state on exit

```
$ .venv/bin/pytest -q
412 passed
$ .venv/bin/ruff check src tests scripts examples benchmarks
All checks passed!
$ git status --porcelain
(empty)
$ gh run list --limit 1
completed  success  ...  CI  main  push
```

Tags stop at `v0.5.0`. **No tag was created and no release was published** —
that is your call, per the plan.

## If GO, the remaining steps

The plan's Step 3 command block still applies, with one caveat: verify the
`sed` range against the CHANGELOG's actual headings before letting it become
public release notes. The 0.6.0 section now begins with
`### Breaking — read this first`, and the extract must include it — that
section is the single most important thing in these notes.
