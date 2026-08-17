# Ship review — recorded baseline

**Pass:** Task 0
**Date:** 2026-08-17
**Commit reviewed:** `e30b667` (`fix: stop two tests assuming optional dependencies are installed`)
**Branch:** `main`
**Scope:** environment construction and baseline measurement only. No source file read for review, none changed.
**Method:** commands below, run from the repo root. Every number here is verbatim output.

## Summary

The package is in good shape at `e30b667`: **375 tests pass, none skip, ruff is
clean across every directory including the one CI does not lint, and both
distributions build and pass `twine check`.** The one real problem was the
environment itself — there was no working interpreter on this machine before
this pass. The dependency resolver pulled the current majors (pandas 3.0.5,
numpy 2.5.2, scipy 1.18.0), so this baseline also demonstrates the package is
clean against the *upper* end of its declared ranges, which the CI matrix does
not separately pin.

---

## Interpreter

| | |
|---|---|
| Interpreter | `python3.12` → **Python 3.12.13** |
| Venv path | `.venv/` (repo root, gitignored at `.gitignore:6`) |
| Executables | `.venv/bin/python`, `.venv/bin/pytest`, `.venv/bin/ruff`, `.venv/bin/mkdocs` |
| pip | 25.0.1 |

**Do not use `python3` or `python3.11` on this machine.**
- `python3` is dyld-broken: `dyld cache '(null)' not loaded: syscall to map cache into shared region failed`
- `python3.11` no longer exists (it was the interpreter used in earlier sessions; it is gone)

Note the venv is Python **3.12**, while the package declares
`requires-python = ">=3.9"` and CI tests 3.9/3.11/3.13. Nothing in this review
exercises the 3.9 floor locally — that remains CI's job, and any pass that wants
to claim something about 3.9 must say so explicitly rather than infer it here.

## Install command used

```bash
python3.12 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e ".[dev,mapping,shapefile,crs,docs]"
.venv/bin/pip install build twine
```

All five extras were installed deliberately so that no test self-skips and every
optional code path (pyproj fallback, pyogrio readers, matplotlib rendering,
mkdocs build) is reachable during the audit.

## Resolved dependency versions

| Package | Version | Declared floor |
|---|---|---|
| numpy | 2.5.2 | `>=1.21` |
| pandas | 3.0.5 | `>=1.3` |
| scipy | 1.18.0 | `>=1.7` |
| shapely | 2.1.2 | `>=2.0` |
| networkx | 3.6.1 | `>=2.6` |
| pyproj | 3.7.2 | `>=3.2` (`crs` extra) |
| pyogrio | 0.13.0 | `>=0.10` (`shapefile` extra) |
| matplotlib | 3.11.1 | `>=3.5` (`mapping` extra) |
| pytest | 9.1.1 | `>=7.0` (`dev`) |
| ruff | 0.16.3 | `>=0.4` (`dev`) |
| mkdocs / mkdocs-material | 1.6.1 / 9.7.7 | `>=1.5` / `>=9.0` |

`redlight` installed as an editable wheel, version **0.6.0**.

## Test baseline

```
$ .venv/bin/pytest -q
375 passed, 6 warnings in 61.64s (0:01:01)
```

**375 passed / 0 failed / 0 skipped / 0 xfailed.**

This is the number every later pass regresses against. It supersedes the stale
figures in older documents — `.plans/2026-08-04-drop-pyproj.md` cites 289 and
`.plans/2026-07-31-mover-mode-screening.md` cites 233; both are historical.

### Skip list

**Empty.** Confirmed explicitly:

```
$ .venv/bin/pytest -q -rs
375 passed, 6 warnings in 17.20s
```

No `SKIPPED` lines. With all five extras installed, every test in the suite
executes. This is the ideal state for Task 6 — any test that *does* skip in a
later pass is a signal something regressed in the environment, not in the code.

### Runtime

- **Cold: 61.64s.** First run of the session.
- **Warm: 17.20s.** Immediately after, unchanged tree.

The 3.6x gap is worth a look in Task 7 — it is larger than bytecode caching
alone usually explains, and if a fixture is doing expensive work that is
incidentally cached across runs, that affects how benchmark numbers should be
read.

### Warnings (6, all intentional)

All six come from tests deliberately exercising warning paths; none indicate a
problem:
- 4 × `tests/test_aggregate.py` — `aggregate.py:1010`, "n_peak + n_offpeak = 2 exceeds the 1 available bins"
- 1 × `tests/test_aggregate.py:316` — `assign_segment_speeds`, "a regime has no observed edges"
- 1 × `tests/test_mapping.py:63` — same `assign_segment_speeds` warning

No `DeprecationWarning` from pandas 3.0 or numpy 2.5, which is a meaningful
signal given how new those majors are.

## Lint baseline

```
$ .venv/bin/ruff check src tests scripts examples     # the exact CI invocation
All checks passed!      (exit 0)

$ .venv/bin/ruff check benchmarks                      # NOT linted by CI
All checks passed!      (exit 0)
```

`benchmarks/` is currently clean even though CI never checks it. So the lint gap
flagged for Task 9 is a **latent** risk, not an active failure — widening the CI
invocation would not turn CI red today.

## Build baseline

`dist/` and `build/` were removed and rebuilt from scratch (both are gitignored;
the previous artifacts dated 2026-08-16 were discarded so no later pass is
misled by a stale wheel).

```
$ .venv/bin/python -m build
Successfully built redlight-0.6.0.tar.gz and redlight-0.6.0-py3-none-any.whl

$ .venv/bin/twine check dist/*
Checking dist/redlight-0.6.0-py3-none-any.whl: PASSED
Checking dist/redlight-0.6.0.tar.gz: PASSED
```

| Artifact | Size |
|---|---|
| `dist/redlight-0.6.0.tar.gz` | 202,452 bytes |
| `dist/redlight-0.6.0-py3-none-any.whl` | 98,855 bytes |

### Contents spot-check

- **sdist carries the test suite**: 22 entries under `tests/`, including
  `tests/conftest.py`. This is what `MANIFEST.in` exists to guarantee, and it is
  working.
- **sdist carries `LICENSE`.**
- **No `.DS_Store`** in either artifact, despite one existing in the repo root
  (gitignored at `.gitignore:4`).
- **Wheel is clean**: only `redlight/` plus `redlight-0.6.0.dist-info/`
  (`LICENSE`, `METADATA`, `WHEEL`, `top_level.txt`, `RECORD`). No tests, no
  examples, no docs.

Task 10 still owns the full packaging audit, including the clean-environment
install of both artifacts from outside the repo — that was **not** done here.

## Machine

| | |
|---|---|
| CPU | Intel(R) Core(TM) i5-7360U @ 2.30GHz |
| Logical CPUs | 4 (2 physical cores, hyperthreaded) |
| Platform | darwin, macOS (Darwin 22.6.0) |

**This matters for Task 7 and Task 8.** Four logical cores on a 2017 dual-core
mobile CPU is modest hardware. Two consequences:
1. The `HMMMatcher` docstring claim that serial beats `n_jobs>1` up to two
   million points will be *easier* to satisfy here than on a many-core machine,
   so confirming it here does not confirm it generally. Task 7 must say so.
2. Absolute throughput numbers from this machine are a floor, not a
   representative figure. Record them as relative.

## Repo state at baseline

```
$ git rev-parse --short HEAD
e30b667

$ git status --porcelain
?? .plans/2026-08-17-redlight-ship-review.md
```

Clean apart from the review plan itself, which is committed alongside this file.

## Corrections to prior beliefs

Recorded here so no later pass wastes effort re-deriving them:

- **`interval_id` cross-call collision: CLOSED.** Older notes describe
  `aggregate_speeds(dedup_intervals=True)` silently dropping rows when two
  `derive_speeds` outputs (each numbering from 0) are concatenated. There is now
  a guard, `_INTERVAL_IDENTITY` at `src/redlight/aggregate.py:187`, which raises
  instead. Task 3 verifies the guard actually fires rather than assuming it.
- **`speed_var` computed but unused: CLOSED.** It is now consumed by the
  `weight_by_variance` option at `src/redlight/aggregate.py:314`. Task 4 audits
  whether the weighting is the correct estimator.
- **Test-count drift in docs: NOT REPRODUCED.** A grep for test-count claims in
  `README.md` and `docs/*.md` found no numeric claim to have drifted.
- **Package rename residue: NOT REPRODUCED.** `grep -rln "roadtraffic"` over
  `src tests docs examples scripts benchmarks README.md mkdocs.yml` returns
  nothing. Task 2 still checks the subtler forms (URLs, figure contents).

## Verdict

**Baseline established. Proceed to Task 1.**

The three gates every later pass is measured against:

| Gate | Baseline |
|---|---|
| `.venv/bin/pytest -q` | **375 passed, 0 skipped** |
| `.venv/bin/ruff check src tests scripts examples` | **clean** |
| `.venv/bin/python -m build && twine check dist/*` | **both PASSED** |
