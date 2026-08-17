# redlight Ship Review Plan

> **For agentic workers:** This is a *review* plan, not an implementation plan.
> Each task is one audit pass run in its own fresh session. Steps use checkbox
> (`- [ ]`) syntax for tracking. Do not run passes in parallel — later passes
> consume earlier findings.

**Goal:** Audit every part of the `redlight` package — numerical accuracy, drift
from the approved specs and published docs, correctness under bad input, test
quality, HMM matching performance, API coherence, and release mechanics — so
v0.6.0 can be tagged and published as a GitHub release with known, recorded
confidence rather than hope.

**Architecture:** Ten sequential audit passes. Each pass is a self-contained
prompt you paste into a fresh session; it reads a defined slice of the package,
proves each claim before writing it down, and emits one findings file under
`.plans/reviews/2026-08-17-ship/`. You then triage that file by writing a
verdict on each finding, and run the shared Fix Cycle prompt to execute the
accepted ones. Findings are never fixed by the pass that found them — the audit
trail and the churn stay separate, and no review pass can quietly become an
unreviewed refactor.

**Tech Stack:** Python 3.9+ (local dev on 3.12.13), numpy, pandas, scipy,
shapely, networkx; pytest, ruff; mkdocs-material. Optional extras: pyproj
(`crs`), pyogrio (`shapefile`), matplotlib (`mapping`).

---

## Global Constraints

These apply to **every** task below. Every pass prompt implicitly includes them.

- **Interpreter: `python3.12` (3.12.13).** The system `python3` is broken
  (`dyld cache '(null)' not loaded`) and `python3.11` no longer exists on this
  machine. All work happens inside `.venv` created by `python3.12` — see Task 0.
  `.venv/` is already in `.gitignore` (line 6).
- **Test command:** `.venv/bin/pytest -q` from the repo root.
  `[tool.pytest.ini_options] testpaths = ["tests"]`, so `benchmarks/` and
  `scripts/` are never collected.
- **Lint command:** `.venv/bin/ruff check src tests scripts examples` — this is
  the exact invocation CI runs (`.github/workflows/ci.yml`). Note it omits
  `benchmarks/`; that gap is itself a Task 9 finding candidate, do not silently
  "fix" it mid-pass.
- **ruff config:** `line-length = 95`, `select = ["E", "F", "W", "I", "B", "UP"]`,
  `ignore = ["E741"]` (the math uses `V`, `L`, `s` per the source papers),
  `extend-exclude = ["*.ipynb"]`.
- **`requires-python = ">=3.9"`.** Every module starts with
  `from __future__ import annotations` — the codebase uses `X | None`
  annotations, which are syntax errors on 3.9 without it. Any new file must too.
- **No new runtime dependencies.** Core is `numpy>=1.21`, `pandas>=1.3`,
  `scipy>=1.7`, `shapely>=2.0`, `networkx>=2.6` — all pure/lightweight wheels,
  no GDAL, no PROJ. This is a headline promise of the package; a performance
  finding that requires numba, Cython or Rust is written up as a **proposal**,
  never implemented.
- **All internal computation is in m/s.** Convert only at the API boundary using
  `redlight.units.to_mps` / `from_mps`. Never hardcode a conversion factor.
- **Public API is frozen for this cycle.** Everything in `redlight.__all__`
  (`src/redlight/__init__.py`) keeps its current name and signature. Additive
  keyword-only arguments with backward-compatible defaults are permitted; renames,
  removals and positional-argument changes are findings to report, not to apply.
- **Findings directory:** `.plans/reviews/2026-08-17-ship/`. Process docs live
  outside the published mkdocs tree by deliberate convention — commit `0403dc5`
  moved them there. Never write review output into `docs/`.
- **Evidence before assertion.** No pass may record a behavioural finding it has
  not demonstrated by running something. "This looks wrong" is not a finding;
  "this returns 41.7 when the paper's formula gives 38.2, here is the script and
  its output" is.

---

## Severity Scale

Every finding carries exactly one severity. Passes assign it; you may change it
during triage.

| Sev | Meaning |
|-----|---------|
| **S1** | Produces a **silently wrong number**. Wrong speeds, wrong matches, wrong aggregates, wrong statistics. The user cannot tell from the output that it is wrong. |
| **S2** | Crashes, hangs, or raises a misleading error on input a real user would plausibly supply. Loud failure — bad, but visible. |
| **S3** | Drift: docs, docstrings, README, examples or specs describe behaviour the code does not have (in either direction). |
| **S4** | Performance: correct but slower or more memory-hungry than it needs to be. |
| **S5** | Polish: naming, consistency, dead code, test-suite hygiene. |

---

## Findings File Format

Every pass writes exactly one file in this format. No pass invents its own.

```markdown
# <Pass name> — findings
**Pass:** Task N
**Date:** 2026-08-17
**Commit reviewed:** <output of `git rev-parse --short HEAD`>
**Scope:** <the exact files this pass read>
**Method:** <what was actually run to produce evidence>

## Summary
<Three sentences max. How healthy is this slice? What is the single worst thing?>

---

### F-N.1 — <one-line title>
- **Severity:** S1
- **Location:** `src/redlight/speeds.py:364`
- **Claim:** <one sentence: what is wrong>
- **Evidence:**
  ```
  <the command that was run and its verbatim output, or the failing script>
  ```
- **Expected vs actual:** <what it should be, per what authority, vs what it is>
- **Suggested fix:** <one or two sentences; not a patch>
- **Verdict:** <LEAVE BLANK — the human fills this in>
- **Outcome:** <LEAVE BLANK — the Fix Cycle fills this in>
```

If a pass finds nothing, it still writes the file with a Summary and the line
`No findings.` A clean pass is a result and must be recorded as one.

---

## The Fix Cycle

After triaging any findings file, paste this prompt into a fresh session,
substituting the real path for `<FINDINGS>`. It is the same prompt every time.

```text
Read <FINDINGS>. Each finding has a Verdict line I have filled in with
ACCEPT, REJECT, or DEFER.

Fix every ACCEPT finding, worst severity first (S1, then S2, S3, S4, S5).
Work one finding at a time. For each:

1. Write a test that fails because of this finding. Put it in the existing
   test file that covers that module. Run it and paste the failure output.
   If the finding cannot be expressed as a failing test, say so explicitly
   and explain why before continuing.
2. Make the smallest change that fixes it.
3. Run: .venv/bin/pytest -q
   Run: .venv/bin/ruff check src tests scripts examples
   Both must be clean. Paste the tail of each.
4. Commit that one finding alone, with a message naming it, e.g.
   "fix: <title> (F-3.2)".

Rules:
- Do not touch REJECT or DEFER findings.
- Do not refactor, rename, or "improve" anything the finding did not name.
- Do not add a runtime dependency. Core deps are numpy, pandas, scipy,
  shapely, networkx and that list is closed.
- Public API is frozen: nothing in redlight.__all__ changes name or
  signature. New keyword-only args with back-compatible defaults are fine.
- Every new module starts with `from __future__ import annotations`
  (the package supports Python 3.9).
- If a fix needs a design decision I have not made, STOP and ask me.
  Do not guess and do not pick "the obvious default".

When all ACCEPT findings are done, append an Outcome line to each finding in
<FINDINGS> — `fixed (<sha>)`, `skipped (<reason>)`, or `no change needed
(<reason>)` — and commit the updated findings file on its own.

Finally, tell me: what did you change, what did you deliberately not change,
and did anything you found while fixing deserve to be a new finding?
```

---

## How To Run This Plan

The loop, per task:

1. `/clear` (or open a fresh session). Passes must not inherit each other's
   context — a pass that already "knows" the answer stops looking.
2. Paste that task's prompt verbatim.
3. Read the findings file. Write `ACCEPT`, `REJECT` or `DEFER` on every
   **Verdict** line. This is the only step only you can do.
4. `/clear`, paste the Fix Cycle prompt with that findings path.
5. Tick the task's checkboxes here and move on.

Do not batch the triage. Findings from Task 3 change what Task 4 should look at.

---

### Task 0: Reproducible environment and recorded baseline

Nothing else in this plan can run until there is a working interpreter, and no
pass can claim a regression without a number to regress from.

**Files:**
- Create: `.venv/` (gitignored, not committed)
- Create: `.plans/reviews/2026-08-17-ship/00-baseline.md`

**Interfaces:**
- Produces: `.venv/bin/python`, `.venv/bin/pytest`, `.venv/bin/ruff` — every
  later task invokes these exact paths. `00-baseline.md` holds the baseline test
  count, lint state, build state and machine description that Tasks 1–10 cite.

- [x] **Step 1: Create the virtual environment**

```bash
cd /Users/corysouthall/Downloads/roadtraffic-corytest
python3.12 -m venv .venv
.venv/bin/python -V
```

Expected: `Python 3.12.13`. If `python3.12` is not found, stop — do not fall
back to `python3`, it is dyld-broken and will fail confusingly later.

- [x] **Step 2: Install the package with every extra**

```bash
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e ".[dev,mapping,shapefile,crs,docs]"
```

All five extras go in deliberately: the audit must be able to exercise the
pyproj fallback path, the pyogrio readers, the matplotlib renderer and the
mkdocs build. Tests that `importorskip` an extra would otherwise skip silently
and hide findings.

- [x] **Step 3: Record the baseline**

```bash
.venv/bin/pytest -q 2>&1 | tail -5
.venv/bin/ruff check src tests scripts examples
.venv/bin/python -c "import redlight; print(redlight.__version__)"
git rev-parse --short HEAD
git status --porcelain
```

Expected: a passing suite (the exact count is unknown until this runs — record
what it actually is, do not assume the 289 figure from the drop-pyproj plan or
the 167 from earlier notes, both are stale), clean ruff, `0.6.0`, `e30b667`,
empty status.

- [x] **Step 4: Verify the skip list is empty-ish and understood**

```bash
.venv/bin/pytest -q -rs 2>&1 | tail -20
```

Write down every skipped test and why. With all extras installed, a test that
still skips is either platform-gated or dead. Both are findings for Task 6.

- [x] **Step 5: Verify the distributions build**

```bash
.venv/bin/pip install build twine
.venv/bin/python -m build
.venv/bin/twine check dist/*
```

Expected: an sdist and a wheel in `dist/`, both PASSED. Note `dist/` already
contains artifacts from a previous build — check the new filenames carry
`0.6.0`, and delete stale ones so Task 10 is not fooled by an old wheel.

- [x] **Step 6: Write the baseline file**

Create `.plans/reviews/2026-08-17-ship/00-baseline.md` recording: interpreter
version and path, the exact `pip install` line used, test count (passed /
skipped / xfailed), the skip list with reasons, ruff result, `redlight.__version__`,
the reviewed commit sha, and the built distribution filenames. Every later pass
cites this file.

- [x] **Step 7: Commit the baseline**

```bash
git add .plans/reviews/2026-08-17-ship/00-baseline.md .plans/2026-08-17-redlight-ship-review.md
git commit -m "docs: ship-review plan and recorded baseline"
```

---

### Task 1: Drift — approved specs vs shipped code

**Files:**
- Read: `.plans/2026-07-31-mover-mode-screening-design.md`,
  `.plans/2026-07-31-mover-mode-screening.md`, `.plans/2026-08-04-drop-pyproj.md`
- Read: `src/redlight/modes.py`, `src/redlight/_geo.py`, `src/redlight/_proj.py`,
  `src/redlight/network.py`
- Create: `.plans/reviews/2026-08-17-ship/01-spec-drift.md`

**Interfaces:**
- Consumes: `00-baseline.md` (commit sha, test count).
- Produces: `01-spec-drift.md`. Task 2 reads its Summary so it does not
  re-report the same doc gaps.

- [ ] **Step 1: Run the pass**

```text
You are auditing the Python package `redlight` at
/Users/corysouthall/Downloads/roadtraffic-corytest for drift between what was
designed and what was built. Read .plans/reviews/2026-08-17-ship/00-baseline.md
first for the environment and baseline.

Three approved specs were implemented in this repo:
  .plans/2026-07-31-mover-mode-screening-design.md  (the design)
  .plans/2026-07-31-mover-mode-screening.md         (the task plan, v0.5.0)
  .plans/2026-08-04-drop-pyproj.md                  (the task plan)

For EACH spec, go requirement by requirement and determine one of:
  IMPLEMENTED AS SPECIFIED / IMPLEMENTED DIFFERENTLY / NOT IMPLEMENTED /
  IMPLEMENTED BUT LATER CHANGED

Do not trust the checkboxes in those files. Verify against the code as it
stands today, and against `git log --oneline` — note that the package was
renamed roadtraffic -> redlight in commit 189a505, so spec text referring to
`roadtraffic.*` module paths is expected and is not itself drift.

Pay specific attention to these binding constraints the specs set, and check
each one still holds:
- drop-pyproj: `grep -rn "pyproj" src/` must hit ONLY function-local deferred
  imports in network.py. No module-level `import pyproj` anywhere in src/.
- drop-pyproj: Network.crs_metric.to_epsg(), Network.project_points(),
  net._transformer_fwd.transform(), net._transformer_inv.transform() must all
  still work unchanged — they are used from tests, scripts and examples.
- mover-mode-screening: classification is per-trajectory and the verdict
  applies to ALL of that mover's observations including its slowest. Verify
  by running it, not by reading it.
- mover-mode-screening: modes.py must depend only on units + scipy.

For anything you flag as IMPLEMENTED DIFFERENTLY, decide and state whether the
difference is an improvement, a regression, or neutral — and whether the spec
or the code is now the thing that is wrong.

Prove every behavioural claim by running code. You have .venv/bin/python.
Write scratch scripts in
/private/tmp/claude-501/-Users-corysouthall-Downloads-roadtraffic-corytest/c8494443-225d-45a5-b2bf-a37253d05b4e/scratchpad
never in the repo.

Write your findings to .plans/reviews/2026-08-17-ship/01-spec-drift.md using
the exact format defined in .plans/2026-08-17-redlight-ship-review.md under
"Findings File Format", with the severity scale defined there. Leave every
Verdict and Outcome line blank. Change no source file. Do not commit.
```

- [ ] **Step 2: Triage** — fill in every Verdict line in `01-spec-drift.md`.

- [ ] **Step 3: Run the Fix Cycle** with `<FINDINGS>` =
`.plans/reviews/2026-08-17-ship/01-spec-drift.md`.

- [ ] **Step 4: Commit the findings file**

```bash
git add .plans/reviews/2026-08-17-ship/01-spec-drift.md
git commit -m "docs: spec-drift findings"
```

---

### Task 2: Drift — docs, docstrings and examples vs behaviour

`tests/test_docs.py` already executes the Python snippets embedded in the docs,
so syntax-level drift is covered. This pass hunts what that cannot catch: prose
that claims a behaviour, a bound, or a number the code does not deliver.

**Files:**
- Read: `README.md`, `docs/index.md`, `docs/quickstart.md`, `docs/statistics.md`,
  `docs/methodology.md`, `docs/api.md`, `examples/**`, every public docstring in
  `src/redlight/`
- Read: `tests/test_docs.py` (to know what is already covered)
- Create: `.plans/reviews/2026-08-17-ship/02-doc-drift.md`

**Interfaces:**
- Consumes: `00-baseline.md`, `01-spec-drift.md` Summary.
- Produces: `02-doc-drift.md`.

- [ ] **Step 1: Run the pass**

```text
You are auditing the Python package `redlight` at
/Users/corysouthall/Downloads/roadtraffic-corytest for drift between its
documentation and its actual behaviour. Read
.plans/reviews/2026-08-17-ship/00-baseline.md for the environment, and the
Summary of .plans/reviews/2026-08-17-ship/01-spec-drift.md so you do not
re-report findings already recorded there.

Scope: README.md, docs/index.md, docs/quickstart.md, docs/statistics.md,
docs/methodology.md, docs/api.md, everything under examples/, and every
docstring of every symbol in redlight.__all__ (see src/redlight/__init__.py).

tests/test_docs.py already executes doc code snippets, so do not spend effort
re-checking that snippets run. Read it first to see exactly what it covers.

What you ARE looking for:
1. PROSE CLAIMS THE CODE DOES NOT SUPPORT. Every sentence that asserts a
   behaviour, a default, a bound, a units convention, an error condition, or a
   numeric result is a testable claim. Extract them and test the ones that
   matter. The module docstring in src/redlight/__init__.py is a dense list of
   such claims — check each line against the function it describes.
2. DOCUMENTED DEFAULTS THAT DISAGREE WITH SIGNATURES. Compare every documented
   parameter default against the actual signature via
   `inspect.signature`. HMMMatcher's docstring (src/redlight/matching.py:125)
   documents sigma_z, beta, max_dist, k, max_route_dist_factor, n_jobs and
   dist_cache_size — verify each stated default is the real one.
3. PERFORMANCE AND SCALE CLAIMS. The HMMMatcher n_jobs docstring asserts "the
   serial path already sustains tens of thousands of points per second" and
   "on the development machine serial matching stayed faster up to at least two
   million points". Flag these as UNVERIFIED here and hand them to Task 7 —
   do not benchmark in this pass, just record that the claim exists and where.
4. NUMBERS IN docs/methodology.md. It cites experiment results (sections 2.3,
   3.4, 4.4) that scripts/paper_experiments.py is supposed to reproduce.
   Do NOT re-run the experiments here — that is Task 4. Instead, list every
   numeric claim in the prose with its section and line, so Task 4 has a
   checklist to reconcile against.
5. EXAMPLES. Run every script under examples/ (see examples/README.md and
   examples/_common.py). Record any that error, produce output contradicting
   their own narration, or use API that no longer exists.
6. STALE COUNTS AND VERSIONS. Any "N tests", "version X", or supported-Python
   claim in prose, checked against reality.
7. THE RENAME. Commit 189a505 renamed roadtraffic -> redlight. A grep for
   "roadtraffic" across the repo currently returns nothing outside .plans/ —
   confirm that, and also check for the subtler residue: stale GitHub URLs,
   old package names in example output, old names in figures under
   docs/figures/.

Prove behavioural claims by running code with .venv/bin/python. Scratch files
go in /private/tmp/claude-501/-Users-corysouthall-Downloads-roadtraffic-corytest/c8494443-225d-45a5-b2bf-a37253d05b4e/scratchpad,
never in the repo.

For each finding state clearly which side is wrong — the doc or the code.
Usually it is the doc; when it is the code, that is a much more serious finding
and should be severity S1 or S2, not S3.

Write findings to .plans/reviews/2026-08-17-ship/02-doc-drift.md in the exact
format from .plans/2026-08-17-redlight-ship-review.md. Leave Verdict and
Outcome blank. Change no source file. Do not commit.
```

- [ ] **Step 2: Triage** — fill in every Verdict line.

- [ ] **Step 3: Run the Fix Cycle** with `<FINDINGS>` =
`.plans/reviews/2026-08-17-ship/02-doc-drift.md`.

- [ ] **Step 4: Commit the findings file**

```bash
git add .plans/reviews/2026-08-17-ship/02-doc-drift.md
git commit -m "docs: documentation-drift findings"
```

---

### Task 3: Accuracy — the numerical core

The geodesy and the speed derivation are where a silent error costs the most:
every downstream number inherits it, and nothing in the output looks wrong.

**Files:**
- Read: `src/redlight/_geo.py`, `src/redlight/_proj.py`, `src/redlight/units.py`,
  `src/redlight/speeds.py`, `src/redlight/matching.py`
- Read: `tests/test_geo.py`, `tests/test_proj.py`, `tests/test_derive_math.py`,
  `tests/test_no_pyproj.py`
- Create: `.plans/reviews/2026-08-17-ship/03-numerical-accuracy.md`

**Interfaces:**
- Consumes: `00-baseline.md`.
- Produces: `03-numerical-accuracy.md`. Task 4 assumes the primitives audited
  here are trustworthy, so this pass must run before it.

- [ ] **Step 1: Run the pass**

```text
You are auditing the numerical core of the Python package `redlight` at
/Users/corysouthall/Downloads/roadtraffic-corytest. Read
.plans/reviews/2026-08-17-ship/00-baseline.md for the environment.

This package computes geodesy and projections in pure numpy specifically to
avoid depending on PROJ. That decision means the correctness of these formulas
is the package's own responsibility. Audit them against external ground truth,
not against the package's own tests.

SCOPE AND AUTHORITY FOR EACH PIECE:

1. src/redlight/_geo.py — Vincenty inverse geodesic distance on WGS84.
   Check against: closed-form expectations and, where available, pyproj.Geod
   (installed in .venv as the `crs` extra — use it as an oracle here even
   though the package must not depend on it at runtime).
   Specifically test: short distances (<1 m), antipodal and near-antipodal
   pairs where Vincenty is known not to converge, equatorial pairs, polar
   pairs, identical points, and points spanning the antimeridian. State what
   the code does on non-convergence and whether that behaviour is safe.

2. src/redlight/_proj.py — Krüger series WGS84<->UTM, plus Web Mercator.
   Check against: pyproj Transformer as an oracle across all 60 northern and
   60 southern UTM zones, at several latitudes each, including near zone
   edges and the zone-width exceptions around Norway/Svalbard if the code
   claims to handle them. Report the max round-trip error in metres and the
   max error vs pyproj in metres, and say plainly whether those magnitudes
   matter for a package whose stated GPS error regime is 1-100 m.
   Also check: what happens outside the valid latitude band for UTM, and
   whether the UtmCrs/UtmTransformer shims report .to_epsg() correctly for
   southern-hemisphere zones (the 327xx vs 326xx distinction).

3. src/redlight/units.py — to_mps / from_mps. Verify exact conversion
   constants (mph, kph, m/s) and round-trip identity to floating-point
   tolerance. This is small but a wrong constant here is invisible and
   corrupts everything.

4. src/redlight/speeds.py — derive_speeds. This is the mathematically load-
   bearing function. Verify the three-piece on-road hop distance
   (`_hop_distance`): remaining length of the previous edge past its snap, the
   node-to-node middle, the current edge's length up to its snap. Construct a
   synthetic network where you know the true on-road distance analytically and
   confirm the computed distance matches. Then verify:
   - the interval midpoint convention for `time` (docstring, speeds.py:37)
   - speed_sigma_mps and speed_var: derive the error model by hand from the
     docstring's stated assumptions and confirm the code implements THAT model
   - the `quality` flag's exact predicate, and that it is advisory not a filter
   - interval_id numbering and the interval_id_start parameter (speeds.py:215),
     including that aggregate.py's _INTERVAL_IDENTITY collision guard
     (aggregate.py:187) actually fires when two derive_speeds outputs that both
     started at 0 are concatenated. Write that test and run it.
   - zero and negative dt, duplicate timestamps, single-point trajectories

5. src/redlight/matching.py — the HMM emission and transition probabilities
   against Newson & Krumm (2009). Verify `_emission_logp` is the log of a
   zero-mean Gaussian in perpendicular snap distance with std sigma_z,
   including the normalising constant, and `_transition_logp` is the log of an
   exponential in |great-circle step - route distance| with mean beta. Then
   scrutinise the three non-paper behaviours the code adds, and for each decide
   whether it is a principled extension or a bug:
     a. the same-edge transition shortcut that sets route distance = gc_step
        (matching.py:296-303)
     b. the saturating penalty when no predecessor is reachable within cutoff
        (matching.py:320-332)
     c. carrying state across candidate-less fixes with the `anchor` array
        (matching.py:256-270)
   For each, construct a trajectory that exercises it and show what the decoded
   path is, and what it would be without that behaviour.

Do NOT evaluate performance in this pass. Do NOT change any source file.
Write scratch scripts to
/private/tmp/claude-501/-Users-corysouthall-Downloads-roadtraffic-corytest/c8494443-225d-45a5-b2bf-a37253d05b4e/scratchpad.

Every finding needs a reproducible script and its verbatim output as evidence.
If you cannot demonstrate it, do not write it down — instead list it under a
final "## Unverified suspicions" heading, clearly separated from the findings.

Write to .plans/reviews/2026-08-17-ship/03-numerical-accuracy.md in the exact
format from .plans/2026-08-17-redlight-ship-review.md. Leave Verdict and
Outcome blank. Do not commit.
```

- [ ] **Step 2: Triage** — fill in every Verdict line. Expect this file to be
the most consequential in the plan; read it slowly.

- [ ] **Step 3: Run the Fix Cycle** with `<FINDINGS>` =
`.plans/reviews/2026-08-17-ship/03-numerical-accuracy.md`.

- [ ] **Step 4: Commit the findings file**

```bash
git add .plans/reviews/2026-08-17-ship/03-numerical-accuracy.md
git commit -m "docs: numerical-accuracy findings"
```

---

### Task 4: Accuracy — statistical claims and the published experiments

`scripts/paper_experiments.py` exists to reproduce every figure and number in
`docs/methodology.md`. Either it still does, or the paper is wrong. This pass
settles that, and audits the statistics the aggregation layer performs.

**Files:**
- Read: `scripts/paper_experiments.py`, `docs/methodology.md`,
  `docs/statistics.md`, `docs/figures/experiment_results.json`
- Read: `src/redlight/aggregate.py`, `src/redlight/modes.py`,
  `src/redlight/cleaning.py`
- Read: `tests/test_aggregate.py`, `tests/test_modes.py`, `tests/test_cleaning.py`
- Create: `.plans/reviews/2026-08-17-ship/04-statistical-accuracy.md`

**Interfaces:**
- Consumes: `03-numerical-accuracy.md` (do not re-audit primitives it cleared),
  `02-doc-drift.md` (the checklist of numeric claims in methodology.md).
- Produces: `04-statistical-accuracy.md`.

- [ ] **Step 1: Run the pass**

```text
You are auditing the statistical layer of the Python package `redlight` at
/Users/corysouthall/Downloads/roadtraffic-corytest, and verifying that its
published empirical results still reproduce.

Read first: .plans/reviews/2026-08-17-ship/00-baseline.md for the environment,
.plans/reviews/2026-08-17-ship/03-numerical-accuracy.md for what has already
been established about the numerical primitives (do not re-audit those), and
the numeric-claims checklist in
.plans/reviews/2026-08-17-ship/02-doc-drift.md.

PART A — REPRODUCE THE PUBLISHED EXPERIMENTS.
scripts/paper_experiments.py claims to reproduce every figure and number in
docs/methodology.md. Run it with .venv/bin/python. Note it writes into
docs/figures/ — copy docs/figures/experiment_results.json to the scratchpad
FIRST so you can diff against it, and `git checkout -- docs/figures/` at the
end so the pass leaves no modified files.

Then reconcile, one by one:
- Experiment A (matching accuracy, methodology.md section 2.3) — does the
  NearestMatcher-vs-HMMMatcher accuracy comparison still produce the numbers
  the prose quotes?
- Experiment B (speed derivation error, section 3.4)
- Experiment C (end-to-end validation, section 4.4)
Report every number that has moved, by how much, and whether the prose's
qualitative claim still holds even where the digits differ. Check the script
seeds its RNG — if any experiment is not deterministic, that is itself a
finding, because an unseeded published result cannot be reproduced by a reader.

PART B — AUDIT THE AGGREGATION STATISTICS in src/redlight/aggregate.py.
- aggregate_speeds: mean vs median paths; the dedup_intervals default and the
  _INTERVAL_IDENTITY collision guard; N-hour block boundary arithmetic (is the
  last block inclusive or exclusive, and does 24 % block_size != 0 behave
  sanely?); what an empty group returns (n=0 and NaN, or a dropped row?).
- weight_by_variance: verify the inverse-variance weighted mean is the correct
  estimator as documented (aggregate.py:314-368), that dropped non-finite or
  non-positive variance rows warn, and that the weighted result reduces to the
  unweighted one when all variances are equal.
- days= filtering and day_type_report: weekday/weekend presets, day names and
  numbers with Mon=0..Sun=6, None meaning all. Critically: what timezone are
  the day-of-week and hour-of-day derived in? If the input timestamps are
  tz-naive, whose local time is being assumed, and is that documented? A
  timezone error here silently shifts every peak-hour result and is S1.
- peak_analysis and classify_hours: the peak-detection rule, ties, and the
  degenerate cases of one hour of data and of perfectly flat speeds.
- congestion_report: observed-over-posted ratio; missing or zero speed limits;
  units at the boundary.

PART C — AUDIT src/redlight/modes.py.
- suggest_mode_threshold uses scipy gaussian_kde to find a density valley.
  Verify it returns None when no walking population exists (commit 367e683
  made this "require a real walking hump" — check the predicate is principled
  and not tuned to one dataset). Test on: vehicles only, pedestrians only,
  a clean bimodal mix, a mix with a 5% pedestrian minority, and n<5 movers.
- classify_movers: confirm the verdict is per-trajectory and applies to every
  observation of that mover including its slowest — this is the package's
  central methodological claim (see the design doc rationale in
  .plans/2026-07-31-mover-mode-screening-design.md).
- mover_features: the speed percentile chosen, and its sensitivity.

PART D — AUDIT src/redlight/cleaning.py.
filter_by_speed and filter_trajectory_speed. The stated design intent is that
cleaning must not bias the answer by deleting congestion (methodology.md 4.1).
Construct a trajectory containing genuine congestion and confirm the dwell-aware
filter keeps it while removing an actual GPS outlier.

Do not change any source file. Leave the repo clean — verify with
`git status --porcelain` before you finish and report the result.
Scratch work goes in
/private/tmp/claude-501/-Users-corysouthall-Downloads-roadtraffic-corytest/c8494443-225d-45a5-b2bf-a37253d05b4e/scratchpad.

Write to .plans/reviews/2026-08-17-ship/04-statistical-accuracy.md in the exact
format from .plans/2026-08-17-redlight-ship-review.md. Leave Verdict and
Outcome blank. Do not commit.
```

- [ ] **Step 2: Triage** — fill in every Verdict line.

- [ ] **Step 3: Run the Fix Cycle** with `<FINDINGS>` =
`.plans/reviews/2026-08-17-ship/04-statistical-accuracy.md`.

- [ ] **Step 4: Commit the findings file**

```bash
git add .plans/reviews/2026-08-17-ship/04-statistical-accuracy.md
git commit -m "docs: statistical-accuracy findings"
```

---

### Task 5: Robustness — bad input, edge cases and error messages

A bring-your-own-data package lives or dies on what it does with data it did not
expect. This pass attacks the boundary.

**Files:**
- Read: `src/redlight/points.py`, `src/redlight/network.py`, `src/redlight/osm.py`,
  `src/redlight/routing.py`, `src/redlight/analysis.py`, `src/redlight/mapping.py`
- Create: `.plans/reviews/2026-08-17-ship/05-robustness.md`

**Interfaces:**
- Consumes: `00-baseline.md`.
- Produces: `05-robustness.md`.

- [ ] **Step 1: Run the pass**

```text
You are stress-testing the input boundary of the Python package `redlight` at
/Users/corysouthall/Downloads/roadtraffic-corytest. Read
.plans/reviews/2026-08-17-ship/00-baseline.md for the environment.

This package's pitch is "bring your own data", so its behaviour on unexpected
input is a feature, not an afterthought. For each case below, actually run it
and record what happens. Classify each outcome as:
  GOOD    — correct result, or a clear error naming the problem and the fix
  LOUD    — it raises, but the message would not help a real user (S2)
  SILENT  — it returns a plausible-looking wrong answer (S1, the worst kind)

POINTS (src/redlight/points.py — load_points, save_points, PointSet):
- empty file; header-only file; one row; duplicate timestamps within a mover
- unsorted timestamps; tz-aware vs tz-naive timestamps; mixed tz in one file
- lat/lon swapped (a very common real error — does anything catch it?)
- out-of-range coordinates (lat 91, lon 181); NaN and null coordinates
- a trajectory id column with NaN, with mixed int/str types
- derive_speed=True on a single-point trajectory
- a file whose speed column is in mph while speed_unit says m/s

NETWORK (src/redlight/network.py, osm.py):
- a GeoJSON with zero features; with only Points (no LineStrings)
- MultiLineString geometry; a LineString with 1 coordinate; zero-length edges
- self-loop edges; duplicate edges between the same node pair
- a network spanning two UTM zones, and one spanning the antimeridian
- missing/blank `highway` tags; missing CRS; a CRS the package cannot project
  without pyproj (confirm the error names the `crs` extra)
- from_overpass: do NOT hit the network. Read the code and report how it
  behaves on HTTP error, empty response and malformed JSON, and whether it
  has a timeout at all.

MATCHING AND SPEEDS at the boundary:
- HMMMatcher on points with no trajectory column (should raise a clear error —
  matching.py:224 claims it does; verify the message names the fix)
- a trajectory entirely outside max_dist of any edge
- a network with one edge; a network of two disconnected components with a
  trajectory that jumps between them

ROUTING (src/redlight/routing.py):
- unreachable origin/destination pair; origin == destination
- a node id that does not exist; a one-way trap
- routing by time when some edges carry no speed (None/NaN weights)

ANALYSIS (src/redlight/analysis.py):
- edge_betweenness_centrality with None/NaN edge weights
- network_stats on a single-edge network and on a disconnected one; the
  automatic study-area detection added in commit 7e14cd5 — what does it do
  with a degenerate or antimeridian-spanning bounding box?
- connectivity_report on a fully connected and on a fully disconnected network

MAPPING (src/redlight/mapping.py):
- to_geojson with no speeds assigned; with NaN speeds; with non-JSON-able
  attribute values (there is a _jsonable helper at mapping.py:48 — test it)
- plot_speed_map without matplotlib installed (temporarily hide it and confirm
  the error names the `mapping` extra)

For every SILENT case, that is the finding — write it up with the exact input
that triggers it. For LOUD cases, quote the actual message and propose the
message that would have helped.

Do not change any source file. Scratch data and scripts go in
/private/tmp/claude-501/-Users-corysouthall-Downloads-roadtraffic-corytest/c8494443-225d-45a5-b2bf-a37253d05b4e/scratchpad.
Leave the repo clean; confirm with `git status --porcelain`.

Write to .plans/reviews/2026-08-17-ship/05-robustness.md in the exact format
from .plans/2026-08-17-redlight-ship-review.md. Leave Verdict and Outcome
blank. Do not commit.
```

- [ ] **Step 2: Triage** — fill in every Verdict line.

- [ ] **Step 3: Run the Fix Cycle** with `<FINDINGS>` =
`.plans/reviews/2026-08-17-ship/05-robustness.md`.

- [ ] **Step 4: Commit the findings file**

```bash
git add .plans/reviews/2026-08-17-ship/05-robustness.md
git commit -m "docs: robustness findings"
```

---

### Task 6: Test-suite quality audit

A green suite proves nothing if the tests do not actually pin behaviour. This
pass audits the auditor.

**Files:**
- Read: all 21 files under `tests/`, `tests/conftest.py`
- Create: `.plans/reviews/2026-08-17-ship/06-test-quality.md`

**Interfaces:**
- Consumes: `00-baseline.md` (the skip list), and every prior findings file.
- Produces: `06-test-quality.md`.

- [ ] **Step 1: Run the pass**

```text
You are auditing the TEST SUITE of the Python package `redlight` at
/Users/corysouthall/Downloads/roadtraffic-corytest — not the source. Read
.plans/reviews/2026-08-17-ship/00-baseline.md for the baseline count and skip
list.

The suite is ~4,300 lines across 21 files and it passes. The question this pass
answers is: what could break without it noticing?

1. MUTATION SPOT-CHECK. This is the core of the pass. Pick 12 load-bearing
   lines across src/redlight/ — at minimum one each in _geo.py, _proj.py,
   speeds.py (the hop-distance sum), matching.py (the emission and the
   transition log-prob), aggregate.py (the weighted mean, the block
   boundary), modes.py (the valley predicate), cleaning.py, routing.py,
   analysis.py. Mutate each one plausibly and one at a time: flip a
   comparison operator, change a + to a -, drop a term, off-by-one an index,
   swap two arguments. After each mutation run `.venv/bin/pytest -q` and
   record whether anything failed and which test.
   ALWAYS `git checkout -- src/` immediately after each measurement. Never
   leave a mutation in place, and verify the tree is clean between mutations.
   Every mutation that the suite does NOT catch is a finding: name the line
   and the specific test that should have caught it and does not.

2. ASSERTION QUALITY. Find tests that call a function and assert only that it
   did not raise, or assert only on shape/length/dtype rather than value.
   Those are smoke tests wearing a unit test's name. List them.

3. TOLERANCE INFLATION. Find numeric assertions whose tolerance is loose
   enough to pass under a real regression — an `abs=1.0` on a metre-scale
   quantity, an rtol wide enough to swallow a 10% error. Quote each and say
   what tolerance the underlying maths actually justifies.

4. SKIPS. For every test still skipping with all extras installed (see the
   baseline skip list), determine whether it is legitimately platform-gated
   or is dead code that will never run again.

5. SHARED-STATE AND ORDER DEPENDENCE. Run the suite in a different order and
   confirm it still passes:
     .venv/bin/pytest -q -p no:randomly
     .venv/bin/pytest -q tests/test_aggregate.py tests/test_geo.py
   Then run each test file individually in a loop and report any that fail
   alone. Check conftest.py fixtures for mutable module-scoped state.

6. COVERAGE OF THE THINGS THAT MATTER. Do not chase a coverage percentage.
   Instead, list the public API symbols (redlight.__all__) that have no test
   asserting their documented behaviour, and any error branch raising a
   custom message that no test exercises.

7. DETERMINISM. Find every RNG use in tests and confirm it is seeded.

Do not fix anything. Do not leave any mutation in the tree — the final action
of this pass is `git status --porcelain` and it must print nothing. Report
that output.

Write to .plans/reviews/2026-08-17-ship/06-test-quality.md in the exact format
from .plans/2026-08-17-redlight-ship-review.md, with the mutation results as a
table (line mutated / mutation / caught? / by which test). Leave Verdict and
Outcome blank. Do not commit.
```

- [ ] **Step 2: Triage** — fill in every Verdict line.

- [ ] **Step 3: Run the Fix Cycle** with `<FINDINGS>` =
`.plans/reviews/2026-08-17-ship/06-test-quality.md`.

- [ ] **Step 4: Commit the findings file**

```bash
git add .plans/reviews/2026-08-17-ship/06-test-quality.md
git commit -m "docs: test-quality findings"
```

---

### Task 7: Performance — measure before touching anything

No optimization is written in this task. The only deliverable is trustworthy
numbers, because the docstring performance claims are currently unverified and
the obvious hot spots may not be the real ones.

**Files:**
- Read: `benchmarks/bench_matching.py`, `src/redlight/matching.py`,
  `src/redlight/network.py`
- Create: `.plans/reviews/2026-08-17-ship/07-performance-baseline.md`
- Create: `benchmarks/profile_hmm.py`

**Interfaces:**
- Consumes: `00-baseline.md`, and the unverified performance claims recorded by
  Task 2.
- Produces: `07-performance-baseline.md` with a profile table, and
  `benchmarks/profile_hmm.py` — the reusable harness Task 8 measures against.

- [ ] **Step 1: Run the pass**

```text
You are establishing a performance baseline for the map matcher in the Python
package `redlight` at /Users/corysouthall/Downloads/roadtraffic-corytest.
Read .plans/reviews/2026-08-17-ship/00-baseline.md for the environment.

WRITE NO OPTIMIZATIONS IN THIS PASS. The deliverable is measurement.

1. RUN THE EXISTING BENCHMARK.
   benchmarks/bench_matching.py builds a synthetic grid network and times
   NearestMatcher and HMMMatcher, serial and parallel. Read it, then run it at
   several sizes, e.g.
     .venv/bin/python benchmarks/bench_matching.py --points 20000 --trajectories 50
     .venv/bin/python benchmarks/bench_matching.py --points 200000 --trajectories 200
   Report points/second for each matcher at each size, and how throughput
   scales with network size, trajectory count and point density.

2. SETTLE THE DOCSTRING CLAIMS. The HMMMatcher docstring
   (src/redlight/matching.py:154-167) asserts that the serial path "sustains
   tens of thousands of points per second" and that "serial matching stayed
   faster [than n_jobs>1] up to at least two million points". Measure both on
   this machine. If either is false here, that is an S3 finding against the
   docstring — record the numbers that would make it true.

3. PROFILE THE REAL HOT PATH. Write benchmarks/profile_hmm.py: a cProfile
   run of HMMMatcher.match over a realistic workload, printing cumulative and
   total time by function, plus a line-level breakdown of _match_one. Follow
   the style and conventions of benchmarks/bench_matching.py (module docstring
   explaining how to run it, `from __future__ import annotations`, argparse,
   ruff-clean at line-length 95). It must not be collected by pytest —
   testpaths is ["tests"], so benchmarks/ is already excluded.
   Report the top 15 functions by cumulative time.

4. ANSWER THESE SPECIFIC QUESTIONS WITH MEASUREMENTS, not reasoning:
   - What fraction of total time is spent inside _CSRDistCache.lookup
     (matching.py:99) — i.e. in bounded Dijkstra — versus in the Viterbi
     bookkeeping around it?
   - What is the cache hit rate of _CSRDistCache during a realistic run?
     Instrument it temporarily to count hits and misses. Does the maxsize
     default of 10,000 bind? Does raising it help?
   - How many times per point are network.edge_endpoints() and
     network.edge_length() called from inside the inner loop
     (matching.py:289-315), and what do those calls cost? They are invoked
     per (candidate, predecessor) pair.
   - What is the average and worst-case number of Viterbi states carried per
     step (len(V[i-1])), and how does it vary with k and max_dist?
   - How much time goes to Network.candidate_edges_batch versus the Viterbi
     itself?
   - Where does memory go on a large run? Measure peak RSS with
     tracemalloc or resource.getrusage.

5. Also profile ONE non-matching path for comparison so effort is not
   misallocated: time derive_speeds and aggregate_speeds on the same
   workload. If matching is 5% of a real pipeline's wall time, that is the
   single most important finding in this pass.

Record every number with the command that produced it and the machine's CPU
count. Numbers are relative to this hardware — say so.

Findings in this pass are mostly S4, plus any S3 against the docstring claims.
Do not change src/. The only file you create is benchmarks/profile_hmm.py.

Write to .plans/reviews/2026-08-17-ship/07-performance-baseline.md in the exact
format from .plans/2026-08-17-redlight-ship-review.md, with a profile table.
Leave Verdict and Outcome blank. Do not commit.
```

- [ ] **Step 2: Triage** — fill in every Verdict line. The decision you are
making here is *where Task 8 is allowed to spend effort*.

- [ ] **Step 3: Commit the harness and the baseline**

```bash
.venv/bin/ruff check src tests scripts examples benchmarks
git add benchmarks/profile_hmm.py .plans/reviews/2026-08-17-ship/07-performance-baseline.md
git commit -m "bench: HMM profiling harness and recorded performance baseline"
```

There is no Fix Cycle for this task — Task 8 is its fix cycle.

---

### Task 8: Performance — safe wins, and the fast-map-matching proposal

Two separable deliverables: optimizations that keep the dependency promise and
provably preserve output, and a written FMM proposal you decide on later.

**Files:**
- Modify: `src/redlight/matching.py`, possibly `src/redlight/network.py`
- Create: `tests/test_matching_invariance.py`
- Create: `.plans/reviews/2026-08-17-ship/08-performance-work.md`
- Create: `.plans/2026-08-17-fast-map-matching-proposal.md`

**Interfaces:**
- Consumes: `07-performance-baseline.md` — the profile decides what is worked
  on. `benchmarks/profile_hmm.py` is the measuring instrument.
- Produces: an optimized matcher with byte-identical output, and a standalone
  FMM proposal document.

- [ ] **Step 1: Run the pass**

```text
You are optimizing the HMM map matcher in the Python package `redlight` at
/Users/corysouthall/Downloads/roadtraffic-corytest.

Read FIRST, and treat as binding:
  .plans/reviews/2026-08-17-ship/07-performance-baseline.md — the profile.
  .plans/reviews/2026-08-17-ship/00-baseline.md — the environment.
Work ONLY on hot spots that profile actually identified. If the profile shows
a thing is cold, it is out of scope no matter how appealing it looks.

THE HARD CONSTRAINT — OUTPUT INVARIANCE.
Every optimization in Part A must leave HMMMatcher's output bit-for-bit
identical. Before optimizing anything:
  Write tests/test_matching_invariance.py. It generates several seeded
  synthetic networks and trajectories (reuse the generators in
  benchmarks/bench_matching.py or scripts/paper_experiments.py rather than
  writing new ones), runs HMMMatcher over them, and asserts the matched frame
  equals a stored expected frame exactly — edge_id sequences identical, snap
  distances equal to within 1e-12. Also assert NearestMatcher output is
  unchanged. Generate the expected values from the CURRENT code, commit that
  test, and confirm it passes BEFORE you change one line of matching.py.
  This test is what makes the rest of the pass safe. Do not skip it and do
  not weaken it later to make an optimization pass.

PART A — SAFE WINS. Candidates, in the order the profile justifies:
- Hoist the repeated network.edge_endpoints() and network.edge_length() calls
  out of the inner loop (matching.py:289-315). These are called per
  (candidate, predecessor) pair. Precompute per-edge length and endpoint
  arrays once per matcher (or per network) and index them with numpy.
- Replace the per-step dict-of-dicts Viterbi state (V, back, snap_t at
  matching.py:251-253) with flat arrays over a compact candidate index, if
  and only if the profile shows dict overhead is material.
- Vectorize the emission log-prob over all candidates of a step at once
  rather than calling _emission_logp per candidate.
- Improve _CSRDistCache (matching.py:82-123) hit rate: the cutoff varies per
  step, which may be fragmenting the cache. Consider quantizing the cutoff to
  a small set of regimes so lookups reuse entries — but only if the baseline
  measured a low hit rate, and only if it cannot change results. Bounded
  Dijkstra results are only reusable at a cutoff >= the requested one; if
  quantizing rounds a cutoff DOWN it changes output and is forbidden.
- Cheap candidate pruning that is provably lossless.

FORBIDDEN in Part A:
- Any new runtime dependency. numba, Cython, Rust, C extensions are out —
  the package's core promise is lightweight pure wheels. If you measure a
  compelling case for one, write it up in the proposal instead.
- Beam pruning, top-m state truncation, or any approximation. These change
  output. If the profile says beam search is the big win, it belongs in the
  proposal, not here.
- Touching the public API.

For EACH optimization: measure with benchmarks/profile_hmm.py before and
after, run tests/test_matching_invariance.py, run the full suite, run ruff,
and commit that single optimization with its measured speedup in the commit
message. If an optimization yields less than ~5% on a realistic workload,
REVERT IT — complexity that does not pay for itself is a net loss in a
package this readable. Say in your report which ones you reverted and why.

PART B — THE FAST MAP MATCHING PROPOSAL. Write, do not build,
.plans/2026-08-17-fast-map-matching-proposal.md covering:
- What FMM actually is: precomputing an upper-bounded origin-destination
  table (UBODT) of shortest-path distances between node pairs within a
  distance bound, so the per-transition Dijkstra in _CSRDistCache.lookup
  becomes a hash lookup. Cite Yang & Gidofalvi (2018).
- Concretely, what changes in this codebase: which functions, what the UBODT
  build would look like on top of the existing Network.csgraph(), how it
  would be stored and invalidated when the network changes.
- The costs, with real estimates computed from the actual networks in
  examples/sample_data and the benchmark grid: UBODT build time and table
  size as a function of node count and the delta bound, and the memory and
  disk footprint. Say at what network size it stops being practical.
- Whether it can be exact. FMM's delta bound means pairs beyond it are
  absent; the current code's cutoff already bounds the search, so state
  precisely under what conditions a UBODT lookup gives identical results to
  the current bounded Dijkstra and when it would not.
- The honest verdict: given the measured baseline, what speedup would this
  actually buy on a realistic pipeline, and is it worth the complexity for a
  package whose selling point is that it is small and readable? A
  recommendation of "not worth it" is a perfectly good outcome — say so
  plainly if that is what the numbers show.
- Alternatives compared on the same terms: precomputed contraction
  hierarchies, A* with a geometric heuristic instead of Dijkstra, caching
  more aggressively, and simply documenting NearestMatcher as the fast path.

Write your work log to .plans/reviews/2026-08-17-ship/08-performance-work.md:
what was optimized, measured before/after for each, what was reverted, and
what remains on the table. Use the standard findings format from
.plans/2026-08-17-redlight-ship-review.md for anything still outstanding.

Commit Part A optimizations individually as you go. Commit the proposal and
the work log separately at the end.
```

- [ ] **Step 2: Review the diff yourself**

```bash
git log --oneline -15
git diff <baseline-sha>..HEAD -- src/redlight/matching.py
```

- [ ] **Step 3: Decide on the FMM proposal.** Read
`.plans/2026-08-17-fast-map-matching-proposal.md` and record your decision at
the top of it: build now, build later, or decline. Commit that decision.

---

### Task 9: API surface and user experience

The last pass over the code itself, viewing the package the way a first-time
user meets it.

**Files:**
- Read: `src/redlight/__init__.py`, every public signature, `examples/**`,
  `scripts/customer_report.py`, `scripts/mover_screen.py`
- Create: `.plans/reviews/2026-08-17-ship/09-api-ux.md`

**Interfaces:**
- Consumes: all prior findings files (to avoid duplicate reports).
- Produces: `09-api-ux.md`. Anything here that would change a public signature
  is recorded for a future major version, not fixed now — the API is frozen.

- [ ] **Step 1: Run the pass**

```text
You are reviewing the public API and user experience of the Python package
`redlight` at /Users/corysouthall/Downloads/roadtraffic-corytest, as if you
were a competent GIS analyst meeting it for the first time.

Read the Summary of every file in .plans/reviews/2026-08-17-ship/ first so you
do not re-report known findings.

1. COHERENCE OF THE 30 PUBLIC SYMBOLS. Use inspect.signature on everything in
   redlight.__all__ and tabulate them. Then check:
   - naming: is the verb/noun convention consistent (aggregate_speeds,
     assign_speeds, peak_analysis, congestion_report, day_type_report,
     network_stats, connectivity_report — are "_report" and "_stats" and bare
     nouns used to mean consistent things?)
   - parameter naming: does the same concept use the same argument name
     everywhere (speed_unit vs output_unit — when is each correct?)
   - keyword-only vs positional: consistent across similar functions?
   - return types: when does a function return a DataFrame, when a dict, when
     a modified Network in place? Is in-place mutation ever a surprise?
     assign_speeds and assign_segment_speeds write onto network edges — is
     that documented at the call site and is it reversible?
   - units: every function that takes or returns a speed — is the unit
     explicit in the signature, and is m/s truly the internal representation
     everywhere (Global Constraints require conversion only at the boundary)?

2. THE FIRST-RUN EXPERIENCE. Work through examples/ in order (00_setup
   through 06_mapping) as a new user would, running each. Note every place
   you had to look at package source to understand what to do next, every
   unexplained magic number, and every step that would fail on a user's own
   data in a way the example does not prepare them for.

3. THE SCRIPTS. Run scripts/customer_report.py and scripts/mover_screen.py
   against examples/sample_data. Check --help is accurate and complete, that
   they fail clearly on missing or malformed arguments, and that output paths
   are not silently overwritten.

4. IMPORT COST. Measure `python -X importtime -c "import redlight"` and report
   the total and the worst offenders. A geospatial package that takes seconds
   to import is unpleasant in a notebook. Check that matplotlib, pyogrio and
   pyproj are all genuinely deferred and not imported at package import time.

5. DEAD AND VESTIGIAL CODE. Find unused private helpers, parameters accepted
   but never used, and code paths unreachable since the rename in 189a505.

6. THE LINT GAP. CI runs `ruff check src tests scripts examples` — it does not
   lint benchmarks/. Run ruff over benchmarks/ and report what it finds, and
   whether the CI invocation should be widened.

Anything that would require changing a name or signature in redlight.__all__
is OUT OF SCOPE to fix — the API is frozen for this release. Record those under
a separate heading "## Deferred to a future major version" with a suggested
migration, and do not assign them a severity.

Do not change any source file. Leave the repo clean.

Write to .plans/reviews/2026-08-17-ship/09-api-ux.md in the exact format from
.plans/2026-08-17-redlight-ship-review.md. Leave Verdict and Outcome blank.
Do not commit.
```

- [ ] **Step 2: Triage** — fill in every Verdict line.

- [ ] **Step 3: Run the Fix Cycle** with `<FINDINGS>` =
`.plans/reviews/2026-08-17-ship/09-api-ux.md`.

- [ ] **Step 4: Commit the findings file**

```bash
git add .plans/reviews/2026-08-17-ship/09-api-ux.md
git commit -m "docs: API and UX findings"
```

---

### Task 10: Release readiness and the v0.6.0 GitHub release

Target is a **GitHub release, not PyPI**. Packaging still has to be right —
people install from git — but there is no index metadata to get wrong.

**Files:**
- Read/Modify: `CHANGELOG.md`, `src/redlight/__init__.py` (version),
  `README.md`, `MANIFEST.in`, `.github/workflows/ci.yml`, `mkdocs.yml`
- Create: `.plans/reviews/2026-08-17-ship/10-release-readiness.md`

**Interfaces:**
- Consumes: every findings file, and the Outcome lines the Fix Cycles wrote.
- Produces: a tagged `v0.6.0` release.

- [ ] **Step 1: Run the readiness pass**

```text
You are preparing the Python package `redlight` at
/Users/corysouthall/Downloads/roadtraffic-corytest for a v0.6.0 GITHUB release
(not PyPI). Read every file in .plans/reviews/2026-08-17-ship/ first,
including the Outcome lines, so you know what was found and what was fixed.

1. UNFINISHED BUSINESS. List every finding across all passes whose Verdict was
   ACCEPT but whose Outcome is not `fixed`, and every DEFER. For each, state
   in one line whether it blocks a release. Anything S1 that is unfixed blocks;
   argue explicitly if you think an exception is warranted.

2. VERSION CONSISTENCY. src/redlight/__init__.py declares __version__ = "0.6.0"
   and pyproject.toml reads it dynamically. Existing tags run v0.1.0..v0.5.0,
   so 0.6.0 is unreleased. Confirm: no other file hardcodes a version, the
   docs do not cite an older one, and 0.6.0 is the right number given what
   changed. Note commit 189a505 is `refactor!:` — a breaking rename of the
   import name from roadtraffic to redlight. Under semver on a 0.x line that
   is defensible as a minor bump, but the CHANGELOG must state it in the
   loudest possible terms, because every existing user's `import roadtraffic`
   breaks. Verify it does.

3. CHANGELOG. Read CHANGELOG.md. A `## [0.6.0] - 2026-08-06` section already
   exists at line 9 and there is no "Unreleased" heading. Two things to check:
   - The date is 2026-08-06, but commits 189a505 (the rename), 7e14cd5 and
     e30b667 all landed after it, and this review's Fix Cycles landed later
     still. Update the date to the actual release date and confirm the section
     now covers every user-visible change in `git log --oneline v0.5.0..HEAD`,
     including everything changed during this review.
   - Confirm the section leads with the roadtraffic -> redlight rename as a
     breaking change, in the loudest possible terms, with migration
     instructions. If it does not, that is the single most important edit in
     this task.

4. PACKAGING. Rebuild cleanly and inspect:
     rm -rf dist build
     .venv/bin/python -m build
     .venv/bin/twine check dist/*
     .venv/bin/python -m tarfile -l dist/redlight-0.6.0.tar.gz
     .venv/bin/python -m zipfile -l dist/redlight-0.6.0-py3-none-any.whl
   Confirm: the sdist carries tests/ and conftest.py (MANIFEST.in exists
   specifically because setuptools once omitted conftest.py and left source
   builds with 7 collection errors), the wheel does NOT ship tests or
   examples, LICENSE is included, and no stray build/ site/ .pytest_cache
   or .DS_Store files are in either. A .DS_Store does exist in the repo root;
   it is gitignored (.gitignore line 4), but confirm MANIFEST.in does not pull
   it into the sdist anyway — gitignore does not govern sdist contents.

5. CLEAN-ENVIRONMENT INSTALL. In a throwaway venv outside the repo, install
   the built wheel, and from a directory that is NOT the repo run a real
   smoke test: import redlight, load a network and points from
   examples/sample_data, match, derive speeds, aggregate, export GeoJSON.
   Then repeat installing the sdist. Both must work with no extras beyond
   the core dependencies.

6. CI. Read .github/workflows/ci.yml. It runs test (ubuntu 3.9/3.11/3.13 plus
   macOS 3.9/3.13), package (build + twine + sdist test suite) and docs jobs.
   Confirm it is currently green on main (`gh run list --limit 5`), that the
   matrix still matches the classifiers in pyproject.toml, and that the lint
   invocation covers everything it should (see the Task 9 lint-gap finding).
   Note .gitlab-ci.yml also exists — determine whether it is live or vestigial
   and say which.

7. DOCS SITE. `.venv/bin/mkdocs build --strict` must succeed with no warnings.
   Check every nav entry in mkdocs.yml resolves and no internal link is broken.
   Confirm site/ is gitignored and not committed.

8. REPO HYGIENE. `git status --porcelain` clean. Confirm build/, dist/, site/,
   .pytest_cache/, .ruff_cache/, __pycache__/ and .DS_Store are all ignored.
   README badges and the repository URLs in pyproject.toml
   ([project.urls]: cesouth/redlight) resolve correctly.

Write your assessment to
.plans/reviews/2026-08-17-ship/10-release-readiness.md, ending with an explicit
GO or NO-GO recommendation and, if NO-GO, the shortest list of things that
would change it. Fix documentation, CHANGELOG and packaging problems as you
find them and commit them. Do NOT create the tag or the release — that is my
call, and I make it after reading your assessment.
```

- [ ] **Step 2: Read the GO/NO-GO recommendation and decide.**

- [ ] **Step 3: Tag and release** — only after a GO.

```bash
.venv/bin/pytest -q
.venv/bin/ruff check src tests scripts examples
git status --porcelain
git tag -a v0.6.0 -m "redlight 0.6.0"
git push origin main --follow-tags
gh release create v0.6.0 dist/redlight-0.6.0.tar.gz dist/redlight-0.6.0-py3-none-any.whl \
  --title "redlight 0.6.0" \
  --notes-file <(sed -n '/## \[0.6.0\]/,/## \[0.5.0\]/p' CHANGELOG.md | head -n -1)
```

Verify the release notes rendered correctly on GitHub before announcing it. If
the `sed` range does not match your CHANGELOG's actual heading style, write the
notes to a file by hand and pass that instead — do not let a malformed extract
become the public release notes.

- [ ] **Step 4: Commit the readiness record**

```bash
git add .plans/reviews/2026-08-17-ship/10-release-readiness.md
git commit -m "docs: v0.6.0 release readiness assessment"
git push
```

---

## Coverage Map

Which pass covers which part of the package. Every source file appears at least
twice — once for correctness, once for drift or robustness.

| Module | Accuracy | Drift | Robustness | Perf | API |
|---|---|---|---|---|---|
| `_geo.py` | T3 | T1 | — | — | — |
| `_proj.py` | T3 | T1 | T5 | — | — |
| `units.py` | T3 | T2 | — | — | T9 |
| `points.py` | — | T2 | T5 | — | T9 |
| `network.py` | T3 | T1 | T5 | T7/T8 | T9 |
| `osm.py` | — | T2 | T5 | — | T9 |
| `matching.py` | T3 | T2 | T5 | T7/T8 | T9 |
| `speeds.py` | T3 | T2 | T5 | T7 | T9 |
| `cleaning.py` | T4 | T2 | — | — | T9 |
| `modes.py` | T4 | T1 | — | — | T9 |
| `aggregate.py` | T4 | T2 | — | T7 | T9 |
| `analysis.py` | — | T2 | T5 | — | T9 |
| `routing.py` | — | T2 | T5 | — | T9 |
| `mapping.py` | — | T2 | T5 | — | T9 |
| `tests/` | — | — | — | — | T6 |
| `docs/`, `examples/` | T4 | T2 | — | — | T9 |
| packaging, CI | — | — | — | — | T10 |

---

## Notes On Running This

- **Order matters.** Task 0 is mandatory first. Task 3 must precede Task 4
  (statistics built on unverified primitives prove nothing). Task 7 must
  precede Task 8 (never optimize without a profile). Task 10 is last.
  Tasks 1, 2, 5, 6 and 9 can be reordered among themselves.
- **Passes 1–7 and 9 change no source.** Only the Fix Cycles and Task 8 do.
  If a pass reports having edited `src/`, that is a process failure — revert it
  and re-run the pass.
- **Two known-stale beliefs, corrected here so no pass wastes time on them:**
  the `interval_id` cross-call collision is now guarded by `_INTERVAL_IDENTITY`
  (`aggregate.py:187`), and `speed_var` is now consumed by `weight_by_variance`
  (`aggregate.py:314`). Both were open findings in older notes; both are closed.
  Tasks 3 and 4 verify the guards actually work rather than assuming they do.
- **If a pass produces more than ~15 findings**, stop and triage before running
  the next one. A large findings file usually means the slice was too big or the
  bar was too low, and the fix cycle will be unmanageable.
