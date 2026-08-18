# Documentation drift — docs, docstrings and examples vs behaviour — findings

**Pass:** Task 2
**Date:** 2026-08-17
**Commit reviewed:** `4492743`
**Scope:** `README.md`, `docs/index.md`, `docs/quickstart.md`,
`docs/statistics.md`, `docs/methodology.md`, `docs/api.md`, everything under
`examples/` (9 scripts, `_common.py`, `examples/README.md`), `docs/figures/`,
`mkdocs.yml`, `CHANGELOG.md`, and the docstring of every symbol in
`redlight.__all__` (30 symbols, via `inspect`). `tests/test_docs.py` read first
to establish what is already covered. Also `src/redlight/_proj.py:7-13`, handed
to this pass by F-1.5.
**Method:** eight scratch scripts under `.venv/bin/python` in the scratchpad
(`d1_peak_vs_classify.py`, `d2_defaults.py`, `d3_defaults2.py`, `d4_claims.py`,
`d5_prose.py`, `d6_more.py`, `d7_sigs.py`, `d8_final.py`), plus all nine
`examples/` scripts executed end to end and their stdout compared against their
own narration. Suite re-run: **378 passed**; `git status --porcelain` empty. No
source file changed, nothing committed.

## Summary

The documentation is in unusually good shape for a package this size: all 28
signatures spelled out in `docs/api.md` match `inspect.signature` exactly, every
documented default across the 30 public symbols is the real one, all nine
examples run clean, and the `roadtraffic` → `redlight` rename left no residue
anywhere — not in prose, not in URLs, not in the figure PNGs. Six findings, all
of them doc-side rather than code-side, which is the good direction. The worst
is an example that tells the reader two functions produce "the same split" when
they use different selection rules and in fact return disjoint answers —
`tests/test_docs.py` cannot catch it because the code runs fine and only the
narration is wrong.

### Already covered by `tests/test_docs.py` — not re-checked

It extracts every fenced `python` block from `README.md` and `docs/*.md` and
executes each against a freshly rebuilt namespace, with `<!-- skip-test -->` and
`<!-- needs: mod -->` opt-outs. 27 blocks pass. So *syntactic* drift — a renamed
parameter, a changed return key — is already caught. This pass looked only for
what executing a block cannot detect: prose that asserts a behaviour, and code
that runs but does not demonstrate what the surrounding text says it does.

### Clean results worth recording

| Check | Result |
|---|---|
| 28 signatures in `docs/api.md` vs `inspect.signature` | **all match** (the one flagged by the parser, `api.md:702`, is a header documenting `to_mps` and `from_mps` in one span) |
| Documented defaults across all 30 public symbols | **no mismatches** — every `Default X` claim is the real default |
| All 9 `examples/` scripts | **exit 0**, output consistent with narration except F-2.2 |
| The rename (`189a505`) | **no residue** — `grep -rn roadtraffic` over source, docs, examples, scripts, tests, benchmarks hits only `CHANGELOG.md:43-44`, which documents the rename and should say it |
| GitHub URLs | all six resolve to `cesouth/redlight`; none stale |
| `docs/figures/*.png` | generated pre-rename (`a9d3189`, 2026-07-08) but `strings` finds no old package name in any of the seven |
| Cross-references into `statistics.md` (§2.2, §3, §7, §8, §9, §10, §11) | every section number and anchor resolves |
| `statistics.md:8-9` exact conversions | `_MPH_TO_MPS = 1609.344/3600`, `_KPH_TO_MPS = 1000/3600` — exactly as documented |
| `statistics.md:43` "cKDTree over segment midpoints" | true — `network.py` builds `cKDTree(mids)` |
| `quickstart.md:162` "6-hour blocks (00–06, 06–12, 12–18, 18–24)" | exactly what `aggregate_speeds(block_hours=6)` emits |
| `methodology.md:157` bounded transition search | matches `matching.py`: `max(self.max_route_dist_factor * gc_step, self.max_dist * 4)` |
| `__init__.py` module docstring, 30 entries | each checked against its function; all accurate |

---

### F-2.1 — `README.md` announces the wrong version

- **Severity:** S3
- **Location:** `README.md:3`
- **Claim:** The README's masthead says **Version 0.5.0**. The package is
  0.6.0, and has been since `189a505`. This is the first line of the first
  document a reader sees, and the only version claim in the docs.
- **Evidence:**
  ```
  $ .venv/bin/python d8_final.py
  == README.md:3 version claim vs reality ==
     README.md:3 says : Version 0.5.0
     redlight.__version__: 0.6.0
     pyproject dynamic?  : ['version'] None
     CHANGELOG newest    : ## [0.6.0] - 2026-08-06
  ```
  `pyproject.toml:67` derives the version from `redlight.__version__`, so
  `__version__`, the built wheel and the changelog all agree at 0.6.0. Only the
  README disagrees.
- **Expected vs actual:** **The doc is wrong.** The version is single-sourced
  from `redlight.__version__` precisely so it cannot drift, and the README is
  the one place that restates it by hand.
- **Suggested fix:** Either update the line to 0.6.0, or drop the version from
  the masthead entirely so there is nothing to keep in sync — the changelog and
  PyPI already carry it. The second is what stops this recurring at 0.7.0.
- **Verdict:** ACCEPT
- **Outcome:** fixed (1dceb74)

---

### F-2.2 — an example calls two different hour-selection rules "the same split", and they disagree

- **Severity:** S3
- **Location:** `examples/02_speed_analysis/peak_and_daytype.py:37`
- **Claim:** The script prints `peak_analysis`' three slowest hours, then prints
  `classify_hours`' output under the heading **"The same split, as reusable hour
  sets"**. They are not the same split: `peak_analysis(n_peak=3)` returns the
  three individually slowest bins, while `classify_hours(n_peak=3)` selects a
  *contiguous* 3-hour window. On the shipped sample data the two answers overlap
  in one hour out of three.
- **Evidence:**
  ```
  $ .venv/bin/python d1_peak_vs_classify.py
  peak_analysis  peak=[7, 10, 19]   offpeak=[15, 21, 23]
  classify_hours peak=[18, 19, 20]   offpeak=[0, 22, 23]
  classify_hours source = 'window'

  peak sets agree?    False   overlap=[19]
  offpeak sets agree? False   overlap=[23]
  ```
  The cause is in the library's own docstring, `aggregate.py:546-551`:
  "**Contiguous windows** -- pass `n_peak` and `n_offpeak`. The peak block is
  the contiguous `n_peak`-hour window (wrapping midnight) with the *lowest*
  network-wide representative speed", against `peak_analysis`
  (`aggregate.py:455`) which simply does `aggregated.sort_values(col)` and takes
  the head and tail. Two different rules behind the same parameter name.
- **Expected vs actual:** **The example's narration is wrong; the code is
  right.** Both functions do exactly what their own docstrings say. But a reader
  working through `examples/` in order — which the plan's Task 9 treats as the
  first-run experience — is told these are two shapes of one answer, and will
  reasonably use `classify_hours`' hour lists believing they name the three
  slowest hours. They do not.
- **Suggested fix:** Change the heading and add one sentence saying the two
  functions answer different questions: `peak_analysis` ranks individual bins,
  `classify_hours(n_peak=)` finds a contiguous block, and the contiguous block
  is what `assign_segment_speeds` needs because a regime has to be a period, not
  a scatter of hours.
- **Verdict:** ACCEPT
- **Outcome:** fixed (fabbe5e) -- narration only; no failing test possible

---

### F-2.3 — `classify_hours` window mode returns hours with no observations, which its docstring only rules out for a different mode

- **Severity:** S3
- **Location:** `src/redlight/aggregate.py:552-556`
- **Claim:** The docstring's mode 3 says "Hours with no observations are left
  unassigned." Mode 2 (contiguous windows) says nothing of the kind, and does
  the opposite: because the window is contiguous by construction, hours with
  zero observations are returned inside it. A reader who has absorbed the mode-3
  sentence will not expect this, and nothing on the page corrects them.
- **Evidence:**
  ```
  $ .venv/bin/python d1_peak_vs_classify.py
  hours classify_hours nominates, and how many observations they have:
    hour 00  offpeak n=0
    hour 18  peak    n=40
    hour 19  peak    n=6
    hour 20  peak    n=0
    hour 22  offpeak n=0
    hour 23  offpeak n=10
  ```
  Three of the six nominated hours — 00, 20 and 22 — carry no observation at
  all, on the sample dataset the examples ship with.
- **Expected vs actual:** **The doc is incomplete; the behaviour is
  defensible.** A contiguous window must span every hour it covers, so excluding
  the empty ones would make it non-contiguous and defeat the mode. The scoring
  is honest — `aggregate.py:551` says windows are "scored on the mean of their
  observed hourly speeds", so an empty hour contributes nothing to the score.
  What is missing is any statement that it still lands in the returned set. The
  consequence is real: those hour lists flow into `assign_segment_speeds`, so an
  edge can be labelled peak or off-peak for an hour in which nothing was ever
  measured.
- **Suggested fix:** Add a sentence to mode 2 saying the returned window is
  contiguous and therefore may include hours with no observations, which are
  scored as absent rather than as slow. No code change — the alternative
  (dropping empty hours) would silently break contiguity.
- **Verdict:** DEFER
- **Outcome:** deferred to Task 4, which owns the peak-detection rule

---

### F-2.4 — the quickstart routes between coordinates 13 km outside its own example network

- **Severity:** S3
- **Location:** `docs/quickstart.md:184`, `docs/quickstart.md:188`
- **Claim:** The routing snippets call
  `router.route((-77.30, 38.68), (-77.27, 38.71), ...)`. Every other document
  and the `test_docs.py` fixture build a network spanning latitude 38.800 to
  38.816. The quickstart's origin is 13.3 km from the nearest node and its
  destination 10.0 km. The block passes its test because `Router` snaps to the
  nearest node at any distance without complaint, so the example runs, prints a
  route, and demonstrates nothing about the coordinates it names.
- **Evidence:**
  ```
  $ .venv/bin/python d6_more.py
  == README:74 / quickstart:184 route endpoints vs the doc network extent ==
      doc network extent: lon -77.3000..-77.2730, lat 38.8000..38.8160
      README.md:74       origin        0 m from nearest node, dest      342 m
      quickstart.md:184  origin    13321 m from nearest node, dest     9994 m
  ```
  `README.md:74` uses `(-77.30, 38.80)` → `(-77.27, 38.81)` and lands on the
  network, so the two documents disagree about where the example study area is.
- **Expected vs actual:** **The doc is wrong** — these look like coordinates
  left behind when the sample network moved north, and the README was updated
  while the quickstart was not. Worth flagging separately: the reason this went
  unnoticed is that `Router.route` accepts an origin arbitrarily far from the
  graph and silently snaps. That is a robustness question rather than a doc one,
  and it belongs to **Task 5** (`routing.py` boundary behaviour) — recorded here
  only because it is what makes this class of doc error invisible to
  `tests/test_docs.py`.
- **Suggested fix:** Change both calls to the README's coordinates so the
  quickstart demonstrates a route on the network it just built.
- **Verdict:** ACCEPT
- **Outcome:** fixed (b4f5384)

---

### F-2.5 — `_proj.py` states a 14 nm inverse bound that a wider sample exceeds

- **Severity:** S3
- **Location:** `src/redlight/_proj.py:10-11`
- **Claim:** The module docstring says the Krüger series agrees with PROJ
  "(measured against pyproj over a full zone: 7.5 nm forward, **14 nm
  inverse**)". Sampled across ten zones in both hemispheres the worst inverse
  error is 15.03 nm, which exceeds the stated figure. Handed to this pass by
  F-1.5 in `01-spec-drift.md`.
- **Evidence:**
  ```
  $ .venv/bin/python d8_final.py
  == _proj.py:11 '14 nm inverse' (deferred to this pass by F-1.5) ==
     worst inverse error vs PROJ: 15.03 nm  (in EPSG:32601)
     docstring states           : 14 nm
     exceeds the stated figure  : True
  ```
  200,000 points across EPSG:32601/10/18/33/56/60 north and 32701/33/56/60
  south, full ±3° zone width, latitudes 0.1–84°N and 0.1–80°S, oracle pyproj
  3.7.2.
- **Expected vs actual:** **The doc is wrong, marginally and harmlessly.** The
  sentence's own leading claim — "a few tens of nanometres" — comfortably covers
  15 nm, and the paragraph's point (eight orders of magnitude below GPS noise)
  is untouched. The defect is only that a figure measured on one sample is
  written as though it bounded all of them. Recorded because a number in a
  docstring reads as a guarantee.
- **Suggested fix:** Restate as "under 20 nm inverse", or name the sample the
  figure came from. Nothing in the code changes.
- **Verdict:** DEFER
- **Outcome:** deferred to Task 3, which measures _proj against pyproj across all zones

---

### F-2.6 — `HMMMatcher` documents a default for three of its seven parameters

- **Severity:** S5
- **Location:** `src/redlight/matching.py:126-178`
- **Claim:** The docstring states defaults for `sigma_z` (6.0), `beta` (30.0)
  and `n_jobs` (1) — all three correct — and states none for `max_dist`, `k`,
  `max_route_dist_factor` and `dist_cache_size`, whose real defaults are 50.0,
  8, 8.0 and 10,000. A reader cannot tell from the docstring whether the silence
  means "no default" or "we didn't say", and `edge_betweenness_centrality`'s
  `weight` genuinely has no default, so the distinction carries meaning
  elsewhere in this API.
- **Evidence:**
  ```
  $ .venv/bin/python d2_defaults.py
  --- HMMMatcher: which documented params state a default at all? ---
    sigma_z                  real=6.0        states a default in prose: True
    beta                     real=30.0       states a default in prose: True
    max_dist                 real=50.0       states a default in prose: True*
    k                        real=8          states a default in prose: True*
    max_route_dist_factor    real=8.0        states a default in prose: True*
    n_jobs                   real=1          states a default in prose: True
    dist_cache_size          real=10000      states a default in prose: False
  ```
  (*the three starred `True`s are the crude chunker over-reading into the next
  parameter's block; reading `matching.py:138-171` directly, `max_dist` is
  "Candidate snap tolerance in metres.", `k` is "Max candidate edges per
  point.", and `max_route_dist_factor`'s paragraph never names 8.0 — none of
  the three states a default. `d3_defaults2.py` prints the raw sentences and
  confirms it.)
- **Expected vs actual:** **The doc is incomplete**, not wrong — no stated
  default is incorrect anywhere in the package. `dist_cache_size`'s 10,000 is
  the one that matters soonest: Task 7 is asked to determine whether that
  default binds, and the docstring does not say what it is.
- **Suggested fix:** Add the four missing defaults, matching the phrasing
  already used for `sigma_z` and `beta`.
- **Verdict:** REJECT
- **Outcome:** no change needed (states nothing incorrect; see note)

---

## Handed to Task 7 — performance claims, recorded UNVERIFIED

Per this pass's instructions these were not benchmarked. Every scale or
throughput assertion in the shipped documentation:

| Location | Claim |
|---|---|
| `src/redlight/matching.py:159` | the serial path "already sustains tens of thousands of points per second" |
| `src/redlight/matching.py:163` | "on the development machine serial matching stayed faster up to at least two million points" |
| `src/redlight/matching.py:170` | larger `dist_cache_size` is "faster on data that revisits the same roads, at more memory (~tens of KB per entry)" |
| `docs/api.md:83` | batch snapping is "10-30x faster for large point sets; `chunk_size` only bounds peak memory" |
| `docs/api.md:172` | `NearestMatcher` does "thousands of points per second" |
| `docs/api.md:186` | `HMMMatcher` does "tens of thousands of points per second serial" |
| `src/redlight/analysis.py:266` | a numpy hull path is "roughly 15x faster than building a shapely `MultiPoint` for the same" |
| `docs/statistics.md:48` | `NearestMatcher` is "O(N log M) for N points and M segments" |

Note `docs/api.md:172` and `:186` sit two paragraphs apart and give
`NearestMatcher` *thousands* and `HMMMatcher` *tens of thousands* of points per
second — i.e. they claim the HMM matcher is an order of magnitude **faster**
than the nearest-edge matcher, which contradicts `docs/quickstart.md:83`
("Speed | Fast | Slower (shortest-path calls)") and `statistics.md:110`. One of
the two figures is almost certainly mis-stated. Task 7 should settle which.

## Handed to Task 4 — numeric claims in `docs/methodology.md`

`d4_claims.py` extracts every prose line carrying a number (180 lines, full dump
in the scratchpad). The load-bearing experimental claims, which Task 4 must
reconcile against a re-run of `scripts/paper_experiments.py`:

| § | Line | Claim |
|---|---|---|
| Abstract | 23-25 | recovers a constant 10 m/s speed with **+0.3 % bias and 15 % relative spread at σ = 5 m**; tracks √2·σ/(Δt·v) |
| Abstract | 27-28 | above **σ ≈ 15 m** on ~110 m grids, per-interval speeds become matching-dominated |
| 2 | 90-91 | at σ = 30 m on a ~110 m grid, "about half of all fixes" are nearer a wrong road |
| 2.3 (Exp A) | 167-168 | 40 trajectories, 25 edges each, fixes every 5 s at 10 m/s, **2,200 fixes per condition** |
| 2.3 (Exp A) | 175-178 | accuracy nearest vs HMM: **91.2/96.7** (5 m), **75.6/80.6** (15 m), **53.4/55.6** (30 m), **34.5/36.4** (50 m) |
| 2.3 | 199-201 | at σ = 15 m, nearest-matcher speed bias **+12.7 %** vs HMM **+4.1 %** |
| 3.3 | 262-263 | a 15 m receiver at 5 s "can never see a 10 m/s road better than **±42 %** per interval" |
| 3.4 (Exp B) | 287-290 | Table 2, theory/spread/bias at σ = 5/15/30/50 m — 12 numbers |
| 3.4 (Exp B) | 296-298 | Table 3 Δt sweep at σ = 30 m: **+31.8 %** (5 s), **+8.3 %** (15 s), **−1.7 %** (30 s) |
| 3.4 | 302-305 | matcher comparison rows: **+0.3/+1.4**, **+4.1/+12.7**, **+32.9/+59.9**, **+105.7/+146.8** |
| 3.4 | 322 | surplus spread **0.52 vs 0.42** at σ = 15; **1.07 vs 0.85** at σ = 30 |
| 3.4 | 330-336 | quality-filtering at Δt = 5 s biases **+67 %** at σ = 15 m; **95–100 %** pass when used correctly |
| 3.4 | 337 | "matcher choice is worth **2–3×** in speed bias" |
| 4.2 | 367-368 | not deduplicating would "shrink confidence intervals by **√12**" |
| 4.4 (Exp C) | 397-399 | **240 movers**, planted 4 m/s at 07–09 and 16–18, 15 m/s at 22–05, 9 m/s otherwise, σ = 15 m, Δt = 10 s, **6,935 fixes** |
| 4.4 (Exp C) | 413 | detector returned **peak = {7, 8, 9}**, **off-peak = {2, 3, 4}** |
| 4.4 (Exp C) | 421-423 | peak hours read **≈ 7.4 m/s** where truth is 4.0; free-flow **15.6 vs 15.0**; SNR **~1.9** |

Also for Task 4: `docs/figures/experiment_results.json` was last written
`a9d3189` (2026-07-08), before both the rename and the `c9a92bb`/`a03733f` fix
commits — so a re-run reproducing it exactly would be mild evidence that those
fixes did not perturb the experiments, and any movement needs attribution.

## Repo state on exit

At the end of the **audit pass** (commit `4492743`), before any triage:

```
$ git status --porcelain
(empty)

$ .venv/bin/pytest -q
378 passed, 6 warnings in 14.11s
```

The nine example scripts write only into `examples/sample_data/`, which is
gitignored (`.gitignore:5`), so running them all left the tree clean. No source
file was modified by the audit and nothing was committed by it. The Fix Cycle
below ran afterwards, as a separate step.

---

## Triage and fix cycle

Triaged 2026-08-17. Three findings accepted, two deferred, one rejected.

| Finding | Sev | Verdict | Outcome |
|---|---|---|---|
| F-2.1 | S3 | ACCEPT | fixed (`1dceb74`) |
| F-2.2 | S3 | ACCEPT | fixed (`fabbe5e`) — narration only |
| F-2.3 | S3 | DEFER | Task 4 owns the peak-detection rule |
| F-2.4 | S3 | ACCEPT | fixed (`b4f5384`) |
| F-2.5 | S3 | DEFER | Task 3 owns the `_proj` measurement |
| F-2.6 | S5 | REJECT | states nothing incorrect |

**F-2.1 — fixed, plus a guard.** The masthead moved to 0.6.0. The finding
offered a second option, dropping the version from the README entirely; the Fix
Cycle asks for the smallest change, so the number was corrected rather than
removed, and the durability concern is handled instead by
`test_prose_version_claims_match_the_package`, which fails on any `**Version
X.Y.Z**` in `README.md` or `docs/*.md` that disagrees with
`redlight.__version__`. Red first: `AssertionError: documentation claims a
version the package does not have (redlight.__version__ == 0.6.0):
[('README.md', '0.5.0')]`.

**F-2.2 — no failing test was possible.** The code is correct and only the
narration was wrong, so a red test would have required breaking working code.
Recorded here per the Fix Cycle's explicit instruction rather than satisfied
with a proxy assertion. The fix was verified by running the script and reading
its output. The new text states which question each function answers, why
`assign_segment_speeds` wants the contiguous window rather than the ranking,
and — carrying F-2.3's substance into the place a reader will actually meet it
— that a contiguous window can span an hour with no observations.

**F-2.4 — the test is the fix.** `test_documented_route_endpoints_lie_on_the_documented_network`
extracts every `.route((lon, lat), (lon, lat))` literal from the docs and
measures it against the grid `tests/test_docs.py` builds, failing beyond 2 km.
It named both offending lines and nothing else, so the README's endpoints were
confirmed good in the same run:

```
E  docs/quickstart.md:184 origin (-77.3, 38.68) is 13,321 m from the nearest node
E  docs/quickstart.md:184 destination (-77.27, 38.71) is 9,994 m from the nearest node
E  docs/quickstart.md:188 origin (-77.3, 38.68) is 13,321 m from the nearest node
E  docs/quickstart.md:188 destination (-77.27, 38.71) is 9,994 m from the nearest node
```

This is the finding's real value: executing a block could never catch it,
because `Router.route` snaps at any distance. **The underlying robustness
question — that routing accepts an origin 13 km off-network without a word —
is untouched and still belongs to Task 5.**

**Why F-2.3 and F-2.5 are deferred rather than rejected.** Both are live gaps in
shipped source, but each sits inside a later pass's remit, and that pass will
have a better answer than this one can give. Task 4 is directed at
"`peak_analysis` and `classify_hours`: the peak-detection rule, ties, and the
degenerate cases", so it may conclude that empty hours should not be in the
window at all — a stronger fix than the sentence F-2.3 proposes, and one this
pass has no standing to choose. Task 3 measures `_proj` against pyproj across
all 120 zones, which will produce a better bound than F-2.5's ten-zone sample;
writing "under 20 nm" now would only have to be rewritten then.

**Why F-2.6 is rejected.** It states nothing incorrect — every default the
`HMMMatcher` docstring gives is right, and the four it omits are one
`inspect.signature` call away. Adding them would restate in prose four numbers
that already have a single authoritative source, which is the exact pattern
that produced F-2.1: the version was wrong precisely because a document
restated something the code already owned. Task 7 needs
`dist_cache_size`'s 10,000, and can read it from the signature.

### Verification after all three fixes

```
$ .venv/bin/pytest -q
380 passed, 6 warnings in 13.43s

$ .venv/bin/ruff check src tests scripts examples
All checks passed!

$ .venv/bin/mkdocs build --strict
exit 0
```

380 = the 378 carried in + 1 test for F-2.1 + 1 for F-2.4. No test was weakened
or removed. `mkdocs build --strict` was run because two of the three fixes touch
published documentation.
