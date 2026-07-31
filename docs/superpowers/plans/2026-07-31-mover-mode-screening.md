# Mover Mode Screening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a per-mover mode classifier to the `roadtraffic` package so mixed GPS feeds (vehicles plus people on foot) can be screened without deleting the congestion the study measures.

**Architecture:** A new leaf module `src/roadtraffic/modes.py` holding four pure functions over pandas frames. It depends only on `roadtraffic.units` and scipy; nothing in the package imports it, so it can be built and tested in isolation before the script and report are rewired onto it. Classification happens per `traj_id` and the verdict applies to every one of that mover's observations.

**Tech Stack:** Python 3.9+, numpy, pandas, scipy (`gaussian_kde`), pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-07-31-mover-mode-screening-design.md`

## Global Constraints

- `requires-python = ">=3.9"`. **Every new module must start with `from __future__ import annotations`** — the codebase uses `X | None` annotations, which are syntax errors on 3.9 without it.
- ruff `line-length = 95`, lint rules `["E", "F", "W", "I", "B", "UP"]`. Run `ruff check src/ tests/ scripts/` before every commit.
- All internal computation is in **m/s**. Convert at the boundary with `roadtraffic.units.to_mps` / `from_mps`. Never hardcode a conversion factor.
- No new dependencies. scipy is already a core dependency (`scipy>=1.7`).
- Test command: `python -m pytest tests/ -q`. Full suite currently **233 passed**; it must stay green.
- Public functions get numpy-style docstrings matching the density and tone of `cleaning.py`. Comments explain *why*, not *what*.
- Do not change existing function signatures. This work is purely additive until Task 7.

---

### Task 1: `modes.py` scaffold and `mover_features`

**Files:**
- Create: `src/roadtraffic/modes.py`
- Create: `tests/test_modes.py`

**Interfaces:**
- Consumes: `roadtraffic.units.SpeedUnit`, `from_mps`
- Produces: `MODE_PEDESTRIAN = "pedestrian"`, `MODE_VEHICLE = "vehicle"`, `MODE_UNKNOWN = "unknown"`, `_percentile_label(percentile: float) -> str`, `_require_traj_columns(obs, fn_name, *, need_speed=True) -> None`, `mover_features(obs, *, percentile=85.0, unit="mph") -> pd.DataFrame`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_modes.py`:

```python
import numpy as np
import pandas as pd
import pytest

import roadtraffic as rt


def obs_frame(specs, *, distance_m=200.0, t0="2026-06-01T08:00:00"):
    """A derive_speeds-shaped frame. ``specs`` maps traj_id -> list of m/s."""
    rows, iid = [], 0
    t0 = pd.Timestamp(t0)
    for tid, speeds in specs.items():
        for v in speeds:
            rows.append({
                "interval_id": iid, "traj_id": tid, "speed_mps": float(v),
                "time": t0 + pd.Timedelta(minutes=iid),
                "distance_m": distance_m, "snap_dist_m": 5.0,
            })
            iid += 1
    return pd.DataFrame(rows)


def test_mover_features_is_one_row_per_mover():
    feat = rt.mover_features(obs_frame({"a": [10.0, 20.0, 30.0], "b": [1.0, 1.5]}),
                             unit="mps")
    assert list(feat.index) == ["a", "b"]
    assert feat.loc["a", "n_intervals"] == 3
    assert feat.loc["b", "speed_median_mps"] == pytest.approx(1.25)
    assert feat.loc["a", "distance_m"] == pytest.approx(600.0)


def test_mover_features_dedups_repeated_intervals():
    """edge_observations repeats an interval once per edge crossed. Without
    dedup each mover's statistics get weighted by how many edges its hops
    spanned, biasing every mover toward its longest intervals."""
    obs = obs_frame({"a": [2.0, 20.0]})
    slow = obs[obs["speed_mps"] == 2.0]
    long_form = pd.concat([obs] + [slow] * 4, ignore_index=True)
    feat = rt.mover_features(long_form, unit="mps")
    assert feat.loc["a", "n_intervals"] == 2
    assert feat.loc["a", "speed_median_mps"] == pytest.approx(11.0)


def test_mover_features_converts_to_the_requested_unit():
    feat = rt.mover_features(obs_frame({"a": [10.0] * 4}), unit="mph")
    assert "speed_p85_mph" in feat.columns
    assert feat.loc["a", "speed_p85_mph"] == pytest.approx(10.0 / 0.44704, rel=1e-6)


def test_mover_features_missing_columns_raises():
    with pytest.raises(ValueError, match="traj_id"):
        rt.mover_features(pd.DataFrame({"speed_mps": [1.0]}))


def test_mover_features_empty_input_returns_typed_empty_frame():
    empty = pd.DataFrame({"interval_id": [], "traj_id": [], "speed_mps": [],
                          "distance_m": []})
    feat = rt.mover_features(empty, unit="mps")
    assert len(feat) == 0
    assert "speed_p85_mps" in feat.columns
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_modes.py -q`
Expected: FAIL — `AttributeError: module 'roadtraffic' has no attribute 'mover_features'`

- [ ] **Step 3: Write the implementation**

Create `src/roadtraffic/modes.py`:

```python
"""Per-mover mode screening for mixed GPS feeds.

A trafficability study reads congestion off *slow* observations, so the one
thing it must not do is drop observations for being slow -- that deletes the
finding. But when a feed also carries people on foot, their fixes match to the
same roads and drag every road's score down.

The resolution is that **mode is a property of the mover, not of the fix**. A
pedestrian is slow for their entire track; a congested vehicle is slow on one
segment and free-flowing elsewhere in the same trip. So this module classifies
whole trajectories on a high percentile of their speed and applies the verdict
to every one of that mover's observations -- a mover judged to be a vehicle
keeps all of its data, crawling included.

Known limitations, stated here rather than buried:

* A vehicle whose **entire** track is gridlocked never shows a fast stretch and
  is classified as a pedestrian. The resulting bias is *upward* -- the same
  direction as ``require_quality`` -- because the vehicles wrongly dropped are
  the slowest. Run the study screened and unscreened, compare, and report the
  gap as uncertainty.
* **Cyclists** occupy 9-16 mph, which is where urban vehicles in traffic also
  sit. They are not separable on speed and will generally be labelled vehicles.
* **Snap distance** was tested as a second discriminator and rejected: alone it
  dropped 60 real vehicles out of 231, and combined with speed it readmitted
  walkers and made per-edge error worse (1.7 vs 0.6 mph). It is reported as a
  diagnostic column but takes no part in the verdict.
* Where the feed carries device type, fleet id or source app, that metadata
  beats inference and should be used instead of this module.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .units import SpeedUnit, from_mps

MODE_PEDESTRIAN = "pedestrian"
MODE_VEHICLE = "vehicle"
MODE_UNKNOWN = "unknown"


def _percentile_label(percentile: float) -> str:
    """``'85'`` for 85.0 and ``'87.5'`` for 87.5 -- keeps column names readable."""
    return f"{percentile:g}"


def _require_traj_columns(obs, fn_name: str, *, need_speed: bool = True) -> None:
    required = {"traj_id"} | ({"speed_mps"} if need_speed else set())
    missing = required - set(obs.columns)
    if missing:
        raise ValueError(
            f"{fn_name} needs columns {sorted(required)}; missing "
            f"{sorted(missing)}. Pass the 'intervals' or 'edge_observations' "
            "frame from derive_speeds, and load points with a trajectory id "
            "(load_points(id_col=...)) so traj_id is present."
        )


def mover_features(obs, *, percentile: float = 85.0, unit="mph") -> pd.DataFrame:
    """Reduce speed observations to one evidence row per mover.

    Parameters
    ----------
    obs : DataFrame
        Any frame carrying ``traj_id`` and ``speed_mps`` -- ``derive_speeds``'
        ``intervals`` or ``edge_observations``, or a matched frame with a
        logged speed. When an ``interval_id`` column is present the frame is
        deduplicated on it first: ``edge_observations`` repeats each interval
        once per edge traversed, so without this a mover's statistics would be
        weighted by how many edges its hops crossed.
    percentile : float
        Which percentile of a mover's speeds to report. The default 85 is high
        enough to see a vehicle's free-flowing stretch -- a mean or median would
        misclassify a mover that spent most of its trip in congestion -- while
        discarding the top of the distribution, where a single GPS jump would
        otherwise promote a pedestrian to a vehicle.
    unit : str or SpeedUnit
        Unit for the emitted speed columns. This is also the unit
        :func:`classify_movers` reads its ``threshold`` in, so one call can
        never compare a threshold against a differently scaled column.

    Returns
    -------
    DataFrame
        Indexed by ``traj_id``, with ``n_intervals``,
        ``speed_p<pct>_<unit>``, ``speed_median_<unit>``, ``distance_m``, and
        ``snap_dist_m`` when the input carries it.
    """
    _require_traj_columns(obs, "mover_features")
    unit = SpeedUnit.parse(unit)
    pct = _percentile_label(percentile)
    pct_col = f"speed_p{pct}_{unit.value}"
    med_col = f"speed_median_{unit.value}"

    df = obs
    if "interval_id" in df.columns:
        df = df.drop_duplicates("interval_id")
    df = df[df["speed_mps"].notna()]

    if not len(df):
        empty = pd.DataFrame(
            {c: pd.Series(dtype=float)
             for c in ("n_intervals", pct_col, med_col, "distance_m")})
        empty.index.name = "traj_id"
        return empty

    g = df.groupby("traj_id", dropna=False)
    out = pd.DataFrame({
        "n_intervals": g.size(),
        pct_col: from_mps(g["speed_mps"].quantile(percentile / 100.0), unit),
        med_col: from_mps(g["speed_mps"].median(), unit),
        "distance_m": (g["distance_m"].sum() if "distance_m" in df.columns
                       else np.nan),
    })
    if "snap_dist_m" in df.columns:
        out["snap_dist_m"] = g["snap_dist_m"].median()
    out.index.name = "traj_id"
    return out
```

Then export it. In `src/roadtraffic/__init__.py`, add to the imports (keep alphabetical order within the block, ruff rule `I` enforces this):

```python
from .modes import (
    MODE_PEDESTRIAN,
    MODE_UNKNOWN,
    MODE_VEHICLE,
    mover_features,
)
```

and add `"mover_features"`, `"MODE_PEDESTRIAN"`, `"MODE_VEHICLE"`, `"MODE_UNKNOWN"` to `__all__`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_modes.py -q && ruff check src/ tests/`
Expected: 5 passed, ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/roadtraffic/modes.py src/roadtraffic/__init__.py tests/test_modes.py
git commit -m "modes: per-mover evidence table"
```

---

### Task 2: `suggest_mode_threshold`

**Files:**
- Modify: `src/roadtraffic/modes.py` (append)
- Modify: `tests/test_modes.py` (append)

**Interfaces:**
- Consumes: `_percentile_label` and the module constants from Task 1
- Produces: `suggest_mode_threshold(mover_speeds, *, unit="mph") -> float | None`

This is the highest-risk function in the plan. Both guards below exist because a version without them nominated 13.2 mph on clean vehicle-only data — a threshold that would have deleted every congested vehicle in a study with nothing in the output looking wrong.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_modes.py`:

```python
def test_suggest_threshold_finds_the_valley_between_two_humps():
    rng = np.random.default_rng(0)
    walkers = rng.normal(1.4, 0.15, 90)
    vehicles = np.clip(rng.normal(12.0, 3.0, 200), 5.0, None)
    t = rt.suggest_mode_threshold(np.concatenate([walkers, vehicles]), unit="mps")
    assert t is not None
    assert 1.8 < t < 5.0


def test_suggest_threshold_rejects_a_valley_between_two_kinds_of_driving():
    """REGRESSION. A vehicle-only feed is itself multi-modal -- gridlock,
    urban free-flow and arterial free-flow form separate humps -- so a density
    valley proves nothing on its own. Ranking valleys by depth alone nominated
    13.2 mph on clean vehicle data, which would have deleted every congested
    vehicle in the study, invisibly. The hump BELOW the valley must be walkers."""
    rng = np.random.default_rng(1)
    gridlock = rng.normal(3.0, 0.4, 60)
    urban = rng.normal(11.0, 1.5, 120)
    arterial = rng.normal(24.0, 3.0, 80)
    speeds = np.concatenate([gridlock, urban, arterial])
    assert rt.suggest_mode_threshold(speeds, unit="mps") is None


def test_suggest_threshold_needs_enough_movers():
    assert rt.suggest_mode_threshold(np.full(10, 1.4), unit="mps") is None


def test_suggest_threshold_accepts_a_series():
    rng = np.random.default_rng(0)
    speeds = np.concatenate([rng.normal(1.4, 0.15, 90),
                             np.clip(rng.normal(12.0, 3.0, 200), 5.0, None)])
    s = pd.Series(speeds, index=[f"m{i}" for i in range(len(speeds))])
    assert rt.suggest_mode_threshold(s, unit="mps") is not None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_modes.py -q -k suggest`
Expected: FAIL — `AttributeError: ... has no attribute 'suggest_mode_threshold'`

- [ ] **Step 3: Write the implementation**

Append to `src/roadtraffic/modes.py`:

```python
# Physical anchors, in m/s. A pedestrian population peaks at walking pace;
# 0.6-2.5 m/s spans a slow stroll to a brisk walk measured at the 85th
# percentile of a track. The split, if there is one, lies below 6 m/s
# (~13 mph) -- above any walking or jogging pace, below urban free flow.
_WALK_MODE_LO_MPS = 0.6
_WALK_MODE_HI_MPS = 2.5
_SEARCH_LO_MPS = 0.5
_SEARCH_HI_MPS = 6.0
_MIN_MOVERS = 20
_MIN_PROMINENCE = 1.5


def suggest_mode_threshold(mover_speeds, *, unit="mph") -> float | None:
    """Speed at the density valley separating walkers from drivers.

    Parameters
    ----------
    mover_speeds : array-like
        One speed per mover, in ``unit`` -- typically the
        ``speed_p85_<unit>`` column of :func:`mover_features`. Any index is
        ignored; non-finite and non-positive values are dropped.
    unit : str or SpeedUnit
        Unit of ``mover_speeds`` and of the returned threshold.

    Returns
    -------
    float or None
        ``None`` when there is no walking-speed population to split off,
        which is the honest answer for a single-mode feed. Callers must not
        substitute a default: a silently chosen threshold that is wrong
        produces a study that looks correct.

    Notes
    -----
    The density is estimated in **log** speed. A fixed relative bandwidth on
    raw speed is set by the spread of the whole sample, and the driving hump
    runs out to motorway speeds -- wide enough to smooth away a valley only a
    couple of mph across. Log speed makes the bandwidth scale-free, so
    resolution near walking pace does not depend on the fastest vehicle.

    Two guards decide whether a valley is a *mode* boundary at all:

    1. Candidates are ranked by **prominence**, ``min(left_peak, right_peak) /
       valley``, and the density is evaluated well beyond the selection window
       in both directions. Ranking by absolute depth instead selects the lowest
       point of a monotone tail at the window edge, which is not a valley.
    2. The density peak **below** the candidate must sit at walking pace. A
       vehicle-only feed has interior valleys of its own; without this guard
       the boundary between gridlock and free flow is accepted as a mode split.
    """
    unit = SpeedUnit.parse(unit)
    x = np.asarray(pd.Series(mover_speeds).to_numpy(), dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    if len(x) < _MIN_MOVERS:
        return None
    try:
        from scipy.stats import gaussian_kde
    except ImportError:  # pragma: no cover - scipy is a core dependency
        return None

    lo = float(from_mps(_SEARCH_LO_MPS, unit))
    hi = float(from_mps(_SEARCH_HI_MPS, unit))
    walk_lo = float(from_mps(_WALK_MODE_LO_MPS, unit))
    walk_hi = float(from_mps(_WALK_MODE_HI_MPS, unit))

    grid = np.linspace(np.log(lo / 4.0), np.log(hi * 8.0), 1200)
    dens = gaussian_kde(np.log(x), bw_method=0.20)(grid)

    interior = np.arange(1, len(grid) - 1)
    is_min = ((dens[interior] < dens[interior - 1])
              & (dens[interior] < dens[interior + 1]))
    in_window = (grid[interior] >= np.log(lo)) & (grid[interior] <= np.log(hi))
    candidates = interior[is_min & in_window]
    if not len(candidates):
        return None

    def prominence(i: int) -> float:
        left = dens[:i].max(initial=0.0)
        right = dens[i + 1:].max(initial=0.0)
        return min(left, right) / dens[i] if dens[i] > 0 else 0.0

    def lower_hump(i: int) -> float:
        return float(np.exp(grid[int(np.argmax(dens[:i]))]))

    viable = [i for i in candidates if prominence(i) >= _MIN_PROMINENCE]
    walkers = [i for i in viable if walk_lo <= lower_hump(i) <= walk_hi]
    if not walkers:
        return None
    return float(np.exp(grid[max(walkers, key=prominence)]))
```

Add `suggest_mode_threshold` to the `from .modes import (...)` block and to `__all__` in `src/roadtraffic/__init__.py`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_modes.py -q && ruff check src/ tests/`
Expected: 9 passed, ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/roadtraffic/modes.py src/roadtraffic/__init__.py tests/test_modes.py
git commit -m "modes: density-valley threshold suggestion with walking-hump guard"
```

---

### Task 3: `classify_movers`

**Files:**
- Modify: `src/roadtraffic/modes.py` (append)
- Modify: `tests/test_modes.py` (append)

**Interfaces:**
- Consumes: `mover_features`, `suggest_mode_threshold`, `_percentile_label`, the mode constants
- Produces: `classify_movers(obs, *, threshold, percentile=85.0, min_intervals=3, min_distance_m=0.0, unit="mph") -> pd.DataFrame` — the `mover_features` table plus a `mode` column

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_modes.py`:

```python
def test_classify_labels_pedestrians_and_vehicles():
    obs = obs_frame({"walker": [1.3, 1.5, 1.4, 1.6],
                     "car": [3.0, 18.0, 20.0, 22.0]})
    m = rt.classify_movers(obs, threshold=3.0, unit="mps")
    assert m.loc["walker", "mode"] == "pedestrian"
    assert m.loc["car", "mode"] == "vehicle"


def test_a_congested_vehicle_is_still_a_vehicle():
    """It crawls for most of its trip but shows one free-flowing stretch, and
    the 85th percentile is chosen precisely so that stretch is visible."""
    obs = obs_frame({"stuck": [1.5, 1.6, 1.4, 1.5, 1.6, 1.5, 14.0, 15.0]})
    m = rt.classify_movers(obs, threshold=3.0, unit="mps")
    assert m.loc["stuck", "mode"] == "vehicle"


def test_unknown_comes_from_too_few_intervals():
    m = rt.classify_movers(obs_frame({"brief": [20.0, 21.0]}),
                           threshold=3.0, min_intervals=3, unit="mps")
    assert m.loc["brief", "mode"] == "unknown"


def test_unknown_comes_from_too_little_distance():
    m = rt.classify_movers(obs_frame({"short": [20.0] * 4}, distance_m=10.0),
                           threshold=3.0, min_distance_m=500.0, unit="mps")
    assert m.loc["short", "mode"] == "unknown"


def test_threshold_is_read_in_the_requested_unit():
    obs = obs_frame({"a": [1.4] * 5, "b": [12.0] * 5})
    mph = rt.classify_movers(obs, threshold=6.0, unit="mph")
    kph = rt.classify_movers(obs, threshold=6.0 * 1.609344, unit="kph")
    assert list(mph["mode"]) == list(kph["mode"])
    assert list(mph["mode"]) == ["pedestrian", "vehicle"]


def test_auto_threshold_raises_rather_than_guessing():
    rng = np.random.default_rng(2)
    obs = obs_frame({f"v{i}": [float(s)] * 4
                     for i, s in enumerate(rng.normal(12.0, 2.0, 60))})
    with pytest.raises(ValueError, match="walking"):
        rt.classify_movers(obs, threshold="auto", unit="mps")


def test_auto_threshold_works_when_there_are_walkers():
    rng = np.random.default_rng(0)
    specs = {f"p{i}": [float(v)] * 4
             for i, v in enumerate(rng.normal(1.4, 0.15, 90))}
    specs.update({f"v{i}": [float(v)] * 4
                  for i, v in enumerate(np.clip(rng.normal(12.0, 3.0, 200), 5.0, None))})
    m = rt.classify_movers(obs_frame(specs), threshold="auto", unit="mps")
    assert set(m["mode"]) == {"pedestrian", "vehicle"}
    assert (m["mode"] == "pedestrian").sum() == 90


def test_bad_threshold_string_raises():
    with pytest.raises(ValueError, match="'auto'"):
        rt.classify_movers(obs_frame({"a": [1.0] * 4}), threshold="fast", unit="mps")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_modes.py -q -k "classify or auto or unknown"`
Expected: FAIL — `AttributeError: ... has no attribute 'classify_movers'`

(The `-k` expression must be quoted; unquoted, the shell splits it on `or`.)

- [ ] **Step 3: Write the implementation**

Append to `src/roadtraffic/modes.py`:

```python
def classify_movers(obs, *, threshold, percentile: float = 85.0,
                    min_intervals: int = 3, min_distance_m: float = 0.0,
                    unit="mph") -> pd.DataFrame:
    """Label each mover ``pedestrian``, ``vehicle`` or ``unknown``.

    Parameters
    ----------
    obs : DataFrame
        As :func:`mover_features`.
    threshold : float or ``"auto"``
        Movers whose ``percentile`` speed is at or above this are vehicles.
        ``"auto"`` delegates to :func:`suggest_mode_threshold` and **raises**
        when it finds no walking population, rather than falling back to a
        default that would silently reshape the study.
    percentile, unit
        As :func:`mover_features`. ``threshold`` is read in ``unit``.
    min_intervals, min_distance_m
        Evidence floors. A mover below either is ``unknown``.

    Returns
    -------
    DataFrame
        :func:`mover_features`' columns plus ``mode``.

    Notes
    -----
    There is deliberately no ``require_quality`` parameter. Classification uses
    every interval, including ``quality=False`` ones. The quality screen rejects
    intervals whose displacement is small relative to GPS noise -- exactly what
    a walking mover produces -- so filtering on it would delete the evidence
    that identifies slow movers and push them into ``unknown``.

    ``unknown`` never arises from speed ambiguity, only from insufficient
    evidence. A congested vehicle is a ``vehicle``.
    """
    unit = SpeedUnit.parse(unit)
    feat = mover_features(obs, percentile=percentile, unit=unit)
    pct_col = f"speed_p{_percentile_label(percentile)}_{unit.value}"
    if not len(feat):
        feat["mode"] = pd.Series(dtype=object)
        return feat

    if isinstance(threshold, str):
        if threshold != "auto":
            raise ValueError(
                f"threshold must be a number or 'auto', got {threshold!r}.")
        auto = suggest_mode_threshold(feat[pct_col], unit=unit)
        if auto is None:
            raise ValueError(
                "classify_movers(threshold='auto') found no walking-speed "
                "population to split off, so there is no defensible cut. "
                "Either the feed is all vehicles (nothing to exclude), or the "
                "two populations overlap too much to separate on speed. "
                "Inspect the distribution of "
                f"mover_features(...)['{pct_col}'] and pass an explicit "
                "threshold if you still want to screen.")
        threshold = auto

    speed = feat[pct_col].to_numpy(dtype=float)
    n_iv = feat["n_intervals"].to_numpy()
    dist = feat["distance_m"].to_numpy(dtype=float)

    insufficient = n_iv < min_intervals
    if min_distance_m > 0:
        # Negated >= rather than <, so a NaN distance (no distance_m column in
        # the source) counts as insufficient instead of silently passing.
        insufficient = insufficient | ~(dist >= min_distance_m)

    feat["mode"] = np.where(
        insufficient, MODE_UNKNOWN,
        np.where(speed >= float(threshold), MODE_VEHICLE, MODE_PEDESTRIAN))
    return feat
```

Add `classify_movers` to the `from .modes import (...)` block and `__all__`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_modes.py -q && ruff check src/ tests/`
Expected: 17 passed, ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/roadtraffic/modes.py src/roadtraffic/__init__.py tests/test_modes.py
git commit -m "modes: classify movers as pedestrian, vehicle or unknown"
```

---

### Task 4: `filter_by_mode` and the core invariant

**Files:**
- Modify: `src/roadtraffic/modes.py` (append)
- Modify: `tests/test_modes.py` (append)

**Interfaces:**
- Consumes: `_require_traj_columns`, `MODE_VEHICLE`
- Produces: `filter_by_mode(obs, movers, *, keep=("vehicle",)) -> pd.DataFrame`

The first test below is the property the entire method rests on. If it ever fails, the screen has become a speed filter and the congestion finding is gone.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_modes.py`:

```python
def test_filter_keeps_every_observation_of_a_kept_mover():
    """THE invariant. A vehicle that crawls through congestion keeps its slow
    rows. That is the whole difference from filter_by_speed, which would delete
    them and take the congestion finding with them."""
    obs = obs_frame({"stuck": [1.5, 1.4, 1.5, 1.6, 14.0, 15.0],
                     "walker": [1.3, 1.4, 1.5, 1.4]})
    movers = rt.classify_movers(obs, threshold=3.0, unit="mps")
    out = rt.filter_by_mode(obs, movers)
    assert set(out["traj_id"]) == {"stuck"}
    assert len(out[out["speed_mps"] < 2.0]) == 4


def test_filter_excludes_unknown_by_default_but_can_include_it():
    obs = obs_frame({"car": [20.0] * 4, "brief": [20.0, 21.0]})
    movers = rt.classify_movers(obs, threshold=3.0, min_intervals=3, unit="mps")
    assert set(rt.filter_by_mode(obs, movers)["traj_id"]) == {"car"}
    both = rt.filter_by_mode(obs, movers, keep=("vehicle", "unknown"))
    assert set(both["traj_id"]) == {"car", "brief"}


def test_filter_warns_instead_of_exiting_when_everything_is_removed():
    obs = obs_frame({"walker": [1.4] * 4})
    movers = rt.classify_movers(obs, threshold=3.0, unit="mps")
    with pytest.warns(UserWarning, match="removed every"):
        out = rt.filter_by_mode(obs, movers)
    assert len(out) == 0


def test_filter_accepts_a_single_mode_string():
    obs = obs_frame({"car": [20.0] * 4, "walker": [1.4] * 4})
    movers = rt.classify_movers(obs, threshold=3.0, unit="mps")
    assert set(rt.filter_by_mode(obs, movers, keep="pedestrian")["traj_id"]) \
        == {"walker"}


def test_filter_requires_a_classification_table():
    obs = obs_frame({"a": [1.0] * 4})
    with pytest.raises(ValueError, match="classify_movers"):
        rt.filter_by_mode(obs, pd.DataFrame({"n_intervals": [4]}))


def test_verdicts_match_between_intervals_and_edge_observations():
    """A long-format frame must classify identically to the interval frame it
    was expanded from -- the dedup path in mover_features."""
    obs = obs_frame({"stuck": [1.5, 1.4, 14.0], "walker": [1.3, 1.4, 1.5, 1.4]})
    slow = obs[obs["speed_mps"] < 2.0]
    long_form = pd.concat([obs] + [slow] * 3, ignore_index=True)
    a = rt.classify_movers(obs, threshold=3.0, min_intervals=3, unit="mps")
    b = rt.classify_movers(long_form, threshold=3.0, min_intervals=3, unit="mps")
    assert a["mode"].to_dict() == b["mode"].to_dict()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_modes.py -q -k filter`
Expected: FAIL — `AttributeError: ... has no attribute 'filter_by_mode'`

- [ ] **Step 3: Write the implementation**

Append to `src/roadtraffic/modes.py`:

```python
def filter_by_mode(obs, movers, *, keep=(MODE_VEHICLE,)) -> pd.DataFrame:
    """Keep the observations of movers whose mode is in ``keep``.

    Parameters
    ----------
    obs : DataFrame
        Observations to filter; needs ``traj_id``. Typically ``derive_speeds``'
        ``edge_observations``, since that is what the aggregators consume.
    movers : DataFrame
        The table returned by :func:`classify_movers`, indexed by ``traj_id``.
    keep : str or iterable of str
        Modes to retain. The default keeps vehicles only, so ``unknown`` is
        excluded unless asked for.

    Returns
    -------
    DataFrame
        Filtered copy with a reset index. Every retained mover keeps **all** of
        its observations, including its slowest -- filtering the slow ones out
        is what this module exists to avoid.

    Warns
    -----
    UserWarning
        When no mover survives. A library returns the empty frame and lets the
        caller decide; it does not exit the process.
    """
    _require_traj_columns(obs, "filter_by_mode", need_speed=False)
    if "mode" not in getattr(movers, "columns", ()):
        raise ValueError(
            "filter_by_mode needs the DataFrame returned by classify_movers "
            "(it must carry a 'mode' column).")
    wanted = {keep} if isinstance(keep, str) else set(keep)
    kept_ids = set(movers.index[movers["mode"].isin(wanted)])
    out = obs[obs["traj_id"].isin(kept_ids)].reset_index(drop=True)
    if not len(out):
        warnings.warn(
            f"filter_by_mode removed every observation: no mover matched "
            f"keep={sorted(wanted)}. Lower the threshold, or widen keep=.",
            UserWarning, stacklevel=2)
    return out
```

Add `filter_by_mode` to the `from .modes import (...)` block and `__all__`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_modes.py -q && ruff check src/ tests/`
Expected: 23 passed, ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/roadtraffic/modes.py src/roadtraffic/__init__.py tests/test_modes.py
git commit -m "modes: filter observations by mover verdict"
```

---

### Task 5: End-to-end premise test

**Files:**
- Modify: `tests/conftest.py`
- Modify: `tests/test_pipeline_e2e.py`

**Interfaces:**
- Consumes: `classify_movers`, `filter_by_mode`, `straight_net` and `make_points_csv` fixtures
- Produces: `track_along_road(n, *, speed_mps, ...)` and `walk_along_road(n, ...)` in `tests/conftest.py`

This validates the premise rather than the plumbing: screening must move the answer *toward* the vehicles-only truth.

- [ ] **Step 1: Add the fixtures helper**

Append to `tests/conftest.py`, beside the existing `drive_along_road`:

```python
def track_along_road(n=10, *, speed_mps, start_lon=0.0002, lat=1e-5,
                     t0="2026-06-01 08:00:00", dt_s=10, traj=None):
    """Rows for a mover travelling east along the equator road at a set speed.

    Spacing is computed from the speed so the planted value is exact: each fix
    sits ``speed_mps * dt_s`` metres further along, and ~111319.5 m is one
    degree of longitude at the equator.
    """
    t0 = pd.Timestamp(t0)
    dlon = speed_mps * dt_s / 111319.5
    rows = []
    for k in range(n):
        row = {"lon": start_lon + dlon * k, "lat": lat,
               "time": (t0 + pd.Timedelta(seconds=dt_s * k)).isoformat()}
        if traj is not None:
            row["id"] = traj
        rows.append(row)
    return rows


def walk_along_road(n=25, *, speed_mps=1.4, dt_s=15, **kw):
    """A mover on foot: ~1.4 m/s is an ordinary walking pace."""
    return track_along_road(n, speed_mps=speed_mps, dt_s=dt_s, **kw)
```

- [ ] **Step 2: Write the failing test**

Append to `tests/test_pipeline_e2e.py`:

```python
def test_mode_screening_recovers_vehicle_only_speeds(straight_net, make_points_csv):
    """Walkers on the same road drag the network speed down; screening must
    move the answer back toward the vehicles-only truth."""
    from conftest import track_along_road, walk_along_road

    veh_rows, ped_rows = [], []
    for i in range(6):
        veh_rows += track_along_road(8, speed_mps=12.0, traj=f"veh{i}",
                                     t0=f"2026-06-01 0{i}:00:00")
    for i in range(6):
        ped_rows += walk_along_road(25, traj=f"ped{i}",
                                    t0=f"2026-06-01 1{i}:00:00")

    def intervals(rows, name):
        path = make_points_csv(rows, name=name)
        pts = rt.load_points(path, id_col="id")
        matched = rt.HMMMatcher(straight_net, max_dist=60).match(pts)
        return rt.derive_speeds(straight_net, matched, pts)["intervals"]

    truth = intervals(veh_rows, "veh.csv")["speed_mps"].median()
    mixed = intervals(veh_rows + ped_rows, "mixed.csv")
    contaminated = mixed["speed_mps"].median()

    movers = rt.classify_movers(mixed, threshold=3.0, unit="mps")
    screened = rt.filter_by_mode(mixed, movers)["speed_mps"].median()

    assert contaminated < truth * 0.9          # contamination is real
    assert abs(screened - truth) < abs(contaminated - truth)
    assert screened == pytest.approx(truth, rel=0.10)
```

- [ ] **Step 3: Run the test**

Run: `python -m pytest tests/test_pipeline_e2e.py -q -k mode_screening`
Expected: PASS.

**This test does not fail-first, and that is expected.** It exercises functions Tasks 1-4 already built, so there is no red phase to observe — it is an integration check on finished units, not the driver of new code. Do not manufacture a failure to satisfy the TDD rhythm.

What it *can* catch, and what to do:

- `contaminated < truth * 0.9` fails → the walkers are not reaching the road. Check `max_dist=60` and confirm the 25 walking fixes at ~21 m spacing stay inside the 1113 m road (start 0.0002 deg ≈ 22 m in, so they end around 526 m).
- `screened == approx(truth, rel=0.10)` fails → check that `threshold=3.0` with `unit="mps"` sits between the planted 1.4 m/s and 12.0 m/s, and that `classify_movers` is being handed the **intervals** frame.
- Both assertions passing while the middle one fails is impossible; if you see it, the fixtures are not deterministic and that is the bug.

- [ ] **Step 4: Run the full suite**

Run: `python -m pytest tests/ -q && ruff check src/ tests/`
Expected: 257 passed (233 existing + 23 from Tasks 1-4 + 1 here), ruff clean.

- [ ] **Step 5: Commit**

```bash
git add tests/conftest.py tests/test_pipeline_e2e.py
git commit -m "tests: end-to-end check that mode screening recovers vehicle-only speeds"
```

---

### Task 6: Documentation and version bump

**Files:**
- Modify: `src/roadtraffic/__init__.py` (module docstring)
- Modify: `docs/api.md`
- Modify: `docs/methodology.md`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: the four public functions from Tasks 1-4
- Produces: no code

- [ ] **Step 1: Add the four entries to the `__init__.py` module docstring**

Insert after the `filter_trajectory_speed` line, matching the existing alignment:

```
mover_features        : per-mover evidence table (speed percentile, sample
                        count, distance) from matched or derived speeds.
suggest_mode_threshold : density-valley speed separating walkers from drivers,
                        or None when no walking population exists.
classify_movers       : label movers pedestrian / vehicle / unknown, judging
                        the whole trajectory rather than each observation.
filter_by_mode        : keep the observations of movers with a given mode --
                        every retained mover keeps its slow rows.
```

- [ ] **Step 2: Add a `## Mode screening` section to `docs/api.md`**

Place it immediately after the `## Cleaning` section (before `## Aggregation & peaks`), using the same `### \`signature\`` + prose format as the surrounding entries. Document all four functions with their full signatures as implemented, and open the section with the reason it exists:

> Mode is a property of the mover, not the fix. A minimum-speed filter on observations cannot distinguish a pedestrian from a vehicle crawling through congestion, so it deletes both. These functions judge whole trajectories and apply the verdict to all of a mover's observations.

- [ ] **Step 3: Add a mode-screening section to `docs/methodology.md`**

Cover, in this order: why per-mover rather than per-observation; why the 85th percentile (a mean or median misclassifies a mover that spent most of its trip congested; a maximum is set by a single GPS jump); how to choose a threshold from the valley; and the measured comparison, copied from the spec:

| strategy | Δ peak speed | per-edge MAE |
|---|---|---|
| no filter (contaminated) | −1.2 mph | 6.0 mph |
| drop observations < 12 mph | +5.0 mph | 3.1 mph |
| `require_quality` only | −2.1 mph | 3.9 mph |
| mover p85 ≥ 6 mph | +0.2 mph | 0.6 mph |

Close with the gridlock limitation and its **upward** bias direction, and the procedural mitigation (run screened and unscreened, report the gap).

- [ ] **Step 4: Bump the version**

Set `__version__ = "0.5.0"` in `src/roadtraffic/__init__.py` and `version = "0.5.0"` in `pyproject.toml`. Confirm they match:

```bash
grep -n '__version__' src/roadtraffic/__init__.py && grep -n '^version' pyproject.toml
```

- [ ] **Step 5: Verify and commit**

Run: `python -m pytest tests/ -q && ruff check src/ tests/ scripts/`

```bash
git add src/roadtraffic/__init__.py docs/api.md docs/methodology.md pyproject.toml
git commit -m "docs: mode screening API reference and methodology; bump to 0.5.0"
```

---

### Task 7: Rewire `scripts/mover_screen.py` onto the API

**Files:**
- Modify: `scripts/mover_screen.py`

**Interfaces:**
- Consumes: `mover_features`, `suggest_mode_threshold`, `classify_movers`, `filter_by_mode`
- Produces: no new API

- [ ] **Step 1: Delete the duplicated logic**

Remove `per_mover_features` and `suggest_threshold` from the script. Keep `histogram` — ASCII rendering is presentation and belongs here, not in the library.

- [ ] **Step 2: Rewrite the body to call the package**

The script keeps its CLI, its histogram, and its `--out-points` / `--out-movers` exports. Replace the computation with:

```python
matched = rt.HMMMatcher(net, max_dist=args.max_dist).match(pts)
acc = args.accuracy_col if args.accuracy_col in pts.df.columns else None
derived = rt.derive_speeds(net, matched, pts, pos_accuracy_col=acc,
                           min_baseline_m=args.min_baseline)
iv = derived["intervals"]
if not len(iv):
    raise SystemExit("No speed intervals could be derived; check --id-col.")

feat = rt.mover_features(iv, percentile=args.percentile, unit=unit)
pct_col = f"speed_p{args.percentile:g}_{unit.value}"
histogram(feat[pct_col].to_numpy(float), f"p{args.percentile:g}",
          focus_hi=2.5 * float(from_mps(6.0, unit)))
sugg = rt.suggest_mode_threshold(feat[pct_col], unit=unit)
```

and, in the apply branch:

```python
movers = rt.classify_movers(iv, threshold=args.threshold,
                            percentile=args.percentile,
                            min_intervals=args.min_intervals, unit=unit)
keep = ("vehicle", "unknown") if args.keep_unclassified else ("vehicle",)
kept_ids = set(movers.index[movers["mode"].isin(keep)])
```

Note the behaviour change to state in the commit message: `--threshold` now also accepts the string `auto`, since `classify_movers` handles it. Change the argparse entry to `type=str` and convert to float when it is not `"auto"`.

- [ ] **Step 3: Verify against the recorded results**

The script must still reproduce the numbers it produced before the rewire. Regenerate the fixtures and check:

```bash
python scripts/mover_screen.py --network mixed_network.geojson \
    --points stress_points.csv --id-col device_id --tz America/New_York
```

Expected: suggests **6.4 mph**, and the histogram shows the walking hump at 2.8-5.6 with the valley at 5.6-8.4. Then:

```bash
python scripts/mover_screen.py --network mixed_network.geojson \
    --points vehicles_only.csv --id-col device_id --tz America/New_York
```

Expected: **no threshold suggested** — this is the negative control, and a suggestion here is a regression in the walking-hump guard.

If the mixed fixtures are absent, regenerate them with the generators in the session scratchpad (`make_mixed_data.py`, `stress_jam.py`) or plant an equivalent set; do not skip this step, since it is the only check that the rewire preserved behaviour.

- [ ] **Step 4: Run lint and the suite**

Run: `ruff check scripts/ src/ tests/ && python -m pytest tests/ -q`

- [ ] **Step 5: Commit**

```bash
git add scripts/mover_screen.py
git commit -m "scripts: rewire mover_screen onto roadtraffic.modes"
```

---

### Task 8: `--mode-threshold` in the customer report

**Files:**
- Modify: `scripts/customer_report.py` (`run_pipeline`, `parse_args`)

**Interfaces:**
- Consumes: `classify_movers`, `filter_by_mode`
- Produces: `R["movers"]` (the classification table) and `R["mode_threshold"]` (float or `None`) in the dict `run_pipeline` returns, both read by Task 9

- [ ] **Step 1: Add the arguments**

In `parse_args`, beside `--require-quality`:

```python
p.add_argument("--mode-threshold", default=None,
               help="exclude movers whose p85 speed is below this (in --unit), "
                    "or 'auto' to pick the density valley. Off by default: "
                    "screening changes the study, so it is never implicit.")
p.add_argument("--keep-unknown", action="store_true",
               help="keep movers with too little data to classify "
                    "(default: excluded along with pedestrians)")
```

- [ ] **Step 2: Screen between derive and clean**

In `run_pipeline`, after the `[4/7]` derive block and before `[5/7] cleaning`. Renumber every existing banner from `[N/7]` to `[N/8]` and add:

`classify_movers` does not report back which threshold `"auto"` resolved to, and the deck section in Task 9 needs that number to draw its threshold line. So resolve it in the report *before* classifying, and pass a plain float in both cases:

```python
movers, mode_threshold = None, None
if args.mode_threshold is not None:
    print("[5/8] mode screening", file=sys.stderr)
    if args.mode_threshold == "auto":
        feat = rt.mover_features(intervals, unit=args.unit)
        thr = rt.suggest_mode_threshold(feat[f"speed_p85_{args.unit}"],
                                        unit=args.unit)
        if thr is None:
            raise SystemExit(
                "--mode-threshold auto found no walking-speed population to "
                "split off. Either the feed is all vehicles (nothing to "
                "exclude), or the populations overlap too much to separate on "
                "speed. Inspect it with scripts/mover_screen.py, then pass an "
                "explicit number.")
    else:
        thr = float(args.mode_threshold)
    mode_threshold = float(thr)

    movers = rt.classify_movers(intervals, threshold=thr, unit=args.unit)
    keep = ("vehicle", "unknown") if args.keep_unknown else ("vehicle",)
    n_before, mov_before = len(obs), int(intervals["traj_id"].nunique())
    obs = rt.filter_by_mode(obs, movers, keep=keep)
    intervals = rt.filter_by_mode(intervals, movers, keep=keep)
    kept = int(movers["mode"].isin(keep).sum())
    notes.append(
        f"Mode screening excluded {mov_before - kept:,} of {mov_before:,} "
        f"movers and {n_before - len(obs):,} of {n_before:,} observations as "
        "pedestrians or other non-vehicle movers. Mode is judged per mover on "
        "its 85th-percentile speed, so a vehicle kept by the screen retains "
        "all of its slow observations.")
    notes.append(
        "WARNING -- mode screening biases speeds UPWARD when it is wrong. A "
        "vehicle whose entire track is gridlocked never shows a fast stretch "
        "and is excluded with the pedestrians. Re-run without "
        "--mode-threshold and compare peak speeds; the gap is the uncertainty.")
```

- [ ] **Step 3: Return the new keys**

Add `"movers": movers, "mode_threshold": mode_threshold` to the dict `run_pipeline` returns.

- [ ] **Step 4: Verify both paths**

```bash
python scripts/customer_report.py --network mixed_network.geojson \
    --points stress_points.csv --id-col device_id --time-col timestamp \
    --lon-col longitude --lat-col latitude --tz America/New_York \
    --min-baseline 150 --mode-threshold 6.4 --out screened.pdf
```

Expected: runs clean, prints 8 steps, and reports at least two data notes about the screening. Then run the same command **without** `--mode-threshold` and confirm the report still builds and reports no screening notes.

Run: `ruff check scripts/ && python -m pytest tests/ -q`

- [ ] **Step 5: Commit**

```bash
git add scripts/customer_report.py
git commit -m "report: opt-in --mode-threshold screening with bias caveat"
```

---

### Task 9: "What the feed is made of" deck section

**Files:**
- Modify: `scripts/customer_report.py` (`SECTIONS`, `build_figures`, `build_tables`, `build_tiles`, `_section_ctx`)

**Interfaces:**
- Consumes: `R["movers"]`, `R["mode_threshold"]` from Task 8
- Produces: figure key `"modes"`, table key `"modes"`, tile key `"modes"`

- [ ] **Step 1: Load the dataviz skill first**

This section introduces the file's first **categorical** palette — three named modes. The existing constants are sequential (`SEQ_BLUE`) and diverging (`DIV_LOW`/`DIV_MID`/`DIV_HIGH`), and neither rule applies. Invoke the `dataviz` skill and follow it before writing any chart code.

- [ ] **Step 2: Add the section entry**

Insert into `SECTIONS` as the second entry, after `coverage`, and renumber the eyebrow prefixes of every later entry (`02 —` … `08 —`):

```python
("modes", "02 — Feed composition", "What the feed is made of",
 "Movers are classified by their own 85th-percentile speed, not by individual "
 "observations: a pedestrian is slow for a whole track, while a congested "
 "vehicle is slow here and free-flowing elsewhere. A vehicle kept by the "
 "screen keeps every one of its slow observations."),
```

- [ ] **Step 3: Add the figure**

A two-panel figure: mover counts by mode (categorical bars), and the per-mover p85 histogram with a vertical line at the threshold. The histogram is the single most useful artifact for explaining the method to a customer, so it must be the larger panel.

Define the categorical palette near the existing constants at `scripts/customer_report.py:82-103`. Take the three hex values from the categorical palette in the dataviz skill's `references/palette.md` — do **not** invent them here, and do **not** reuse `DIV_LOW`/`DIV_HIGH`, which encode a diverging scale and would imply an ordering these three categories do not have.

The values must satisfy: three distinct hues (not a lightness ramp), distinguishable under the common colour-vision deficiencies, and legible on the deck's near-white surface — the same bar that caught the invisible diverging midpoint earlier in this file's history.

```python
# Categorical: three unordered classes. Distinct hues, not a ramp.
# Hex values from the dataviz skill's categorical palette.
CAT_1, CAT_2, CAT_3 = "...", "...", "..."   # fill from references/palette.md
MODE_COLORS = {"vehicle": CAT_1, "pedestrian": CAT_2, "unknown": CAT_3}
MODE_ORDER = ("vehicle", "pedestrian", "unknown")
```

```python
def fig_modes(plt, movers, threshold, unit, percentile=85.0):
    """Feed composition, and the distribution the screen actually cut on."""
    if movers is None or not len(movers):
        return None
    pct_col = f"speed_p{percentile:g}_{unit}"
    if pct_col not in movers.columns:
        return None
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9, 4.0),
                                   gridspec_kw={"width_ratios": [1, 2]})

    counts = [int((movers["mode"] == m).sum()) for m in MODE_ORDER]
    present = [(m, c) for m, c in zip(MODE_ORDER, counts) if c]
    ax0.barh([m for m, _ in present], [c for _, c in present],
             color=[MODE_COLORS[m] for m, _ in present])
    ax0.set_xlabel("Movers")
    ax0.invert_yaxis()
    for i, (_, c) in enumerate(present):
        ax0.text(c, i, f" {c:,}", va="center", fontsize=8)
    ax0.spines[["top", "right"]].set_visible(False)

    speeds = movers[pct_col].to_numpy(dtype=float)
    speeds = speeds[np.isfinite(speeds)]
    hi = float(np.percentile(speeds, 98)) if len(speeds) else 1.0
    ax1.hist(speeds, bins=np.linspace(0, max(hi, 1.0), 40),
             color=SEQ_BLUE[-2], edgecolor="none")
    if threshold is not None:
        ax1.axvline(threshold, color=INK, lw=1.4, ls="--")
        ax1.annotate(f"screen at {threshold:.1f} {unit}",
                     xy=(threshold, ax1.get_ylim()[1]),
                     xytext=(4, -10), textcoords="offset points",
                     fontsize=8, color=INK, ha="left", va="top")
    ax1.set_xlabel(f"Per-mover {percentile:g}th-percentile speed ({unit})")
    ax1.set_ylabel("Movers")
    ax1.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig
```

`INK` and `SEQ_BLUE` already exist in the file; reuse them rather than introducing new greys. Register the figure in `build_figures`, guarded on `R["movers"]` being present, so a report run without screening simply omits the section — the same pattern `congestion` already uses:

```python
if R.get("movers") is not None:
    f = fig_modes(plt, R["movers"], R["mode_threshold"], args.unit)
    if f:
        figs["modes"] = f
```

- [ ] **Step 4: Add the table and tiles**

In `build_tables`:

```python
if R.get("movers") is not None:
    mv = R["movers"]
    total = int(mv["n_intervals"].sum()) or 1
    rows = []
    for m in MODE_ORDER:
        sub = mv[mv["mode"] == m]
        if not len(sub):
            continue
        n_iv = int(sub["n_intervals"].sum())
        rows.append((m.capitalize(), f"{len(sub):,}", f"{n_iv:,}",
                     f"{100.0 * n_iv / total:.0f}%"))
    out["modes"] = ("Feed composition by mover mode",
                    ["Mode", "Movers", "Intervals", "Share of intervals"], rows)
```

In `build_tiles`:

```python
if R.get("movers") is not None:
    mv = R["movers"]
    kept = int((mv["mode"] == "vehicle").sum())
    thr = R["mode_threshold"]
    out["modes"] = [
        (f"{kept:,}", "Movers kept", "classified as vehicles"),
        (f"{len(mv) - kept:,}", "Movers excluded", "pedestrian or unclassified"),
        (f"{thr:.1f}" if thr is not None else "--", f"Screen ({unit})",
         "per-mover 85th percentile"),
        (f"{100.0 * kept / max(len(mv), 1):.0f}%", "Of all movers", "")]
```

- [ ] **Step 5: Verify by looking at the output**

Generate all three formats and **open them**:

```bash
python scripts/customer_report.py ... --mode-threshold 6.4 --out modes.pdf
python scripts/customer_report.py ... --mode-threshold 6.4 --out modes.pptx
python scripts/customer_report.py ... --mode-threshold 6.4 --format png --out modes_png/
```

Check, by inspecting the rendered images rather than by the script exiting 0: the threshold line lands in the valley; the legend does not collide with the bars or title; the picture fits inside the PPTX slide bounds; and the section is absent entirely when `--mode-threshold` is omitted. Every layout defect in this file's history was found this way and by no other means.

- [ ] **Step 6: Commit**

```bash
git add scripts/customer_report.py
git commit -m "report: feed-composition section showing modes and the screening cut"
```

---

## Verification

Before opening a PR:

```bash
python -m pytest tests/ -q          # expect 257 passed
ruff check src/ tests/ scripts/     # expect clean
grep -n '__version__' src/roadtraffic/__init__.py   # expect 0.5.0
```

Plus the two behavioural checks that no unit test covers:

1. `scripts/mover_screen.py` on vehicles-only data suggests **nothing** (the negative control).
2. The report's mode section renders correctly in PDF **and** PPTX, verified by looking at the output.
