"""Temporal aggregation, peak/off-peak detection, and edge speed assignment.

Aggregation bins observations by hour-of-day (0-23) or by an N-hour block, then
summarises speed per bin using either the mean (with std and SEM) or the median
(with IQR), selectable by the caller.

Statistical notes
------------------
- The arithmetic mean and standard deviation assume roughly symmetric,
  outlier-light data; pair with ``filter_by_speed(mad_outliers=True)``.
- The standard error of the mean, SEM = s / sqrt(n), quantifies uncertainty in
  the *mean* speed of a bin; a 95% normal-approx CI is mean +/- 1.96 * SEM.
  Bins with a single observation report NaN std/SEM/CI -- one observation
  carries no information about spread.
- The median and interquartile range (IQR = Q3 - Q1) are robust alternatives
  recommended when speed distributions are skewed (common in congested flow).
- Bins with few observations are statistically weak. ``min_samples`` lets you
  suppress under-sampled bins; ``n`` is always reported so users can judge.
- Rows with ``edge_id == -1`` (points that snapped to nothing) are excluded.
- Network-wide statistics deduplicate ``derive_speeds`` edge observations on
  ``interval_id`` when present, so an interval that traversed many edges still
  counts as one measurement.
"""
from __future__ import annotations

import warnings
from collections.abc import Iterable
from functools import reduce

import numpy as np
import pandas as pd

from .units import SpeedUnit, _usable_speed, from_mps


def _require_speed_mps(df: pd.DataFrame, fn_name: str) -> None:
    """Raise a clear, actionable error if ``df`` has no ``speed_mps`` column.

    Shared by every function in this module (and ``cleaning.py``) that needs
    a speed to operate on, so points loaded without a speed column
    (position+time only -- see :func:`redlight.points.load_points`) get a
    consistent, specific error message pointing at the fix instead of a
    generic pandas ``KeyError`` deep in a groupby.
    """
    if "speed_mps" not in df.columns:
        raise ValueError(
            f"{fn_name} needs a 'speed_mps' column. If points were loaded "
            "without a speed column, run a matcher then "
            "redlight.speeds.derive_speeds first and pass its "
            "'edge_observations' frame here."
        )


def _drop_unmatched(df: pd.DataFrame) -> pd.DataFrame:
    """Remove edge_id == -1 sentinel rows (points that snapped to nothing)."""
    if "edge_id" in df.columns:
        return df[df["edge_id"] != -1]
    return df


# Day-of-week filtering. Weekday numbering follows pandas' ``dt.dayofweek``:
# Monday == 0 ... Sunday == 6. A resolved selector is a ``frozenset[int]`` of
# those numbers, or ``None`` meaning "no filter -- every day" (so the whole
# feature is opt-in and back-compatible: ``days=None`` changes nothing).
_DAY_PRESETS: dict[str, frozenset | None] = {
    "all": None,
    "weekday": frozenset({0, 1, 2, 3, 4}),
    "weekdays": frozenset({0, 1, 2, 3, 4}),
    "weekend": frozenset({5, 6}),
    "weekends": frozenset({5, 6}),
}
# Every day-name spelling accepted, mapped to its ``dt.dayofweek`` number.
_DAY_NAMES: dict[str, int] = {
    "mon": 0, "monday": 0,
    "tue": 1, "tues": 1, "tuesday": 1,
    "wed": 2, "weds": 2, "wednesday": 2,
    "thu": 3, "thur": 3, "thurs": 3, "thursday": 3,
    "fri": 4, "friday": 4,
    "sat": 5, "saturday": 5,
    "sun": 6, "sunday": 6,
}


def _day_tokens(token) -> frozenset | None:
    """Resolve one day token to a set of weekday ints, or ``None`` for 'all'.

    A token is a preset (``"weekday"``/``"weekend"``/``"all"``), a day name
    (``"Mon"``/``"Monday"``/...), or a weekday number 0-6 (Mon=0), in any case
    and with surrounding whitespace ignored.
    """
    k = str(token).strip().lower()
    if k in _DAY_PRESETS:
        return _DAY_PRESETS[k]           # None for 'all', else a frozenset
    if k in _DAY_NAMES:
        return frozenset({_DAY_NAMES[k]})
    try:
        di = int(k)
    except ValueError:
        raise ValueError(
            f"Unrecognised day {token!r}. Use a weekday number 0-6 (Mon=0), a "
            "day name ('Mon'/'Monday'), or a preset "
            "('weekday'/'weekend'/'all')."
        ) from None
    if not 0 <= di <= 6:
        raise ValueError(f"day {di} out of range; use 0=Mon .. 6=Sun.")
    return frozenset({di})


def _resolve_days(days) -> frozenset | None:
    """Normalise a ``days=`` argument to a ``frozenset[int]`` or ``None``.

    ``None`` (the default everywhere) means no day filter. A bare scalar is one
    token; anything iterable is a union of tokens. If any token is ``"all"``
    the result is ``None`` (no filter), since including every day is the same
    as not filtering. See :func:`_day_tokens` for the token grammar.

    Scalar-vs-iterable is decided by *trying* to iterate rather than by an
    ``isinstance(days, int)`` test, because numpy integers are not Python
    ``int`` subclasses: ``days=np.int64(0)`` -- what you get straight out of
    ``df["time"].dt.dayofweek.unique()[0]`` -- would otherwise be sent down
    the iterate-it path and die as "numpy.int64 object is not iterable",
    an error naming neither ``days`` nor the real problem.
    """
    if days is None:
        return None
    if isinstance(days, str):
        tokens = [days]
    else:
        try:
            tokens = list(days)
        except TypeError:          # a non-iterable scalar (int, np.int64, ...)
            tokens = [days]
    result: set[int] = set()
    for tok in tokens:
        got = _day_tokens(tok)
        if got is None:                  # 'all' short-circuits to "no filter"
            return None
        result |= set(got)
    if not result:
        raise ValueError("days= resolved to no days; pass at least one day.")
    return frozenset(result)


def _filter_days(df: pd.DataFrame, days: frozenset | None) -> pd.DataFrame:
    """Keep only rows whose ``time`` falls on a weekday in ``days`` (or all)."""
    if days is None:
        return df
    dow = pd.to_datetime(df["time"]).dt.dayofweek.to_numpy()
    return df[np.isin(dow, list(days))]


def _prepare(matched: pd.DataFrame, *, days=None,
             require_quality: bool = False,
             fn_name: str = "aggregate") -> pd.DataFrame:
    """The cleaning prologue every speed consumer in this module shares.

    Drops unmatched rows and missing speeds, applies the ``days=`` weekday
    filter, and -- when asked -- keeps only observations ``derive_speeds``
    flagged ``quality=True``. Kept in one place so the three entry points
    (:func:`aggregate_speeds`, :func:`assign_speeds`,
    :func:`assign_segment_speeds`) cannot drift apart on what "usable
    observation" means.
    """
    df = _drop_unmatched(matched)
    df = df[~df["speed_mps"].isna()]
    df = _filter_days(df, _resolve_days(days))
    if require_quality:
        if "quality" not in df.columns:
            raise ValueError(
                f"{fn_name}: require_quality=True needs a 'quality' column, "
                "which derive_speeds emits. A frame straight from a matcher "
                "has no per-observation quality assessment to filter on."
            )
        df = df[df["quality"].astype(bool)]
    return df


# The columns that identify the measurement an ``interval_id`` names. In
# ``derive_speeds``' edge_observations the rows sharing an interval_id are the
# same measurement attributed to several edges, so they agree on all of these
# and differ only in ``edge_id``. Disagreement therefore means two genuinely
# different intervals have been given the same id.
#
# ``traj_id`` earns its place: two runs over separately-recorded trajectories
# can produce intervals that coincide exactly in time and speed, and those are
# still two independent measurements that the dedup would collapse into one.
_INTERVAL_IDENTITY = ("interval_id", "traj_id", "time", "speed_mps")


def _require_unique_intervals(df: pd.DataFrame, fn_name: str) -> None:
    """Raise if one ``interval_id`` names more than one distinct measurement.

    ``derive_speeds`` numbers intervals from 0 on every call, so concatenating
    the output of two runs silently makes distinct intervals collide. The
    network-wide dedup would then keep one row per colliding group and discard
    the rest -- roughly half the data for two equal-sized runs -- with no error
    and no warning. Cheap to detect: dedup on the identity columns and see
    whether any id survives more than once.
    """
    cols = [c for c in _INTERVAL_IDENTITY if c in df.columns]
    if len(cols) < 2:            # nothing to cross-check the id against
        return
    distinct = df[cols].drop_duplicates()
    collided = distinct["interval_id"].duplicated()
    if not collided.any():
        return
    ids = distinct.loc[collided, "interval_id"].unique()
    raise ValueError(
        f"{fn_name}: {len(ids)} interval_id value(s) name more than one "
        f"distinct measurement (e.g. {ids[0]!r}), so this frame concatenates "
        "the output of more than one derive_speeds call -- each numbers its "
        "intervals from 0. Deduplicating on interval_id would keep one row "
        "per colliding group and silently drop the rest. Fix by one of: "
        "(1) run derive_speeds once over all trajectories; (2) offset each "
        "run before concatenating, via derive_speeds(..., "
        "interval_id_start=...) or df['interval_id'] += n; (3) pass "
        "dedup_intervals=False if these ids are not interval identifiers."
    )


def aggregate_speeds(
    matched: pd.DataFrame,
    *,
    block_hours: int = 1,
    statistic: str = "mean",
    output_unit="mph",
    by_edge: bool = False,
    min_samples: int = 1,
    dedup_intervals: bool = True,
    days=None,
    weight_by_variance: bool = False,
    require_quality: bool = False,
) -> pd.DataFrame:
    """Aggregate speeds into time bins.

    Parameters
    ----------
    matched : DataFrame
        Must contain ``time`` (datetime) and ``speed_mps``. ``edge_id`` needed
        if ``by_edge=True``. Rows with ``edge_id == -1`` are excluded.
    block_hours : int
        Width of each time bin in hours. 1 -> hour-of-day (24 bins). 2, 3, 4, 6,
        8, 12 -> blocks; must divide 24 evenly for clean coverage (a warning is
        emitted otherwise).
    statistic : {"mean", "median", "both"}
        Which summary to compute. ``"mean"`` adds std/sem/ci columns;
        ``"median"`` adds q1/q3/iqr; ``"both"`` adds all.
    output_unit : str or SpeedUnit
        Unit for the reported speed columns.
    by_edge : bool
        If True, aggregate per (edge_id, time-bin) instead of network-wide.
    min_samples : int
        Bins with fewer than this many observations are dropped.
    dedup_intervals : bool
        When aggregating network-wide (``by_edge=False``) and the input carries
        an ``interval_id`` column (from :func:`redlight.speeds.derive_speeds`),
        drop duplicate rows of the same interval first, so an interval that
        traversed many edges counts once. Default True. Ignored for
        ``by_edge=True``, where the per-edge duplication is the point.
    days : optional
        Restrict to observations falling on particular days of the week before
        binning -- the way to separate weekday from weekend traffic (the
        hour-of-day bins otherwise pool, say, Tuesday 09:00 with Saturday
        09:00). Accepts a preset (``"weekday"`` = Mon-Fri, ``"weekend"`` =
        Sat-Sun, ``"all"``), a day name or number (``"Mon"``/``0`` ...
        ``"Sun"``/``6``), or an iterable of those (e.g. ``[0, 1, 2]`` or
        ``["Sat", "Sun"]``). Default ``None`` = every day (no filter). The
        filter uses the stored **local-clock** weekday, so load points with
        ``tz=`` if the study area isn't already local. See
        :func:`day_type_report` for a ready-made weekday-vs-weekend comparison.
    weight_by_variance : bool
        Weight each observation by ``1 / speed_var`` when computing the mean,
        instead of counting every observation equally. ``speed_var`` is the
        per-interval speed uncertainty :func:`~redlight.speeds.derive_speeds`
        already reports (``speed_sigma_mps ** 2``), which grows as the GPS noise
        floor divided by the interval's duration -- so a speed measured over a
        long, clean baseline stops being averaged on equal terms with a noisy
        short hop. ``mean_speed`` becomes ``sum(x/var) / sum(1/var)`` and
        ``sem_speed`` becomes ``sqrt(1 / sum(1/var))``, a propagated measurement
        uncertainty rather than a spread-over-``sqrt(n)`` -- so it is defined
        even for a single observation. ``std_speed`` stays the *unweighted*
        sample spread, which describes the traffic rather than the measurement.
        Rows whose ``speed_var`` is non-finite or non-positive carry no usable
        weight and are dropped with a warning. Requires a ``speed_var`` column
        and a mean (``statistic="mean"`` or ``"both"``; a ``"both"`` median is
        left unweighted). Default False.

    Returns
    -------
    DataFrame
        One row per bin (or per edge x bin). Always includes ``block_start_hour``,
        ``block_label`` and ``n``.
    """
    unit = SpeedUnit.parse(output_unit)
    if statistic not in {"mean", "median", "both"}:
        raise ValueError("statistic must be 'mean', 'median' or 'both'.")
    if weight_by_variance and statistic == "median":
        raise ValueError(
            "weight_by_variance applies to the mean; the median has no "
            "inverse-variance form here. Use statistic='mean' (or 'both', "
            "which leaves the median unweighted)."
        )
    if 24 % block_hours != 0:
        warnings.warn(
            f"block_hours={block_hours} does not divide 24 evenly; the last "
            "block will be narrower.",
            stacklevel=2,
        )

    _require_speed_mps(matched, "aggregate_speeds")
    df = _prepare(matched, days=days, require_quality=require_quality,
                  fn_name="aggregate_speeds")
    if weight_by_variance:
        if "speed_var" not in df.columns:
            raise ValueError(
                "weight_by_variance=True needs a 'speed_var' column, which "
                "derive_speeds emits (speed_sigma_mps ** 2). A frame from a "
                "matcher alone carries no per-observation uncertainty, so "
                "there is nothing to weight by."
            )
        var = pd.to_numeric(df["speed_var"], errors="coerce").to_numpy(float)
        # A zero/negative variance is an infinite or negative weight and a NaN
        # would poison the whole sum, so these rows cannot participate.
        usable = np.isfinite(var) & (var > 0)
        if not usable.all():
            warnings.warn(
                f"aggregate_speeds: dropped {int((~usable).sum())} row(s) with "
                "a non-finite or non-positive 'speed_var'; they carry no "
                "usable inverse-variance weight.",
                stacklevel=2,
            )
            df = df[usable]
    if not by_edge and dedup_intervals and "interval_id" in df.columns:
        _require_unique_intervals(df, "aggregate_speeds")
        df = df.drop_duplicates(subset="interval_id")
    df = df.copy()
    t = pd.to_datetime(df["time"])
    hour = t.dt.hour.values
    df["block_start_hour"] = (hour // block_hours) * block_hours

    group_cols = (["edge_id", "block_start_hour"] if by_edge
                  else ["block_start_hour"])

    rows = []
    for key, g in df.groupby(group_cols):
        if isinstance(key, tuple):
            keyd = dict(zip(group_cols, key))
        else:
            keyd = {group_cols[0]: key}
        sp = g["speed_mps"].values
        n = len(sp)
        if n < min_samples:
            continue
        rec = dict(keyd)
        rec["n"] = int(n)
        bs = int(rec["block_start_hour"])
        be = min(bs + block_hours, 24)
        rec["block_label"] = f"{bs:02d}:00-{be:02d}:00"
        if statistic in {"mean", "both"} and weight_by_variance:
            # Inverse-variance (a.k.a. precision) weighting: each observation
            # counts in proportion to 1/var, so a fix pair measured over a long
            # baseline outweighs a noisy short one instead of counting equally.
            # For independent measurements with known variances the combined
            # mean is sum(x/var)/sum(1/var) and the variance OF that mean is
            # 1/sum(1/var) -- so the SEM here is a propagated measurement
            # uncertainty, not a spread-over-sqrt(n). It is therefore defined
            # even for a single observation, unlike the unweighted branch.
            w = 1.0 / g["speed_var"].to_numpy(float)
            sum_w = float(w.sum())
            m = float(np.dot(w, sp) / sum_w)
            sem = float(np.sqrt(1.0 / sum_w))
            rec["mean_speed"] = from_mps(m, unit)
            rec["sem_speed"] = from_mps(sem, unit)
            rec["ci95_low"] = from_mps(m - 1.96 * sem, unit)
            rec["ci95_high"] = from_mps(m + 1.96 * sem, unit)
            # std stays the *unweighted* sample spread: it describes how much
            # the observed speeds actually varied, which is a property of the
            # traffic, not of how precisely each fix pair was measured.
            rec["std_speed"] = (from_mps(float(np.std(sp, ddof=1)), unit)
                                if n > 1 else np.nan)
        elif statistic in {"mean", "both"}:
            m = float(np.mean(sp))
            rec["mean_speed"] = from_mps(m, unit)
            if n > 1:
                s = float(np.std(sp, ddof=1))
                sem = s / np.sqrt(n)
                rec["std_speed"] = from_mps(s, unit)
                rec["sem_speed"] = from_mps(sem, unit)
                rec["ci95_low"] = from_mps(m - 1.96 * sem, unit)
                rec["ci95_high"] = from_mps(m + 1.96 * sem, unit)
            else:
                # a single observation carries no spread information; a
                # zero-width CI would claim perfect certainty
                rec["std_speed"] = np.nan
                rec["sem_speed"] = np.nan
                rec["ci95_low"] = np.nan
                rec["ci95_high"] = np.nan
        if statistic in {"median", "both"}:
            med = float(np.median(sp))
            q1 = float(np.percentile(sp, 25))
            q3 = float(np.percentile(sp, 75))
            rec["median_speed"] = from_mps(med, unit)
            rec["q1_speed"] = from_mps(q1, unit)
            rec["q3_speed"] = from_mps(q3, unit)
            rec["iqr_speed"] = from_mps(q3 - q1, unit)
        rec["unit"] = unit.value
        rows.append(rec)

    out = pd.DataFrame(rows)
    if len(out):
        sort_cols = (["edge_id", "block_start_hour"] if by_edge
                     else ["block_start_hour"])
        out = out.sort_values(sort_cols).reset_index(drop=True)
    return out


def peak_analysis(
    aggregated: pd.DataFrame,
    *,
    statistic: str = "mean",
    n_peak: int = 1,
    n_offpeak: int = 1,
) -> dict:
    """Identify peak (slowest) and off-peak (fastest) time bins.

    Peak traffic == lowest speeds. Operates on a network-wide aggregation
    (not ``by_edge``). Returns a dict with the ranked bins and the speed column
    used.

    Parameters
    ----------
    aggregated : DataFrame
        Output of :func:`aggregate_speeds` (network-wide).
    statistic : {"mean", "median"}
        Which speed column to rank on.
    n_peak, n_offpeak : int
        How many peak / off-peak bins to return.
    """
    if len(aggregated) == 0:
        raise ValueError(
            "peak_analysis: the aggregation is empty (no observations survived "
            "filtering, or min_samples suppressed every bin)."
        )
    if "edge_id" in aggregated.columns:
        raise ValueError(
            "peak_analysis needs a network-wide aggregation; re-run "
            "aggregate_speeds with by_edge=False (this input is per-edge)."
        )
    col = "mean_speed" if statistic == "mean" else "median_speed"
    if col not in aggregated.columns:
        raise ValueError(
            f"Column {col!r} not in aggregation; re-run aggregate_speeds with "
            f"statistic='{statistic}' or 'both'."
        )
    ordered = aggregated.sort_values(col).reset_index(drop=True)
    if n_peak + n_offpeak > len(ordered):
        warnings.warn(
            f"n_peak + n_offpeak = {n_peak + n_offpeak} exceeds the "
            f"{len(ordered)} available bins; the peak and off-peak lists will "
            "overlap.",
            stacklevel=2,
        )
    peak = ordered.head(n_peak)          # slowest = peak congestion
    offpeak = ordered.tail(n_offpeak)    # fastest = free flow
    # Ranking always yields a "slowest" and a "fastest" block, even when every
    # bin holds the same speed -- and then which bins those are is an artifact
    # of sort order, not of traffic. classify_hours already refuses to pass
    # that off as a result; say the same thing here rather than let a caller
    # read peak[] and report hours that are in no way peak.
    if len(peak) and len(offpeak):
        spread = float(offpeak[col].max()) - float(peak[col].min())
        if not spread > 0:
            warnings.warn(
                "peak_analysis: every bin has the same representative speed, "
                "so this data has no peak -- the returned peak and off-peak "
                "blocks are an artifact of sort order, not of congestion.",
                stacklevel=2,
            )
    return {
        "speed_column": col,
        "peak": peak.to_dict("records"),
        "off_peak": offpeak.to_dict("records"),
        "ranked": ordered[["block_label", col, "n"]].to_dict("records"),
    }


def _validate_hours(hours: Iterable, name: str) -> set:
    """Coerce an hour-of-day iterable to a ``set[int]``, validating 0-23.

    Parameters
    ----------
    hours : iterable of int
        Candidate hour-of-day values, e.g. ``[7, 8, 9]``.
    name : str
        The caller's parameter name (``"peak_hours"``/``"offpeak_hours"``),
        used only to make the error message point at the right argument.

    Returns
    -------
    set of int
    """
    out = {int(h) for h in hours}
    bad = {h for h in out if not 0 <= h <= 23}
    if bad:
        raise ValueError(f"{name} contains invalid hours {sorted(bad)}; use 0-23.")
    return out


def _best_window(hour_speeds: dict, width: int, *, minimize: bool,
                 exclude: frozenset = frozenset()):
    """Best contiguous ``width``-hour window over the 24-hour clock (wrapping).

    ``hour_speeds`` maps hour-of-day -> representative speed (m/s) for hours
    with data. A window is scored by the mean speed of its observed hours;
    windows containing no observed hour, or any excluded hour, are ineligible.
    Ties break toward the earliest start hour. Returns (hours, score) or None.
    """
    best = None
    for start in range(24):
        hrs = [(start + k) % 24 for k in range(width)]
        if any(h in exclude for h in hrs):
            continue
        vals = [hour_speeds[h] for h in hrs if h in hour_speeds]
        if not vals:
            continue
        score = float(np.mean(vals))
        key = (score if minimize else -score, start)
        if best is None or key < best[0]:
            best = (key, hrs, score)
    if best is None:
        return None
    return best[1], best[2]


def classify_hours(
    matched: pd.DataFrame,
    *,
    statistic: str = "median",
    peak_hours=None,
    offpeak_hours=None,
    n_peak: int | None = None,
    n_offpeak: int | None = None,
    min_samples: int = 1,
    days=None,
    require_quality: bool = False,
) -> dict:
    """Split the 24 hours of the day into a peak and an off-peak block.

    ``days`` restricts which weekdays contribute (e.g. ``"weekday"`` /
    ``"weekend"`` / ``[0, 1, 2]``) so peak windows can be classified separately
    for weekday and weekend traffic; see :func:`aggregate_speeds` for the
    accepted forms. It has no effect in explicit-override mode (the hour lists
    are taken as given). Default ``None`` = every day.

    Three modes, in order of precedence:

    1. **Explicit override** -- pass ``peak_hours`` and/or ``offpeak_hours``
       (iterables of 0-23). If only one is given, the other becomes the
       remaining hours of the day. The two sets must be disjoint.
    2. **Contiguous windows** -- pass ``n_peak`` and ``n_offpeak``. The peak
       block is the contiguous ``n_peak``-hour window (wrapping midnight) with
       the *lowest* network-wide representative speed (peak = most congested);
       the off-peak block is the contiguous ``n_offpeak``-hour window with the
       *highest* speed among windows disjoint from the peak block. Windows are
       scored on the mean of their observed hourly speeds. Because a window is
       contiguous by construction it **may include hours with no observations
       at all** -- unlike mode 3 below, which leaves those unassigned. Such an
       hour is scored as absent rather than as slow (it contributes nothing to
       the mean), but it still lands in the returned list, and from there in
       :func:`assign_segment_speeds`, so an edge can be labelled peak or
       off-peak for an hour nothing was ever measured in. Ties between equally
       scoring windows go to the earliest.
    3. **Data-driven median split** (default) -- each hour-of-day's
       network-wide representative speed (median or mean) is computed, and
       hours at or below the median of those hourly speeds are labelled *peak*
       (slower = busier), the rest *off-peak*. Hours with no observations are
       left unassigned.

    Returns
    -------
    dict
        ``peak_hours`` (sorted list), ``offpeak_hours`` (sorted list),
        ``threshold_speed_mps`` (median-split point, or None otherwise),
        ``source`` (``"override"``, ``"window"`` or ``"auto"``), and -- in
        window mode -- ``peak_speed_mps`` / ``offpeak_speed_mps`` (the window
        scores).
    """
    if peak_hours is not None or offpeak_hours is not None:
        all_h = set(range(24))
        if peak_hours is not None and offpeak_hours is not None:
            peak = _validate_hours(peak_hours, "peak_hours")
            off = _validate_hours(offpeak_hours, "offpeak_hours")
            overlap = peak & off
            if overlap:
                raise ValueError(
                    f"peak_hours and offpeak_hours overlap on {sorted(overlap)}; "
                    "an hour cannot belong to both regimes."
                )
        elif peak_hours is not None:
            peak = _validate_hours(peak_hours, "peak_hours")
            off = all_h - peak
        else:
            off = _validate_hours(offpeak_hours, "offpeak_hours")
            peak = all_h - off
        return {
            "peak_hours": sorted(peak),
            "offpeak_hours": sorted(off),
            "threshold_speed_mps": None,
            "source": "override",
        }

    if (n_peak is None) != (n_offpeak is None):
        raise ValueError(
            "Pass both n_peak and n_offpeak (window mode), or neither "
            "(median split)."
        )

    agg = aggregate_speeds(
        matched, block_hours=1, statistic=statistic, output_unit="mps",
        by_edge=False, min_samples=min_samples, days=days,
        require_quality=require_quality,
    )
    if len(agg) == 0:
        raise ValueError(
            "classify_hours: no observations to classify (every hour bin was "
            "empty or suppressed by min_samples"
            + (", or the days= filter matched no rows)." if days is not None
               else ").")
        )
    col = "median_speed" if statistic == "median" else "mean_speed"
    hours = agg["block_start_hour"].to_numpy()
    speeds = agg[col].to_numpy(dtype=float)

    if n_peak is not None:
        if not (1 <= n_peak <= 23 and 1 <= n_offpeak <= 23):
            raise ValueError("n_peak and n_offpeak must each be between 1 and 23.")
        if n_peak + n_offpeak > 24:
            raise ValueError(
                f"n_peak + n_offpeak = {n_peak + n_offpeak} exceeds 24 hours; "
                "the windows cannot be disjoint."
            )
        hour_speeds = {int(h): float(s) for h, s in zip(hours, speeds)}
        peak_win = _best_window(hour_speeds, n_peak, minimize=True)
        if peak_win is None:  # pragma: no cover - agg non-empty implies a window
            raise ValueError("classify_hours: no window with data found.")
        peak_hrs, peak_score = peak_win
        off_win = _best_window(hour_speeds, n_offpeak, minimize=False,
                               exclude=frozenset(peak_hrs))
        if off_win is None:
            raise ValueError(
                "classify_hours: no off-peak window disjoint from the peak "
                "window contains data. Collect data across more of the day, or "
                "reduce n_peak/n_offpeak."
            )
        off_hrs, off_score = off_win
        return {
            "peak_hours": sorted(peak_hrs),
            "offpeak_hours": sorted(off_hrs),
            "threshold_speed_mps": None,
            "peak_speed_mps": peak_score,
            "offpeak_speed_mps": off_score,
            "source": "window",
        }

    threshold = float(np.median(speeds))
    peak = sorted(int(h) for h, s in zip(hours, speeds) if s <= threshold)
    off = sorted(int(h) for h, s in zip(hours, speeds) if s > threshold)
    if not off:
        warnings.warn(
            "classify_hours: every observed hour tied at or below the median "
            "speed, so the off-peak set is empty. Consider window mode "
            "(n_peak=/n_offpeak=) or explicit hour lists.",
            stacklevel=2,
        )
    return {
        "peak_hours": peak,
        "offpeak_hours": off,
        "threshold_speed_mps": threshold,
        "source": "auto",
    }


def _per_edge_speed(df: pd.DataFrame, statistic: str) -> pd.Series:
    """Representative speed (m/s) per edge_id, median or mean."""
    if len(df) == 0:
        return pd.Series(dtype=float)
    agg_fn = np.median if statistic == "median" else np.mean
    return df.groupby("edge_id")["speed_mps"].apply(
        lambda s: float(agg_fn(s.values))
    )


def assign_segment_speeds(
    network,
    matched: pd.DataFrame,
    *,
    statistic: str = "median",
    peak_hours=None,
    offpeak_hours=None,
    n_peak: int | None = None,
    n_offpeak: int | None = None,
    default_speed_mps: float | None = None,
    min_samples: int = 1,
    days=None,
    require_quality: bool = False,
) -> dict:
    """Write three representative speeds per edge: overall, peak, off-peak.

    For every edge, computes the median (default) or mean speed over (a) the whole
    timeframe, (b) the peak-hour block, and (c) the off-peak-hour block, and writes
    them onto the graph as ``obs_speed_mps_{overall,peak,offpeak}`` with matching
    ``travel_time_s_{overall,peak,offpeak}``. The overall values are also written
    to the plain ``obs_speed_mps`` / ``travel_time_s`` so a default
    :class:`~redlight.routing.Router` keeps working unchanged.

    Pooling into two broad blocks (rather than 24 hourly slices) is deliberate:
    it keeps far more observations per segment, which is what makes per-edge
    speeds -- and therefore time routing -- stable on sparse GPS data.

    Peak/off-peak hours come from :func:`classify_hours`: pass explicit
    ``peak_hours``/``offpeak_hours``, or ``n_peak``/``n_offpeak`` for the
    contiguous-window mode, or nothing for the automatic median split.

    Parameters
    ----------
    statistic : {"median", "mean"}
        Per-edge summary statistic.
    peak_hours, offpeak_hours : list of int, optional
        Explicit hour-of-day lists (0-23) for each regime, passed straight
        through to :func:`classify_hours`. Overrides ``n_peak``/``n_offpeak``
        and the automatic median split when given.
    n_peak, n_offpeak : int, optional
        Widths (hours) of the contiguous peak and off-peak windows -- see
        :func:`classify_hours`.
    default_speed_mps : float, optional
        Fallback speed for edges with no observations in a given regime.
    min_samples : int
        Passed to auto hour classification.
    days : optional
        Restrict to particular weekdays before both classifying hours and
        pooling per-edge speeds -- e.g. ``"weekday"`` / ``"weekend"`` /
        ``[0, 1, 2]``. Build one annotated network per day-type to compare
        weekday and weekend congestion on the same segments (:func:`to_geojson`
        / :func:`~redlight.mapping.plot_speed_map`). See
        :func:`aggregate_speeds` for the accepted forms; default ``None`` =
        every day. ``day_type_report`` wraps this comparison up for you.

    Returns
    -------
    dict
        ``peak_hours``, ``offpeak_hours``, ``threshold_speed_mps``, ``source``,
        ``days`` (the resolved weekday set, or None), ``n_edges_total`` and
        ``coverage`` (observed-edge counts per regime).
    """
    cls = classify_hours(
        matched, statistic=statistic, peak_hours=peak_hours,
        offpeak_hours=offpeak_hours, n_peak=n_peak, n_offpeak=n_offpeak,
        min_samples=min_samples, days=days, require_quality=require_quality,
    )
    peak_set = set(cls["peak_hours"])
    off_set = set(cls["offpeak_hours"])

    day_set = _resolve_days(days)
    df = _prepare(matched, days=days, require_quality=require_quality,
                  fn_name="assign_segment_speeds")
    hod = pd.to_datetime(df["time"]).dt.hour.to_numpy()

    regimes = {
        "overall": _per_edge_speed(df, statistic),
        "peak": _per_edge_speed(df[np.isin(hod, list(peak_set))], statistic),
        "offpeak": _per_edge_speed(df[np.isin(hod, list(off_set))], statistic),
    }

    coverage = {r: 0 for r in regimes}
    for _u, _v, data in network.graph.edges(data=True):
        eid = data["edge_id"]
        for r, per_edge in regimes.items():
            spd = per_edge.get(eid, None)
            # 0.0 m/s (gridlock) is a real observation, not "unobserved" --
            # only a missing per-edge entry means that.
            observed = spd is not None
            if not observed:
                spd = default_speed_mps
            if spd is not None and spd >= 0:
                data[f"obs_speed_mps_{r}"] = float(spd)
                if spd > 0:
                    data[f"travel_time_s_{r}"] = data["length_m"] / float(spd)
                else:
                    # travel time is undefined/infinite at 0 m/s
                    data.pop(f"travel_time_s_{r}", None)
            else:
                data.pop(f"obs_speed_mps_{r}", None)
                data.pop(f"travel_time_s_{r}", None)
            if observed:
                coverage[r] += 1
        # Keep the plain attributes pointing at the overall regime for
        # back-compatible default routing. travel_time_s_overall may be
        # absent even when obs_speed_mps_overall is present (0 m/s has no
        # defined travel time).
        if "obs_speed_mps_overall" in data:
            data["obs_speed_mps"] = data["obs_speed_mps_overall"]
            if "travel_time_s_overall" in data:
                data["travel_time_s"] = data["travel_time_s_overall"]
            else:
                data.pop("travel_time_s", None)
        else:
            data.pop("obs_speed_mps", None)
            data.pop("travel_time_s", None)

    if coverage["peak"] == 0 or coverage["offpeak"] == 0:
        warnings.warn(
            "assign_segment_speeds: a regime has no observed edges "
            f"(coverage={coverage}). Routes in that period fall back to the "
            "default speed everywhere. Check that the data spans both peak and "
            "off-peak hours.",
            stacklevel=2,
        )

    return {
        "peak_hours": cls["peak_hours"],
        "offpeak_hours": cls["offpeak_hours"],
        "threshold_speed_mps": cls["threshold_speed_mps"],
        "source": cls["source"],
        "days": sorted(day_set) if day_set is not None else None,
        "n_edges_total": network.graph.number_of_edges(),
        "coverage": coverage,
    }


def assign_speeds(
    network,
    matched: pd.DataFrame,
    *,
    statistic: str = "median",
    default_speed_mps: float | None = None,
    block_hours: int = 24,
    target_hour: int | None = None,
    days=None,
    require_quality: bool = False,
) -> dict:
    """Compute a representative speed per edge and write it onto the graph.

    Stores the result as edge attribute ``obs_speed_mps`` -- always m/s, since
    this is what :class:`~redlight.routing.Router` reads -- and
    ``travel_time_s = length_m / obs_speed_mps``. Edges with no observations
    get ``default_speed_mps`` if provided, else are left without a travel time
    (router can fall back to length). Unlike :func:`aggregate_speeds`, there is
    no ``output_unit``: this function's output is for routing, not display, so
    it is never converted out of m/s.

    Parameters
    ----------
    statistic : {"mean", "median"}
        Per-edge summary statistic.
    block_hours, target_hour : int
        If ``target_hour`` is given, only observations whose hour-of-day falls in
        the block starting at ``(target_hour // block_hours) * block_hours`` are
        used, enabling time-of-day routing.
    default_speed_mps : float, optional
        Fallback speed for unobserved edges.
    days : optional
        Restrict to particular weekdays before computing per-edge speeds -- e.g.
        ``"weekday"`` / ``"weekend"`` / ``[0, 1, 2]`` -- so a routing network can
        be built from weekday-only (or weekend-only) travel times. Combines with
        ``target_hour``. See :func:`aggregate_speeds` for the accepted forms;
        default ``None`` = every day.

    Returns
    -------
    dict
        ``{"n_edges_observed": int, "n_edges_total": int}`` --
        ``n_edges_observed`` counts only edges whose written speed came from
        observations (not the default fallback).
    """
    _require_speed_mps(matched, "assign_speeds")
    df = _prepare(matched, days=days, require_quality=require_quality,
                  fn_name="assign_speeds")
    if target_hour is not None:
        t = pd.to_datetime(df["time"])
        bs = (target_hour // block_hours) * block_hours
        be = bs + block_hours
        h = t.dt.hour.values
        df = df[(h >= bs) & (h < be)]

    per_edge = _per_edge_speed(df, statistic)

    observed = 0
    for _u, _v, data in network.graph.edges(data=True):
        eid = data["edge_id"]
        spd = per_edge.get(eid, None)
        # 0.0 m/s (gridlock) is a real observation, not "unobserved" -- only
        # a missing per-edge entry means that.
        from_observation = spd is not None
        if not from_observation:
            spd = default_speed_mps
        if spd is not None and spd >= 0:
            data["obs_speed_mps"] = float(spd)
            if spd > 0:
                data["travel_time_s"] = data["length_m"] / float(spd)
            else:
                # travel time is undefined/infinite at 0 m/s
                data.pop("travel_time_s", None)
            if from_observation:
                observed += 1
        else:
            data.pop("obs_speed_mps", None)
            data.pop("travel_time_s", None)
    return {
        "n_edges_observed": observed,
        "n_edges_total": network.graph.number_of_edges(),
    }


def day_type_report(
    matched: pd.DataFrame,
    *,
    statistic: str = "median",
    output_unit="mph",
    block_hours: int = 1,
    groups: dict | None = None,
    n_peak: int = 1,
    n_offpeak: int = 1,
    min_samples: int = 1,
    weight_by_variance: bool = False,
    require_quality: bool = False,
) -> dict:
    """Compare traffic across day-types -- weekday vs weekend by default.

    A ready-made summary of how speeds differ between groups of weekdays, built
    on the same ``days=`` filter the rest of this module now accepts. For each
    group it reports the network-wide overall speed, the full per-hour (or
    per-block) profile, and the peak/off-peak blocks; then it lines the groups
    up block-by-block so the weekday-vs-weekend **difference** is a single table
    you can print or export.

    Parameters
    ----------
    matched : DataFrame
        Speed observations with ``time`` and ``speed_mps`` (a matcher output, or
        ``derive_speeds``' ``edge_observations``). Rows with ``edge_id == -1``
        are excluded and, when an ``interval_id`` column is present, network-wide
        counts deduplicate on it -- exactly as :func:`aggregate_speeds` does.
    statistic : {"median", "mean"}
        Summary statistic for every reported speed. Median (default) is the
        robust choice for skewed congested-flow speeds.
    output_unit : str or SpeedUnit
        Unit for every speed in the result.
    block_hours : int
        Time-bin width for the per-hour profile and peak detection (see
        :func:`aggregate_speeds`). 1 (default) = hour-of-day.
    groups : dict, optional
        ``{label: day-selector}`` mapping, where each selector is anything
        :func:`aggregate_speeds`' ``days=`` accepts (a preset, day name/number,
        or iterable). Default
        ``{"weekday": "weekday", "weekend": "weekend"}``. Provide your own to
        compare arbitrary day groupings, e.g.
        ``{"Mon-Thu": [0, 1, 2, 3], "Fri": "fri"}``.
    n_peak, n_offpeak : int
        How many peak (slowest) / off-peak (fastest) blocks to report per group.
    min_samples : int
        Bins with fewer observations than this are dropped before ranking.

    Returns
    -------
    dict
        ``statistic``, ``unit``, ``block_hours`` : echoes of the inputs.
        ``groups`` : dict of ``label -> {days, n, overall_speed, hourly,
            peak, offpeak}``. ``days`` is the resolved weekday set (or None for
            all); ``n`` is the number of independent measurements; ``hourly`` is
            the full :func:`aggregate_speeds` frame; ``peak``/``offpeak`` are
            lists of ``{block_label, speed, n}`` (``None`` if the group had no
            data).
        ``overall`` : dict of ``{label}_speed`` per group, plus -- when there
            are exactly two groups -- ``delta_speed`` and ``delta_pct`` (the
            second group minus the first: with the default groups that is
            *weekend minus weekday*, so a positive number means weekends run
            faster / less congested).
        ``comparison`` : a tidy DataFrame, one row per time block, with a
            ``{label}_speed`` column per group and (for two groups)
            ``delta_speed`` / ``delta_pct``. Print it to read the weekday-vs-
            weekend change at a glance, or feed it straight to a plot.

    Notes
    -----
    A day-type with no observations (e.g. a dataset that never sampled a
    weekend) is reported honestly with ``n=0`` and NaN speeds, and a warning --
    not an error. Weekdays are taken on the stored **local clock**, so load
    points with ``tz=`` if the study area isn't already in local time.
    """
    if statistic not in {"mean", "median"}:
        raise ValueError("statistic must be 'mean' or 'median'.")
    _require_speed_mps(matched, "day_type_report")
    unit = SpeedUnit.parse(output_unit)
    if groups is None:
        groups = {"weekday": "weekday", "weekend": "weekend"}
    if not groups:
        raise ValueError("groups must contain at least one day-type.")

    speed_col = "median_speed" if statistic == "median" else "mean_speed"

    results: dict = {}
    empties: list = []
    pieces: list = []           # each group's hourly column, for the comparison
    for label, selector in groups.items():
        day_set = _resolve_days(selector)

        # A single 24-hour bin is the network-wide overall speed and count,
        # interval-deduplicated exactly the way the hourly profile is -- so both
        # numbers in the report come from the one aggregation path instead of a
        # hand-rolled second copy of the dedup/filter/aggregate logic.
        overall_agg = aggregate_speeds(
            matched, block_hours=24, statistic=statistic,
            output_unit=unit, by_edge=False, days=selector,
            weight_by_variance=weight_by_variance,
            require_quality=require_quality,
        )
        if len(overall_agg):
            overall = float(overall_agg[speed_col].iloc[0])
            n = int(overall_agg["n"].iloc[0])
        else:
            overall, n = float("nan"), 0

        hourly = aggregate_speeds(
            matched, block_hours=block_hours, statistic=statistic,
            output_unit=unit, by_edge=False, min_samples=min_samples,
            days=selector, weight_by_variance=weight_by_variance,
            require_quality=require_quality,
        )
        peak = offpeak = None
        if len(hourly):
            pk = peak_analysis(hourly, statistic=statistic,
                               n_peak=n_peak, n_offpeak=n_offpeak)
            peak = [{"block_label": r["block_label"], "speed": r[speed_col],
                     "n": r["n"]} for r in pk["peak"]]
            offpeak = [{"block_label": r["block_label"], "speed": r[speed_col],
                        "n": r["n"]} for r in pk["off_peak"]]
        if n == 0:
            empties.append(label)
        results[label] = {
            "days": sorted(day_set) if day_set is not None else None,
            "n": n,
            "overall_speed": overall,
            "hourly": hourly,
            "peak": peak,
            "offpeak": offpeak,
        }

        # accumulate this group's column for the block-by-block comparison
        col = f"{label}_speed"
        if len(hourly):
            piece = hourly[["block_start_hour", "block_label", speed_col]].rename(
                columns={speed_col: col})
        else:
            # aggregate_speeds returns a column-less frame when empty, so build
            # the typed empty schema the outer merge below needs.
            piece = pd.DataFrame({
                "block_start_hour": pd.Series([], dtype="int64"),
                "block_label": pd.Series([], dtype="object"),
                col: pd.Series([], dtype="float64"),
            })
        pieces.append(piece)

    labels = list(groups)
    # block-by-block comparison table: outer-merge every group's column
    comp = reduce(
        lambda left, right: left.merge(
            right, on=["block_start_hour", "block_label"], how="outer"),
        pieces,
    )
    if len(comp):
        comp = comp.sort_values("block_start_hour").reset_index(drop=True)

    overall_summary = {f"{label}_speed": results[label]["overall_speed"]
                       for label in labels}
    if len(labels) == 2:
        g1, g2 = labels
        # second group minus first (weekend - weekday, with the defaults):
        # positive => the second day-type is faster / less congested.
        comp["delta_speed"] = comp[f"{g2}_speed"] - comp[f"{g1}_speed"]
        with np.errstate(divide="ignore", invalid="ignore"):
            comp["delta_pct"] = 100.0 * comp["delta_speed"] / comp[f"{g1}_speed"]
        base = overall_summary[f"{g1}_speed"]
        d = overall_summary[f"{g2}_speed"] - base
        overall_summary["delta_speed"] = d
        overall_summary["delta_pct"] = (100.0 * d / base) if base else float("nan")

    if empties:
        warnings.warn(
            f"day_type_report: day-type(s) {empties} had no observations; their "
            "speeds are NaN. Check the data spans those days (and that points "
            "were loaded with the right tz= so weekdays are on the local clock).",
            stacklevel=2,
        )

    return {
        "statistic": statistic,
        "unit": unit.value,
        "block_hours": block_hours,
        "groups": results,
        "overall": overall_summary,
        "comparison": comp,
    }


def congestion_report(
    network,
    matched: pd.DataFrame,
    *,
    statistic: str = "median",
    output_unit="mph",
    block_hours: int | None = None,
    days=None,
    min_samples: int = 1,
    require_quality: bool = False,
    weight_by_variance: bool = False,
) -> dict:
    """Observed speed as a fraction of the posted limit, per edge.

    The standard level-of-service framing: congestion is the gap between how
    fast a road is *allowed* to run and how fast it *actually* ran. A ratio of
    1.0 means traffic moved at the posted limit; 0.45 means it crawled at 45%
    of it. Needs the ``maxspeed_mps`` edge attribute, which
    :meth:`~redlight.network.Network.from_geojson` and friends parse from an
    OSM ``maxspeed`` tag at load time.

    Parameters
    ----------
    network : redlight.network.Network
        Supplies the posted limits. Edges with no usable ``maxspeed_mps``
        (absent, or a NaN/non-positive value from the caller's own source
        data) are reported with a NaN limit and ratio rather than dropped, so
        the coverage gap stays visible.
    matched : DataFrame
        Speed observations, as :func:`aggregate_speeds` takes them.
    statistic : {"median", "mean"}
        Summary statistic for the observed speed. Median is the robust default.
    output_unit : str or SpeedUnit
        Unit for the two reported speeds. ``ratio`` is dimensionless.
    block_hours : int, optional
        ``None`` (default) gives one row per edge over the whole period. An
        integer additionally bins by time of day, giving one row per edge per
        block and a ``block_start_hour`` column -- the way to see a corridor
        drop to 40% of its limit at 08:00 and recover by 14:00.
    days, min_samples, require_quality, weight_by_variance
        Passed through to :func:`aggregate_speeds`.

    Returns
    -------
    dict
        ``edges`` : DataFrame with ``edge_id``, ``observed_speed``,
            ``speed_limit``, ``ratio``, ``n`` (plus ``block_start_hour`` and
            ``block_label`` when ``block_hours`` is set). ``ratio`` is
            ``observed / limit``, **not clipped at 1.0** -- traffic observed
            above the posted limit is a real finding, and silently capping it
            would hide it.
        ``summary`` : ``n_rows``, ``n_edges_observed``, ``n_edges_rated``
            (observed *and* carrying a usable limit -- the only ones a ratio
            exists for), ``median_ratio``/``mean_ratio`` over those, and
            ``unit``/``statistic`` echoes.

    Notes
    -----
    A posted limit is a legal maximum, not a free-flow speed: roads routinely
    run below their limit for reasons other than congestion (geometry, signals,
    parking), and a ratio near 1.0 on a road whose limit is unrealistically low
    is not evidence of free flow. Treat the ratio as a screening indicator, and
    prefer comparing a road against *itself* across time blocks over comparing
    different roads against each other.
    """
    if statistic not in {"mean", "median"}:
        raise ValueError("statistic must be 'mean' or 'median'.")
    unit = SpeedUnit.parse(output_unit)
    speed_col = "median_speed" if statistic == "median" else "mean_speed"

    agg = aggregate_speeds(
        matched, by_edge=True, statistic=statistic, output_unit=unit,
        block_hours=24 if block_hours is None else block_hours,
        days=days, min_samples=min_samples, require_quality=require_quality,
        weight_by_variance=weight_by_variance,
    )

    keep = ["edge_id", speed_col, "n"]
    if block_hours is not None:
        keep = ["edge_id", "block_start_hour", "block_label", speed_col, "n"]
    edges = (agg[keep].rename(columns={speed_col: "observed_speed"})
             if len(agg) else
             pd.DataFrame({c: pd.Series([], dtype="float64") for c in keep}
                          | {"observed_speed": pd.Series([], dtype="float64")}))

    # posted limit per edge, in the reported unit; NaN where unusable/absent
    limits = {}
    for eid in network.edge_ids:
        mps = _usable_speed(network.edge_data(int(eid)).get("maxspeed_mps"))
        limits[int(eid)] = (float(from_mps(mps, unit)) if mps is not None
                            else float("nan"))
    edges = edges.copy()
    edges["speed_limit"] = edges["edge_id"].map(limits).astype(float)
    edges["ratio"] = edges["observed_speed"] / edges["speed_limit"]
    if len(edges):
        sort_cols = (["edge_id"] if block_hours is None
                     else ["edge_id", "block_start_hour"])
        edges = edges.sort_values(sort_cols).reset_index(drop=True)

    rated = edges["ratio"].notna() if len(edges) else edges.index
    ratios = edges.loc[rated, "ratio"] if len(edges) else pd.Series(dtype=float)
    summary = {
        "statistic": statistic,
        "unit": unit.value,
        "block_hours": block_hours,
        "n_rows": int(len(edges)),
        "n_edges_observed": int(edges["edge_id"].nunique()) if len(edges) else 0,
        "n_edges_rated": (int(edges.loc[rated, "edge_id"].nunique())
                          if len(edges) else 0),
        "median_ratio": float(ratios.median()) if len(ratios) else float("nan"),
        "mean_ratio": float(ratios.mean()) if len(ratios) else float("nan"),
    }
    return {"edges": edges, "summary": summary}
