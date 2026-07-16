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

import numpy as np
import pandas as pd

from .units import SpeedUnit, from_mps


def _require_speed_mps(df: pd.DataFrame, fn_name: str) -> None:
    """Raise a clear, actionable error if ``df`` has no ``speed_mps`` column.

    Shared by every function in this module (and ``cleaning.py``) that needs
    a speed to operate on, so points loaded without a speed column
    (position+time only -- see :func:`roadtraffic.points.load_points`) get a
    consistent, specific error message pointing at the fix instead of a
    generic pandas ``KeyError`` deep in a groupby.
    """
    if "speed_mps" not in df.columns:
        raise ValueError(
            f"{fn_name} needs a 'speed_mps' column. If points were loaded "
            "without a speed column, run a matcher then "
            "roadtraffic.speeds.derive_speeds first and pass its "
            "'edge_observations' frame here."
        )


def _drop_unmatched(df: pd.DataFrame) -> pd.DataFrame:
    """Remove edge_id == -1 sentinel rows (points that snapped to nothing)."""
    if "edge_id" in df.columns:
        return df[df["edge_id"] != -1]
    return df


def aggregate_speeds(
    matched: pd.DataFrame,
    *,
    block_hours: int = 1,
    statistic: str = "mean",
    output_unit="mph",
    by_edge: bool = False,
    min_samples: int = 1,
    dedup_intervals: bool = True,
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
        an ``interval_id`` column (from :func:`roadtraffic.speeds.derive_speeds`),
        drop duplicate rows of the same interval first, so an interval that
        traversed many edges counts once. Default True. Ignored for
        ``by_edge=True``, where the per-edge duplication is the point.

    Returns
    -------
    DataFrame
        One row per bin (or per edge x bin). Always includes ``block_start_hour``,
        ``block_label`` and ``n``.
    """
    unit = SpeedUnit.parse(output_unit)
    if statistic not in {"mean", "median", "both"}:
        raise ValueError("statistic must be 'mean', 'median' or 'both'.")
    if 24 % block_hours != 0:
        warnings.warn(
            f"block_hours={block_hours} does not divide 24 evenly; the last "
            "block will be narrower.",
            stacklevel=2,
        )

    _require_speed_mps(matched, "aggregate_speeds")
    df = _drop_unmatched(matched)
    df = df[~df["speed_mps"].isna()]
    if not by_edge and dedup_intervals and "interval_id" in df.columns:
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
        if statistic in {"mean", "both"}:
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
) -> dict:
    """Split the 24 hours of the day into a peak and an off-peak block.

    Three modes, in order of precedence:

    1. **Explicit override** -- pass ``peak_hours`` and/or ``offpeak_hours``
       (iterables of 0-23). If only one is given, the other becomes the
       remaining hours of the day. The two sets must be disjoint.
    2. **Contiguous windows** -- pass ``n_peak`` and ``n_offpeak``. The peak
       block is the contiguous ``n_peak``-hour window (wrapping midnight) with
       the *lowest* network-wide representative speed (peak = most congested);
       the off-peak block is the contiguous ``n_offpeak``-hour window with the
       *highest* speed among windows disjoint from the peak block. Windows are
       scored on the mean of their observed hourly speeds.
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
        by_edge=False, min_samples=min_samples,
    )
    if len(agg) == 0:
        raise ValueError(
            "classify_hours: no observations to classify (every hour bin was "
            "empty or suppressed by min_samples)."
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
) -> dict:
    """Write three representative speeds per edge: overall, peak, off-peak.

    For every edge, computes the median (default) or mean speed over (a) the whole
    timeframe, (b) the peak-hour block, and (c) the off-peak-hour block, and writes
    them onto the graph as ``obs_speed_mps_{overall,peak,offpeak}`` with matching
    ``travel_time_s_{overall,peak,offpeak}``. The overall values are also written
    to the plain ``obs_speed_mps`` / ``travel_time_s`` so a default
    :class:`~roadtraffic.routing.Router` keeps working unchanged.

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

    Returns
    -------
    dict
        ``peak_hours``, ``offpeak_hours``, ``threshold_speed_mps``, ``source``,
        ``n_edges_total`` and ``coverage`` (observed-edge counts per regime).
    """
    cls = classify_hours(
        matched, statistic=statistic, peak_hours=peak_hours,
        offpeak_hours=offpeak_hours, n_peak=n_peak, n_offpeak=n_offpeak,
        min_samples=min_samples,
    )
    peak_set = set(cls["peak_hours"])
    off_set = set(cls["offpeak_hours"])

    df = _drop_unmatched(matched)
    df = df[~df["speed_mps"].isna()]
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
) -> dict:
    """Compute a representative speed per edge and write it onto the graph.

    Stores the result as edge attribute ``obs_speed_mps`` -- always m/s, since
    this is what :class:`~roadtraffic.routing.Router` reads -- and
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

    Returns
    -------
    dict
        ``{"n_edges_observed": int, "n_edges_total": int}`` --
        ``n_edges_observed`` counts only edges whose written speed came from
        observations (not the default fallback).
    """
    _require_speed_mps(matched, "assign_speeds")
    df = _drop_unmatched(matched)
    df = df[~df["speed_mps"].isna()]
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
