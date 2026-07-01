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
- The median and interquartile range (IQR = Q3 - Q1) are robust alternatives
  recommended when speed distributions are skewed (common in congested flow).
- Bins with few observations are statistically weak. ``min_samples`` lets you
  suppress under-sampled bins; ``n`` is always reported so users can judge.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .units import SpeedUnit, from_mps


def _require_speed_mps(df: pd.DataFrame, fn_name: str) -> None:
    if "speed_mps" not in df.columns:
        raise ValueError(
            f"{fn_name} needs a 'speed_mps' column. If points were loaded "
            "without a speed column, run a matcher then "
            "roadtraffic.speeds.derive_speeds first and pass its "
            "'edge_observations' frame here."
        )


def aggregate_speeds(
    matched: pd.DataFrame,
    *,
    block_hours: int = 1,
    statistic: str = "mean",
    output_unit="mph",
    by_edge: bool = False,
    min_samples: int = 1,
) -> pd.DataFrame:
    """Aggregate speeds into time bins.

    Parameters
    ----------
    matched : DataFrame
        Must contain ``time`` (datetime) and ``speed_mps``. ``edge_id`` needed
        if ``by_edge=True``.
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
        import warnings
        warnings.warn(
            f"block_hours={block_hours} does not divide 24 evenly; the last "
            "block will be narrower.",
            stacklevel=2,
        )

    _require_speed_mps(matched, "aggregate_speeds")
    df = matched.copy()
    df = df[~df["speed_mps"].isna()]
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
            s = float(np.std(sp, ddof=1)) if n > 1 else 0.0
            sem = s / np.sqrt(n) if n > 0 else np.nan
            rec["mean_speed"] = from_mps(m, unit)
            rec["std_speed"] = from_mps(s, unit)
            rec["sem_speed"] = from_mps(sem, unit)
            rec["ci95_low"] = from_mps(m - 1.96 * sem, unit)
            rec["ci95_high"] = from_mps(m + 1.96 * sem, unit)
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
    col = "mean_speed" if statistic == "mean" else "median_speed"
    if col not in aggregated.columns:
        raise ValueError(
            f"Column {col!r} not in aggregation; re-run aggregate_speeds with "
            f"statistic='{statistic}' or 'both'."
        )
    ordered = aggregated.sort_values(col).reset_index(drop=True)
    peak = ordered.head(n_peak)          # slowest = peak congestion
    offpeak = ordered.tail(n_offpeak)    # fastest = free flow
    return {
        "speed_column": col,
        "peak": peak.to_dict("records"),
        "off_peak": offpeak.to_dict("records"),
        "ranked": ordered[["block_label", col, "n"]].to_dict("records"),
    }


def classify_hours(
    matched: pd.DataFrame,
    *,
    statistic: str = "median",
    peak_hours=None,
    offpeak_hours=None,
    min_samples: int = 1,
) -> dict:
    """Split the 24 hours of the day into a peak and an off-peak block.

    By default the split is **data-driven**: each hour-of-day's network-wide
    representative speed (median or mean) is computed, and hours at or below the
    median of those hourly speeds are labelled *peak* (slower = busier), the rest
    *off-peak*. Hours with no observations are left unassigned.

    Pass ``peak_hours`` and/or ``offpeak_hours`` (iterables of 0-23) to override.
    If only one is given, the other becomes the remaining hours of the day.

    Returns
    -------
    dict
        ``peak_hours`` (sorted list), ``offpeak_hours`` (sorted list),
        ``threshold_speed_mps`` (the split point, or None when overridden), and
        ``source`` (``"override"`` or ``"auto"``).
    """
    if peak_hours is not None or offpeak_hours is not None:
        all_h = set(range(24))
        if peak_hours is not None and offpeak_hours is not None:
            peak = {int(h) for h in peak_hours}
            off = {int(h) for h in offpeak_hours}
        elif peak_hours is not None:
            peak = {int(h) for h in peak_hours}
            off = all_h - peak
        else:
            off = {int(h) for h in offpeak_hours}
            peak = all_h - off
        return {
            "peak_hours": sorted(peak),
            "offpeak_hours": sorted(off),
            "threshold_speed_mps": None,
            "source": "override",
        }

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
    threshold = float(np.median(speeds))
    peak = sorted(int(h) for h, s in zip(hours, speeds) if s <= threshold)
    off = sorted(int(h) for h, s in zip(hours, speeds) if s > threshold)
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
    default_speed_mps: Optional[float] = None,
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

    Peak/off-peak hours are auto-detected via :func:`classify_hours` unless you
    pass ``peak_hours`` / ``offpeak_hours``.

    Parameters
    ----------
    statistic : {"median", "mean"}
        Per-edge summary statistic.
    default_speed_mps : float, optional
        Fallback speed for edges with no observations in a given regime.
    min_samples : int
        Passed to auto hour classification.

    Returns
    -------
    dict
        ``peak_hours``, ``offpeak_hours``, ``threshold_speed_mps``,
        ``n_edges_total`` and ``coverage`` (observed-edge counts per regime).
    """
    cls = classify_hours(
        matched, statistic=statistic, peak_hours=peak_hours,
        offpeak_hours=offpeak_hours, min_samples=min_samples,
    )
    peak_set = set(cls["peak_hours"])
    off_set = set(cls["offpeak_hours"])

    df = matched.copy()
    df = df[(df["edge_id"] != -1) & (~df["speed_mps"].isna())]
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
            observed = spd is not None and spd > 0
            if not observed:
                spd = default_speed_mps
            if spd and spd > 0:
                data[f"obs_speed_mps_{r}"] = float(spd)
                data[f"travel_time_s_{r}"] = data["length_m"] / float(spd)
            else:
                data.pop(f"obs_speed_mps_{r}", None)
                data.pop(f"travel_time_s_{r}", None)
            if observed:
                coverage[r] += 1
        # Keep the plain attributes pointing at the overall regime for
        # back-compatible default routing.
        if "obs_speed_mps_overall" in data:
            data["obs_speed_mps"] = data["obs_speed_mps_overall"]
            data["travel_time_s"] = data["travel_time_s_overall"]
        else:
            data.pop("obs_speed_mps", None)
            data.pop("travel_time_s", None)

    if coverage["peak"] == 0 or coverage["offpeak"] == 0:
        import warnings
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
        "n_edges_total": network.graph.number_of_edges(),
        "coverage": coverage,
    }


def assign_speeds(
    network,
    matched: pd.DataFrame,
    *,
    statistic: str = "median",
    output_unit="mps",
    default_speed_mps: Optional[float] = None,
    block_hours: int = 24,
    target_hour: Optional[int] = None,
) -> dict:
    """Compute a representative speed per edge and write it onto the graph.

    Stores the result as edge attribute ``obs_speed_mps`` (always m/s for use by
    the router) and ``travel_time_s = length_m / obs_speed_mps``. Edges with no
    observations get ``default_speed_mps`` if provided, else are left without a
    travel time (router can fall back to length).

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
        ``{"n_edges_observed": int, "n_edges_total": int}``.
    """
    _require_speed_mps(matched, "assign_speeds")
    df = matched.copy()
    df = df[(df["edge_id"] != -1) & (~df["speed_mps"].isna())]
    if target_hour is not None:
        t = pd.to_datetime(df["time"])
        bs = (target_hour // block_hours) * block_hours
        be = bs + block_hours
        h = t.dt.hour.values
        df = df[(h >= bs) & (h < be)]

    agg_fn = np.median if statistic == "median" else np.mean
    per_edge = df.groupby("edge_id")["speed_mps"].apply(
        lambda s: float(agg_fn(s.values))
    )

    observed = 0
    for u, v, data in network.graph.edges(data=True):
        eid = data["edge_id"]
        spd = per_edge.get(eid, None)
        if spd is None or spd <= 0:
            spd = default_speed_mps
        if spd and spd > 0:
            data["obs_speed_mps"] = float(spd)
            data["travel_time_s"] = data["length_m"] / float(spd)
            if eid in per_edge.index:
                observed += 1
        else:
            data.pop("obs_speed_mps", None)
            data.pop("travel_time_s", None)
    return {
        "n_edges_observed": observed,
        "n_edges_total": network.graph.number_of_edges(),
    }
