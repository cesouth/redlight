"""Outlier and threshold filtering of speed observations.

Two complementary strategies:

1. Hard physical bounds (``min_speed`` / ``max_speed``): drop observations that
   are implausible for the study (e.g. parked vehicles at 0, GPS-jump
   artefacts above a sane ceiling). Bounds are specified in the unit you pass.

2. Robust statistical outlier removal (optional): the modified Z-score based on
   the median absolute deviation (MAD). MAD is preferred over mean/std because
   it has a ~50% breakdown point, so a few wild GPS errors do not inflate the
   threshold the way the standard deviation would. A common cutoff is
   |M_i| > 3.5 (Iglewicz & Hoaglin, 1993):

       M_i = 0.6745 * (x_i - median(x)) / MAD,
       MAD = median(|x_i - median(x)|).

   The 0.6745 factor scales MAD to be a consistent estimator of the standard
   deviation for normally distributed data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._geo import GEOD_WGS84
from .aggregate import _require_speed_mps
from .units import SpeedUnit, to_mps


def filter_by_speed(
    matched: pd.DataFrame,
    *,
    min_speed: float | None = None,
    max_speed: float | None = None,
    unit="mph",
    drop_unmatched: bool = True,
    mad_outliers: bool = False,
    mad_threshold: float = 3.5,
    per_edge: bool = False,
) -> pd.DataFrame:
    """Filter a matched-observations DataFrame on speed.

    Parameters
    ----------
    matched : DataFrame
        Output of a matcher (must contain ``speed_mps``; ``edge_id`` if using
        ``drop_unmatched`` or ``per_edge``).
    min_speed, max_speed : float, optional
        Hard bounds in ``unit``. Observations strictly below ``min_speed`` or
        strictly above ``max_speed`` are removed.
    unit : str or SpeedUnit
        Unit the bounds are expressed in.
    drop_unmatched : bool
        Drop rows with ``edge_id == -1`` (points that snapped to nothing).
    mad_outliers : bool
        If True, also drop modified-Z-score outliers.
    mad_threshold : float
        Modified Z-score cutoff (default 3.5).
    per_edge : bool
        Apply MAD outlier removal within each edge group rather than globally.
        Recommended when edges have very different free-flow speeds.

    Returns
    -------
    DataFrame
        Filtered copy with a reset index.
    """
    _require_speed_mps(matched, "filter_by_speed")
    unit = SpeedUnit.parse(unit)
    df = matched.copy()

    if drop_unmatched and "edge_id" in df.columns:
        df = df[df["edge_id"] != -1]

    if min_speed is not None:
        df = df[df["speed_mps"] >= to_mps(min_speed, unit)]
    if max_speed is not None:
        df = df[df["speed_mps"] <= to_mps(max_speed, unit)]

    if mad_outliers and len(df) > 0:
        if per_edge and "edge_id" in df.columns:
            masks = []
            for _, g in df.groupby("edge_id", sort=False):
                m = _mad_mask(g["speed_mps"].values, mad_threshold)
                masks.append(pd.Series(m, index=g.index))
            keep_mask = pd.concat(masks).reindex(df.index).fillna(False)
            df = df[keep_mask.values]
        else:
            df = df[_mad_mask(df["speed_mps"].values, mad_threshold)]

    return df.reset_index(drop=True)


def filter_trajectory_speed(
    matched: pd.DataFrame,
    *,
    drop_unmatched: bool = True,
    drop_missing_speed: bool = True,
    dwell_radius_m: float = 25.0,
    dwell_min_s: float = 120.0,
    max_speed: float | None = None,
    unit="mph",
) -> pd.DataFrame:
    """Trajectory-aware cleaning for HMM-matched observations.

    Unlike :func:`filter_by_speed`, this **does not drop points merely for being
    slow** -- doing so would delete the very congestion a trafficability study
    exists to measure and bias every segment's speed upward. Instead it uses the
    trajectory (a ``traj_id`` and the carried ``lon``/``lat``) to separate two
    cases:

    * **Genuine travel that happens to be slow** (a vehicle creeping through
      congestion). The vehicle still makes ground over time, so it is **kept**,
      low speed and all.
    * **Stationary dwells** (parked / idling / GPS jitter while stopped). The
      vehicle stays inside ``dwell_radius_m`` for at least ``dwell_min_s``; these
      points are not travel on the segment and are **dropped**.

    A dwell is a maximal run of consecutive same-trajectory points (time-ordered)
    all within ``dwell_radius_m`` (geodesic) of the run's first point, spanning at
    least ``dwell_min_s``. With the defaults (25 m, 120 s) a vehicle must average
    below ~0.2 m/s (~0.47 mph) to be treated as stopped, so ordinary signal stops
    and stop-and-go congestion survive while true parking is removed. Raise
    ``dwell_min_s`` to keep longer operational stops; lower it to be stricter.

    Parameters
    ----------
    matched : DataFrame
        Matcher output. Must carry ``traj_id``, ``time``, ``lon``, ``lat`` and
        ``speed_mps`` (both matchers now emit ``lon``/``lat``).
    drop_unmatched : bool
        Drop rows with ``edge_id == -1``.
    drop_missing_speed : bool
        Drop rows with no logged speed (``speed_mps`` is NaN).
    dwell_radius_m, dwell_min_s : float
        Stationary-cluster thresholds described above.
    max_speed : float, optional
        Optional hard upper bound (in ``unit``) to discard GPS-jump artefacts.
        There is deliberately no ``min_speed`` here -- see above.
    unit : str or SpeedUnit
        Unit for ``max_speed``.

    Returns
    -------
    DataFrame
        Filtered copy with a reset index.
    """
    required = {"traj_id", "time", "lon", "lat", "speed_mps"}
    missing = required - set(matched.columns)
    if missing:
        raise ValueError(
            "filter_trajectory_speed needs columns "
            f"{sorted(required)}; missing {sorted(missing)}. Match with a "
            "trajectory id (load_points(id_col=...)) so lon/lat and traj_id are "
            "present, or use filter_by_speed for independent points."
        )

    df = matched.copy()
    if drop_unmatched and "edge_id" in df.columns:
        df = df[df["edge_id"] != -1]
    if max_speed is not None:
        df = df[df["speed_mps"] <= to_mps(max_speed, SpeedUnit.parse(unit))]
    df = df.reset_index(drop=True)
    if len(df) == 0:
        return df

    time = pd.to_datetime(df["time"])
    # Force ns resolution before the int cast: a bare astype("int64") yields the
    # datetime64's native unit (us/ns/s vary by pandas version), which would
    # silently mis-scale dwell durations.
    t = time.to_numpy(dtype="datetime64[ns]").astype("float64") / 1e9  # seconds
    t[time.isna().to_numpy()] = np.nan
    lon = df["lon"].to_numpy(float)
    lat = df["lat"].to_numpy(float)

    dwell = np.zeros(len(df), dtype=bool)
    for idx in df.groupby("traj_id", sort=False).indices.values():
        idx = np.asarray(idx)
        order = np.argsort(t[idx], kind="stable")
        sidx = idx[order]
        dwell[sidx] = _dwell_mask(
            lon[sidx], lat[sidx], t[sidx], dwell_radius_m, dwell_min_s
        )

    keep = ~dwell
    if drop_missing_speed:
        keep &= ~df["speed_mps"].isna().to_numpy()
    return df[keep].reset_index(drop=True)


def _dwell_mask(lon, lat, t, radius_m, min_s) -> np.ndarray:
    """Boolean mask (True = stationary dwell) for one time-ordered trajectory."""
    n = len(lon)
    mask = np.zeros(n, dtype=bool)
    i = 0
    while i < n:
        j = i
        while j + 1 < n:
            _, _, dist = GEOD_WGS84.inv(lon[i], lat[i], lon[j + 1], lat[j + 1])
            if not np.isfinite(dist) or dist > radius_m:
                break
            j += 1
        dur = t[j] - t[i]
        if j > i and np.isfinite(dur) and dur >= min_s:
            mask[i:j + 1] = True
            i = j + 1
        else:
            i += 1
    return mask


def _mad_mask(x: np.ndarray, threshold: float) -> np.ndarray:
    """Boolean mask of non-outliers by modified Z-score (MAD-based)."""
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad == 0:
        # Degenerate spread: fall back to keeping everything (no robust signal).
        return np.ones(len(x), dtype=bool)
    mz = 0.6745 * (x - med) / mad
    return np.abs(mz) <= threshold
