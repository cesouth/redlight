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

from .units import SpeedUnit, from_mps, to_mps

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
        # Built from the same rule the populated path below uses, so an emptied
        # feed cannot change the shape of the answer. Hardcoding the list here
        # instead let the two branches disagree about snap_dist_m, and a caller
        # reading that column got a KeyError only once its data ran out.
        cols = ["n_intervals", pct_col, med_col, "distance_m"]
        if "snap_dist_m" in df.columns:
            cols.append("snap_dist_m")
        empty = pd.DataFrame({c: pd.Series(dtype=float) for c in cols})
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
# A walking hump has to carry real mass to be a population rather than a ripple
# in the left tail. Measured on vehicle-only feeds, the spurious "humps" that
# used to pass the location check sat at ~1% of the peak density.
_MIN_HUMP_SHARE = 0.15


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
        ``None`` when no walking-speed population can be split off, which is
        the honest answer for a single-mode feed. Callers must not substitute
        a default: a silently chosen threshold that is wrong produces a study
        that looks correct.

        Note that ``None`` also covers a walking population that is real but
        too *small* to identify. Guard 3 below asks the walking hump to carry
        mass, and a modest minority does not clear it -- measured over eight
        seeds on synthetic mixes (vehicles N(28, 6) mph, walkers N(3, 0.8) mph,
        n = 300), a 20% minority is found every time, 15% is marginal, and 10%
        or below is never found. The exact floor depends on how well separated
        the two populations are, so treat 20% as the reliable figure rather
        than a guarantee. That is deliberate, since the same guard is what
        stops a vehicle-only feed's left-tail ripple from inventing pedestrians
        out of gridlocked vehicles. If a small walking minority is known to be present,
        pass an explicit threshold rather than reading ``None`` as its absence.

    Notes
    -----
    The density is estimated in **log** speed. A fixed relative bandwidth on
    raw speed is set by the spread of the whole sample, and the driving hump
    runs out to motorway speeds -- wide enough to smooth away a valley only a
    couple of mph across. Log speed makes the bandwidth scale-free, so
    resolution near walking pace does not depend on the fastest vehicle.

    Three guards decide whether a valley is a *mode* boundary at all:

    1. Candidates are ranked by **prominence**, ``min(left_peak, right_peak) /
       valley``, and the density is evaluated well beyond the selection window
       in both directions. Ranking by absolute depth instead selects the lowest
       point of a monotone tail at the window edge, which is not a valley.
    2. There must be a **hump** below the candidate, not merely a lower value:
       an interior local maximum, a place where the density turns over. On a
       monotone left tail the grid point just below the candidate always sits
       just below it, so a position test alone verifies nothing.
    3. That hump must be **at walking pace and carry mass** -- located between
       ``_WALK_MODE_LO_MPS`` and ``_WALK_MODE_HI_MPS``, and at least
       ``_MIN_HUMP_SHARE`` of the peak density. A vehicle-only feed has interior
       valleys of its own and ripples in its left tail at a fraction of a
       percent of the peak; without both halves of this guard the boundary
       between gridlock and free flow is accepted as a mode split.
    """
    unit = SpeedUnit.parse(unit)
    x = np.asarray(pd.Series(mover_speeds).to_numpy(), dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    if len(x) < _MIN_MOVERS:
        return None
    x = to_mps(x, unit)
    try:
        from scipy.stats import gaussian_kde
    except ImportError:  # pragma: no cover - scipy is a core dependency
        return None

    lo = _SEARCH_LO_MPS
    hi = _SEARCH_HI_MPS
    walk_lo = _WALK_MODE_LO_MPS
    walk_hi = _WALK_MODE_HI_MPS

    grid = np.linspace(np.log(lo / 4.0), np.log(hi * 8.0), 1200)
    try:
        dens = gaussian_kde(np.log(x), bw_method=0.20)(grid)
    except np.linalg.LinAlgError:
        # Every mover at one speed (a quantised or synthetic field) gives a
        # singular covariance. That distribution is a spike, not two humps with
        # a valley between them, so there is nothing to find -- report the same
        # "no walking population" answer rather than aborting the caller.
        return None

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

    peak_density = float(dens.max())

    def lower_hump(i: int) -> int | None:
        """Index of the tallest interior local maximum strictly below ``i``.

        Deliberately not ``argmax(dens[:i])``. On a monotone left tail that
        returns the grid point immediately below the candidate, whose position
        is by construction just under the candidate's own -- so a candidate
        anywhere inside the walking band satisfies a "the hump below is at
        walking pace" test automatically, and the test verifies nothing. A hump
        is a place where the density turns over, so require that it does.
        """
        below = np.arange(1, i)
        if not len(below):
            return None
        turns_over = ((dens[below] > dens[below - 1])
                      & (dens[below] > dens[below + 1]))
        humps = below[turns_over]
        if not len(humps):
            return None
        return int(humps[np.argmax(dens[humps])])

    def has_walking_hump(i: int) -> bool:
        j = lower_hump(i)
        if j is None:
            return False
        if not (walk_lo <= float(np.exp(grid[j])) <= walk_hi):
            return False
        # Location alone is not enough: a vehicle-only feed ripples in its far
        # left tail at a fraction of a percent of the peak, and treating that as
        # a walking population invents pedestrians -- out of noise, and worse,
        # out of genuinely gridlocked vehicles, which is the very congestion the
        # study exists to measure.
        return dens[j] >= _MIN_HUMP_SHARE * peak_density

    viable = [i for i in candidates if prominence(i) >= _MIN_PROMINENCE]
    walkers = [i for i in viable if has_walking_hump(i)]
    if not walkers:
        return None
    return float(from_mps(np.exp(grid[max(walkers, key=prominence)]), unit))


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
    kept = movers.index[movers["mode"].isin(wanted)]
    mask = obs["traj_id"].isin(set(kept[kept.notna()]))
    # A frame loaded without a trajectory id carries traj_id = None, which
    # groupby collapses into a single null-indexed mover. Nulls are not equal to
    # each other, so `isin` matches that mover against zero rows and a "keep"
    # verdict silently deletes all of its data -- and dtype-dependently, since
    # an object column of None kept nothing where a float column of NaN kept
    # everything. Match the null-id mover explicitly instead.
    if kept.isna().any():
        mask = mask | obs["traj_id"].isna()
    out = obs[mask].reset_index(drop=True)
    if not len(out):
        warnings.warn(
            f"filter_by_mode removed every observation: no mover matched "
            f"keep={sorted(wanted)}. Lower the threshold, or widen keep=.",
            UserWarning, stacklevel=2)
    return out
