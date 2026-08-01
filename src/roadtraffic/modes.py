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
    return float(from_mps(np.exp(grid[max(walkers, key=prominence)]), unit))
