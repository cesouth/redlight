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
