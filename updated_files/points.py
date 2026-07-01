"""GPS point ingestion and the PointSet container.

A *point file* is any delimited text file (CSV/TSV) or GeoJSON FeatureCollection
of Point geometries carrying, per observation:

  - longitude / latitude (WGS84 degrees),
  - a timestamp,
  - a speed value (mph, kph or m/s),
  - optionally a unique trajectory id (required for HMM map matching).

The loader normalises these into a pandas DataFrame with canonical columns:

  ``point_id`` (int, row index), ``traj_id`` (object, optional),
  ``lon`` (float), ``lat`` (float), ``time`` (datetime64[ns, UTC] or naive),
  ``speed_mps`` (float).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .units import SpeedUnit, to_mps


@dataclass
class PointSet:
    """Normalised GPS observations.

    Attributes
    ----------
    df : pandas.DataFrame
        Canonical columns: point_id, traj_id (optional), lon, lat, time,
        speed_mps. Additional source columns are preserved.
    has_traj : bool
        Whether a trajectory-id column is present (needed for HMM matching).
    """

    df: pd.DataFrame
    has_traj: bool

    def __len__(self) -> int:
        return len(self.df)

    def trajectories(self):
        """Yield (traj_id, sub-DataFrame sorted by time) for each trajectory.

        If no trajectory id is present the whole set is yielded once under id
        ``None``. Sub-frames are time-sorted, which HMM matching requires.
        """
        if self.has_traj:
            for tid, sub in self.df.groupby("traj_id", sort=False):
                yield tid, sub.sort_values("time").reset_index(drop=True)
        else:
            yield None, self.df.sort_values("time").reset_index(drop=True)

    def copy(self) -> "PointSet":
        return PointSet(self.df.copy(), self.has_traj)


def _autodetect(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def load_points(
    path: str,
    *,
    speed_unit="mph",
    lon_col: Optional[str] = None,
    lat_col: Optional[str] = None,
    time_col: Optional[str] = None,
    speed_col: Optional[str] = None,
    id_col: Optional[str] = None,
    timestamp_unit: Optional[str] = None,
    sep: Optional[str] = None,
) -> PointSet:
    """Load a GPS point file into a :class:`PointSet`.

    Parameters
    ----------
    path : str
        CSV/TSV (``.csv``/``.tsv``/``.txt``) or GeoJSON (``.geojson``/``.json``).
    speed_unit : str or SpeedUnit
        Unit the source speed column is expressed in: ``"mph"``, ``"kph"`` or
        ``"mps"``. Converted internally to m/s.
    lon_col, lat_col, time_col, speed_col, id_col : str, optional
        Column names. If omitted, common names are auto-detected (e.g. ``lon``,
        ``longitude``, ``x``; ``timestamp``, ``time``, ``datetime``; ``speed``;
        ``id``, ``uid``, ``track_id``). For GeoJSON, geometry supplies lon/lat.
    timestamp_unit : str, optional
        If timestamps are numeric epoch values, the unit to interpret them in
        (``"s"``, ``"ms"``, ``"us"``, ``"ns"``). Otherwise parsed as datetimes.
    sep : str, optional
        Delimiter for text files; inferred from extension if omitted.

    Returns
    -------
    PointSet
    """
    unit = SpeedUnit.parse(speed_unit)
    lower = path.lower()
    if lower.endswith((".geojson",)) or lower.endswith(".json"):
        df, has_geom_coords = _read_geojson_points(path)
    else:
        if sep is None:
            sep = "\t" if lower.endswith((".tsv", ".tab")) else ","
        df = pd.read_csv(path, sep=sep)
        has_geom_coords = False

    cols = list(df.columns)
    if not has_geom_coords:
        lon_col = lon_col or _autodetect(cols, ["lon", "longitude", "lng", "x", "long"])
        lat_col = lat_col or _autodetect(cols, ["lat", "latitude", "y"])
        if lon_col is None or lat_col is None:
            raise ValueError(
                "Could not find longitude/latitude columns. "
                "Pass lon_col= and lat_col= explicitly."
            )
    time_col = time_col or _autodetect(
        cols, ["time", "timestamp", "datetime", "date_time", "t", "utc"]
    )
    speed_col = speed_col or _autodetect(
        cols, ["speed", "speed_mph", "speed_kph", "speed_mps", "velocity", "spd"]
    )
    id_col = id_col or _autodetect(
        cols, ["traj_id", "track_id", "trip_id", "id", "uid", "unit_id", "device_id"]
    )
    if time_col is None:
        raise ValueError("Could not find a timestamp column. Pass time_col=.")
    # speed_col is optional: GPS that records position+time only is fine; speed
    # can be derived downstream from on-road displacement (see roadtraffic.speeds).

    out = pd.DataFrame()
    out["point_id"] = np.arange(len(df), dtype=np.int64)
    has_traj = id_col is not None
    if has_traj:
        out["traj_id"] = df[id_col].values
    if has_geom_coords:
        out["lon"] = df["lon"].astype(float).values
        out["lat"] = df["lat"].astype(float).values
    else:
        out["lon"] = pd.to_numeric(df[lon_col], errors="coerce").values
        out["lat"] = pd.to_numeric(df[lat_col], errors="coerce").values

    if timestamp_unit is not None:
        out["time"] = pd.to_datetime(
            pd.to_numeric(df[time_col], errors="coerce"), unit=timestamp_unit
        )
    else:
        out["time"] = pd.to_datetime(df[time_col], errors="coerce", utc=False)

    raw_speed = (pd.to_numeric(df[speed_col], errors="coerce").values
                 if speed_col is not None else None)
    if raw_speed is not None:
        out["speed_mps"] = to_mps(raw_speed, unit)

    required = ["lon", "lat", "time"] + (["speed_mps"] if speed_col is not None else [])
    n_before = len(out)
    out = out.dropna(subset=required).reset_index(drop=True)
    out["point_id"] = np.arange(len(out), dtype=np.int64)
    dropped = n_before - len(out)
    if dropped:
        import warnings

        warnings.warn(
            f"Dropped {dropped} row(s) with missing/unparseable "
            + ("lon/lat/time/speed." if speed_col is not None else "lon/lat/time."),
            stacklevel=2,
        )
    return PointSet(out, has_traj)


def _read_geojson_points(path: str):
    with open(path, "r", encoding="utf-8") as fh:
        gj = json.load(fh)
    feats = gj.get("features", []) if isinstance(gj, dict) else []
    rows = []
    for feat in feats:
        geom = (feat or {}).get("geometry") or {}
        if geom.get("type") != "Point":
            continue
        coords = geom.get("coordinates") or [None, None]
        props = dict(feat.get("properties") or {})
        props["lon"] = coords[0]
        props["lat"] = coords[1]
        rows.append(props)
    if not rows:
        raise ValueError("GeoJSON contained no Point features.")
    return pd.DataFrame(rows), True
