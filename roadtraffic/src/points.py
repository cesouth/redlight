"""GPS point ingestion and the PointSet container.

A *point file* is any delimited text file (CSV/TSV) or GeoJSON FeatureCollection
of Point geometries carrying, per observation:

  - longitude / latitude (WGS84 degrees),
  - a timestamp,
  - a speed value (mph, kph or m/s) -- optional if ``derive_speed=True``,
  - optionally a unique trajectory id (required for HMM map matching, and for
    deriving speed from positions).

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
from pyproj import Geod

from .units import SpeedUnit, from_mps, to_mps


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

    def to_frame(self, *, speed_unit="mph") -> pd.DataFrame:
        """Return a tidy DataFrame with speed in both m/s and ``speed_unit``.

        Columns: ``point_id``, ``traj_id`` (if present), ``lon``, ``lat``,
        ``time`` (ISO 8601), ``speed_mps`` and ``speed_<unit>``. This is the
        canonical, save-ready view -- handy after ``derive_speed=True`` to persist
        the computed speeds.
        """
        unit = SpeedUnit.parse(speed_unit)
        cols = ["point_id"]
        if self.has_traj and "traj_id" in self.df.columns:
            cols.append("traj_id")
        cols += ["lon", "lat", "time", "speed_mps"]
        out = self.df[cols].copy()
        out["time"] = pd.to_datetime(out["time"]).map(
            lambda x: x.isoformat() if pd.notna(x) else ""
        )
        out[f"speed_{unit.value}"] = from_mps(out["speed_mps"].to_numpy(float), unit)
        return out

    def to_csv(self, path: str, *, speed_unit="mph") -> str:
        """Write the point set (incl. speed) to CSV. Returns ``path``."""
        self.to_frame(speed_unit=speed_unit).to_csv(path, index=False)
        return path

    def to_geojson(self, path: str, *, speed_unit="mph") -> str:
        """Write the point set to a GeoJSON Point FeatureCollection."""
        frame = self.to_frame(speed_unit=speed_unit)
        feats = []
        for rec in frame.to_dict("records"):
            lon = rec.pop("lon")
            lat = rec.pop("lat")
            feats.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": rec,
            })
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"type": "FeatureCollection", "features": feats}, fh)
        return path


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
    derive_speed: bool = False,
) -> PointSet:
    """Load a GPS point file into a :class:`PointSet`.

    Parameters
    ----------
    path : str
        CSV/TSV (``.csv``/``.tsv``/``.txt``) or GeoJSON (``.geojson``/``.json``).
    speed_unit : str or SpeedUnit
        Unit the source speed column is expressed in: ``"mph"``, ``"kph"`` or
        ``"mps"``. Converted internally to m/s. Ignored when
        ``derive_speed=True``.
    derive_speed : bool
        If True, compute speed from successive GPS positions instead of reading a
        speed column (opt-in). Requires a unique-id column (``id_col``) so speeds
        are never differenced across distinct movements, and timestamps. Speed is
        the ellipsoidal (geodesic) distance between consecutive same-id points
        divided by their time gap; see ``docs/statistics.md`` for the method.
        When False (default), a speed column must be present or a ``ValueError``
        is raised.
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
    has_traj = id_col is not None
    if derive_speed:
        if not has_traj:
            raise ValueError(
                "derive_speed=True needs a unique-id column so speeds are never "
                "differenced across distinct movements. Pass id_col= (or include "
                "a recognisable id column)."
            )
    elif speed_col is None:
        raise ValueError(
            "Could not find a speed column. Pass speed_col=, or set "
            "derive_speed=True to compute speed from successive GPS positions "
            "(needs a unique-id column)."
        )

    out = pd.DataFrame()
    out["point_id"] = np.arange(len(df), dtype=np.int64)
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

    if derive_speed:
        out["speed_mps"] = _derive_speed_mps(out)
    else:
        raw_speed = pd.to_numeric(df[speed_col], errors="coerce").values
        out["speed_mps"] = to_mps(raw_speed, unit)

    n_before = len(out)
    out = out.dropna(subset=["lon", "lat", "time", "speed_mps"]).reset_index(drop=True)
    out["point_id"] = np.arange(len(out), dtype=np.int64)
    dropped = n_before - len(out)
    if dropped:
        import warnings

        warnings.warn(
            f"Dropped {dropped} row(s) with missing/unparseable "
            "lon/lat/time/speed.",
            stacklevel=2,
        )
    return PointSet(out, has_traj)


def save_points(points: PointSet, path: str, *, speed_unit="mph") -> str:
    """Write a :class:`PointSet` to disk, format chosen by extension.

    ``.geojson``/``.json`` -> GeoJSON Point features; anything else -> CSV. Speed
    is written in both m/s (``speed_mps``) and ``speed_unit`` (``speed_<unit>``),
    so a set loaded with ``derive_speed=True`` can be persisted with its computed
    speeds. Returns ``path``.
    """
    low = path.lower()
    if low.endswith((".geojson", ".json")):
        return points.to_geojson(path, speed_unit=speed_unit)
    return points.to_csv(path, speed_unit=speed_unit)


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


_GEOD_WGS84 = Geod(ellps="WGS84")


def _derive_speed_mps(out: pd.DataFrame) -> np.ndarray:
    """Speed (m/s) at each point, differenced within each trajectory.

    Within a single ``traj_id`` (sorted by time) the speed at point ``i`` is the
    ellipsoidal distance from point ``i-1`` to ``i`` divided by their time gap --
    i.e. the *average* speed over the interval ending at ``i``. The first point of
    each trajectory inherits the speed of its first interval. Intervals with a
    non-positive time gap (duplicate/out-of-order timestamps) or a missing
    coordinate/time yield NaN, and such points are dropped downstream.

    Differencing is strictly per-trajectory so two distinct movements are never
    connected. WGS84 lon/lat are used directly with a geodesic (Geod) inverse, so
    no projection is needed and distances are true ground metres.
    """
    n = len(out)
    speed = np.full(n, np.nan)
    if n == 0:
        return speed
    lon = out["lon"].to_numpy(dtype=float)
    lat = out["lat"].to_numpy(dtype=float)
    t = pd.to_datetime(out["time"]).to_numpy(dtype="datetime64[ns]").astype("float64")
    t[pd.isna(out["time"]).to_numpy()] = np.nan  # NaT -> NaN seconds

    if "traj_id" in out.columns:
        groups = out.groupby("traj_id", sort=False).indices.values()
    else:
        groups = [np.arange(n)]

    for idx in groups:
        idx = np.asarray(idx)
        if idx.size < 2:
            continue
        order = np.argsort(t[idx], kind="stable")  # time order; NaN sorts last
        sidx = idx[order]
        _, _, dist = _GEOD_WGS84.inv(
            lon[sidx[:-1]], lat[sidx[:-1]], lon[sidx[1:]], lat[sidx[1:]]
        )
        dt = (t[sidx[1:]] - t[sidx[:-1]]) / 1e9  # ns -> s
        with np.errstate(divide="ignore", invalid="ignore"):
            seg = np.where(dt > 0, dist / dt, np.nan)
        s = np.empty(sidx.size)
        s[1:] = seg
        s[0] = seg[0]  # first point inherits first interval's speed
        speed[sidx] = s
    return speed
