"""GPS point ingestion and the PointSet container.

A *point file* is any delimited text file (CSV/TSV) or GeoJSON FeatureCollection
of Point geometries carrying, per observation:

  - longitude / latitude (WGS84 degrees),
  - a timestamp,
  - a speed value (mph, kph or m/s) -- optional,
  - optionally a unique trajectory id (required for HMM map matching, and for
    deriving speed from positions).

The loader normalises these into a pandas DataFrame with canonical columns:

  ``point_id`` (int, row index), ``traj_id`` (object, optional),
  ``lon`` (float), ``lat`` (float), ``time`` (datetime64, timezone-naive),
  ``speed_mps`` (float, optional).

Timezones
---------
Hour-of-day statistics (peak / off-peak detection) are only meaningful on the
**local clock** of the study area. ``load_points`` therefore normalises times
to timezone-naive values:

  - timezone-aware timestamps (e.g. ISO strings with ``Z`` or an offset, or
    numeric epochs, which are UTC by definition) are converted to the zone
    given by ``tz=`` and the zone info is dropped;
  - if aware timestamps arrive and no ``tz`` is given, a warning explains that
    hour-of-day statistics will use that clock (usually UTC);
  - naive timestamps are assumed to already be local, unless ``tz`` is given,
    in which case they are treated as UTC and converted.

Speed
-----
Speed is not always required at load time. Three ways to end up with a
:class:`PointSet`:

1. **Speed column present** (default). Read and converted to m/s. If the
   column name embeds a unit (``speed_kph``, ``speed_mps``, ``speed_mph``)
   that unit is used automatically; an explicit ``speed_unit=`` always wins
   (with a warning if the two disagree).
2. ``derive_speed=True``. No speed column needed; speed is reconstructed
   per point from the *straight-line* (geodesic) distance to the previous
   same-trajectory fix. Cheap, but biased low on curves and by GPS noise
   over short gaps -- see ``docs/statistics.md`` §7.
3. **Neither.** Position + time only. ``speed_mps`` is omitted entirely from
   ``.df``. Pair this with a matcher (:mod:`roadtraffic.matching`) and
   :func:`roadtraffic.speeds.derive_speeds`, which reconstructs speed from
   *on-road* displacement after map matching -- more accurate than (2) for
   noisy/sparse fixes, at the cost of needing a network and a matching step
   first. See ``docs/statistics.md`` §10.
"""
from __future__ import annotations

import json
import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ._geo import geodesic_distance
from .units import _SPEED_UNIT_ALIASES, SpeedUnit, from_mps, to_mps


@dataclass
class PointSet:
    """Normalised GPS observations.

    Attributes
    ----------
    df : pandas.DataFrame
        Canonical columns: point_id, traj_id (optional), lon, lat, time,
        speed_mps (optional -- absent when loaded with no speed column and
        ``derive_speed=False``). Additional source columns are preserved.
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

    def copy(self) -> PointSet:
        return PointSet(self.df.copy(), self.has_traj)

    def to_frame(self, *, speed_unit="mph") -> pd.DataFrame:
        """Return a tidy DataFrame with speed in both m/s and ``speed_unit``.

        Columns: ``point_id``, ``traj_id`` (if present), ``lon``, ``lat``,
        ``time`` (ISO 8601), and -- if the set carries speed --  ``speed_mps``
        and ``speed_<unit>``. This is the canonical, save-ready view -- handy
        after ``derive_speed=True`` to persist the computed speeds. Sets loaded
        with position+time only (no speed column, ``derive_speed=False``) omit
        the speed columns.
        """
        unit = SpeedUnit.parse(speed_unit)
        has_speed = "speed_mps" in self.df.columns
        cols = ["point_id"]
        if self.has_traj and "traj_id" in self.df.columns:
            cols.append("traj_id")
        cols += ["lon", "lat", "time"]
        if has_speed:
            cols.append("speed_mps")
        out = self.df[cols].copy()
        out["time"] = pd.to_datetime(out["time"]).map(
            lambda x: x.isoformat() if pd.notna(x) else ""
        )
        if has_speed:
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


def _autodetect(columns: Sequence[str], candidates: Sequence[str]) -> str | None:
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def _infer_speed_unit(colname: str) -> str | None:
    """Infer the unit embedded in a speed column name, if any.

    Recognizes every alias :meth:`~roadtraffic.units.SpeedUnit.parse` accepts
    (e.g. ``kmh``, ``kmph``, ``km/h``), not just the canonical mph/kph/mps
    tokens -- so an explicit ``speed_unit=`` and an auto-detected column name
    never silently disagree just because they spelled the same unit differently.
    """
    low = colname.strip().lower()
    for alias, canonical in _SPEED_UNIT_ALIASES.items():
        if low == alias or low.endswith(f"_{alias}") or low.endswith(f"({alias})"):
            return canonical
    return None


def _parse_times(values, timestamp_unit: str | None, tz: str | None) -> pd.Series:
    """Parse timestamps to a timezone-naive local-clock datetime Series.

    See the module docstring ("Timezones") for the exact semantics.
    """
    if timestamp_unit is not None:
        # numeric epochs are UTC by definition
        t = pd.to_datetime(
            pd.to_numeric(values, errors="coerce"), unit=timestamp_unit, utc=True
        )
        if tz is not None:
            t = t.dt.tz_convert(tz)
        else:
            warnings.warn(
                "Timestamps are numeric epochs (UTC by definition); hour-of-day "
                "statistics (peak/off-peak) will use UTC. Pass tz= to convert to "
                "the study area's local time.",
                stacklevel=3,
            )
        return t.dt.tz_localize(None)

    try:
        t = pd.to_datetime(values, errors="coerce", utc=False)
    except (ValueError, TypeError):
        # mixed UTC offsets (e.g. a DST change): parse as UTC, convert below
        t = pd.to_datetime(values, errors="coerce", utc=True)
    if t.dtype == object:  # older pandas: mixed offsets degrade to object
        t = pd.to_datetime(values, errors="coerce", utc=True)

    aware = getattr(t.dt, "tz", None) is not None
    if aware:
        if tz is not None:
            t = t.dt.tz_convert(tz)
        else:
            warnings.warn(
                f"Timestamps are timezone-aware ({t.dt.tz}); hour-of-day "
                "statistics (peak/off-peak) will use that clock. Pass tz= to "
                "convert to the study area's local time.",
                stacklevel=3,
            )
        t = t.dt.tz_localize(None)
    elif tz is not None:
        # naive input with an explicit tz: treat the values as UTC and convert
        t = t.dt.tz_localize("UTC").dt.tz_convert(tz).dt.tz_localize(None)
    return t


def load_points(
    path: str,
    *,
    speed_unit=None,
    lon_col: str | None = None,
    lat_col: str | None = None,
    time_col: str | None = None,
    speed_col: str | None = None,
    id_col: str | None = None,
    timestamp_unit: str | None = None,
    tz: str | None = None,
    sep: str | None = None,
    derive_speed: bool = False,
    keep_cols=None,
) -> PointSet:
    """Load a GPS point file into a :class:`PointSet`.

    Parameters
    ----------
    path : str
        CSV/TSV (``.csv``/``.tsv``/``.txt``) or GeoJSON (``.geojson``/``.json``).
    speed_unit : str or SpeedUnit, optional
        Unit the source speed column is expressed in: ``"mph"``, ``"kph"`` or
        ``"mps"``. If omitted (default), the unit is inferred from the column
        name when it embeds one (``speed_kph`` -> kph) and falls back to mph
        otherwise. An explicit value always wins; a warning is emitted if it
        contradicts the column name. Ignored when ``derive_speed=True`` or no
        speed column is found.
    derive_speed : bool
        If True, compute speed from successive GPS positions instead of reading a
        speed column (opt-in). Requires a unique-id column (``id_col``) so speeds
        are never differenced across distinct movements, and timestamps. Speed is
        the ellipsoidal (geodesic) distance between consecutive same-id points
        divided by their time gap; see ``docs/statistics.md`` §7 for the method
        and its limits, and §10 for the more accurate on-road alternative
        (:func:`roadtraffic.speeds.derive_speeds`, used after map matching).
        When False (default) and no speed column is found or given, the
        returned ``PointSet`` simply has no ``speed_mps`` column -- position and
        time are still valid on their own for matching.
    lon_col, lat_col, time_col, speed_col, id_col : str, optional
        Column names. If omitted, common names are auto-detected (e.g. ``lon``,
        ``longitude``, ``x``; ``timestamp``, ``time``, ``datetime``; ``speed``;
        ``id``, ``uid``, ``track_id``). For GeoJSON, geometry supplies lon/lat.
        ``speed_col`` is optional -- position+time-only data is valid input.
    timestamp_unit : str, optional
        If timestamps are numeric epoch values, the unit to interpret them in
        (``"s"``, ``"ms"``, ``"us"``, ``"ns"``). Otherwise parsed as datetimes.
    tz : str, optional
        IANA timezone of the study area (e.g. ``"America/New_York"``). Aware
        timestamps and numeric epochs are converted to it; naive timestamps
        are treated as UTC and converted. Stored times are always naive local
        clock -- which is what hour-of-day statistics need. If omitted, naive
        input is used as-is and aware input triggers a warning.
    sep : str, optional
        Delimiter for text files; inferred from extension if omitted.

    Returns
    -------
    PointSet

    Notes
    -----
    Rows with a missing trajectory id (when an id column is present) are
    dropped with a warning: an unknown mover cannot be assigned to any
    trajectory, and keeping such rows previously made the two matchers return
    different row counts for the same input.
    """
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
    if derive_speed and not has_traj:
        raise ValueError(
            "derive_speed=True needs a unique-id column so speeds are never "
            "differenced across distinct movements. Pass id_col= (or include "
            "a recognisable id column)."
        )
    # speed_col is optional otherwise: position+time-only data is valid input
    # (e.g. for HMMMatcher + roadtraffic.speeds.derive_speeds).

    # Resolve the unit of the source speed column. Skipped entirely when
    # deriving speed from positions: speed_col (if any) is never read in that
    # mode, so warning about a unit that has no effect would just be noise.
    if not derive_speed:
        inferred = _infer_speed_unit(speed_col) if speed_col is not None else None
        if speed_unit is None:
            unit = SpeedUnit.parse(inferred) if inferred else SpeedUnit.MPH
        else:
            unit = SpeedUnit.parse(speed_unit)
            if inferred is not None and SpeedUnit.parse(inferred) is not unit:
                warnings.warn(
                    f"speed_unit={unit.value!r} contradicts the detected column "
                    f"name {speed_col!r} (which implies {inferred!r}); using the "
                    f"explicit speed_unit={unit.value!r}.",
                    stacklevel=2,
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

    out["time"] = _parse_times(df[time_col], timestamp_unit, tz)

    # Carry through source columns the canonical schema does not name -- a
    # per-point horizontal accuracy above all, since
    # :func:`roadtraffic.speeds.derive_speeds` reads its ``pos_accuracy_col``
    # straight off this frame and silently falls back to an assumed constant
    # sigma when the column is absent. Attached here, BEFORE the row drops
    # below, so the extras are filtered along with their own rows: any scheme
    # that re-attaches them afterwards has to reconstruct the alignment, and
    # ``point_id`` cannot serve as the key because it is renumbered after the
    # drop.
    consumed = {lon_col, lat_col, time_col, speed_col, id_col} - {None}
    if has_geom_coords:
        consumed |= {"lon", "lat"}
    extras = [c for c in df.columns if c not in consumed]
    if keep_cols is not None:
        wanted = set(keep_cols)
        missing = wanted - set(df.columns)
        if missing:
            warnings.warn(
                f"keep_cols requested column(s) not in the source: "
                f"{sorted(missing)}.",
                stacklevel=2,
            )
        extras = [c for c in extras if c in wanted]
    for col in extras:
        # A source column named like one this loader writes must not clobber
        # it; preserve it under "<name>_src", as Network._build does for
        # colliding edge attributes.
        name = f"{col}_src" if col in out.columns else col
        out[name] = df[col].values

    if has_traj:
        # a row with no id cannot belong to any trajectory; keeping it would
        # make matcher outputs inconsistent and silently corrupt derived speeds
        missing_id = out["traj_id"].isna()
        if missing_id.any():
            warnings.warn(
                f"Dropped {int(missing_id.sum())} row(s) with a missing id in "
                f"column {id_col!r}: ids are required to separate movers.",
                stacklevel=2,
            )
            out = out[~missing_id].reset_index(drop=True)

    if derive_speed:
        out["speed_mps"] = _derive_speed_mps(out)
    elif speed_col is not None:
        # rows may have been dropped above; re-read speeds aligned via index
        raw = pd.to_numeric(df[speed_col], errors="coerce")
        raw_speed = raw.iloc[out["point_id"].values].to_numpy() \
            if len(out) != len(df) else raw.values
        out["speed_mps"] = to_mps(raw_speed, unit)
    # else: position+time only -- no speed_mps column at all.

    required = ["lon", "lat", "time"] + (["speed_mps"] if "speed_mps" in out.columns else [])
    n_before = len(out)
    out = out.dropna(subset=required).reset_index(drop=True)
    out["point_id"] = np.arange(len(out), dtype=np.int64)
    dropped = n_before - len(out)
    if dropped:
        warnings.warn(
            f"Dropped {dropped} row(s) with missing/unparseable "
            + ("lon/lat/time/speed." if "speed_mps" in out.columns else "lon/lat/time."),
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
    with open(path, encoding="utf-8") as fh:
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
        dist = geodesic_distance(
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
