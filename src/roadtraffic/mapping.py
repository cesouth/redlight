"""Turn a speed-annotated road network into a trafficability map.

After running the pipeline up through :func:`roadtraffic.aggregate.assign_speeds`,
every edge of ``network.graph`` carries ``obs_speed_mps`` (and ``travel_time_s``).
This module exposes those per-edge speeds as map output:

* :func:`to_geojson` — write a GeoJSON ``FeatureCollection`` of the road segments,
  each with its average speed and retained OSM tags, ready to style by speed in
  QGIS / Kepler / Leaflet / Mapbox.
* :func:`plot_speed_map` — render a quick static PNG coloured by speed (no map
  tiles or network needed), for an immediate look.

The network's stored edge geometry is in the projected (metric) CRS, so both
functions reproject back to lon/lat with the network's inverse transformer.
"""
from __future__ import annotations

import json
from typing import Optional

import numpy as np

_MPS_TO_MPH = 1.0 / 0.44704
_MPS_TO_KPH = 3.6


def _speed_in_unit(mps: Optional[float], unit: str) -> Optional[float]:
    if mps is None:
        return None
    if unit == "mph":
        return mps * _MPS_TO_MPH
    if unit == "kph":
        return mps * _MPS_TO_KPH
    return mps


def _edge_lonlat(network, eid: int):
    """Reproject one edge's projected geometry back to (lon, lat) coordinate pairs."""
    geom = network._edge_geoms_proj[eid]
    xs, ys = np.asarray(geom.coords)[:, 0], np.asarray(geom.coords)[:, 1]
    lon, lat = network._transformer_inv.transform(xs, ys)
    return list(zip([float(a) for a in lon], [float(b) for b in lat]))


def _jsonable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)


def _collect_edges(network, directional: bool):
    """Yield (edge_ids, representative_data, obs_speed_mps_or_None) per feature.

    When ``directional`` is False, the two directed edges of a two-way road are
    merged into one feature (speed = mean of whichever directions were observed).
    """
    if directional:
        for u, v, data in network.graph.edges(data=True):
            yield [int(data["edge_id"])], data, data.get("obs_speed_mps")
        return

    seen: dict = {}
    for u, v, data in network.graph.edges(data=True):
        key = frozenset((u, v))
        seen.setdefault(key, []).append(data)
    for key, datas in seen.items():
        speeds = [d["obs_speed_mps"] for d in datas if d.get("obs_speed_mps") is not None]
        spd = float(np.mean(speeds)) if speeds else None
        yield [int(d["edge_id"]) for d in datas], datas[0], spd


def to_geojson(
    network,
    path: Optional[str] = None,
    *,
    directional: bool = False,
    speed_unit: str = "mph",
    keep_tags: Optional[list] = ("name", "highway", "maxspeed", "oneway", "ref"),
) -> dict:
    """Export the speed-annotated network as a GeoJSON FeatureCollection.

    Parameters
    ----------
    path : str, optional
        If given, write the GeoJSON to this file.
    directional : bool
        ``True`` keeps one feature per directed edge (two overlapping lines per
        two-way road). ``False`` (default) emits one feature per physical road.
    speed_unit : {"mph", "kph", "mps"}
        Unit for the ``speed`` property.
    keep_tags : list of str
        Original OSM tags to copy into each feature's properties when present.

    Returns
    -------
    dict
        The GeoJSON FeatureCollection (also written to ``path`` if provided).
    """
    feats = []
    n_obs = n_total = 0
    for eids, data, spd_mps in _collect_edges(network, directional):
        n_total += 1
        if spd_mps is not None and spd_mps > 0:
            n_obs += 1
        coords = _edge_lonlat(network, eids[0])
        props = {
            "edge_id": eids[0],
            "edge_ids": eids,
            "has_speed": bool(spd_mps),
            "speed": _speed_in_unit(spd_mps, speed_unit),
            "speed_unit": speed_unit,
            "length_m": _jsonable(data.get("length_m")),
            "travel_time_s": _jsonable(data.get("travel_time_s")),
        }
        if keep_tags:
            for tag in keep_tags:
                if tag in data:
                    props[tag] = _jsonable(data[tag])
        feats.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords},
            "properties": props,
        })

    fc = {
        "type": "FeatureCollection",
        "features": feats,
        "properties": {"n_edges_observed": n_obs, "n_edges_total": n_total,
                       "speed_unit": speed_unit},
    }
    if path:
        with open(path, "w") as fh:
            json.dump(fc, fh)
    return fc


def plot_speed_map(
    network,
    path: Optional[str] = None,
    *,
    speed_unit: str = "mph",
    cmap: str = "RdYlGn",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    no_data_color: str = "#cccccc",
    linewidth: float = 2.0,
    figsize=(10, 10),
    dpi: int = 150,
):
    """Render a static trafficability map coloured by per-edge speed.

    Observed edges are coloured on ``cmap`` (green = fast/free-flowing, red =
    slow by default); unobserved edges are drawn in ``no_data_color``. Requires
    matplotlib. Returns the matplotlib Figure (and saves to ``path`` if given).
    """
    import matplotlib
    if path:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    lines_obs, speeds, lines_nodata = [], [], []
    for eids, data, spd_mps in _collect_edges(network, directional=False):
        coords = _edge_lonlat(network, eids[0])
        val = _speed_in_unit(spd_mps, speed_unit) if spd_mps and spd_mps > 0 else None
        if val is None:
            lines_nodata.append(coords)
        else:
            lines_obs.append(coords)
            speeds.append(val)

    speeds = np.array(speeds, dtype=float)
    if vmin is None:
        vmin = float(np.nanpercentile(speeds, 5)) if speeds.size else 0.0
    if vmax is None:
        vmax = float(np.nanpercentile(speeds, 95)) if speeds.size else 1.0
    norm = Normalize(vmin=vmin, vmax=max(vmax, vmin + 1e-6))

    fig, ax = plt.subplots(figsize=figsize)
    if lines_nodata:
        ax.add_collection(LineCollection(lines_nodata, colors=no_data_color,
                                         linewidths=linewidth * 0.6, zorder=1))
    if lines_obs:
        lc = LineCollection(lines_obs, cmap=cmap, norm=norm,
                            linewidths=linewidth, zorder=2)
        lc.set_array(speeds)
        ax.add_collection(lc)
        sm = ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
        cb.set_label(f"average speed ({speed_unit})")

    ax.autoscale()
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.set_title("Road network trafficability — average speed per segment")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return fig
