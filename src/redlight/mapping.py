"""Turn a speed-annotated road network into a trafficability map.

After running the pipeline up through :func:`redlight.aggregate.assign_speeds`
or :func:`redlight.aggregate.assign_segment_speeds`, edges of
``network.graph`` carry per-edge observed speeds. This module exposes those
speeds as map output:

* :func:`to_geojson` — write a GeoJSON ``FeatureCollection`` of the road segments,
  each with its average speed and retained OSM tags, ready to style by speed in
  QGIS / Kepler / Leaflet / Mapbox.
* :func:`plot_speed_map` — render a quick static PNG coloured by speed (no map
  tiles or network needed), for an immediate look.

Both accept ``period={"overall", "peak", "offpeak"}`` to export one of the three
regimes written by ``assign_segment_speeds`` (default: overall, which also works
with plain ``assign_speeds``).

Units go through :mod:`redlight.units`, so every alias accepted elsewhere in
the API (``"km/h"``, ``"m/s"``, ...) works here too and unknown units raise.
"""
from __future__ import annotations

import json

import numpy as np

from .units import SpeedUnit, from_mps

_PERIODS = ("overall", "peak", "offpeak")

# Computed properties to_geojson always writes. A keep_tags entry sharing one
# of these names is renamed to "<tag>_src" instead of silently overwriting it
# (mirrors network.py's _sanitize_props convention for the same collision).
_COMPUTED_PROPS = frozenset({
    "edge_id", "edge_ids", "has_speed", "speed", "speed_unit", "period",
    "length_m", "travel_time_s",
})


def _speed_attr(period: str) -> str:
    if period not in _PERIODS:
        raise ValueError(f"period must be one of {_PERIODS}, got {period!r}.")
    # plain obs_speed_mps is written by both assign_speeds and
    # assign_segment_speeds (as the overall regime)
    return "obs_speed_mps" if period == "overall" else f"obs_speed_mps_{period}"


def _jsonable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)


def _collect_edges(network, directional: bool, attr: str):
    """Yield (edge_ids, representative_data, speed_mps_or_None) per feature.

    When ``directional`` is False, the two directed edges of a two-way road are
    merged into one feature (speed = mean of whichever directions carry the
    requested regime). Roads are paired via
    :meth:`~redlight.network.Network.road_edge_ids`, so distinct parallel
    roads between the same nodes are NOT conflated.
    """
    if directional:
        for _u, _v, data in network.graph.edges(data=True):
            yield [int(data["edge_id"])], data, data.get(attr)
        return

    groups: dict = {}
    for _u, _v, data in network.graph.edges(data=True):
        eid = int(data["edge_id"])
        rid = min(network.road_edge_ids(eid))  # canonical id of the physical road
        groups.setdefault(rid, []).append(data)
    for rid in sorted(groups):
        datas = groups[rid]
        speeds = [d[attr] for d in datas if d.get(attr) is not None]
        spd = float(np.mean(speeds)) if speeds else None
        yield sorted(int(d["edge_id"]) for d in datas), datas[0], spd


def to_geojson(
    network,
    path: str | None = None,
    *,
    directional: bool = False,
    period: str = "overall",
    speed_unit: str = "mph",
    keep_tags: list | None = ("name", "highway", "maxspeed", "oneway", "ref"),
) -> dict:
    """Export the speed-annotated network as a GeoJSON FeatureCollection.

    Parameters
    ----------
    path : str, optional
        If given, write the GeoJSON to this file.
    directional : bool
        ``True`` keeps one feature per directed edge (two overlapping lines per
        two-way road). ``False`` (default) emits one feature per physical road,
        with the mean of the observed directions' speeds and a travel time
        recomputed from that merged speed (so the two properties always agree).
    period : {"overall", "peak", "offpeak"}
        Which regime's speeds to export (see
        :func:`~redlight.aggregate.assign_segment_speeds`).
    speed_unit : str or SpeedUnit
        Unit for the ``speed`` property. Any alias `units.SpeedUnit.parse`
        accepts (``"mph"``, ``"km/h"``, ``"m/s"``, ...).
    keep_tags : list of str
        Original OSM tags to copy into each feature's properties when present.
        A tag whose name collides with a computed property (e.g. a source tag
        literally named ``speed``) is copied under ``"<tag>_src"`` instead of
        overwriting the computed value.

    Returns
    -------
    dict
        The GeoJSON FeatureCollection (also written to ``path`` if provided).
    """
    unit = SpeedUnit.parse(speed_unit)
    attr = _speed_attr(period)
    feats = []
    n_obs = n_total = 0
    for eids, data, spd_mps in _collect_edges(network, directional, attr):
        n_total += 1
        # 0.0 m/s is a real observation (gridlock), not "no data" -- only
        # None (never observed in this regime) means that.
        has_speed = spd_mps is not None
        if has_speed:
            n_obs += 1
        coords = network.edge_coords_lonlat(eids[0])
        length_m = data.get("length_m")
        props = {
            "edge_id": eids[0],
            "edge_ids": eids,
            "has_speed": has_speed,
            "speed": from_mps(float(spd_mps), unit) if has_speed else None,
            "speed_unit": unit.value,
            "period": period,
            "length_m": _jsonable(length_m),
            # recomputed from the (possibly direction-merged) speed so speed
            # and travel time never contradict each other. spd_mps == 0.0
            # (gridlock) has an undefined/infinite travel time -- left None
            # rather than raising a ZeroDivisionError.
            "travel_time_s": (float(length_m) / float(spd_mps)
                              if has_speed and length_m and spd_mps > 0 else None),
        }
        if keep_tags:
            for tag in keep_tags:
                if tag in data:
                    key = f"{tag}_src" if tag in _COMPUTED_PROPS else tag
                    props[key] = _jsonable(data[tag])
        feats.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords},
            "properties": props,
        })

    fc = {
        "type": "FeatureCollection",
        "features": feats,
        "properties": {"n_edges_observed": n_obs, "n_edges_total": n_total,
                       "speed_unit": unit.value, "period": period},
    }
    if path:
        with open(path, "w") as fh:
            json.dump(fc, fh)
    return fc


def plot_speed_map(
    network,
    path: str | None = None,
    *,
    period: str = "overall",
    speed_unit: str = "mph",
    cmap: str = "RdYlGn",
    vmin: float | None = None,
    vmax: float | None = None,
    no_data_color: str = "#cccccc",
    linewidth: float = 2.0,
    figsize=(10, 10),
    dpi: int = 150,
):
    """Render a static trafficability map coloured by per-edge speed.

    Observed edges are coloured on ``cmap`` (green = fast/free-flowing, red =
    slow by default); unobserved edges are drawn in ``no_data_color``. Requires
    matplotlib (``pip install redlight[mapping]``), imported lazily so the
    core package has no hard dependency on it.

    Parameters
    ----------
    network : redlight.network.Network
        A network that has been annotated with per-edge speeds via
        :func:`~redlight.aggregate.assign_speeds` or
        :func:`~redlight.aggregate.assign_segment_speeds`.
    path : str, optional
        If given, save the figure here (format inferred from the extension,
        e.g. ``.png``); the matplotlib backend is switched to the
        non-interactive ``"Agg"`` first so this works headless (no display).
        If omitted, the figure is only returned -- useful in a notebook.
    period : {"overall", "peak", "offpeak"}
        Which regime's speeds to colour by (see
        :func:`~redlight.aggregate.assign_segment_speeds`).
    speed_unit : str or SpeedUnit
        Unit for the colour scale and colourbar label. Any alias
        :meth:`~redlight.units.SpeedUnit.parse` accepts.
    cmap : str
        Matplotlib colormap name for the speed scale. Default ``"RdYlGn"``
        (red = slow, yellow = medium, green = fast) -- pick a colormap where
        low values read as "bad" if you swap it.
    vmin, vmax : float, optional
        Colour-scale bounds, in ``speed_unit``. Default: the 5th/95th
        percentile of observed speeds (robust to a few extreme outliers
        stretching the scale). Pass both explicitly to compare multiple maps
        (e.g. peak vs. off-peak) on the same colour scale.
    no_data_color : str
        Matplotlib color spec for edges with no observation in ``period``.
        Default a light grey (``"#cccccc"``) so they read as background.
    linewidth : float
        Line width (points) for observed edges; unobserved edges are drawn
        thinner (``0.6 * linewidth``) so they recede visually.
    figsize : tuple of float
        Figure size in inches, passed straight to ``matplotlib.pyplot.subplots``.
    dpi : int
        Resolution for the saved file (ignored if ``path`` is omitted).

    Returns
    -------
    matplotlib.figure.Figure
        The rendered figure. Callers embedding it elsewhere (e.g. a notebook
        or GUI) are responsible for closing it (``plt.close(fig)``) if they
        don't want it to accumulate in matplotlib's global figure registry.
    """
    try:
        import matplotlib
    except ImportError as exc:  # pragma: no cover - exercised in a subprocess
        raise ImportError(
            "plot_speed_map needs matplotlib, which redlight does not install "
            "by default. Install the extra:\n"
            "    pip install 'redlight[mapping]'\n"
            "or use to_geojson() and render the result in QGIS or a web map."
        ) from exc
    if path:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize

    unit = SpeedUnit.parse(speed_unit)
    attr = _speed_attr(period)

    lines_obs, speeds, lines_nodata = [], [], []
    for eids, _data, spd_mps in _collect_edges(network, False, attr):
        coords = network.edge_coords_lonlat(eids[0])
        # 0.0 m/s is a real observation (gridlock) and must be drawn on the
        # speed colormap, not lumped in with genuinely unobserved edges.
        if spd_mps is not None:
            lines_obs.append(coords)
            speeds.append(from_mps(float(spd_mps), unit))
        else:
            lines_nodata.append(coords)

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
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
        cb.set_label(f"average speed ({unit.value})")

    ax.autoscale()
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    title_period = "" if period == "overall" else f" — {period} hours"
    ax.set_title(f"Road network trafficability — average speed per segment{title_period}")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return fig
