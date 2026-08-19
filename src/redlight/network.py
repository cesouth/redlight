"""Road network container.

Loads a road network from GeoJSON (native, no extra deps), Shapefile/GPKG
(optional, requires the ``shapefile`` extra -> pyogrio, Python 3.10+), or the
Overpass API (:meth:`Network.from_overpass`, stdlib only). Builds a NetworkX
multigraph whose nodes are endpoint coordinates and whose edges carry
geometry, length, and any source attributes (e.g. OSM ``highway``,
``maxspeed``, ``oneway``).

The graph is a :class:`networkx.MultiDiGraph` keyed by ``edge_id``, so two
distinct roads that happen to share both endpoints (dual carriageways,
service loops) coexist instead of overwriting each other. Two-way roads get
one directed edge per direction; the pair is linked via
:meth:`Network.road_edge_ids`. OSM ``oneway`` semantics are honoured,
including ``oneway=-1`` (one-way *against* the digitized direction).

Coordinates are stored in WGS84 (EPSG:4326) and also projected to a local
metric CRS (auto UTM, or user-specified) for distance-correct snapping and
length computation. Snapping/length math must never be done in degrees. UTM
and Web Mercator are projected in numpy (see ``_proj``); any other CRS needs
the optional ``crs`` extra, which pulls in pyproj.
"""
from __future__ import annotations

import json
import math
import re
import warnings

import numpy as np
import shapely
from shapely.geometry import LineString, shape
from shapely.ops import transform as shapely_transform

from . import _proj

try:  # networkx is a core dependency
    import networkx as nx
except ImportError as exc:  # pragma: no cover
    raise ImportError("redlight requires networkx. pip install networkx") from exc


def _auto_utm_epsg(lon: float, lat: float) -> int:
    """Return the EPSG code of the UTM zone containing (lon, lat)."""
    zone = int(math.floor((lon + 180.0) / 6.0) % 60) + 1
    return (32600 if lat >= 0 else 32700) + zone


# What each caller of _require_pyproj can natively handle. These differ: a
# source file may be read from any of the three, but only UTM is safe as the
# *metric* CRS, because Web Mercator inflates ground distance by sec(latitude)
# -- some 55% at 50 deg N -- which would corrupt every length and speed.
_SOURCE_CRS_NATIVE = (
    "EPSG:4326 (WGS84), OGC:CRS84, EPSG:3857 (Web Mercator), or a WGS84 UTM "
    "zone (EPSG:32601-32660 and 32701-32760)"
)
_METRIC_CRS_NATIVE = (
    "a WGS84 UTM zone (EPSG:32601-32660 for the northern hemisphere, "
    "32701-32760 for the southern), which is also the default"
)


def _require_pyproj(what: str, supported: str):
    """Import pyproj, or raise an error that names the extra and the way out.

    pyproj is not a core dependency: it ships PROJ's native library and its
    coordinate database, which is both the bulk of the install and the usual
    source of ``proj.db`` conflicts. The natively supported systems cover
    essentially all road data, so this is reached only for the long tail.

    Parameters
    ----------
    what : str
        The operation that needs pyproj, e.g. ``"metric_epsg=EPSG:3035"``.
    supported : str
        The alternatives *this* caller can handle without pyproj -- pass
        ``_SOURCE_CRS_NATIVE`` or ``_METRIC_CRS_NATIVE``. The two sets are not
        the same, and suggesting the wrong one sends the user back to the CRS
        that just failed.
    """
    try:
        import pyproj
    except ImportError as exc:
        raise ImportError(
            f"{what} needs pyproj, which redlight does not install by "
            f"default. Either install the extra:\n"
            f"    pip install 'redlight[crs]'\n"
            f"or use a natively supported CRS: {supported}."
        ) from exc
    return pyproj


def _crs_excerpt(crs, width: int = 60) -> str:
    """Shorten a CRS for an error message, marking that it was cut.

    A WKT string is hundreds of characters, and a bare slice of one ends
    mid-token (``SPHEROID["WGS 84",6``) and reads like corruption rather than
    an excerpt. ``textwrap.shorten`` is not used here: WKT's only spaces are
    inside quoted names, so a word-boundary cut can throw away almost the
    whole excerpt. The trailing marker is what makes the cut legible.
    """
    text = " ".join(str(crs).split())
    if len(text) <= width:
        return text
    return text[:width - 4].rstrip() + " ..."


def _metric_crs_and_transformers(epsg: int):
    """Return ``(crs, forward, inverse)`` for the metric CRS.

    UTM -- the default and the overwhelmingly common case -- is handled in
    numpy. Any other projected CRS the caller asks for via ``metric_epsg``
    falls back to pyproj.
    """
    try:
        return _proj.utm_crs_and_transformers(epsg)
    except ValueError:
        pass
    pyproj = _require_pyproj(f"metric_epsg=EPSG:{epsg}", _METRIC_CRS_NATIVE)
    crs = pyproj.CRS.from_epsg(epsg)
    wgs84 = pyproj.CRS.from_epsg(_proj.EPSG_WGS84)
    return (crs,
            pyproj.Transformer.from_crs(wgs84, crs, always_xy=True),
            pyproj.Transformer.from_crs(crs, wgs84, always_xy=True))


def _source_to_wgs84(crs):
    """Return an ``(x, y) -> (lon, lat)`` callable for a file's source CRS.

    Returns None when the source is already WGS84 and no transform is needed --
    including the CRS84 and 4979 spellings of it, which carry no usable EPSG
    code but describe the very same lon/lat. UTM and Web Mercator are handled
    in numpy; anything else -- national grids, non-WGS84 datums, raw WKT with
    no authority code -- needs PROJ's database and falls back to pyproj.
    """
    # No CRS recorded at all means WGS84 by convention; a CRS that is recorded
    # but unparseable (raw WKT) is a real one we cannot read.
    if not crs or _proj.is_wgs84(crs):
        return None
    epsg = _proj.parse_epsg(crs)
    if epsg is None:
        pyproj = _require_pyproj(f"Reading a file in {_crs_excerpt(crs)}",
                                 _SOURCE_CRS_NATIVE)
        return pyproj.Transformer.from_crs(
            pyproj.CRS.from_user_input(crs),
            pyproj.CRS.from_epsg(_proj.EPSG_WGS84),
            always_xy=True,
        ).transform
    if epsg == _proj.EPSG_WEB_MERCATOR:
        return _proj.web_mercator_inverse
    try:
        zone, north = _proj.utm_epsg_to_zone(epsg)
    except ValueError:
        pyproj = _require_pyproj(f"Reading a file in EPSG:{epsg}",
                                 _SOURCE_CRS_NATIVE)
        return pyproj.Transformer.from_crs(
            pyproj.CRS.from_epsg(epsg),
            pyproj.CRS.from_epsg(_proj.EPSG_WGS84),
            always_xy=True,
        ).transform
    return lambda x, y: _proj.utm_inverse(x, y, zone, north)


def _geojson_crs(gj) -> str | None:
    """The CRS a GeoJSON document declares, normalised to an ``EPSG:``/``OGC:`` token.

    RFC 7946 fixed GeoJSON at WGS84 lon/lat and dropped the ``crs`` member, but
    QGIS and ogr2ogr still emit the GeoJSON-2008 form and people still hand
    those files to this package. Ignoring the member means reading eastings and
    northings as degrees, which does not fail -- it silently produces a road
    thousands of kilometres long.

    Handles both spellings of the name: the OGC URN
    (``urn:ogc:def:crs:EPSG::27700``, ``urn:ogc:def:crs:OGC:1.3:CRS84``) and the
    short form (``EPSG:27700``). Returns None when no member is present, which
    :func:`_source_to_wgs84` already treats as "WGS84 by convention".
    """
    if not isinstance(gj, dict):
        return None
    name = ((gj.get("crs") or {}).get("properties") or {}).get("name")
    if not name:
        return None
    text = str(name).strip()
    m = re.match(r"^urn:ogc:def:crs:([A-Za-z]+):[^:]*:(.+)$", text, re.IGNORECASE)
    if m:
        authority, code = m.group(1).upper(), m.group(2).strip()
        return f"{authority}:{code.upper() if authority == 'OGC' else code}"
    return text


def _round_node(x: float, y: float, ndigits: int = 7):
    """Quantise a coordinate so shared endpoints hash to the same node.

    7 decimal degrees ~ 1.1 cm at the equator: enough to merge endpoints that
    are meant to be identical without collapsing genuinely distinct nodes.
    Note that endpoints serialized with different float noise can still
    straddle a rounding boundary; if two features are meant to connect, make
    sure they share exact endpoint coordinates in the source data.
    """
    return (round(x, ndigits), round(y, ndigits))


# Edge attributes the builder itself writes; colliding source properties are
# preserved under "<name>_src" instead of crashing add_edge.
_RESERVED_EDGE_ATTRS = frozenset({"edge_id", "length_m", "geometry"})

# OSM oneway values. Forward = travel only in the digitized direction;
# reverse (oneway=-1) = travel only AGAINST the digitized direction.
_ONEWAY_FORWARD = frozenset({"yes", "true", "1", "t"})
_ONEWAY_REVERSE = frozenset({"-1", "reverse"})


def _sanitize_props(props: dict):
    """Rename source properties that collide with reserved edge attributes."""
    out, renamed = {}, []
    for k, v in props.items():
        if k in _RESERVED_EDGE_ATTRS:
            out[f"{k}_src"] = v
            renamed.append(k)
        else:
            out[k] = v
    return out, renamed


class Network:
    """A road network as a graph plus projected geometry.

    Attributes
    ----------
    graph : networkx.MultiDiGraph
        Directed multigraph, edge key == ``edge_id``. Each edge has attrs:
        ``geometry`` (projected LineString, metres), ``length_m`` (float),
        ``edge_id`` (int), plus source attributes. Two-way roads get edges in
        both directions (linked via :meth:`road_edge_ids`).
    crs_metric : _proj.UtmCrs or pyproj.CRS
        Projected CRS used for all distance math: ``_proj.UtmCrs`` on the
        default UTM path, a real ``pyproj.CRS`` only when a non-UTM
        ``metric_epsg`` was supplied.
    """

    def __init__(self, graph, crs_metric, edge_index, edge_geoms_proj,
                 edge_ids, transformer_fwd, transformer_inv,
                 edge_lengths=None, reverse_of=None):
        """Assemble a ``Network`` from already-built graph/index pieces.

        Not normally called directly -- use :meth:`from_geojson`,
        :meth:`from_file`, or :meth:`from_overpass` to build one from source
        data; those all funnel through the internal :meth:`_build` classmethod,
        which constructs the arguments below and calls this constructor.

        Parameters
        ----------
        graph : networkx.MultiDiGraph
            The road graph itself, nodes = (lon, lat) tuples, edges keyed by
            ``edge_id``.
        crs_metric : _proj.UtmCrs or pyproj.CRS
            The projected CRS all distance/length/snapping math is done in:
            ``_proj.UtmCrs`` on the default UTM path, a real ``pyproj.CRS``
            only when a non-UTM ``metric_epsg`` was supplied.
        edge_index : dict
            ``edge_id -> (u, v)`` node tuple, mirroring the graph's own edge
            endpoints for O(1) lookup without a graph traversal.
        edge_geoms_proj : dict
            ``edge_id -> shapely.LineString`` in the projected metric CRS
            (the same geometry stored on the graph edge, kept here too for
            fast access from hot paths like :meth:`_build_segment_table`).
        edge_ids : numpy.ndarray
            Sorted array of every edge id, backing the :attr:`edge_ids`
            property.
        transformer_fwd, transformer_inv : _proj.UtmTransformer or pyproj.Transformer
            WGS84 -> metric and metric -> WGS84 coordinate transformers
            (``_proj.UtmTransformer`` on the default UTM path, real
            ``pyproj.Transformer`` only for a non-UTM ``metric_epsg``),
            reused so every projection in the object uses the exact same CRS
            pair (see :meth:`project_points`, :meth:`edge_coords_lonlat`).
        edge_lengths : dict, optional
            ``edge_id -> length_m``, backing :meth:`edge_length`. Defaults to
            an empty dict.
        reverse_of : dict, optional
            ``edge_id -> opposite-direction edge_id`` (or ``None`` for a
            one-way road with no opposite edge), backing
            :meth:`road_edge_ids`. Defaults to an empty dict.
        """
        self.graph = graph
        self.crs_metric = crs_metric
        self._edge_index = edge_index          # edge_id -> (u, v)
        self._edge_geoms_proj = edge_geoms_proj  # edge_id -> projected LineString
        self._edge_ids = edge_ids              # ordered array of edge_ids
        self._transformer_fwd = transformer_fwd  # wgs84 -> metric
        self._transformer_inv = transformer_inv  # metric -> wgs84
        self._edge_lengths = edge_lengths or {}  # edge_id -> length (m)
        self._reverse_of = reverse_of or {}    # edge_id -> opposite-direction id or None
        self._kdtree = None
        self._seg_table = None  # (edge_id, x0, y0, x1, y1) per segment for snap
        self._csr = None        # scipy CSR adjacency (lazy, see csgraph())
        self._node_int = None   # node tuple -> integer index into _csr
        self._int_node = None   # integer index -> node tuple

    # ------------------------------------------------------------------ loaders
    @classmethod
    def from_geojson(cls, path: str, *, metric_epsg: int | None = None,
                     directed: bool = True, oneway_attr: str = "oneway",
                     length_attr: str | None = None) -> Network:
        """Load a network from a GeoJSON LineString FeatureCollection.

        Parameters
        ----------
        path : str
            Path to a ``.geojson``/``.json`` file. ``LineString`` and
            ``MultiLineString`` features are read; each part of a
            ``MultiLineString`` becomes its own edge (or edge pair).
        metric_epsg : int, optional
            Projected CRS (EPSG code) used for all distance-correct math
            (snapping, length). Default: auto-picked UTM zone from the first
            coordinate of the first geometry.
        directed : bool
            If ``True`` (default), the ``oneway_attr`` source property is
            honoured -- a oneway road gets a single directed edge in the legal
            direction of travel. If ``False``, oneway restrictions are ignored
            and every road gets edges in both directions. Note this does *not*
            change the graph's type: the result is always a
            :class:`networkx.MultiDiGraph` either way.
        oneway_attr : str
            Source property name holding the oneway tag. Default ``"oneway"``
            (the OSM convention); values ``"yes"``/``"true"``/``"1"``/``"t"``
            mean forward-only, ``"-1"``/``"reverse"`` mean reverse-only. Any
            other value (including absent) is treated as two-way.
        length_attr : str, optional
            Source property to use as edge length in metres instead of
            computing it from the projected geometry (e.g. a pre-measured
            OSM ``length`` tag). Default: compute from geometry.

        Returns
        -------
        Network
        """
        with open(path, encoding="utf-8") as fh:
            gj = json.load(fh)
        feats = gj.get("features", []) if isinstance(gj, dict) else []
        # Same treatment the Shapefile/GPKG loader gives its source CRS, so the
        # two produce identical structure for equivalent data: WGS84, CRS84,
        # Web Mercator and UTM are handled in numpy, anything else goes through
        # the `crs` extra with an error that names it.
        to_wgs84 = _source_to_wgs84(_geojson_crs(gj))
        records = []
        for feat in feats:
            geom = (feat or {}).get("geometry") or {}
            gtype = geom.get("type")
            props = dict(feat.get("properties") or {})
            parts = []
            if gtype == "LineString":
                parts = [shape(geom)]
            elif gtype == "MultiLineString":
                parts = list(shape(geom).geoms)
            for part in parts:
                if to_wgs84 is not None:
                    part = shapely_transform(to_wgs84, part)
                records.append((part, dict(props)))
        if not records:
            raise ValueError("GeoJSON contained no LineString features.")
        return cls._build(records, metric_epsg, directed, oneway_attr, length_attr)

    @classmethod
    def from_file(cls, path: str, *, metric_epsg: int | None = None,
                  directed: bool = True, oneway_attr: str = "oneway",
                  length_attr: str | None = None,
                  layer: str | int | None = None) -> Network:
        """Load a network from Shapefile (``.shp``) or GeoPackage (``.gpkg``).

        Requires the optional ``shapefile`` extra (``pip install
        redlight[shapefile]``; needs Python 3.10+), which pulls in
        ``pyogrio`` for reading non-GeoJSON vector formats -- GeoJSON itself
        needs no extra dependency and is dispatched straight to
        :meth:`from_geojson` without importing ``pyogrio`` at all. Only
        ``LineString`` and ``MultiLineString`` features are read (other
        geometry types in the file are skipped); the source CRS is read from
        the file and every coordinate is reprojected to WGS84 before being
        handed to the same builder :meth:`from_geojson` uses, so the two
        loaders produce identical `Network` structure for equivalent data.

        Parameters
        ----------
        path : str
            Path to a ``.shp``, ``.gpkg``, or ``.geojson``/``.json`` file
            (the latter two are dispatched to :meth:`from_geojson`).
        metric_epsg, directed, oneway_attr, length_attr :
            Same meaning as in :meth:`from_geojson`.
        layer : str or int, optional
            Layer name or index to read from a multi-layer file (e.g. one
            table in a GeoPackage that holds several). Default: pyogrio's own
            default layer (the file's only layer for a Shapefile; the first
            layer for a multi-layer GeoPackage unless overridden here).
            Ignored when ``path`` is dispatched to :meth:`from_geojson`.

        Returns
        -------
        Network
        """
        low = path.lower()
        if low.endswith((".geojson", ".json")):
            return cls.from_geojson(
                path, metric_epsg=metric_epsg, directed=directed,
                oneway_attr=oneway_attr, length_attr=length_attr,
            )
        try:
            import pyogrio
        except ImportError as exc:
            raise ImportError(
                "Reading Shapefile/GPKG requires the optional 'shapefile' extra. "
                "Install with: pip install redlight[shapefile]"
            ) from exc
        # force_2d=True: drop any Z coordinate, matching the (x, y)-only
        # reprojection below (2D distance-correct math is all this package does).
        meta, _fids, wkb, field_data = pyogrio.raw.read(
            path, layer=layer, force_2d=True,
        )
        # A feature with no geometry comes back as a None entry in `wkb`;
        # drop it from every parallel array (geometry + each field column)
        # before converting, mirroring fiona's `if geom is None: continue`.
        has_geom = np.array([g is not None for g in wkb], dtype=bool)
        geoms = shapely.from_wkb(wkb[has_geom])
        field_names = list(meta["fields"])
        field_data = tuple(np.asarray(col)[has_geom] for col in field_data)

        to_wgs84 = _source_to_wgs84(meta.get("crs"))

        records = []
        for g, row in zip(geoms, zip(*field_data)):
            props = dict(zip(field_names, row))
            parts = g.geoms if g.geom_type == "MultiLineString" else [g]
            for part in parts:
                if part.geom_type != "LineString":
                    continue
                if to_wgs84 is not None:
                    part = shapely_transform(to_wgs84, part)
                records.append((part, dict(props)))
        if not records:
            raise ValueError("File contained no LineString features.")
        return cls._build(records, metric_epsg, directed, oneway_attr, length_attr)

    @classmethod
    def from_overpass(cls, bbox, *, metric_epsg: int | None = None,
                      directed: bool = True, oneway_attr: str = "oneway",
                      highway_regex: str | None = None,
                      url: str | None = None,
                      timeout: float = 90.0) -> Network:
        """Fetch an OSM road network for a bounding box via the Overpass API.

        Requires network access; no extra dependencies (stdlib urllib).

        Parameters
        ----------
        bbox : tuple of float
            ``(min_lon, min_lat, max_lon, max_lat)`` in WGS84 degrees.
        metric_epsg : int, optional
            Projected CRS (EPSG code) for distance-correct snapping/length
            math. Default: auto-picked UTM zone from the first coordinate.
        directed : bool
            If ``True`` (default), OSM ``oneway`` restrictions are honoured --
            a oneway street gets a single directed edge. If ``False``, oneway
            restrictions are ignored and every road gets edges in both
            directions. Note this does *not* change the graph's type: the
            result is always a :class:`networkx.MultiDiGraph` either way.
        oneway_attr : str
            Source property name holding the oneway tag. Default ``"oneway"``
            (the OSM convention); values ``"yes"``/``"true"``/``"1"``/``"t"``
            mean forward-only, ``"-1"``/``"reverse"`` mean reverse-only.
        highway_regex : str, optional
            Regex of OSM ``highway`` values to include. Defaults to drivable
            road classes (see :data:`redlight.osm.DEFAULT_HIGHWAY_REGEX`).
        url : str, optional
            Overpass endpoint (default: the public overpass-api.de instance).
        timeout : float
            HTTP + Overpass query timeout in seconds.
        """
        from .osm import fetch_network_records
        records = fetch_network_records(
            bbox, highway_regex=highway_regex, url=url, timeout=timeout,
        )
        if not records:
            raise ValueError(
                "Overpass returned no ways for that bounding box / highway filter."
            )
        return cls._build(records, metric_epsg, directed, oneway_attr, None)

    # ------------------------------------------------------------------ builder
    @classmethod
    def _build(cls, records, metric_epsg, directed, oneway_attr, length_attr):
        """Shared graph-construction logic behind every ``from_*`` loader.

        Parameters
        ----------
        records : list of (shapely.LineString, dict)
            One entry per road, geometry in WGS84 lon/lat and its source
            property dict. This is the common currency every loader (GeoJSON,
            Shapefile/GPKG, Overpass) converts its own format into before
            calling here -- the only place that needs to know about graph
            construction, CRS projection, and oneway semantics.
        metric_epsg, directed, oneway_attr, length_attr :
            Passed straight through from whichever ``from_*`` classmethod was
            called; see :meth:`from_geojson` for what each one means.

        Returns
        -------
        Network
        """
        # Pick a metric CRS from the first coordinate of the first geometry.
        first = records[0][0]
        c = first.coords[0]
        if metric_epsg is None:
            metric_epsg = _auto_utm_epsg(c[0], c[1])
        crs_metric, fwd, inv = _metric_crs_and_transformers(metric_epsg)

        # Local import (once, not per record): keeps urllib -- pulled in by
        # osm.py's Overpass client -- off the cost of importing redlight,
        # matching the deferred import in from_overpass below.
        from .osm import parse_maxspeed

        graph = nx.MultiDiGraph()
        edge_index, edge_geoms_proj = {}, {}   # edge_id -> (u, v) / projected geometry
        edge_lengths, reverse_of = {}, {}      # edge_id -> length_m / opposite edge_id
        next_id = 0       # monotonically increasing edge_id counter
        n_loops = 0        # count of skipped closed-loop features, for the warning below
        renamed_keys = set()  # source property names that collided and got "_src"-suffixed

        def _add(a, b, line, length_m, attrs):
            """Add one directed edge a->b, assign it the next edge_id, and
            record it in every per-edge index the Network needs. Returns the
            new edge_id so the caller can link forward/reverse pairs."""
            nonlocal next_id
            eid = next_id
            next_id += 1
            graph.add_node(a, lon=a[0], lat=a[1])
            graph.add_node(b, lon=b[0], lat=b[1])
            graph.add_edge(a, b, key=eid, edge_id=eid, length_m=length_m,
                           geometry=line, **attrs)
            edge_index[eid] = (a, b)
            edge_geoms_proj[eid] = line
            edge_lengths[eid] = float(length_m)
            reverse_of[eid] = None
            return eid

        for geom_wgs, props in records:
            # geom_wgs.coords: the road's vertices in WGS84 lon/lat order, as
            # digitized; xs/ys split them into parallel coordinate arrays for
            # the vectorised WGS84 -> metric projection below.
            xs, ys = zip(*[(pt[0], pt[1]) for pt in geom_wgs.coords])
            px, py = fwd.transform(np.asarray(xs), np.asarray(ys))
            proj_line = LineString(np.column_stack([px, py]))  # same geometry, metric CRS
            length_m = (float(props[length_attr]) if length_attr and length_attr in props
                        else proj_line.length)
            u = _round_node(xs[0], ys[0])   # start-node key (rounded WGS84 coord)
            v = _round_node(xs[-1], ys[-1])  # end-node key
            if u == v:
                n_loops += 1  # closed ways (roundabouts/loops) cannot be modelled
                continue

            # oneway_val: the source oneway tag, normalised for comparison
            # against _ONEWAY_FORWARD / _ONEWAY_REVERSE below.
            oneway_val = str(props.get(oneway_attr, "")).strip().lower()
            # base_attrs: props with any reserved-name collision renamed to
            # "<name>_src" (see _sanitize_props); this becomes the edge's
            # attribute dict on the graph.
            base_attrs, renamed = _sanitize_props(props)
            renamed_keys.update(renamed)
            # Posted legal limit, parsed to m/s for routing fallbacks. The raw
            # tag is left untouched alongside it; an unparseable value (OSM's
            # "none"/"walk"/country defaults) simply writes no numeric attr, so
            # consumers can test presence rather than sniff for sentinels.
            if "maxspeed" in base_attrs and "maxspeed_mps" not in base_attrs:
                maxspeed_mps = parse_maxspeed(base_attrs["maxspeed"])
                if maxspeed_mps is not None:
                    base_attrs["maxspeed_mps"] = maxspeed_mps
            rev_line = LineString(list(proj_line.coords)[::-1])  # v -> u geometry

            if directed and oneway_val in _ONEWAY_FORWARD:
                _add(u, v, proj_line, length_m, base_attrs)
            elif directed and oneway_val in _ONEWAY_REVERSE:
                # OSM oneway=-1: travel is legal only AGAINST the digitized
                # direction, so the only edge runs v -> u.
                _add(v, u, rev_line, length_m, base_attrs)
            else:
                e_fwd = _add(u, v, proj_line, length_m, base_attrs)
                e_rev = _add(v, u, rev_line, length_m, base_attrs)
                reverse_of[e_fwd] = e_rev
                reverse_of[e_rev] = e_fwd

        if n_loops:
            warnings.warn(
                f"Skipped {n_loops} closed-loop feature(s) whose endpoints "
                "coincide (e.g. roundabouts digitized as one closed way). "
                "Split closed ways into open segments to include them.",
                stacklevel=3,
            )
        if renamed_keys:
            warnings.warn(
                "Source properties colliding with reserved edge attributes were "
                f"renamed with an '_src' suffix: {sorted(renamed_keys)}.",
                stacklevel=3,
            )

        edge_ids = np.array(sorted(edge_index.keys()), dtype=np.int64)
        net = cls(graph, crs_metric, edge_index, edge_geoms_proj, edge_ids,
                  fwd, inv, edge_lengths, reverse_of)
        net._build_segment_table()
        return net

    # ------------------------------------------------------ spatial index/snap
    # Long segments are split so a query point near a segment end is never
    # crowded out of the midpoint KDTree shortlist by nearer-midpoint segments.
    # The exact foot-of-perpendicular distance is still computed against the
    # original segment endpoints, so the matched edge is identical -- this only
    # makes candidate retrieval reliable. 25 m is well below typical snap
    # tolerances yet keeps the table small.
    _SNAP_DENSIFY_M = 25.0

    def _build_segment_table(self):
        """Flatten every edge into its constituent 2-point segments for snapping.

        A KDTree over segment midpoints gives fast candidate retrieval; exact
        point-to-segment distance is then computed for the shortlist. Segments
        longer than ``_SNAP_DENSIFY_M`` are subdivided so every midpoint stays
        close to the geometry it represents.
        """
        d = self._SNAP_DENSIFY_M
        rows = []
        for eid in self._edge_ids:
            coords = np.asarray(self._edge_geoms_proj[eid].coords)
            arc0 = 0.0          # arc length from the edge start to this segment
            for i in range(len(coords) - 1):
                x0, y0 = coords[i]
                x1, y1 = coords[i + 1]
                seg_len = math.hypot(x1 - x0, y1 - y0)
                n_sub = max(1, int(math.ceil(seg_len / d))) if d and d > 0 else 1
                if n_sub == 1:
                    rows.append((eid, x0, y0, x1, y1, arc0))
                else:
                    ts = np.linspace(0.0, 1.0, n_sub + 1)
                    xs = x0 + ts * (x1 - x0)
                    ys = y0 + ts * (y1 - y0)
                    for j in range(n_sub):
                        rows.append((eid, xs[j], ys[j], xs[j + 1], ys[j + 1],
                                     arc0 + ts[j] * seg_len))
                arc0 += seg_len
        if not rows:
            raise ValueError(
                "Network has no usable edges: every feature was degenerate "
                "(zero length, or a closed loop whose endpoints coincide). "
                "Split closed ways into open segments before loading."
            )
        self._seg_table = np.array(rows, dtype=float)  # cols: eid,x0,y0,x1,y1,arc0
        mids = np.column_stack([
            (self._seg_table[:, 1] + self._seg_table[:, 3]) / 2.0,
            (self._seg_table[:, 2] + self._seg_table[:, 4]) / 2.0,
        ])
        from scipy.spatial import cKDTree
        self._kdtree = cKDTree(mids)

    def project_points(self, lon, lat):
        """Project WGS84 lon/lat arrays to the metric CRS (metres)."""
        return self._transformer_fwd.transform(np.asarray(lon), np.asarray(lat))

    def candidate_edges(self, px, py, *, k: int = 10, max_dist: float = 50.0):
        """For metric point (px, py), return candidate (edge_id, perp_dist, t).

        ``t`` is the fractional position (0..1) of the foot of the perpendicular
        along the matched segment. ``max_dist`` is the snap tolerance in metres.
        Results are unique per edge_id, keeping the closest segment, and are
        ordered by increasing perpendicular distance.
        """
        k = min(k, len(self._seg_table))
        dists, idxs = self._kdtree.query([px, py], k=k)
        idxs = np.atleast_1d(idxs)
        seg = self._seg_table[idxs]
        eid = seg[:, 0]
        x0, y0, x1, y1 = seg[:, 1], seg[:, 2], seg[:, 3], seg[:, 4]
        dx, dy = x1 - x0, y1 - y0
        seg_len2 = dx * dx + dy * dy
        seg_len2[seg_len2 == 0] = 1e-12
        t = ((px - x0) * dx + (py - y0) * dy) / seg_len2
        t = np.clip(t, 0.0, 1.0)
        fx, fy = x0 + t * dx, y0 + t * dy
        perp = np.hypot(px - fx, py - fy)
        order = np.argsort(perp)
        best = {}
        for j in order:
            e = int(eid[j])
            if perp[j] > max_dist:
                continue
            if e not in best:
                best[e] = (perp[j], float(t[j]))
        return [(e, d, tt) for e, (d, tt) in best.items()]

    # -------------------------------------------------------- batch snapping
    def _candidates_matrix(self, px, py, k: int):
        """Vectorised candidate retrieval for many points at once.

        One batch KDTree query plus one vectorised foot-of-perpendicular
        computation over the whole (n, k) candidate matrix — the same math as
        :meth:`candidate_edges`, minus the per-point Python overhead. Returns
        ``(edge_ids, perp_dists, ts, arcs)``, each of shape ``(n, k)``, where
        ``ts`` are fractions of a densified sub-segment and ``arcs`` are the
        corresponding arc lengths in metres from the edge's start node.
        """
        pts = np.column_stack([np.asarray(px, dtype=float),
                               np.asarray(py, dtype=float)])
        k = min(k, len(self._seg_table))
        _, idxs = self._kdtree.query(pts, k=k)
        idxs = np.asarray(idxs).reshape(len(pts), -1)  # k=1 gives (n,) -> (n,1)
        seg = self._seg_table[idxs]                    # (n, k, 5)
        x0, y0, x1, y1 = seg[..., 1], seg[..., 2], seg[..., 3], seg[..., 4]
        dx, dy = x1 - x0, y1 - y0
        seg_len2 = dx * dx + dy * dy
        seg_len2 = np.where(seg_len2 == 0, 1e-12, seg_len2)
        qx, qy = pts[:, :1], pts[:, 1:2]
        t = np.clip(((qx - x0) * dx + (qy - y0) * dy) / seg_len2, 0.0, 1.0)
        fx, fy = x0 + t * dx, y0 + t * dy
        perp = np.hypot(qx - fx, qy - fy)
        arc = seg[..., 5] + t * np.hypot(dx, dy)
        return seg[..., 0].astype(np.int64), perp, t, arc

    def nearest_edges(self, px, py, *, k: int = 10, max_dist: float = 50.0,
                      chunk_size: int = 200_000):
        """Nearest edge per point, vectorised over many points.

        Returns ``(edge_ids, snap_dists)`` arrays: ``edge_ids[i] == -1`` and
        ``snap_dists[i] == NaN`` where no edge lies within ``max_dist``.
        Identical results to calling :meth:`candidate_edges` per point and
        taking the closest candidate; ``chunk_size`` only bounds peak memory.
        """
        px = np.asarray(px, dtype=float)
        py = np.asarray(py, dtype=float)
        n = px.size
        edge_ids = np.full(n, -1, dtype=np.int64)
        snap = np.full(n, np.nan)
        for s in range(0, n, chunk_size):
            e = min(s + chunk_size, n)
            em, pm, _tm, _am = self._candidates_matrix(px[s:e], py[s:e], k)
            j = np.argmin(pm, axis=1)
            rows = np.arange(e - s)
            best = pm[rows, j]
            ok = best <= max_dist
            edge_ids[s:e][ok] = em[rows, j][ok]
            snap[s:e][ok] = best[ok]
        return edge_ids, snap

    def candidate_edges_batch(self, px, py, *, k: int = 10,
                              max_dist: float = 50.0):
        """Per-point candidate lists for many points, in one vectorised pass.

        Returns a list (one entry per point) of ``(edge_id, perp_dist, t)``
        candidate lists with the same content and ordering as
        :meth:`candidate_edges`.
        """
        em, pm, tm, _am = self._candidates_matrix(px, py, k)
        order_all = np.argsort(pm, axis=1, kind="stable")  # one call, all rows
        out = []
        for i in range(len(em)):
            best = {}
            for j in order_all[i]:
                d = pm[i, j]
                if d > max_dist:
                    break  # sorted ascending: nothing further qualifies
                e = int(em[i, j])
                if e not in best:
                    best[e] = (float(d), float(tm[i, j]))
            out.append([(e, d, tt) for e, (d, tt) in best.items()])
        return out

    def _candidate_arcs_batch(self, px, py, *, k: int = 10,
                              max_dist: float = 50.0):
        """Per-point candidates as ``(edge_id, perp_dist, arc_m)``.

        Same candidates, ordering and tolerance as
        :meth:`candidate_edges_batch`, but the third element is the snap's
        **arc length along the edge in metres**, measured from the directed
        edge's start node -- the quantity
        :func:`redlight.speeds._arc_position` computes with shapely, obtained
        here from the snap table at no extra cost.

        It is deliberately a separate method rather than a third element added
        to :meth:`candidate_edges_batch`: that method's ``t`` is a fraction of
        one densified sub-segment, and the two must not be confused. Scaling
        ``t`` by the edge length is wrong by up to a full edge length, which is
        exactly the bug this method exists to make unavailable.
        """
        em, pm, _tm, am = self._candidates_matrix(px, py, k)
        order_all = np.argsort(pm, axis=1, kind="stable")
        out = []
        for i in range(len(em)):
            best = {}
            for j in order_all[i]:
                d = pm[i, j]
                if d > max_dist:
                    break  # sorted ascending: nothing further qualifies
                e = int(em[i, j])
                if e not in best:
                    best[e] = (float(d), float(am[i, j]))
            out.append([(e, d, a) for e, (d, a) in best.items()])
        return out

    # ------------------------------------------------------------ CSR graph
    def csgraph(self):
        """The directed graph as a scipy CSR matrix, plus node index maps.

        Returns ``(csr, node_to_int, int_to_node)`` where ``csr[i, j]`` is the
        minimum ``length_m`` over parallel edges from node ``i`` to node ``j``
        (the same distance Dijkstra would use on the multigraph). Built lazily
        on first use and cached; scipy's C Dijkstra on this matrix replaces
        pure-Python networkx searches in hot paths.
        """
        if self._csr is None:
            nodes = list(self.graph.nodes())
            node_int = {node: i for i, node in enumerate(nodes)}
            best: dict = {}
            for u, v, d in self.graph.edges(data=True):
                key = (node_int[u], node_int[v])
                length = float(d["length_m"])
                if key not in best or length < best[key]:
                    best[key] = length
            from scipy.sparse import csr_matrix
            rows = np.fromiter((k[0] for k in best), dtype=np.int64, count=len(best))
            cols = np.fromiter((k[1] for k in best), dtype=np.int64, count=len(best))
            vals = np.fromiter(best.values(), dtype=float, count=len(best))
            self._csr = csr_matrix((vals, (rows, cols)),
                                   shape=(len(nodes), len(nodes)))
            self._node_int = node_int
            self._int_node = nodes
        return self._csr, self._node_int, self._int_node

    # ------------------------------------------------------------- convenience
    def edge_endpoints(self, edge_id: int):
        """Return the (u, v) node tuple of a directed edge."""
        return self._edge_index[edge_id]

    def edge_length(self, edge_id: int) -> float:
        """Length in metres of one directed edge."""
        return self._edge_lengths[edge_id]

    def edge_geometry(self, edge_id: int):
        """Projected (metric CRS) LineString geometry of one directed edge."""
        return self._edge_geoms_proj[edge_id]

    def edge_data(self, edge_id: int) -> dict:
        """The attribute dict of one directed edge (mutating it updates the graph)."""
        u, v = self._edge_index[edge_id]
        return self.graph[u][v][edge_id]

    def edge_coords_lonlat(self, edge_id: int):
        """Edge geometry as a list of (lon, lat) pairs (WGS84)."""
        geom = self._edge_geoms_proj[edge_id]
        xs, ys = np.asarray(geom.coords)[:, 0], np.asarray(geom.coords)[:, 1]
        lon, lat = self._transformer_inv.transform(xs, ys)
        return [(float(a), float(b)) for a, b in zip(lon, lat)]

    def road_edge_ids(self, edge_id: int):
        """All directed edge ids of the physical road ``edge_id`` belongs to.

        One id for a one-way road, two (the edge and its opposite direction)
        for a two-way road. Distinct parallel roads between the same nodes are
        NOT conflated.
        """
        rev = self._reverse_of.get(edge_id)
        return [edge_id] if rev is None else sorted((edge_id, rev))

    def edges_between(self, x, y):
        """All directed edges between nodes ``x`` and ``y`` in either direction.

        Returns a list of ``(edge_id, length_m)`` covering parallel roads too.
        """
        out = []
        for a, b in ((x, y), (y, x)):
            data = self.graph.get_edge_data(a, b)
            if data:
                for attrs in data.values():
                    out.append((int(attrs["edge_id"]), float(attrs["length_m"])))
        return out

    def number_of_edges(self) -> int:
        """Total directed edge count (each direction of a two-way road counts
        separately)."""
        return self.graph.number_of_edges()

    def number_of_nodes(self) -> int:
        """Total node (intersection/endpoint) count."""
        return self.graph.number_of_nodes()

    @property
    def edge_ids(self):
        """Sorted ``numpy.ndarray`` of every directed edge id in the network."""
        return self._edge_ids
