#!/usr/bin/env python3
"""Produce a customer-ready congestion study from raw GPS and a road network.

End-to-end pipeline for GPS data that carries **no speed column** -- position,
time, a mover id and a horizontal-accuracy value are enough:

    load network (GeoJSON)
      -> load points (tz-aware, local clock)
      -> HMM / Viterbi map matching (Newson & Krumm)
      -> derive_speeds from on-road displacement, using per-point accuracy
      -> dwell-aware + robust cleaning
      -> assign overall / peak / off-peak speeds onto the network
      -> analytics (temporal, day-type, congestion vs posted limit, structure)
      -> a single self-contained HTML deck

The report is one ``.html`` file with every image inlined as base64, so it can
be emailed as-is and opened with no dependencies, and printed to PDF from any
browser (Ctrl/Cmd-P). It commits deliberately to a light, print-oriented look
rather than following the reader's OS theme -- the embedded raster maps cannot
re-theme, and a document that prints predictably is worth more here than one
that adapts to a screen.

Usage
-----
    python scripts/customer_report.py \\
        --network roads.geojson \\
        --points  probes.csv \\
        --accuracy-col accuracy \\
        --tz America/New_York \\
        --title "Route 1 Corridor Study" \\
        --customer "Example County DOT" \\
        --out report.html

Requires the ``mapping`` extra for figures: ``pip install roadtraffic[mapping]``
(matplotlib). Every stage degrades gracefully -- a study area with no posted
speed limits, or GPS that never sampled a weekend, drops the affected section
and says so in "Data notes" rather than failing or inventing a number.
"""
from __future__ import annotations

import argparse
import base64
import io
import sys
import textwrap
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import roadtraffic as rt

# --------------------------------------------------------------------------- #
# Palette. Light-mode values from the project's validated data-viz palette.
# Colour choices that matter, and why they are not the conventional ones:
#
#   * Absolute speed is a *magnitude*, so it gets a single-hue sequential ramp
#     (blue, light->dark). The familiar red-yellow-green traffic ramp is a
#     rainbow -- it puts a hue at the midpoint, and red-vs-green is precisely
#     the pair that deuteranopic and protanopic readers cannot separate. On a
#     printed page handed to a room, that is a real failure, not a theoretical
#     one.
#   * The congestion ratio *does* have a natural midpoint (1.0 = running at the
#     posted limit), so it gets a true diverging scale: red below, neutral grey
#     at, blue above. Two hues that read as opposite, and nothing at the middle.
#   * Map line-work starts at ramp step 250 rather than 100: the palest
#     sequential steps are designed to recede into the surface, which is right
#     for a filled choropleth and wrong for a 2px line that then vanishes.
# --------------------------------------------------------------------------- #
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
PLANE = "#f9f9f7"
SERIES_1 = "#2a78d6"     # blue
SERIES_2 = "#eb6834"     # orange
GOOD = "#0ca30c"
WARNING = "#fab219"
CRITICAL = "#d03b3b"

# Sequential blue, steps 250..700 -- see note above on why not 100.
SEQ_BLUE = ["#86b6ef", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
            "#1c5cab", "#184f95", "#104281", "#0d366b"]
# Diverging poles + neutral midpoint. The midpoint is the *baseline* grey, not
# the palette's pale diverging grey: on a near-white surface a pale midpoint
# makes every road near the middle of the scale disappear, and on this map the
# middle is where most roads live. A mid grey still reads as "neither pole"
# (it carries no hue) while staying legible as a 2px line.
DIV_LOW, DIV_MID, DIV_HIGH = "#d03b3b", "#c3c2b7", "#2a78d6"

FONT_STACK = ["system-ui", "-apple-system", "Segoe UI", "Helvetica", "Arial",
              "DejaVu Sans", "sans-serif"]


def _mpl():
    """Import and configure matplotlib, with an actionable error if absent."""
    try:
        import matplotlib
    except ImportError as exc:  # pragma: no cover - depends on user's env
        raise SystemExit(
            "This report needs matplotlib for its figures.\n"
            "    pip install 'roadtraffic[mapping]'"
        ) from exc
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": FONT_STACK,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.dpi": 150,
    })
    return plt


def _cmaps():
    from matplotlib.colors import LinearSegmentedColormap
    seq = LinearSegmentedColormap.from_list("rt_seq", SEQ_BLUE)
    div = LinearSegmentedColormap.from_list(
        "rt_div", [DIV_LOW, DIV_MID, DIV_HIGH])
    return seq, div


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _fig_to_b64(fig, plt) -> str:
    """Render a figure to a base64 PNG and close it."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _esc(text) -> str:
    """Minimal HTML escaping for interpolated text."""
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def _fmt(value, digits=1, dash="--"):
    """Format a number for display, rendering NaN/None as an em-dash."""
    if value is None:
        return dash
    try:
        f = float(value)
    except (TypeError, ValueError):
        return _esc(value)
    if not np.isfinite(f):
        return dash
    return f"{f:,.{digits}f}"


def _map_axes(ax, network):
    """Configure an axis for a lon/lat map: correct aspect, no chrome."""
    lats = [n[1] for n in network.graph.nodes()]
    mid = float(np.mean(lats)) if lats else 0.0
    # 1 degree of longitude is cos(lat) as long as 1 degree of latitude; without
    # this correction every map is horizontally stretched.
    ax.set_aspect(1.0 / max(np.cos(np.radians(mid)), 1e-6))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _draw_network(ax, network, values=None, cmap=None, vmin=None, vmax=None,
                  base_color=GRID, lw=1.6, base_lw=0.9):
    """Draw edges, optionally coloured by ``values`` (edge_id -> float).

    Edges with no value are drawn in the recessive base colour first, so the
    unmeasured extent of the network stays visible instead of silently
    disappearing -- a map that hides its own coverage gaps overstates the study.
    """
    from matplotlib.collections import LineCollection
    plain, coloured, cvals = [], [], []
    for eid in network.edge_ids:
        coords = network.edge_coords_lonlat(int(eid))
        v = None if values is None else values.get(int(eid))
        if v is None or not np.isfinite(v):
            plain.append(coords)
        else:
            coloured.append(coords)
            cvals.append(v)
    if plain:
        ax.add_collection(LineCollection(plain, colors=base_color,
                                         linewidths=base_lw, zorder=1))
    sm = None
    if coloured:
        lc = LineCollection(coloured, cmap=cmap, linewidths=lw, zorder=2)
        lc.set_array(np.asarray(cvals))
        if vmin is not None:
            lc.set_clim(vmin, vmax)
        sm = ax.add_collection(lc)
    ax.autoscale_view()
    all_coords = plain + coloured
    if all_coords:
        xs = [p[0] for c in all_coords for p in c]
        ys = [p[1] for c in all_coords for p in c]
        pad_x = (max(xs) - min(xs)) * 0.04 or 0.001
        pad_y = (max(ys) - min(ys)) * 0.04 or 0.001
        ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
        ax.set_ylim(min(ys) - pad_y, max(ys) + pad_y)
    return sm


def _colorbar(fig, sm, ax, label):
    cb = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label(label, color=INK_2, fontsize=9)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, color=MUTED, labelcolor=INK_2)
    return cb


def _runs(sorted_ints):
    """Yield (start, end) of each run of consecutive integers."""
    if not sorted_ints:
        return
    start = prev = sorted_ints[0]
    for h in sorted_ints[1:]:
        if h == prev + 1:
            prev = h
            continue
        yield start, prev
        start = prev = h
    yield start, prev


def _table(headers, rows, caption=None) -> str:
    """A table view for a figure -- the accessible twin of every chart here.

    The maps and charts are rasterised PNGs, so they cannot carry a hover
    tooltip. Every figure therefore ships the underlying numbers as a real
    table rather than leaving colour as the only way to read a value.
    """
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows
    )
    cap = f"<caption>{_esc(caption)}</caption>" if caption else ""
    return (f'<details class="tv"><summary>Data table</summary>'
            f'<div class="tw"><table>{cap}<thead><tr>{head}</tr></thead>'
            f"<tbody>{body}</tbody></table></div></details>")


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def run_pipeline(args, notes: list) -> dict:
    """Load, match, derive, clean and aggregate. Returns everything the deck needs."""
    print(f"[1/7] network  <- {args.network}", file=sys.stderr)
    net = rt.Network.from_geojson(args.network)
    print(f"      {net.number_of_nodes():,} nodes / {net.number_of_edges():,} edges",
          file=sys.stderr)

    print(f"[2/7] points   <- {args.points}", file=sys.stderr)
    pts = rt.load_points(args.points, tz=args.tz, id_col=args.id_col,
                         time_col=args.time_col, lon_col=args.lon_col,
                         lat_col=args.lat_col)
    pdf = pts.df
    if not pts.has_traj:
        raise SystemExit(
            "HMM matching needs a trajectory/mover id column to order fixes "
            "within a track. Pass --id-col explicitly."
        )
    if "speed_mps" in pdf.columns:
        notes.append(
            "The input carried a speed column; it was ignored. Speeds here are "
            "reconstructed from on-road displacement after matching, which is "
            "robust to the receiver's own instantaneous-speed estimate.")

    acc = args.accuracy_col
    if acc and acc in pdf.columns:
        a = pd.to_numeric(pdf[acc], errors="coerce")
        med = float(a.median()) if a.notna().any() else float("nan")
        print(f"      accuracy '{acc}': median {med:.1f} m", file=sys.stderr)
        # A horizontal accuracy in metres is typically ~3-50 m. A median below
        # ~1.5 is the signature of an HDOP/dimensionless column, which would be
        # silently interpreted as metres and make the error model far too
        # confident -- inflating the share of intervals that pass the quality
        # screen. Surface it rather than let it bias the study invisibly.
        if np.isfinite(med) and med < 1.5:
            notes.append(
                f"Accuracy column '{acc}' has a median of {med:.2f}, which is "
                "low for a horizontal accuracy in metres and typical of an HDOP "
                "(dimensionless) field. It is being used as a per-point sigma in "
                "metres; if it is HDOP, multiply it by the receiver's UERE "
                "(commonly 3-5 m) before running this report.")
    elif acc:
        notes.append(f"Accuracy column '{acc}' not found; every point fell back "
                     f"to the default sigma of {args.default_sigma} m.")
        acc = None

    print("[3/7] matching (HMM/Viterbi)", file=sys.stderr)
    matched = rt.HMMMatcher(net, max_dist=args.max_dist).match(pts)
    n_unmatched = int((matched["edge_id"] == -1).sum())
    if n_unmatched:
        pct = 100.0 * n_unmatched / max(len(matched), 1)
        notes.append(f"{n_unmatched:,} of {len(matched):,} fixes ({pct:.1f}%) "
                     f"snapped to no road within {args.max_dist} m and are "
                     "excluded from every statistic below.")

    print("[4/7] deriving speeds from on-road displacement", file=sys.stderr)
    derived = rt.derive_speeds(
        net, matched, pts,
        pos_accuracy_col=acc,
        default_pos_sigma_m=args.default_sigma,
        min_baseline_m=args.min_baseline,
    )
    intervals, obs = derived["intervals"], derived["edge_observations"]
    if not len(intervals):
        raise SystemExit(
            "No speed intervals could be derived. Common causes: fixes too far "
            "apart in time (see --max-dt), only one fix per mover, or a "
            "trajectory id that is not actually grouping the tracks."
        )
    n_bad = int((~intervals["quality"]).sum())
    bad_frac = n_bad / max(len(intervals), 1)
    if n_bad:
        notes.append(
            f"{n_bad:,} of {len(intervals):,} derived intervals "
            f"({100.0 * bad_frac:.1f}%) failed the quality screen "
            "-- displacement not clearly above the GPS noise floor, an implausible "
            "implied speed, or a poor snap. "
            + ("They are excluded (--require-quality)." if args.require_quality
               else "They are INCLUDED; re-run with --require-quality to exclude them."))
    # The quality screen is not neutral with respect to speed. It rejects
    # intervals whose displacement is small relative to GPS noise, and slow
    # traffic covers the least ground per fix -- so excluding failures
    # preferentially deletes the slowest measurements and biases every speed
    # UPWARD, understating exactly the congestion a study like this exists to
    # measure. Measured on a synthetic set with known ground truth, peak-hour
    # congestion was understated by 18% with --require-quality alone, and by
    # only 3% once --min-baseline merged hops to lift displacement clear of the
    # noise floor (which also lifted the pass rate from 75% to 95%).
    if args.require_quality and args.min_baseline is None and bad_frac > 0.10:
        notes.append(
            "WARNING -- possible upward speed bias. --require-quality is on, "
            f"{100.0 * bad_frac:.0f}% of intervals were rejected, and no "
            "--min-baseline was set. Rejected intervals are disproportionately "
            "the slow ones, so congestion here is likely UNDERSTATED. Re-run "
            "with --min-baseline 150 (or larger) to merge hops until the "
            "displacement clears the noise floor, then compare the two runs.")

    print("[5/7] cleaning", file=sys.stderr)
    # filter_trajectory_speed is deliberately NOT used here: it is the cleaner
    # for matched *point* frames (it needs lon/lat/traj_id to detect dwells),
    # whereas derive_speeds emits interval observations that carry no position.
    # The dwell case it exists to catch is already covered on this path -- a
    # parked vehicle produces displacement below the GPS noise floor, which the
    # quality screen flags -- so --require-quality is the equivalent control.
    before = len(obs)
    clean = rt.filter_by_speed(obs, max_speed=args.max_speed, unit=args.unit,
                               mad_outliers=True, per_edge=True)
    dropped = before - len(clean)
    if dropped:
        notes.append(
            f"Cleaning removed {dropped:,} of {before:,} edge observations "
            f"({100.0 * dropped / before:.1f}%) as above {args.max_speed} "
            f"{args.unit} or as robust (MAD) per-edge outliers.")
    if not len(clean):
        raise SystemExit("Cleaning removed every observation; raise --max-speed.")

    rq = args.require_quality
    print("[6/7] assigning speeds + aggregating", file=sys.stderr)
    seg = rt.assign_segment_speeds(net, clean, statistic=args.statistic,
                                   n_peak=args.n_peak, n_offpeak=args.n_offpeak,
                                   require_quality=rq)
    hourly = rt.aggregate_speeds(clean, block_hours=1, statistic="both",
                                 output_unit=args.unit, require_quality=rq,
                                 min_samples=args.min_samples)
    peaks = (rt.peak_analysis(hourly, statistic=args.statistic,
                              n_peak=args.n_peak, n_offpeak=args.n_offpeak)
             if len(hourly) else None)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        daytype = rt.day_type_report(clean, statistic=args.statistic,
                                     output_unit=args.unit, require_quality=rq)
        for w in caught:
            if "no observations" in str(w.message):
                notes.append("One day-type had no observations, so the "
                             "weekday/weekend comparison is partial.")

    congestion = rt.congestion_report(net, clean, statistic=args.statistic,
                                      output_unit=args.unit, require_quality=rq)
    if congestion["summary"]["n_edges_rated"] == 0:
        congestion = None
        notes.append("No edge carried a usable OSM 'maxspeed' tag, so the "
                     "congestion-vs-posted-limit section is omitted. Re-export "
                     "the network with maxspeed to enable it.")

    print("[7/7] network structure", file=sys.stderr)
    stats = rt.network_stats(net, area_km2=args.area_km2)
    conn = rt.connectivity_report(net)
    try:
        bc = rt.edge_betweenness_centrality(net, weight="travel_time_s")
    except ValueError as exc:
        bc = None
        notes.append(f"Chokepoint analysis skipped: {exc}".split(" Fix by")[0])

    return {
        "net": net, "points": pdf, "matched": matched, "intervals": intervals,
        "clean": clean, "seg": seg, "hourly": hourly, "peaks": peaks,
        "daytype": daytype, "congestion": congestion, "stats": stats,
        "conn": conn, "bc": bc,
    }


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_coverage(plt, net, seg):
    fig, ax = plt.subplots(figsize=(9, 6.2))
    observed = {}
    for eid in net.edge_ids:
        d = net.edge_data(int(eid))
        if d.get("obs_speed_mps_overall") is not None or \
           d.get("obs_speed_mps") is not None:
            observed[int(eid)] = 1.0
    _draw_network(ax, net, values=observed, cmap=None, base_color=GRID, lw=1.8)
    from matplotlib.collections import LineCollection
    seen = [net.edge_coords_lonlat(e) for e in observed]
    if seen:
        ax.add_collection(LineCollection(seen, colors=SERIES_1,
                                         linewidths=1.8, zorder=3))
    _map_axes(ax, net)
    cov = 100.0 * len(observed) / max(net.number_of_edges(), 1)
    ax.set_title(f"Survey coverage — {len(observed):,} of "
                 f"{net.number_of_edges():,} edges observed ({cov:.0f}%)",
                 fontsize=12, pad=12, loc="left")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color=SERIES_1, lw=2, label="Observed"),
                       Line2D([], [], color=GRID, lw=2, label="No data")],
              loc="lower right", fontsize=9)
    return fig


def fig_speed_map(plt, net, period, unit, vlim=None):
    seq, _ = _cmaps()
    attr = ("obs_speed_mps_overall" if period == "overall"
            else f"obs_speed_mps_{period}")
    vals = {}
    for eid in net.edge_ids:
        v = net.edge_data(int(eid)).get(attr)
        if v is None:
            v = net.edge_data(int(eid)).get("obs_speed_mps")
        if v is not None and np.isfinite(v):
            vals[int(eid)] = float(rt.from_mps(v, unit))
    if not vals:
        return None, None
    lo, hi = (vlim if vlim else (min(vals.values()), max(vals.values())))
    fig, ax = plt.subplots(figsize=(9, 6.2))
    sm = _draw_network(ax, net, values=vals, cmap=seq, vmin=lo, vmax=hi, lw=2.0)
    _map_axes(ax, net)
    if sm:
        _colorbar(fig, sm, ax, f"Speed ({unit})")
    ax.set_title(f"{period.capitalize()} speed", fontsize=12, pad=12, loc="left")
    return fig, (lo, hi)


def fig_hourly(plt, hourly, unit, peaks):
    col = "median_speed" if "median_speed" in hourly.columns else "mean_speed"
    h = hourly.sort_values("block_start_hour")
    x = h["block_start_hour"].to_numpy()
    y = h[col].to_numpy()
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.grid(axis="y", zorder=0)
    if {"ci95_low", "ci95_high"} <= set(h.columns) and h["ci95_low"].notna().any():
        ax.fill_between(x, h["ci95_low"], h["ci95_high"], color=SERIES_1,
                        alpha=0.15, linewidth=0, zorder=2,
                        label="95% CI of the mean")
    ax.plot(x, y, color=SERIES_1, lw=2, zorder=3, marker="o", ms=4,
            label=f"{col.split('_')[0].capitalize()} speed")
    # Direct-label only the extremes -- a number on every point is unreadable.
    if len(y):
        i_lo, i_hi = int(np.argmin(y)), int(np.argmax(y))
        for i, va in ((i_lo, "top"), (i_hi, "bottom")):
            ax.annotate(f"{y[i]:.0f}", (x[i], y[i]), textcoords="offset points",
                        xytext=(0, -14 if va == "top" else 10), ha="center",
                        fontsize=9, color=INK, fontweight="bold")
    if peaks:
        # Merge consecutive peak hours into one span. Drawing them per-hour
        # leaves a visible seam between neighbours that reads as two separate
        # windows when it is really one continuous rush period.
        hrs = sorted(r["block_start_hour"] for r in peaks["peak"])
        for run_start, run_end in _runs(hrs):
            ax.axvspan(run_start - 0.5, run_end + 0.5, color=CRITICAL,
                       alpha=0.08, lw=0, zorder=1)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlim(-0.6, 23.6)
    ax.set_xlabel("Hour of day (local clock)", fontsize=10)
    ax.set_ylabel(f"Speed ({unit})", fontsize=10)
    # Below the axes. The curve's shape changes with the data, so any in-axes
    # corner eventually sits on a line, and the top-right strip collides with
    # the title once the title is more than a few words.
    ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.17),
              ncol=2)
    ax.set_title("Speed by hour of day", fontsize=12, pad=12, loc="left")
    return fig


def fig_daytype(plt, daytype, unit):
    comp = daytype["comparison"]
    labels = [k for k in daytype["groups"]]
    cols = [f"{lb}_speed" for lb in labels]
    if comp is None or not len(comp) or not all(c in comp.columns for c in cols):
        return None
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.grid(axis="y", zorder=0)
    palette = [SERIES_1, SERIES_2]
    for i, (lb, c) in enumerate(zip(labels, cols)):
        sub = comp[["block_start_hour", c]].dropna()
        if not len(sub):
            continue
        ax.plot(sub["block_start_hour"], sub[c], lw=2, marker="o", ms=4,
                color=palette[i % 2], label=lb.capitalize(), zorder=3)
        last = sub.iloc[-1]
        ax.annotate(lb.capitalize(), (last["block_start_hour"], last[c]),
                    textcoords="offset points", xytext=(6, 0), fontsize=9,
                    color=INK_2, va="center")
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlim(-0.6, 25.5)
    ax.set_xlabel("Hour of day (local clock)", fontsize=10)
    ax.set_ylabel(f"Speed ({unit})", fontsize=10)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_title("Weekday vs weekend", fontsize=12, pad=12, loc="left")
    return fig


def fig_congestion_map(plt, net, congestion):
    _, div = _cmaps()
    e = congestion["edges"]
    rated = e[e["ratio"].notna()]
    if not len(rated):
        return None
    vals = dict(zip(rated["edge_id"].astype(int), rated["ratio"].astype(float)))
    fig, ax = plt.subplots(figsize=(9, 6.2))
    # Centre the diverging scale on 1.0 (running at the posted limit) with
    # symmetric arms, so equal visual distance means equal deviation. The span
    # comes from the 5th/95th percentile rather than the extremes: one freak
    # segment would otherwise compress every other road into the midpoint.
    # Values beyond the span saturate at the pole colour, which is the intended
    # reading ("as congested as the scale goes").
    v = np.asarray(list(vals.values()))
    span = max(abs(np.percentile(v, 5) - 1.0),
               abs(np.percentile(v, 95) - 1.0), 0.15)
    sm = _draw_network(ax, net, values=vals, cmap=div,
                       vmin=1 - span, vmax=1 + span, lw=2.0)
    _map_axes(ax, net)
    if sm:
        cb = _colorbar(fig, sm, ax, "Observed ÷ posted limit")
        cb.ax.axhline(1.0, color=INK_2, lw=1)
    ax.set_title("Congestion relative to the posted speed limit",
                 fontsize=12, pad=12, loc="left")
    return fig


def _by_road(net, edges):
    """Collapse directed edges to physical roads, keeping the worst direction.

    A two-way street is two directed edges, so ranking on edge_id lists every
    street twice -- which reads as a bug to anyone looking at the deck. Group
    on the canonical road id (the same ``min(road_edge_ids)`` idiom the rest of
    the package uses) and keep the slower direction, since this is a ranking of
    worst-case congestion.
    """
    out = edges.copy()
    out["road_id"] = [min(net.road_edge_ids(int(e))) for e in out["edge_id"]]
    idx = out.groupby("road_id")["ratio"].idxmin()
    return out.loc[idx.dropna()]


def fig_worst_corridors(plt, net, congestion, top=12):
    e = congestion["edges"]
    rated = _by_road(net, e[e["ratio"].notna()]).sort_values("ratio").head(top)
    if not len(rated):
        return None
    names = []
    for eid in rated["edge_id"].astype(int):
        d = net.edge_data(int(eid))
        names.append(str(d.get("name") or d.get("highway") or f"edge {eid}")[:34])
    fig, ax = plt.subplots(figsize=(9, max(3.0, 0.38 * len(rated) + 1.4)))
    ax.grid(axis="x", zorder=0)
    y = np.arange(len(rated))
    # One series -> one colour. A value-ramp here would double-encode length.
    ax.barh(y, rated["ratio"], color=SERIES_1, height=0.62, zorder=3)
    ax.axvline(1.0, color=BASELINE, lw=1.2, zorder=4)
    ax.text(1.0, -0.9, "posted limit", fontsize=8, color=MUTED, ha="center")
    for i, v in enumerate(rated["ratio"]):
        ax.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(y, names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, max(1.15, float(rated["ratio"].max()) * 1.18))
    ax.set_xlabel("Observed speed ÷ posted limit (slower direction)", fontsize=10)
    ax.set_title(f"Most congested roads (worst {len(rated)})",
                 fontsize=12, pad=12, loc="left")
    return fig


def fig_chokepoints(plt, net, bc):
    seq, _ = _cmaps()
    vals = {int(k): float(v) for k, v in bc.items() if np.isfinite(v)}
    if not vals or max(vals.values()) <= 0:
        return None
    fig, ax = plt.subplots(figsize=(9, 6.2))
    sm = _draw_network(ax, net, values=vals, cmap=seq, vmin=0,
                       vmax=max(vals.values()), lw=2.0)
    _map_axes(ax, net)
    if sm:
        _colorbar(fig, sm, ax, "Edge betweenness")
    ax.set_title("Chokepoints — travel-time-weighted edge betweenness",
                 fontsize=12, pad=12, loc="left")
    return fig


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #
CSS = """
:root{--plane:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;
 --muted:#898781;--grid:#e1e0d9;--warning:#fab219}
*{box-sizing:border-box}
body{margin:0;background:var(--plane);color:var(--ink);
 font-family:system-ui,-apple-system,"Segoe UI",Helvetica,Arial,sans-serif;
 line-height:1.55;-webkit-print-color-adjust:exact;print-color-adjust:exact}
.slide{background:var(--surface);max-width:1080px;margin:0 auto 22px;padding:40px 46px;
 border:1px solid rgba(11,11,11,.10);border-radius:10px}
.slide.cover{padding:64px 46px}
h1{font-size:31px;line-height:1.2;margin:0 0 6px;letter-spacing:-.02em}
h2{font-size:21px;margin:0 0 4px;letter-spacing:-.01em}
.eyebrow{font-size:11px;letter-spacing:.13em;text-transform:uppercase;
 color:var(--muted);margin:0 0 14px;font-weight:600}
.sub{color:var(--ink2);font-size:14px;margin:0 0 22px;max-width:74ch}
p{font-size:14px;color:var(--ink2);max-width:80ch}
img{width:100%;height:auto;display:block;margin:8px 0 4px}
.tiles{display:flex;flex-wrap:wrap;gap:12px;margin:20px 0 6px}
.tile{flex:1 1 165px;background:var(--plane);border:1px solid rgba(11,11,11,.08);
 border-radius:8px;padding:14px 16px}
.tile .v{font-size:27px;font-weight:650;letter-spacing:-.02em;line-height:1.15}
.tile .k{font-size:11px;color:var(--muted);text-transform:uppercase;
 letter-spacing:.07em;margin-top:5px;font-weight:600}
.tile .n{font-size:11.5px;color:var(--ink2);margin-top:5px}
.tv{margin-top:10px}
.tv summary{cursor:pointer;font-size:12px;color:var(--ink2);padding:5px 0;
 font-weight:600}
.tw{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:12.5px;
 font-variant-numeric:tabular-nums;margin-top:6px}
th,td{text-align:right;padding:6px 10px;border-bottom:1px solid var(--grid)}
th:first-child,td:first-child{text-align:left}
th{color:var(--muted);font-weight:600;text-transform:uppercase;font-size:10.5px;
 letter-spacing:.05em}
caption{caption-side:top;text-align:left;color:var(--muted);font-size:11.5px;
 padding-bottom:5px}
.notes{border-left:3px solid var(--warning);padding:2px 0 2px 16px;margin:16px 0}
.notes li{font-size:13px;color:var(--ink2);margin:7px 0}
.foot{max-width:1080px;margin:0 auto 40px;padding:0 46px;color:var(--muted);
 font-size:11.5px}
.kv{display:flex;gap:26px;flex-wrap:wrap;margin:18px 0 0}
.kv div{font-size:13px;color:var(--ink2)}
.kv b{display:block;color:var(--muted);font-size:10.5px;text-transform:uppercase;
 letter-spacing:.07em;font-weight:600;margin-bottom:2px}
@media print{body{background:#fff}
 .slide{break-after:page;border:none;border-radius:0;margin:0;max-width:none}
 .tv[open] summary{display:none}}
"""


def tile(value, key, note=""):
    n = f'<div class="n">{_esc(note)}</div>' if note else ""
    return (f'<div class="tile"><div class="v">{value}</div>'
            f'<div class="k">{_esc(key)}</div>{n}</div>')


def slide(body, cover=False):
    return f'<section class="slide{" cover" if cover else ""}">{body}</section>'


def img(b64, alt):
    return f'<img src="data:image/png;base64,{b64}" alt="{_esc(alt)}">'


def build_html(args, R, figs, notes) -> str:
    net, unit = R["net"], args.unit
    pdf, intervals = R["points"], R["intervals"]
    seg, stats, conn = R["seg"], R["stats"], R["conn"]
    t0, t1 = pdf["time"].min(), pdf["time"].max()
    days = max((t1 - t0).days + 1, 1)
    n_mov = int(pdf["traj_id"].nunique()) if "traj_id" in pdf.columns else 0
    S = []

    # ---- 1. cover ------------------------------------------------------- #
    S.append(slide(f"""
      <p class="eyebrow">Trafficability study</p>
      <h1>{_esc(args.title)}</h1>
      <p class="sub">{_esc(args.customer)}</p>
      <div class="tiles">
        {tile(f"{len(pdf):,}", "GPS fixes")}
        {tile(f"{n_mov:,}", "Movers")}
        {tile(f"{len(intervals):,}", "Speed intervals", "independent measurements")}
        {tile(f"{days:,}", "Days observed")}
      </div>
      <div class="kv">
        <div><b>Period</b>{t0:%Y-%m-%d %H:%M} — {t1:%Y-%m-%d %H:%M}</div>
        <div><b>Network</b>{net.number_of_edges():,} edges /
             {net.number_of_nodes():,} nodes</div>
        <div><b>Timezone</b>{_esc(args.tz or "as supplied")}</div>
        <div><b>Generated</b>{datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC</div>
      </div>
      <p style="margin-top:26px">Speeds in this report were not read from the
      GPS receiver. Each fix was map-matched to the road network with a hidden
      Markov model, and speed reconstructed from <b>on-road displacement</b>
      between consecutive fixes of the same mover — the distance actually
      travelled along the roadway, divided by elapsed time.</p>""", cover=True))

    # ---- 2. coverage ---------------------------------------------------- #
    if figs.get("coverage"):
        obs_edges = seg["coverage"]["overall"]
        S.append(slide(f"""
          <p class="eyebrow">01 — Survey coverage</p>
          <h2>Where the data actually is</h2>
          <p class="sub">Roads in grey were never traversed by the survey and
          carry no measured speed. Every statistic in this report describes the
          blue extent only.</p>
          {img(figs["coverage"], "Survey coverage map")}
          <div class="tiles">
            {tile(f"{obs_edges:,}", "Edges observed")}
            {tile(f"{100.0 * obs_edges / max(net.number_of_edges(), 1):.0f}%", "Of network")}
            {tile(f"{seg['coverage']['peak']:,}", "In peak window")}
            {tile(f"{seg['coverage']['offpeak']:,}", "In off-peak window")}
          </div>"""))

    # ---- 3-4. speed maps ------------------------------------------------ #
    if figs.get("speed_overall"):
        S.append(slide(f"""
          <p class="eyebrow">02 — Measured speed</p>
          <h2>Overall speed by segment</h2>
          <p class="sub">The {args.statistic} speed across the whole survey
          period. Darker is faster.</p>
          {img(figs["speed_overall"], "Overall speed map")}"""))

    if figs.get("speed_peak") or figs.get("speed_offpeak"):
        peak_h = ", ".join(f"{h:02d}:00" for h in seg["peak_hours"][:6])
        off_h = ", ".join(f"{h:02d}:00" for h in seg["offpeak_hours"][:6])
        both = "".join(img(figs[k], f"{k} speed map")
                       for k in ("speed_peak", "speed_offpeak") if figs.get(k))
        S.append(slide(f"""
          <p class="eyebrow">03 — Peak vs off-peak</p>
          <h2>The same roads, at their worst and their best</h2>
          <p class="sub">Both maps share one colour scale, so they are directly
          comparable. Peak hours ({_esc(peak_h)}) were chosen from the data as
          the slowest contiguous window; off-peak ({_esc(off_h)}) the fastest.</p>
          {both}"""))

    # ---- 5. hourly profile ---------------------------------------------- #
    if figs.get("hourly"):
        h = R["hourly"].sort_values("block_start_hour")
        col = "median_speed" if "median_speed" in h.columns else "mean_speed"
        rows = [(f"{int(r.block_start_hour):02d}:00", _fmt(getattr(r, col)),
                 f"{int(r.n):,}") for r in h.itertuples()]
        S.append(slide(f"""
          <p class="eyebrow">04 — Time of day</p>
          <h2>When the network slows down</h2>
          <p class="sub">Network-wide {args.statistic} speed by hour. Each
          interval counts once regardless of how many segments it crossed, so
          the sample sizes below are independent measurements.</p>
          {img(figs["hourly"], "Speed by hour of day")}
          {_table(["Hour", f"Speed ({unit})", "n"], rows,
                  "Network-wide speed by hour of day")}"""))

    # ---- 6. weekday vs weekend ------------------------------------------ #
    if figs.get("daytype"):
        ov = R["daytype"]["overall"]
        dt_tiles = "".join(
            tile(_fmt(ov.get(f"{lb}_speed")), f"{lb} ({unit})",
                 f"n = {R['daytype']['groups'][lb]['n']:,}")
            for lb in R["daytype"]["groups"])
        delta = ov.get("delta_pct")
        if delta is not None and np.isfinite(delta):
            dt_tiles += tile(f"{delta:+.1f}%", "Weekend vs weekday",
                             "positive = weekends run faster")
        S.append(slide(f"""
          <p class="eyebrow">05 — Day type</p>
          <h2>Weekday and weekend traffic are different populations</h2>
          <p class="sub">Pooling them into one hour-of-day average hides both.
          Split here so a Tuesday 09:00 is never averaged with a Saturday 09:00.</p>
          {img(figs["daytype"], "Weekday versus weekend speed")}
          <div class="tiles">{dt_tiles}</div>"""))

    # ---- 7. congestion --------------------------------------------------- #
    if figs.get("congestion_map") or figs.get("worst"):
        c = R["congestion"]["summary"]
        e = R["congestion"]["edges"]
        # Same one-row-per-physical-road collapse the chart uses, so the table
        # and the figure cannot tell different stories.
        rated = _by_road(net, e[e["ratio"].notna()]).sort_values("ratio")
        rows = []
        for r in rated.head(15).itertuples():
            d = net.edge_data(int(r.edge_id))
            rows.append((_esc(d.get("name") or d.get("highway") or r.edge_id),
                         _fmt(r.observed_speed), _fmt(r.speed_limit),
                         f"{r.ratio:.2f}", f"{int(r.n):,}"))
        maps = "".join(img(figs[k], k) for k in ("congestion_map", "worst")
                       if figs.get(k))
        S.append(slide(f"""
          <p class="eyebrow">06 — Congestion vs posted limit</p>
          <h2>How the network performs against its own speed limits</h2>
          <p class="sub">A ratio of 1.00 means traffic moved at the posted
          limit; 0.45 means it crawled at 45% of it. This is the comparison raw
          speeds cannot make — a 30&nbsp;mph arterial and a 55&nbsp;mph highway
          both at 25&nbsp;mph are in completely different states.</p>
          {maps}
          <div class="tiles">
            {tile(f"{c['median_ratio']:.2f}", "Median ratio", "network-wide")}
            {tile(f"{c['n_edges_rated']:,}", "Segments rated", "observed and posted")}
            {tile(f"{int((rated['ratio'] < 0.5).sum()):,}", "Below 50% of limit")}
            {tile(f"{int((rated['ratio'] > 1.0).sum()):,}", "Above the limit")}
          </div>
          {_table(["Segment", f"Observed ({unit})", f"Limit ({unit})", "Ratio", "n"],
                  rows, "Most congested segments")}"""))

    # ---- 8. chokepoints -------------------------------------------------- #
    if figs.get("chokepoints"):
        top = sorted(R["bc"].items(), key=lambda kv: -kv[1])[:12]
        rows = []
        for eid, v in top:
            d = net.edge_data(int(eid))
            rows.append((_esc(d.get("name") or d.get("highway") or eid),
                         f"{v:.4f}", _fmt(d.get("length_m"), 0)))
        S.append(slide(f"""
          <p class="eyebrow">07 — Network structure</p>
          <h2>Chokepoints</h2>
          <p class="sub">Edge betweenness weighted by <i>measured travel time</i>:
          the share of fastest routes that must cross each segment. This finds
          roads that are structurally load-bearing, which is not the same as
          roads that are merely central on a map.</p>
          {img(figs["chokepoints"], "Chokepoint map")}
          {_table(["Segment", "Betweenness", "Length (m)"], rows,
                  "Highest-betweenness segments")}"""))

    # ---- 9. structure stats ---------------------------------------------- #
    dens = ""
    if stats.get("edge_density_km2") is not None:
        dens = tile(_fmt(stats["edge_density_km2"], 1), "Edges / km²")
    scc = conn.get("largest_scc_frac")
    S.append(slide(f"""
      <p class="eyebrow">08 — Network structure</p>
      <h2>Structural summary</h2>
      <p class="sub">Properties of the road network itself, independent of the
      GPS survey.</p>
      <div class="tiles">
        {tile(f"{stats['n_intersections']:,}", "Intersections")}
        {tile(f"{stats.get('n_dead_ends', 0):,}", "Dead ends")}
        {tile(_fmt(stats.get('streets_per_node_avg'), 2), "Streets per node")}
        {tile(_fmt(stats.get('circuity_avg'), 3), "Circuity",
              "1.00 = perfectly straight")}
        {dens}
      </div>
      <div class="tiles">
        {tile("Yes" if conn.get("is_strongly_connected") else "No",
              "Fully drivable", "every segment reaches every other")}
        {tile(f"{100.0 * scc:.1f}%" if scc is not None else "--",
              "Largest connected part")}
        {tile("Yes" if conn.get("is_weakly_connected") else "No",
              "Geometrically joined",
              "if yes but not fully drivable, one-way rules are the cause")}
      </div>"""))

    # ---- 10. method ------------------------------------------------------ #
    note_html = ""
    if notes:
        note_html = ('<h2 style="margin-top:26px">Data notes</h2><ul class="notes">'
                     + "".join(f"<li>{_esc(n)}</li>" for n in notes) + "</ul>")
    S.append(slide(f"""
      <p class="eyebrow">09 — Method</p>
      <h2>How these numbers were produced</h2>
      <p><b>Matching.</b> Each fix was assigned to a road with a hidden Markov
      model decoded by Viterbi (Newson &amp; Krumm, 2009), which uses the whole
      trajectory rather than snapping each point independently — near
      intersections, independent snapping makes correlated errors that bias
      speed.</p>
      <p><b>Speed.</b> Derived from on-road displacement between consecutive
      fixes of one mover, divided by elapsed time, with per-point GPS accuracy
      propagated into an explicit uncertainty for every interval. Intervals
      whose displacement is not clearly above the noise floor are flagged.</p>
      <p><b>Aggregation.</b> {args.statistic.capitalize()} statistics; each
      interval counts once network-wide however many segments it crossed.
      Peak and off-peak windows were selected from the data as the slowest and
      fastest contiguous blocks.</p>
      <p><b>Limitations.</b> Speeds describe <i>the surveyed vehicles</i>, not
      all traffic, and only where and when they drove — see the coverage map.
      A posted limit is a legal maximum, not a free-flow speed; roads run below
      their limit for reasons other than congestion, so the congestion ratio is
      a screening indicator. Comparing a road against itself across time blocks
      is sounder than comparing different roads to each other.</p>
      {note_html}"""))

    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{_esc(args.title)}</title><style>{CSS}</style></head><body>"
            + "".join(S)
            + f'<div class="foot">Generated with roadtraffic v{rt.__version__}. '
              f'Speeds reconstructed from on-road displacement after HMM map '
              f'matching; see the method section for assumptions and limits.</div>'
            + "</body></html>")


# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Build a customer-ready congestion report from GPS + a road network.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            example:
              python scripts/customer_report.py --network roads.geojson \\
                --points probes.csv --accuracy-col accuracy \\
                --tz America/New_York --title "Route 1 Study" \\
                --customer "Example DOT" --out report.html
            """))
    p.add_argument("--network", required=True, help="road network GeoJSON")
    p.add_argument("--points", required=True, help="GPS points CSV/TSV/GeoJSON")
    p.add_argument("--out", default="report.html", help="output HTML file")
    p.add_argument("--title", default="Road Network Trafficability Study")
    p.add_argument("--customer", default="")
    p.add_argument("--tz", default=None,
                   help="IANA timezone of the study area, e.g. America/New_York")
    p.add_argument("--accuracy-col", default="accuracy",
                   help="per-point horizontal accuracy in METRES (default: accuracy)")
    p.add_argument("--id-col", default=None, help="mover/trajectory id column")
    p.add_argument("--time-col", default=None)
    p.add_argument("--lon-col", default=None)
    p.add_argument("--lat-col", default=None)
    p.add_argument("--unit", default="mph", choices=["mph", "kph", "mps"])
    p.add_argument("--statistic", default="median", choices=["median", "mean"])
    p.add_argument("--max-dist", type=float, default=50.0,
                   help="max snap distance, metres (default 50)")
    p.add_argument("--default-sigma", type=float, default=15.0,
                   help="fallback GPS sigma when accuracy is missing (default 15 m)")
    p.add_argument("--min-baseline", type=float, default=None,
                   help="merge hops until this on-road distance; raises SNR on "
                        "dense/noisy fixes")
    p.add_argument("--max-speed", type=float, default=80.0,
                   help="physical upper bound for cleaning, in --unit")
    p.add_argument("--min-samples", type=int, default=1,
                   help="suppress time bins with fewer observations than this "
                        "(default 1; raise to stop thin hours reading as spikes)")
    p.add_argument("--n-peak", type=int, default=3)
    p.add_argument("--n-offpeak", type=int, default=3)
    p.add_argument("--area-km2", type=float, default=None,
                   help="study-area size, enables per-km2 densities")
    p.add_argument("--require-quality", action="store_true",
                   help="drop intervals that failed the quality screen "
                        "(recommended for customer deliverables)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    notes: list = []
    R = run_pipeline(args, notes)
    plt = _mpl()

    print("      rendering figures", file=sys.stderr)
    figs = {}
    net = R["net"]
    f = fig_coverage(plt, net, R["seg"])
    figs["coverage"] = _fig_to_b64(f, plt)

    f, lim = fig_speed_map(plt, net, "overall", args.unit)
    if f:
        figs["speed_overall"] = _fig_to_b64(f, plt)
    # Peak and off-peak share the overall scale so the two maps compare directly.
    for period in ("peak", "offpeak"):
        f, _ = fig_speed_map(plt, net, period, args.unit, vlim=lim)
        if f:
            figs[f"speed_{period}"] = _fig_to_b64(f, plt)

    if len(R["hourly"]):
        figs["hourly"] = _fig_to_b64(
            fig_hourly(plt, R["hourly"], args.unit, R["peaks"]), plt)
    f = fig_daytype(plt, R["daytype"], args.unit)
    if f:
        figs["daytype"] = _fig_to_b64(f, plt)
    if R["congestion"]:
        f = fig_congestion_map(plt, net, R["congestion"])
        if f:
            figs["congestion_map"] = _fig_to_b64(f, plt)
        f = fig_worst_corridors(plt, net, R["congestion"])
        if f:
            figs["worst"] = _fig_to_b64(f, plt)
    if R["bc"]:
        f = fig_chokepoints(plt, net, R["bc"])
        if f:
            figs["chokepoints"] = _fig_to_b64(f, plt)

    html = build_html(args, R, figs, notes)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(html)
    size_mb = len(html.encode("utf-8")) / 1e6
    print(f"\nWrote {args.out}  ({size_mb:.1f} MB, {len(figs)} figures)",
          file=sys.stderr)
    if notes:
        print(f"{len(notes)} data note(s) included in the report.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
