"""Reproduce every figure and number in docs/methodology.md.

Synthetic ground-truth experiments defending the package's method choices:

  A. Map-matching accuracy (nearest-edge vs HMM/Viterbi) under a GPS-noise
     sweep, against known true paths.
  B. Speed-recovery accuracy of the on-road interval estimator, against the
     analytic error model sigma_v = sqrt(2)*sigma / dt, with and without the
     quality gate and baseline merging.
  C. An end-to-end day study: hour-varying true speeds -> matching -> speed
     derivation -> contiguous peak/off-peak window detection -> per-segment
     regime speeds.

Everything is seeded; run time is a couple of minutes:

    python scripts/paper_experiments.py

Figures land in docs/figures/, and the headline numbers in
docs/figures/experiment_results.json.
"""
from __future__ import annotations

import json
import os
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from shapely.geometry import Point

import redlight as rl

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "..", "docs", "figures")
GRID_N = 12
GRID_SPACING = 0.001  # deg, ~111 m at the equator
V_TRUE = 10.0         # m/s ground-truth speed for experiments A/B
SEED = 42

RESULTS: dict = {}


# ------------------------------------------------------------------ scaffolding
def build_grid_network() -> rl.Network:
    feats = []
    for i in range(GRID_N):
        for j in range(GRID_N):
            lon, lat = j * GRID_SPACING, i * GRID_SPACING
            if j < GRID_N - 1:
                feats.append({"type": "Feature",
                              "properties": {"highway": "residential"},
                              "geometry": {"type": "LineString",
                                           "coordinates": [[lon, lat],
                                                           [lon + GRID_SPACING, lat]]}})
            if i < GRID_N - 1:
                feats.append({"type": "Feature",
                              "properties": {"highway": "residential"},
                              "geometry": {"type": "LineString",
                                           "coordinates": [[lon, lat],
                                                           [lon, lat + GRID_SPACING]]}})
    path = os.path.join(tempfile.mkdtemp(prefix="rt_paper_"), "grid.json")
    with open(path, "w") as fh:
        json.dump({"type": "FeatureCollection", "features": feats}, fh)
    return rl.Network.from_geojson(path)


def random_walk_edges(net: rl.Network, rng: np.random.Generator,
                      n_edges: int, p_straight: float = 0.6) -> list[int]:
    """A ground-truth path: a random walk that prefers going straight.

    Vehicles mostly continue straight through intersections; turning at every
    block (a uniform walk) would be an unrealistic worst case for any matcher
    that uses path consistency.
    """
    nodes = list(net.graph.nodes())
    u = nodes[int(rng.integers(len(nodes)))]
    prev = None
    eids: list[int] = []
    for _ in range(n_edges):
        succs = [v for v in net.graph.successors(u) if v != prev]
        if not succs:
            succs = list(net.graph.successors(u))
        v = None
        if prev is not None and rng.random() < p_straight:
            heading = (u[0] - prev[0], u[1] - prev[1])
            for cand in succs:
                if (round(cand[0] - u[0], 9), round(cand[1] - u[1], 9)) == \
                        (round(heading[0], 9), round(heading[1], 9)):
                    v = cand
                    break
        if v is None:
            v = succs[int(rng.integers(len(succs)))]
        data = net.graph[u][v]
        eids.append(int(data[next(iter(data))]["edge_id"]))
        prev, u = u, v
    return eids


def sample_trajectory(net: rl.Network, eids: list[int], *, v_mps: float,
                      dt_s: float, sigma_m: float, rng: np.random.Generator,
                      t0: pd.Timestamp):
    """Emit noisy GPS fixes along a true path at constant speed.

    Returns (lon, lat, time, true_edge) arrays. Noise is isotropic Gaussian in
    the metric plane -- the standard GPS horizontal-error model.
    """
    geoms = [net.edge_geometry(e) for e in eids]
    cum = np.concatenate([[0.0], np.cumsum([net.edge_length(e) for e in eids])])
    n = int(cum[-1] // (v_mps * dt_s))
    lon = np.empty(n)
    lat = np.empty(n)
    true_edge = np.empty(n, dtype=np.int64)
    times = []
    for i in range(n):
        s = i * v_mps * dt_s
        j = min(int(np.searchsorted(cum, s, side="right")) - 1, len(eids) - 1)
        p = geoms[j].interpolate(s - cum[j])
        x = p.x + rng.normal(0.0, sigma_m)
        y = p.y + rng.normal(0.0, sigma_m)
        lo, la = net._transformer_inv.transform(x, y)
        lon[i], lat[i] = lo, la
        true_edge[i] = eids[j]
        times.append(t0 + pd.Timedelta(seconds=i * dt_s))
    return lon, lat, pd.to_datetime(times), true_edge


def make_pointset(net, rng, *, n_traj, n_edges, sigma_m, dt_s,
                  v_mps=V_TRUE, hour_speeds=None):
    """PointSet + ground-truth edge per fix for n_traj random-walk movers.

    If ``hour_speeds`` is given (len-24 array), each trajectory starts at a
    random hour and moves at that hour's speed instead of ``v_mps``.
    """
    frames = []
    truths = []
    pid = 0
    for t in range(n_traj):
        if hour_speeds is not None:
            hour = int(rng.integers(0, 24))
            v = float(hour_speeds[hour])
            t0 = (pd.Timestamp("2026-06-01") + pd.Timedelta(hours=hour)
                  + pd.Timedelta(minutes=int(rng.integers(0, 40))))
        else:
            v = v_mps
            t0 = pd.Timestamp("2026-06-01 08:00:00") + pd.Timedelta(minutes=2 * t)
        lon, lat, times, true_edge = sample_trajectory(
            net, random_walk_edges(net, rng, n_edges),
            v_mps=v, dt_s=dt_s, sigma_m=sigma_m, rng=rng, t0=t0)
        n = len(lon)
        frames.append(pd.DataFrame({
            "point_id": np.arange(pid, pid + n, dtype=np.int64),
            "traj_id": f"veh{t:04d}", "lon": lon, "lat": lat, "time": times,
        }))
        truths.append(true_edge)
        pid += n
    df = pd.concat(frames, ignore_index=True)
    return rl.PointSet(df, has_traj=True), np.concatenate(truths)


def road_level_accuracy(net, matched: pd.DataFrame, truth: np.ndarray):
    """Fraction of fixes matched to the true physical road (either direction)."""
    m = matched.sort_values("point_id")["edge_id"].to_numpy()
    ok = np.zeros(len(m), dtype=bool)
    road_cache: dict = {}
    for i, (got, want) in enumerate(zip(m, truth)):
        if got == -1:
            continue
        if want not in road_cache:
            road_cache[want] = set(net.road_edge_ids(int(want)))
        ok[i] = int(got) in road_cache[want]
    return float(ok.mean()), float((m == -1).mean())


# ------------------------------------------------------------- experiment A
def experiment_a(net):
    print("Experiment A: matching accuracy vs GPS noise")
    sigmas = [5.0, 15.0, 30.0, 50.0]
    res = {"sigma_m": sigmas, "nearest_acc": [], "hmm_acc": [],
           "nearest_unmatched": [], "hmm_unmatched": []}
    matched_store = {}
    for sig in sigmas:
        rng = np.random.default_rng(SEED)
        pts, truth = make_pointset(net, rng, n_traj=40, n_edges=25,
                                   sigma_m=sig, dt_s=5.0)
        max_dist = max(60.0, 3.0 * sig)
        # k counts SEGMENTS in the KDTree shortlist; at a 150 m search radius
        # k=16 is needed to keep all plausible roads in the candidate set
        near = rl.NearestMatcher(net, max_dist=max_dist, k=16).match(pts)
        hmm = rl.HMMMatcher(net, sigma_z=sig, max_dist=max_dist, k=16).match(pts)
        na, nu = road_level_accuracy(net, near, truth)
        ha, hu = road_level_accuracy(net, hmm, truth)
        res["nearest_acc"].append(na)
        res["hmm_acc"].append(ha)
        res["nearest_unmatched"].append(nu)
        res["hmm_unmatched"].append(hu)
        matched_store[sig] = (pts, hmm, near)
        print(f"  sigma={sig:>4.0f} m  nearest={na:6.1%}  hmm={ha:6.1%}  n={len(truth)}")
    RESULTS["experiment_a"] = res

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(res["sigma_m"], np.array(res["nearest_acc"]) * 100, "o-",
            color="#d95f02", label="Nearest-edge matcher")
    ax.plot(res["sigma_m"], np.array(res["hmm_acc"]) * 100, "s-",
            color="#1b9e77", label="HMM / Viterbi matcher")
    ax.set_xlabel("GPS noise σ (metres)")
    ax.set_ylabel("fixes matched to the true road (%)")
    ax.set_title("Map-matching accuracy vs GPS noise\n"
                 f"({GRID_N}×{GRID_N} grid, ~111 m blocks, ground-truth random walks)")
    ax.set_ylim(0, 102)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig3_matching_accuracy.png"), dpi=150)
    plt.close(fig)
    return matched_store


# ------------------------------------------------------------- experiment B
def experiment_b(net, matched_store):
    print("Experiment B: speed recovery vs the analytic error model")
    rows = []
    for sig, (pts, hmm, near) in matched_store.items():
        out = rl.derive_speeds(net, hmm, pts, default_pos_sigma_m=sig)
        iv = out["intervals"]
        # downstream effect of matcher choice: same points, nearest matches
        ivn = rl.derive_speeds(net, near, pts,
                               default_pos_sigma_m=sig)["intervals"]
        rel_n = (ivn["speed_mps"] - V_TRUE) / V_TRUE
        q = iv[iv["quality"]]
        rel_all = (iv["speed_mps"] - V_TRUE) / V_TRUE
        rel_q = (q["speed_mps"] - V_TRUE) / V_TRUE
        # merged-baseline variant for the same data
        out_m = rl.derive_speeds(net, hmm, pts, default_pos_sigma_m=sig,
                                 min_baseline_m=3 * np.sqrt(2) * sig)
        ivm = out_m["intervals"]
        qm = ivm[ivm["quality"]]
        rel_m = (qm["speed_mps"] - V_TRUE) / V_TRUE
        rows.append({
            "sigma_m": sig, "dt_s": 5.0,
            "theory_rel_std": np.sqrt(2) * sig / (5.0 * V_TRUE),
            "raw_rel_std": float(rel_all.std()),
            "raw_bias": float(rel_all.mean()),
            "quality_frac": float(iv["quality"].mean()),
            "quality_rel_std": float(rel_q.std()) if len(q) else np.nan,
            "quality_bias": float(rel_q.mean()) if len(q) else np.nan,
            "merged_rel_std": float(rel_m.std()) if len(qm) else np.nan,
            "merged_bias": float(rel_m.mean()) if len(qm) else np.nan,
            "merged_quality_frac": float(ivm["quality"].mean()),
            "nearest_raw_bias": float(rel_n.mean()),
            "nearest_raw_std": float(rel_n.std()),
        })
        print(f"  sigma={sig:>4.0f} m: theory σᵥ/v={rows[-1]['theory_rel_std']:.2f} "
              f"measured={rows[-1]['raw_rel_std']:.2f} "
              f"quality kept={rows[-1]['quality_frac']:.0%} "
              f"merged σᵥ/v={rows[-1]['merged_rel_std']:.2f} | "
              f"speed bias hmm={rows[-1]['raw_bias']:+.2f} "
              f"nearest={rows[-1]['nearest_raw_bias']:+.2f}")

    # dt sweep at sigma = 30 m (raw error vs theory: no quality selection)
    dt_rows = []
    for dt in (5.0, 15.0, 30.0):
        rng = np.random.default_rng(SEED + 1)
        pts, _tr = make_pointset(net, rng, n_traj=40, n_edges=25,
                                 sigma_m=30.0, dt_s=dt)
        hmm = rl.HMMMatcher(net, sigma_z=30.0, max_dist=90.0, k=16).match(pts)
        iv = rl.derive_speeds(net, hmm, pts,
                              default_pos_sigma_m=30.0)["intervals"]
        rel = (iv["speed_mps"] - V_TRUE) / V_TRUE
        dt_rows.append({"dt_s": dt,
                        "theory_rel_std": np.sqrt(2) * 30.0 / (dt * V_TRUE),
                        "measured_rel_std": float(rel.std()),
                        "measured_bias": float(rel.mean()),
                        "quality_frac": float(iv["quality"].mean())})
        print(f"  dt={dt:>4.0f} s: theory={dt_rows[-1]['theory_rel_std']:.2f} "
              f"measured={dt_rows[-1]['measured_rel_std']:.2f} "
              f"bias={dt_rows[-1]['measured_bias']:+.2f}")
    RESULTS["experiment_b"] = {"sigma_sweep": rows, "dt_sweep": dt_rows}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    sig_arr = [r["sigma_m"] for r in rows]
    axes[0].plot(sig_arr, [r["theory_rel_std"] for r in rows], "k--",
                 label=r"theory: $\sqrt{2}\,\sigma/(\Delta t\, v)$")
    axes[0].plot(sig_arr, [r["raw_rel_std"] for r in rows], "o-",
                 color="#7570b3", label="measured (all intervals)")
    axes[0].plot(sig_arr, [r["merged_rel_std"] for r in rows], "s-",
                 color="#1b9e77",
                 label=r"measured (baseline-merged, $\geq 3\sqrt{2}\sigma$)")
    axes[0].set_xlabel("GPS noise σ (metres)")
    axes[0].set_ylabel("relative speed error (std)")
    axes[0].set_title(f"Spread vs noise (Δt = 5 s, v = {V_TRUE:.0f} m/s)")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].plot(sig_arr, [r["raw_bias"] for r in rows], "o-",
                 color="#7570b3", label="all intervals")
    axes[1].plot(sig_arr, [r["merged_bias"] for r in rows], "s-",
                 color="#1b9e77", label="baseline-merged")
    axes[1].plot(sig_arr, [r["quality_bias"] for r in rows], "^-",
                 color="#d62728",
                 label="quality-filtered only\n(selection bias — see text)")
    axes[1].set_xlabel("GPS noise σ (metres)")
    axes[1].set_ylabel("relative speed bias (mean error)")
    axes[1].set_title("Bias vs noise (Δt = 5 s)")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    dts = [r["dt_s"] for r in dt_rows]
    axes[2].plot(dts, [r["theory_rel_std"] for r in dt_rows], "k--",
                 label="theory")
    axes[2].plot(dts, [r["measured_rel_std"] for r in dt_rows], "o-",
                 color="#7570b3", label="measured (all intervals)")
    axes[2].set_xlabel("fix spacing Δt (seconds)")
    axes[2].set_ylabel("relative speed error (std)")
    axes[2].set_title("Spread vs fix spacing (σ = 30 m)")
    axes[2].grid(alpha=0.3)
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig5_speed_error.png"), dpi=150)
    plt.close(fig)


# ------------------------------------------------------------- experiment C
def experiment_c(net):
    print("Experiment C: end-to-end day study")
    hour_speeds = np.full(24, 9.0)
    hour_speeds[[7, 8, 9, 16, 17, 18]] = 4.0     # true congestion
    hour_speeds[[22, 23, 0, 1, 2, 3, 4, 5]] = 15.0  # true free flow
    rng = np.random.default_rng(SEED + 2)
    pts, _tr = make_pointset(net, rng, n_traj=240, n_edges=20, sigma_m=15.0,
                             dt_s=10.0, hour_speeds=hour_speeds)
    hmm = rl.HMMMatcher(net, sigma_z=15.0, max_dist=60.0).match(pts)
    eo = rl.derive_speeds(net, hmm, pts, default_pos_sigma_m=15.0)["edge_observations"]
    eo = eo[eo["quality"]]
    cls = rl.classify_hours(eo, n_peak=3, n_offpeak=3)
    info = rl.assign_segment_speeds(net, eo, n_peak=3, n_offpeak=3,
                                    statistic="median")
    agg = rl.aggregate_speeds(eo, statistic="median", output_unit="mps")
    RESULTS["experiment_c"] = {
        "true_peak_hours": [7, 8, 9, 16, 17, 18],
        "detected_peak_hours": cls["peak_hours"],
        "detected_offpeak_hours": cls["offpeak_hours"],
        "peak_speed_mps": cls["peak_speed_mps"],
        "offpeak_speed_mps": cls["offpeak_speed_mps"],
        "coverage": info["coverage"],
        "n_points": len(pts),
    }
    print(f"  detected peak={cls['peak_hours']} offpeak={cls['offpeak_hours']}"
          f"  (true slow: 7-9 & 16-18, true fast: 22-05)")

    # hourly profile + windows
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(agg["block_start_hour"], agg["median_speed"], "o-",
            color="#33517e", label="measured hourly median")
    ax.step(range(24), hour_speeds, where="post", color="0.6", lw=1,
            label="true speed profile")
    for h in cls["peak_hours"]:
        ax.axvspan(h, h + 1, color="#d95f02", alpha=0.18,
                   label="detected peak window" if h == cls["peak_hours"][0] else None)
    for h in cls["offpeak_hours"]:
        ax.axvspan(h, h + 1, color="#1b9e77", alpha=0.18,
                   label="detected off-peak window" if h == cls["offpeak_hours"][0] else None)
    ax.set_xlabel("hour of day (local)")
    ax.set_ylabel("speed (m/s)")
    ax.set_title("Network-wide hourly speed and detected contiguous windows "
                 "(n_peak=3, n_offpeak=3)")
    ax.set_xticks(range(0, 25, 2))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig6_peak_windows.png"), dpi=150)
    plt.close(fig)

    # per-segment regime maps
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharex=True, sharey=True)
    for ax, regime, title in ((axes[0], "peak", "Peak window (07:00–10:00)"),
                              (axes[1], "offpeak", "Off-peak window")):
        obs_lines, obs_speeds, nodata = [], [], []
        seen = set()
        for _u, _v, d in net.graph.edges(data=True):
            rid = min(net.road_edge_ids(int(d["edge_id"])))
            if rid in seen:
                continue
            seen.add(rid)
            coords = net.edge_coords_lonlat(rid)
            spd = d.get(f"obs_speed_mps_{regime}")
            if spd:
                obs_lines.append(coords)
                obs_speeds.append(spd)
            else:
                nodata.append(coords)
        if nodata:
            ax.add_collection(LineCollection(nodata, colors="#cccccc", linewidths=1.2))
        lc = LineCollection(obs_lines, cmap="RdYlGn", linewidths=2.4)
        lc.set_array(np.array(obs_speeds))
        lc.set_clim(3, 16)
        ax.add_collection(lc)
        ax.autoscale()
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    cb = fig.colorbar(lc, ax=axes, shrink=0.75, pad=0.02)
    cb.set_label("median segment speed (m/s)")
    fig.suptitle("Per-segment trafficability by regime (grey = no observations)")
    fig.savefig(os.path.join(FIG_DIR, "fig7_segment_map.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------- schematics
def figure_matching_problem(net):
    """One noisy trajectory: nearest-edge errors vs HMM consistency.

    Illustrative example: trajectories are scanned for a case where the
    aggregate behaviour (Fig. 3) is visible in a single picture -- nearest
    snapping fixes to cross/parallel streets that the HMM keeps on-path.
    The caption in the paper says so explicitly.
    """
    best = None
    for seed in range(60):
        rng = np.random.default_rng(seed)
        p, tr = make_pointset(net, rng, n_traj=1, n_edges=10, sigma_m=15.0,
                              dt_s=5.0)
        nr = rl.NearestMatcher(net, max_dist=60.0, k=16).match(p)
        hm = rl.HMMMatcher(net, sigma_z=15.0, max_dist=60.0, k=16).match(p)
        na, _ = road_level_accuracy(net, nr, tr)
        ha, _ = road_level_accuracy(net, hm, tr)
        score = ha - na
        if best is None or score > best[0]:
            best = (score, p, tr, nr, hm)
    _, pts, truth, near, hmm = best
    px, py = net.project_points(pts.df["lon"].values, pts.df["lat"].values)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    for ax, matched, name in ((axes[0], near, "Nearest-edge matching"),
                              (axes[1], hmm, "HMM / Viterbi matching")):
        for eid in net.edge_ids:  # light street grid
            xy = np.asarray(net.edge_geometry(int(eid)).coords)
            ax.plot(xy[:, 0], xy[:, 1], color="0.85", lw=1, zorder=1)
        drawn = set()
        for want in truth:  # true path
            rid = min(net.road_edge_ids(int(want)))
            if rid in drawn:
                continue
            drawn.add(rid)
            xy = np.asarray(net.edge_geometry(rid).coords)
            ax.plot(xy[:, 0], xy[:, 1], color="#33517e", lw=3, zorder=2,
                    label="true path" if len(drawn) == 1 else None)
        m = matched.sort_values("point_id")["edge_id"].to_numpy()
        for i in range(len(m)):
            correct = (m[i] != -1 and
                       int(m[i]) in net.road_edge_ids(int(truth[i])))
            ax.scatter(px[i], py[i], s=26, zorder=4,
                       color="#1b9e77" if correct else "#d62728",
                       label=None)
            if m[i] != -1:  # snap line to the assigned road
                g = net.edge_geometry(int(m[i]))
                q = g.interpolate(g.project(Point(px[i], py[i])))
                ax.plot([px[i], q.x], [py[i], q.y], color="0.55", lw=0.7,
                        zorder=3)
        n_ok = sum(m[i] != -1 and int(m[i]) in net.road_edge_ids(int(truth[i]))
                   for i in range(len(m)))
        ax.set_title(f"{name}\n{n_ok}/{len(m)} fixes on the true road",
                     fontsize=11)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        lo_x, hi_x = px.min() - 150, px.max() + 150
        lo_y, hi_y = py.min() - 150, py.max() + 150
        ax.set_xlim(lo_x, hi_x)
        ax.set_ylim(lo_y, hi_y)
    handles = [plt.Line2D([], [], color="#33517e", lw=3, label="true path"),
               plt.Line2D([], [], marker="o", ls="", color="#1b9e77",
                          label="fix matched to true road"),
               plt.Line2D([], [], marker="o", ls="", color="#d62728",
                          label="fix matched to wrong road / unmatched"),
               plt.Line2D([], [], color="0.55", lw=0.7, label="snap to assigned road")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9,
               frameon=False)
    fig.suptitle("The matching problem at σ = 15 m GPS noise (~111 m blocks) — "
                 "illustrative trajectory; aggregate results in Fig. 3")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(os.path.join(FIG_DIR, "fig1_matching_problem.png"), dpi=150)
    plt.close(fig)


def figure_hmm_mechanics():
    """Schematic: emission and transition probabilities."""
    fig, ax = plt.subplots(figsize=(9, 5.2))
    # two roads: horizontal A, vertical B crossing at (0, 0)
    ax.plot([-260, 260], [0, 0], color="0.4", lw=5, solid_capstyle="round")
    ax.plot([0, 0], [-160, 220], color="0.4", lw=5, solid_capstyle="round")
    ax.annotate("road A", (215, 12), fontsize=11, color="0.25")
    ax.annotate("road B", (12, 195), fontsize=11, color="0.25")
    # fixes
    z1, z2 = (-170.0, 40.0), (-60.0, 90.0)
    ax.scatter([z1[0]], [z1[1]], s=90, color="#33517e", zorder=5)
    ax.annotate("$z_t$", (z1[0] - 34, z1[1] + 6), fontsize=13, color="#33517e")
    ax.scatter([z2[0]], [z2[1]], s=90, color="#33517e", zorder=5)
    ax.annotate("$z_{t+1}$", (z2[0] - 14, z2[1] + 18), fontsize=13,
                color="#33517e")
    # emission: perpendicular snap distances to candidates
    ax.plot([z2[0], z2[0]], [z2[1], 0], ls="--", color="#1b9e77", lw=1.6)
    ax.annotate("$d_A$", (z2[0] + 7, 40), color="#1b9e77", fontsize=12)
    ax.plot([z2[0], 0], [z2[1], z2[1]], ls="--", color="#d95f02", lw=1.6)
    ax.annotate("$d_B$", (-36, z2[1] + 8), color="#d95f02", fontsize=12)
    ax.annotate(r"emission: $p(z\,|\,c)\ \propto\ \exp(-d^2/2\sigma_z^2)$"
                "\n(closer road = likelier candidate)",
                (-255, -105), fontsize=11)
    # transition: straight-line step vs on-road distance
    ax.annotate("", xy=z2, xytext=z1,
                arrowprops=dict(arrowstyle="->", color="#33517e", lw=1.6))
    ax.annotate("GPS step", (-160, 74), fontsize=10, color="#33517e",
                rotation=22)
    ax.plot([z1[0], z2[0]], [4, 4], color="#1b9e77", lw=3)
    ax.annotate(r"transition: $p(c\to c')\ \propto\ \exp(-|\Delta|/\beta)$,"
                r"  $\Delta = |$route dist $-$ GPS step$|$"
                "\n(a candidate pair whose on-road distance matches the\n"
                "GPS step is likelier: paths must be drivable, not teleport)",
                (-255, -160), fontsize=11)
    ax.annotate("on-road distance A→A", (-172, -22), fontsize=10, color="#1b9e77")
    ax.set_xlim(-270, 270)
    ax.set_ylim(-185, 235)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("HMM map matching: two probabilities decide each fix\n"
                 "(Viterbi then picks the jointly most probable road sequence)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig2_hmm_mechanics.png"), dpi=150)
    plt.close(fig)


def figure_speed_derivation():
    """Schematic: the three-piece on-road interval distance."""
    fig, ax = plt.subplots(figsize=(9, 4.6))
    # an L-shaped road: edge A east, edge B north
    ax.plot([-40, 300], [0, 0], color="0.4", lw=5, solid_capstyle="round")
    ax.plot([300, 300], [0, 240], color="0.4", lw=5, solid_capstyle="round")
    ax.annotate("edge A", (100, -26), fontsize=11, color="0.25")
    ax.annotate("edge B", (312, 110), fontsize=11, color="0.25")
    # fixes + snapped positions
    f1, s1 = (80.0, 42.0), (80.0, 0.0)
    f2, s2 = (262.0, 158.0), (300.0, 150.0)
    for (fx, fy), (sx, sy), lab in ((f1, s1, "fix $i$"), (f2, s2, "fix $i{+}1$")):
        ax.scatter([fx], [fy], s=90, color="#33517e", zorder=5)
        ax.plot([fx, sx], [fy, sy], ls="--", color="0.55", lw=1.2)
        ax.scatter([sx], [sy], s=55, color="#1b9e77", zorder=5, marker="s")
        ax.annotate(lab, (fx - 10, fy + 14), fontsize=12, color="#33517e")
    # three-piece distance
    ax.plot([80, 300], [6, 6], color="#d95f02", lw=3.5)
    ax.plot([294, 294], [6, 150], color="#d95f02", lw=3.5)
    ax.annotate("① rest of edge A after the snap", (110, 16), fontsize=10,
                color="#d95f02")
    ax.annotate("② along edge B\nup to the snap", (196, 92), fontsize=10,
                color="#d95f02", ha="center")
    ax.annotate(
        "on-road distance = ① + ②   (never the straight line between fixes)\n"
        r"speed over the interval $=\ $ on-road distance $/\ \Delta t$,"
        " attributed to every traversed edge",
        (-40, 216), fontsize=11)
    ax.annotate("snapped positions (arc-length on the matched edge)",
                (86, -52), fontsize=10, color="#1b9e77")
    ax.set_xlim(-55, 420)
    ax.set_ylim(-70, 260)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Interval speed from on-road displacement", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig4_speed_derivation.png"), dpi=150)
    plt.close(fig)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    net = build_grid_network()
    figure_hmm_mechanics()
    figure_speed_derivation()
    figure_matching_problem(net)
    matched_store = experiment_a(net)
    experiment_b(net, matched_store)
    experiment_c(net)
    out = os.path.join(FIG_DIR, "experiment_results.json")
    with open(out, "w") as fh:
        json.dump(RESULTS, fh, indent=2)
    print(f"figures -> {os.path.abspath(FIG_DIR)}")
    print(f"numbers -> {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
