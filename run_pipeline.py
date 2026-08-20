"""End-to-end redlight pipeline: GPS + a road network -> road speeds and analysis.

Point the CONFIG block below at your own data and run:

    python run_pipeline.py

Every stage is numbered and every tunable is explained where it is used, not
just where it is declared. Stages marked [OPTIONAL] can be deleted outright;
the ones that are not are load -> match -> derive -> aggregate.

Requires only redlight's core dependencies (numpy, pandas, scipy, shapely,
networkx). The map export in stage 12 additionally needs the `mapping` extra:

    pip install 'redlight[mapping]'

IMPORTANT: parallel matching (N_JOBS != 1) requires the
``if __name__ == "__main__":`` guard at the bottom of this file. Python's
'spawn' start method re-imports this module in every worker, so without the
guard each worker would re-run the pipeline. Do not remove it.
"""
from __future__ import annotations

import os

import redlight as rl

# =============================================================================
# CONFIG — everything you are likely to change lives here
# =============================================================================

# --- inputs ------------------------------------------------------------------
# NETWORK accepts .geojson/.json directly. For .shp/.gpkg use rl.Network.from_file
# instead of from_geojson below; that path needs the `shapefile` extra (pyogrio).
NETWORK = "roads.geojson"
POINTS = "gps.csv"
OUT_DIR = "results"

# --- your point file's column names ------------------------------------------
# redlight infers common spellings (lon/longitude/x, lat/latitude/y, time/
# timestamp/datetime) but naming them is faster and removes all doubt.
LON_COL = "longitude"
LAT_COL = "latitude"
TIME_COL = "timestamp"
# ID_COL is the mover/vehicle/device id. It is REQUIRED for HMM matching --
# without it there are no trajectories to decode, only a cloud of points, and
# HMMMatcher raises rather than guessing. Set it.
ID_COL = "device_id"
# Per-fix horizontal accuracy in metres, if your receiver reports it (often
# 'accuracy', 'hdop'-derived, or 'eph'). Feeds the speed error model directly:
# a fix that is known to be good gets a tighter uncertainty than one that is
# not. Set to None if you have no such column.
ACCURACY_COL = "accuracy_m"
# Speed column, if your file already carries one. Leave as None to derive speed
# from on-road displacement instead, which is what this pipeline does and is
# more trustworthy than a receiver's instantaneous reading.
SPEED_COL = None
SPEED_COL_UNIT = "mph"          # only used when SPEED_COL is not None

# --- timezone ----------------------------------------------------------------
# Hour-of-day and day-of-week statistics are read off the STORED clock. If your
# timestamps are UTC (or numeric epochs, which are UTC by definition) and your
# study area is not, set TZ to the study area's zone or every peak-hour result
# shifts by the offset. Set to None if the timestamps are already local.
TZ = "America/New_York"
# Set only if your time column holds numeric epochs rather than text:
# "s", "ms", "us" or "ns". Leave None for ISO-8601 strings.
TIMESTAMP_UNIT = None

# --- units -------------------------------------------------------------------
# redlight computes everything internally in m/s and converts only at the API
# boundary. This is the unit for reports and exports: "mph", "kph" or "mps".
OUTPUT_UNIT = "mph"

# --- matching ----------------------------------------------------------------
# MAX_DIST: candidate search radius in metres. A fix with no road inside this
# gets edge_id = -1 and is reported unmatched rather than forced onto a road.
# Set it to roughly 3-4x your typical GPS error. Too small loses real fixes;
# too large mainly costs time, since the emission term still prefers near roads.
MAX_DIST = 60.0
# SIGMA_Z: assumed GPS noise std-dev in metres -- the spread of the emission
# term. Consumer phone GPS in the open is 4-10 m; urban canyon 20-40 m. Set it
# to your data's REAL error, not the optimistic figure: too small makes the
# matcher over-trust position and ignore route consistency.
SIGMA_Z = 15.0
# BETA: how many metres of disagreement between the straight-line step and the
# on-road route distance costs one nat of likelihood. Smaller = stricter about
# implausible jumps; larger = more tolerant of detours and roundabouts.
BETA = 30.0
# K: maximum candidate roads considered per fix. This is the dominant cost
# knob -- k=16 roughly doubles match time versus k=8 on a dense grid. 8 is
# ample for most networks.
K = 8
# N_JOBS: worker processes for decoding independent trajectories.
#   1  = serial (default, and faster than you would expect)
#  -1  = all cores
# MEASURE BEFORE ENABLING. Worker start-up (a fresh interpreter plus the
# network pickled into each process) dominates on small and medium jobs. On a
# 2-core laptop serial was 1.9-6.1x FASTER below ~200k points and only lost
# above ~1M. Where your crossover falls depends on core count, trajectory
# length and how many of your fixes are near a road at all.
N_JOBS = -1

# --- speed derivation --------------------------------------------------------
# DEFAULT_POS_SIGMA_M: fallback per-fix position uncertainty in metres, used
# for any fix where ACCURACY_COL is missing or non-positive. The per-interval
# speed uncertainty is sqrt(sigma_i^2 + sigma_j^2) / dt, so this number
# directly sets how wide your error bars are.
DEFAULT_POS_SIGMA_M = 15.0
# MIN_BASELINE_M: merge consecutive hops until their summed on-road distance
# reaches this many metres before emitting a speed. Trades time and edge
# resolution for signal-to-noise, and it is the single most effective knob on
# noisy or densely-sampled data: a 10 m receiver sampling every 5 s at 10 m/s
# cannot see a road better than +-42% per interval, but merging to 150 m fixes
# that. Set to None to emit one interval per fix pair.
MIN_BASELINE_M = 150.0
# Quality-flag thresholds. These FLAG rows (quality=False), they never drop
# them -- filtering is your decision, made below via REQUIRE_QUALITY.
MIN_DT_S = 0.5                  # shorter gaps than this are implausible
MAX_DT_S = 120.0                # longer gaps mean the mover was not tracked
MAX_SNAP_DIST_M = 60.0          # a fix further than this from its road is suspect
MAX_SPEED_MPS = 60.0            # 60 m/s = 134 mph; above this is a data error
MIN_SNR = 3.0                   # displacement must beat the noise floor 3:1

# --- cleaning [OPTIONAL] -----------------------------------------------------
# Two different tools for two different jobs; do not confuse them.
#
# (a) DWELL REMOVAL runs on the MATCHED FIXES, before speeds are derived, and
#     needs those fixes to already carry a speed -- so it applies only when
#     your file has SPEED_COL, or you loaded with derive_speed=True. A mover
#     that stays inside DWELL_RADIUS_M for DWELL_MIN_S is parked or idling
#     (<= ~0.2 m/s sustained) and is dropped. Note this is NOT a slow-speed
#     filter: a vehicle creeping through congestion keeps making ground and is
#     kept, low speed and all. A minimum-speed floor would delete exactly the
#     congestion a trafficability study exists to measure.
DWELL_REMOVAL = True
DWELL_RADIUS_M = 25.0
DWELL_MIN_S = 120.0
#
# (b) OUTLIER SCREENING runs on the DERIVED OBSERVATIONS. MAD is the modified
#     Z-score on the median absolute deviation (cutoff 3.5): it has a 50%
#     breakdown point, so a handful of GPS jumps cannot inflate the screen the
#     way they inflate a standard deviation. per_edge=True judges each edge
#     against itself, so a motorway is not measured against a service road.
MAD_OUTLIERS = True
MAD_THRESHOLD = 3.5
MAX_PLAUSIBLE_SPEED = 80.0      # in OUTPUT_UNIT; a hard ceiling before MAD

# --- mode screening [OPTIONAL, stage 2] --------------------------------------
# Separates pedestrians from vehicles so foot traffic does not drag road speeds
# down. The verdict is per MOVER, not per observation: a vehicle stuck in
# gridlock stays a vehicle and its slow observations are kept, which is the
# whole point.
SCREEN_MODES = True
# None = infer the vehicle/pedestrian threshold from a density valley in the
# data. Returns None (and this pipeline then skips screening) unless there is a
# real walking population -- reliably found at ~20% of movers, not at 10%.
# Set a number (in OUTPUT_UNIT) if you know your fleet.
MODE_THRESHOLD = None
# Percentile of each mover's speeds used as its signature. 85 is deliberate:
# at the median, a congested vehicle looks exactly like a pedestrian.
MODE_PERCENTILE = 85.0
MODE_MIN_INTERVALS = 3          # movers with fewer intervals stay "unknown"

# --- aggregation -------------------------------------------------------------
# BLOCK_HOURS: width of each time bin. 1 = hour-of-day (24 bins); 6 gives
# 00-06/06-12/12-18/18-24. Must divide 24 evenly or the last block is narrower.
BLOCK_HOURS = 1
# "median" is robust to the outliers GPS data always has; "mean" is the one
# with usable confidence intervals. "both" emits both sets of columns.
STATISTIC = "median"
# Bins with fewer than this many observations are dropped as too thin to trust.
MIN_SAMPLES = 5
# Restrict to particular days: None (all), "weekday", "weekend", day names, or
# numbers with Monday=0. Weekday and weekend traffic are different populations
# and pooling them hides both.
DAYS = None
# REQUIRE_QUALITY: drop rows the quality flag marked bad. Leave False and read
# the `n` column, OR set True -- but if you set it, set MIN_BASELINE_M too.
# Filtering on quality at short intervals without merging first selects the
# upward noise fluctuations and biases speeds high by ~55%. Merge, then gate.
REQUIRE_QUALITY = False
# Weight the mean by inverse variance, so an interval measured over a long
# baseline counts for more than a noisy short one. Requires statistic="mean"
# or "both"; has no meaning for a median.
WEIGHT_BY_VARIANCE = False

# --- peak detection ----------------------------------------------------------
# N_PEAK / N_OFFPEAK: width in hours of the peak and off-peak windows.
# classify_hours returns a CONTIGUOUS window, so a day with two rush periods
# (a normal commuting day) yields one of them -- the slower. Raising the width
# does not capture both, because they are disjoint; peak_analysis below ranks
# individual hours instead and will name both.
N_PEAK = 3
N_OFFPEAK = 3

# --- network analysis --------------------------------------------------------
# AREA_METHOD is the study-area detector: "convex_hull" (default), "bbox", or
# None to disable and report no densities. The area comes from the network's
# own projected geometry, so it needs no reprojection and no extra dependency.
# CAVEAT: the hull OVER-states a non-convex extract -- the inside of an L, a
# ring road's doughnut hole, the wedges of a radial network -- so read the
# densities as a lower bound there, or set AREA_KM2 by hand.
AREA_METHOD = "convex_hull"
AREA_KM2 = None                 # a supplied area always wins over detection
# Chokepoint analysis [OPTIONAL, stage 10]. "length_m" ranks by distance;
# "time" ranks by travel time and needs speeds assigned first (stage 8 does
# that); None treats every edge as equal cost.
CENTRALITY_WEIGHT = "length_m"
CENTRALITY_SAMPLE_K = None      # e.g. 200 to approximate on a large network

# --- export ------------------------------------------------------------------
# DIRECTIONAL=False merges the two directed edges of a two-way road into one
# map feature. Note redlight does not currently produce direction-specific
# speeds -- an interval is attributed to both directions of every road it
# traversed -- so True gives you two features carrying the same number.
DIRECTIONAL = False
EXPORT_PERIOD = "overall"       # "overall", "peak" or "offpeak"
MAKE_MAP = True                 # needs the `mapping` extra (matplotlib)

# =============================================================================
# PIPELINE
# =============================================================================


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    # -- 1. load ---------------------------------------------------------------
    # The network is projected to a metric CRS on load (a UTM zone picked from
    # the data) so every length and snap distance is in real metres. If your
    # GeoJSON declares a `crs` member in a projected CRS it is honoured; a
    # non-WGS84 CRS with no native support needs the `crs` extra (pyproj).
    print("[1/12] loading")
    net = rl.Network.from_geojson(NETWORK)
    # For Shapefile / GeoPackage instead:
    #   net = rl.Network.from_file(NETWORK)        # needs the `shapefile` extra
    # For a live OSM extract instead:
    #   net = rl.Network.from_overpass((min_lon, min_lat, max_lon, max_lat))

    pts = rl.load_points(
        POINTS,
        lon_col=LON_COL, lat_col=LAT_COL, time_col=TIME_COL, id_col=ID_COL,
        speed_col=SPEED_COL, speed_unit=SPEED_COL_UNIT,
        timestamp_unit=TIMESTAMP_UNIT, tz=TZ,
        # keep_cols carries extra source columns through the pipeline; the
        # accuracy column must survive to reach derive_speeds below.
        keep_cols=[ACCURACY_COL] if ACCURACY_COL else None,
    )
    print(f"        {len(net.edge_ids):,} directed edges, {len(pts.df):,} fixes")

    # -- 2. mode screening [OPTIONAL] -----------------------------------------
    # This needs speeds, which need matching -- so it runs a cheap first pass
    # with NearestMatcher, screens, then matches the survivors properly. If you
    # know your feed is vehicles only, delete this whole stage.
    if SCREEN_MODES:
        print("[2/12] screening movers")
        rough = rl.NearestMatcher(net, max_dist=MAX_DIST).match(pts)
        rough_obs = rl.derive_speeds(
            net, rough, pts,
            pos_accuracy_col=ACCURACY_COL,
            default_pos_sigma_m=DEFAULT_POS_SIGMA_M,
            min_baseline_m=MIN_BASELINE_M,
        )["edge_observations"]

        threshold = MODE_THRESHOLD
        if threshold is None:
            feats = rl.mover_features(rough_obs, percentile=MODE_PERCENTILE,
                                      unit=OUTPUT_UNIT)
            threshold = rl.suggest_mode_threshold(
                feats[f"speed_p{int(MODE_PERCENTILE)}_{OUTPUT_UNIT}"],
                unit=OUTPUT_UNIT)

        if threshold is None:
            # No walking population found. That is the honest answer for a
            # single-mode feed -- do NOT substitute a default, since a silently
            # chosen threshold produces a study that looks correct.
            print("        no pedestrian population detected; keeping all movers")
        else:
            movers = rl.classify_movers(
                rough_obs, threshold=threshold, percentile=MODE_PERCENTILE,
                min_intervals=MODE_MIN_INTERVALS, unit=OUTPUT_UNIT)
            keep = set(movers.index[movers["mode"] == rl.MODE_VEHICLE])
            n_before = len(pts.df)
            pts.df = pts.df[pts.df["traj_id"].isin(keep)].reset_index(drop=True)
            print(f"        threshold {threshold:.1f} {OUTPUT_UNIT}; kept "
                  f"{len(keep):,} vehicles, dropped {n_before - len(pts.df):,} fixes")

    # -- 3. map matching -------------------------------------------------------
    # Hidden Markov Model with Viterbi decoding (Newson & Krumm 2009). The
    # emission term prefers roads the fix is near; the transition term prefers
    # sequences whose on-road distance matches the straight-line step. Deciding
    # the whole trajectory jointly is what separates this from NearestMatcher.
    print(f"[3/12] HMM matching (n_jobs={N_JOBS})")
    matcher = rl.HMMMatcher(net, sigma_z=SIGMA_Z, beta=BETA, max_dist=MAX_DIST,
                            k=K, n_jobs=N_JOBS)
    matched = matcher.match(pts)
    n_matched = int((matched["edge_id"] != -1).sum())
    print(f"        matched {n_matched:,}/{len(matched):,} "
          f"({100 * n_matched / max(len(matched), 1):.1f}%)")

    # -- 4. dwell removal [OPTIONAL] -------------------------------------------
    # Runs on the MATCHED FIXES, before speeds are derived, so parked periods
    # never become intervals in the first place. Needs a speed on the points:
    # with SPEED_COL=None and derive_speed unset there is none, so this is
    # skipped -- derive_speeds' own MIN_SNR and MAX_DT_S gates cover the same
    # ground for that case.
    if DWELL_REMOVAL:
        if "speed_mps" in matched.columns:
            n_before = len(matched)
            matched = rl.filter_trajectory_speed(
                matched, dwell_radius_m=DWELL_RADIUS_M, dwell_min_s=DWELL_MIN_S,
                max_speed=MAX_PLAUSIBLE_SPEED, unit=OUTPUT_UNIT)
            print(f"[4/12] dwell removal: {n_before:,} -> {len(matched):,} fixes")
        else:
            print("[4/12] dwell removal skipped (points carry no speed column; "
                  "set SPEED_COL, or load with derive_speed=True)")

    # -- 5. derive speeds ------------------------------------------------------
    # Speed comes from ON-ROAD displacement between consecutive fixes, measured
    # along the graph, never as the crow flies. Two frames come back:
    #   intervals         -- one row per fix pair; use for network-wide stats
    #   edge_observations -- one row per (interval, traversed edge); use for
    #                        per-edge stats. Deliberately duplicated, and
    #                        deduplicated automatically where it matters.
    print("[5/12] deriving speeds")
    speeds = rl.derive_speeds(
        net, matched, pts,
        pos_accuracy_col=ACCURACY_COL,
        default_pos_sigma_m=DEFAULT_POS_SIGMA_M,
        min_dt_s=MIN_DT_S, max_dt_s=MAX_DT_S,
        max_snap_dist_m=MAX_SNAP_DIST_M, max_speed_mps=MAX_SPEED_MPS,
        min_snr=MIN_SNR, min_baseline_m=MIN_BASELINE_M,
    )
    intervals, obs = speeds["intervals"], speeds["edge_observations"]
    if intervals.empty:
        raise SystemExit(
            "No speed intervals were derived. Common causes: MAX_DIST too "
            "small for your GPS error, ID_COL not identifying real movers, or "
            "timestamps that do not increase within a mover.")
    good = int(intervals["quality"].sum())
    print(f"        {len(intervals):,} intervals ({good:,} pass quality), "
          f"{len(obs):,} edge observations")

    # -- 6. outlier screening [OPTIONAL] ---------------------------------------
    # Note there is no min_speed here, deliberately: see the CONFIG note above.
    if MAD_OUTLIERS:
        n_before = len(obs)
        obs = rl.filter_by_speed(
            obs, max_speed=MAX_PLAUSIBLE_SPEED, unit=OUTPUT_UNIT,
            mad_outliers=True, mad_threshold=MAD_THRESHOLD, per_edge=True)
        print(f"[6/12] outlier screening: {n_before:,} -> {len(obs):,} "
              f"observations")

    # -- 7. aggregate ----------------------------------------------------------
    # Network-wide by time block. Deduplicates on interval_id automatically, so
    # the edge-level duplication above does not inflate sample sizes.
    print("[7/12] aggregating")
    hourly = rl.aggregate_speeds(
        obs, block_hours=BLOCK_HOURS, statistic=STATISTIC,
        output_unit=OUTPUT_UNIT, min_samples=MIN_SAMPLES, days=DAYS,
        require_quality=REQUIRE_QUALITY, weight_by_variance=WEIGHT_BY_VARIANCE)
    per_edge = rl.aggregate_speeds(
        obs, block_hours=BLOCK_HOURS, statistic=STATISTIC, by_edge=True,
        output_unit=OUTPUT_UNIT, min_samples=MIN_SAMPLES, days=DAYS,
        require_quality=REQUIRE_QUALITY)
    hourly.to_csv(os.path.join(OUT_DIR, "speeds_hourly.csv"), index=False)
    per_edge.to_csv(os.path.join(OUT_DIR, "speeds_per_edge.csv"), index=False)
    print(f"        {len(hourly)} time blocks, {len(per_edge):,} edge-blocks")

    # -- 8. peak / off-peak ----------------------------------------------------
    print("[8/12] peak analysis")
    windows = rl.classify_hours(obs, n_peak=N_PEAK, n_offpeak=N_OFFPEAK,
                                statistic=STATISTIC, days=DAYS,
                                require_quality=REQUIRE_QUALITY)
    print(f"        peak {windows['peak_hours']}  "
          f"off-peak {windows['offpeak_hours']}")
    # peak_analysis ranks INDIVIDUAL hours rather than returning a contiguous
    # window, so on a two-rush-period day it names both. It takes the
    # AGGREGATED frame, not the raw observations.
    ranked = rl.peak_analysis(hourly, statistic=STATISTIC,
                              n_peak=N_PEAK, n_offpeak=N_OFFPEAK)
    print(f"        slowest hours: "
          f"{[r['block_label'] for r in ranked['peak']]}")
    # Weekday vs weekend, which are different populations.
    daytype = rl.day_type_report(obs, statistic=STATISTIC,
                                 output_unit=OUTPUT_UNIT,
                                 n_peak=N_PEAK, n_offpeak=N_OFFPEAK,
                                 require_quality=REQUIRE_QUALITY)
    print(f"        weekday vs weekend: "
          f"{daytype['overall'].get('delta_pct', float('nan')):.1f}% difference")

    # -- 9. write speeds onto the network --------------------------------------
    # Both of these MUTATE `net` in place, attaching edge attributes that the
    # router, the exports and the congestion report all read.
    print("[9/12] assigning speeds to edges")
    rl.assign_speeds(net, obs, statistic=STATISTIC, days=DAYS,
                     require_quality=REQUIRE_QUALITY)
    coverage = rl.assign_segment_speeds(
        net, obs, statistic=STATISTIC, n_peak=N_PEAK, n_offpeak=N_OFFPEAK,
        days=DAYS, require_quality=REQUIRE_QUALITY)
    print(f"        {coverage['coverage']['overall']:,} of "
          f"{coverage['n_edges_total']:,} edges have an observed speed")

    # -- 10. congestion + network area analysis ---------------------------------
    print("[10/12] congestion and network structure")
    # Observed speed over posted limit. Edges with no usable maxspeed tag get a
    # NaN ratio and are excluded from the summary rather than assumed free.
    cong = rl.congestion_report(net, obs, statistic=STATISTIC,
                                output_unit=OUTPUT_UNIT, days=DAYS,
                                min_samples=MIN_SAMPLES,
                                require_quality=REQUIRE_QUALITY)
    cong["edges"].to_csv(os.path.join(OUT_DIR, "congestion_by_edge.csv"),
                         index=False)
    s = cong["summary"]
    print(f"        median observed/posted ratio {s['median_ratio']:.2f} "
          f"across {s['n_edges_rated']:,} rated edges")

    # THE ROAD NETWORK AREA ANALYSER. network_stats measures the study area
    # from the network's own projected geometry and reports densities per km2.
    stats = rl.network_stats(net, area_km2=AREA_KM2, area_method=AREA_METHOD)
    print(f"        {stats['n_physical_roads']:,} roads, "
          f"{stats['n_intersections']:,} intersections, "
          f"{stats['n_dead_ends']:,} dead ends")
    # edge_density_km2 is METRES OF PHYSICAL ROAD per km2 -- a two-way road is
    # counted once, not twice -- so it is a road-density figure, not a count.
    print(f"        area {stats['area_km2']:.2f} km2 "
          f"(from {stats['area_method']})")
    print(f"        {stats['intersection_density_km2']:.1f} intersections/km2, "
          f"{stats['edge_density_km2']:,.0f} m of road/km2")
    print(f"        streets per node {stats['streets_per_node_avg']:.2f}, "
          f"circuity {stats['circuity_avg']:.3f}")
    conn = rl.connectivity_report(net)
    if not conn["is_strongly_connected"]:
        # Worth reading closely: unexpected components usually mean the source
        # data is not noded at intersections, which silently breaks routing.
        print(f"        WARNING: {conn['n_strongly_connected_components']} "
              f"components; largest holds "
              f"{100 * conn['largest_component_fraction_nodes']:.1f}% of nodes")

    # -- 11. chokepoints [OPTIONAL] -------------------------------------------
    print("[11/12] chokepoints")
    bc = rl.edge_betweenness_centrality(
        net, weight=CENTRALITY_WEIGHT, normalized=True,
        k=CENTRALITY_SAMPLE_K, seed=0 if CENTRALITY_SAMPLE_K else None,
        write_attr="betweenness")
    # Collapse to physical roads: a two-way road is two directed edges with the
    # same score, and listing both just fills the top five with duplicates.
    by_road: dict[int, tuple[float, str]] = {}
    for eid, score in bc.items():
        rid = min(int(x) for x in net.road_edge_ids(int(eid)))
        name = net.edge_data(int(eid)).get("name") or f"edge {rid}"
        if rid not in by_road or score > by_road[rid][0]:
            by_road[rid] = (score, name, rid)
    # The id is printed too: several distinct segments often share a street
    # name, and identical scores on a regular grid are real, not duplicates.
    for score, name, rid in sorted(by_road.values(), key=lambda v: -v[0])[:5]:
        print(f"        {score:.4f}  {name}  (road {rid})")

    # -- 12. export ------------------------------------------------------------
    print("[12/12] exporting")
    gj_path = os.path.join(OUT_DIR, f"speeds_{EXPORT_PERIOD}.geojson")
    rl.to_geojson(net, gj_path, directional=DIRECTIONAL, period=EXPORT_PERIOD,
                  speed_unit=OUTPUT_UNIT)
    print(f"        {gj_path}")
    if MAKE_MAP:
        try:
            png = os.path.join(OUT_DIR, f"speed_map_{EXPORT_PERIOD}.png")
            rl.plot_speed_map(net, png, period=EXPORT_PERIOD,
                              speed_unit=OUTPUT_UNIT)
            print(f"        {png}")
        except ImportError as exc:
            print(f"        skipped map: {exc}")

    print(f"\nDone. Outputs in {os.path.abspath(OUT_DIR)}/")


# Required for N_JOBS != 1: the 'spawn' start method re-imports this module in
# every worker process, and without this guard each one would re-run main().
if __name__ == "__main__":
    main()
