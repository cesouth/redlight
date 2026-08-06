import numpy as np
import pandas as pd
import pytest

import redlight as rl


def obs(hour, speed_mps, *, edge_id=0, n=1, minute=15, interval_id=None):
    rows = []
    for k in range(n):
        r = {"edge_id": edge_id, "speed_mps": speed_mps,
             "time": pd.Timestamp(f"2026-06-01 {hour:02d}:{minute:02d}:{k:02d}")}
        if interval_id is not None:
            r["interval_id"] = interval_id + k
        rows.append(r)
    return rows


def day_profile(*, slow_hours=(7, 8), fast_hours=(22, 23), n_per_hour=4,
                slow=4.0, base=10.0, fast=16.0):
    rows = []
    for h in range(24):
        sp = slow if h in slow_hours else (fast if h in fast_hours else base)
        rows += obs(h, sp, n=n_per_hour)
    return pd.DataFrame(rows)


def test_aggregate_hourly_mean_exact():
    df = pd.DataFrame(obs(8, 10.0) + obs(8, 14.0) + obs(9, 6.0))
    out = rl.aggregate_speeds(df, statistic="both", output_unit="mps")
    r8 = out[out["block_start_hour"] == 8].iloc[0]
    assert r8["mean_speed"] == pytest.approx(12.0)
    assert r8["median_speed"] == pytest.approx(12.0)
    assert r8["n"] == 2
    assert r8["block_label"] == "08:00-09:00"


def test_single_observation_bin_has_nan_uncertainty():
    """Regression: n=1 bins used to report a zero-width 95% CI."""
    out = rl.aggregate_speeds(pd.DataFrame(obs(9, 6.0)), output_unit="mps")
    r = out.iloc[0]
    assert np.isnan(r["std_speed"]) and np.isnan(r["sem_speed"])
    assert np.isnan(r["ci95_low"]) and np.isnan(r["ci95_high"])
    assert r["mean_speed"] == pytest.approx(6.0)


def test_unmatched_rows_excluded():
    """Regression: edge_id == -1 rows used to flow into network stats."""
    df = pd.DataFrame(obs(8, 21.0) + obs(8, 1.0, edge_id=-1))
    out = rl.aggregate_speeds(df, output_unit="mps")
    assert out.iloc[0]["n"] == 1
    assert out.iloc[0]["mean_speed"] == pytest.approx(21.0)


def test_interval_dedup_network_wide():
    """Regression: edge_observations duplication inflated n/SEM."""
    rows = []
    for eid in range(8):  # one interval attributed to 8 edges
        rows += obs(8, 10.0, edge_id=eid, interval_id=0)
    df = pd.DataFrame(rows)
    net_wide = rl.aggregate_speeds(df, output_unit="mps")
    assert net_wide.iloc[0]["n"] == 1
    per_edge = rl.aggregate_speeds(df, output_unit="mps", by_edge=True)
    assert len(per_edge) == 8  # per-edge keeps the deliberate duplication


def test_min_samples_suppresses_bins():
    df = pd.DataFrame(obs(8, 10.0, n=3) + obs(9, 6.0))
    out = rl.aggregate_speeds(df, min_samples=2, output_unit="mps")
    assert out["block_start_hour"].tolist() == [8]


def test_block_hours_warns_when_not_dividing_24():
    with pytest.warns(UserWarning, match="does not divide 24"):
        rl.aggregate_speeds(day_profile(), block_hours=5, output_unit="mps")


def test_peak_analysis_ranks_slowest_first():
    agg = rl.aggregate_speeds(day_profile(), output_unit="mps")
    res = rl.peak_analysis(agg, n_peak=2, n_offpeak=2)
    peak_hours = {r["block_start_hour"] for r in res["peak"]}
    off_hours = {r["block_start_hour"] for r in res["off_peak"]}
    assert peak_hours == {7, 8}
    assert off_hours == {22, 23}


def test_peak_analysis_rejects_per_edge_aggregation():
    agg = rl.aggregate_speeds(day_profile(), by_edge=True, output_unit="mps")
    with pytest.raises(ValueError, match="by_edge"):
        rl.peak_analysis(agg)


def test_peak_analysis_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        rl.peak_analysis(pd.DataFrame())


def test_peak_analysis_warns_on_overlap():
    df = pd.DataFrame(obs(8, 10.0, n=2) + obs(9, 6.0, n=2))
    agg = rl.aggregate_speeds(df, output_unit="mps")
    with pytest.warns(UserWarning, match="overlap"):
        rl.peak_analysis(agg, n_peak=2, n_offpeak=2)


def test_classify_hours_median_split():
    res = rl.classify_hours(day_profile())
    assert res["source"] == "auto"
    assert 7 in res["peak_hours"] and 8 in res["peak_hours"]
    assert 22 in res["offpeak_hours"] and 23 in res["offpeak_hours"]
    assert set(res["peak_hours"]).isdisjoint(res["offpeak_hours"])


def test_classify_hours_override_validation():
    df = day_profile()
    with pytest.raises(ValueError, match="overlap"):
        rl.classify_hours(df, peak_hours=[7, 8], offpeak_hours=[8, 22])
    with pytest.raises(ValueError, match="invalid hours"):
        rl.classify_hours(df, peak_hours=[7, 24])
    res = rl.classify_hours(df, peak_hours=[7, 8])
    assert res["source"] == "override"
    assert res["offpeak_hours"] == [h for h in range(24) if h not in (7, 8)]


def test_classify_hours_window_mode():
    res = rl.classify_hours(day_profile(), n_peak=2, n_offpeak=2)
    assert res["source"] == "window"
    assert res["peak_hours"] == [7, 8]
    assert res["offpeak_hours"] == [22, 23]
    assert res["peak_speed_mps"] == pytest.approx(4.0)
    assert res["offpeak_speed_mps"] == pytest.approx(16.0)


def test_classify_hours_window_wraps_midnight():
    df = day_profile(fast_hours=(23, 0), slow_hours=(7, 8))
    res = rl.classify_hours(df, n_peak=2, n_offpeak=2)
    assert res["offpeak_hours"] == [0, 23]  # the 23:00-01:00 window


def test_classify_hours_window_validation():
    df = day_profile()
    with pytest.raises(ValueError, match="both n_peak and n_offpeak"):
        rl.classify_hours(df, n_peak=3)
    with pytest.raises(ValueError, match="exceeds 24"):
        rl.classify_hours(df, n_peak=20, n_offpeak=10)
    with pytest.raises(ValueError, match="between 1 and 23"):
        rl.classify_hours(df, n_peak=0, n_offpeak=2)


def test_classify_hours_degenerate_ties_warns():
    rows = []
    for h in (8, 9, 10):
        rows += obs(h, 10.0, n=2)
    with pytest.warns(UserWarning, match="off-peak set is empty"):
        res = rl.classify_hours(pd.DataFrame(rows))
    assert res["offpeak_hours"] == []


def test_assign_speeds_observed_count_excludes_default(straight_net):
    """Regression: edges written with the default used to count as observed."""
    df = pd.DataFrame(obs(8, 10.0, edge_id=0, n=3))  # edge 1 has no observations
    info = rl.assign_speeds(straight_net, df, default_speed_mps=5.0)
    assert info["n_edges_observed"] == 1
    assert info["n_edges_total"] == 2
    # default still written so time routing works
    assert straight_net.edge_data(1)["obs_speed_mps"] == 5.0


def test_assign_speeds_zero_speed_is_observed_gridlock(straight_net):
    """Regression: 0.0 m/s (gridlock) used to be treated as 'unobserved', so
    a genuinely stopped edge silently fell back to the default speed instead
    of reporting the real gridlock."""
    df = pd.DataFrame(obs(8, 0.0, edge_id=0, n=3))
    info = rl.assign_speeds(straight_net, df, default_speed_mps=5.0)
    assert info["n_edges_observed"] == 1
    assert straight_net.edge_data(0)["obs_speed_mps"] == pytest.approx(0.0)
    assert "travel_time_s" not in straight_net.edge_data(0)  # undefined at 0 m/s
    assert straight_net.edge_data(1)["obs_speed_mps"] == 5.0  # truly unobserved


def test_assign_segment_speeds_regimes(straight_net):
    df = day_profile()
    info = rl.assign_segment_speeds(straight_net, df, n_peak=2, n_offpeak=2)
    assert info["source"] == "window"
    assert info["peak_hours"] == [7, 8]
    assert info["offpeak_hours"] == [22, 23]
    d = straight_net.edge_data(0)
    assert d["obs_speed_mps_peak"] == pytest.approx(4.0)
    assert d["obs_speed_mps_offpeak"] == pytest.approx(16.0)
    assert d["obs_speed_mps_overall"] == pytest.approx(10.0)
    assert d["obs_speed_mps"] == d["obs_speed_mps_overall"]
    assert info["coverage"]["peak"] == 1  # only edge 0 observed


# --------------------------------------------------------------- day-of-week

# June 2026: the 1st is a Monday, so 2026-06-01 is a weekday (dow 0) and
# 2026-06-06 / 2026-06-07 are Saturday / Sunday (dow 5 / 6).
WEEKDAY = "2026-06-01"   # Monday
SATURDAY = "2026-06-06"
SUNDAY = "2026-06-07"


def obs_on(date, hour, speed_mps, *, edge_id=0, n=1, minute=15):
    rows = []
    for k in range(n):
        rows.append({"edge_id": edge_id, "speed_mps": speed_mps,
                     "time": pd.Timestamp(f"{date} {hour:02d}:{minute:02d}:{k:02d}")})
    return rows


def day_profile_on(date, *, slow_hours=(7, 8), fast_hours=(22, 23), n_per_hour=4,
                   slow=4.0, base=10.0, fast=16.0, edge_id=0):
    rows = []
    for h in range(24):
        sp = slow if h in slow_hours else (fast if h in fast_hours else base)
        rows += obs_on(date, h, sp, n=n_per_hour, edge_id=edge_id)
    return rows


def test_days_filter_selects_weekday_vs_weekend():
    df = pd.DataFrame(obs_on(WEEKDAY, 8, 5.0) + obs_on(SATURDAY, 8, 20.0))
    wd = rl.aggregate_speeds(df, days="weekday", output_unit="mps")
    we = rl.aggregate_speeds(df, days="weekend", output_unit="mps")
    both = rl.aggregate_speeds(df, output_unit="mps")  # days=None default
    assert wd.iloc[0]["mean_speed"] == pytest.approx(5.0) and wd.iloc[0]["n"] == 1
    assert we.iloc[0]["mean_speed"] == pytest.approx(20.0) and we.iloc[0]["n"] == 1
    assert both.iloc[0]["n"] == 2  # unfiltered pools the two days


def test_days_presets_names_numbers_agree():
    df = pd.DataFrame(obs_on(SATURDAY, 8, 20.0) + obs_on(SUNDAY, 8, 22.0)
                      + obs_on(WEEKDAY, 8, 5.0))
    by_preset = rl.aggregate_speeds(df, days="weekend", output_unit="mps").iloc[0]
    by_names = rl.aggregate_speeds(df, days=["sat", "Sunday"], output_unit="mps").iloc[0]
    by_nums = rl.aggregate_speeds(df, days=[5, 6], output_unit="mps").iloc[0]
    assert by_preset["n"] == by_names["n"] == by_nums["n"] == 2
    assert by_preset["mean_speed"] == pytest.approx(by_nums["mean_speed"])


def test_days_all_is_noop():
    df = pd.DataFrame(obs_on(WEEKDAY, 8, 5.0) + obs_on(SATURDAY, 8, 20.0))
    a = rl.aggregate_speeds(df, days="all", output_unit="mps").iloc[0]
    b = rl.aggregate_speeds(df, output_unit="mps").iloc[0]
    assert a["n"] == b["n"] == 2


def test_days_invalid_raises():
    df = pd.DataFrame(obs_on(WEEKDAY, 8, 5.0))
    with pytest.raises(ValueError, match="Unrecognised day"):
        rl.aggregate_speeds(df, days="funday")
    with pytest.raises(ValueError, match="out of range"):
        rl.aggregate_speeds(df, days=[7])


def test_days_accepts_numpy_scalar():
    """A numpy integer is not a Python int subclass, so a bare
    ``isinstance(days, int)`` test sends it down the iterate-it path and it
    dies as 'numpy.int64 object is not iterable' -- with no mention of days=.
    Reachable straight from pandas: ``df['time'].dt.dayofweek.unique()[0]``."""
    import numpy as np
    df = pd.DataFrame(obs_on(WEEKDAY, 8, 5.0) + obs_on(SATURDAY, 8, 9.0))
    out = rl.aggregate_speeds(df, days=np.int64(0), output_unit="mps")
    assert len(out) == 1
    assert out["mean_speed"].iloc[0] == pytest.approx(5.0)


def test_day_type_report_weekday_vs_weekend():
    weekday = day_profile_on(WEEKDAY)                       # slow 7-8, fast 22-23
    weekend = day_profile_on(SATURDAY, slow=14.0, base=18.0, fast=22.0)
    rep = rl.day_type_report(pd.DataFrame(weekday + weekend),
                             statistic="median", output_unit="mps")
    g = rep["groups"]
    assert g["weekday"]["n"] == 96 and g["weekend"]["n"] == 96
    assert g["weekday"]["days"] == [0, 1, 2, 3, 4]
    assert g["weekend"]["days"] == [5, 6]
    assert g["weekday"]["overall_speed"] == pytest.approx(10.0)
    assert g["weekend"]["overall_speed"] == pytest.approx(18.0)
    # weekend is less congested overall -> positive delta (weekend - weekday)
    assert rep["overall"]["delta_speed"] == pytest.approx(8.0)
    # peak = slowest block, recovered independently per day-type
    assert g["weekday"]["peak"][0]["speed"] == pytest.approx(4.0)
    assert g["weekend"]["peak"][0]["speed"] == pytest.approx(14.0)
    comp = rep["comparison"]
    assert len(comp) == 24
    assert {"weekday_speed", "weekend_speed", "delta_speed", "delta_pct"} <= set(comp.columns)
    # the 08:00 block: weekday 4, weekend 14 -> delta 10
    row8 = comp[comp["block_start_hour"] == 8].iloc[0]
    assert row8["delta_speed"] == pytest.approx(10.0)


def test_day_type_report_empty_group_is_graceful():
    """A dataset with no weekend fixes reports n=0/NaN and warns, not raises."""
    weekday = day_profile_on(WEEKDAY)
    with pytest.warns(UserWarning, match="no observations"):
        rep = rl.day_type_report(pd.DataFrame(weekday), output_unit="mps")
    assert rep["groups"]["weekend"]["n"] == 0
    assert np.isnan(rep["groups"]["weekend"]["overall_speed"])
    assert rep["groups"]["weekend"]["peak"] is None
    assert rep["comparison"]["weekend_speed"].isna().all()


def test_day_type_report_custom_groups():
    df = pd.DataFrame(obs_on(WEEKDAY, 8, 5.0, n=2)        # Monday
                      + obs_on("2026-06-05", 8, 9.0, n=2))  # Friday
    rep = rl.day_type_report(df, groups={"Mon": "mon", "Fri": [4]},
                             output_unit="mps")
    assert rep["groups"]["Mon"]["overall_speed"] == pytest.approx(5.0)
    assert rep["groups"]["Fri"]["overall_speed"] == pytest.approx(9.0)
    assert "delta_speed" in rep["comparison"].columns


def test_assign_segment_speeds_days_filter(straight_net):
    """days= restricts which weekdays feed per-edge speeds."""
    df = pd.DataFrame(obs_on(WEEKDAY, 8, 4.0, edge_id=0, n=3)
                      + obs_on(SATURDAY, 8, 16.0, edge_id=0, n=3))
    info = rl.assign_segment_speeds(straight_net, df, peak_hours=[8],
                                    offpeak_hours=[20], days="weekday",
                                    default_speed_mps=10.0)
    assert info["days"] == [0, 1, 2, 3, 4]
    assert straight_net.edge_data(0)["obs_speed_mps_peak"] == pytest.approx(4.0)


def test_assign_segment_speeds_zero_speed_is_observed_gridlock(straight_net):
    """Regression: a regime with only 0.0 m/s observations used to write
    nothing at all (spd=0.0 is falsy), leaving the edge with no speed data
    instead of the real gridlock reading."""
    df = pd.DataFrame(obs(8, 0.0, edge_id=0, n=2) + obs(22, 16.0, edge_id=0, n=2))
    info = rl.assign_segment_speeds(straight_net, df, peak_hours=[8],
                                    offpeak_hours=[22])
    d = straight_net.edge_data(0)
    assert d["obs_speed_mps_peak"] == pytest.approx(0.0)
    assert "travel_time_s_peak" not in d  # undefined at 0 m/s
    assert info["coverage"]["peak"] == 1


# ------------------------------------------------ interval_id collisions
def test_colliding_interval_ids_raise():
    """derive_speeds numbers intervals from 0 per call, so concatenating two
    runs makes genuinely different intervals share an id -- and the
    network-wide dedup would silently drop one of every colliding pair."""
    run_a = pd.DataFrame(obs(8, 10.0, interval_id=0) + obs(9, 11.0, interval_id=1))
    run_b = pd.DataFrame(obs(10, 20.0, interval_id=0) + obs(11, 21.0, interval_id=1))
    both = pd.concat([run_a, run_b], ignore_index=True)
    with pytest.raises(ValueError, match="interval_id"):
        rl.aggregate_speeds(both, output_unit="mps")


def test_colliding_interval_ids_ignored_when_dedup_off():
    run_a = pd.DataFrame(obs(8, 10.0, interval_id=0))
    run_b = pd.DataFrame(obs(10, 20.0, interval_id=0))
    both = pd.concat([run_a, run_b], ignore_index=True)
    out = rl.aggregate_speeds(both, output_unit="mps", dedup_intervals=False)
    assert out["n"].sum() == 2


# ------------------------------------------------ inverse-variance weighting
def _var_obs(hour, speed_mps, var, *, interval_id, edge_id=0):
    return [{"edge_id": edge_id, "speed_mps": speed_mps, "speed_var": var,
             "interval_id": interval_id,
             "time": pd.Timestamp(f"2026-06-01 {hour:02d}:15:00")}]


def test_inverse_variance_weighting_favours_precise_observations():
    """A speed measured to var=1 should outweigh one measured to var=4."""
    df = pd.DataFrame(_var_obs(8, 10.0, 1.0, interval_id=0)
                      + _var_obs(8, 20.0, 4.0, interval_id=1))
    plain = rl.aggregate_speeds(df, output_unit="mps", statistic="mean")
    wtd = rl.aggregate_speeds(df, output_unit="mps", statistic="mean",
                              weight_by_variance=True)
    assert plain["mean_speed"].iloc[0] == pytest.approx(15.0)
    # (10/1 + 20/4) / (1/1 + 1/4) = 15 / 1.25
    assert wtd["mean_speed"].iloc[0] == pytest.approx(12.0)
    # SEM of an inverse-variance weighted mean is sqrt(1 / sum of weights)
    assert wtd["sem_speed"].iloc[0] == pytest.approx((1 / 1.25) ** 0.5)
    assert wtd["n"].iloc[0] == 2


def test_weighting_requires_speed_var_column():
    df = pd.DataFrame(obs(8, 10.0, interval_id=0))
    with pytest.raises(ValueError, match="speed_var"):
        rl.aggregate_speeds(df, weight_by_variance=True)


def test_weighting_requires_a_mean():
    df = pd.DataFrame(_var_obs(8, 10.0, 1.0, interval_id=0))
    with pytest.raises(ValueError, match="weight_by_variance"):
        rl.aggregate_speeds(df, statistic="median", weight_by_variance=True)


def test_unusable_variances_are_dropped_with_a_warning():
    """var <= 0 would be an infinite weight; NaN would poison the sum."""
    df = pd.DataFrame(_var_obs(8, 10.0, 1.0, interval_id=0)
                      + _var_obs(8, 99.0, 0.0, interval_id=1)
                      + _var_obs(8, 99.0, float("nan"), interval_id=2))
    with pytest.warns(UserWarning, match="speed_var"):
        out = rl.aggregate_speeds(df, output_unit="mps", statistic="mean",
                                  weight_by_variance=True)
    assert out["n"].iloc[0] == 1
    assert out["mean_speed"].iloc[0] == pytest.approx(10.0)


# ------------------------------------------------ quality flag consumption
def _q(hour, speed_mps, quality, *, interval_id, edge_id=0):
    return [{"edge_id": edge_id, "speed_mps": speed_mps, "quality": quality,
             "interval_id": interval_id,
             "time": pd.Timestamp(f"2026-06-01 {hour:02d}:15:00")}]


def test_require_quality_drops_flagged_observations():
    """derive_speeds flags intervals it does not trust; until now nothing in
    the package could act on that flag."""
    df = pd.DataFrame(_q(8, 10.0, True, interval_id=0)
                      + _q(8, 99.0, False, interval_id=1))
    every = rl.aggregate_speeds(df, output_unit="mps")
    good = rl.aggregate_speeds(df, output_unit="mps", require_quality=True)
    assert every["n"].iloc[0] == 2
    assert good["n"].iloc[0] == 1
    assert good["mean_speed"].iloc[0] == pytest.approx(10.0)


def test_require_quality_without_the_column_raises():
    df = pd.DataFrame(obs(8, 10.0, interval_id=0))
    with pytest.raises(ValueError, match="quality"):
        rl.aggregate_speeds(df, require_quality=True)


def test_require_quality_reaches_assign_speeds(straight_net):
    df = pd.DataFrame(_q(8, 10.0, True, interval_id=0)
                      + _q(8, 99.0, False, interval_id=1))
    rl.assign_speeds(straight_net, df, require_quality=True)
    assert straight_net.edge_data(0)["obs_speed_mps"] == pytest.approx(10.0)


def test_require_quality_reaches_day_type_report():
    df = pd.DataFrame(_q(8, 10.0, True, interval_id=0)
                      + _q(8, 99.0, False, interval_id=1))
    rep = rl.day_type_report(df, groups={"wk": "weekday"}, output_unit="mps",
                             require_quality=True)
    assert rep["groups"]["wk"]["n"] == 1
    assert rep["groups"]["wk"]["overall_speed"] == pytest.approx(10.0)


def test_day_type_report_can_weight_by_variance():
    rows = (_var_obs(8, 10.0, 1.0, interval_id=0)
            + _var_obs(8, 20.0, 4.0, interval_id=1))
    rep = rl.day_type_report(pd.DataFrame(rows), groups={"wk": "weekday"},
                             statistic="mean", output_unit="mps",
                             weight_by_variance=True)
    assert rep["groups"]["wk"]["overall_speed"] == pytest.approx(12.0)


# ------------------------------------------------ congestion vs posted limit
def _limit_net(tmp_path, maxspeed="72 km/h"):   # 72 km/h == exactly 20 m/s
    from conftest import line_feature, write_geojson
    return rl.Network.from_geojson(write_geojson(tmp_path / "lim.json", [
        line_feature([[0, 0], [0.01, 0]], highway="primary", maxspeed=maxspeed),
    ]))


def test_congestion_report_ratio_against_posted_limit(tmp_path):
    net = _limit_net(tmp_path)
    df = pd.DataFrame(obs(8, 10.0, edge_id=0, n=3))     # half the limit
    rep = rl.congestion_report(net, df, output_unit="mps")
    row = rep["edges"].set_index("edge_id").loc[0]
    assert row["observed_speed"] == pytest.approx(10.0)
    assert row["speed_limit"] == pytest.approx(20.0)
    assert row["ratio"] == pytest.approx(0.5)


def test_congestion_report_unrated_edge_has_nan_ratio(tmp_path):
    from conftest import line_feature, write_geojson
    net = rl.Network.from_geojson(write_geojson(tmp_path / "n.json", [
        line_feature([[0, 0], [0.01, 0]], highway="primary"),   # no maxspeed
    ]))
    df = pd.DataFrame(obs(8, 10.0, edge_id=0, n=3))
    rep = rl.congestion_report(net, df, output_unit="mps")
    row = rep["edges"].set_index("edge_id").loc[0]
    assert np.isnan(row["speed_limit"]) and np.isnan(row["ratio"])
    assert rep["summary"]["n_edges_rated"] == 0


def test_congestion_report_reports_speeding_honestly(tmp_path):
    """Observed above the limit is real information, not something to clip."""
    net = _limit_net(tmp_path)
    df = pd.DataFrame(obs(8, 25.0, edge_id=0, n=3))
    rep = rl.congestion_report(net, df, output_unit="mps")
    assert rep["edges"].set_index("edge_id").loc[0]["ratio"] == pytest.approx(1.25)


def test_congestion_report_by_hour(tmp_path):
    net = _limit_net(tmp_path)
    df = pd.DataFrame(obs(8, 5.0, edge_id=0, n=3) + obs(14, 20.0, edge_id=0, n=3))
    rep = rl.congestion_report(net, df, output_unit="mps", block_hours=1)
    e = rep["edges"].set_index("block_start_hour")
    assert e.loc[8, "ratio"] == pytest.approx(0.25)
    assert e.loc[14, "ratio"] == pytest.approx(1.0)


def test_congestion_report_summary(tmp_path):
    net = _limit_net(tmp_path)
    df = pd.DataFrame(obs(8, 10.0, edge_id=0, n=3))
    s = rl.congestion_report(net, df, output_unit="mps")["summary"]
    assert s["n_edges_rated"] == 1
    assert s["median_ratio"] == pytest.approx(0.5)
