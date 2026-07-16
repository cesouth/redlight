import numpy as np
import pandas as pd
import pytest

import roadtraffic as rt


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
    out = rt.aggregate_speeds(df, statistic="both", output_unit="mps")
    r8 = out[out["block_start_hour"] == 8].iloc[0]
    assert r8["mean_speed"] == pytest.approx(12.0)
    assert r8["median_speed"] == pytest.approx(12.0)
    assert r8["n"] == 2
    assert r8["block_label"] == "08:00-09:00"


def test_single_observation_bin_has_nan_uncertainty():
    """Regression: n=1 bins used to report a zero-width 95% CI."""
    out = rt.aggregate_speeds(pd.DataFrame(obs(9, 6.0)), output_unit="mps")
    r = out.iloc[0]
    assert np.isnan(r["std_speed"]) and np.isnan(r["sem_speed"])
    assert np.isnan(r["ci95_low"]) and np.isnan(r["ci95_high"])
    assert r["mean_speed"] == pytest.approx(6.0)


def test_unmatched_rows_excluded():
    """Regression: edge_id == -1 rows used to flow into network stats."""
    df = pd.DataFrame(obs(8, 21.0) + obs(8, 1.0, edge_id=-1))
    out = rt.aggregate_speeds(df, output_unit="mps")
    assert out.iloc[0]["n"] == 1
    assert out.iloc[0]["mean_speed"] == pytest.approx(21.0)


def test_interval_dedup_network_wide():
    """Regression: edge_observations duplication inflated n/SEM."""
    rows = []
    for eid in range(8):  # one interval attributed to 8 edges
        rows += obs(8, 10.0, edge_id=eid, interval_id=0)
    df = pd.DataFrame(rows)
    net_wide = rt.aggregate_speeds(df, output_unit="mps")
    assert net_wide.iloc[0]["n"] == 1
    per_edge = rt.aggregate_speeds(df, output_unit="mps", by_edge=True)
    assert len(per_edge) == 8  # per-edge keeps the deliberate duplication


def test_min_samples_suppresses_bins():
    df = pd.DataFrame(obs(8, 10.0, n=3) + obs(9, 6.0))
    out = rt.aggregate_speeds(df, min_samples=2, output_unit="mps")
    assert out["block_start_hour"].tolist() == [8]


def test_block_hours_warns_when_not_dividing_24():
    with pytest.warns(UserWarning, match="does not divide 24"):
        rt.aggregate_speeds(day_profile(), block_hours=5, output_unit="mps")


def test_peak_analysis_ranks_slowest_first():
    agg = rt.aggregate_speeds(day_profile(), output_unit="mps")
    res = rt.peak_analysis(agg, n_peak=2, n_offpeak=2)
    peak_hours = {r["block_start_hour"] for r in res["peak"]}
    off_hours = {r["block_start_hour"] for r in res["off_peak"]}
    assert peak_hours == {7, 8}
    assert off_hours == {22, 23}


def test_peak_analysis_rejects_per_edge_aggregation():
    agg = rt.aggregate_speeds(day_profile(), by_edge=True, output_unit="mps")
    with pytest.raises(ValueError, match="by_edge"):
        rt.peak_analysis(agg)


def test_peak_analysis_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        rt.peak_analysis(pd.DataFrame())


def test_peak_analysis_warns_on_overlap():
    df = pd.DataFrame(obs(8, 10.0, n=2) + obs(9, 6.0, n=2))
    agg = rt.aggregate_speeds(df, output_unit="mps")
    with pytest.warns(UserWarning, match="overlap"):
        rt.peak_analysis(agg, n_peak=2, n_offpeak=2)


def test_classify_hours_median_split():
    res = rt.classify_hours(day_profile())
    assert res["source"] == "auto"
    assert 7 in res["peak_hours"] and 8 in res["peak_hours"]
    assert 22 in res["offpeak_hours"] and 23 in res["offpeak_hours"]
    assert set(res["peak_hours"]).isdisjoint(res["offpeak_hours"])


def test_classify_hours_override_validation():
    df = day_profile()
    with pytest.raises(ValueError, match="overlap"):
        rt.classify_hours(df, peak_hours=[7, 8], offpeak_hours=[8, 22])
    with pytest.raises(ValueError, match="invalid hours"):
        rt.classify_hours(df, peak_hours=[7, 24])
    res = rt.classify_hours(df, peak_hours=[7, 8])
    assert res["source"] == "override"
    assert res["offpeak_hours"] == [h for h in range(24) if h not in (7, 8)]


def test_classify_hours_window_mode():
    res = rt.classify_hours(day_profile(), n_peak=2, n_offpeak=2)
    assert res["source"] == "window"
    assert res["peak_hours"] == [7, 8]
    assert res["offpeak_hours"] == [22, 23]
    assert res["peak_speed_mps"] == pytest.approx(4.0)
    assert res["offpeak_speed_mps"] == pytest.approx(16.0)


def test_classify_hours_window_wraps_midnight():
    df = day_profile(fast_hours=(23, 0), slow_hours=(7, 8))
    res = rt.classify_hours(df, n_peak=2, n_offpeak=2)
    assert res["offpeak_hours"] == [0, 23]  # the 23:00-01:00 window


def test_classify_hours_window_validation():
    df = day_profile()
    with pytest.raises(ValueError, match="both n_peak and n_offpeak"):
        rt.classify_hours(df, n_peak=3)
    with pytest.raises(ValueError, match="exceeds 24"):
        rt.classify_hours(df, n_peak=20, n_offpeak=10)
    with pytest.raises(ValueError, match="between 1 and 23"):
        rt.classify_hours(df, n_peak=0, n_offpeak=2)


def test_classify_hours_degenerate_ties_warns():
    rows = []
    for h in (8, 9, 10):
        rows += obs(h, 10.0, n=2)
    with pytest.warns(UserWarning, match="off-peak set is empty"):
        res = rt.classify_hours(pd.DataFrame(rows))
    assert res["offpeak_hours"] == []


def test_assign_speeds_observed_count_excludes_default(straight_net):
    """Regression: edges written with the default used to count as observed."""
    df = pd.DataFrame(obs(8, 10.0, edge_id=0, n=3))  # edge 1 has no observations
    info = rt.assign_speeds(straight_net, df, default_speed_mps=5.0)
    assert info["n_edges_observed"] == 1
    assert info["n_edges_total"] == 2
    # default still written so time routing works
    assert straight_net.edge_data(1)["obs_speed_mps"] == 5.0


def test_assign_speeds_zero_speed_is_observed_gridlock(straight_net):
    """Regression: 0.0 m/s (gridlock) used to be treated as 'unobserved', so
    a genuinely stopped edge silently fell back to the default speed instead
    of reporting the real gridlock."""
    df = pd.DataFrame(obs(8, 0.0, edge_id=0, n=3))
    info = rt.assign_speeds(straight_net, df, default_speed_mps=5.0)
    assert info["n_edges_observed"] == 1
    assert straight_net.edge_data(0)["obs_speed_mps"] == pytest.approx(0.0)
    assert "travel_time_s" not in straight_net.edge_data(0)  # undefined at 0 m/s
    assert straight_net.edge_data(1)["obs_speed_mps"] == 5.0  # truly unobserved


def test_assign_segment_speeds_regimes(straight_net):
    df = day_profile()
    info = rt.assign_segment_speeds(straight_net, df, n_peak=2, n_offpeak=2)
    assert info["source"] == "window"
    assert info["peak_hours"] == [7, 8]
    assert info["offpeak_hours"] == [22, 23]
    d = straight_net.edge_data(0)
    assert d["obs_speed_mps_peak"] == pytest.approx(4.0)
    assert d["obs_speed_mps_offpeak"] == pytest.approx(16.0)
    assert d["obs_speed_mps_overall"] == pytest.approx(10.0)
    assert d["obs_speed_mps"] == d["obs_speed_mps_overall"]
    assert info["coverage"]["peak"] == 1  # only edge 0 observed


def test_assign_segment_speeds_zero_speed_is_observed_gridlock(straight_net):
    """Regression: a regime with only 0.0 m/s observations used to write
    nothing at all (spd=0.0 is falsy), leaving the edge with no speed data
    instead of the real gridlock reading."""
    df = pd.DataFrame(obs(8, 0.0, edge_id=0, n=2) + obs(22, 16.0, edge_id=0, n=2))
    info = rt.assign_segment_speeds(straight_net, df, peak_hours=[8],
                                    offpeak_hours=[22])
    d = straight_net.edge_data(0)
    assert d["obs_speed_mps_peak"] == pytest.approx(0.0)
    assert "travel_time_s_peak" not in d  # undefined at 0 m/s
    assert info["coverage"]["peak"] == 1
