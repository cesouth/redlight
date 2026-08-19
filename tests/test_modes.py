import numpy as np
import pandas as pd
import pytest

import redlight as rl


def obs_frame(specs, *, distance_m=200.0, t0="2026-06-01T08:00:00"):
    """A derive_speeds-shaped frame. ``specs`` maps traj_id -> list of m/s."""
    rows, iid = [], 0
    t0 = pd.Timestamp(t0)
    for tid, speeds in specs.items():
        for v in speeds:
            rows.append({
                "interval_id": iid, "traj_id": tid, "speed_mps": float(v),
                "time": t0 + pd.Timedelta(minutes=iid),
                "distance_m": distance_m, "snap_dist_m": 5.0,
            })
            iid += 1
    return pd.DataFrame(rows)


def test_mover_features_is_one_row_per_mover():
    feat = rl.mover_features(obs_frame({"a": [10.0, 20.0, 30.0], "b": [1.0, 1.5]}),
                             unit="mps")
    assert list(feat.index) == ["a", "b"]
    assert feat.loc["a", "n_intervals"] == 3
    assert feat.loc["b", "speed_median_mps"] == pytest.approx(1.25)
    assert feat.loc["a", "distance_m"] == pytest.approx(600.0)


def test_mover_features_dedups_repeated_intervals():
    """edge_observations repeats an interval once per edge crossed. Without
    dedup each mover's statistics get weighted by how many edges its hops
    spanned, biasing every mover toward its longest intervals."""
    obs = obs_frame({"a": [2.0, 20.0]})
    slow = obs[obs["speed_mps"] == 2.0]
    long_form = pd.concat([obs] + [slow] * 4, ignore_index=True)
    feat = rl.mover_features(long_form, unit="mps")
    assert feat.loc["a", "n_intervals"] == 2
    assert feat.loc["a", "speed_median_mps"] == pytest.approx(11.0)


def test_mover_features_converts_to_the_requested_unit():
    feat = rl.mover_features(obs_frame({"a": [10.0] * 4}), unit="mph")
    assert "speed_p85_mph" in feat.columns
    assert feat.loc["a", "speed_p85_mph"] == pytest.approx(10.0 / 0.44704, rel=1e-6)


def test_mover_features_missing_columns_raises():
    with pytest.raises(ValueError, match="traj_id"):
        rl.mover_features(pd.DataFrame({"speed_mps": [1.0]}))


def test_mover_features_empty_input_returns_typed_empty_frame():
    empty = pd.DataFrame({"interval_id": [], "traj_id": [], "speed_mps": [],
                          "distance_m": []})
    feat = rl.mover_features(empty, unit="mps")
    assert len(feat) == 0
    assert "speed_p85_mps" in feat.columns


def test_mover_features_empty_input_keeps_the_same_columns_as_a_full_one():
    """An emptied feed must not change the shape of the answer. A feed can
    filter down to nothing upstream, and a caller reading the documented
    snap_dist_m diagnostic should not get a KeyError for the privilege."""
    full = obs_frame({"a": [5.0] * 4})
    empty = full.iloc[:0]
    assert "snap_dist_m" in empty.columns, "the fixture must carry the column"
    assert (list(rl.mover_features(empty, unit="mps").columns)
            == list(rl.mover_features(full, unit="mps").columns))


def test_suggest_threshold_finds_the_valley_between_two_humps():
    rng = np.random.default_rng(0)
    walkers = rng.normal(1.4, 0.15, 90)
    vehicles = np.clip(rng.normal(12.0, 3.0, 200), 5.0, None)
    t = rl.suggest_mode_threshold(np.concatenate([walkers, vehicles]), unit="mps")
    assert t is not None
    assert 1.8 < t < 5.0


def test_suggest_threshold_rejects_a_valley_between_two_kinds_of_driving():
    """REGRESSION. A vehicle-only feed is itself multi-modal -- gridlock,
    urban free-flow and arterial free-flow form separate humps -- so a density
    valley proves nothing on its own. Ranking valleys by depth alone nominated
    13.2 mph on clean vehicle data, which would have deleted every congested
    vehicle in the study, invisibly. The hump BELOW the valley must be walkers."""
    rng = np.random.default_rng(1)
    gridlock = rng.normal(3.0, 0.4, 60)
    urban = rng.normal(11.0, 1.5, 120)
    arterial = rng.normal(24.0, 3.0, 80)
    speeds = np.concatenate([gridlock, urban, arterial])
    assert rl.suggest_mode_threshold(speeds, unit="mps") is None


def test_suggest_threshold_needs_enough_movers():
    assert rl.suggest_mode_threshold(np.full(10, 1.4), unit="mps") is None


@pytest.mark.parametrize("n_walkers,found", [(90, True), (15, False)])
def test_suggest_threshold_has_a_minority_detection_floor(n_walkers, found):
    """CHARACTERIZATION. The walking hump must carry real mass, not just sit at
    the right speed, so a small walking minority is reported as no split at all.
    That floor is a documented consequence of the mass guard, not an accident:
    it is the same guard that stops a vehicle-only feed's left-tail ripple from
    inventing pedestrians out of genuinely gridlocked vehicles. Pinned here so
    the docstring's stated floor cannot drift away from the behaviour."""
    rng = np.random.default_rng(0)
    speeds = np.concatenate([
        rng.normal(1.4, 0.15, n_walkers),
        np.clip(rng.normal(12.0, 3.0, 300 - n_walkers), 5.0, None),
    ])
    assert (rl.suggest_mode_threshold(speeds, unit="mps") is not None) is found


def test_suggest_threshold_accepts_a_series():
    rng = np.random.default_rng(0)
    speeds = np.concatenate([rng.normal(1.4, 0.15, 90),
                             np.clip(rng.normal(12.0, 3.0, 200), 5.0, None)])
    s = pd.Series(speeds, index=[f"m{i}" for i in range(len(speeds))])
    assert rl.suggest_mode_threshold(s, unit="mps") is not None


def test_suggest_threshold_is_unit_consistent():
    """The same population must yield the same physical speed whatever unit it
    is expressed in. Every existing test uses mps, where both conversions are
    the identity, so a swapped to_mps/from_mps would go undetected."""
    rng = np.random.default_rng(0)
    mps = np.concatenate([rng.normal(1.4, 0.15, 90),
                          np.clip(rng.normal(12.0, 3.0, 200), 5.0, None)])
    t_mps = rl.suggest_mode_threshold(mps, unit="mps")
    t_mph = rl.suggest_mode_threshold(mps / 0.44704, unit="mph")
    assert t_mps is not None and t_mph is not None
    # same physical speed, expressed two ways
    assert t_mph * 0.44704 == pytest.approx(t_mps, rel=1e-6)


def test_classify_labels_pedestrians_and_vehicles():
    obs = obs_frame({"walker": [1.3, 1.5, 1.4, 1.6],
                     "car": [3.0, 18.0, 20.0, 22.0]})
    m = rl.classify_movers(obs, threshold=3.0, unit="mps")
    assert m.loc["walker", "mode"] == "pedestrian"
    assert m.loc["car", "mode"] == "vehicle"


def test_a_congested_vehicle_is_still_a_vehicle():
    """It crawls for most of its trip but shows one free-flowing stretch, and
    the 85th percentile is chosen precisely so that stretch is visible."""
    obs = obs_frame({"stuck": [1.5, 1.6, 1.4, 1.5, 1.6, 1.5, 14.0, 15.0]})
    m = rl.classify_movers(obs, threshold=3.0, unit="mps")
    assert m.loc["stuck", "mode"] == "vehicle"


def test_unknown_comes_from_too_few_intervals():
    m = rl.classify_movers(obs_frame({"brief": [20.0, 21.0]}),
                           threshold=3.0, min_intervals=3, unit="mps")
    assert m.loc["brief", "mode"] == "unknown"


def test_unknown_comes_from_too_little_distance():
    m = rl.classify_movers(obs_frame({"short": [20.0] * 4}, distance_m=10.0),
                           threshold=3.0, min_distance_m=500.0, unit="mps")
    assert m.loc["short", "mode"] == "unknown"


def test_threshold_is_read_in_the_requested_unit():
    obs = obs_frame({"a": [1.4] * 5, "b": [12.0] * 5})
    mph = rl.classify_movers(obs, threshold=6.0, unit="mph")
    kph = rl.classify_movers(obs, threshold=6.0 * 1.609344, unit="kph")
    assert list(mph["mode"]) == list(kph["mode"])
    assert list(mph["mode"]) == ["pedestrian", "vehicle"]


def test_auto_threshold_raises_rather_than_guessing():
    rng = np.random.default_rng(2)
    obs = obs_frame({f"v{i}": [float(s)] * 4
                     for i, s in enumerate(rng.normal(12.0, 2.0, 60))})
    with pytest.raises(ValueError, match="walking"):
        rl.classify_movers(obs, threshold="auto", unit="mps")


def test_auto_threshold_works_when_there_are_walkers():
    rng = np.random.default_rng(0)
    specs = {f"p{i}": [float(v)] * 4
             for i, v in enumerate(rng.normal(1.4, 0.15, 90))}
    specs.update({f"v{i}": [float(v)] * 4
                  for i, v in enumerate(np.clip(rng.normal(12.0, 3.0, 200), 5.0, None))})
    m = rl.classify_movers(obs_frame(specs), threshold="auto", unit="mps")
    assert set(m["mode"]) == {"pedestrian", "vehicle"}
    assert (m["mode"] == "pedestrian").sum() == 90


def test_bad_threshold_string_raises():
    with pytest.raises(ValueError, match="'auto'"):
        rl.classify_movers(obs_frame({"a": [1.0] * 4}), threshold="fast", unit="mps")


def test_filter_keeps_every_observation_of_a_kept_mover():
    """THE invariant. A vehicle that crawls through congestion keeps its slow
    rows. That is the whole difference from filter_by_speed, which would delete
    them and take the congestion finding with them."""
    obs = obs_frame({"stuck": [1.5, 1.4, 1.5, 1.6, 14.0, 15.0],
                     "walker": [1.3, 1.4, 1.5, 1.4]})
    movers = rl.classify_movers(obs, threshold=3.0, unit="mps")
    out = rl.filter_by_mode(obs, movers)
    assert set(out["traj_id"]) == {"stuck"}
    assert len(out[out["speed_mps"] < 2.0]) == 4


def test_filter_excludes_unknown_by_default_but_can_include_it():
    obs = obs_frame({"car": [20.0] * 4, "brief": [20.0, 21.0]})
    movers = rl.classify_movers(obs, threshold=3.0, min_intervals=3, unit="mps")
    assert set(rl.filter_by_mode(obs, movers)["traj_id"]) == {"car"}
    both = rl.filter_by_mode(obs, movers, keep=("vehicle", "unknown"))
    assert set(both["traj_id"]) == {"car", "brief"}


def test_filter_warns_instead_of_exiting_when_everything_is_removed():
    obs = obs_frame({"walker": [1.4] * 4})
    movers = rl.classify_movers(obs, threshold=3.0, unit="mps")
    with pytest.warns(UserWarning, match="removed every"):
        out = rl.filter_by_mode(obs, movers)
    assert len(out) == 0


def test_filter_accepts_a_single_mode_string():
    obs = obs_frame({"car": [20.0] * 4, "walker": [1.4] * 4})
    movers = rl.classify_movers(obs, threshold=3.0, unit="mps")
    assert set(rl.filter_by_mode(obs, movers, keep="pedestrian")["traj_id"]) \
        == {"walker"}


def test_filter_requires_a_classification_table():
    obs = obs_frame({"a": [1.0] * 4})
    with pytest.raises(ValueError, match="classify_movers"):
        rl.filter_by_mode(obs, pd.DataFrame({"n_intervals": [4]}))


def test_verdicts_match_between_intervals_and_edge_observations():
    """A long-format frame must classify identically to the interval frame it
    was expanded from -- the dedup path in mover_features."""
    obs = obs_frame({"stuck": [1.5, 1.4, 14.0], "walker": [1.3, 1.4, 1.5, 1.4]})
    slow = obs[obs["speed_mps"] < 2.0]
    long_form = pd.concat([obs] + [slow] * 3, ignore_index=True)
    a = rl.classify_movers(obs, threshold=3.0, min_intervals=3, unit="mps")
    b = rl.classify_movers(long_form, threshold=3.0, min_intervals=3, unit="mps")
    assert a["mode"].to_dict() == b["mode"].to_dict()


def test_suggest_threshold_rejects_a_monotone_tail_with_no_walking_hump():
    """REGRESSION: a candidate valley in the far-left tail of a vehicle-only
    feed had no hump below it at all, but `argmax(dens[:i])` returned the point
    immediately below the candidate, so the walking-band guard passed
    trivially. Measured before the fix: 4 of these 20 seeds returned a
    threshold, and adding gridlocked vehicles made it label 37 of 430 real
    vehicles as pedestrians."""
    for seed in range(20):
        rng = np.random.default_rng(seed)
        mph = rng.lognormal(np.log(13), 0.45, 400)
        assert rl.suggest_mode_threshold(mph, unit="mph") is None, f"seed {seed}"


def test_gridlock_is_indistinguishable_from_walking_on_speed_alone():
    """Gridlocked vehicles and pedestrians at the same speed are the SAME input.

    The reviewer asked for a test asserting that 400 vehicles plus 30
    gridlocked ones at 2-4 mph yields no threshold. That is not achievable on
    speed alone, and this test pins down why: the "gridlocked vehicles" array
    and a "real pedestrians" array drawn identically are bit-for-bit equal, so
    no function of the speed distribution can return None for one and a
    threshold for the other. Rejecting this feed would require
    `_MIN_HUMP_SHARE > 0.22`, which would also blind the detector to any
    genuine walking population under ~8% of the feed.

    This is the limitation documented at the top of redlight.modes: a
    fully-gridlocked vehicle is classified as a pedestrian, the bias is upward,
    and the mitigation is to run the study screened and unscreened and report
    the gap. It is a known cost of the method, not a bug in the detector.
    """
    rng = np.random.default_rng(0)
    vehicles = rng.lognormal(np.log(13), 0.45, 400)
    gridlocked = rng.uniform(2.0, 4.0, 30)

    other = np.random.default_rng(0)
    other.lognormal(np.log(13), 0.45, 400)
    pedestrians = other.uniform(2.0, 4.0, 30)

    assert np.array_equal(gridlocked, pedestrians)
    assert (rl.suggest_mode_threshold(np.concatenate([vehicles, gridlocked]),
                                      unit="mph")
            == rl.suggest_mode_threshold(np.concatenate([vehicles, pedestrians]),
                                         unit="mph"))


def test_suggest_threshold_survives_a_degenerate_distribution():
    """A quantised/synthetic speed field where every mover shares one speed
    makes gaussian_kde raise LinAlgError on the singular covariance. A spike
    has no valley, so the answer is None -- not a traceback out of the CLI."""
    assert rl.suggest_mode_threshold(np.full(40, 2.0), unit="mps") is None


def test_a_null_trajectory_id_keeps_its_rows_through_classify_and_filter():
    """REGRESSION: speeds.py sets traj_id = None when the input carries no
    trajectory id. classify_movers grouped those into one null-indexed mover
    and labelled it a vehicle, but filter_by_mode's `isin` never matches null
    against null -- so a "keep" verdict silently deleted every row. Measured
    before the fix: all-None kept 0 of 6 rows while all-NaN kept 6 of 6."""
    obs = pd.DataFrame({
        "traj_id": [None] * 6,
        "speed_mps": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
        "distance_m": [100.0] * 6,
    })
    movers = rl.classify_movers(obs, threshold=5.0, unit="mps")
    assert list(movers["mode"]) == ["vehicle"]
    assert len(rl.filter_by_mode(obs, movers)) == len(obs)


def test_null_and_nan_trajectory_ids_filter_identically():
    """The None and NaN spellings of "no trajectory id" must not disagree."""
    base = {"speed_mps": [10.0, 11.0, 12.0, 13.0], "distance_m": [100.0] * 4}
    none_obs = pd.DataFrame({"traj_id": [None] * 4, **base})
    nan_obs = pd.DataFrame({"traj_id": [np.nan] * 4, **base})
    kept = [len(rl.filter_by_mode(o, rl.classify_movers(o, threshold=5.0,
                                                        unit="mps")))
            for o in (none_obs, nan_obs)]
    assert kept == [4, 4]


def _bimodal(frac_ped, *, n=300, seed=0):
    """Vehicles N(28, 6) mph and pedestrians N(3, 0.8) mph, ``frac_ped`` walkers."""
    r = np.random.default_rng(seed)
    n_ped = int(round(n * frac_ped))
    return np.clip(np.concatenate([r.normal(28.0, 6.0, n - n_ped),
                                   r.normal(3.0, 0.8, n_ped)]), 0.1, None)


def test_mode_threshold_minority_floor_matches_the_docstring():
    """Pin the minority fraction the valley guard can actually resolve.

    The docstring quotes this floor, so a test has to hold it honest: it was
    documented as 10% and measured at 0/8 seeds (F-4.5). Detection is reliable
    at 20% and absent at or below 10%.
    """
    found = {frac: sum(rl.suggest_mode_threshold(_bimodal(frac, seed=s),
                                                 unit="mph") is not None
                       for s in range(8))
             for frac in (0.05, 0.10, 0.20)}
    assert found[0.05] == 0, found
    assert found[0.10] == 0, found
    assert found[0.20] == 8, found
