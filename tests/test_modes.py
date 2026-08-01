import numpy as np
import pandas as pd
import pytest

import roadtraffic as rt


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
    feat = rt.mover_features(obs_frame({"a": [10.0, 20.0, 30.0], "b": [1.0, 1.5]}),
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
    feat = rt.mover_features(long_form, unit="mps")
    assert feat.loc["a", "n_intervals"] == 2
    assert feat.loc["a", "speed_median_mps"] == pytest.approx(11.0)


def test_mover_features_converts_to_the_requested_unit():
    feat = rt.mover_features(obs_frame({"a": [10.0] * 4}), unit="mph")
    assert "speed_p85_mph" in feat.columns
    assert feat.loc["a", "speed_p85_mph"] == pytest.approx(10.0 / 0.44704, rel=1e-6)


def test_mover_features_missing_columns_raises():
    with pytest.raises(ValueError, match="traj_id"):
        rt.mover_features(pd.DataFrame({"speed_mps": [1.0]}))


def test_mover_features_empty_input_returns_typed_empty_frame():
    empty = pd.DataFrame({"interval_id": [], "traj_id": [], "speed_mps": [],
                          "distance_m": []})
    feat = rt.mover_features(empty, unit="mps")
    assert len(feat) == 0
    assert "speed_p85_mps" in feat.columns


def test_suggest_threshold_finds_the_valley_between_two_humps():
    rng = np.random.default_rng(0)
    walkers = rng.normal(1.4, 0.15, 90)
    vehicles = np.clip(rng.normal(12.0, 3.0, 200), 5.0, None)
    t = rt.suggest_mode_threshold(np.concatenate([walkers, vehicles]), unit="mps")
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
    assert rt.suggest_mode_threshold(speeds, unit="mps") is None


def test_suggest_threshold_needs_enough_movers():
    assert rt.suggest_mode_threshold(np.full(10, 1.4), unit="mps") is None


def test_suggest_threshold_accepts_a_series():
    rng = np.random.default_rng(0)
    speeds = np.concatenate([rng.normal(1.4, 0.15, 90),
                             np.clip(rng.normal(12.0, 3.0, 200), 5.0, None)])
    s = pd.Series(speeds, index=[f"m{i}" for i in range(len(speeds))])
    assert rt.suggest_mode_threshold(s, unit="mps") is not None


def test_suggest_threshold_is_unit_consistent():
    """The same population must yield the same physical speed whatever unit it
    is expressed in. Every existing test uses mps, where both conversions are
    the identity, so a swapped to_mps/from_mps would go undetected."""
    rng = np.random.default_rng(0)
    mps = np.concatenate([rng.normal(1.4, 0.15, 90),
                          np.clip(rng.normal(12.0, 3.0, 200), 5.0, None)])
    t_mps = rt.suggest_mode_threshold(mps, unit="mps")
    t_mph = rt.suggest_mode_threshold(mps / 0.44704, unit="mph")
    assert t_mps is not None and t_mph is not None
    # same physical speed, expressed two ways
    assert t_mph * 0.44704 == pytest.approx(t_mps, rel=1e-6)


def test_classify_labels_pedestrians_and_vehicles():
    obs = obs_frame({"walker": [1.3, 1.5, 1.4, 1.6],
                     "car": [3.0, 18.0, 20.0, 22.0]})
    m = rt.classify_movers(obs, threshold=3.0, unit="mps")
    assert m.loc["walker", "mode"] == "pedestrian"
    assert m.loc["car", "mode"] == "vehicle"


def test_a_congested_vehicle_is_still_a_vehicle():
    """It crawls for most of its trip but shows one free-flowing stretch, and
    the 85th percentile is chosen precisely so that stretch is visible."""
    obs = obs_frame({"stuck": [1.5, 1.6, 1.4, 1.5, 1.6, 1.5, 14.0, 15.0]})
    m = rt.classify_movers(obs, threshold=3.0, unit="mps")
    assert m.loc["stuck", "mode"] == "vehicle"


def test_unknown_comes_from_too_few_intervals():
    m = rt.classify_movers(obs_frame({"brief": [20.0, 21.0]}),
                           threshold=3.0, min_intervals=3, unit="mps")
    assert m.loc["brief", "mode"] == "unknown"


def test_unknown_comes_from_too_little_distance():
    m = rt.classify_movers(obs_frame({"short": [20.0] * 4}, distance_m=10.0),
                           threshold=3.0, min_distance_m=500.0, unit="mps")
    assert m.loc["short", "mode"] == "unknown"


def test_threshold_is_read_in_the_requested_unit():
    obs = obs_frame({"a": [1.4] * 5, "b": [12.0] * 5})
    mph = rt.classify_movers(obs, threshold=6.0, unit="mph")
    kph = rt.classify_movers(obs, threshold=6.0 * 1.609344, unit="kph")
    assert list(mph["mode"]) == list(kph["mode"])
    assert list(mph["mode"]) == ["pedestrian", "vehicle"]


def test_auto_threshold_raises_rather_than_guessing():
    rng = np.random.default_rng(2)
    obs = obs_frame({f"v{i}": [float(s)] * 4
                     for i, s in enumerate(rng.normal(12.0, 2.0, 60))})
    with pytest.raises(ValueError, match="walking"):
        rt.classify_movers(obs, threshold="auto", unit="mps")


def test_auto_threshold_works_when_there_are_walkers():
    rng = np.random.default_rng(0)
    specs = {f"p{i}": [float(v)] * 4
             for i, v in enumerate(rng.normal(1.4, 0.15, 90))}
    specs.update({f"v{i}": [float(v)] * 4
                  for i, v in enumerate(np.clip(rng.normal(12.0, 3.0, 200), 5.0, None))})
    m = rt.classify_movers(obs_frame(specs), threshold="auto", unit="mps")
    assert set(m["mode"]) == {"pedestrian", "vehicle"}
    assert (m["mode"] == "pedestrian").sum() == 90


def test_bad_threshold_string_raises():
    with pytest.raises(ValueError, match="'auto'"):
        rt.classify_movers(obs_frame({"a": [1.0] * 4}), threshold="fast", unit="mps")


def test_filter_keeps_every_observation_of_a_kept_mover():
    """THE invariant. A vehicle that crawls through congestion keeps its slow
    rows. That is the whole difference from filter_by_speed, which would delete
    them and take the congestion finding with them."""
    obs = obs_frame({"stuck": [1.5, 1.4, 1.5, 1.6, 14.0, 15.0],
                     "walker": [1.3, 1.4, 1.5, 1.4]})
    movers = rt.classify_movers(obs, threshold=3.0, unit="mps")
    out = rt.filter_by_mode(obs, movers)
    assert set(out["traj_id"]) == {"stuck"}
    assert len(out[out["speed_mps"] < 2.0]) == 4


def test_filter_excludes_unknown_by_default_but_can_include_it():
    obs = obs_frame({"car": [20.0] * 4, "brief": [20.0, 21.0]})
    movers = rt.classify_movers(obs, threshold=3.0, min_intervals=3, unit="mps")
    assert set(rt.filter_by_mode(obs, movers)["traj_id"]) == {"car"}
    both = rt.filter_by_mode(obs, movers, keep=("vehicle", "unknown"))
    assert set(both["traj_id"]) == {"car", "brief"}


def test_filter_warns_instead_of_exiting_when_everything_is_removed():
    obs = obs_frame({"walker": [1.4] * 4})
    movers = rt.classify_movers(obs, threshold=3.0, unit="mps")
    with pytest.warns(UserWarning, match="removed every"):
        out = rt.filter_by_mode(obs, movers)
    assert len(out) == 0


def test_filter_accepts_a_single_mode_string():
    obs = obs_frame({"car": [20.0] * 4, "walker": [1.4] * 4})
    movers = rt.classify_movers(obs, threshold=3.0, unit="mps")
    assert set(rt.filter_by_mode(obs, movers, keep="pedestrian")["traj_id"]) \
        == {"walker"}


def test_filter_requires_a_classification_table():
    obs = obs_frame({"a": [1.0] * 4})
    with pytest.raises(ValueError, match="classify_movers"):
        rt.filter_by_mode(obs, pd.DataFrame({"n_intervals": [4]}))


def test_verdicts_match_between_intervals_and_edge_observations():
    """A long-format frame must classify identically to the interval frame it
    was expanded from -- the dedup path in mover_features."""
    obs = obs_frame({"stuck": [1.5, 1.4, 14.0], "walker": [1.3, 1.4, 1.5, 1.4]})
    slow = obs[obs["speed_mps"] < 2.0]
    long_form = pd.concat([obs] + [slow] * 3, ignore_index=True)
    a = rt.classify_movers(obs, threshold=3.0, min_intervals=3, unit="mps")
    b = rt.classify_movers(long_form, threshold=3.0, min_intervals=3, unit="mps")
    assert a["mode"].to_dict() == b["mode"].to_dict()
