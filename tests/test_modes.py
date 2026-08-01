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
