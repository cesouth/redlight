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
