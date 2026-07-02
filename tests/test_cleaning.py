import numpy as np
import pandas as pd
import pytest

import roadtraffic as rt


def _matched_frame(speeds_mps, *, edge_id=0, traj="a", t0="2026-06-01 08:00:00"):
    t0 = pd.Timestamp(t0)
    n = len(speeds_mps)
    return pd.DataFrame({
        "point_id": range(n),
        "edge_id": edge_id,
        "snap_dist_m": 1.0,
        "lon": np.linspace(0, 0.001 * n, n),
        "lat": 0.0,
        "time": [t0 + pd.Timedelta(seconds=10 * k) for k in range(n)],
        "speed_mps": speeds_mps,
        "traj_id": traj,
    })


def test_filter_by_speed_bounds():
    df = _matched_frame([0.0, 5.0, 10.0, 50.0])
    out = rt.filter_by_speed(df, min_speed=1.0, max_speed=20.0, unit="mps")
    assert out["speed_mps"].tolist() == [5.0, 10.0]


def test_filter_by_speed_requires_speed_column():
    df = _matched_frame([1.0]).drop(columns=["speed_mps"])
    with pytest.raises(ValueError, match="speed_mps"):
        rt.filter_by_speed(df)


def test_filter_by_speed_drops_unmatched():
    df = _matched_frame([5.0, 6.0])
    df.loc[0, "edge_id"] = -1
    out = rt.filter_by_speed(df, unit="mps")
    assert out["edge_id"].tolist() == [0]


def test_mad_outliers_removed():
    # jitter keeps MAD > 0 (identical values would trigger the degenerate
    # MAD=0 keep-everything fallback)
    speeds = [10.0 + 0.1 * (k % 5) for k in range(20)] + [200.0]
    out = rt.filter_by_speed(_matched_frame(speeds), unit="mps",
                             mad_outliers=True)
    assert 200.0 not in out["speed_mps"].values
    assert len(out) == 20


def test_trajectory_filter_keeps_slow_moving_traffic():
    # creeping at ~1 m/s: slow but making ground -> kept, low speed and all
    n = 20
    lon = np.cumsum(np.full(n, 10.0 / 111320))  # 10 m per 10 s fix
    df = _matched_frame([1.0] * n)
    df["lon"] = lon
    out = rt.filter_trajectory_speed(df)
    assert len(out) == n


def test_trajectory_filter_drops_parked_dwell():
    # 10 fixes moving (30 m per 10 s), then 30 fixes parked for ~300 s
    n_move, n_park = 10, 30
    lon = list(np.cumsum(np.full(n_move, 30.0 / 111320)))
    lon += [lon[-1] + 2e-7 * k for k in range(n_park)]  # ~2 cm jitter
    df = _matched_frame([3.0] * n_move + [0.01] * n_park)
    df["lon"] = lon
    out = rt.filter_trajectory_speed(df, dwell_radius_m=25, dwell_min_s=120)
    # the parked block is gone; the final moving fix sits at the parking
    # position, so the anchor-based dwell absorbs it too
    assert len(out) == n_move - 1
    assert (out["speed_mps"] == 3.0).all()
