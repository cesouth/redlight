import numpy as np
import pandas as pd
import pytest

import redlight as rl


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
    out = rl.filter_by_speed(df, min_speed=1.0, max_speed=20.0, unit="mps")
    assert out["speed_mps"].tolist() == [5.0, 10.0]


def test_filter_by_speed_requires_speed_column():
    df = _matched_frame([1.0]).drop(columns=["speed_mps"])
    with pytest.raises(ValueError, match="speed_mps"):
        rl.filter_by_speed(df)


def test_filter_by_speed_drops_unmatched():
    df = _matched_frame([5.0, 6.0])
    df.loc[0, "edge_id"] = -1
    out = rl.filter_by_speed(df, unit="mps")
    assert out["edge_id"].tolist() == [0]


def test_mad_outliers_removed():
    # jitter keeps MAD > 0 (identical values would trigger the degenerate
    # MAD=0 keep-everything fallback)
    speeds = [10.0 + 0.1 * (k % 5) for k in range(20)] + [200.0]
    out = rl.filter_by_speed(_matched_frame(speeds), unit="mps",
                             mad_outliers=True)
    assert 200.0 not in out["speed_mps"].values
    assert len(out) == 20


def test_mad_outliers_nan_speed_does_not_wipe_dataset():
    """Regression: a single NaN speed poisoned np.median, failing every row."""
    speeds = [10.0 + 0.1 * (k % 5) for k in range(20)] + [np.nan]
    out = rl.filter_by_speed(_matched_frame(speeds), unit="mps",
                             mad_outliers=True)
    assert len(out) == 20  # the 19 finite non-outliers, minus the NaN row


def test_trajectory_filter_keeps_slow_moving_traffic():
    # creeping at ~1 m/s: slow but making ground -> kept, low speed and all
    n = 20
    lon = np.cumsum(np.full(n, 10.0 / 111320))  # 10 m per 10 s fix
    df = _matched_frame([1.0] * n)
    df["lon"] = lon
    out = rl.filter_trajectory_speed(df)
    assert len(out) == n


def test_trajectory_filter_drops_parked_dwell():
    # 10 fixes moving (30 m per 10 s), then 30 fixes parked for ~300 s
    n_move, n_park = 10, 30
    lon = list(np.cumsum(np.full(n_move, 30.0 / 111320)))
    lon += [lon[-1] + 2e-7 * k for k in range(n_park)]  # ~2 cm jitter
    df = _matched_frame([3.0] * n_move + [0.01] * n_park)
    df["lon"] = lon
    out = rl.filter_trajectory_speed(df, dwell_radius_m=25, dwell_min_s=120)
    # the parked block is gone; the final moving fix sits at the parking
    # position, so the anchor-based dwell absorbs it too
    assert len(out) == n_move - 1
    assert (out["speed_mps"] == 3.0).all()


def test_dwell_run_that_never_leaves_the_radius_reaches_the_end():
    """The block scan finds the first point past the radius with argmax, which
    returns 0 on an all-False block -- i.e. 'the very first point already left'.
    A trajectory that never leaves the radius must still dwell to its last fix.
    """
    from redlight.cleaning import _dwell_mask

    n = 500  # longer than the probe window and several doubling blocks
    rng = np.random.default_rng(3)
    d = 5.0 / 111319.5
    lon = 15.0 + rng.uniform(-d, d, n)
    lat = 50.0 + rng.uniform(-d, d, n)
    t = np.arange(n, dtype=float) * 10.0
    assert _dwell_mask(lon, lat, t, 25.0, 120.0).all()


@pytest.mark.parametrize("run_len", [1, 7, 8, 9, 31, 32, 33, 100])
def test_dwell_run_ends_exactly_where_the_radius_does(run_len):
    """Run boundaries must not shift with the scan's probe/block sizes."""
    from redlight.cleaning import _dwell_mask

    rng = np.random.default_rng(run_len)
    d = 5.0 / 111319.5
    lon = np.concatenate([15.0 + rng.uniform(-d, d, run_len + 1), [15.5]])
    lat = np.concatenate([50.0 + rng.uniform(-d, d, run_len + 1), [50.0]])
    t = np.arange(run_len + 2, dtype=float) * 30.0
    mask = _dwell_mask(lon, lat, t, 25.0, 30.0)
    assert mask[:run_len + 1].all()
    assert not mask[run_len + 1]


def test_dwell_tolerates_a_missing_coordinate():
    """A NaN fix ends the run it lands in rather than failing the whole clean:
    the geodesic raises on a non-finite input instead of returning one, so the
    coordinates have to be screened before they reach it."""
    from redlight.cleaning import _dwell_mask

    rng = np.random.default_rng(11)
    d = 5.0 / 111319.5
    lon = 15.0 + rng.uniform(-d, d, 40)
    lat = 50.0 + rng.uniform(-d, d, 40)
    lon[20] = np.nan
    t = np.arange(40, dtype=float) * 10.0
    mask = _dwell_mask(lon, lat, t, 25.0, 120.0)
    assert mask[:20].all()          # the run ends at the missing fix
    assert not mask[20]             # which is itself not part of any dwell
    assert mask[21:].all()          # and a fresh run starts after it
