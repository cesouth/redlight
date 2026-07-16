
import numpy as np
import pandas as pd
import pytest

import roadtraffic as rt
from conftest import drive_along_road


def test_basic_load_autodetect(make_points_csv):
    path = make_points_csv([
        {"Longitude": 1.0, "Latitude": 2.0, "Timestamp": "2026-06-01T08:00:00",
         "Speed": 30.0, "track_id": "a"},
    ])
    pts = rt.load_points(path)
    assert list(pts.df["lon"]) == [1.0]
    assert pts.has_traj
    # default unit is mph
    assert pts.df["speed_mps"][0] == pytest.approx(30 * 0.44704)


def test_speed_unit_inferred_from_column_name(make_points_csv):
    """Regression: speed_kph used to be converted with the mph factor."""
    path = make_points_csv([
        {"lon": 0.0, "lat": 0.0, "time": "2026-06-01T08:00:00", "speed_kph": 100.0},
    ])
    pts = rt.load_points(path)
    assert pts.df["speed_mps"][0] == pytest.approx(100 / 3.6)


def test_speed_unit_inferred_from_alias_column_name(make_points_csv):
    """Regression: only the exact 'kph' token was recognized, not aliases
    like 'kmph' -- speed_kmph silently fell back to the mph conversion."""
    path = make_points_csv([
        {"lon": 0.0, "lat": 0.0, "time": "2026-06-01T08:00:00", "speed_kmph": 100.0},
    ])
    pts = rt.load_points(path, speed_col="speed_kmph")
    assert pts.df["speed_mps"][0] == pytest.approx(100 / 3.6)


def test_explicit_unit_beats_column_name_with_warning(make_points_csv):
    path = make_points_csv([
        {"lon": 0.0, "lat": 0.0, "time": "2026-06-01T08:00:00", "speed_kph": 100.0},
    ])
    with pytest.warns(UserWarning, match="contradicts"):
        pts = rt.load_points(path, speed_unit="mph")
    assert pts.df["speed_mps"][0] == pytest.approx(100 * 0.44704)


def test_tz_converts_utc_to_local_clock(make_points_csv):
    """Regression: UTC-stamped data produced peak hours on the UTC clock."""
    path = make_points_csv([
        {"lon": 0.0, "lat": 0.0, "time": "2026-06-01T13:02:11Z"},
    ])
    pts = rt.load_points(path, tz="America/New_York")
    t = pd.to_datetime(pts.df["time"])
    assert t.dt.hour[0] == 9          # 13:02 UTC == 09:02 EDT
    assert getattr(t.dt, "tz", None) is None  # stored naive local


def test_aware_without_tz_warns(make_points_csv):
    path = make_points_csv([
        {"lon": 0.0, "lat": 0.0, "time": "2026-06-01T13:02:11Z"},
    ])
    with pytest.warns(UserWarning, match="timezone-aware"):
        pts = rt.load_points(path)
    assert pd.to_datetime(pts.df["time"]).dt.hour[0] == 13


def test_mixed_offsets_do_not_crash(make_points_csv):
    """Regression: DST-spanning offsets raised 'Mixed timezones detected'."""
    path = make_points_csv([
        {"lon": 0.0, "lat": 0.0, "time": "2026-03-08T01:30:00-05:00"},
        {"lon": 0.0, "lat": 0.0, "time": "2026-03-08T03:30:00-04:00"},
    ])
    pts = rt.load_points(path, tz="America/New_York")
    hours = pd.to_datetime(pts.df["time"]).dt.hour.tolist()
    assert hours == [1, 3]


def test_epoch_with_tz(make_points_csv):
    # 2026-06-01 13:00:00 UTC
    epoch = int(pd.Timestamp("2026-06-01T13:00:00Z").timestamp())
    path = make_points_csv([{"lon": 0.0, "lat": 0.0, "time": epoch}])
    pts = rt.load_points(path, timestamp_unit="s", tz="America/New_York")
    assert pd.to_datetime(pts.df["time"]).dt.hour[0] == 9


def test_epoch_without_tz_warns(make_points_csv):
    """Regression: numeric epochs (UTC-aware by construction) never warned,
    while an equivalent ISO 'Z' string did -- an inconsistent silent gap."""
    epoch = int(pd.Timestamp("2026-06-01T13:00:00Z").timestamp())
    path = make_points_csv([{"lon": 0.0, "lat": 0.0, "time": epoch}])
    with pytest.warns(UserWarning, match="epoch"):
        pts = rt.load_points(path, timestamp_unit="s")
    assert pd.to_datetime(pts.df["time"]).dt.hour[0] == 13


def test_missing_id_rows_dropped_with_warning(make_points_csv):
    """Regression: NaN-id rows silently vanished from HMM output only."""
    rows = drive_along_road(4, traj="a")
    rows[1]["id"] = None
    rows[2]["id"] = None
    path = make_points_csv(rows)
    with pytest.warns(UserWarning, match="missing id"):
        pts = rt.load_points(path)
    assert len(pts) == 2
    assert pts.df["point_id"].tolist() == [0, 1]  # renumbered


def test_derive_speed_geodesic(make_points_csv):
    # 0.001 deg lon at equator ~ 111.32 km / 1000 in 60 s
    path = make_points_csv([
        {"id": "a", "lon": 0.000, "lat": 0.0, "time": "2026-06-01T08:00:00"},
        {"id": "a", "lon": 0.001, "lat": 0.0, "time": "2026-06-01T08:01:00"},
    ])
    pts = rt.load_points(path, derive_speed=True)
    expected = 111319.5 / 1000 / 60  # ~1.855 m/s
    np.testing.assert_allclose(pts.df["speed_mps"], expected, rtol=1e-3)


def test_derive_speed_ignores_unused_speed_column_unit(make_points_csv, recwarn):
    """Regression: a stale/irrelevant speed column's inferred unit could still
    fire a 'contradicts' warning even though derive_speed=True never reads it."""
    path = make_points_csv([
        {"id": "a", "lon": 0.000, "lat": 0.0, "time": "2026-06-01T08:00:00",
         "speed_kph": 999.0},
        {"id": "a", "lon": 0.001, "lat": 0.0, "time": "2026-06-01T08:01:00",
         "speed_kph": 999.0},
    ])
    pts = rt.load_points(path, derive_speed=True, speed_unit="mph")
    assert not any("contradicts" in str(w.message) for w in recwarn.list)
    expected = 111319.5 / 1000 / 60
    np.testing.assert_allclose(pts.df["speed_mps"], expected, rtol=1e-3)


def test_derive_speed_requires_id(make_points_csv):
    path = make_points_csv([
        {"lon": 0.0, "lat": 0.0, "time": "2026-06-01T08:00:00"},
    ])
    with pytest.raises(ValueError, match="unique-id column"):
        rt.load_points(path, derive_speed=True)


def test_position_time_only_has_no_speed_column(make_points_csv):
    path = make_points_csv(drive_along_road(3))
    pts = rt.load_points(path)
    assert "speed_mps" not in pts.df.columns


def test_save_round_trip_csv_and_geojson(make_points_csv, tmp_path):
    path = make_points_csv([
        {"id": "a", "lon": 0.0, "lat": 0.0, "time": "2026-06-01T08:00:00",
         "speed": 10.0},
        {"id": "a", "lon": 0.001, "lat": 0.0, "time": "2026-06-01T08:01:00",
         "speed": 12.0},
    ])
    pts = rt.load_points(path, speed_unit="mps")
    for name in ("out.csv", "out.geojson"):
        out = rt.save_points(pts, str(tmp_path / name), speed_unit="mps")
        again = rt.load_points(out, speed_unit="mps")
        np.testing.assert_allclose(again.df["speed_mps"], pts.df["speed_mps"])
        np.testing.assert_allclose(again.df["lon"], pts.df["lon"])
