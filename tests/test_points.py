
import numpy as np
import pandas as pd
import pytest

import redlight as rl
from conftest import drive_along_road


def test_basic_load_autodetect(make_points_csv):
    path = make_points_csv([
        {"Longitude": 1.0, "Latitude": 2.0, "Timestamp": "2026-06-01T08:00:00",
         "Speed": 30.0, "track_id": "a"},
    ])
    pts = rl.load_points(path)
    assert list(pts.df["lon"]) == [1.0]
    assert pts.has_traj
    # default unit is mph
    assert pts.df["speed_mps"][0] == pytest.approx(30 * 0.44704)


def test_speed_unit_inferred_from_column_name(make_points_csv):
    """Regression: speed_kph used to be converted with the mph factor."""
    path = make_points_csv([
        {"lon": 0.0, "lat": 0.0, "time": "2026-06-01T08:00:00", "speed_kph": 100.0},
    ])
    pts = rl.load_points(path)
    assert pts.df["speed_mps"][0] == pytest.approx(100 / 3.6)


def test_speed_unit_inferred_from_alias_column_name(make_points_csv):
    """Regression: only the exact 'kph' token was recognized, not aliases
    like 'kmph' -- speed_kmph silently fell back to the mph conversion."""
    path = make_points_csv([
        {"lon": 0.0, "lat": 0.0, "time": "2026-06-01T08:00:00", "speed_kmph": 100.0},
    ])
    pts = rl.load_points(path, speed_col="speed_kmph")
    assert pts.df["speed_mps"][0] == pytest.approx(100 / 3.6)


def test_explicit_unit_beats_column_name_with_warning(make_points_csv):
    path = make_points_csv([
        {"lon": 0.0, "lat": 0.0, "time": "2026-06-01T08:00:00", "speed_kph": 100.0},
    ])
    with pytest.warns(UserWarning, match="contradicts"):
        pts = rl.load_points(path, speed_unit="mph")
    assert pts.df["speed_mps"][0] == pytest.approx(100 * 0.44704)


def test_tz_converts_utc_to_local_clock(make_points_csv):
    """Regression: UTC-stamped data produced peak hours on the UTC clock."""
    path = make_points_csv([
        {"lon": 0.0, "lat": 0.0, "time": "2026-06-01T13:02:11Z"},
    ])
    pts = rl.load_points(path, tz="America/New_York")
    t = pd.to_datetime(pts.df["time"])
    assert t.dt.hour[0] == 9          # 13:02 UTC == 09:02 EDT
    assert getattr(t.dt, "tz", None) is None  # stored naive local


def test_aware_without_tz_warns(make_points_csv):
    path = make_points_csv([
        {"lon": 0.0, "lat": 0.0, "time": "2026-06-01T13:02:11Z"},
    ])
    with pytest.warns(UserWarning, match="timezone-aware"):
        pts = rl.load_points(path)
    assert pd.to_datetime(pts.df["time"]).dt.hour[0] == 13


def test_mixed_offsets_do_not_crash(make_points_csv):
    """Regression: DST-spanning offsets raised 'Mixed timezones detected'."""
    path = make_points_csv([
        {"lon": 0.0, "lat": 0.0, "time": "2026-03-08T01:30:00-05:00"},
        {"lon": 0.0, "lat": 0.0, "time": "2026-03-08T03:30:00-04:00"},
    ])
    pts = rl.load_points(path, tz="America/New_York")
    hours = pd.to_datetime(pts.df["time"]).dt.hour.tolist()
    assert hours == [1, 3]


def test_epoch_with_tz(make_points_csv):
    # 2026-06-01 13:00:00 UTC
    epoch = int(pd.Timestamp("2026-06-01T13:00:00Z").timestamp())
    path = make_points_csv([{"lon": 0.0, "lat": 0.0, "time": epoch}])
    pts = rl.load_points(path, timestamp_unit="s", tz="America/New_York")
    assert pd.to_datetime(pts.df["time"]).dt.hour[0] == 9


def test_epoch_without_tz_warns(make_points_csv):
    """Regression: numeric epochs (UTC-aware by construction) never warned,
    while an equivalent ISO 'Z' string did -- an inconsistent silent gap."""
    epoch = int(pd.Timestamp("2026-06-01T13:00:00Z").timestamp())
    path = make_points_csv([{"lon": 0.0, "lat": 0.0, "time": epoch}])
    with pytest.warns(UserWarning, match="epoch"):
        pts = rl.load_points(path, timestamp_unit="s")
    assert pd.to_datetime(pts.df["time"]).dt.hour[0] == 13


def test_missing_id_rows_dropped_with_warning(make_points_csv):
    """Regression: NaN-id rows silently vanished from HMM output only."""
    rows = drive_along_road(4, traj="a")
    rows[1]["id"] = None
    rows[2]["id"] = None
    path = make_points_csv(rows)
    with pytest.warns(UserWarning, match="missing id"):
        pts = rl.load_points(path)
    assert len(pts) == 2
    assert pts.df["point_id"].tolist() == [0, 1]  # renumbered


def test_derive_speed_geodesic(make_points_csv):
    # 0.001 deg lon at equator ~ 111.32 km / 1000 in 60 s
    path = make_points_csv([
        {"id": "a", "lon": 0.000, "lat": 0.0, "time": "2026-06-01T08:00:00"},
        {"id": "a", "lon": 0.001, "lat": 0.0, "time": "2026-06-01T08:01:00"},
    ])
    pts = rl.load_points(path, derive_speed=True)
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
    pts = rl.load_points(path, derive_speed=True, speed_unit="mph")
    assert not any("contradicts" in str(w.message) for w in recwarn.list)
    expected = 111319.5 / 1000 / 60
    np.testing.assert_allclose(pts.df["speed_mps"], expected, rtol=1e-3)


def test_derive_speed_requires_id(make_points_csv):
    path = make_points_csv([
        {"lon": 0.0, "lat": 0.0, "time": "2026-06-01T08:00:00"},
    ])
    with pytest.raises(ValueError, match="unique-id column"):
        rl.load_points(path, derive_speed=True)


def test_position_time_only_has_no_speed_column(make_points_csv):
    path = make_points_csv(drive_along_road(3))
    pts = rl.load_points(path)
    assert "speed_mps" not in pts.df.columns


def test_save_round_trip_csv_and_geojson(make_points_csv, tmp_path):
    path = make_points_csv([
        {"id": "a", "lon": 0.0, "lat": 0.0, "time": "2026-06-01T08:00:00",
         "speed": 10.0},
        {"id": "a", "lon": 0.001, "lat": 0.0, "time": "2026-06-01T08:01:00",
         "speed": 12.0},
    ])
    pts = rl.load_points(path, speed_unit="mps")
    for name in ("out.csv", "out.geojson"):
        out = rl.save_points(pts, str(tmp_path / name), speed_unit="mps")
        again = rl.load_points(out, speed_unit="mps")
        np.testing.assert_allclose(again.df["speed_mps"], pts.df["speed_mps"])
        np.testing.assert_allclose(again.df["lon"], pts.df["lon"])


# ------------------------------------------------ extra source columns
def test_extra_columns_are_preserved(make_points_csv):
    """A per-point accuracy column must survive loading: derive_speeds'
    pos_accuracy_col is otherwise unreachable and silently degrades to the
    assumed default sigma."""
    path = make_points_csv([
        {"lon": 1.0, "lat": 2.0, "time": "2026-06-01T08:00:00",
         "accuracy": 5.0, "sats": 11, "id": "a"},
    ])
    pts = rl.load_points(path)
    assert pts.df["accuracy"].tolist() == [5.0]
    assert pts.df["sats"].tolist() == [11]


def test_extra_columns_stay_aligned_when_rows_are_dropped(make_points_csv):
    """The alignment case that matters: a dropped row must take its own extra
    values with it, not shift the column against the surviving points."""
    path = make_points_csv([
        {"lon": 1.0, "lat": 2.0, "time": "2026-06-01T08:00:00",
         "accuracy": 5.0, "id": "a"},
        {"lon": 1.0, "lat": 2.0, "time": "not-a-timestamp",
         "accuracy": 99.0, "id": "a"},          # dropped: unparseable time
        {"lon": 1.1, "lat": 2.1, "time": "2026-06-01T08:00:30",
         "accuracy": 7.0, "id": "a"},
    ])
    with pytest.warns(UserWarning, match="Dropped"):
        pts = rl.load_points(path)
    assert pts.df["accuracy"].tolist() == [5.0, 7.0]     # never [5.0, 99.0]


def test_keep_cols_selects_a_subset(make_points_csv):
    path = make_points_csv([
        {"lon": 1.0, "lat": 2.0, "time": "2026-06-01T08:00:00",
         "accuracy": 5.0, "sats": 11, "id": "a"},
    ])
    pts = rl.load_points(path, keep_cols=["accuracy"])
    assert "accuracy" in pts.df.columns
    assert "sats" not in pts.df.columns


def test_keep_cols_empty_restores_lean_frame(make_points_csv):
    path = make_points_csv([
        {"lon": 1.0, "lat": 2.0, "time": "2026-06-01T08:00:00",
         "accuracy": 5.0, "id": "a"},
    ])
    pts = rl.load_points(path, keep_cols=[])
    assert "accuracy" not in pts.df.columns


def test_extra_column_colliding_with_canonical_is_suffixed(make_points_csv):
    """A source column named like a canonical one must not overwrite it."""
    path = make_points_csv([
        {"longitude": 1.0, "latitude": 2.0, "time": "2026-06-01T08:00:00",
         "point_id": "device-A-0007", "id": "a"},
    ])
    pts = rl.load_points(path)
    assert pts.df["point_id"].tolist() == [0]             # canonical wins
    assert pts.df["point_id_src"].tolist() == ["device-A-0007"]


def test_accuracy_reaches_derive_speeds(straight_net, make_points_csv):
    """End-to-end: the whole point of preserving the column."""
    rows = drive_along_road(8, traj="a")
    for i, r in enumerate(rows):
        r["accuracy"] = 4.0 + i * 0.5
    path = make_points_csv(rows)
    pts = rl.load_points(path, id_col="id")
    m = rl.NearestMatcher(straight_net, max_dist=60).match(pts)
    tight = rl.derive_speeds(straight_net, m, pts, pos_accuracy_col="accuracy")
    loose = rl.derive_speeds(straight_net, m, pts, default_pos_sigma_m=40.0)
    # a ~4 m accuracy must yield a tighter speed uncertainty than an assumed 40 m
    assert (tight["intervals"]["speed_sigma_mps"].median()
            < loose["intervals"]["speed_sigma_mps"].median())


def test_mixed_iso8601_spellings_are_all_parsed(make_points_csv):
    """'T' and space separators, and optional fractional seconds, in one column.

    Regression for F-3.6: pandas >= 2.0 infers one format from the first value
    and coerces the rest to NaT, so rows were dropped behind a warning that
    blamed missing data. Every value here is unambiguous ISO 8601.
    """
    rows = [
        {"id": "a", "lon": 0.0005, "lat": 1e-5, "time": "2026-06-01T08:00:00"},
        {"id": "a", "lon": 0.0011, "lat": 1e-5, "time": "2026-06-01 08:00:10"},
        {"id": "a", "lon": 0.0017, "lat": 1e-5, "time": "2026-06-01T08:00:20.5"},
        {"id": "a", "lon": 0.0023, "lat": 1e-5, "time": "2026-06-01 08:00:30"},
    ]
    pts = rl.load_points(make_points_csv(rows), id_col="id")
    assert len(pts.df) == 4
    assert pts.df["time"].notna().all()
    assert pts.df["time"].is_monotonic_increasing


def test_out_of_range_coordinates_are_dropped_with_a_warning(make_points_csv):
    """|lat| > 90 is not a position; it must not reach the geodesy.

    Regression for F-5.2 (and the loader half of F-3.4): such a point loaded
    cleanly, and geodesic_distance then returned a confident wrong distance
    rather than NaN.
    """
    rows = [
        {"lon": 0.0005, "lat": 1e-5, "time": "2026-06-01T08:00:00"},
        {"lon": 181.0, "lat": 91.0, "time": "2026-06-01T08:00:10"},
        {"lon": 0.0007, "lat": 1e-5, "time": "2026-06-01T08:00:20"},
    ]
    with pytest.warns(UserWarning, match="outside the valid range"):
        pts = rl.load_points(make_points_csv(rows))
    assert len(pts.df) == 2
    assert pts.df["lat"].abs().max() <= 90.0
    assert pts.df["point_id"].tolist() == [0, 1]


def test_in_range_coordinates_are_untouched(make_points_csv):
    """The boundary values themselves are valid positions."""
    rows = [{"lon": 180.0, "lat": 90.0, "time": "2026-06-01T08:00:00"},
            {"lon": -180.0, "lat": -90.0, "time": "2026-06-01T08:00:10"}]
    pts = rl.load_points(make_points_csv(rows))
    assert len(pts.df) == 2


def test_implausible_speeds_warn(make_points_csv):
    """Negative and supersonic speeds are wrong input, not data. F-5.3."""
    def rows(v):
        return [{"lon": 0.0005, "lat": 1e-5, "time": "2026-06-01T08:00:00", "sp": v},
                {"lon": 0.0006, "lat": 1e-5, "time": "2026-06-01T08:00:10", "sp": v}]

    with pytest.warns(UserWarning, match="implausible"):
        rl.load_points(make_points_csv(rows(-5.0)), speed_col="sp", speed_unit="mps")
    with pytest.warns(UserWarning, match="implausible"):
        rl.load_points(make_points_csv(rows(400.0)), speed_col="sp", speed_unit="mps")


def test_plausible_speeds_do_not_warn(make_points_csv, recwarn):
    """A normal speed column must stay quiet."""
    rows = [{"lon": 0.0005, "lat": 1e-5, "time": "2026-06-01T08:00:00", "sp": 13.0},
            {"lon": 0.0006, "lat": 1e-5, "time": "2026-06-01T08:00:10", "sp": 0.0}]
    rl.load_points(make_points_csv(rows), speed_col="sp", speed_unit="mps")
    assert not [w for w in recwarn if "implausible" in str(w.message)]


def test_single_fix_mover_is_not_called_unparseable(make_points_csv):
    """A mover with one fix has nothing to difference against. F-5.4."""
    rows = [{"lon": 0.0005, "lat": 1e-5, "time": "2026-06-01T08:00:00", "id": "solo"},
            {"lon": 0.0006, "lat": 1e-5, "time": "2026-06-01T08:00:00", "id": "pair"},
            {"lon": 0.0007, "lat": 1e-5, "time": "2026-06-01T08:00:10", "id": "pair"}]
    with pytest.warns(UserWarning, match="too few fixes"):
        pts = rl.load_points(make_points_csv(rows), id_col="id", derive_speed=True)
    assert pts.df["traj_id"].tolist() == ["pair", "pair"]


def test_mixed_naive_and_aware_timestamps_name_the_real_conflict(make_points_csv):
    """Half the rows vanished behind a 'missing/unparseable' message. F-5.5."""
    rows = [{"lon": 0.0005, "lat": 1e-5, "time": "2026-06-01T08:00:00"},
            {"lon": 0.0006, "lat": 1e-5, "time": "2026-06-01T08:00:10+01:00"}]
    with pytest.warns(UserWarning, match="mix of timezone-aware and timezone-naive"):
        rl.load_points(make_points_csv(rows))
