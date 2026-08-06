import pandas as pd
import pytest

import redlight as rl


def _annotate(net, *, peak=4.0, offpeak=16.0):
    rows = []
    for h, sp in ((8, peak), (22, offpeak)):
        rows += [{"edge_id": 0, "speed_mps": sp,
                  "time": pd.Timestamp(f"2026-06-01 {h:02d}:10:{k:02d}")}
                 for k in range(3)]
    rl.assign_segment_speeds(net, pd.DataFrame(rows), peak_hours=[8],
                             offpeak_hours=[22])


def test_to_geojson_units_and_alias(straight_net):
    _annotate(straight_net)
    fc = rl.to_geojson(straight_net, speed_unit="km/h")
    props = fc["features"][0]["properties"]
    assert props["speed_unit"] == "kph"
    assert props["speed"] == pytest.approx(10.0 * 3.6)  # overall = mean(4,16)


def test_to_geojson_unknown_unit_raises(straight_net):
    """Regression: unknown units were silently exported as m/s."""
    _annotate(straight_net)
    with pytest.raises(ValueError, match="Unknown speed unit"):
        rl.to_geojson(straight_net, speed_unit="furlongs")


def test_to_geojson_period_export(straight_net):
    _annotate(straight_net)
    peak = rl.to_geojson(straight_net, period="peak", speed_unit="mps")
    off = rl.to_geojson(straight_net, period="offpeak", speed_unit="mps")
    assert peak["features"][0]["properties"]["speed"] == pytest.approx(4.0)
    assert off["features"][0]["properties"]["speed"] == pytest.approx(16.0)
    with pytest.raises(ValueError, match="period"):
        rl.to_geojson(straight_net, period="rush")


def test_to_geojson_travel_time_consistent(straight_net):
    """Regression: merged speed and datas[0] travel time used to contradict."""
    _annotate(straight_net)
    props = rl.to_geojson(straight_net, speed_unit="mps")["features"][0]["properties"]
    assert props["travel_time_s"] == pytest.approx(
        props["length_m"] / props["speed"])


def test_to_geojson_one_feature_per_road(straight_net):
    _annotate(straight_net)
    fc = rl.to_geojson(straight_net)  # non-directional
    assert len(fc["features"]) == 1
    assert sorted(fc["features"][0]["properties"]["edge_ids"]) == [0, 1]
    fc_dir = rl.to_geojson(straight_net, directional=True)
    assert len(fc_dir["features"]) == 2


def test_to_geojson_zero_speed_is_observed_not_nodata(straight_net):
    """Regression: 0.0 m/s (gridlock) was treated identically to no data."""
    rows = [{"edge_id": 0, "speed_mps": 0.0,
             "time": pd.Timestamp(f"2026-06-01 08:10:{k:02d}")} for k in range(3)]
    rl.assign_segment_speeds(straight_net, pd.DataFrame(rows), peak_hours=[8],
                             offpeak_hours=[22])
    fc = rl.to_geojson(straight_net, speed_unit="mps")
    props = fc["features"][0]["properties"]
    assert props["has_speed"] is True
    assert props["speed"] == pytest.approx(0.0)
    assert props["travel_time_s"] is None  # infinite/undefined, not ZeroDivisionError
    assert fc["properties"]["n_edges_observed"] == 1


def test_to_geojson_keep_tags_collision_renamed(straight_net):
    """Regression: a source tag named e.g. 'speed' silently overwrote the
    computed numeric speed property."""
    _annotate(straight_net)
    for _u, _v, data in straight_net.graph.edges(data=True):
        data["speed"] = "posted 30mph zone"
    fc = rl.to_geojson(straight_net, keep_tags=["speed"], speed_unit="mps")
    props = fc["features"][0]["properties"]
    assert isinstance(props["speed"], float)  # computed value untouched
    assert props["speed_src"] == "posted 30mph zone"


def test_to_geojson_unobserved_edges(grid_net):
    fc = rl.to_geojson(grid_net, speed_unit="mps")
    assert fc["properties"]["n_edges_observed"] == 0
    assert all(p["properties"]["has_speed"] is False for p in fc["features"])
    assert all(p["properties"]["speed"] is None for p in fc["features"])


def test_to_geojson_writes_file(straight_net, tmp_path):
    _annotate(straight_net)
    out = tmp_path / "map.geojson"
    rl.to_geojson(straight_net, str(out))
    assert out.exists() and out.stat().st_size > 0


def test_plot_speed_map_smoke(straight_net, tmp_path):
    pytest.importorskip("matplotlib")
    _annotate(straight_net)
    out = tmp_path / "map.png"
    fig = rl.plot_speed_map(straight_net, str(out), period="peak")
    assert out.exists() and out.stat().st_size > 0
    import matplotlib.pyplot as plt
    plt.close(fig)
