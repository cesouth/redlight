import numpy as np
import pytest

from roadtraffic.units import SpeedUnit, from_mps, to_mps


def test_parse_aliases():
    assert SpeedUnit.parse("mph") is SpeedUnit.MPH
    assert SpeedUnit.parse("mi/h") is SpeedUnit.MPH
    assert SpeedUnit.parse("km/h") is SpeedUnit.KPH
    assert SpeedUnit.parse("KMH") is SpeedUnit.KPH
    assert SpeedUnit.parse("m/s") is SpeedUnit.MPS
    assert SpeedUnit.parse(SpeedUnit.MPS) is SpeedUnit.MPS


def test_parse_unknown_raises():
    with pytest.raises(ValueError, match="Unknown speed unit"):
        SpeedUnit.parse("furlongs")


def test_conversion_exact():
    assert to_mps(1.0, "mph") == pytest.approx(0.44704)
    assert to_mps(3.6, "kph") == pytest.approx(1.0)
    assert to_mps(5.0, "mps") == 5.0


def test_round_trip_arrays():
    x = np.array([0.0, 1.0, 30.0])
    for unit in ("mph", "kph", "mps"):
        np.testing.assert_allclose(to_mps(from_mps(x, unit), unit), x)
