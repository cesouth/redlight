"""Generate the sample dataset every other example uses. Run this first.

    python examples/00_setup/generate_sample_data.py

Writes into ``examples/sample_data/``:

    network.geojson   a small road grid with three road classes and posted
                      ``maxspeed`` tags, so congestion-vs-limit analysis works
    points.csv        GPS fixes with position, time, a mover id and a per-fix
                      horizontal accuracy in metres -- and deliberately **no
                      speed column**

Why no speed column
-------------------
Plenty of real feeds carry position and time but no usable speed: the receiver
never logged it, or logged an instantaneous value too noisy to trust. That is
the harder and more common case, so it is what the examples demonstrate.
:func:`roadtraffic.derive_speeds` reconstructs speed from on-road displacement
after matching, which is more robust than a per-fix reading anyway.

What is planted in the data
---------------------------
Ground truth is deliberate so the examples' output can be checked:

* **Free-flow speed by road class** -- the arterial really is faster than the
  residential streets, and each is tagged with a matching posted limit.
* **Rush hours.** Weekday 07-09 and 16-18 run at roughly a third of free flow;
  midday around three quarters; overnight free.
* **A quieter weekend**, so weekday/weekend comparison has something to find.
* **Pedestrians.** About a fifth of the movers are people on foot at walking
  pace, on the same streets. They are what ``03_mode_screening`` exists to
  remove, and they carry a ``mode`` column recording the truth so the example
  can score its own accuracy. Real feeds will not have that column -- it is a
  teaching aid, not an input the library uses.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "sample_data")

rng = np.random.default_rng(20260801)

LON0, LAT0 = -77.300, 38.800
NROW, NCOL = 4, 5
DLAT, DLON = 0.0080, 0.0090        # ~890 m N-S, ~780 m E-W here

# Free-flow speed by class, with the posted limit that matches it.
CLASSES = {
    "primary":     {"maxspeed": "55 mph", "free_mps": 24.6},
    "secondary":   {"maxspeed": "45 mph", "free_mps": 20.1},
    "residential": {"maxspeed": "25 mph", "free_mps": 11.2},
}
EW_NAMES = ["Beltway Reach", "Commerce Ave", "Maple St", "Orchard Ln"]
NS_NAMES = ["Depot Rd", "Mill St", "Kingsway", "Quarry Rd", "Foundry St"]

KX = math.cos(math.radians(LAT0))
DT = 15.0                          # seconds between fixes
DATES = pd.date_range("2026-06-01", periods=7, freq="D")   # Mon..Sun


def node(r: int, c: int) -> tuple[float, float]:
    return (LON0 + c * DLON, LAT0 + r * DLAT)


def build_network(path: str) -> list:
    """A grid of named two-way streets; row 0 is the arterial.

    Each road is emitted as one feature **per block between intersections**,
    which is how OSM extracts are actually shaped. Emitting a whole road as a
    single LineString would leave crossings as mere geometric intersections
    with no shared node, so the graph would be far sparser than it looks and
    routing between two arbitrary points would fail.

    The returned list keeps each road's *full* polyline, which the trajectory
    generator walks along; the split only concerns the network's topology.
    """
    feats, roads = [], []
    for r in range(NROW):
        cls = "primary" if r == 0 else ("secondary" if r == 2 else "residential")
        coords = [list(node(r, c)) for c in range(NCOL)]
        for a, b in zip(coords, coords[1:]):
            feats.append(_feature([a, b], EW_NAMES[r], cls))
        roads.append((EW_NAMES[r], cls, coords))
    for c in range(NCOL):
        cls = "secondary" if c in (0, NCOL - 1) else "residential"
        coords = [list(node(r, c)) for r in range(NROW)]
        for a, b in zip(coords, coords[1:]):
            feats.append(_feature([a, b], NS_NAMES[c], cls))
        roads.append((NS_NAMES[c], cls, coords))
    with open(path, "w") as fh:
        json.dump({"type": "FeatureCollection", "features": feats}, fh, indent=0)
    return roads


def _feature(coords, name, cls):
    return {
        "type": "Feature",
        "properties": {"name": name, "highway": cls,
                       "maxspeed": CLASSES[cls]["maxspeed"], "oneway": "no"},
        "geometry": {"type": "LineString", "coordinates": coords},
    }


def congestion(hour: int, weekend: bool) -> float:
    """Multiplier on free-flow speed. Weekday rush hours bite hardest."""
    if weekend:
        return 0.95 if 11 <= hour <= 17 else 1.0
    if 7 <= hour <= 9:
        return 0.34
    if 16 <= hour <= 18:
        return 0.40
    if 10 <= hour <= 15:
        return 0.78
    return 1.0


def _cumulative(coords):
    cum = [0.0]
    for a, b in zip(coords, coords[1:]):
        cum.append(cum[-1] + math.hypot((b[0] - a[0]) * KX * 111320.0,
                                        (b[1] - a[1]) * 111320.0))
    return cum


def point_at(coords, cum, dist_m):
    """The point exactly ``dist_m`` along the polyline.

    Sampling by arc length rather than by fraction is what makes the planted
    speed exact: each fix sits v*dt metres further on regardless of how long
    the road happens to be.
    """
    if dist_m <= 0:
        return tuple(coords[0])
    if dist_m >= cum[-1]:
        return tuple(coords[-1])
    for i in range(len(cum) - 1):
        if cum[i + 1] >= dist_m:
            span = cum[i + 1] - cum[i]
            t = (dist_m - cum[i]) / span if span else 0.0
            a, b = coords[i], coords[i + 1]
            return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
    return tuple(coords[-1])


def _track(rows, roads, tid, speed_mps, hour, date, mode, acc_scale, n_fix_range,
           offset_m=0.0):
    """Walk one mover along one road at a fixed speed, emitting fixes."""
    _, _, coords = roads[rng.integers(len(roads))]
    cum = _cumulative(coords)
    length = cum[-1]
    step = speed_mps * DT
    n_fix = min(int(rng.integers(*n_fix_range)), int(length / step))
    if n_fix < 4:
        return False
    reverse = bool(rng.integers(2))
    start = float(rng.uniform(0.0, max(length - n_fix * step, 0.0)))
    t0 = date + pd.Timedelta(hours=hour, minutes=int(rng.integers(60)))
    off_deg = offset_m / 111320.0
    for i in range(n_fix + 1):
        travelled = start + i * step
        if reverse:
            travelled = length - travelled
        lon, lat = point_at(coords, cum, travelled)
        acc = float(np.clip(rng.gamma(4.0, acc_scale), 3.0, 45.0))
        sd = acc / 111320.0
        # Offset-aware ISO8601, as a real telematics feed emits. Naive local
        # timestamps would be read as UTC when a --tz is supplied, silently
        # shifting every hour-of-day statistic by the UTC offset.
        stamp = (t0 + pd.Timedelta(seconds=i * DT)).tz_localize("America/New_York")
        rows.append({
            "device_id": tid,
            "timestamp": stamp.isoformat(),
            "longitude": lon + (off_deg + float(rng.normal(0, sd))) / KX,
            "latitude": lat + off_deg + float(rng.normal(0, sd)),
            "accuracy_m": round(acc, 1),
            "mode": mode,
        })
    return True


def build_points(path: str, roads: list) -> pd.DataFrame:
    rows, tid = [], 0
    hours = [7, 8, 8, 9, 12, 13, 14, 17, 17, 18, 21, 23, 3]
    probs = [.10, .14, .12, .08, .06, .06, .06, .10, .09, .08, .05, .04, .02]

    for date in DATES:
        weekend = date.dayofweek >= 5
        for _ in range(14 if not weekend else 7):
            hour = int(rng.choice(hours, p=probs))
            name, cls, _ = roads[rng.integers(len(roads))]
            v = CLASSES[cls]["free_mps"] * congestion(hour, weekend)
            v = max(v * float(rng.normal(1.0, 0.07)), 1.0)
            tid += 1
            _track(rows, roads, f"veh_{tid:04d}", v, hour, date, "vehicle",
                   1.9, (9, 22))

    # Pedestrians on the sidewalk: walking pace, offset from the centreline,
    # phone-grade accuracy. 03_mode_screening exists to remove these.
    for _ in range(18):
        date = DATES[rng.integers(len(DATES))]
        hour = int(rng.choice([8, 9, 12, 13, 17, 18]))
        tid += 1
        _track(rows, roads, f"ped_{tid:04d}", float(rng.uniform(1.1, 1.7)),
               hour, date, "pedestrian", 3.2, (14, 34),
               offset_m=float(rng.choice([-1.0, 1.0])) * 7.0)

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    net_path = os.path.join(OUT, "network.geojson")
    pts_path = os.path.join(OUT, "points.csv")

    roads = build_network(net_path)
    df = build_points(pts_path, roads)

    n = df.groupby("mode")["device_id"].nunique()
    print(f"Wrote {net_path}")
    print(f"      {len(roads)} named roads, three classes, maxspeed tagged")
    print(f"Wrote {pts_path}")
    print(f"      {len(df):,} fixes / {df.device_id.nunique()} movers "
          f"({n.get('vehicle', 0)} vehicle, {n.get('pedestrian', 0)} pedestrian)")
    print(f"      columns: {list(df.columns)}")
    print("      note: no speed column -- speeds are derived from positions")


if __name__ == "__main__":
    main()
