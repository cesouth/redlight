"""Example 0 — generate sample data.

Creates a small grid road network (GeoJSON) and a GPS point file (CSV) with
realistic time-of-day speed variation, so the other examples can run without
any data of your own. Run this first.

    python examples/00_generate_sample_data.py

Outputs (into examples/sample_data/):
    network.geojson
    points.csv
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "sample_data")

LON0, LAT0 = -77.30, 38.68
SPACING = 0.005   # ~0.5 km per grid cell at this latitude
N = 6             # 6x6 grid of intersections


def make_network(path):
    feats = []
    for i in range(N):
        for j in range(N):
            lon = LON0 + j * SPACING
            lat = LAT0 + i * SPACING
            if j < N - 1:
                feats.append({"type": "Feature",
                              "properties": {"highway": "residential",
                                             "oneway": "no"},
                              "geometry": {"type": "LineString",
                                           "coordinates": [[lon, lat],
                                                           [lon + SPACING, lat]]}})
            if i < N - 1:
                feats.append({"type": "Feature",
                              "properties": {"highway": "residential",
                                             "oneway": "no"},
                              "geometry": {"type": "LineString",
                                           "coordinates": [[lon, lat],
                                                           [lon, lat + SPACING]]}})
    with open(path, "w") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f, indent=0)


def make_points(path, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for hour in range(24):
        if hour in (7, 8, 9, 16, 17, 18):       # rush hours: slow
            base = 12.0
        elif hour in (0, 1, 2, 3, 4, 5):        # overnight: fast
            base = 38.0
        else:
            base = 28.0
        for trip in range(4):
            tid = f"veh_{hour:02d}_{trip}"
            for j in range(N - 1):              # drive east along bottom row
                for frac in (0.25, 0.5, 0.75):
                    lon = LON0 + (j + frac) * SPACING
                    lat = LAT0 + rng.normal(0, 2e-5)   # small GPS noise
                    spd = max(1.0, base + rng.normal(0, 2.5))
                    ts = pd.Timestamp("2024-06-01") + pd.Timedelta(
                        hours=hour, minutes=trip * 7, seconds=j * 12)
                    rows.append({"vehicle_id": tid, "longitude": lon,
                                 "latitude": lat, "timestamp": ts.isoformat(),
                                 "speed": round(spd, 2)})
    pd.DataFrame(rows).to_csv(path, index=False)


def main():
    os.makedirs(OUT, exist_ok=True)
    net_path = os.path.join(OUT, "network.geojson")
    pts_path = os.path.join(OUT, "points.csv")
    make_network(net_path)
    make_points(pts_path)
    print(f"Wrote {net_path}")
    print(f"Wrote {pts_path}")


if __name__ == "__main__":
    main()
