# roadtraffic

**Version 0.5.0** · MIT licensed · Python 3.9+

**Trafficability analysis for road networks, from your own GPS data.**

Turn a road network and a pile of GPS fixes into a defensible study of how
traffic actually moves: speed by time of day, peak and off-peak windows,
congestion against posted limits, chokepoints, and routing on measured speeds.

Built on a small, well-understood dependency set — `numpy`, `pandas`, `scipy`,
`shapely`, `pyproj`, `networkx`. **No GDAL required** for the core. GeoJSON
works out of the box; Shapefile and GeoPackage are an opt-in extra.

---

## Install

```bash
pip install roadtraffic                 # core
pip install roadtraffic[shapefile]      # + Shapefile / GeoPackage (needs 3.10+)
pip install roadtraffic[mapping]        # + static PNG maps
```

From source:

```bash
git clone https://github.com/cesouth/roadtraffic.git
cd roadtraffic
pip install -e ".[dev]"
pytest
```

## A worked example

Your GPS has position and time but no usable speed — the common case.
`roadtraffic` reconstructs speed from how far each vehicle moved *along the
road*, which is more trustworthy than a receiver's instantaneous reading.

```python
import roadtraffic as rt

# 1. Road network. GeoJSON here; .shp/.gpkg via Network.from_file,
#    or straight from OpenStreetMap with Network.from_overpass(bbox).
net = rt.Network.from_geojson("network.geojson")

# 2. GPS fixes. Columns are auto-detected; tz= puts the timestamps on the
#    local clock so "rush hour" means the local rush hour.
pts = rt.load_points("points.csv", id_col="vehicle_id", tz="America/New_York")

# 3. Match each fix to a road. HMMMatcher decodes the whole trajectory, so a
#    fix that is nearer the wrong road still lands on the road travelled.
matched = rt.HMMMatcher(net, max_dist=50).match(pts)

# 4. Reconstruct speed from on-road displacement.
derived = rt.derive_speeds(net, matched, pts, min_baseline_m=150)
obs = derived["edge_observations"]

# 5. Clean. Cap the top end for GPS-jump artefacts; leave the bottom alone.
clean = rt.filter_by_speed(obs, max_speed=80, unit="mph",
                           mad_outliers=True, per_edge=True)

# 6. Speed by hour, then the peak and off-peak windows, found from the data.
hourly = rt.aggregate_speeds(clean, block_hours=1, statistic="median",
                             output_unit="mph")
peaks = rt.peak_analysis(hourly, statistic="median", n_peak=3, n_offpeak=3)

# 7. Write per-regime speeds onto the graph, then map and route on them.
rt.assign_segment_speeds(net, clean, n_peak=3, n_offpeak=3)
rt.to_geojson(net, "trafficability_peak.geojson", period="peak")

router = rt.Router(net)
route = router.route((-77.30, 38.80), (-77.27, 38.81), mode="time", period="peak")
print(f"{route['travel_time_s']:.0f} s over {route['distance_m']:.0f} m")
```

Runnable versions of this and everything below live in
[`examples/`](examples/) — start with
[`01_basics`](examples/01_basics/load_match_derive.py).

## What it does

**Getting data in.** Networks from GeoJSON, Shapefile/GeoPackage, or
OpenStreetMap via Overpass. Points from CSV/TSV/GeoJSON with auto-detected
columns, unit-aware speed parsing, and timezone handling.

**Speed without a speed column.** `derive_speeds` reconstructs speed from
on-road displacement after matching, with a per-fix error model, an explicit
uncertainty on every measurement, and a quality flag you can filter on — or
deliberately not.

**Two matchers, one interface.** `NearestMatcher` snaps each fix
independently and is fast. `HMMMatcher` decodes the trajectory with Viterbi
(Newson & Krumm, 2009) and is right more often. Neither needs extra
dependencies.

**Mixed feeds.** If your data also contains people on foot, `roadtraffic.modes`
classifies whole *movers* rather than observations, so a vehicle crawling
through a chokepoint keeps its slow rows while a pedestrian is removed
entirely. A minimum-speed filter cannot tell those two apart.

**Time.** Aggregate by hour or N-hour block, mean (with SEM and 95% CI) or
median (with IQR). Split weekday from weekend rather than pooling them.
Optionally weight each observation by the precision it was measured to.

**Peak and off-peak.** Rank bins by congestion, pick contiguous windows of
your chosen width, or pass explicit hours — then assign overall, peak and
off-peak speeds per segment.

**Congestion against posted limits.** `congestion_report` divides measured
speed by each road's `maxspeed`. A motorway at 18 mph and a side street at
20 mph look alike until you see they are at 0.28 and 0.81 of their limits.

**Structure.** Travel-time-weighted betweenness to find genuine chokepoints,
descriptive network statistics, and connectivity diagnostics that distinguish
a one-way trap from a clipped extract.

**Routing and mapping.** Dijkstra by time, distance, or a custom cost, per
regime, with parallel roads and OSM one-way semantics handled. Unobserved
roads fall back to their posted limit before any global default. Export
GeoJSON for QGIS/Kepler/Leaflet, or render a static PNG.

## Two things worth knowing up front

**Never filter out slow observations.** A vehicle in gridlock and a pedestrian
look identical in a single fix, so a `min_speed` floor deletes exactly the
congestion a trafficability study exists to find. Cap the top end only. If
non-vehicle movers are the problem, screen them per mover.

**Every filter has a direction of bias, and this package says which.**
`require_quality` biases speeds upward, because slow traffic covers the least
ground per fix and fails the screen most often. Mode screening biases upward
too, when it is wrong. Both are documented where they are offered, not buried.

## Documentation

- [Quickstart & concepts](docs/quickstart.md)
- [Statistical methodology](docs/statistics.md) — what every number means
- [Methodology paper](docs/methodology.md) — the empirical argument for the
  HMM matcher and the on-road speed estimator, with reproducible experiments
- [API reference](docs/api.md)
- [Examples](examples/)
- [Changelog](CHANGELOG.md)

Build the docs site with `pip install roadtraffic[docs] && mkdocs serve`.

## Development

This package was developed with the assistance of AI tooling. Every change
was reviewed by the author before being merged, and the methods it implements
are defended empirically rather than asserted: the
[methodology paper](docs/methodology.md) reproduces its figures and numbers
from fixed seeds using only the public API, and the test suite executes every
Python example in this documentation, so the prose cannot drift from the code
it describes.

## License

MIT. See [`LICENSE`](LICENSE).
