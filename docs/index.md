# roadtraffic

**Trafficability analysis for road networks, from your own GPS data.**

Turn a road network and a pile of GPS fixes into a defensible study of how
traffic actually moves: speed by time of day, peak and off-peak windows,
congestion against posted limits, chokepoints, and routing on measured speeds.

Built on `numpy`, `pandas`, `scipy`, `shapely` and `networkx`.
**No GDAL and no PROJ required** for the core.

## Where to go

| If you want to… | Read |
|---|---|
| Get something running in ten minutes | [Quickstart & concepts](quickstart.md) |
| Understand what a number means | [Statistical methodology](statistics.md) |
| See why the methods were chosen, with evidence | [Methodology paper](methodology.md) |
| Look up a function | [API reference](api.md) |
| Copy a runnable script | [`examples/`](https://github.com/cesouth/roadtraffic/tree/main/examples) |

## Install

```bash
pip install roadtraffic                 # core
pip install roadtraffic[shapefile]      # + Shapefile / GeoPackage (needs 3.10+)
pip install roadtraffic[mapping]        # + static PNG maps
```

## The shape of a study

Most work follows the same path. Each step is a documented function you can
stop at, inspect, and swap out.

```python
import roadtraffic as rt

net = rt.Network.from_geojson("network.geojson")
pts = rt.load_points("points.csv", id_col="vehicle_id", tz="America/New_York")

# Match fixes to roads, then reconstruct speed from on-road displacement.
matched = rt.HMMMatcher(net, max_dist=50).match(pts)
derived = rt.derive_speeds(net, matched, pts, min_baseline_m=150)

# Cap the top end only -- a minimum-speed filter would delete the congestion.
clean = rt.filter_by_speed(derived["edge_observations"], max_speed=80,
                           unit="mph", mad_outliers=True, per_edge=True)

hourly = rt.aggregate_speeds(clean, block_hours=1, statistic="median",
                             output_unit="mph")
peaks = rt.peak_analysis(hourly, statistic="median", n_peak=3, n_offpeak=3)
rt.assign_segment_speeds(net, clean, n_peak=3, n_offpeak=3)
```

From there: `congestion_report` for performance against posted limits,
`edge_betweenness_centrality` for chokepoints, `Router` for travel times, and
`to_geojson` for mapping. All are in the [API reference](api.md).

## What makes it different

**Speed without a speed column.** Plenty of feeds carry position and time and
nothing else. `derive_speeds` reconstructs speed from displacement along the
matched road, with a per-fix error model and an explicit uncertainty on every
measurement.

**Mixed feeds are handled honestly.** If the data also contains people on
foot, mode screening classifies whole *movers* rather than observations — so a
vehicle crawling through a chokepoint keeps its slow rows while a pedestrian is
removed. A minimum-speed filter cannot tell those two apart, and using one
destroys the finding.

**Bias is named, not hidden.** `require_quality` biases speeds upward, because
slow traffic covers the least ground per fix and fails the screen most often.
Mode screening biases upward when it is wrong. Both say so where they are
offered.

**The methods are defended.** [The methodology paper](methodology.md) is not a
description; it is an argument with reproducible ground-truth experiments
behind it.

## License

MIT.
