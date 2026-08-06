# Examples

Runnable scripts, grouped by topic. Every one works against the same generated
sample dataset, so you can run them in any order once the data exists.

## Start here

```bash
pip install -e .            # or: pip install redlight
python examples/00_setup/generate_sample_data.py
python examples/01_basics/load_match_derive.py
```

The generator writes `examples/sample_data/` (git-ignored). Every other script
fails with an actionable message if you skip it.

## The examples

| Script | What it shows |
|---|---|
| [`00_setup/generate_sample_data.py`](00_setup/generate_sample_data.py) | Builds the sample road network and GPS feed. Read this to understand what is planted in the data — free-flow speeds by road class, rush hours, a quieter weekend, and pedestrians mixed in. |
| [`01_basics/load_match_derive.py`](01_basics/load_match_derive.py) | **The core pipeline.** Load GPS → HMM map matching → derive speed from on-road displacement. The input has no speed column, which is the common and harder case. Start here. |
| [`02_speed_analysis/clean_and_aggregate.py`](02_speed_analysis/clean_and_aggregate.py) | Cleaning derived speeds, and why a minimum-speed filter destroys the finding you are measuring. Aggregation by hour and by N-hour block. |
| [`02_speed_analysis/peak_and_daytype.py`](02_speed_analysis/peak_and_daytype.py) | Discovering peak windows from the data, comparing weekdays against weekends, and writing per-regime speeds onto the graph. |
| [`03_mode_screening/screen_pedestrians.py`](03_mode_screening/screen_pedestrians.py) | Removing non-vehicle movers from a mixed feed without deleting congestion. Scores itself against the planted truth. |
| [`04_congestion/congestion_vs_limits.py`](04_congestion/congestion_vs_limits.py) | Measured speed as a fraction of the posted limit — the comparison raw speeds cannot make. |
| [`05_network/structure_and_chokepoints.py`](05_network/structure_and_chokepoints.py) | Network shape, connectivity diagnostics, and betweenness-weighted chokepoints. |
| [`05_network/routing.py`](05_network/routing.py) | Fastest routes on measured speeds, how the answer changes by regime, and how the router falls back where nothing was observed. |
| [`06_mapping/export_map.py`](06_mapping/export_map.py) | GeoJSON export for real mapping tools, plus a quick static PNG. |
| [`notebooks/overpass_demo.ipynb`](notebooks/overpass_demo.ipynb) | Fetching a real road network from OpenStreetMap via Overpass. Needs network access. |

## Two ideas worth taking away

**Speed is derived, not read.** The sample feed carries position and time but
no speed, because plenty of real feeds do not. `derive_speeds` reconstructs it
from displacement along the matched road, which is more robust than a
receiver's instantaneous estimate even when one is available.

**Never filter out slow observations.** A vehicle crawling through a chokepoint
and a pedestrian look identical in a single fix, so a `min_speed` floor deletes
exactly the congestion a trafficability study exists to find. If non-vehicle
movers are the problem, screen them per *mover*
(`03_mode_screening`), which keeps every slow observation belonging to a real
vehicle.

## Notes

- `sample_data/` is generated and git-ignored; regenerate it any time.
- Scripts import a shared `_common.py` for the load→match→derive prelude.
  `01_basics` spells that out longhand instead — read it first.
- Everything runs offline except the Overpass notebook.
