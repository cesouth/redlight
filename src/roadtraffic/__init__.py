"""roadtraffic: lightweight, bring-your-own-data trafficability analysis for road networks.

Public API
----------
Network               : road graph container (load from GeoJSON / Shapefile /
                        GPKG, or fetch from OSM via ``Network.from_overpass``).
load_points           : read a GPS point file into a normalised PointSet
                        (position+time only is valid; ``derive_speed=True``
                        reconstructs a per-point speed from positions).
save_points           : write a PointSet (incl. derived speed) to CSV / GeoJSON.
PointSet              : container of GPS observations with units handling.
NearestMatcher        : fast independent nearest-edge snapping.
HMMMatcher            : trajectory-aware HMM/Viterbi map matching.
derive_speeds         : reconstruct speed from on-road displacement after matching
                        (more accurate than ``load_points(derive_speed=True)``).
filter_by_speed       : drop observations outside a plausible speed band.
filter_trajectory_speed : dwell-aware cleaning that keeps slow-but-moving traffic.
aggregate_speeds      : average speed by hour or by N-hour block, mean or median.
peak_analysis         : identify peak / off-peak periods.
classify_hours        : split the day into peak vs off-peak hour blocks.
assign_speeds         : write a single aggregated speed onto network edges.
assign_segment_speeds : write overall / peak / off-peak speeds onto network edges.
Router                : shortest path by time (per regime), distance, or cost.
to_geojson            : export a speed-annotated network as a GeoJSON map.
plot_speed_map        : render a quick static PNG map coloured by speed.
"""

from .aggregate import (
    aggregate_speeds,
    assign_segment_speeds,
    assign_speeds,
    classify_hours,
    peak_analysis,
)
from .cleaning import filter_by_speed, filter_trajectory_speed
from .mapping import plot_speed_map, to_geojson
from .matching import HMMMatcher, NearestMatcher
from .network import Network
from .points import PointSet, load_points, save_points
from .routing import Router
from .speeds import derive_speeds
from .units import SpeedUnit, from_mps, to_mps

__version__ = "0.2.0"

__all__ = [
    "SpeedUnit",
    "to_mps",
    "from_mps",
    "PointSet",
    "load_points",
    "save_points",
    "Network",
    "NearestMatcher",
    "HMMMatcher",
    "derive_speeds",
    "filter_by_speed",
    "filter_trajectory_speed",
    "aggregate_speeds",
    "peak_analysis",
    "assign_speeds",
    "classify_hours",
    "assign_segment_speeds",
    "Router",
    "to_geojson",
    "plot_speed_map",
    "__version__",
]
