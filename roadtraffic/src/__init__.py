"""roadtraffic: lightweight, bring-your-own-data trafficability analysis for road networks.

Public API
----------
Network               : road graph container (load from GeoJSON / Shapefile / GPKG).
load_points           : read a GPS point file into a normalised PointSet
                        (``derive_speed=True`` reconstructs speed from positions).
save_points           : write a PointSet (incl. derived speed) to CSV / GeoJSON.
PointSet              : container of GPS observations with units handling.
NearestMatcher        : fast independent nearest-edge snapping.
HMMMatcher            : trajectory-aware HMM/Viterbi map matching.
filter_by_speed       : drop observations outside a plausible speed band.
filter_trajectory_speed : dwell-aware cleaning that keeps slow-but-moving traffic.
aggregate_speeds      : average speed by hour or by N-hour block, mean or median.
peak_analysis         : identify peak / off-peak periods.
classify_hours        : split the day into peak vs off-peak hour blocks.
assign_speeds         : write a single aggregated speed onto network edges.
assign_segment_speeds : write overall / peak / off-peak speeds onto network edges.
Router                : shortest path by time (per regime), distance, or cost.
"""

from .units import SpeedUnit, to_mps, from_mps
from .points import PointSet, load_points, save_points
from .network import Network
from .matching import NearestMatcher, HMMMatcher
from .cleaning import filter_by_speed, filter_trajectory_speed
from .aggregate import (
    aggregate_speeds,
    peak_analysis,
    assign_speeds,
    classify_hours,
    assign_segment_speeds,
)
from .routing import Router

__version__ = "0.1.0"

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
    "filter_by_speed",
    "filter_trajectory_speed",
    "aggregate_speeds",
    "peak_analysis",
    "assign_speeds",
    "classify_hours",
    "assign_segment_speeds",
    "Router",
    "__version__",
]
