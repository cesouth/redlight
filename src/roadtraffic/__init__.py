"""roadtraffic: lightweight, bring-your-own-data trafficability analysis for road networks.

Public API
----------
SpeedUnit, to_mps, from_mps : the mph/kph/m/s unit type and its conversion
                        helpers, used throughout the API wherever a
                        ``speed_unit=``/``output_unit=`` argument is accepted.
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
mover_features        : per-mover evidence table (speed percentile, sample
                        count, distance) from matched or derived speeds.
suggest_mode_threshold : density-valley speed separating walkers from drivers,
                        or None when no walking population exists.
classify_movers       : label movers pedestrian / vehicle / unknown, judging
                        the whole trajectory rather than each observation.
filter_by_mode        : keep the observations of movers with a given mode --
                        every retained mover keeps its slow rows.
aggregate_speeds      : average speed by hour or by N-hour block, mean or median
                        (optional ``days=`` filter for weekday/weekend splits).
peak_analysis         : identify peak / off-peak periods.
classify_hours        : split the day into peak vs off-peak hour blocks.
day_type_report       : compare traffic across day-types (weekday vs weekend by
                        default) -- overall/hourly/peak speeds and their delta.
congestion_report     : observed speed as a fraction of the posted speed limit.
assign_speeds         : write a single aggregated speed onto network edges.
assign_segment_speeds : write overall / peak / off-peak speeds onto network edges.
Router                : shortest path by time (per regime), distance, or cost.
to_geojson            : export a speed-annotated network as a GeoJSON map.
plot_speed_map        : render a quick static PNG map coloured by speed.
edge_betweenness_centrality : which roads carry a disproportionate share of
                        shortest paths -- trafficability chokepoints, not
                        just topologically central roads.
network_stats         : circuity, streets-per-node, intersection/dead-end
                        counts, optional area-based densities.
connectivity_report    : largest strongly-connected-component diagnostics and
                        a one-way-trap vs. disconnected-extract distinction.
"""

from .aggregate import (
    aggregate_speeds,
    assign_segment_speeds,
    assign_speeds,
    classify_hours,
    congestion_report,
    day_type_report,
    peak_analysis,
)
from .analysis import connectivity_report, edge_betweenness_centrality, network_stats
from .cleaning import filter_by_speed, filter_trajectory_speed
from .mapping import plot_speed_map, to_geojson
from .matching import HMMMatcher, NearestMatcher
from .modes import (
    MODE_PEDESTRIAN,
    MODE_UNKNOWN,
    MODE_VEHICLE,
    classify_movers,
    filter_by_mode,
    mover_features,
    suggest_mode_threshold,
)
from .network import Network
from .points import PointSet, load_points, save_points
from .routing import Router
from .speeds import derive_speeds
from .units import SpeedUnit, from_mps, to_mps

__version__ = "0.5.0"

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
    "day_type_report",
    "congestion_report",
    "Router",
    "to_geojson",
    "plot_speed_map",
    "edge_betweenness_centrality",
    "network_stats",
    "connectivity_report",
    "mover_features",
    "suggest_mode_threshold",
    "classify_movers",
    "filter_by_mode",
    "MODE_PEDESTRIAN",
    "MODE_VEHICLE",
    "MODE_UNKNOWN",
    "__version__",
]
