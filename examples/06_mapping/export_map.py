"""Exporting a speed-annotated network for mapping.

    python examples/06_mapping/export_map.py

``to_geojson`` needs no extra dependency and is the one to reach for: the
result opens in QGIS, geojson.io, Kepler, or a web map, and keeps the speed
attributes so the styling can happen wherever you actually make maps.
``plot_speed_map`` renders a quick static PNG and needs the ``mapping`` extra
(``pip install roadtraffic[mapping]``).

Outputs land in ``examples/sample_data/``.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _common import DATA, prepare, rule  # noqa: E402

import roadtraffic as rt  # noqa: E402


def main() -> None:
    net, pts, derived = prepare()
    clean = rt.filter_by_speed(derived["edge_observations"], max_speed=80,
                               unit="mph", mad_outliers=True, per_edge=True)
    rt.assign_segment_speeds(net, clean, statistic="median",
                             n_peak=3, n_offpeak=3)

    # ------------------------------------------------------------- GeoJSON
    rule("GeoJSON export")
    out = os.path.join(DATA, "speeds.geojson")
    fc = rt.to_geojson(net, out, period="overall", speed_unit="mph")
    print(f"wrote {out}")
    print(f"  {len(fc['features'])} features")
    props = fc["features"][0]["properties"]
    print(f"  properties on each feature: {sorted(props)}")
    print("\nperiod= selects which regime's speed is written: 'overall',")
    print("'peak' or 'offpeak'. Export each separately to style them as")
    print("layers. directional=True keeps the two directions of a two-way")
    print("street as separate features instead of collapsing them.")

    # -------------------------------------------------------- static PNG
    rule("Static PNG (needs the mapping extra)")
    png = os.path.join(DATA, "speed_map.png")
    try:
        rt.plot_speed_map(net, png, period="overall", speed_unit="mph")
    except ImportError as exc:
        print(f"skipped: {exc}")
        print("install with: pip install roadtraffic[mapping]")
    else:
        print(f"wrote {png}")
        print("  Good enough for a quick look. For anything a customer sees,")
        print("  scripts/customer_report.py builds a full PDF/PPTX deck.")

    rule("Colour, briefly")
    print("The default colormap here is a diverging red-yellow-green, which")
    print("is conventional for traffic but is exactly the pairing colourblind")
    print("readers cannot separate. If the map is going in a report, prefer a")
    print("single-hue sequential ramp for absolute speed, and reserve a")
    print("diverging scale (with a neutral midpoint) for a ratio like")
    print("observed-over-posted, where the midpoint genuinely means something.")


if __name__ == "__main__":
    main()
