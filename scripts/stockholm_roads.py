# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib>=3.9"]
# ///
"""Download the roads of central Stockholm from Trafikverket and save them as a PNG map.

The roads come from NVDB, Trafikverket's national road database, through its open INSPIRE WFS
(CC0 licence, no account needed). Each road segment is drawn by its functional road class
(funktionell vägklass): the more important the road, the darker and wider the line. The map shows
nothing but Trafikverket's data, and the script uses nothing else from this repository.

    uv run scripts/stockholm_roads.py
    uv run scripts/stockholm_roads.py --bbox 660000 6570000 690000 6595000 --out results/roads.png

uv reads the dependencies from the block at the top and installs them for this script only.
"""

import argparse
import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

WFS = "https://geo-inspire.trafikverket.se/MapService/wfs.axd/TN_RoadTransportNetwork"
MAX_FEATURES = 100_000  # the service returns at most this many segments per request
STOCKHOLM = (667_500.0, 6_575_000.0, 682_500.0, 6_587_000.0)  # SWEREF 99 TM: min E, min N, max E, max N
CLASS_NAMES = ("mainRoad", "firstClass", "secondClass", "thirdClass", "fourthClass",
               "fifthClass", "sixthClass", "seventhClass", "eighthClass", "ninthClass")  # classes 0-9

SURFACE, INK, INK_SECONDARY, INK_MUTED, HAIRLINE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"


@dataclass(frozen=True)
class Tier:
    """Functional road classes drawn with one colour and line width."""

    classes: range
    description: str
    color: str
    width: float  # points

    @property
    def label(self) -> str:
        """Return the legend text, e.g. '0–1  Europavägar och nationella huvudvägar'."""
        return f"{self.classes.start}–{self.classes.stop - 1}  {self.description}"


# Least important first, so the major roads are drawn on top: one blue ramp, light to dark.
TIERS = (
    Tier(range(7, 10), "Lokalgator, gågator och småvägar", "#86b6ef", 0.3),
    Tier(range(4, 7), "Länsvägar, huvudgator och uppsamlingsgator", "#3987e5", 0.6),
    Tier(range(2, 4), "Nationella och regionala vägar", "#1c5cab", 1.1),
    Tier(range(0, 2), "Europavägar och nationella huvudvägar", "#0d366b", 1.7),
)


def fetch_roads(bbox: tuple[float, float, float, float]) -> list[dict]:
    """Download every road segment in `bbox` (SWEREF 99 TM) as GeoJSON features from Trafikverket's WFS."""
    min_e, min_n, max_e, max_n = bbox
    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typenames": "tn-ro:FunctionalRoadClass",
        "outputFormat": "application/json",
        "count": str(MAX_FEATURES),
        # EPSG:3006 lists northing before easting, so the box is min N, min E, max N, max E.
        "bbox": f"{min_n:.0f},{min_e:.0f},{max_n:.0f},{max_e:.0f},urn:ogc:def:crs:EPSG::3006",
    }
    request = urllib.request.Request(f"{WFS}?{urllib.parse.urlencode(params)}", headers={"User-Agent": "stockholm-roads"})
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            features = json.load(response)["features"]
    except (urllib.error.URLError, json.JSONDecodeError) as error:
        raise SystemExit(f"Could not download the roads from Trafikverket: {error}") from None
    if len(features) >= MAX_FEATURES:
        raise SystemExit(f"The area has at least {MAX_FEATURES} road segments, the most the service returns at once. "
                         "Use a smaller --bbox.")
    return features


def lines_by_tier(features: list[dict]) -> dict[Tier, list[list[tuple[float, float]]]]:
    """Group the segments by tier as lists of (easting, northing) points; the GeoJSON lists northing first."""
    tier_of_class = {CLASS_NAMES[number]: tier for tier in TIERS for number in tier.classes}
    lines: dict[Tier, list[list[tuple[float, float]]]] = {tier: [] for tier in TIERS}
    for feature in features:
        tier = tier_of_class.get(feature["properties"].get("functionalRoadClass"))
        geometry = feature["geometry"]
        if tier is None or geometry is None:
            continue
        parts = geometry["coordinates"] if geometry["type"] == "MultiLineString" else [geometry["coordinates"]]
        lines[tier].extend([(point[1], point[0]) for point in part] for part in parts)
    return lines


def scale_bar_length(width_m: float) -> int:
    """Return a round scale bar length in metres, at most a fifth of the map's width."""
    lengths = (100, 200, 500, 1000, 2000, 5000, 10_000, 20_000, 50_000)
    return max((length for length in lengths if length <= width_m / 5), default=lengths[0])


def draw_map(lines: dict[Tier, list], bbox: tuple[float, float, float, float], title: str, subtitle: str,
             credit: str) -> plt.Figure:
    """Draw the roads tier by tier with a title, legend, scale bar and source line, and return the figure."""
    min_e, min_n, max_e, max_n = bbox
    fig_w, margin, header, footer = 12.0, 0.4, 1.0, 1.3  # inches
    map_w = fig_w - 2 * margin
    map_h = map_w * (max_n - min_n) / (max_e - min_e)
    fig_h = header + map_h + footer
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=SURFACE)
    ax = fig.add_axes((margin / fig_w, footer / fig_h, map_w / fig_w, map_h / fig_h), facecolor=SURFACE)
    for tier in TIERS:
        ax.add_collection(LineCollection(lines[tier], colors=tier.color, linewidths=tier.width,
                                         capstyle="round", joinstyle="round"))
    ax.set_xlim(min_e, max_e)
    ax.set_ylim(min_n, max_n)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(HAIRLINE)

    inches = fig.dpi_scale_trans
    fig.text(margin, fig_h - 0.5, title, transform=inches, fontsize=18, fontweight="bold", color=INK)
    fig.text(margin, fig_h - 0.82, subtitle, transform=inches, fontsize=11, color=INK_SECONDARY)
    handles = [Line2D([], [], color=tier.color, linewidth=max(1.2, 2.2 * tier.width), solid_capstyle="round",
                      label=tier.label) for tier in reversed(TIERS)]
    legend = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(margin - 0.1, footer - 0.12),
                        bbox_transform=inches, ncol=2, frameon=False, fontsize=10, labelcolor=INK,
                        title="Funktionell vägklass", title_fontsize=10, alignment="left", columnspacing=2.5)
    legend.get_title().set_color(INK_SECONDARY)

    metres_per_inch = (max_n - min_n) / map_h
    length = scale_bar_length(max_e - min_e)
    bar_y = min_n - 0.3 * metres_per_inch
    ax.plot([max_e - length, max_e], [bar_y, bar_y], color=INK_SECONDARY, linewidth=2, solid_capstyle="butt",
            clip_on=False)
    ax.text(max_e - length / 2, bar_y - 0.08 * metres_per_inch, f"{length // 1000} km" if length >= 1000 else f"{length} m",
            ha="center", va="top", fontsize=10, color=INK_SECONDARY, clip_on=False)
    fig.text(fig_w - margin, 0.25, credit, transform=inches, ha="right", fontsize=8.5, color=INK_MUTED)
    return fig


def main() -> None:
    """Download the roads, draw them and save the map as a PNG."""
    parser = argparse.ArgumentParser(description="Download roads from Trafikverket (NVDB) and save them as a PNG map.")
    parser.add_argument("--bbox", nargs=4, type=float, default=STOCKHOLM, metavar=("MIN_E", "MIN_N", "MAX_E", "MAX_N"),
                        help="Area in SWEREF 99 TM metres (EPSG:3006). Default: central Stockholm.")
    parser.add_argument("--title", default="Vägar i Stockholm")
    parser.add_argument("--out", default="results/stockholm-roads.png", help="Where to save the PNG.")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()
    bbox = tuple(args.bbox)

    print(f"Downloading the roads in {bbox} from Trafikverket ...")
    lines = lines_by_tier(fetch_roads(bbox))
    count = f"{sum(len(segments) for segments in lines.values()):,}".replace(",", " ")
    subtitle = f"Trafikverkets nationella vägdatabas (NVDB): {count} vägsegment efter funktionell vägklass"
    credit = f"Källa: Trafikverket, NVDB (CC0), hämtad {date.today().isoformat()}"
    fig = draw_map(lines, bbox, args.title, subtitle, credit)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, facecolor=SURFACE)
    plt.close(fig)
    for tier in reversed(TIERS):
        print(f"  {tier.label}: {len(lines[tier])} segments")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
