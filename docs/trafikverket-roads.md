# Swedish road data from Trafikverket

Trafikverket, the Swedish Transport Administration, publishes NVDB (*Nationell vägdatabas*), the
official database of every road in Sweden: state, municipal and private. This page explains how to
fetch it, for example to draw roads on the Sentinel-2 maps of the [satellite demo](satellite-demo.md).

In short: ask Trafikverket's WFS for the roads in a bounding box as GeoJSON, keep functional road
class 0–3, and convert the coordinates from SWEREF 99 TM to the scene's UTM zone. No account or API
key is needed, and the data is free under CC0.

## Quick start

```bash
uv run scripts/stockholm_roads.py   # saves results/stockholm-roads.png
```

[scripts/stockholm_roads.py](../scripts/stockholm_roads.py) downloads every road in central Stockholm
from the WFS described below and draws them by functional road class, with nothing else on the map. It
is standalone: it uses no other code from this repository, and uv installs its only dependency,
matplotlib, from the block at the top of the file. `--bbox` takes another area in SWEREF 99 TM metres.

## Three ways to get the data

| Service | What you get | Account | Use it for |
|---|---|---|---|
| WFS (download service) | Road lines as GeoJSON or GML, selected by area and attributes | No | Drawing roads in code (recommended) |
| WMS (view service) | Ready-drawn, transparent PNG images | No | A quick look, or a GIS program such as QGIS |
| [Lastkajen](https://www.trafikverket.se/e-tjanster/lastkajen--sveriges-vag--och-jarnvagsdata/) | Files (GeoPackage) for the whole country or a region | Free registration | Bulk downloads and offline use |

The WFS and the WMS are described in Trafikverket's
[service description](https://bransch.trafikverket.se/contentassets/f4d7a64a5cb84b6f919d5e4645e868c7/tjanstebeskrivning_inspire_tn_giltig-version.pdf)
(INSPIRE TN, version 1.2, June 2024). They serve NVDB in the EU's INSPIRE format for transport
networks, which is why layer and attribute names are in English.

## The WFS

```text
https://geo-inspire.trafikverket.se/MapService/wfs.axd/TN_RoadTransportNetwork
```

- WFS 1.1.0 and 2.0.0. `outputFormat=application/json` returns GeoJSON.
- Coordinates are always in SWEREF 99 TM (EPSG:3006), Sweden's national grid.
- At most 100,000 features per response; page through larger results with `count` and `startIndex`.

**Layers.** Each layer (feature type) is a set of road segments, each with its own geometry and one
attribute:

| Feature type | Contents |
|---|---|
| `tn-ro:FunctionalRoadClass` | Every road with its functional road class (below); the one to use for maps |
| `tn-ro:RoadLink` | Every link in NVDB, without class; 130,000 in the Stockholm demo area |
| `tn-ro:ERoad` | European routes, with `europeanRouteNumber` (E4, E18, …) |
| `tn-ro:RoadName` | Road names and numbers |
| `tn:VerticalPosition` | `onGroundSurface`, `suspendedOrElevated` (bridges) or `underground` (tunnels) |
| `tn-ro:FormOfWay`, `tn-ro:NumberOfLanes`, `tn-ro:RoadWidth`, `tn-ro:SpeedLimit`, `tn-ro:RoadSurfaceCategory` | Further attributes |

The GeoJSON properties do not include a reference to the underlying road link, so combining two
layers, such as class and tunnels, means matching their geometries.

## Functional road class

*Funktionell vägklass* ranks roads by how important they are for connections in the road network,
from 0 (most important) to 9. In the GeoJSON it is the attribute `functionalRoadClass`, and `CAB_V`
holds the same class as a digit. The descriptions follow Trafikverket's
[data product specification](https://bransch.trafikverket.se/TrvSeFiler/Dataproduktspecifikationer/V%C3%A4gdataprodukter/DPS_E-G/1010funktionell_vagklass.pdf)
(version 9.0, 2020). The last column counts the segments in the Stockholm demo area in October 2026.

| Class | `functionalRoadClass` | Typical roads | Stockholm |
|---|---|---|---|
| 0 | `mainRoad` | All European routes and the motorways that form or join them | 142 |
| 1 | `firstClass` | The next most important national roads, which link class 0 roads | 373 |
| 2 | `secondClass` | Other national roads, mostly road numbers below 100 | 51 |
| 3 | `thirdClass` | Regional roads, mostly primary county roads (numbers 100–499); city streets with through traffic | 339 |
| 4 | `fourthClass` | Secondary county roads, often 90 km/h; other main city streets | 1,021 |
| 5 | `fifthClass` | Secondary and tertiary county roads, often 70 km/h; main streets with trunk buses or cycles in the carriageway | 2,396 |
| 6 | `sixthClass` | State roads of the lowest standard; collector streets | 1,762 |
| 7 | `seventhClass` | Streets inside residential areas; main forest roads | 13,183 |
| 8 | `eighthClass` | Pedestrian streets; small private roads; most forest roads | 394 |
| 9 | `ninthClass` | Roads in car parks and fenced industrial areas; the smallest forest roads | 7,120 |

Sentinel-2's 10 m pixels show the large roads, so classes 0–3 suit the satellite maps. Classes 4–5
add the main streets.

## Requests

```bash
WFS=https://geo-inspire.trafikverket.se/MapService/wfs.axd/TN_RoadTransportNetwork

# How many segments are in the Stockholm demo area? (WGS84 box, latitude first)
curl "$WFS?service=WFS&version=2.0.0&request=GetFeature&typenames=tn-ro:FunctionalRoadClass&resultType=hits&bbox=59.28,17.95,59.38,18.20,urn:ogc:def:crs:EPSG::4326"

# The European routes in the same area, as GeoJSON
curl -o eroads.json "$WFS?service=WFS&version=2.0.0&request=GetFeature&typenames=tn-ro:ERoad&outputFormat=application/json&bbox=59.28,17.95,59.38,18.20,urn:ogc:def:crs:EPSG::4326"
```

The first request answers with XML that holds the count in `numberMatched`. In WFS 2.0, a bounding
box in EPSG:4326 is written latitude first.

**Filtering on class.** WFS does not allow the `bbox` parameter together with a `filter`, so the box
goes into the filter. This server is strict about two things: the GML namespace must be
`http://www.opengis.net/gml` (the newer `http://www.opengis.net/gml/3.2` is rejected), and the
envelope is in EPSG:3006 with northing first. The envelope below is the Stockholm demo box, 17.95–18.20° E
and 59.28–59.38° N, converted to SWEREF 99 TM:

```bash
WFS=https://geo-inspire.trafikverket.se/MapService/wfs.axd/TN_RoadTransportNetwork
FILTER='<fes:Filter xmlns:fes="http://www.opengis.net/fes/2.0" xmlns:gml="http://www.opengis.net/gml">
  <fes:And>
    <fes:BBOX>
      <gml:Envelope srsName="urn:ogc:def:crs:EPSG::3006">
        <gml:lowerCorner>6574952 667578</gml:lowerCorner>
        <gml:upperCorner>6586736 682307</gml:upperCorner>
      </gml:Envelope>
    </fes:BBOX>
    <fes:Or>
      <fes:PropertyIsEqualTo><fes:ValueReference>functionalRoadClass</fes:ValueReference><fes:Literal>mainRoad</fes:Literal></fes:PropertyIsEqualTo>
      <fes:PropertyIsEqualTo><fes:ValueReference>functionalRoadClass</fes:ValueReference><fes:Literal>firstClass</fes:Literal></fes:PropertyIsEqualTo>
      <fes:PropertyIsEqualTo><fes:ValueReference>functionalRoadClass</fes:ValueReference><fes:Literal>secondClass</fes:Literal></fes:PropertyIsEqualTo>
      <fes:PropertyIsEqualTo><fes:ValueReference>functionalRoadClass</fes:ValueReference><fes:Literal>thirdClass</fes:Literal></fes:PropertyIsEqualTo>
    </fes:Or>
  </fes:And>
</fes:Filter>'

curl -G "$WFS" -o major-roads.json \
  --data-urlencode service=WFS --data-urlencode version=2.0.0 --data-urlencode request=GetFeature \
  --data-urlencode typenames=tn-ro:FunctionalRoadClass --data-urlencode outputFormat=application/json \
  --data-urlencode "filter=$FILTER"
```

The same kind of filter with `typenames=tn:VerticalPosition` and the condition `verticalPosition` =
`underground` returns the tunnels, 123 segments in the Stockholm demo area.

## The response

A shortened feature from `major-roads.json`:

```json
{
  "type": "FeatureCollection",
  "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::3006"}},
  "features": [{
    "type": "Feature",
    "id": "SE05843E6B-F5E8-4631-97D6-B376FDEF1740",
    "properties": {
      "functionalRoadClass": "firstClass",
      "CAB_V": "1",
      "fromPosition": 0.0,
      "toPosition": 1.0,
      "validFrom": "2006-05-10T00:00:00+01:00",
      "beginLifespanVersion": "2026-10-14T00:00:00+01:00"
    },
    "geometry": {
      "type": "LineString",
      "coordinates": [[6581285.43, 673466.6], [6581223.168, 673508.865], [6581195.595, 673523.821]]
    }
  }]
}
```

- **Coordinates are [northing, easting]**, the axis order of EPSG:3006, not the x, y order GeoJSON
  normally uses. Swap them before converting.
- **Whole segments are returned**, so lines can reach outside the box. Clip them when drawing.
- **`beginLifespanVersion` is the date of the data load**, 2026-10-14 on every segment fetched on
  21 October 2026.

## Drawing the roads on a Sentinel-2 scene

`requests` and `pyproj` are already installed through `sentinelhub`. The demo's scenes are in the UTM
zone of their area (`utm_bbox` in [sentinel.py](../src/intro_to_vit/satellite/sentinel.py)), so the
roads are converted from SWEREF 99 TM to that zone and then to pixels:

```python
import numpy as np
import requests
from pyproj import Transformer
from sentinelhub import CRS, BBox, to_utm_bbox

WFS = "https://geo-inspire.trafikverket.se/MapService/wfs.axd/TN_RoadTransportNetwork"
MAJOR_ROADS = ("mainRoad", "firstClass", "secondClass", "thirdClass")  # functional road class 0-3


def road_filter(bbox: tuple[float, float, float, float], classes: tuple[str, ...]) -> str:
    """Return an FES 2.0 filter for the segments in a WGS84 bbox whose functional road class is in `classes`."""
    to_sweref = Transformer.from_crs("EPSG:4326", "EPSG:3006", always_xy=True)
    east, north = to_sweref.transform([bbox[0], bbox[0], bbox[2], bbox[2]], [bbox[1], bbox[3], bbox[1], bbox[3]])
    envelope = (  # EPSG:3006 puts northing first
        '<gml:Envelope srsName="urn:ogc:def:crs:EPSG::3006">'
        f"<gml:lowerCorner>{min(north):.0f} {min(east):.0f}</gml:lowerCorner>"
        f"<gml:upperCorner>{max(north):.0f} {max(east):.0f}</gml:upperCorner></gml:Envelope>"
    )
    condition = "".join(
        "<fes:PropertyIsEqualTo><fes:ValueReference>functionalRoadClass</fes:ValueReference>"
        f"<fes:Literal>{name}</fes:Literal></fes:PropertyIsEqualTo>"
        for name in classes
    )
    if len(classes) > 1:
        condition = f"<fes:Or>{condition}</fes:Or>"
    return (
        '<fes:Filter xmlns:fes="http://www.opengis.net/fes/2.0" xmlns:gml="http://www.opengis.net/gml">'
        f"<fes:And><fes:BBOX>{envelope}</fes:BBOX>{condition}</fes:And></fes:Filter>"
    )


def fetch_roads(bbox: tuple[float, float, float, float], classes: tuple[str, ...] = MAJOR_ROADS) -> dict:
    """Download the roads in a WGS84 bbox as GeoJSON, with [northing, easting] coordinates in EPSG:3006."""
    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typenames": "tn-ro:FunctionalRoadClass",
        "outputFormat": "application/json",
        "filter": road_filter(bbox, classes),
    }
    response = requests.get(WFS, params=params, timeout=180)
    response.raise_for_status()
    return response.json()


def to_scene_pixels(roads: dict, bbox: tuple[float, float, float, float], width: int, height: int) -> list[np.ndarray]:
    """Return every segment as (n, 2) pixel coordinates in the scene sentinel.py fetched for `bbox`."""
    utm = to_utm_bbox(BBox(bbox, crs=CRS.WGS84))  # the UTM box the Sentinel Hub request covers
    to_utm = Transformer.from_crs("EPSG:3006", f"EPSG:{utm.crs.epsg}", always_xy=True)
    lines = []
    for feature in roads["features"]:
        geometry = feature["geometry"]
        parts = geometry["coordinates"] if geometry["type"] == "MultiLineString" else [geometry["coordinates"]]
        for part in parts:
            north, east = np.asarray(part).T
            x, y = to_utm.transform(east, north)
            column = (np.asarray(x) - utm.min_x) / (utm.max_x - utm.min_x) * width
            row = (utm.max_y - np.asarray(y)) / (utm.max_y - utm.min_y) * height
            lines.append(np.column_stack([column, row]))
    return lines
```

Run from the repository root, after `make satellite-fetch`:

```python
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from intro_to_vit.satellite.sentinel import load_scene

image, meta = load_scene("results/satellite/stockholm")
bbox = tuple(meta["bbox"])
lines = to_scene_pixels(fetch_roads(bbox), bbox, meta["width"], meta["height"])

fig, ax = plt.subplots(figsize=(10, 7))
ax.imshow(image, extent=(0, meta["width"], meta["height"], 0))
ax.add_collection(LineCollection(lines, colors="#FFD400", linewidths=1.2))
ax.set_xlim(0, meta["width"])  # segments can reach past the scene
ax.set_ylim(meta["height"], 0)
ax.set_axis_off()
fig.savefig("stockholm-roads.png", dpi=150)
```

The pixel coordinates start at the top-left corner of the scene, like the `extent` that `render_map`
in [mapping.py](../src/intro_to_vit/satellite/mapping.py) uses, so the same lines fit both of its
panels.

## The WMS

```text
https://geo-inspire.trafikverket.se/MapService/wms.axd/TN_RoadTransportNetwork
```

A transparent PNG of the Stockholm demo area, 1000 × 800 pixels:

```text
https://geo-inspire.trafikverket.se/MapService/wms.axd/TN_RoadTransportNetwork?service=WMS&version=1.1.1&request=GetMap&layers=TN.RoadTransportNetwork.FunctionalRoadClass&styles=&srs=EPSG:3006&bbox=667578,6574952,682307,6586736&width=1000&height=800&format=image/png&transparent=true
```

- Layers are named after the WFS feature types, for example `TN.RoadTransportNetwork.FunctionalRoadClass`
  and `TN.RoadTransportNetwork.ERoad`.
- Each layer has a single style, `Default`, which draws all classes in Trafikverket's colours.
- The projections are EPSG:3006, EPSG:4326, EPSG:4258 and CRS:84, but not UTM, so the images do not
  line up with the demo's scenes without reprojection.
- In WMS 1.1.1 the box is given as easting, northing.

## Lastkajen

[Lastkajen](https://www.trafikverket.se/e-tjanster/lastkajen--sveriges-vag--och-jarnvagsdata/) is
Trafikverket's download portal for road and rail data. It needs a free registration with an email
address and acceptance of a licence agreement. It offers ready-made packages, or orders that combine
data products and areas, and the INSPIRE road network comes as GeoPackage. Lastkajen also has an API,
with a manual in the portal.

## Practical notes

- **Fetch before a demo.** The service is normally up around the clock, but support is weekdays
  08:00–16:30, and there is a maintenance window on the third weekend of each month, Friday 18:00 to
  Sunday 23:00. Save the GeoJSON next to the scene.
- **Tunnels.** NVDB includes roads in tunnels, such as Norra länken and Södra länken in Stockholm,
  which cannot be seen from space. `tn:VerticalPosition` marks them as `underground`.
- **Ferry routes** served by road ferries are part of the road network and included.
- **Speed.** The class 0–3 request for the Stockholm demo area returns 905 segments in a few seconds.

## Licence and attribution

The data in the WFS and the WMS is released under
[CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/): free to use, without conditions.
Credit is not required, but a line such as "Vägdata: Trafikverket (NVDB)" next to "Contains modified
Copernicus Sentinel data" tells viewers where the roads come from.

## References

- Trafikverket, [Tjänstebeskrivning INSPIRE TN](https://bransch.trafikverket.se/contentassets/f4d7a64a5cb84b6f919d5e4645e868c7/tjanstebeskrivning_inspire_tn_giltig-version.pdf),
  version 1.2, 2024: the WFS and WMS, their layers, and the licence.
- Trafikverket, [Dataproduktspecifikation Funktionell vägklass](https://bransch.trafikverket.se/TrvSeFiler/Dataproduktspecifikationer/V%C3%A4gdataprodukter/DPS_E-G/1010funktionell_vagklass.pdf),
  version 9.0, 2020: the definitions of the classes.
- Trafikverket, [Lastkajen – Sveriges väg- och järnvägsdata](https://www.trafikverket.se/e-tjanster/lastkajen--sveriges-vag--och-jarnvagsdata/).
- European Commission, [INSPIRE data specification on transport networks](https://inspire.ec.europa.eu/id/document/tg/tn).
