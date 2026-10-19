"""Fetch Copernicus Sentinel-2 images through Sentinel Hub on the Copernicus Data Space Ecosystem."""

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from dotenv import find_dotenv, load_dotenv
from PIL import Image
from sentinelhub import (
    CRS,
    BBox,
    DataCollection,
    MimeType,
    MosaickingOrder,
    SentinelHubRequest,
    SHConfig,
    bbox_to_dimensions,
    to_utm_bbox,
)

from intro_to_vit.satellite.eurosat import REFLECTANCE_GAIN

CDSE_BASE_URL = "https://sh.dataspace.copernicus.eu"
CDSE_TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"


@dataclass(frozen=True)
class Provider:
    """A Sentinel Hub deployment: where to authenticate, where to send requests, and its name."""

    base_url: str
    token_url: str
    name: str


# SH_PROVIDER picks one of these; "cdse" is the free Copernicus Data Space Ecosystem.
PROVIDERS = {
    "cdse": Provider(CDSE_BASE_URL, CDSE_TOKEN_URL, "Sentinel Hub, Copernicus Data Space Ecosystem"),
    "sentinelhub": Provider(
        "https://services.sentinel-hub.com",
        "https://services.sentinel-hub.com/auth/realms/main/protocol/openid-connect/token",
        "Sentinel Hub",
    ),
}
MAX_PIXELS = 2500  # the Process API returns at most 2500 x 2500 pixels per request
DEFAULT_TIME = ("2026-06-01", "2026-08-31")

# True colour from Sentinel-2 L1C, scaled exactly like EuroSAT: reflectance x 10000 / 2750.
EVALSCRIPT = f"""//VERSION=3
function setup() {{
  return {{
    input: [{{ bands: ["B02", "B03", "B04"], units: "REFLECTANCE" }}],
    output: {{ bands: 3, sampleType: "AUTO" }}
  }};
}}
function evaluatePixel(sample) {{
  const gain = {REFLECTANCE_GAIN:.6f};
  return [gain * sample.B04, gain * sample.B03, gain * sample.B02];
}}
"""

CREDENTIALS_HELP = """Sentinel Hub credentials are missing.

The images come from the Copernicus Data Space Ecosystem, which is free but needs an account:
  1. Register at https://dataspace.copernicus.eu
  2. Open https://shapps.dataspace.copernicus.eu/dashboard/#/account/settings,
     find "OAuth clients" and click "Create". Copy the client id and the secret.
  3. Put them in a .env file in the repository root (copy .env.example; .env is git-ignored):
       SH_CLIENT_ID=<client id>
       SH_CLIENT_SECRET=<client secret>
     You can also export the same two variables in the shell, or save them in sentinelhub's
     own config file:
       uv run sentinelhub.config --sh_client_id <client id> --sh_client_secret <client secret>
"""


@dataclass(frozen=True)
class Area:
    """A named area of interest with a WGS84 bounding box (lon_min, lat_min, lon_max, lat_max)."""

    key: str
    name: str
    bbox: tuple[float, float, float, float]
    description: str


AREAS: dict[str, Area] = {
    area.key: area
    for area in (
        Area("stockholm", "Stockholm", (17.95, 59.28, 18.20, 59.38), "Innerstad, förorter, skog och mycket vatten"),
        Area("uppsala", "Uppsala", (17.55, 59.80, 17.75, 59.90), "Stad, åkrar, skog och Fyrisån"),
        Area("lund", "Lund och Skånes slättland", (13.10, 55.66, 13.35, 55.76), "Åkermark, stad och motorvägar"),
    )
}


class MissingCredentialsError(RuntimeError):
    """Raised when no Sentinel Hub OAuth client is configured."""


def load_env_file() -> str:
    """Load variables from the nearest .env file (searching up from the working directory).

    Variables already set in the environment win. Returns the file's path, or "" if none was found.
    """
    path = find_dotenv(usecwd=True)
    if path:
        load_dotenv(path, override=False)
    return path


def selected_provider() -> Provider:
    """Return the provider named by SH_PROVIDER (default "cdse")."""
    key = os.environ.get("SH_PROVIDER", "cdse").strip().lower()
    if key not in PROVIDERS:
        raise ValueError(f"SH_PROVIDER={key!r} is not supported; use one of {', '.join(PROVIDERS)}")
    return PROVIDERS[key]


def sentinel_hub_config() -> SHConfig:
    """Return a sentinelhub config with credentials from .env, the environment or sentinelhub's config file.

    SH_PROVIDER chooses the deployment (default: the Copernicus Data Space Ecosystem), and
    SH_BASE_URL and SH_TOKEN_URL override its endpoints.
    """
    load_env_file()
    provider = selected_provider()
    config = SHConfig()  # reads SH_CLIENT_ID and SH_CLIENT_SECRET from the environment
    config.sh_base_url = os.environ.get("SH_BASE_URL", provider.base_url)
    config.sh_token_url = os.environ.get("SH_TOKEN_URL", provider.token_url)
    if not config.sh_client_id or not config.sh_client_secret:
        raise MissingCredentialsError(CREDENTIALS_HELP)
    return config


@lru_cache
def sentinel2_l1c(service_url: str) -> DataCollection:
    """Return the Sentinel-2 L1C collection as served at `service_url` (defined once per URL and process)."""
    name = "s2l1c-" + (urlparse(service_url).hostname or "default").replace(".", "-")
    return DataCollection.SENTINEL2_L1C.define_from(name, service_url=service_url)


def utm_bbox(bbox: tuple[float, float, float, float]) -> BBox:
    """Convert a WGS84 bounding box to its UTM zone, so that pixels are square in metres."""
    return to_utm_bbox(BBox(bbox, crs=CRS.WGS84))


def image_size(bbox: tuple[float, float, float, float], resolution: int) -> tuple[int, int]:
    """Return the (width, height) in pixels of `bbox` at `resolution` metres per pixel."""
    width, height = bbox_to_dimensions(utm_bbox(bbox), resolution=resolution)
    if max(width, height) > MAX_PIXELS:
        raise ValueError(
            f"The area is {width} x {height} pixels at {resolution} m; Sentinel Hub returns at most "
            f"{MAX_PIXELS} x {MAX_PIXELS} per request. Use a smaller bounding box."
        )
    return width, height


def build_request(
    bbox: tuple[float, float, float, float],
    time_interval: tuple[str, str],
    config: SHConfig,
    resolution: int = 10,
    max_cloud: float = 0.2,
) -> SentinelHubRequest:
    """Build a Process API request for a EuroSAT-scaled true-colour image of the least cloudy data."""
    return SentinelHubRequest(
        evalscript=EVALSCRIPT,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=sentinel2_l1c(config.sh_base_url),
                time_interval=time_interval,
                mosaicking_order=MosaickingOrder.LEAST_CC,
                maxcc=max_cloud,
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
        bbox=utm_bbox(bbox),
        size=image_size(bbox, resolution),
        config=config,
    )


def fetch_true_color(
    bbox: tuple[float, float, float, float],
    time_interval: tuple[str, str],
    config: SHConfig,
    resolution: int = 10,
    max_cloud: float = 0.2,
) -> np.ndarray:
    """Download the scene and return it as an (height, width, 3) uint8 array."""
    return build_request(bbox, time_interval, config, resolution, max_cloud).get_data()[0]


def save_scene(image: np.ndarray, out_dir: str | Path, **metadata) -> Path:
    """Write the scene as sentinel2.png plus scene.json with where and when it was fetched."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(out / "sentinel2.png")
    record = {**metadata, "width": image.shape[1], "height": image.shape[0],
              "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (out / "scene.json").write_text(json.dumps(record, indent=2, ensure_ascii=False))
    return out


def load_scene(scene_dir: str | Path) -> tuple[np.ndarray, dict]:
    """Return (image, metadata) for a scene saved by `save_scene`."""
    scene_dir = Path(scene_dir)
    image_path = scene_dir / "sentinel2.png"
    if not image_path.exists():
        raise FileNotFoundError(f"{image_path} not found. Fetch it first: make satellite-fetch")
    meta_path = scene_dir / "scene.json"
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    with Image.open(image_path) as image:
        return np.asarray(image.convert("RGB")), metadata
