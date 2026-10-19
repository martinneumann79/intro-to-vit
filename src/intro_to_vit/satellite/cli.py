"""Command-line entry points: satellite-train, satellite-fetch and satellite-map."""

import argparse
import json
from pathlib import Path

from intro_to_vit.satellite.sentinel import AREAS, DEFAULT_TIME

RESULTS = Path("results/satellite")
DEFAULT_MODEL = RESULTS / "vit-eurosat.pt"


def train_main(argv: list[str] | None = None) -> None:
    """Fine-tune the ViT on EuroSAT and save it."""
    from intro_to_vit.satellite.train import fine_tune

    parser = argparse.ArgumentParser(description="Fine-tune a pretrained ViT on EuroSAT (Sentinel-2 land use).")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out", default=str(DEFAULT_MODEL))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    fine_tune(args.data_dir, args.out, args.epochs, args.batch_size, args.lr, args.seed, args.workers)


def area_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the shared --area / --bbox options."""
    parser.add_argument("--area", default="stockholm", choices=list(AREAS), help="A predefined area of interest.")
    parser.add_argument("--bbox", nargs=4, type=float, metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
                        help="Your own WGS84 bounding box instead of --area (results go to results/satellite/custom).")


def scene_dir(args: argparse.Namespace) -> Path:
    """Return the folder that holds the scene for --area or --bbox."""
    return RESULTS / ("custom" if args.bbox else args.area)


def fetch_main(argv: list[str] | None = None) -> None:
    """Download a Sentinel-2 true-colour scene from the Copernicus Data Space Ecosystem."""
    from intro_to_vit.satellite.sentinel import (
        MissingCredentialsError,
        fetch_true_color,
        save_scene,
        selected_provider,
        sentinel_hub_config,
    )

    parser = argparse.ArgumentParser(description="Fetch a Sentinel-2 scene through Sentinel Hub (Copernicus).")
    area_arguments(parser)
    parser.add_argument("--start", default=DEFAULT_TIME[0], help="First day to search, YYYY-MM-DD.")
    parser.add_argument("--end", default=DEFAULT_TIME[1], help="Last day to search, YYYY-MM-DD.")
    parser.add_argument("--max-cloud", type=float, default=0.2, help="Maximum cloud cover per tile, 0-1.")
    parser.add_argument("--resolution", type=int, default=10, help="Metres per pixel (EuroSAT uses 10).")
    args = parser.parse_args(argv)
    try:
        config = sentinel_hub_config()
    except (MissingCredentialsError, ValueError) as error:
        raise SystemExit(str(error)) from None
    area = AREAS[args.area]
    bbox = tuple(args.bbox) if args.bbox else area.bbox
    name = "Eget område" if args.bbox else area.name
    print(f"Fetching Sentinel-2 L1C for {name} {bbox}, {args.start} to {args.end}, max cloud {args.max_cloud:.0%} ...")
    image = fetch_true_color(bbox, (args.start, args.end), config, args.resolution, args.max_cloud)
    out = save_scene(image, scene_dir(args), name=name, bbox=list(bbox), time_interval=[args.start, args.end],
                     max_cloud=args.max_cloud, resolution_m=args.resolution,
                     source=f"Copernicus Sentinel-2 L1C via {selected_provider().name}")
    print(f"Saved {image.shape[1]} x {image.shape[0]} px to {out / 'sentinel2.png'}")


def map_main(argv: list[str] | None = None) -> None:
    """Classify a fetched scene with the fine-tuned ViT and draw the land-use map."""
    from intro_to_vit.satellite.mapping import classify_scene, format_summary, render_map
    from intro_to_vit.satellite.model import load_checkpoint
    from intro_to_vit.satellite.sentinel import load_scene
    from intro_to_vit.training import get_device

    parser = argparse.ArgumentParser(description="Draw a land-use map of a Sentinel-2 scene with the ViT.")
    area_arguments(parser)
    parser.add_argument("--scene-dir", help="Folder with sentinel2.png (default: results/satellite/<area>).")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    args = parser.parse_args(argv)
    folder = Path(args.scene_dir) if args.scene_dir else scene_dir(args)
    device = get_device()
    try:
        image, meta = load_scene(folder)
        model, class_names, metrics = load_checkpoint(args.model, device)
    except FileNotFoundError as error:
        raise SystemExit(str(error)) from None
    land_use = classify_scene(model, class_names, image, device)
    name = meta.get("name", folder.name)
    title = f"Markanvändning enligt en Vision Transformer: {name}"
    period = " till ".join(meta.get("time_interval", []))
    subtitle = f"{meta.get('source', 'Sentinel-2')}" + (f", {period}" if period else "")
    years = sorted({day[:4] for day in meta.get("time_interval", [])})
    credit = f"Contains modified Copernicus Sentinel data {'–'.join(years)}".strip()
    render_map(image, land_use, folder / "map.png", title, subtitle, credit)
    summary = format_summary(land_use, f"{title} ({land_use.classes.size} rutor à 640 × 640 m)")
    (folder / "summary.md").write_text(summary + "\n")
    (folder / "land_use.json").write_text(json.dumps(
        {"class_names": class_names, "classes": land_use.classes.tolist(),
         "confidence": land_use.confidence.round(3).tolist(), "shares": land_use.shares(),
         "model_val_accuracy": metrics.get("val_accuracy")}, indent=1))
    print(summary)
    print(f"\nSaved {folder / 'map.png'}")
