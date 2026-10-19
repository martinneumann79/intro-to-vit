"""Command-line entry points for the robustness benchmark, the demo app and data download."""

import argparse
from pathlib import Path

from intro_to_vit.robustness.imagenet_data import DATASETS, load_dataset
from intro_to_vit.robustness.perturbations import PERTURBATIONS
from intro_to_vit.robustness.pretrained import MODEL_SPECS, load_classifiers
from intro_to_vit.training import get_device

DEFAULT_MODELS = ["vit", "cnn"]
MODEL_HELP = (
    "Model aliases (" + ", ".join(MODEL_SPECS) + ") or any timm model name. The first two are compared pairwise."
)


def parse_benchmark_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse arguments for `robustness-benchmark`."""
    parser = argparse.ArgumentParser(
        description="Measure ViT vs CNN accuracy under increasingly strong image corruptions."
    )
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help=MODEL_HELP)
    parser.add_argument("--perturbations", nargs="+", default=["noise"], choices=list(PERTURBATIONS))
    parser.add_argument("--dataset", default="imagenette", choices=DATASETS)
    parser.add_argument("--severities", nargs="+", type=float, help="Override the default severity sweep.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="results/robustness")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--examples", type=int, default=3, help="Images in the example figure (0 to skip).")
    return parser.parse_args(argv)


def benchmark_main(argv: list[str] | None = None) -> None:
    """Run the benchmark for each requested perturbation and save tables and figures."""
    from intro_to_vit.robustness.benchmark import run_benchmark
    from intro_to_vit.robustness.plots import plot_accuracy, plot_examples
    from intro_to_vit.robustness.report import format_markdown, save_result

    args = parse_benchmark_args(argv)
    device = get_device()
    images = load_dataset(args.dataset, args.data_dir)
    classifiers = load_classifiers(args.models, device)
    print(f"Device: {device}  dataset: {args.dataset} ({len(images):,} images)")
    for key in args.perturbations:
        perturbation = PERTURBATIONS[key]
        severities = args.severities or perturbation.severities
        print(f"\n== {perturbation.title}: {', '.join(c.spec.display_name for c in classifiers)}")
        result = run_benchmark(classifiers, images, perturbation, severities, args.dataset,
                               seed=args.seed, batch_size=args.batch_size, num_workers=args.workers)
        out = save_result(result, Path(args.out_dir) / args.dataset / key)
        for theme in ("light", "dark"):
            suffix = "" if theme == "light" else "-dark"
            plot_accuracy(result, out / f"accuracy{suffix}.png", theme)
            if args.examples:
                plot_examples(classifiers, images, perturbation, out / f"examples{suffix}.jpg",
                              count=args.examples, seed=args.seed, theme=theme)
        print("\n" + format_markdown(result))
        print(f"\nSaved to {out}/ ({result.seconds:.0f}s)")


def parse_demo_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse arguments for `robustness-demo`."""
    parser = argparse.ArgumentParser(description="Interactive ViT vs CNN demo in the browser.")
    parser.add_argument("--models", nargs=2, default=DEFAULT_MODELS, help=MODEL_HELP)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--results-dir", default="results/robustness")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser tab.")
    return parser.parse_args(argv)


def demo_main(argv: list[str] | None = None) -> None:
    """Start the Gradio demo app."""
    from intro_to_vit.robustness.app import launch

    args = parse_demo_args(argv)
    launch(args.models, args.data_dir, args.results_dir, args.port, open_browser=not args.no_browser)


def download_main(argv: list[str] | None = None) -> None:
    """Download datasets and model weights so the demo also works offline."""
    parser = argparse.ArgumentParser(description="Download evaluation images and pretrained weights.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help=MODEL_HELP + " Use 'all' for every alias.")
    parser.add_argument("--datasets", nargs="+", default=["imagenette"], choices=DATASETS)
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args(argv)
    for name in args.datasets:
        print(f"{name}: {len(load_dataset(name, args.data_dir)):,} images")
    names = list(MODEL_SPECS) if args.models == ["all"] else args.models
    for classifier in load_classifiers(names, get_device()):
        print(f"{classifier.spec.display_name}: {classifier.num_parameters / 1e6:.1f}M parameters")
