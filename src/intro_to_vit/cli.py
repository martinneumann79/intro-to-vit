"""Command-line entry point: `uv run compare-models --help`."""

import argparse

from intro_to_vit.compare import run_comparison
from intro_to_vit.models import MODEL_NAMES


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Compare a Vision Transformer with a CNN on CIFAR-10.")
    parser.add_argument("--models", nargs="+", default=list(MODEL_NAMES), choices=MODEL_NAMES)
    parser.add_argument(
        "--fractions", nargs="+", type=float, default=[1.0],
        help="Fractions of the training set to use, e.g. 0.1 1.0 to test data efficiency.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the comparison described by the command-line arguments."""
    args = parse_args(argv)
    run_comparison(
        models=args.models,
        fractions=args.fractions,
        epochs=args.epochs,
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
