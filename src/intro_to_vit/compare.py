"""Run the ViT-vs-CNN experiment grid and summarise the results."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from intro_to_vit.data import build_loaders
from intro_to_vit.metrics import count_parameters, measure_throughput
from intro_to_vit.models import build_model
from intro_to_vit.training import fit, get_device


@dataclass
class RunResult:
    """Outcome of training one model on one fraction of the training data."""

    model: str
    train_fraction: float
    parameters: int
    best_test_acc: float
    final_test_acc: float
    final_train_acc: float
    train_seconds: float
    images_per_second: float
    history: dict = field(default_factory=dict)


def run_single(model_name: str, train_fraction: float, epochs: int, batch_size: int, data_dir: str, seed: int, device: torch.device) -> RunResult:
    """Train and evaluate one model on one data fraction."""
    torch.manual_seed(seed)
    model = build_model(model_name)
    train_loader, test_loader = build_loaders(data_dir, batch_size, train_fraction, seed=seed)
    print(f"\n== {model_name.upper()}  fraction={train_fraction}  params={count_parameters(model):,}  train images={len(train_loader.dataset):,}")
    history = fit(model, train_loader, test_loader, epochs, device)
    return RunResult(
        model=model_name,
        train_fraction=train_fraction,
        parameters=count_parameters(model),
        best_test_acc=max(history["test_acc"]),
        final_test_acc=history["test_acc"][-1],
        final_train_acc=history["train_acc"][-1],
        train_seconds=sum(history["epoch_seconds"]),
        images_per_second=measure_throughput(model, device),
        history=history,
    )


def format_table(results: list[RunResult]) -> str:
    """Render results as a Markdown table."""
    lines = [
        "| model | data fraction | params | best test acc | final train acc | train time (s) | inference img/s |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in sorted(results, key=lambda r: (r.train_fraction, r.model)):
        lines.append(
            f"| {r.model} | {r.train_fraction:g} | {r.parameters:,} | {r.best_test_acc:.3f} "
            f"| {r.final_train_acc:.3f} | {r.train_seconds:.0f} | {r.images_per_second:,.0f} |"
        )
    return "\n".join(lines)


def save_results(results: list[RunResult], out_dir: str | Path) -> Path:
    """Write results.json and results.md to `out_dir`; return the directory."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps([asdict(r) for r in results], indent=2))
    (out / "results.md").write_text(format_table(results) + "\n")
    return out


def run_comparison(
    models: list[str],
    fractions: list[float],
    epochs: int,
    batch_size: int = 128,
    data_dir: str = "data",
    out_dir: str = "results",
    seed: int = 0,
) -> list[RunResult]:
    """Train every model on every data fraction, save and print a summary table."""
    device = get_device()
    print(f"Device: {device}")
    results = [
        run_single(m, f, epochs, batch_size, data_dir, seed, device) for f in fractions for m in models
    ]
    save_results(results, out_dir)
    print("\n" + format_table(results))
    print(f"\nSaved to {out_dir}/results.json and {out_dir}/results.md")
    return results
