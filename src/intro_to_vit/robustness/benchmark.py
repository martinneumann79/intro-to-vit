"""Measure how each model's accuracy falls as a perturbation gets stronger."""

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import torch
from torch.utils.data import DataLoader

from intro_to_vit.robustness.imagenet_data import ImageListDataset, LabeledImage
from intro_to_vit.robustness.perturbations import Perturbation
from intro_to_vit.robustness.pretrained import Classifier


@dataclass
class BenchmarkResult:
    """Accuracy per model and severity, plus a paired comparison of the first two models."""

    dataset: str
    perturbation: str
    severities: list[float]
    models: list[dict]
    accuracy: dict[str, list[float]]
    paired: list[dict]
    num_images: int
    seed: int
    seconds: float
    correct: dict[str, torch.Tensor] = field(repr=False, default_factory=dict)


def image_seeds(seed: int, indices: Sequence[int]) -> list[int]:
    """Return one deterministic random seed per image index."""
    return [seed * 1_000_003 + int(i) for i in indices]


def mcnemar_p_value(only_first: int, only_second: int) -> float:
    """Exact two-sided McNemar test on the images where exactly one model is right.

    Returns the probability of a split at least this uneven if both models were equally good.
    """
    n = only_first + only_second
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(only_first, only_second) + 1))
    return min(1.0, 2 * tail / 2**n)


def paired_counts(first: torch.Tensor, second: torch.Tensor) -> dict:
    """Count images where both, only the first, only the second, or neither model is right."""
    only_first = int((first & ~second).sum())
    only_second = int((~first & second).sum())
    return {
        "both": int((first & second).sum()),
        "only_first": only_first,
        "only_second": only_second,
        "neither": int((~first & ~second).sum()),
        "p_value": mcnemar_p_value(only_first, only_second),
    }


def run_benchmark(
    classifiers: list[Classifier],
    images: list[LabeledImage],
    perturbation: Perturbation,
    severities: Sequence[float],
    dataset: str,
    seed: int = 0,
    batch_size: int = 50,
    num_workers: int = 4,
    log: Callable[[str], None] = print,
) -> BenchmarkResult:
    """Evaluate every classifier on every image at every severity of `perturbation`."""
    loader = DataLoader(ImageListDataset(images), batch_size=batch_size, num_workers=num_workers)
    correct = {c.spec.alias: torch.zeros(len(severities), len(images), dtype=torch.bool) for c in classifiers}
    start = time.perf_counter()
    done = 0
    log_every = max(1, len(loader) // 10)
    for batch_number, (batch, labels, indices) in enumerate(loader, start=1):
        seeds = image_seeds(seed, indices.tolist())
        for row, severity in enumerate(severities):
            corrupted = perturbation.apply(batch, severity, seeds)
            for classifier in classifiers:
                predictions = classifier.predict_proba(corrupted).argmax(dim=1)
                correct[classifier.spec.alias][row, indices] = predictions == labels
        done += len(labels)
        if batch_number % log_every == 0 or done == len(images):
            log(f"  {done:>6,}/{len(images):,} images  ({time.perf_counter() - start:.0f}s)")
    aliases = list(correct)
    paired = []
    if len(aliases) >= 2:
        first, second = correct[aliases[0]], correct[aliases[1]]
        paired = [paired_counts(first[row], second[row]) for row in range(len(severities))]
    return BenchmarkResult(
        dataset=dataset,
        perturbation=perturbation.key,
        severities=[float(s) for s in severities],
        models=[c.describe() for c in classifiers],
        accuracy={alias: c.float().mean(dim=1).tolist() for alias, c in correct.items()},
        paired=paired,
        num_images=len(images),
        seed=seed,
        seconds=time.perf_counter() - start,
        correct=correct,
    )
