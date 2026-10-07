"""Architecture-level metrics: size and inference speed."""

import time

import torch
from torch import nn


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters in `model`."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _synchronize(device: torch.device) -> None:
    """Wait for queued GPU work so wall-clock timings are accurate."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def measure_throughput(
    model: nn.Module,
    device: torch.device,
    batch_size: int = 256,
    image_size: int = 32,
    iterations: int = 20,
    warmup: int = 5,
) -> float:
    """Return inference throughput in images per second on random input."""
    model.eval().to(device)
    x = torch.randn(batch_size, 3, image_size, image_size, device=device)
    for _ in range(warmup):
        model(x)
    _synchronize(device)
    start = time.perf_counter()
    for _ in range(iterations):
        model(x)
    _synchronize(device)
    return batch_size * iterations / (time.perf_counter() - start)
