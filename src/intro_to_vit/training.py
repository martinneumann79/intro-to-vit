"""Training and evaluation loops shared by both architectures."""

import math
import time
from collections.abc import Iterable

import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR


def get_device() -> torch.device:
    """Pick the best available device: CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def warmup_cosine_schedule(optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int) -> LambdaLR:
    """Linear warmup followed by cosine decay to zero. ViTs train poorly without warmup."""

    def factor(step: int) -> float:
        """Return the learning-rate multiplier for `step`."""
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    return LambdaLR(optimizer, factor)


def train_one_epoch(
    model: nn.Module,
    loader: Iterable,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR | None,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Train for one pass over `loader`; return (mean loss, accuracy)."""
    model.train()
    # Accumulate on-device; calling .item() every step would force a GPU sync.
    total_loss = torch.zeros((), device=device)
    correct = torch.zeros((), device=device, dtype=torch.long)
    seen = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = loss_fn(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.detach() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum()
        seen += labels.size(0)
    return total_loss.item() / seen, correct.item() / seen


@torch.no_grad()
def evaluate(model: nn.Module, loader: Iterable, device: torch.device) -> float:
    """Return classification accuracy of `model` on `loader`."""
    model.eval()
    correct = torch.zeros((), device=device, dtype=torch.long)
    seen = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        correct += (model(images).argmax(1) == labels).sum()
        seen += labels.size(0)
    return correct.item() / seen


def fit(
    model: nn.Module,
    train_loader,
    test_loader,
    epochs: int,
    device: torch.device,
    lr: float = 1e-3,
    weight_decay: float = 0.05,
    warmup_epochs: int = 1,
    log=print,
) -> dict:
    """Train `model` with AdamW + warmup/cosine and return a history dict."""
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = len(train_loader)
    scheduler = warmup_cosine_schedule(optimizer, warmup_epochs * steps_per_epoch, epochs * steps_per_epoch)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    history = {"train_loss": [], "train_acc": [], "test_acc": [], "epoch_seconds": []}
    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        loss, acc = train_one_epoch(model, train_loader, optimizer, scheduler, loss_fn, device)
        elapsed = time.perf_counter() - start
        test_acc = evaluate(model, test_loader, device)
        for key, value in zip(history, (loss, acc, test_acc, elapsed)):
            history[key].append(value)
        log(f"  epoch {epoch:3d}/{epochs}  loss {loss:.3f}  train {acc:.3f}  test {test_acc:.3f}  ({elapsed:.1f}s)")
    return history
