"""Fine-tune the ViT on EuroSAT and report how well it recognises each land-use class."""

import time
from collections.abc import Callable
from pathlib import Path

import torch
from torch import nn

from intro_to_vit.satellite.eurosat import build_loaders
from intro_to_vit.satellite.model import create_vit, save_checkpoint
from intro_to_vit.training import evaluate, get_device, train_one_epoch, warmup_cosine_schedule


@torch.inference_mode()
def predict_all(model: nn.Module, loader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (predictions, targets) for every example in `loader`."""
    model.eval()
    predictions, targets = [], []
    for images, labels in loader:
        predictions.append(model(images.to(device)).argmax(dim=1).cpu())
        targets.append(labels)
    return torch.cat(predictions), torch.cat(targets)


def per_class_accuracy(predictions: torch.Tensor, targets: torch.Tensor, class_names: list[str]) -> dict[str, float]:
    """Return the share of correctly classified examples for each class."""
    return {
        name: float((predictions[targets == i] == i).float().mean())
        for i, name in enumerate(class_names)
    }


def fine_tune(
    data_dir: str = "data",
    out_path: str | Path = "results/satellite/vit-eurosat.pt",
    epochs: int = 3,
    batch_size: int = 64,
    lr: float = 1e-4,
    seed: int = 0,
    num_workers: int = 4,
    log: Callable[[str], None] = print,
) -> dict:
    """Fine-tune the pretrained ViT on 80% of EuroSAT, evaluate on the other 20% and save it."""
    torch.manual_seed(seed)
    device = get_device()
    train_loader, val_loader, class_names = build_loaders(data_dir, batch_size, seed=seed, num_workers=num_workers)
    model = create_vit(len(class_names)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    steps = epochs * len(train_loader)
    scheduler = warmup_cosine_schedule(optimizer, warmup_steps=len(train_loader) // 2, total_steps=steps)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    log(f"Device: {device}  train: {len(train_loader.dataset):,}  validation: {len(val_loader.dataset):,} images")
    history = []
    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        loss, train_acc = train_one_epoch(model, train_loader, optimizer, scheduler, loss_fn, device)
        val_acc = evaluate(model, val_loader, device)
        seconds = time.perf_counter() - start
        history.append({"epoch": epoch, "loss": loss, "train_acc": train_acc, "val_acc": val_acc, "seconds": seconds})
        log(f"  epoch {epoch}/{epochs}  loss {loss:.3f}  train {train_acc:.3f}  val {val_acc:.3f}  ({seconds:.0f}s)")
    predictions, targets = predict_all(model, val_loader, device)
    metrics = {
        "val_accuracy": float((predictions == targets).float().mean()),
        "per_class_accuracy": per_class_accuracy(predictions, targets, class_names),
        "epochs": epochs,
        "history": history,
    }
    path = save_checkpoint(model.cpu(), class_names, metrics, out_path)
    log(f"Validation accuracy {metrics['val_accuracy']:.1%}. Saved to {path}")
    return metrics
