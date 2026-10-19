"""The Vision Transformer that classifies land use, and saving and loading it."""

from pathlib import Path

import timm
import torch
from torch import nn

MODEL_NAME = "deit_small_patch16_224.fb_in1k"  # ViT-S/16, 22M parameters, pretrained on ImageNet


def create_vit(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Create the ViT with a new classification head for `num_classes` classes."""
    return timm.create_model(MODEL_NAME, pretrained=pretrained, num_classes=num_classes)


def save_checkpoint(model: nn.Module, class_names: list[str], metrics: dict, path: str | Path) -> Path:
    """Save the fine-tuned weights together with the class names and training metrics."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_name": MODEL_NAME, "class_names": class_names, "metrics": metrics,
                "state_dict": model.state_dict()}, path)
    return path


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[nn.Module, list[str], dict]:
    """Load a fine-tuned ViT saved by `save_checkpoint`; returns (model, class_names, metrics)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Train the model first: make satellite-train")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = create_vit(len(checkpoint["class_names"]), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval().to(device), checkpoint["class_names"], checkpoint["metrics"]
