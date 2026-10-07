"""Model definitions and a name-based factory."""

from torch import nn

from intro_to_vit.models.cnn import SmallResNet
from intro_to_vit.models.vit import VisionTransformer

MODEL_NAMES = ("vit", "cnn")


def build_model(name: str, num_classes: int = 10) -> nn.Module:
    """Create a model by name ('vit' or 'cnn') with CIFAR-10 sized defaults."""
    if name == "vit":
        return VisionTransformer(num_classes=num_classes)
    if name == "cnn":
        return SmallResNet(num_classes=num_classes)
    raise ValueError(f"Unknown model {name!r}; choose from {MODEL_NAMES}")


__all__ = ["MODEL_NAMES", "SmallResNet", "VisionTransformer", "build_model"]
