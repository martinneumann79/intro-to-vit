"""Pretrained ImageNet-1k classifiers (loaded with timm) behind one prediction interface."""

from collections.abc import Iterable
from dataclasses import dataclass

import timm
import torch
from torch import nn

from intro_to_vit.metrics import count_parameters


@dataclass(frozen=True)
class ModelSpec:
    """A pretrained ImageNet-1k model and how to present it."""

    alias: str
    timm_name: str
    display_name: str
    family: str
    description: str


# Every model here was trained on ImageNet-1k only (1.3M images), with modern augmentation.
# The default pair, "vit" vs "cnn", is matched on size (~22M vs ~26M parameters) and on clean
# ImageNet accuracy (~80% vs ~80%), so the architecture is the main difference.
MODEL_SPECS: dict[str, ModelSpec] = {
    spec.alias: spec
    for spec in (
        ModelSpec("vit", "deit_small_patch16_224.fb_in1k", "ViT-S/16 (DeiT)", "transformer",
                  "Plain Vision Transformer, 16x16 patches, DeiT training recipe"),
        ModelSpec("cnn", "resnet50.a1_in1k", "ResNet-50", "cnn",
                  "Classic CNN trained with the 'ResNet strikes back' A1 recipe"),
        ModelSpec("vit-base", "deit_base_patch16_224.fb_in1k", "ViT-B/16 (DeiT)", "transformer",
                  "Larger plain Vision Transformer (86M parameters)"),
        ModelSpec("convnext", "convnext_tiny.fb_in1k", "ConvNeXt-T", "cnn",
                  "Modern CNN (2022) that borrows design choices from transformers"),
        ModelSpec("swin", "swin_tiny_patch4_window7_224.ms_in1k", "Swin-T", "transformer",
                  "Hierarchical transformer that attends within local windows"),
        ModelSpec("resnet50-2015", "resnet50.tv_in1k", "ResNet-50 (2015-recept)", "cnn",
                  "Original torchvision ResNet-50 weights, simple augmentation"),
    )
}


def resolve_spec(name: str) -> ModelSpec:
    """Return the spec for a known alias, or wrap any other timm model name."""
    if name in MODEL_SPECS:
        return MODEL_SPECS[name]
    return ModelSpec(name, name, name, "unknown", "timm model")


class Classifier:
    """A pretrained ImageNet classifier that takes 224x224 RGB images with values in [0, 1]."""

    def __init__(self, spec: ModelSpec, device: torch.device):
        """Load the pretrained weights (downloaded on first use) onto `device`."""
        self.spec = spec
        self.device = device
        self.model: nn.Module = timm.create_model(spec.timm_name, pretrained=True).eval().to(device)
        config = self.model.pretrained_cfg
        self.mean = torch.tensor(config["mean"], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(config["std"], device=device).view(1, 3, 1, 1)
        self.num_parameters = count_parameters(self.model)

    @torch.inference_mode()
    def predict_proba(self, images: torch.Tensor) -> torch.Tensor:
        """Return class probabilities of shape (batch, 1000) as a CPU tensor."""
        x = (images.to(self.device) - self.mean) / self.std
        return self.model(x).softmax(dim=1).cpu()

    def describe(self) -> dict:
        """Return a JSON-friendly description of the model."""
        return {
            "alias": self.spec.alias,
            "timm_name": self.spec.timm_name,
            "display_name": self.spec.display_name,
            "family": self.spec.family,
            "parameters": self.num_parameters,
        }


def load_classifiers(names: Iterable[str], device: torch.device) -> list[Classifier]:
    """Load one classifier per alias or timm model name, skipping duplicates."""
    unique = list(dict.fromkeys(names))
    return [Classifier(resolve_spec(name), device) for name in unique]
