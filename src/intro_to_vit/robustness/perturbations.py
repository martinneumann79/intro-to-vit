"""Image corruptions controlled by a single severity value.

Each function takes a batch of images with values in [0, 1], shape (B, 3, H, W), a severity and
one seed per image, and returns a corrupted copy. The random pattern depends only on the seed:
raising the severity strengthens the same noise or occlusion pattern, and every model sees
exactly the same corrupted pixels.
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial

import torch
from torchvision.transforms.functional import gaussian_blur as tv_gaussian_blur

PerturbFn = Callable[[torch.Tensor, float, Sequence[int]], torch.Tensor]


def per_image_random(seeds: Sequence[int], sample: Callable[[torch.Generator], torch.Tensor]) -> torch.Tensor:
    """Draw one random tensor per seed with its own generator and stack them."""
    return torch.stack([sample(torch.Generator().manual_seed(int(seed))) for seed in seeds])


def gaussian_noise(images: torch.Tensor, sigma: float, seeds: Sequence[int]) -> torch.Tensor:
    """Add Gaussian noise with standard deviation `sigma` to every pixel value."""
    if sigma == 0:
        return images.clone()
    _, channels, height, width = images.shape
    noise = per_image_random(seeds, lambda g: torch.randn(channels, height, width, generator=g))
    return (images + sigma * noise).clamp(0, 1)


def salt_and_pepper(images: torch.Tensor, amount: float, seeds: Sequence[int]) -> torch.Tensor:
    """Turn a fraction `amount` of the pixels pure black or pure white (impulse noise)."""
    _, _, height, width = images.shape
    u = per_image_random(seeds, lambda g: torch.rand(1, height, width, generator=g))
    out = torch.where(u < amount / 2, torch.zeros_like(images), images)
    return torch.where(u > 1 - amount / 2, torch.ones_like(images), out)


def gaussian_blur(images: torch.Tensor, sigma: float, seeds: Sequence[int]) -> torch.Tensor:
    """Blur with a Gaussian kernel of standard deviation `sigma` pixels; `seeds` is unused."""
    if sigma == 0:
        return images.clone()
    size = 2 * math.ceil(3 * sigma) + 1
    return tv_gaussian_blur(images, [size, size], [sigma, sigma])


def patch_occlusion(
    images: torch.Tensor, ratio: float, seeds: Sequence[int], patch_size: int = 16, offset: int = 0
) -> torch.Tensor:
    """Black out a fraction `ratio` of the squares in a grid of `patch_size` pixel squares.

    With offset 0 the grid matches the 16x16 patches a ViT-*/16 model cuts the image into. A
    non-zero offset shifts the grid, so each black square covers parts of four ViT patches.
    """
    batch, _, height, width = images.shape
    rows = math.ceil((height + offset) / patch_size)
    cols = math.ceil((width + offset) / patch_size)
    ranks = per_image_random(seeds, lambda g: torch.randperm(rows * cols, generator=g))
    keep = (ranks >= round(ratio * rows * cols)).float().view(batch, 1, rows, cols)
    keep = keep.repeat_interleave(patch_size, 2).repeat_interleave(patch_size, 3)
    return images * keep[:, :, offset : offset + height, offset : offset + width]


@dataclass(frozen=True)
class Perturbation:
    """A corruption type with its benchmark sweep and slider settings for the demo app."""

    key: str
    title: str
    severity_label: str
    severities: tuple[float, ...]
    maximum: float
    step: float
    default: float
    examples: tuple[float, ...]
    apply: PerturbFn


OCCLUSION_RATIOS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

PERTURBATIONS: dict[str, Perturbation] = {
    p.key: p
    for p in (
        Perturbation("noise", "Gaussiskt brus", "Brusnivå σ (pixelvärden 0–1)",
                     (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5), 0.6, 0.01, 0.3,
                     (0.0, 0.2, 0.3, 0.4), gaussian_noise),
        Perturbation("salt-pepper", "Salt-och-peppar-brus", "Andel förstörda pixlar",
                     (0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3), 0.5, 0.01, 0.1,
                     (0.0, 0.05, 0.1, 0.2), salt_and_pepper),
        Perturbation("blur", "Oskärpa", "Oskärpa σ (pixlar)",
                     (0.0, 1.0, 2.0, 3.0, 4.0, 6.0), 8.0, 0.5, 3.0,
                     (0.0, 2.0, 3.0, 4.0), gaussian_blur),
        Perturbation("occlusion", "Ocklusion i linje med ViT:s rutnät", "Andel svarta rutor",
                     OCCLUSION_RATIOS, 0.95, 0.05, 0.5, (0.0, 0.3, 0.5, 0.7), patch_occlusion),
        Perturbation("occlusion-shifted", "Ocklusion förskjuten 8 px", "Andel svarta rutor",
                     OCCLUSION_RATIOS, 0.95, 0.05, 0.5, (0.0, 0.3, 0.5, 0.7),
                     partial(patch_occlusion, offset=8)),
    )
}
