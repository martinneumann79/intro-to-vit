"""Classify a Sentinel-2 scene patch by patch with the fine-tuned ViT and draw a land-use map."""

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from torch import nn  # noqa: E402

from intro_to_vit.robustness.plots import FONT_FAMILY, PX, THEMES  # noqa: E402
from intro_to_vit.satellite.eurosat import (  # noqa: E402
    CLASS_COLORS,
    PATCH_SIZE,
    SWEDISH_NAMES,
    vit_input_transform,
)

NO_DATA = -1


@dataclass
class LandUseMap:
    """The predicted class and confidence for every patch of a scene."""

    class_names: list[str]
    classes: np.ndarray  # (rows, cols) class index, NO_DATA where the scene has no pixels
    confidence: np.ndarray  # (rows, cols) probability of the predicted class

    def shares(self) -> dict[str, float]:
        """Return each class's share of the patches that have data, largest first."""
        valid = self.classes[self.classes != NO_DATA]
        counts = {name: int((valid == i).sum()) for i, name in enumerate(self.class_names)}
        total = max(1, len(valid))
        return dict(sorted(((n, c / total) for n, c in counts.items() if c), key=lambda kv: -kv[1]))

    def mean_confidence(self) -> dict[str, float]:
        """Return the mean probability the model gave its prediction, per class, in the order of `shares`."""
        return {
            name: float(self.confidence[self.classes == self.class_names.index(name)].mean())
            for name in self.shares()
        }


def format_share(share: float) -> str:
    """Format a share in Swedish style, e.g. 0.27 -> '27 %', and anything below 0.5 % as '<1 %'."""
    return "<1 %" if share < 0.005 else f"{share * 100:.0f} %"


def tile_image(image: np.ndarray, patch: int = PATCH_SIZE) -> tuple[torch.Tensor, tuple[int, int]]:
    """Cut an (H, W, 3) uint8 image into (N, 3, patch, patch) tiles in [0, 1], row by row.

    A strip at the right or bottom edge narrower than one patch is left out.
    """
    rows, cols = image.shape[0] // patch, image.shape[1] // patch
    pixels = torch.from_numpy(np.ascontiguousarray(image[: rows * patch, : cols * patch])).permute(2, 0, 1)
    tiles = pixels.float().div(255).unfold(1, patch, patch).unfold(2, patch, patch)  # (3, rows, cols, p, p)
    return tiles.permute(1, 2, 0, 3, 4).reshape(rows * cols, 3, patch, patch), (rows, cols)


@torch.inference_mode()
def classify_tiles(model: nn.Module, tiles: torch.Tensor, device: torch.device, batch_size: int = 128) -> torch.Tensor:
    """Return class probabilities, shape (N, classes), for every tile."""
    prepare = vit_input_transform()
    model.eval()
    return torch.cat([model(prepare(batch).to(device)).softmax(dim=1).cpu() for batch in tiles.split(batch_size)])


def classify_scene(model: nn.Module, class_names: list[str], image: np.ndarray, device: torch.device) -> LandUseMap:
    """Classify every 64x64 patch of `image`; patches that are mostly black (no data) are skipped."""
    tiles, (rows, cols) = tile_image(image)
    probabilities = classify_tiles(model, tiles, device)
    confidence, classes = probabilities.max(dim=1)
    no_data = (tiles.amax(dim=1) < 1 / 255).float().mean(dim=(1, 2)) > 0.5
    classes[no_data] = NO_DATA
    return LandUseMap(class_names, classes.view(rows, cols).numpy(), confidence.view(rows, cols).numpy())


def format_summary(land_use: LandUseMap, title: str) -> str:
    """Render each class's share and the model's mean confidence as a Markdown table."""
    lines = [title, "", "| Klass | Andel av rutorna | Medelsäkerhet |", "|---|---|---|"]
    confidences = land_use.mean_confidence()
    for name, share in land_use.shares().items():
        lines.append(f"| {SWEDISH_NAMES.get(name, name)} | {format_share(share)} | {format_share(confidences[name])} |")
    return "\n".join(lines)


def class_overlay(land_use: LandUseMap, alpha: float = 0.55) -> np.ndarray:
    """Return an RGBA image (rows, cols, 4) with each patch in its class colour; no data is transparent."""
    overlay = np.zeros((*land_use.classes.shape, 4))
    for index, name in enumerate(land_use.class_names):
        overlay[land_use.classes == index] = to_rgba(CLASS_COLORS.get(name, "#000000"), alpha)
    return overlay


def render_map(
    image: np.ndarray, land_use: LandUseMap, path: str | Path, title: str, subtitle: str, credit: str = ""
) -> Path:
    """Save the scene and the ViT's land-use map side by side, with each class's mean confidence in the legend.

    `credit` is printed in the bottom-right corner, e.g. the attribution Copernicus data requires.
    """
    colors = THEMES["light"]
    rows, cols = land_use.classes.shape
    extent = (0, cols * PATCH_SIZE, rows * PATCH_SIZE, 0)
    shown = image[: rows * PATCH_SIZE, : cols * PATCH_SIZE]
    with plt.rc_context({"font.family": FONT_FAMILY}):
        fig, axes = plt.subplots(1, 2, figsize=(16, 8.4), dpi=100)
        fig.patch.set_facecolor(colors["surface"])
        for ax, label in zip(axes, ("Sentinel-2, sann färg", "ViT:ens klassning, en ruta = 640 × 640 m")):
            ax.imshow(shown, extent=extent)
            ax.set_title(label, fontsize=14 * PX * 1.4, color=colors["ink2"], loc="left", pad=8)
            ax.set_axis_off()
        axes[1].imshow(class_overlay(land_use), extent=extent, interpolation="nearest")
        handles = [
            Patch(facecolor=CLASS_COLORS.get(name, "#000000"), edgecolor="none",
                  label=f"{SWEDISH_NAMES.get(name, name)} · {format_share(confidence)}")
            for name, confidence in land_use.mean_confidence().items()
        ]
        legend = fig.legend(handles=handles, loc="lower center", ncol=min(5, len(handles)), frameon=False,
                            fontsize=13 * PX * 1.4, bbox_to_anchor=(0.5, 0.01),
                            title="Modellens medelsäkerhet per klass", title_fontsize=13 * PX * 1.4)
        legend.get_title().set_color(colors["ink2"])
        for text in legend.get_texts():
            text.set_color(colors["ink"])
        fig.text(0.02, 0.955, title, fontsize=18 * PX * 1.4, fontweight="bold", color=colors["ink"])
        fig.text(0.02, 0.915, subtitle, fontsize=13 * PX * 1.4, color=colors["ink2"])
        if credit:
            fig.text(0.98, 0.012, credit, fontsize=11 * PX * 1.4, color=colors["muted"], ha="right")
        fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.155, wspace=0.04)
        path = Path(path)
        fig.savefig(path, dpi=150, facecolor=colors["surface"])
        plt.close(fig)
    return path
