"""Slide-ready figures: accuracy curves and example images with each model's prediction."""

import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from intro_to_vit.robustness.benchmark import BenchmarkResult, image_seeds  # noqa: E402
from intro_to_vit.robustness.imagenet_data import (  # noqa: E402
    DATASET_TITLES,
    LabeledImage,
    imagenet_class_names,
    load_image,
)
from intro_to_vit.robustness.perturbations import PERTURBATIONS, Perturbation  # noqa: E402
from intro_to_vit.robustness.pretrained import Classifier  # noqa: E402

# Chart chrome per theme. Light and dark are separate, validated steps, not an automatic flip.
THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "axis": "#c3c2b7", "good": "#0ca30c", "bad": "#d03b3b"},
    "dark": {"surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781",
             "grid": "#2c2c2a", "axis": "#383835", "good": "#0ca30c", "bad": "#d03b3b"},
}
# Categorical slots in fixed order (validated for colour-vision deficiencies in both themes).
SERIES_COLORS = {
    "light": ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"),
    "dark": ("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"),
}
# A model keeps the same colour in every chart, whatever else is plotted with it.
MODEL_SLOTS = {"vit": 0, "cnn": 1, "vit-base": 2, "convnext": 3, "swin": 4, "resnet50-2015": 5}
FAMILY_MARKERS = {"transformer": "o", "cnn": "s"}
FONT_FAMILY = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
PX = 72 / 100  # points per pixel at the 1x layout density (figures are saved at 2x)


def swedish_number(value: float) -> str:
    """Format a number with a decimal comma, e.g. 0.25 -> '0,25'."""
    return f"{value:g}".replace(".", ",")


def swedish_millions(count: int) -> str:
    """Format a parameter count in millions with a decimal comma, e.g. 22052968 -> '22,1 M'."""
    return f"{count / 1e6:.1f} M".replace(".", ",")


def assign_colors(aliases: list[str], theme: str) -> dict[str, str]:
    """Give known models their fixed slot and unknown models the first free slots."""
    palette = SERIES_COLORS[theme]
    used = {MODEL_SLOTS[a] for a in aliases if a in MODEL_SLOTS}
    free = iter(i for i in range(len(palette)) if i not in used)
    return {a: palette[MODEL_SLOTS[a] if a in MODEL_SLOTS else next(free) % len(palette)] for a in aliases}


def style_axes(ax: plt.Axes, colors: dict) -> None:
    """Apply recessive hairline axes and grid on the chart surface."""
    ax.set_facecolor(colors["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(colors["axis"])
    ax.spines["bottom"].set_linewidth(1 * PX)
    ax.grid(axis="y", color=colors["grid"], linewidth=1 * PX)
    ax.set_axisbelow(True)
    ax.tick_params(colors=colors["muted"], labelcolor=colors["ink2"], length=0, labelsize=13 * PX)


def label_index(result: BenchmarkResult) -> int:
    """Return the severity index to annotate: the demo's default severity, else the last one."""
    default = PERTURBATIONS[result.perturbation].default
    return result.severities.index(default) if default in result.severities else len(result.severities) - 1


def plot_accuracy(result: BenchmarkResult, path: str | Path, theme: str = "light") -> Path:
    """Plot accuracy against severity for every model and save it as a PNG."""
    colors = THEMES[theme]
    perturbation = PERTURBATIONS[result.perturbation]
    aliases = [m["alias"] for m in result.models]
    series_colors = assign_colors(aliases, theme)
    with plt.rc_context({"font.family": FONT_FAMILY}):
        fig, ax = plt.subplots(figsize=(9.6, 5.6), dpi=100)
        fig.patch.set_facecolor(colors["surface"])
        style_axes(ax, colors)
        x = result.severities
        mark = label_index(result)
        ax.axvline(x[mark], color=colors["grid"], linewidth=1 * PX, zorder=0)
        for model in result.models:
            y = [100 * a for a in result.accuracy[model["alias"]]]
            ax.plot(x, y, color=series_colors[model["alias"]], linewidth=2 * PX, solid_capstyle="round",
                    solid_joinstyle="round", marker=FAMILY_MARKERS.get(model["family"], "D"),
                    markersize=8 * PX, markeredgecolor=colors["surface"], markeredgewidth=2 * PX,
                    label=f"{model['display_name']} · {swedish_millions(model['parameters'])} parametrar", zorder=3)
            if len(result.models) <= 2:
                ax.annotate(f"{y[mark]:.0f} %", (x[mark], y[mark]), xytext=(8, 6), textcoords="offset points",
                            va="bottom", fontsize=14 * PX, fontweight="bold", color=colors["ink"],
                            bbox={"boxstyle": "round,pad=0.25", "facecolor": colors["surface"], "edgecolor": "none"})
        ax.set_ylim(0, 100)
        ax.set_xlim(min(x), max(x) * 1.04)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f} %"))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: swedish_number(round(v, 3))))
        ax.set_xticks(x)
        ax.set_xlabel(perturbation.severity_label, color=colors["ink2"], fontsize=13 * PX, labelpad=8)
        fig.text(0.07, 0.95, f"Träffsäkerhet med {perturbation.title.lower()}", fontsize=18 * PX,
                 fontweight="bold", color=colors["ink"])
        count = f"{result.num_images:,}".replace(",", " ")
        fig.text(0.07, 0.905, f"{count} bilder som modellerna aldrig sett ({DATASET_TITLES[result.dataset]}) · "
                 "rätt klass bland 1000 möjliga", fontsize=13 * PX, color=colors["ink2"])
        legend = ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=min(len(result.models), 3),
                           frameon=False, fontsize=13 * PX, handlelength=1.8, borderaxespad=0.3)
        for text in legend.get_texts():
            text.set_color(colors["ink"])
        fig.subplots_adjust(left=0.07, right=0.97, top=0.80, bottom=0.12)
        path = Path(path)
        fig.savefig(path, dpi=200, facecolor=colors["surface"])
        plt.close(fig)
    return path


def pick_examples(images: list[LabeledImage], count: int, seed: int) -> list[tuple[int, LabeledImage]]:
    """Pick `count` images at random (not cherry-picked), returning their dataset indices too."""
    indices = sorted(random.Random(seed).sample(range(len(images)), count))
    return [(i, images[i]) for i in indices]


def shorten(text: str, limit: int = 20) -> str:
    """Truncate `text` to `limit` characters with an ellipsis."""
    return text if len(text) <= limit else text[: limit - 1] + "…"


def draw_prediction(ax: plt.Axes, y: float, marker: str, color: str, ok: bool, text: str, colors: dict) -> None:
    """Draw one prediction line: series marker, a status icon (✓/✗) and the label in ink."""
    ax.plot([0.03], [y], marker=marker, color=color, markersize=8 * PX, linestyle="none")
    ax.text(0.09, y, "✓" if ok else "✗", va="center", fontsize=13 * PX, fontweight="bold",
            color=colors["good"] if ok else colors["bad"])
    ax.text(0.17, y, text, va="center", fontsize=12 * PX, color=colors["ink"])


def plot_examples(
    classifiers: list[Classifier],
    images: list[LabeledImage],
    perturbation: Perturbation,
    path: str | Path,
    count: int = 3,
    seed: int = 0,
    theme: str = "light",
) -> Path:
    """Show randomly picked images at a few severities with each model's top-1 prediction."""
    colors = THEMES[theme]
    names = imagenet_class_names()
    examples = pick_examples(images, count, seed)
    severities = perturbation.examples
    series_colors = assign_colors([c.spec.alias for c in classifiers], theme)
    batch = torch.stack([load_image(item.path) for _, item in examples])
    seeds = image_seeds(seed, [index for index, _ in examples])
    text_height = 0.3 * len(classifiers) + 0.1
    with plt.rc_context({"font.family": FONT_FAMILY}):
        fig = plt.figure(figsize=(2.5 * len(severities) + 0.4, (2.5 + text_height) * count + 0.4), dpi=100)
        fig.patch.set_facecolor(colors["surface"])
        grid = fig.add_gridspec(2 * count, len(severities), height_ratios=[2.5, text_height] * count,
                                left=0.05, right=0.99, top=0.95, bottom=0.01, wspace=0.06, hspace=0.04)
        for col, severity in enumerate(severities):
            corrupted = perturbation.apply(batch, severity, seeds)
            predictions = {c.spec.alias: c.predict_proba(corrupted).argmax(dim=1) for c in classifiers}
            for row, (_, item) in enumerate(examples):
                image_ax = fig.add_subplot(grid[2 * row, col])
                image_ax.imshow(corrupted[row].permute(1, 2, 0).numpy())
                image_ax.set_axis_off()
                text_ax = fig.add_subplot(grid[2 * row + 1, col])
                text_ax.set_xlim(0, 1)
                text_ax.set_ylim(0, 1)
                text_ax.set_axis_off()
                for line, classifier in enumerate(classifiers):
                    predicted = int(predictions[classifier.spec.alias][row])
                    model_name = classifier.spec.display_name.split(" ")[0]
                    draw_prediction(text_ax, 1 - (line + 0.75) / (len(classifiers) + 0.5),
                                    FAMILY_MARKERS.get(classifier.spec.family, "D"),
                                    series_colors[classifier.spec.alias], predicted == item.label,
                                    f"{model_name}: {shorten(names[predicted])}", colors)
                if row == 0:
                    image_ax.set_title(f"{perturbation.severity_label.split(' (')[0]}: {swedish_number(severity)}",
                                       fontsize=13 * PX, color=colors["ink2"], pad=6)
                if col == 0:
                    image_ax.text(-0.04, 0.5, f"Rätt svar: {names[item.label]}", transform=image_ax.transAxes,
                                  rotation=90, ha="right", va="center", fontsize=12 * PX, color=colors["ink2"])
        path = Path(path)
        # Photos with noise compress badly as PNG; JPEG keeps the grid around 1 MB.
        options = {"pil_kwargs": {"quality": 90}} if path.suffix.lower() in (".jpg", ".jpeg") else {}
        fig.savefig(path, dpi=200, facecolor=colors["surface"], **options)
        plt.close(fig)
    return path
