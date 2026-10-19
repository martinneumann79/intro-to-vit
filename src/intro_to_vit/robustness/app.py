"""Interactive browser demo: corrupt an image and watch a ViT and a CNN classify it live."""

import random
from collections import Counter
from pathlib import Path

import gradio as gr
import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image

from intro_to_vit.robustness.imagenet_data import imagenet_class_names, imagenette_unseen, to_model_input
from intro_to_vit.robustness.perturbations import PERTURBATIONS
from intro_to_vit.robustness.plots import swedish_millions
from intro_to_vit.robustness.pretrained import Classifier, load_classifiers
from intro_to_vit.training import get_device

DEMO_SEED = 7  # one fixed random pattern: the slider changes its strength, not its shape
TOP_K = 5
EXAMPLE_COUNT = 10
DEFAULT_PERTURBATION = "noise"

INTRO = """
# {first} mot {second}: vem känner igen en störd bild?

Båda modellerna är tränade på samma 1,3 miljoner ImageNet-bilder och känner igen rena bilder
ungefär lika bra (cirka 80 % rätt bland 1000 klasser). Skillnaden ligger i arkitekturen:
**{first}** {first_how}. **{second}** {second_how}.

Välj en bild, eller ta en med kameran, och dra i reglaget.
"""
HOW_FAMILIES_SEE = {
    "transformer": "delar bilden i rutor och låter rutorna titta på varandra med self-attention",
    "cnn": "bygger upp bilden lager för lager med små lokala filter",
}


def unique_class_names() -> list[str]:
    """Return ImageNet class names, with the index appended to names that occur twice."""
    names = imagenet_class_names()
    counts = Counter(names)
    return [name if counts[name] == 1 else f"{name} ({i})" for i, name in enumerate(names)]


def how_it_sees(classifier: Classifier) -> str:
    """Describe in Swedish how the model's architecture family processes an image."""
    return HOW_FAMILIES_SEE.get(classifier.spec.family, "är en förtränad bildmodell")


def model_title(classifier: Classifier) -> str:
    """Return a short heading such as 'ViT-S/16 (DeiT) · 22,1 M parametrar'."""
    return f"{classifier.spec.display_name} · {swedish_millions(classifier.num_parameters)} parametrar"


class DemoModels:
    """The two classifiers being compared, ImageNet class names and the example images."""

    def __init__(self, model_names: list[str], data_dir: str | Path):
        """Load both classifiers and index the unseen evaluation images by file name."""
        self.classifiers = load_classifiers(model_names, get_device())
        if len(self.classifiers) != 2:
            raise ValueError("The demo compares exactly two different models")
        self.class_names = unique_class_names()
        self.images = imagenette_unseen(data_dir)
        self.labels_by_name = {item.path.name: item.label for item in self.images}

    def examples(self, count: int = EXAMPLE_COUNT, seed: int = 0) -> list[str]:
        """Return one random unseen image per class (up to `count`) as file paths."""
        rng = random.Random(seed)
        by_label: dict[int, list[Path]] = {}
        for item in self.images:
            by_label.setdefault(item.label, []).append(item.path)
        return [str(rng.choice(paths)) for paths in list(by_label.values())[:count]]

    def random_image(self) -> str:
        """Return the path of a random unseen evaluation image."""
        return str(random.choice(self.images).path)

    def true_label(self, image_path: str) -> int | None:
        """Return the ImageNet class of an evaluation image (matched by file name), else None."""
        return self.labels_by_name.get(Path(image_path).name)

    def top_k(self, probabilities: torch.Tensor) -> dict[str, float]:
        """Return the TOP_K most likely classes as {class name: probability}."""
        values, indices = probabilities.topk(TOP_K)
        return {self.class_names[int(i)]: float(v) for v, i in zip(values, indices)}

    def verdict(self, image_path: str, predictions: list[int]) -> str:
        """Describe in Swedish which model got the image right."""
        guesses = [
            f"**{c.spec.display_name}** gissar *{self.class_names[p]}*"
            for c, p in zip(self.classifiers, predictions)
        ]
        label = self.true_label(image_path)
        if label is None:
            return "Egen bild, så rätt svar är okänt. " + " · ".join(guesses)
        marks = [
            f"{'✅' if p == label else '❌'} {c.spec.display_name}" for c, p in zip(self.classifiers, predictions)
        ]
        return f"**Rätt svar:** {self.class_names[label]}  \n" + " · ".join(marks) + "  \n" + " · ".join(guesses)

    def analyze(self, image_path: str | None, perturbation_key: str, severity: float):
        """Corrupt the image and return (shown image, top-k per model, verdict text)."""
        if not image_path:
            return None, None, None, ""
        with Image.open(image_path) as image:
            batch = to_model_input(image).unsqueeze(0)
        corrupted = PERTURBATIONS[perturbation_key].apply(batch, float(severity), [DEMO_SEED])
        probabilities = [c.predict_proba(corrupted)[0] for c in self.classifiers]
        shown = to_pil_image(corrupted[0]).resize((448, 448), Image.Resampling.NEAREST)
        predictions = [int(p.argmax()) for p in probabilities]
        return shown, *[self.top_k(p) for p in probabilities], self.verdict(image_path, predictions)


def accuracy_plot(results_dir: str | Path, perturbation_key: str) -> str | None:
    """Return the benchmark figure for a perturbation if `robustness-benchmark` has made one."""
    path = Path(results_dir) / "imagenette" / perturbation_key / "accuracy.png"
    return str(path) if path.exists() else None


def slider_for(perturbation_key: str) -> gr.Slider:
    """Return a slider configured for the chosen perturbation."""
    p = PERTURBATIONS[perturbation_key]
    return gr.Slider(minimum=0, maximum=p.maximum, step=p.step, value=p.default, label=p.severity_label)


def build_app(models: DemoModels, results_dir: str | Path) -> gr.Blocks:
    """Lay out the demo and wire up its events."""
    first, second = models.classifiers
    examples = models.examples()

    def plot_for(perturbation_key: str) -> str | None:
        """Return the benchmark figure for the chosen perturbation, if there is one."""
        return accuracy_plot(results_dir, perturbation_key)

    with gr.Blocks(title="ViT mot CNN", analytics_enabled=False) as app:
        gr.Markdown(INTRO.format(first=first.spec.display_name, second=second.spec.display_name,
                                 first_how=how_it_sees(first), second_how=how_it_sees(second)))
        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(value=examples[0], type="filepath", sources=["upload", "webcam", "clipboard"],
                                 label="Bild", height=300)
                random_button = gr.Button("Slumpa en bild modellerna aldrig sett", variant="secondary")
                perturbation = gr.Dropdown(choices=[(p.title, p.key) for p in PERTURBATIONS.values()],
                                           value=DEFAULT_PERTURBATION, label="Störning")
                severity = slider_for(DEFAULT_PERTURBATION)
            with gr.Column(scale=1):
                shown = gr.Image(label="Det modellerna ser (224×224 pixlar)", interactive=False, height=448)
            with gr.Column(scale=1):
                first_output = gr.Label(num_top_classes=TOP_K, label=model_title(first))
                second_output = gr.Label(num_top_classes=TOP_K, label=model_title(second))
        verdict = gr.Markdown()
        gr.Examples(examples=examples, inputs=image, label="Exempel ur ImageNets valideringsbilder",
                    examples_per_page=EXAMPLE_COUNT)
        with gr.Accordion("Resultat på 500 osedda bilder", open=False):
            plot = gr.Image(value=plot_for(DEFAULT_PERTURBATION), show_label=False,
                            interactive=False)

        inputs = [image, perturbation, severity]
        outputs = [shown, first_output, second_output, verdict]
        image.change(models.analyze, inputs, outputs)
        severity.change(models.analyze, inputs, outputs)
        perturbation.change(slider_for, perturbation, severity).then(models.analyze, inputs, outputs)
        perturbation.change(plot_for, perturbation, plot)
        random_button.click(models.random_image, None, image)
        app.load(models.analyze, inputs, outputs)
    return app


def launch(model_names: list[str], data_dir: str | Path, results_dir: str | Path, port: int = 7860,
           open_browser: bool = True) -> None:
    """Build the demo and serve it on http://127.0.0.1:<port>."""
    models = DemoModels(model_names, data_dir)
    app = build_app(models, results_dir)
    allowed = [str(Path(data_dir).resolve()), str(Path(results_dir).resolve())]
    # System fonts and no analytics, so the demo also works without internet during a talk.
    theme = gr.themes.Soft(font=["system-ui", "sans-serif"], font_mono=["ui-monospace", "monospace"])
    app.launch(server_name="127.0.0.1", server_port=port, inbrowser=open_browser, allowed_paths=allowed, theme=theme)
