"""Evaluation images that the pretrained models never saw during training."""

import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.models import ResNet50_Weights

# Imagenette's 10 classes, keyed by WordNet id, with their index among the 1000 ImageNet classes.
IMAGENETTE_CLASSES = {
    "n01440764": 0,  # tench
    "n02102040": 217,  # English springer
    "n02979186": 482,  # cassette player
    "n03000684": 491,  # chain saw
    "n03028079": 497,  # church
    "n03394916": 566,  # French horn
    "n03417042": 569,  # garbage truck
    "n03425413": 571,  # gas pump
    "n03445777": 574,  # golf ball
    "n03888257": 701,  # parachute
}
# Imagenette keeps ImageNet's file names. Files with this prefix come from the official
# ImageNet validation set; all other Imagenette files come from the ImageNet training set.
IMAGENET_VAL_PREFIX = "ILSVRC2012_val_"
IMAGENETV2_URL = (
    "https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-matched-frequency.tar.gz"
)
IMAGENETV2_FOLDER = "imagenetv2-matched-frequency-format-val"
DATASETS = ("imagenette", "imagenetv2")
DATASET_TITLES = {
    "imagenette": "ImageNet-val, 10 klasser",
    "imagenetv2": "ImageNetV2, 1000 klasser",
}

PREPROCESS = transforms.Compose(
    [
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
)


@dataclass(frozen=True)
class LabeledImage:
    """Path to an image file and its ImageNet-1k class index."""

    path: Path
    label: int


def imagenet_class_names() -> list[str]:
    """Return the 1000 ImageNet-1k class names in index order (no download needed)."""
    return list(ResNet50_Weights.IMAGENET1K_V2.meta["categories"])


def imagenette_unseen(data_dir: str | Path = "data", download: bool = True) -> list[LabeledImage]:
    """Return the 500 Imagenette images (50 per class) taken from the ImageNet validation set.

    The rest of Imagenette comes from the ImageNet training set. The pretrained models were
    trained on those images, so evaluating on them would overstate accuracy.
    """
    root = Path(data_dir) / "imagenette2-320"
    if not root.exists():
        if not download:
            raise FileNotFoundError(f"{root} not found; run with download enabled first")
        datasets.Imagenette(str(data_dir), split="val", size="320px", download=True)
    files = root.glob(f"*/*/{IMAGENET_VAL_PREFIX}*.JPEG")
    images = [LabeledImage(f, IMAGENETTE_CLASSES[f.parent.name]) for f in files]
    return sorted(images, key=lambda image: image.path.name)


def imagenetv2(data_dir: str | Path = "data", download: bool = True) -> list[LabeledImage]:
    """Return ImageNetV2 (matched frequency): 10,000 newly collected images, 10 per class."""
    root = Path(data_dir) / "imagenetv2" / IMAGENETV2_FOLDER
    if not root.exists():
        if not download:
            raise FileNotFoundError(f"{root} not found; run with download enabled first")
        download_and_extract(IMAGENETV2_URL, root.parent)
    images = [LabeledImage(f, int(f.parent.name)) for f in root.glob("*/*.jpeg")]
    return sorted(images, key=lambda image: (image.label, image.path.name))


def download_and_extract(url: str, target_dir: Path) -> None:
    """Download a .tar.gz archive into `target_dir`, unpack it safely and delete the archive."""
    target_dir.mkdir(parents=True, exist_ok=True)
    archive = target_dir / url.rsplit("/", 1)[-1]
    print(f"Downloading {url} (this can take a few minutes) ...")
    urllib.request.urlretrieve(url, archive)
    with tarfile.open(archive) as tar:
        tar.extractall(target_dir, filter="data")
    archive.unlink()


def load_dataset(name: str, data_dir: str | Path = "data", download: bool = True) -> list[LabeledImage]:
    """Return the evaluation images of dataset `name` ('imagenette' or 'imagenetv2')."""
    if name == "imagenette":
        return imagenette_unseen(data_dir, download)
    if name == "imagenetv2":
        return imagenetv2(data_dir, download)
    raise ValueError(f"Unknown dataset {name!r}; choose from {DATASETS}")


def to_model_input(image: Image.Image) -> torch.Tensor:
    """Resize to 256, centre-crop to 224 and return a (3, 224, 224) tensor in [0, 1]."""
    return PREPROCESS(image.convert("RGB"))


def load_image(path: str | Path) -> torch.Tensor:
    """Load an image file as a (3, 224, 224) tensor in [0, 1]."""
    with Image.open(path) as image:
        return to_model_input(image)


class ImageListDataset(Dataset):
    """A dataset over a list of labelled images; items are (image, label, index)."""

    def __init__(self, images: list[LabeledImage]):
        """Store the list; images are loaded lazily."""
        self.images = images

    def __len__(self) -> int:
        """Return the number of images."""
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        """Load image `index` and return it with its label and index."""
        item = self.images[index]
        return load_image(item.path), item.label, index
