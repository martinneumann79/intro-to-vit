"""EuroSAT: 27,000 Copernicus Sentinel-2 patches (64x64 px, 10 m per pixel) in ten land-use classes."""

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# EuroSAT's RGB images are Sentinel-2 L1C reflectance x 10000 / 2750, clipped to 8 bits. A scene
# fetched with the same gain looks to the model exactly like its training data.
REFLECTANCE_GAIN = 10000 / 2750
PATCH_SIZE = 64  # pixels; 640 x 640 m on the ground at 10 m per pixel
VIT_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

SWEDISH_NAMES = {
    "AnnualCrop": "Åker, ettåriga grödor",
    "Forest": "Skog",
    "HerbaceousVegetation": "Örtvegetation",
    "Highway": "Väg och motorväg",
    "Industrial": "Industri",
    "Pasture": "Betesmark",
    "PermanentCrop": "Fleråriga grödor",
    "Residential": "Bostadsområde",
    "River": "Vattendrag",
    "SeaLake": "Hav och sjö",
}

# Conventional land-cover colours (as on land-cover maps): water blue, forest dark green, etc.
CLASS_COLORS = {
    "AnnualCrop": "#F4C542",
    "Forest": "#1D5E35",
    "HerbaceousVegetation": "#5FAE57",
    "Highway": "#8C8C8C",
    "Industrial": "#9C4FB0",
    "Pasture": "#B7D95B",
    "PermanentCrop": "#E07B39",
    "Residential": "#D64545",
    "River": "#4FA3E0",
    "SeaLake": "#1F4E9C",
}


class RandomRot90:
    """Rotate an image tensor by a random multiple of 90 degrees (satellite images have no 'up')."""

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Return `image` rotated by 0, 90, 180 or 270 degrees."""
        return torch.rot90(image, int(torch.randint(4, ())), dims=(-2, -1))


def vit_input_transform() -> transforms.Compose:
    """Return the shared step from a [0, 1] image tensor to ViT input: resize to 224 and normalise.

    Works on single images (C, H, W) during training and on batches (N, C, H, W) at inference, so
    both see exactly the same preprocessing.
    """
    return transforms.Compose(
        [
            transforms.Resize(VIT_SIZE, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_transforms(train: bool) -> transforms.Compose:
    """Return EuroSAT transforms; training adds flips, 90-degree rotations and mild colour jitter."""
    steps: list = [transforms.ToTensor()]
    if train:
        steps += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            RandomRot90(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
        ]
    steps.append(vit_input_transform())
    return transforms.Compose(steps)


def stratified_split(targets: list[int], val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    """Split indices into train and validation sets with the same class mix in both."""
    generator = torch.Generator().manual_seed(seed)
    labels = torch.tensor(targets)
    train, val = [], []
    for label in labels.unique():
        indices = torch.nonzero(labels == label).flatten()
        indices = indices[torch.randperm(len(indices), generator=generator)]
        n_val = round(len(indices) * val_fraction)
        val += indices[:n_val].tolist()
        train += indices[n_val:].tolist()
    return sorted(train), sorted(val)


def build_loaders(
    data_dir: str = "data", batch_size: int = 64, val_fraction: float = 0.2, seed: int = 0, num_workers: int = 4
) -> tuple[DataLoader, DataLoader, list[str]]:
    """Download EuroSAT if needed and return (train_loader, val_loader, class_names)."""
    train_set = datasets.EuroSAT(data_dir, transform=build_transforms(train=True), download=True)
    val_set = datasets.EuroSAT(data_dir, transform=build_transforms(train=False))
    train_idx, val_idx = stratified_split(train_set.targets, val_fraction, seed)
    persistent = num_workers > 0
    train_loader = DataLoader(Subset(train_set, train_idx), batch_size, shuffle=True,
                              num_workers=num_workers, persistent_workers=persistent)
    val_loader = DataLoader(Subset(val_set, val_idx), 256, shuffle=False,
                            num_workers=num_workers, persistent_workers=persistent)
    return train_loader, val_loader, list(train_set.classes)
