"""CIFAR-10 data loading, with optional subsetting to study data efficiency."""

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def build_transforms(train: bool) -> transforms.Compose:
    """Return CIFAR-10 transforms; training adds random crop and horizontal flip."""
    steps = []
    if train:
        steps += [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    steps += [transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
    return transforms.Compose(steps)


def subset_fraction(dataset: Dataset, fraction: float, seed: int = 0) -> Dataset:
    """Return a reproducible random subset containing `fraction` of `dataset`."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    if fraction == 1:
        return dataset
    n = max(1, int(len(dataset) * fraction))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:n].tolist()
    return Subset(dataset, indices)


def build_loaders(
    data_dir: str = "data",
    batch_size: int = 128,
    train_fraction: float = 1.0,
    num_workers: int = 4,
    seed: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Download CIFAR-10 if needed and return (train_loader, test_loader)."""
    train_set = datasets.CIFAR10(data_dir, train=True, download=True, transform=build_transforms(True))
    test_set = datasets.CIFAR10(data_dir, train=False, download=True, transform=build_transforms(False))
    train_set = subset_fraction(train_set, train_fraction, seed)
    persistent = num_workers > 0
    train_loader = DataLoader(
        train_set, batch_size, shuffle=True, num_workers=num_workers, persistent_workers=persistent
    )
    test_loader = DataLoader(
        test_set, 512, shuffle=False, num_workers=num_workers, persistent_workers=persistent
    )
    return train_loader, test_loader
