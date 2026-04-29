# Responsible for: loading images from disk, applying transforms,
# and exposing a PyTorch Dataset that the DataLoader can consume.

import os
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


# Standard ImageNet normalisation — appropriate when using pretrained torchvision backbones.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_transforms(image_size: tuple[int, int], augment: bool = False) -> transforms.Compose:
    """Return a torchvision transform pipeline.

    Args:
        image_size: (H, W) resize target.
        augment: if True, adds random flips and colour jitter for training.
    """
    base = [
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]

    if augment:
        augmentation = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]
        # Insert augmentation steps before the tensor conversion.
        base = augmentation + base

    return transforms.Compose(base)


class ChestXrayDataset(Dataset):
    """Dataset for the three-class chest X-ray classification task.

    Expected directory layout inside ``root_dir``::

        root_dir/
          COVID-19/
            img001.jpg
            ...
          viral_pneumonia/
            ...
          normal/
            ...

    Args:
        root_dir: path to the split folder (e.g. ``data/processed/train``).
        transform: torchvision transform to apply to each image.
    """

    # Class-to-index mapping is fixed across all splits so predictions are consistent.
    CLASS_TO_IDX = {"COVID-19": 0, "viral_pneumonia": 1, "normal": 2}

    def __init__(self, root_dir: str | Path, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []  # (image_path, label)

        # TODO: walk root_dir and populate self.samples
        raise NotImplementedError("Implement _load_samples() and call it here.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
