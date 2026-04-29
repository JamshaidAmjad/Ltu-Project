# Responsible for: defining the neural network architectures used in this project.
# Two approaches are provided as placeholders:
#   1. CustomCNN   — a small CNN built from scratch.
#   2. TransferModel — a pretrained torchvision backbone with a replaced head.

import torch
import torch.nn as nn
from torchvision import models


class CustomCNN(nn.Module):
    """Lightweight CNN built from scratch (baseline / ablation).

    Architecture sketch:
        Conv → BN → ReLU → MaxPool  (×3 blocks)
        GlobalAvgPool
        FC → Dropout → FC (num_classes)
    """

    def __init__(self, num_classes: int = 3, dropout: float = 0.5):
        super().__init__()
        # TODO: define self.features (convolutional blocks)
        # TODO: define self.classifier (fully-connected head)
        raise NotImplementedError("Build the convolutional blocks and classifier head.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: x → features → flatten → classifier
        raise NotImplementedError


class TransferModel(nn.Module):
    """Pretrained backbone (ResNet-50 by default) with a custom classification head.

    The backbone is loaded from torchvision with ImageNet weights.
    Only the final fully-connected layer is replaced for our 3-class task.
    During the first training phase the backbone weights are frozen;
    unfreeze them for fine-tuning in a second phase.
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        num_classes: int = 3,
        pretrained: bool = True,
        dropout: float = 0.5,
    ):
        super().__init__()
        # TODO: load backbone via getattr(models, backbone)(weights=...)
        # TODO: freeze backbone parameters
        # TODO: replace the final FC layer with nn.Sequential(Dropout, Linear(num_classes))
        raise NotImplementedError("Load backbone and attach the classification head.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: pass x through backbone
        raise NotImplementedError

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters for fine-tuning."""
        # TODO: set requires_grad=True on backbone parameters
        raise NotImplementedError
