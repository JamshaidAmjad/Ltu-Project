# Responsible for: the training and validation loop, checkpointing,
# and logging metrics to the console / a results file.

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# from src.data.dataset import ChestXrayDataset, get_transforms
# from src.models.model import TransferModel
# from src.utils.utils import load_config, set_seed


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Run a single training epoch.

    Returns:
        (average_loss, accuracy) for this epoch.
    """
    model.train()
    # TODO: iterate loader, forward pass, backward pass, update weights
    raise NotImplementedError


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate the model on a validation split.

    Returns:
        (average_loss, accuracy)
    """
    model.eval()
    with torch.no_grad():
        # TODO: iterate loader, compute loss and accuracy
        raise NotImplementedError


def train(config: dict) -> None:
    """Full training pipeline driven by a config dict.

    Steps (to be implemented):
        1. set_seed for reproducibility
        2. Build train / val datasets and DataLoaders
        3. Instantiate model, loss function, optimiser, scheduler
        4. Loop over epochs: train_one_epoch → validate → checkpoint if improved
        5. Save final model weights
    """
    # TODO: implement full training pipeline
    raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train chest X-ray classifier")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()

    # TODO: load config then call train(config)
    raise NotImplementedError("Wire up config loading and call train().")
