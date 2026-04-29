# Responsible for: loading a trained checkpoint, running inference on the
# test split, and producing classification metrics and visualisations.

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# from src.data.dataset import ChestXrayDataset, get_transforms
# from src.models.model import TransferModel
# from src.utils.utils import load_config


CLASSES = ["COVID-19", "viral_pneumonia", "normal"]


def run_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int]]:
    """Collect ground-truth labels and model predictions over a DataLoader.

    Returns:
        (all_labels, all_predictions) — flat lists of ints.
    """
    model.eval()
    all_labels: list[int] = []
    all_preds:  list[int] = []

    with torch.no_grad():
        # TODO: iterate loader, run forward pass, collect argmax predictions
        raise NotImplementedError

    return all_labels, all_preds


def compute_metrics(labels: list[int], predictions: list[int]) -> dict:
    """Compute per-class and macro-averaged metrics.

    Returns a dict with keys: accuracy, precision, recall, f1 (macro),
    and a per-class breakdown.
    """
    # TODO: use sklearn.metrics (classification_report, confusion_matrix)
    raise NotImplementedError


def plot_confusion_matrix(
    labels: list[int],
    predictions: list[int],
    save_path: str | Path | None = None,
) -> None:
    """Plot and optionally save a normalised confusion matrix."""
    # TODO: use seaborn heatmap
    raise NotImplementedError


def evaluate(config: dict) -> None:
    """End-to-end evaluation pipeline.

    Steps:
        1. Load test dataset
        2. Load model weights from config checkpoint path
        3. run_inference → compute_metrics → plot_confusion_matrix
        4. Save metrics report to outputs/reports/
    """
    # TODO: implement
    raise NotImplementedError


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate chest X-ray classifier")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()

    # TODO: load config then call evaluate(config)
    raise NotImplementedError("Wire up config loading and call evaluate().")
