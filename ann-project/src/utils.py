# utils.py — Shared helpers
# Contains utilities for noise mixing, metrics, and plotting used across the project.

def add_noise(data, noise_factor=0.1):
    """Mix Gaussian noise into the data."""
    raise NotImplementedError("Implement noise mixing here.")


def compute_metrics(y_true, y_pred):
    """Compute and return evaluation metrics (accuracy, F1, etc.)."""
    raise NotImplementedError("Implement metrics computation here.")


def plot_results(history, save_path=None):
    """Plot training history curves and optionally save to file."""
    raise NotImplementedError("Implement result plotting here.")


def plot_confusion_matrix(y_true, y_pred, labels=None, save_path=None):
    """Plot and optionally save a confusion matrix."""
    raise NotImplementedError("Implement confusion matrix plotting here.")
