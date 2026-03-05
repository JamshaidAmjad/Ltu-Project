# utils.py — Shared helpers
# Contains utilities for noise mixing, metrics, and plotting used across the project.
# Used by CNN (P2), RNN (P3), Combined (P5), and evaluate.py (P4).

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend (works on servers and in scripts)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix


def add_noise(data, noise_factor=0.1):
    """
    Mix Gaussian noise into a numpy array or torch tensor.

    Args:
        data        : audio waveform (numpy array or torch.Tensor)
        noise_factor: amplitude of the added Gaussian noise (default 0.1)

    Returns:
        Noisy waveform, same type as input.
    """
    import torch

    if isinstance(data, torch.Tensor):
        noise = torch.randn_like(data) * noise_factor
        return data + noise
    else:
        noise = np.random.randn(*data.shape).astype(data.dtype) * noise_factor
        return data + noise


def compute_metrics(y_true, y_pred):
    """
    Compute evaluation metrics for classification.

    Args:
        y_true: ground-truth integer class labels (list or array)
        y_pred: predicted integer class labels (list or array)

    Returns:
        dict with keys: 'accuracy', 'f1_macro'
    """
    acc      = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    return {
        'accuracy' : round(acc,      4),
        'f1_macro' : round(f1_macro, 4),
    }


def plot_results(history, title='Model', save_path=None):
    """
    Plot training/validation loss and accuracy curves side by side.

    Args:
        history  : dict with keys 'train_loss', 'val_loss',
                   'train_acc', 'val_acc' (each a list of per-epoch values)
        title    : model name for plot title (e.g. 'CNN', 'RNN (BiGRU)')
        save_path: if given, save the figure to this path (PNG)
    """
    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{title} — Training History', fontsize=15, fontweight='bold')

    # Loss
    ax1.plot(epochs, history['train_loss'], 'b-',  linewidth=2, label='Train Loss')
    ax1.plot(epochs, history['val_loss'],   'r--', linewidth=2, label='Val Loss')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Loss Curves', fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(epochs, [a * 100 for a in history['train_acc']], 'b-',  linewidth=2, label='Train Acc')
    ax2.plot(epochs, [a * 100 for a in history['val_acc']],   'r--', linewidth=2, label='Val Acc')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Accuracy Curves', fontsize=13)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 100])

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  Plot saved to {save_path}')
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, labels=None, title='Model',
                          save_path=None):
    """
    Plot a seaborn heatmap confusion matrix.

    Args:
        y_true   : ground-truth integer class labels
        y_pred   : predicted integer class labels
        labels   : list of class name strings (optional, used as tick labels)
        title    : title shown above the plot
        save_path: if given, save the figure to this path (PNG)
    """
    cm = confusion_matrix(y_true, y_pred)

    fig_size = max(10, len(cm) // 3)
    fig, ax  = plt.subplots(figsize=(fig_size, fig_size))

    sns.heatmap(
        cm,
        annot=len(cm) <= 20,
        fmt='d',
        cmap='Blues',
        xticklabels=labels if labels else 'auto',
        yticklabels=labels if labels else 'auto',
        ax=ax,
        linewidths=0.3,
    )

    ax.set_xlabel('Predicted Label', fontsize=13)
    ax.set_ylabel('True Label', fontsize=13)
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  Confusion matrix saved to {save_path}')
    plt.close(fig)


def plot_noise_robustness(noise_results, title='Model', save_path=None):
    """
    Plot a bar chart of test accuracy at different noise levels.

    Args:
        noise_results: dict mapping noise level float → accuracy float
                       e.g. {0.0: 0.941, 0.1: 0.914, 0.5: 0.754}
        title        : model name for plot title
        save_path    : if given, save the figure to this path (PNG)
    """
    level_labels = []
    accuracies   = []
    colors       = {'0%': '#2ecc71', '10%': '#f39c12', '50%': '#e74c3c'}

    for level, acc in sorted(noise_results.items()):
        tag = f'{int(level*100)}%'
        label = f'{tag}\n(clean)' if level == 0.0 else tag
        level_labels.append(label)
        accuracies.append(acc * 100)

    bar_colors = [colors.get(f'{int(l*100)}%', '#3498db') for l in sorted(noise_results)]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(level_labels, accuracies, color=bar_colors, width=0.5,
                  edgecolor='white', linewidth=2)

    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax.set_ylabel('Test Accuracy (%)', fontsize=13)
    ax.set_xlabel('Background Noise Level', fontsize=13)
    ax.set_title(f'{title} — Noise Robustness', fontsize=15, fontweight='bold')
    ax.set_ylim([0, 105])
    ax.axhline(y=75, color='gray', linestyle=':', alpha=0.5, label='Min required (75%)')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  Noise robustness chart saved to {save_path}')
    plt.close(fig)
