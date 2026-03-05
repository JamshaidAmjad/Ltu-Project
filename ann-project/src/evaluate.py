# evaluate.py — Evaluation & Reporting
# Universal evaluation pipeline that works for CNN, RNN, and Combined models.
# Generates all deliverables required by the instructor:
#   - Training curves (loss + accuracy)
#   - Confusion matrices at 0%, 10%, 50% noise
#   - Noise robustness bar chart
#   - Printed metrics (accuracy, F1)
#
# Usage:
#   python evaluate.py --model rnn
#   python evaluate.py --model cnn
#   python evaluate.py --model combined

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from data_pipeline import get_dataloaders
from utils import (compute_metrics, plot_results, plot_confusion_matrix,
                   plot_noise_robustness)

# Class labels (same order as data_pipeline.py)
LABELS = [
    'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five',
    'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn', 'left',
    'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila',
    'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero'
]

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
NOISE_LEVELS = [0.0, 0.1, 0.5]


def evaluate_model(model, test_loader, criterion=None):
    """
    Evaluate a model on a test DataLoader.

    Works with any model (CNN, RNN, Combined) — just pass the model and loader.

    Args:
        model       : trained nn.Module (already in eval mode)
        test_loader : DataLoader yielding (spectrograms, labels)
        criterion   : loss function (default: CrossEntropyLoss)

    Returns:
        dict with keys:
            'accuracy'  : float
            'f1_macro'  : float
            'loss'      : float
            'y_true'    : np.array of true labels
            'y_pred'    : np.array of predicted labels
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    model.eval()
    all_preds, all_labels = [], []
    total_loss, total = 0.0, 0

    with torch.no_grad():
        for specs, labels in test_loader:
            logits = model(specs)
            loss   = criterion(logits, labels)
            preds  = logits.argmax(1)

            total_loss += loss.item() * labels.size(0)
            total      += labels.size(0)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    metrics = compute_metrics(y_true, y_pred)
    metrics['loss']   = total_loss / total
    metrics['y_true'] = y_true
    metrics['y_pred'] = y_pred

    return metrics


def generate_report(model, model_name, mel_transform, build_cached_loader,
                    history=None, output_dir=RESULTS_DIR):
    """
    Generate the complete evaluation report for a model.

    This single function produces ALL instructor-required deliverables:
      1. Training curves (if history is provided)
      2. Confusion matrices at 0%, 10%, 50% noise
      3. Noise robustness bar chart
      4. Printed accuracy table

    Works identically for CNN, RNN, or Combined — just pass the model.

    Args:
        model              : trained nn.Module
        model_name         : string like 'CNN', 'RNN (BiGRU)', 'Combined'
        mel_transform      : the mel spectrogram transform (from the model's module)
        build_cached_loader: the caching function (from the model's module)
        history            : optional training history dict
                             {'train_loss': [], 'val_loss': [],
                              'train_acc': [], 'val_acc': []}
        output_dir         : directory to save plots (default: results/)
    """
    os.makedirs(output_dir, exist_ok=True)
    # File prefix based on model name (e.g. 'rnn', 'cnn', 'combined')
    prefix = model_name.lower().split()[0].split('(')[0].strip()

    print(f'\n{"="*60}')
    print(f'  Generating Report: {model_name}')
    print(f'{"="*60}')

    # ── 1. Training Curves ──────────────────────────────────────
    if history:
        print(f'\n[1] Training curves...')
        save_path = os.path.join(output_dir, f'{prefix}_training_curves.png')
        plot_results(history, title=model_name, save_path=save_path)
    else:
        print(f'\n[1] Training curves — skipped (no history provided)')

    # ── 2 & 3. Evaluate at all noise levels + confusion matrices ─
    noise_results = {}
    criterion = nn.CrossEntropyLoss()

    for noise_level in NOISE_LEVELS:
        tag = int(noise_level * 100)
        print(f'\n[Noise {tag}%] Loading test data...')

        _, _, test_loader = get_dataloaders(
            noise_level=noise_level,
            batch_size=32,
            transform=mel_transform,
        )
        cached_test = build_cached_loader(
            test_loader, 'full_test', noise_level, 128
        )

        # Evaluate
        metrics = evaluate_model(model, cached_test, criterion)
        noise_results[noise_level] = metrics['accuracy']

        print(f'  Accuracy : {metrics["accuracy"]*100:.1f}%')
        print(f'  F1 Macro : {metrics["f1_macro"]:.4f}')
        print(f'  Loss     : {metrics["loss"]:.4f}')

        # Confusion matrix
        cm_path = os.path.join(output_dir, f'{prefix}_confusion_matrix_{tag}pct.png')
        cm_title = f'{model_name} — Confusion Matrix at {tag}% Noise  (Accuracy: {metrics["accuracy"]*100:.1f}%)'
        plot_confusion_matrix(
            metrics['y_true'], metrics['y_pred'],
            labels=LABELS, title=cm_title, save_path=cm_path
        )

    # ── 4. Noise Robustness Chart ───────────────────────────────
    print(f'\n[Summary] Noise robustness chart...')
    rob_path = os.path.join(output_dir, f'{prefix}_noise_robustness.png')
    plot_noise_robustness(noise_results, title=model_name, save_path=rob_path)

    # ── 5. Print Summary Table ──────────────────────────────────
    print(f'\n{"="*60}')
    print(f'  {model_name} — Final Results')
    print(f'{"="*60}')
    print(f'  {"Noise Level":<15} {"Test Accuracy":<15} {"Status"}')
    print(f'  {"─"*45}')
    for level, acc in sorted(noise_results.items()):
        tag    = f'{int(level*100)}%'
        status = '✅' if acc >= 0.75 else '❌ Below 75%'
        print(f'  {tag:<15} {acc*100:>6.1f}%         {status}')
    print(f'{"="*60}\n')

    return noise_results


# ── Command-Line Interface ──────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate a trained model and generate all required plots.'
    )
    parser.add_argument(
        '--model', type=str, required=True,
        choices=['cnn', 'rnn', 'combined'],
        help='Which model to evaluate: cnn, rnn, or combined'
    )
    args = parser.parse_args()

    if args.model == 'rnn':
        from rnn_model import (load_rnn_model, mel_transform,
                               build_cached_loader)
        model = load_rnn_model()
        generate_report(model, 'RNN (BiGRU)', mel_transform,
                        build_cached_loader)

    elif args.model == 'cnn':
        from cnn_model import (load_cnn_model, mel_transform,
                               build_cached_loader)
        model = load_cnn_model()
        generate_report(model, 'CNN', mel_transform,
                        build_cached_loader)

    elif args.model == 'combined':
        from combined_model import SmallCombinedModel, loadCombinedModel
        from cnn_model import mel_transform, build_cached_loader
        model = SmallCombinedModel()
        state, _ = loadCombinedModel()
        model.load_state_dict(state)
        generate_report(model, 'Combined (CNN+RNN)', mel_transform,
                        build_cached_loader)
