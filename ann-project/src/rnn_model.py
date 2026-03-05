# rnn_model.py — Owned by P3
# Responsible for defining, training, and saving the RNN model.
#
# Architecture: Bidirectional GRU on Mel-Spectrogram time frames.
# The same mel_transform used in cnn_model.py is used here so both
# models can share cached dataloaders from the same data_pipeline.

import os
import sys
import math
import torch
import torch.nn as nn
import torchaudio.transforms as T
from torch.utils.data import DataLoader, TensorDataset

torch.set_num_threads(4)

# ── Constants  (match CNN exactly so pipelines are interchangeable) ──────
SAMPLE_RATE = 16000
N_CLASSES   = 35
N_MELS      = 64
N_FFT       = 512
HOP_LENGTH  = 160
CACHE_DIR   = '../results/spec_cache'

# Number of time frames produced by the mel transform on 1-second audio:
#   T = floor(SAMPLE_RATE / HOP_LENGTH) + 1  ≈ 101
N_FRAMES = SAMPLE_RATE // HOP_LENGTH + 1   # 101

# ── Mel-Spectrogram Transform  (identical to CNN) ────────────────────────
mel_transform = nn.Sequential(
    T.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    ),
    T.AmplitudeToDB(),
)

# ── Spectrogram Cache  (same helper pattern as cnn_model.py) ─────────────
def build_cached_loader(loader, split_name, noise_level, batch_size,
                        cache_dir=CACHE_DIR):
    """Pre-compute and cache spectrograms for fast repeated training."""
    os.makedirs(cache_dir, exist_ok=True)
    noise_tag  = f'noise{int(noise_level * 100):03d}'
    cache_path = os.path.join(cache_dir, f'{split_name}_{noise_tag}.pt')

    if os.path.exists(cache_path):
        print(f'  Loading cache: {cache_path}')
        data   = torch.load(cache_path, weights_only=True)
        specs  = data['specs']
        labels = data['labels']
    else:
        print(f'  Building cache for {split_name} '
              f'(noise={int(noise_level*100)}%)... ', end='', flush=True)
        all_specs, all_labels = [], []
        for specs_batch, labels_batch in loader:
            all_specs.append(specs_batch)
            all_labels.append(labels_batch)
        specs  = torch.cat(all_specs,  dim=0)
        labels = torch.cat(all_labels, dim=0)
        torch.save({'specs': specs, 'labels': labels}, cache_path)
        print('done.')

    shuffle = (split_name == 'train')
    return DataLoader(
        TensorDataset(specs, labels),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
    )

# ── Model ────────────────────────────────────────────────────────────────
def build_rnn_model(n_classes=N_CLASSES):
    """Build and return the RNN model."""
    return AudioRNN(n_classes=n_classes)


class AudioRNN(nn.Module):
    """
    Bidirectional GRU classifier for mel-spectrogram sequences.

    Input shape : [B, 1, N_MELS=64, T=101]
                  (same tensor produced by mel_transform)
    Output shape: [B, N_CLASSES=35]

    Processing steps:
      1. Squeeze channel dim  → [B, 64, 101]
      2. Permute              → [B, T=101, F=64]  (time-major for RNN)
      3. BatchNorm on features → stabilises training on raw dB values
      4. Bidirectional GRU (1 layer, 128 hidden, dropout 0.3)
      5. Concatenate final forward + backward hidden states → [B, 256]
      6. FC head: 256 → 128 → n_classes   (~185K params, matches CNN's 181K)
    """

    def __init__(self, n_classes=N_CLASSES, hidden_size=128, num_layers=1,
                 dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # Normalise the 64-dim mel feature vector at each time step
        self.feature_norm = nn.BatchNorm1d(N_MELS)

        # Bidirectional GRU
        self.gru = nn.GRU(
            input_size=N_MELS,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),   # *2 because bidirectional
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        # x: [B, 1, N_MELS, T]
        x = x.squeeze(1)                        # [B, N_MELS, T]

        # BatchNorm1d expects [B, C, L]
        x = self.feature_norm(x)                # [B, N_MELS, T]

        x = x.permute(0, 2, 1)                  # [B, T, N_MELS]

        # GRU — we only need the final hidden state
        _, h_n = self.gru(x)
        # h_n: [num_layers * 2, B, hidden_size]  (2 = bidirectional)

        # Take the last layer's forward and backward hidden states
        h_forward  = h_n[-2]                    # [B, hidden_size]
        h_backward = h_n[-1]                    # [B, hidden_size]
        h_cat      = torch.cat([h_forward, h_backward], dim=1)  # [B, 512]

        return self.classifier(h_cat)           # [B, n_classes]


# ── Training ─────────────────────────────────────────────────────────────
def train_rnn(model, train_loader, val_loader, epochs=30, lr=1e-3):
    """
    Train the RNN model and return (trained_model, history).

    Uses AdamW + cosine LR schedule with linear warmup + label smoothing,
    identical to the CNN training setup so results are directly comparable.
    """
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    warmup_epochs = 3

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    history       = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_loss = float('inf')
    best_weights  = None

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0

        for specs, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(specs)
            loss   = criterion(logits, labels)
            loss.backward()
            # Gradient clipping — important for RNNs to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            tr_loss    += loss.item() * labels.size(0)
            tr_correct += (logits.argmax(1) == labels).sum().item()
            tr_total   += labels.size(0)

        scheduler.step()
        tr_loss /= tr_total
        tr_acc   = tr_correct / tr_total

        val_loss, val_acc = _evaluate(model, val_loader, criterion)

        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights  = {k: v.clone() for k, v in model.state_dict().items()}

        # Periodic checkpoint every 2 epochs
        if epoch % 2 == 0:
            save_rnn_model(model, f'../results/rnn_model_epoch{epoch}.pth')

        print(
            f'  Epoch {epoch:3d}/{epochs} | '
            f'Train loss {tr_loss:.4f} acc {tr_acc:.3f} | '
            f'Val loss {val_loss:.4f} acc {val_acc:.3f} | '
            f'LR {scheduler.get_last_lr()[0]:.5f}'
        )

    # Restore best weights before returning
    if best_weights is not None:
        model.load_state_dict(best_weights)
    print(f'\n  Training complete. Best val loss: {best_val_loss:.4f}')
    return model, history


# ── Evaluation ───────────────────────────────────────────────────────────
@torch.no_grad()
def _evaluate(model, loader, criterion):
    """Return (avg_loss, accuracy) on a given DataLoader."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for specs, labels in loader:
        logits      = model(specs)
        loss        = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += labels.size(0)
    return total_loss / total, correct / total


# ── Save / Load ───────────────────────────────────────────────────────────
def save_rnn_model(model, path='../results/rnn_model.pth'):
    """Save model state dict to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f'  Model saved to {path}')


def load_rnn_model(path='../results/rnn_model.pth', n_classes=N_CLASSES):
    """Load a saved RNN model from disk."""
    model = AudioRNN(n_classes=n_classes)
    state = torch.load(path, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


# ── Main ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    sys.path.insert(0, '.')
    from data_pipeline import get_dataloaders

    model    = build_rnn_model()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'RNN trainable parameters: {n_params:,}')

    print('\nLoading full dataset...')
    train_loader, val_loader, test_loader = get_dataloaders(
        noise_level=0.0,
        batch_size=32,
        transform=mel_transform,
    )

    print('Building / loading spectrogram cache...')
    cached_train = build_cached_loader(train_loader, 'full_train', 0.0, 128)
    cached_val   = build_cached_loader(val_loader,   'full_val',   0.0, 128)
    cached_test  = build_cached_loader(test_loader,  'full_test',  0.0, 128)

    model, history = train_rnn(model, cached_train, cached_val, epochs=30)

    criterion_test = nn.CrossEntropyLoss()
    _, test_acc = _evaluate(model, cached_test, criterion_test)
    print('\nFinal Results:')
    print(f'  Train acc : {history["train_acc"][-1]:.3f}')
    print(f'  Val acc   : {history["val_acc"][-1]:.3f}')
    print(f'  Test acc  : {test_acc:.3f}')
    save_rnn_model(model)
