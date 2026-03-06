# cnn_model.py — Created by Sathish Kumaresan
# Responsible for defining, training, and saving the CNN model.

import os
import sys
import math
import torch
import torch.nn as nn
import torchaudio.transforms as T
from torch.utils.data import DataLoader, TensorDataset

torch.set_num_threads(4)

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Constants ───────────────────────────────────────────────────────────
SAMPLE_RATE = 16000
N_CLASSES   = 35
N_MELS      = 64
N_FFT       = 512
HOP_LENGTH  = 160
CACHE_DIR   = '../results/spec_cache'

# ── Mel-Spectrogram Transform ───────────────────────────────────────────
mel_transform = nn.Sequential(
    T.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    ),
    T.AmplitudeToDB()
)

# ── Spectrogram Cache ───────────────────────────────────────────────────
def build_cached_loader(loader, split_name, noise_level, batch_size,
                        cache_dir=CACHE_DIR):
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

# ── Model ───────────────────────────────────────────────────────────────
def build_cnn_model(n_classes=N_CLASSES):
    return AudioCNN(n_classes=n_classes)

class AudioCNN(nn.Module):
    def __init__(self, n_classes=N_CLASSES):
        super().__init__()

        # Normalise raw dB-scale spectrograms
        self.input_norm = nn.BatchNorm2d(1)

        # Frequency and Time masking (only active during training)
        self.freq_mask = T.FrequencyMasking(freq_mask_param=8)
        self.time_mask = T.TimeMasking(time_mask_param=16)

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.15),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        x = self.input_norm(x)

        # Mask during training only
        if self.training:
            x = self.freq_mask(x)
            x = self.freq_mask(x)
            x = self.time_mask(x)
            x = self.time_mask(x)

        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x

# ── Training ────────────────────────────────────────────────────────────
def train_cnn(model, train_loader, val_loader, epochs=30, lr=1e-3):
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
            specs, labels = specs.to(device), labels.to(device)  # Move tensors to the GPU
            optimizer.zero_grad(set_to_none=True)
            logits = model(specs)
            loss   = criterion(logits, labels)
            loss.backward()
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

        if epoch % 2 == 0:
            save_cnn_model(model, f'../results/cnn_model_epoch{epoch}.pth')

        print(
            f'  Epoch {epoch:3d}/{epochs} | '
            f'Train loss {tr_loss:.4f} acc {tr_acc:.3f} | '
            f'Val loss {val_loss:.4f} acc {val_acc:.3f} | '
            f'LR {scheduler.get_last_lr()[0]:.5f}'
        )

    if best_weights is not None:
        model.load_state_dict(best_weights)
    print(f'\n  Training complete. Best val loss: {best_val_loss:.4f}')
    return model, history

# ── Evaluation ──────────────────────────────────────────────────────────
@torch.no_grad()
def _evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for specs, labels in loader:
        specs, labels = specs.to(device), labels.to(device) # Move tensors to the GPU
        logits      = model(specs)
        loss        = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += labels.size(0)
    return total_loss / total, correct / total

# ── Save / Load ─────────────────────────────────────────────────────────
def save_cnn_model(model, path='../results/cnn_model.pth'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f'  Model saved to {path}')

def load_cnn_model(path='../results/cnn_model.pth', n_classes=N_CLASSES):
    model = AudioCNN(n_classes=n_classes)
    state = torch.load(path, weights_only=True)
    model.load_state_dict(state)
    return model

# ── Main ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    sys.path.insert(0, '.')
    from data_pipeline import get_dataloaders

    model = build_cnn_model().to(device)  # Move model to GPU if available
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'CNN trainable parameters: {n_params:,}')

    print('\nLoading full dataset...')
    train_loader, val_loader, test_loader = get_dataloaders(
        noise_level=0.0,
        batch_size=32,
        transform=mel_transform,
    )

    print('Loading spectrogram cache...')
    cached_train = build_cached_loader(train_loader, 'full_train', 0.0, 128)
    cached_val   = build_cached_loader(val_loader,   'full_val',   0.0, 128)
    cached_test  = build_cached_loader(test_loader,  'full_test',  0.0, 128)

    model, history = train_cnn(model, cached_train, cached_val, epochs=30)

    criterion_test = nn.CrossEntropyLoss()
    _, test_acc = _evaluate(model, cached_test, criterion_test)
    print('\nFinal Results:')
    print(f'  Train acc : {history["train_acc"][-1]:.3f}')
    print(f'  Val acc   : {history["val_acc"][-1]:.3f}')
    print(f'  Test acc  : {test_acc:.3f}')
    save_cnn_model(model)
