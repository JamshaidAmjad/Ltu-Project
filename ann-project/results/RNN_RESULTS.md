# RNN Model — P3 Results

## Architecture: Bidirectional GRU

```
Input:  Mel Spectrogram [B, 1, 64, 101]
  → BatchNorm1d(64)
  → Bidirectional GRU (1 layer, 128 hidden)
  → Concatenate forward + backward hidden states → [B, 256]
  → Linear(256 → 128) → ReLU → Dropout(0.5) → Linear(128 → 35)
Output: [B, 35]
```

**Trainable Parameters:** 186,531 (CNN has 181,507 — 2.8% difference)

## Training Configuration

| Setting | Value |
|---------|-------|
| Epochs | 30 |
| Batch size | 128 |
| Optimizer | AdamW (lr=1e-3, weight_decay=1e-4) |
| LR Schedule | Cosine decay with 3-epoch linear warmup |
| Loss | CrossEntropyLoss (label_smoothing=0.1) |
| Gradient clipping | max_norm=5.0 |
| Trained on | Clean data (0% noise) |

## Results — Noise Robustness

| Noise Level | Test Accuracy | Test Loss |
|-------------|---------------|-----------|
| **0% (clean)** | **94.1%** | 0.2844 |
| **10% noise** | **91.4%** | 0.3803 |
| **50% noise** | **75.4%** | 0.9534 |

### Training History (0% noise)

| Epoch | Train Acc | Val Acc |
|-------|-----------|---------|
| 5 | 88.4% | 90.3% |
| 10 | 93.2% | 92.9% |
| 15 | 95.2% | 93.5% |
| 20 | 96.6% | 94.0% |
| 25 | 97.4% | 94.3% |
| 30 | 97.6% | 94.4% |

## Key Design Decisions

- **GRU over LSTM:** Fewer parameters, faster training, comparable accuracy on speech tasks
- **Bidirectional:** Speech requires both past and future context for disambiguation
- **BatchNorm before GRU:** Stabilises training on raw dB-scale mel features
- **Gradient clipping (5.0):** Prevents exploding gradients across 101 time steps — essential for RNNs
- **Single layer:** Chosen to match CNN parameter count (~186K) for fair comparison
