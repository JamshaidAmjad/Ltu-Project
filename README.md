# Chest X-Ray Classification — AI in Healthcare

**Course:** AI in Healthcare (Master's level)  
**Institution:** Luleå University of Technology  
**Team size:** 3

---

## Problem Statement

Chest X-rays are one of the most common and cost-effective diagnostic tools available, yet their interpretation requires specialist expertise that is not always accessible. This project builds a **medical decision support system** that automatically classifies chest X-ray images into three categories:

| Class | Description |
|---|---|
| `COVID-19` | Radiological findings consistent with COVID-19 pneumonia |
| `viral_pneumonia` | Other viral pneumonia patterns |
| `normal` | Healthy chest X-ray |

The goal is **not** to replace radiologists but to provide a fast, consistent second opinion — particularly relevant in resource-limited or high-volume settings.

---

## Approach

### Baseline — Custom CNN
A small convolutional neural network trained from scratch to establish a performance baseline.

### Main model — Transfer Learning
A pretrained **ResNet-50** backbone (ImageNet weights) with a replaced classification head. Training proceeds in two phases:

1. **Phase 1 — Head-only training:** backbone frozen, only the new FC head is optimised.
2. **Phase 2 — Fine-tuning:** backbone unfrozen, all layers trained with a smaller learning rate.

Both models are implemented in `src/models/model.py`.

---

## Repository Structure

```
Ltu-Project/
│
├── data/
│   ├── raw/            ← original, unmodified images (not committed to git)
│   └── processed/      ← train / val / test splits after preprocessing
│
├── notebooks/          ← exploratory data analysis and result visualisation
│
├── src/
│   ├── data/
│   │   └── dataset.py  ← PyTorch Dataset class and image transforms
│   ├── models/
│   │   └── model.py    ← CustomCNN and TransferModel (ResNet-50) definitions
│   ├── training/
│   │   └── train.py    ← training loop, validation, checkpointing
│   ├── evaluation/
│   │   └── evaluate.py ← inference, metrics (accuracy, F1, AUC), confusion matrix
│   └── utils/
│       └── utils.py    ← shared helpers: config loader, seed setter, device selector
│
├── outputs/
│   ├── models/         ← saved model checkpoints (.pth) — not committed
│   ├── figures/        ← training curves, confusion matrices — not committed
│   └── reports/        ← metric summaries (JSON / CSV) — not committed
│
├── config/
│   └── config.yaml     ← all hyperparameters and paths in one place
│
├── tests/              ← unit tests (dataset loading, model forward pass, etc.)
│
├── requirements.txt
└── .gitignore
```

---

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/jamshaidamjad/ltu-project.git
cd ltu-project

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place the dataset inside data/raw/
#    Expected layout:
#      data/raw/COVID-19/
#      data/raw/viral_pneumonia/
#      data/raw/normal/
```

---

## How to Run

> **Note:** The scripts below are placeholders — implementation is in progress.

```bash
# Preprocess and split the dataset
# python src/data/preprocess.py --config config/config.yaml

# Train the model
python src/training/train.py --config config/config.yaml

# Evaluate on the test set
python src/evaluation/evaluate.py --config config/config.yaml
```

---

## Configuration

All hyperparameters live in `config/config.yaml` — edit that file rather than touching source code.

Key settings:

| Parameter | Default | Description |
|---|---|---|
| `model.architecture` | `resnet50` | Torchvision backbone name |
| `training.epochs` | `30` | Total training epochs |
| `training.batch_size` | `32` | Samples per batch |
| `training.learning_rate` | `1e-4` | Initial learning rate |
| `data.image_size` | `[224, 224]` | Resize target for all images |

---

## Team

| Name | Responsibility |
|---|---|
| Member 1 | Data preprocessing & augmentation |
| Member 2 | Model architecture & training |
| Member 3 | Evaluation, metrics & reporting |

---

## Ethical Considerations

- This system is intended as a **decision support tool**, not an autonomous diagnostic device.
- All data must be de-identified before use in accordance with applicable data protection regulations.
- Model performance must be validated on a diverse, representative dataset before any clinical deployment.
