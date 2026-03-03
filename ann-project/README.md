# ANN Project

A collaborative Artificial Neural Network project exploring CNN and RNN architectures.

## Project Structure

```
ann-project/
├── data/
│   └── README.md          # Instructions to download datasets (actual data not committed)
├── src/
│   ├── data_pipeline.py   # P1 — Data loading and preprocessing
│   ├── cnn_model.py       # P2 — CNN model definition and training
│   ├── rnn_model.py       # P3 — RNN model definition and training
│   ├── evaluate.py        # P4 — Model evaluation and reporting
│   └── utils.py           # Shared helpers: noise mixing, metrics, plotting
├── notebooks/
│   └── ANN_Project.ipynb  # P5 — Final integration notebook
├── results/
│   └── .gitkeep           # Plots and confusion matrices are saved here
├── .gitignore
└── README.md
```

## Team Responsibilities

| Member | File | Responsibility |
|--------|------|----------------|
| P1 | `src/data_pipeline.py` | Data loading & preprocessing |
| P2 | `src/cnn_model.py` | CNN architecture & training |
| P3 | `src/rnn_model.py` | RNN architecture & training |
| P4 | `src/evaluate.py` | Evaluation & result reports |
| P5 | `notebooks/ANN_Project.ipynb` | Final integration |

## Getting Started

1. **Clone the repo** and switch to your working branch.
2. **Download the dataset** — see `data/README.md` for instructions.
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt  # (add this file when dependencies are finalized)
   ```
4. **Run the notebook** `notebooks/ANN_Project.ipynb` for the full pipeline.

## Results

All plots, confusion matrices, and output files go in the `results/` directory.
