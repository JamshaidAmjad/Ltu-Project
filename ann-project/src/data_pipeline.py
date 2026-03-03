# data_pipeline.py — Owned by P1
# Responsible for loading, preprocessing, and preparing datasets for training.

def load_data(path):
    """Load dataset from the given path."""
    raise NotImplementedError("Implement data loading here.")


def preprocess(data):
    """Apply preprocessing steps to the raw data."""
    raise NotImplementedError("Implement preprocessing here.")


def get_dataloaders(train_path, test_path, batch_size=32):
    """Return train and test data loaders/splits."""
    raise NotImplementedError("Implement dataloader creation here.")
