# Responsible for: small, reusable helpers used across the project —
# config loading, reproducibility seeding, logging, and path resolution.

import random
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(config_path: str | Path) -> dict:
    """Load a YAML config file and return it as a plain dict."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """Fix all relevant random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic mode — may slow down GPU training slightly.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(preferred: str = "cuda") -> torch.device:
    """Return a CUDA device if available, otherwise fall back to CPU."""
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_checkpoint(state: dict, path: str | Path) -> None:
    """Save a model checkpoint dict to disk."""
    # TODO: call torch.save(state, path) and log the save location
    raise NotImplementedError


def load_checkpoint(path: str | Path, model: torch.nn.Module, device: torch.device) -> dict:
    """Load a checkpoint from disk into model (in-place).

    Returns the full checkpoint dict so the caller can restore optimizer
    state, epoch counter, and best metric if needed.
    """
    # TODO: torch.load with map_location, model.load_state_dict
    raise NotImplementedError
