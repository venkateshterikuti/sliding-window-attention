from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch


def select_device(preferred: Optional[str] = None) -> torch.device:
    """Pick an available accelerator (cuda/mps) or fall back to cpu."""
    if preferred:
        return torch.device(preferred)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synchronize(device: torch.device) -> None:
    """Barrier across accelerators so timing is correct."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()
