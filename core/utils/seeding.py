"""Global seeding for python, numpy and torch. Envs are seeded through ``reset(seed=...)``."""

import random

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Seed python ``random``, numpy and torch (CPU and CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
