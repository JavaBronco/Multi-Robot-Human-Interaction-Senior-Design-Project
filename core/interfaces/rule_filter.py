"""Abstract rule filter: game/world state -> boolean mask of legal actions."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class RuleFilter(ABC):
    """Encodes an app's legality rules. The env calls it to fill ``obs["action_mask"]``."""

    def __init__(self, n_actions: int) -> None:
        self.n_actions = n_actions

    @abstractmethod
    def mask(self, state: Any) -> np.ndarray:
        """Return an int8 array of shape (n_actions,), 1 = legal. At least one entry must be 1."""
