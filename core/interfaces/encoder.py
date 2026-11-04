"""Abstract state encoder ("translator"): raw observation dict -> feature vector."""

from abc import ABC, abstractmethod

import torch
from torch import nn


class StateEncoder(nn.Module, ABC):
    """Maps a batched observation dict to a flat feature tensor of shape (B, out_dim).

    Encoders must ignore the ``action_mask`` key; the policy applies the mask itself.
    """

    @property
    @abstractmethod
    def out_dim(self) -> int:
        """Size of the feature vector produced by ``forward``."""

    @abstractmethod
    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode a batch of observations."""
