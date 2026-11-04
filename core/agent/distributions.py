"""Categorical distribution with logit masking: illegal actions get exactly zero probability."""

import torch
from torch.distributions import Categorical


class MaskedCategorical(Categorical):
    """``Categorical`` whose illegal logits are replaced by the dtype minimum before softmax.

    Args:
        logits: (B, n) unnormalised scores.
        mask: (B, n) bool/int tensor, nonzero = legal. Every row needs a legal action.
    """

    def __init__(self, logits: torch.Tensor, mask: torch.Tensor) -> None:
        self.mask = mask.bool()
        if not self.mask.any(dim=-1).all():
            raise ValueError("every row of the action mask needs at least one legal action")
        masked = torch.where(self.mask, logits, torch.finfo(logits.dtype).min)
        super().__init__(logits=masked)

    def entropy(self) -> torch.Tensor:
        """Entropy over legal actions only (avoids 0 * -inf = nan on masked entries)."""
        p_log_p = torch.where(self.mask, self.probs * self.logits, 0.0)
        return -p_log_p.sum(dim=-1)
