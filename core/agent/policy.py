"""App-agnostic actor-critic: pluggable encoder -> shared trunk -> per-app action head + value."""

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn

from core.agent.distributions import MaskedCategorical
from core.interfaces.encoder import StateEncoder
from core.interfaces.env_api import ACTION_MASK_KEY


def obs_to_tensor(obs: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    """Convert a batched numpy observation dict to tensors (mask -> bool, rest -> float32)."""
    out = {}
    for key, value in obs.items():
        dtype = torch.bool if key == ACTION_MASK_KEY else torch.float32
        out[key] = torch.as_tensor(np.asarray(value), device=device).to(dtype)
    return out


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias: float = 0.0) -> nn.Linear:
    """Orthogonal init as in CleanRL."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class Policy(nn.Module):
    """Actor-critic with an app-specific encoder and action head around a shared MLP trunk.

    Each app builds its own instance (own weights); the trunk/head/value code is shared.
    """

    def __init__(
        self, encoder: StateEncoder, n_actions: int, hidden_sizes: Sequence[int] = (128, 128)
    ) -> None:
        super().__init__()
        self.encoder = encoder
        layers: list[nn.Module] = []
        in_dim = encoder.out_dim
        for size in hidden_sizes:
            layers += [layer_init(nn.Linear(in_dim, size)), nn.Tanh()]
            in_dim = size
        self.trunk = nn.Sequential(*layers)
        self.action_head = layer_init(nn.Linear(in_dim, n_actions), std=0.01)
        self.value_head = layer_init(nn.Linear(in_dim, 1), std=1.0)

    def forward(self, obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (logits (B, n), value (B,))."""
        h = self.trunk(self.encoder(obs))
        return self.action_head(h), self.value_head(h).squeeze(-1)

    def distribution(self, obs: dict[str, torch.Tensor]) -> MaskedCategorical:
        """Masked action distribution for a batch of observations."""
        logits, _ = self(obs)
        return MaskedCategorical(logits, obs[ACTION_MASK_KEY])

    def act(
        self, obs: dict[str, torch.Tensor], deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample (or argmax) a legal action. Returns (action, log_prob, value)."""
        logits, value = self(obs)
        dist = MaskedCategorical(logits, obs[ACTION_MASK_KEY])
        action = dist.probs.argmax(-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), value

    def evaluate_actions(
        self, obs: dict[str, torch.Tensor], actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (log_prob, entropy, value) of ``actions`` under the current masked policy."""
        logits, value = self(obs)
        dist = MaskedCategorical(logits, obs[ACTION_MASK_KEY])
        return dist.log_prob(actions), dist.entropy(), value
