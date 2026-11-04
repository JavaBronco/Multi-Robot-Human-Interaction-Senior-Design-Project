"""Stub encoder: MLP over both teams' [hp, is_active] tables.

TODO(pokemon team): replace with an encoder for the real battle state (species, moves,
types, stats, status, field) once the simulator observation is defined.
"""

import torch
from gymnasium import spaces
from torch import nn

from core.interfaces.encoder import StateEncoder


class TeamMLPEncoder(StateEncoder):
    """Flattens ``own_team`` and ``opp_team`` and applies a small MLP."""

    def __init__(self, obs_space: spaces.Dict, hidden: int = 64) -> None:
        super().__init__()
        in_dim = obs_space["own_team"].shape[0] * obs_space["own_team"].shape[1] * 2
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU())
        self._out_dim = hidden

    @property
    def out_dim(self) -> int:
        return self._out_dim

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        x = torch.cat([obs["own_team"].flatten(1), obs["opp_team"].flatten(1)], dim=-1)
        return self.net(x)
