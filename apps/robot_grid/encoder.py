"""Robot grid encoder: small CNN over the view window + MLP over the status vector."""

import torch
from gymnasium import spaces
from torch import nn

from core.interfaces.encoder import StateEncoder


class GridEncoder(StateEncoder):
    """Concatenates CNN(window) and MLP(status) features."""

    def __init__(
        self,
        obs_space: spaces.Dict,
        channels: int = 32,
        window_dim: int = 128,
        status_dim: int = 32,
    ) -> None:
        super().__init__()
        in_ch, h, w = obs_space["window"].shape
        self.cnn = nn.Sequential(
            nn.Conv2d(in_ch, channels // 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels // 2, channels, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(channels * h * w, window_dim),
            nn.ReLU(),
        )
        self.status_mlp = nn.Sequential(
            nn.Linear(obs_space["status"].shape[0], status_dim), nn.ReLU()
        )
        self._out_dim = window_dim + status_dim

    @property
    def out_dim(self) -> int:
        return self._out_dim

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([self.cnn(obs["window"]), self.status_mlp(obs["status"])], dim=-1)
