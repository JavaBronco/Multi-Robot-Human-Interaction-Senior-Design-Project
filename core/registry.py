"""How core/ finds an app: the config names a module, and that module exposes ``APP``."""

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
from gymnasium import spaces

from core.interfaces.encoder import StateEncoder
from core.selfplay.opponent_pool import Opponent, OpponentPool


@dataclass
class AppSpec:
    """Everything core/ needs from an app.

    Attributes:
        name: Short app name used in logs.
        make_env: (env config, opponent pool or None) -> env satisfying ``core.interfaces.env_api``.
        make_encoder: (observation space, encoder config) -> StateEncoder.
        train_opponents: Scripted opponents seen during training (None for single-agent apps).
        eval_opponents: Held-out scripted opponents used only by evaluation.
    """

    name: str
    make_env: Callable[[dict[str, Any], OpponentPool | None], gym.Env]
    make_encoder: Callable[[spaces.Dict, dict[str, Any]], StateEncoder]
    train_opponents: Callable[[], list[Opponent]] | None = None
    eval_opponents: Callable[[], list[Opponent]] | None = None


def load_app(module: str) -> AppSpec:
    """Import ``module`` (e.g. ``apps.robot_grid``) and return its ``APP`` spec."""
    app = getattr(importlib.import_module(module), "APP", None)
    if not isinstance(app, AppSpec):
        raise TypeError(f"{module} must define APP = AppSpec(...)")
    return app
