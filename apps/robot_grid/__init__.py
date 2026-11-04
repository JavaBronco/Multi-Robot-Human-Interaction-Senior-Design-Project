"""Robot navigation, Phase 1: 2D grid. Registers ``APP`` for core/."""

from apps.robot_grid.encoder import GridEncoder
from apps.robot_grid.env import RobotGridEnv
from core.registry import AppSpec

APP = AppSpec(
    name="robot_grid",
    make_env=lambda env_cfg, pool: RobotGridEnv(**env_cfg),
    make_encoder=lambda obs_space, enc_cfg: GridEncoder(obs_space, **enc_cfg),
)
