import numpy as np
import torch
from gymnasium.utils.env_checker import check_env

from apps.robot_grid.encoder import GridEncoder
from apps.robot_grid.env import RobotGridEnv
from apps.robot_grid.rules import BIG_ROCK, CLIMB, JUMP, WALK, GridRuleFilter, MoveCosts
from core.agent.policy import obs_to_tensor
from core.interfaces.env_api import EVAL_SEED_START, SCENARIO_SEED_KEY, check_env_contract


def test_gymnasium_env_checker():
    check_env(RobotGridEnv(), skip_render_check=True)


def test_env_contract():
    check_env_contract(RobotGridEnv(), n_steps=500)


def test_same_seed_same_map_different_seed_different_map():
    env = RobotGridEnv()
    env.reset(options={SCENARIO_SEED_KEY: EVAL_SEED_START})
    a = env.grid.copy()
    env.reset(options={SCENARIO_SEED_KEY: EVAL_SEED_START})
    assert np.array_equal(a, env.grid)
    env.reset(options={SCENARIO_SEED_KEY: EVAL_SEED_START + 1})
    assert not np.array_equal(a, env.grid)


def test_rule_filter_masks_impossible_moves():
    rules = GridRuleFilter(MoveCosts())
    grid = np.zeros((3, 3), dtype=int)
    grid[0, 1] = BIG_ROCK  # rock above the centre
    mask = rules.mask((grid, (1, 1)))
    up = 0
    assert mask[WALK * 4 + up] == 0  # can't walk into rock
    assert mask[CLIMB * 4 + up] == 1  # can climb it
    assert mask[CLIMB * 4 + 1] == 0  # can't climb soil
    assert not mask[JUMP * 4 : JUMP * 4 + 4].any()  # every jump lands off a 3x3 grid
    corner = rules.mask((grid, (0, 0)))
    assert corner[WALK * 4 + 0] == 0 and corner[WALK * 4 + 3] == 0  # off-grid


def test_encoder_output_shape():
    env = RobotGridEnv()
    obs, _ = env.reset(seed=0)
    enc = GridEncoder(env.observation_space)
    batch = obs_to_tensor({k: v[None] for k, v in obs.items()}, torch.device("cpu"))
    assert enc(batch).shape == (1, enc.out_dim)
