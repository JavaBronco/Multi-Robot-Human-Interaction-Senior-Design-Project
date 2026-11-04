"""The contract every app env must satisfy so core/ can train and evaluate it unchanged.

An app env is a ``gymnasium.Env`` (multi-agent envs are adapted with
``core.selfplay.opponent_env.OpponentEnv``) where:

* ``observation_space`` is a ``spaces.Dict`` containing ``ACTION_MASK_KEY`` as a
  ``spaces.MultiBinary(n)`` alongside whatever keys the app's encoder consumes.
* ``action_space`` is ``spaces.Discrete(n)`` with the same ``n``.
* Every observation's mask has at least one legal action.
* ``step`` returns ``(obs, reward: float, terminated, truncated, info)``; on the final
  step of an episode ``info["is_success"]`` is a bool (goal reached / battle won).
* Scenarios (maps, battles) are generated from a scenario seed. ``reset`` draws one
  from ``TRAIN_SEED_RANGE`` unless ``options[SCENARIO_SEED_KEY]`` is given, and reports
  the seed used in ``info[SCENARIO_SEED_KEY]``. Evaluation uses seeds >= ``EVAL_SEED_START``,
  so test scenarios never appear during training.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

ACTION_MASK_KEY = "action_mask"
SCENARIO_SEED_KEY = "scenario_seed"
TRAIN_SEED_RANGE = range(0, 10**6)
EVAL_SEED_START = 10**9


def check_env_contract(env: gym.Env, n_steps: int = 200, seed: int = 0) -> None:
    """Assert that ``env`` satisfies the contract above. Raises AssertionError otherwise."""
    obs_space, act_space = env.observation_space, env.action_space
    assert isinstance(obs_space, spaces.Dict), "observation_space must be spaces.Dict"
    assert ACTION_MASK_KEY in obs_space.spaces, f"observation must contain '{ACTION_MASK_KEY}'"
    mask_space = obs_space[ACTION_MASK_KEY]
    assert isinstance(mask_space, spaces.MultiBinary), "action_mask must be spaces.MultiBinary"
    assert isinstance(act_space, spaces.Discrete), "action_space must be spaces.Discrete"
    assert mask_space.shape == (act_space.n,), "mask size must equal number of actions"

    rng = np.random.default_rng(seed)
    obs, info = env.reset(seed=seed)
    assert info[SCENARIO_SEED_KEY] in TRAIN_SEED_RANGE, "default reset must use a training seed"
    episodes = 0
    for _ in range(n_steps):
        assert obs_space.contains(obs), "observation outside observation_space"
        mask = obs[ACTION_MASK_KEY]
        assert mask.any(), "at least one action must be legal"
        action = int(rng.choice(np.flatnonzero(mask)))
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(reward, float), "reward must be a Python float"
        if terminated or truncated:
            assert isinstance(info.get("is_success"), bool), "final info needs bool 'is_success'"
            episodes += 1
            obs, info = env.reset()

    eval_seed = EVAL_SEED_START + 7
    obs_a, info = env.reset(options={SCENARIO_SEED_KEY: eval_seed})
    assert info[SCENARIO_SEED_KEY] == eval_seed, "reset must honour options['scenario_seed']"
    obs_b, _ = env.reset(options={SCENARIO_SEED_KEY: eval_seed})
    for key in obs_a:
        assert np.array_equal(obs_a[key], obs_b[key]), f"scenario seed not deterministic ({key})"
