"""Turn a two-player PettingZoo ``ParallelEnv`` into a single-agent Gymnasium env.

The second player is driven by an ``Opponent`` sampled from an ``OpponentPool`` at every
reset, so scripted trainers and self-play snapshots look identical to the trainer.
"""

from typing import Any

import gymnasium as gym
import numpy as np
from pettingzoo import ParallelEnv

from core.interfaces.env_api import SCENARIO_SEED_KEY, TRAIN_SEED_RANGE
from core.selfplay.opponent_pool import Opponent, OpponentPool


class OpponentEnv(gym.Env):
    """Single-agent view of ``par_env`` from ``possible_agents[0]``'s perspective.

    The wrapped env must accept ``options[SCENARIO_SEED_KEY]`` in ``reset``.
    """

    metadata = {"render_modes": []}

    def __init__(self, par_env: ParallelEnv, pool: OpponentPool) -> None:
        self.par_env, self.pool = par_env, pool
        self.learner, self.other = par_env.possible_agents
        self.observation_space = par_env.observation_space(self.learner)
        self.action_space = par_env.action_space(self.learner)
        self.opponent: Opponent | None = None

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        options = dict(options or {})
        if options.get(SCENARIO_SEED_KEY) is None:
            options[SCENARIO_SEED_KEY] = int(
                self.np_random.integers(TRAIN_SEED_RANGE.start, TRAIN_SEED_RANGE.stop)
            )
        self.opponent = self.pool.sample(self.np_random)
        obs, infos = self.par_env.reset(seed=options[SCENARIO_SEED_KEY], options=options)
        self._opp_obs = obs[self.other]
        info = {
            **infos[self.learner],
            "opponent": self.opponent.name,
            SCENARIO_SEED_KEY: options[SCENARIO_SEED_KEY],
        }
        return obs[self.learner], info

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        opp_action = self.opponent.act(self._opp_obs, self.np_random)
        obs, rewards, terms, truncs, infos = self.par_env.step(
            {self.learner: int(action), self.other: opp_action}
        )
        self._opp_obs = obs[self.other]
        info = {**infos[self.learner], "opponent": self.opponent.name}
        return (
            obs[self.learner],
            float(rewards[self.learner]),
            bool(terms[self.learner]),
            bool(truncs[self.learner]),
            info,
        )
