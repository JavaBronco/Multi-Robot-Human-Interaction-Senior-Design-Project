"""Phase 1 robot navigation: seeded procedural 2D grid, partial view, energy/health budgets."""

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from apps.robot_grid.rules import BIG_ROCK, N_ACTIONS, TERRAINS, GridRuleFilter, MoveCosts, Outcome
from core.interfaces.env_api import ACTION_MASK_KEY, SCENARIO_SEED_KEY, TRAIN_SEED_RANGE

N_CHANNELS = len(TERRAINS) + 2  # terrain one-hot, out-of-bounds, goal


class RobotGridEnv(gym.Env):
    """One robot crosses a procedurally generated grid from start to goal.

    Observation:
        window: (6, k, k) one-hot terrain + out-of-bounds + goal around the robot, k = 2r+1.
        status: [energy frac, health frac, goal drow / H, goal dcol / W].
        action_mask: legal actions from ``GridRuleFilter``.
    Reward: ``goal_bonus`` on reaching the goal, minus small per-step energy/damage/time costs.
    An illegal action (only possible if the mask is ignored) is a no-op that still costs a step.
    Episode ends on goal (success), energy or health exhausted, or ``max_steps`` (truncation).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        height: int = 8,
        width: int = 8,
        view_radius: int = 2,
        terrain_freqs: dict[str, float] | None = None,
        min_goal_distance: int | None = None,
        energy: float = 40.0,
        health: float = 5.0,
        max_steps: int | None = None,
        goal_bonus: float = 10.0,
        energy_cost: float = 0.01,
        damage_cost: float = 0.05,
        step_cost: float = 0.01,
        move_costs: dict[str, float] | None = None,
    ) -> None:
        self.height, self.width, self.view_radius = height, width, view_radius
        freqs = terrain_freqs or {"soil": 0.5, "bedrock": 0.2, "sand": 0.2, "big_rock": 0.1}
        p = np.array([freqs[t] for t in TERRAINS], dtype=np.float64)
        self.terrain_p = p / p.sum()
        self.min_goal_distance = min_goal_distance or (height + width) // 2
        self.max_energy, self.max_health = energy, health
        self.max_steps = max_steps or 4 * (height + width)
        self.goal_bonus, self.energy_cost = goal_bonus, energy_cost
        self.damage_cost, self.step_cost = damage_cost, step_cost
        self.rules = GridRuleFilter(MoveCosts(**(move_costs or {})))

        k = 2 * view_radius + 1
        self.observation_space = spaces.Dict(
            {
                "window": spaces.Box(0.0, 1.0, (N_CHANNELS, k, k), np.float32),
                "status": spaces.Box(-1.0, 1.0, (4,), np.float32),
                ACTION_MASK_KEY: spaces.MultiBinary(N_ACTIONS),
            }
        )
        self.action_space = spaces.Discrete(N_ACTIONS)

    def _generate(self, scenario_seed: int) -> None:
        rng = np.random.default_rng(scenario_seed)
        self.grid = rng.choice(len(TERRAINS), size=(self.height, self.width), p=self.terrain_p)
        free = np.argwhere(self.grid != BIG_ROCK)
        while True:  # resample start/goal until far enough apart
            start, goal = free[rng.choice(len(free), size=2, replace=False)]
            if np.abs(start - goal).sum() >= self.min_goal_distance:
                break
        self.pos, self.goal = (int(start[0]), int(start[1])), (int(goal[0]), int(goal[1]))

    def _obs(self) -> dict[str, np.ndarray]:
        r = self.view_radius
        window = np.zeros(self.observation_space["window"].shape, dtype=np.float32)
        for i in range(2 * r + 1):
            for j in range(2 * r + 1):
                gr, gc = self.pos[0] + i - r, self.pos[1] + j - r
                if 0 <= gr < self.height and 0 <= gc < self.width:
                    window[self.grid[gr, gc], i, j] = 1.0
                    window[-1, i, j] = float((gr, gc) == self.goal)
                else:
                    window[len(TERRAINS), i, j] = 1.0
        status = np.array(
            [
                self.energy / self.max_energy,
                self.health / self.max_health,
                (self.goal[0] - self.pos[0]) / self.height,
                (self.goal[1] - self.pos[1]) / self.width,
            ],
            dtype=np.float32,
        )
        mask = self.rules.mask((self.grid, self.pos))
        return {"window": window, "status": status, ACTION_MASK_KEY: mask}

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}
        scenario_seed = options.get(SCENARIO_SEED_KEY)
        if scenario_seed is None:
            scenario_seed = int(
                self.np_random.integers(TRAIN_SEED_RANGE.start, TRAIN_SEED_RANGE.stop)
            )
        self._generate(scenario_seed)
        self.energy, self.health, self.steps = self.max_energy, self.max_health, 0
        return self._obs(), {SCENARIO_SEED_KEY: scenario_seed}

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        outcome = self.rules.outcome(self.grid, self.pos, int(action))
        illegal = outcome is None
        if illegal:  # masked policies never get here; unmasked callers waste a step
            outcome = Outcome(self.pos, 0.0, 0.0)
        self.pos = outcome.pos
        self.energy -= outcome.energy
        self.health -= outcome.damage
        self.steps += 1

        reward = -(
            self.energy_cost * outcome.energy + self.damage_cost * outcome.damage + self.step_cost
        )
        success = self.pos == self.goal
        if success:
            reward += self.goal_bonus
        terminated = success or self.energy <= 0 or self.health <= 0
        truncated = not terminated and self.steps >= self.max_steps
        info: dict[str, Any] = {"illegal_action": True} if illegal else {}
        if terminated or truncated:
            info["is_success"] = bool(success)
        return self._obs(), float(reward), terminated, truncated, info
