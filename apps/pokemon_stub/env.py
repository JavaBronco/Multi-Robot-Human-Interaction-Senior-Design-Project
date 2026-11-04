"""PLACEHOLDER two-player battle env that only demonstrates the core contract.

There are no real battle mechanics here. Each move deals a fixed placeholder fraction
of HP; switching swaps the active mon. Its only purpose is to show the obs dict,
action space (4 moves + 5 switches), action mask and opponent plumbing.

TODO(pokemon team): replace ``_resolve_turn`` and the observation with a real Gen 3
(Ruby/Sapphire) simulator, e.g. a Pokémon Showdown server driven through poke-env
(gen3randombattle / gen3ou formats), keeping the same observation keys or updating
``apps/pokemon_stub/encoder.py`` to match.
"""

from typing import Any

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from apps.pokemon_stub.rules import N_ACTIONS, N_MOVES, TEAM_SIZE, PokemonRuleFilter, bench
from core.interfaces.env_api import ACTION_MASK_KEY, SCENARIO_SEED_KEY

# TODO(pokemon team): real damage comes from the simulator (stats, types, STAB, crits, RNG).
PLACEHOLDER_MOVE_DAMAGE = np.array([0.10, 0.15, 0.20, 0.25])


class PokemonStubParallelEnv(ParallelEnv):
    """Symmetric 6v6 placeholder battle. Reward +1 win / -1 loss; 0 on turn-limit draw."""

    metadata = {"name": "pokemon_stub_v0", "render_modes": []}

    def __init__(self, max_turns: int = 100) -> None:
        self.possible_agents = ["player_0", "player_1"]
        self.agents: list[str] = []
        self.max_turns = max_turns
        self.rules = PokemonRuleFilter()
        team = spaces.Box(0.0, 1.0, (TEAM_SIZE, 2), np.float32)  # [hp fraction, is_active]
        self._obs_space = spaces.Dict(
            {"own_team": team, "opp_team": team, ACTION_MASK_KEY: spaces.MultiBinary(N_ACTIONS)}
        )
        self._act_space = spaces.Discrete(N_ACTIONS)

    def observation_space(self, agent: str) -> spaces.Dict:
        return self._obs_space

    def action_space(self, agent: str) -> spaces.Discrete:
        return self._act_space

    def _team_array(self, agent: str) -> np.ndarray:
        arr = np.zeros((TEAM_SIZE, 2), dtype=np.float32)
        arr[:, 0] = self.hp[agent]
        arr[self.active[agent], 1] = 1.0
        return arr

    def _obs(self) -> dict[str, dict[str, np.ndarray]]:
        a, b = self.possible_agents
        out = {}
        for me, opp in ((a, b), (b, a)):
            out[me] = {
                "own_team": self._team_array(me),
                "opp_team": self._team_array(opp),  # TODO: real battles hide opponent info
                ACTION_MASK_KEY: self.rules.mask((self.hp[me], self.active[me])),
            }
        return out

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, Any]]]:
        scenario = (options or {}).get(SCENARIO_SEED_KEY, seed)
        # TODO(pokemon team): the scenario seed should pick the NPC trainer's team and battle RNG.
        self.rng = np.random.default_rng(scenario)
        self.agents = list(self.possible_agents)
        self.hp = {a: np.ones(TEAM_SIZE, dtype=np.float32) for a in self.agents}
        self.active = {a: 0 for a in self.agents}
        self.turn = 0
        return self._obs(), {a: {} for a in self.agents}

    def _resolve_turn(self, actions: dict[str, int]) -> None:
        """PLACEHOLDER turn: switches first, then moves in random order. Not real mechanics."""
        a, b = self.possible_agents
        masks = {ag: self.rules.mask((self.hp[ag], self.active[ag])) for ag in (a, b)}
        legal = {ag: masks[ag][actions[ag]] == 1 for ag in (a, b)}  # illegal action = no-op
        for ag in (a, b):
            if legal[ag] and actions[ag] >= N_MOVES:
                self.active[ag] = bench(self.active[ag])[actions[ag] - N_MOVES]
        order = [a, b] if self.rng.random() < 0.5 else [b, a]  # TODO: speed / priority
        for ag in order:
            opp = b if ag == a else a
            if legal[ag] and actions[ag] < N_MOVES and self.hp[ag][self.active[ag]] > 0:
                target = self.active[opp]
                damage = PLACEHOLDER_MOVE_DAMAGE[actions[ag]]
                self.hp[opp][target] = max(0.0, self.hp[opp][target] - damage)

    def step(self, actions: dict[str, int]) -> tuple[dict, dict, dict, dict, dict]:
        self._resolve_turn({ag: int(act) for ag, act in actions.items()})
        self.turn += 1
        alive = {ag: bool((self.hp[ag] > 0).any()) for ag in self.agents}
        over = not all(alive.values())
        truncated = not over and self.turn >= self.max_turns
        rewards, infos = {}, {}
        for ag in self.agents:
            opp = next(o for o in self.agents if o != ag)
            won = alive[ag] and not alive[opp]
            rewards[ag] = float(won) - float(alive[opp] and not alive[ag])
            infos[ag] = {"is_success": won} if over or truncated else {}
        terms = {ag: over for ag in self.agents}
        truncs = {ag: truncated for ag in self.agents}
        obs = self._obs()
        if over or truncated:
            self.agents = []
        return obs, rewards, terms, truncs, infos
