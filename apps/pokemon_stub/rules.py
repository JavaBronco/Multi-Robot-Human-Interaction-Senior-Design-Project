"""Legality rules for the stub battle: 4 moves + 5 switches.

TODO(pokemon team): replace with real Gen 3 legality from the simulator state
(PP left, Disable/Taunt/Encore, Choice Band lock, trapping moves/abilities like
Shadow Tag and Arena Trap, forced switches after a faint, Struggle).
"""

import numpy as np

from core.interfaces.rule_filter import RuleFilter

N_MOVES, TEAM_SIZE = 4, 6
N_SWITCHES = TEAM_SIZE - 1
N_ACTIONS = N_MOVES + N_SWITCHES


def bench(active: int) -> list[int]:
    """Team indices reachable by switch actions 4..8, in team order."""
    return [i for i in range(TEAM_SIZE) if i != active]


class PokemonRuleFilter(RuleFilter):
    """Moves are legal while the active mon is alive; switches only to living bench mons."""

    def __init__(self) -> None:
        super().__init__(N_ACTIONS)

    def mask(self, state: tuple[np.ndarray, int]) -> np.ndarray:
        """state = (team HP fractions (6,), active index)."""
        hp, active = state
        mask = np.zeros(N_ACTIONS, dtype=np.int8)
        if hp[active] > 0:
            mask[:N_MOVES] = 1
        for j, idx in enumerate(bench(active)):
            mask[N_MOVES + j] = hp[idx] > 0
        if not mask.any():  # whole team fainted; battle is over, keep the mask non-empty
            mask[0] = 1
        return mask
