"""Movement rules for the 2D grid: what each action does on each terrain, and what is legal."""

from dataclasses import dataclass

import numpy as np

from core.interfaces.rule_filter import RuleFilter

TERRAINS = ("soil", "bedrock", "sand", "big_rock")
SOIL, BEDROCK, SAND, BIG_ROCK = range(len(TERRAINS))
KINDS = ("walk", "climb", "jump", "crawl")
WALK, CLIMB, JUMP, CRAWL = range(len(KINDS))
DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))  # up, right, down, left as (drow, dcol)
N_ACTIONS = len(KINDS) * len(DIRECTIONS)


def decode_action(action: int) -> tuple[int, tuple[int, int]]:
    """Action index -> (kind, direction). Index = kind * 4 + direction."""
    return action // len(DIRECTIONS), DIRECTIONS[action % len(DIRECTIONS)]


@dataclass
class MoveCosts:
    """Energy and damage per move; all tunable from the config."""

    walk_energy: float = 1.0
    walk_sand_energy: float = 2.0
    walk_sand_damage: float = 1.0
    crawl_energy: float = 2.0
    climb_energy: float = 4.0
    jump_energy: float = 3.0
    jump_bedrock_damage: float = 1.0


@dataclass
class Outcome:
    """Result of a legal move."""

    pos: tuple[int, int]
    energy: float
    damage: float


class GridRuleFilter(RuleFilter):
    """Legality and cost of moves.

    walk/crawl: target cell not big_rock. climb: target cell is big_rock.
    jump: moves two cells, landing cell in bounds and not big_rock. Never leave the grid.
    """

    def __init__(self, costs: MoveCosts) -> None:
        super().__init__(N_ACTIONS)
        self.costs = costs

    def outcome(self, grid: np.ndarray, pos: tuple[int, int], action: int) -> Outcome | None:
        """Return the move's outcome, or None if the move is illegal."""
        kind, (dr, dc) = decode_action(action)
        dist = 2 if kind == JUMP else 1
        r, c = pos[0] + dr * dist, pos[1] + dc * dist
        if not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]):
            return None
        target = grid[r, c]
        k = self.costs
        if kind == CLIMB:
            return Outcome((r, c), k.climb_energy, 0.0) if target == BIG_ROCK else None
        if target == BIG_ROCK:
            return None
        if kind == WALK:
            if target == SAND:
                return Outcome((r, c), k.walk_sand_energy, k.walk_sand_damage)
            return Outcome((r, c), k.walk_energy, 0.0)
        if kind == CRAWL:
            return Outcome((r, c), k.crawl_energy, 0.0)
        damage = k.jump_bedrock_damage if target == BEDROCK else 0.0
        return Outcome((r, c), k.jump_energy, damage)

    def mask(self, state: tuple[np.ndarray, tuple[int, int]]) -> np.ndarray:
        """state = (grid, robot position)."""
        grid, pos = state
        return np.array(
            [self.outcome(grid, pos, a) is not None for a in range(N_ACTIONS)], dtype=np.int8
        )
