"""Scripted NPC trainers for the stub.

TODO(pokemon team): implement the real Gen 3 NPC trainer AI scripts here as
``ScriptedOpponent``s (they follow fixed algorithms), split into training and
held-out evaluation trainers.
"""

import numpy as np

from apps.pokemon_stub.rules import N_MOVES
from core.interfaces.env_api import ACTION_MASK_KEY
from core.selfplay.opponent_pool import Opponent, ScriptedOpponent, random_legal_action


def strongest_move(obs: dict[str, np.ndarray], rng: np.random.Generator) -> int:
    """Use the highest-index legal move (placeholder 'strongest'); otherwise first legal switch."""
    legal = np.flatnonzero(obs[ACTION_MASK_KEY])
    moves = legal[legal < N_MOVES]
    return int(moves.max() if len(moves) else legal[0])


def train_opponents() -> list[Opponent]:
    """Trainers seen during training."""
    return [ScriptedOpponent("random", random_legal_action)]


def eval_opponents() -> list[Opponent]:
    """Held-out trainers used only by evaluation."""
    return [ScriptedOpponent("strongest_move", strongest_move)]
