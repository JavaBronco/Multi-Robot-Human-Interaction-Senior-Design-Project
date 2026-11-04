"""Scripted policies and frozen self-play snapshots behind one ``Opponent`` interface."""

import copy
from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np
import torch

from core.agent.policy import Policy, obs_to_tensor
from core.interfaces.env_api import ACTION_MASK_KEY


class Opponent(ABC):
    """Anything that picks an action for the other side from its own observation."""

    name: str

    @abstractmethod
    def act(self, obs: dict[str, np.ndarray], rng: np.random.Generator) -> int:
        """Return a legal action for a single (unbatched) observation."""


class ScriptedOpponent(Opponent):
    """Fixed algorithm, e.g. a Pokémon NPC trainer. ``fn(obs, rng) -> action``."""

    def __init__(self, name: str, fn: Callable[[dict[str, np.ndarray], np.random.Generator], int]):
        self.name, self.fn = name, fn

    def act(self, obs: dict[str, np.ndarray], rng: np.random.Generator) -> int:
        return int(self.fn(obs, rng))


class SnapshotOpponent(Opponent):
    """Frozen copy of a policy (a past checkpoint) that samples from its masked distribution."""

    def __init__(self, name: str, policy: Policy) -> None:
        self.name = name
        self.policy = copy.deepcopy(policy).cpu().eval()
        self.policy.requires_grad_(False)

    def act(self, obs: dict[str, np.ndarray], rng: np.random.Generator) -> int:
        batch = obs_to_tensor({k: np.asarray(v)[None] for k, v in obs.items()}, torch.device("cpu"))
        with torch.no_grad():
            probs = self.policy.distribution(batch).probs[0].double().numpy()
        return int(rng.choice(len(probs), p=probs / probs.sum()))


def random_legal_action(obs: dict[str, np.ndarray], rng: np.random.Generator) -> int:
    """Uniformly random legal action; used for scripted random opponents and baselines."""
    return int(rng.choice(np.flatnonzero(obs[ACTION_MASK_KEY])))


class OpponentPool:
    """Scripted opponents plus up to ``max_snapshots`` most recent self-play snapshots."""

    def __init__(self, scripted: list[Opponent], max_snapshots: int = 10) -> None:
        if not scripted:
            raise ValueError("an opponent pool needs at least one scripted opponent")
        self.scripted = list(scripted)
        self.snapshots: list[SnapshotOpponent] = []
        self.max_snapshots = max_snapshots

    @property
    def opponents(self) -> list[Opponent]:
        return self.scripted + self.snapshots

    def add_snapshot(self, policy: Policy, name: str) -> None:
        """Freeze the current policy into the pool, evicting the oldest snapshot if full."""
        self.snapshots.append(SnapshotOpponent(name, policy))
        if len(self.snapshots) > self.max_snapshots:
            self.snapshots.pop(0)

    def sample(self, rng: np.random.Generator) -> Opponent:
        """Pick an opponent uniformly at random."""
        opponents = self.opponents
        return opponents[int(rng.integers(len(opponents)))]
