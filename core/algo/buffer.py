"""On-policy rollout storage, including the action mask seen at every step."""

from collections.abc import Iterator

import torch
from gymnasium import spaces

from core.interfaces.env_api import ACTION_MASK_KEY


class RolloutBuffer:
    """Fixed-size (num_steps, num_envs) storage for one PPO rollout.

    ``dones[t]`` is 1 when the episode ended *after* the action at step t, so obs t+1
    belongs to a new episode and must not be bootstrapped across.
    """

    def __init__(
        self, obs_space: spaces.Dict, num_steps: int, num_envs: int, device: torch.device
    ) -> None:
        self.num_steps, self.num_envs = num_steps, num_envs
        shape = (num_steps, num_envs)
        self.obs = {
            key: torch.zeros(
                shape + space.shape,
                dtype=torch.bool if key == ACTION_MASK_KEY else torch.float32,
                device=device,
            )
            for key, space in obs_space.spaces.items()
        }
        self.actions = torch.zeros(shape, dtype=torch.long, device=device)
        self.logprobs = torch.zeros(shape, device=device)
        self.rewards = torch.zeros(shape, device=device)
        self.dones = torch.zeros(shape, device=device)
        self.values = torch.zeros(shape, device=device)
        self.advantages = torch.zeros(shape, device=device)
        self.returns = torch.zeros(shape, device=device)

    def add(
        self,
        step: int,
        obs: dict[str, torch.Tensor],
        action: torch.Tensor,
        logprob: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """Store one vectorised transition at index ``step``."""
        for key, tensor in obs.items():
            self.obs[key][step] = tensor
        self.actions[step] = action
        self.logprobs[step] = logprob
        self.rewards[step] = reward
        self.dones[step] = done
        self.values[step] = value

    def minibatches(
        self, minibatch_size: int, generator: torch.Generator | None = None
    ) -> Iterator[dict[str, torch.Tensor | dict[str, torch.Tensor]]]:
        """Yield shuffled flat minibatches (masks included) for the PPO update."""
        total = self.num_steps * self.num_envs
        flat_obs = {k: v.reshape((total,) + v.shape[2:]) for k, v in self.obs.items()}
        flat = {
            name: getattr(self, name).reshape(total)
            for name in ("actions", "logprobs", "values", "advantages", "returns")
        }
        perm = torch.randperm(total, generator=generator).to(self.actions.device)
        for start in range(0, total, minibatch_size):
            idx = perm[start : start + minibatch_size]
            batch: dict = {name: t[idx] for name, t in flat.items()}
            batch["obs"] = {k: v[idx] for k, v in flat_obs.items()}
            yield batch
