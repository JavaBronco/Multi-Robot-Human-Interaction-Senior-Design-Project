"""Masked PPO (clipped objective, GAE, value + entropy losses), CleanRL-style.

Masks are stored in the rollout buffer and re-applied in the update, so the
probability ratio is always computed over the same legal-action set the
behaviour policy sampled from.
"""

from dataclasses import dataclass

import torch
from torch import nn

from core.agent.policy import Policy
from core.algo.buffer import RolloutBuffer


@dataclass
class PPOConfig:
    """PPO hyperparameters (defaults follow CleanRL's discrete-action PPO)."""

    num_steps: int = 128
    learning_rate: float = 2.5e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalised Advantage Estimation over a (T, N) rollout.

    Args:
        rewards, values, dones: (T, N); ``dones[t]`` = episode ended after step t.
        last_value: (N,) value of the observation following the last step.

    Returns:
        (advantages, returns), each (T, N).
    """
    num_steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros_like(last_value)
    for t in reversed(range(num_steps)):
        next_value = last_value if t == num_steps - 1 else values[t + 1]
        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * not_done - values[t]
        last_gae = delta + gamma * gae_lambda * not_done * last_gae
        advantages[t] = last_gae
    return advantages, advantages + values


class PPO:
    """Owns the optimiser and performs the clipped-surrogate update on a filled buffer."""

    def __init__(self, policy: Policy, cfg: PPOConfig) -> None:
        self.policy, self.cfg = policy, cfg
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.learning_rate, eps=1e-5)

    def set_progress(self, fraction_done: float) -> None:
        """Linearly anneal the learning rate given the fraction of training completed."""
        if self.cfg.anneal_lr:
            lr = (1.0 - fraction_done) * self.cfg.learning_rate
            for group in self.optimizer.param_groups:
                group["lr"] = lr

    def update(self, buffer: RolloutBuffer, last_value: torch.Tensor) -> dict[str, float]:
        """Compute GAE into ``buffer`` and run the PPO epochs. Returns scalar diagnostics."""
        cfg = self.cfg
        buffer.advantages, buffer.returns = compute_gae(
            buffer.rewards, buffer.values, buffer.dones, last_value, cfg.gamma, cfg.gae_lambda
        )
        batch_size = buffer.num_steps * buffer.num_envs
        minibatch_size = batch_size // cfg.num_minibatches
        clipfracs, approx_kl = [], torch.tensor(0.0)
        for _ in range(cfg.update_epochs):
            for mb in buffer.minibatches(minibatch_size):
                new_logp, entropy, new_value = self.policy.evaluate_actions(
                    mb["obs"], mb["actions"]
                )
                log_ratio = new_logp - mb["logprobs"]
                ratio = log_ratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item())

                adv = mb["advantages"]
                if cfg.norm_adv:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                pg_loss = torch.max(
                    -adv * ratio, -adv * ratio.clamp(1 - cfg.clip_coef, 1 + cfg.clip_coef)
                ).mean()

                if cfg.clip_vloss:
                    v_clipped = mb["values"] + (new_value - mb["values"]).clamp(
                        -cfg.clip_coef, cfg.clip_coef
                    )
                    v_loss = (
                        0.5
                        * torch.max(
                            (new_value - mb["returns"]) ** 2, (v_clipped - mb["returns"]) ** 2
                        ).mean()
                    )
                else:
                    v_loss = 0.5 * ((new_value - mb["returns"]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - cfg.ent_coef * entropy_loss + cfg.vf_coef * v_loss
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                self.optimizer.step()
            if cfg.target_kl is not None and approx_kl > cfg.target_kl:
                break

        y_pred, y_true = buffer.values.flatten(), buffer.returns.flatten()
        var_y = y_true.var()
        explained_var = float("nan") if var_y == 0 else (1 - (y_true - y_pred).var() / var_y).item()
        return {
            "losses/policy_loss": pg_loss.item(),
            "losses/value_loss": v_loss.item(),
            "losses/entropy": entropy_loss.item(),
            "losses/approx_kl": approx_kl.item(),
            "losses/clipfrac": sum(clipfracs) / len(clipfracs),
            "losses/explained_variance": explained_var,
            "charts/learning_rate": self.optimizer.param_groups[0]["lr"],
        }
