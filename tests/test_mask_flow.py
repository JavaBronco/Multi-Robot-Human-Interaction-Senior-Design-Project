"""Masks must flow env -> buffer -> policy in both rollout and the PPO update."""

import torch
from gymnasium import spaces

from core.agent.policy import Policy
from core.algo.buffer import RolloutBuffer
from core.algo.ppo import PPO, PPOConfig
from core.interfaces.encoder import StateEncoder

N_ACTIONS = 6


class FlatEncoder(StateEncoder):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Linear(4, 8)

    @property
    def out_dim(self) -> int:
        return 8

    def forward(self, obs):
        return self.net(obs["x"])


def make_obs(n: int, gen: torch.Generator) -> dict[str, torch.Tensor]:
    mask = torch.rand(n, N_ACTIONS, generator=gen) > 0.5
    mask[torch.arange(n), torch.randint(0, N_ACTIONS, (n,), generator=gen)] = True
    return {"x": torch.randn(n, 4, generator=gen), "action_mask": mask}


def test_masked_actions_never_sampled_in_rollout_or_update():
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    obs_space = spaces.Dict(
        {"x": spaces.Box(-10, 10, (4,)), "action_mask": spaces.MultiBinary(N_ACTIONS)}
    )
    policy = Policy(FlatEncoder(), N_ACTIONS, hidden_sizes=(16,))
    cfg = PPOConfig(num_steps=32, num_minibatches=2, update_epochs=2)
    ppo = PPO(policy, cfg)
    buf = RolloutBuffer(obs_space, cfg.num_steps, num_envs=4, device=torch.device("cpu"))

    for step in range(cfg.num_steps):
        obs = make_obs(4, gen)
        with torch.no_grad():
            action, logp, value = policy.act(obs)
        assert obs["action_mask"].gather(1, action[:, None]).all()
        reward = torch.ones(4)
        buf.add(step, obs, action, logp, reward, torch.zeros(4), value)

    stored_mask = buf.obs["action_mask"].reshape(-1, N_ACTIONS)
    assert stored_mask.gather(1, buf.actions.reshape(-1, 1)).all()

    stats = ppo.update(buf, last_value=torch.zeros(4))
    assert all(torch.isfinite(torch.tensor(v)) for v in stats.values())

    # After the update, masked actions still have exactly zero probability on stored obs.
    flat_obs = {k: v.reshape((-1,) + v.shape[2:]) for k, v in buf.obs.items()}
    with torch.no_grad():
        probs = policy.distribution(flat_obs).probs
        logp, _, _ = policy.evaluate_actions(flat_obs, buf.actions.reshape(-1))
    assert torch.all(probs[~stored_mask] == 0)
    assert torch.isfinite(logp).all()
