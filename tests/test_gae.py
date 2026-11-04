import torch

from core.algo.ppo import compute_gae


def test_gae_matches_hand_computed_example():
    # gamma=0.9, lambda=0.8; episode ends after step 1, so step 1 must not bootstrap.
    rewards = torch.tensor([[1.0], [0.0], [2.0]])
    values = torch.tensor([[0.5], [1.0], [1.5]])
    dones = torch.tensor([[0.0], [1.0], [0.0]])
    last_value = torch.tensor([2.0])
    # t=2: delta = 2 + 0.9*2.0 - 1.5 = 2.3                    adv = 2.3
    # t=1: delta = 0 + 0 - 1.0 = -1.0 (done, no bootstrap)     adv = -1.0
    # t=0: delta = 1 + 0.9*1.0 - 0.5 = 1.4                    adv = 1.4 + 0.72*(-1.0) = 0.68
    adv, ret = compute_gae(rewards, values, dones, last_value, gamma=0.9, gae_lambda=0.8)
    assert torch.allclose(adv.squeeze(), torch.tensor([0.68, -1.0, 2.3]))
    assert torch.allclose(ret.squeeze(), torch.tensor([1.18, 0.0, 3.8]))


def test_gae_lambda_one_equals_discounted_return_minus_value():
    rewards = torch.tensor([[1.0], [1.0], [1.0]])
    values = torch.zeros(3, 1)
    dones = torch.tensor([[0.0], [0.0], [1.0]])
    adv, _ = compute_gae(rewards, values, dones, torch.tensor([5.0]), gamma=0.5, gae_lambda=1.0)
    assert torch.allclose(adv.squeeze(), torch.tensor([1.75, 1.5, 1.0]))
