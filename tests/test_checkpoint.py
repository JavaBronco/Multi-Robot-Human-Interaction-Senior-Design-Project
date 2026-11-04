import torch

from apps.robot_grid import APP
from core.agent.policy import obs_to_tensor
from core.algo.ppo import PPO, PPOConfig
from core.run_logger import load_checkpoint, save_checkpoint
from core.train import build_policy


def test_checkpoint_round_trip(tmp_path):
    torch.manual_seed(0)
    cfg = {"policy": {"hidden_sizes": [32]}, "encoder": {"channels": 8}}
    env = APP.make_env({}, None)
    obs, _ = env.reset(seed=0)
    batch = obs_to_tensor({k: v[None] for k, v in obs.items()}, torch.device("cpu"))

    policy = build_policy(cfg, APP, env)
    ppo = PPO(policy, PPOConfig())
    loss = sum(o.sum() for o in policy(batch))  # take one step so optimiser has state
    loss.backward()
    ppo.optimizer.step()
    save_checkpoint(tmp_path / "ck.pt", policy, ppo.optimizer, global_step=123, cfg=cfg)

    torch.manual_seed(1)
    restored = build_policy(cfg, APP, env)
    restored_ppo = PPO(restored, PPOConfig())
    ckpt = load_checkpoint(tmp_path / "ck.pt", restored, restored_ppo.optimizer)

    assert ckpt["global_step"] == 123 and ckpt["config"] == cfg
    for a, b in zip(policy(batch), restored(batch), strict=True):
        assert torch.equal(a, b)
    state_a = ppo.optimizer.state_dict()["state"]
    state_b = restored_ppo.optimizer.state_dict()["state"]
    assert all(torch.equal(state_a[k]["exp_avg"], state_b[k]["exp_avg"]) for k in state_a)
