"""Training entrypoint for every app: ``python -m core.train --config configs/<name>.yaml``."""

import argparse
import time
from collections import deque
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector import AutoresetMode, SyncVectorEnv

from core.agent.policy import Policy, obs_to_tensor
from core.algo.buffer import RolloutBuffer
from core.algo.ppo import PPO, PPOConfig
from core.registry import AppSpec, load_app
from core.run_logger import RunLogger, save_checkpoint
from core.selfplay.opponent_pool import OpponentPool
from core.utils.config import load_config
from core.utils.device import get_device
from core.utils.seeding import set_global_seed


def build_policy(cfg: dict[str, Any], app: AppSpec, env: gym.Env) -> Policy:
    """Encoder from the app + shared trunk/heads sized to the env's action space."""
    encoder = app.make_encoder(env.observation_space, cfg.get("encoder", {}))
    return Policy(encoder, int(env.action_space.n), **cfg.get("policy", {}))


def make_pool(app: AppSpec, cfg: dict[str, Any], split: str) -> OpponentPool | None:
    """Opponent pool for ``split`` ('train' or 'eval'); None for single-agent apps."""
    factory = app.train_opponents if split == "train" else app.eval_opponents
    if factory is None:
        return None
    return OpponentPool(factory(), cfg.get("selfplay", {}).get("max_snapshots", 10))


def stack_obs(obs_list: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """List of single observations -> batched observation dict."""
    return {k: np.stack([o[k] for o in obs_list]) for k in obs_list[0]}


def train(cfg: dict[str, Any]) -> Path:
    """Run masked PPO as configured. Returns the run directory."""
    set_global_seed(cfg["seed"])
    device = get_device(cfg.get("device", "auto"))
    app = load_app(cfg["app"])
    pool = make_pool(app, cfg, "train")
    num_envs = cfg["num_envs"]
    envs = SyncVectorEnv(
        [lambda: app.make_env(cfg.get("env", {}), pool) for _ in range(num_envs)],
        autoreset_mode=AutoresetMode.SAME_STEP,
    )
    single_env = envs.envs[0]
    policy = build_policy(cfg, app, single_env).to(device)
    ppo_cfg = PPOConfig(**cfg.get("ppo", {}))
    ppo = PPO(policy, ppo_cfg)
    buffer = RolloutBuffer(single_env.observation_space, ppo_cfg.num_steps, num_envs, device)
    logger = RunLogger(cfg)
    ckpt_dir = logger.run_dir / "checkpoints"
    snapshot_every = cfg.get("selfplay", {}).get("snapshot_every", 0)

    num_updates = cfg["total_timesteps"] // (ppo_cfg.num_steps * num_envs)
    ep_returns, ep_lengths = np.zeros(num_envs), np.zeros(num_envs, dtype=int)
    recent: deque[tuple[float, int, bool]] = deque(maxlen=100)
    global_step, start = 0, time.time()
    obs_np, _ = envs.reset(seed=cfg["seed"])
    print(f"run dir: {logger.run_dir}  device: {device}  updates: {num_updates}")

    for update in range(1, num_updates + 1):
        ppo.set_progress((update - 1) / num_updates)
        for step in range(ppo_cfg.num_steps):
            obs = obs_to_tensor(obs_np, device)
            with torch.no_grad():
                action, logp, value = policy.act(obs)
            obs_np, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
            global_step += num_envs
            done = terminated | truncated
            ep_returns += reward
            ep_lengths += 1

            # Truncation is not failure: bootstrap from the value of the final observation.
            train_reward = reward.astype(np.float32)
            trunc_idx = np.flatnonzero(truncated & ~terminated)
            if len(trunc_idx):
                final = stack_obs([info["final_obs"][i] for i in trunc_idx])
                with torch.no_grad():
                    _, final_value = policy(obs_to_tensor(final, device))
                train_reward[trunc_idx] += ppo_cfg.gamma * final_value.cpu().numpy()

            buffer.add(
                step,
                obs,
                action,
                logp,
                torch.as_tensor(train_reward, device=device),
                torch.as_tensor(done, dtype=torch.float32, device=device),
                value,
            )
            for i in np.flatnonzero(done):
                success = bool(info["final_info"]["is_success"][i])
                recent.append((ep_returns[i], ep_lengths[i], success))
                ep_returns[i], ep_lengths[i] = 0.0, 0

        with torch.no_grad():
            _, last_value = policy(obs_to_tensor(obs_np, device))
        stats = ppo.update(buffer, last_value)
        stats["charts/sps"] = global_step / (time.time() - start)
        if recent:
            rets, lens, succ = zip(*recent, strict=True)
            stats |= {
                "charts/episode_return": float(np.mean(rets)),
                "charts/episode_length": float(np.mean(lens)),
                "charts/success_rate": float(np.mean(succ)),
            }
        logger.log(stats, global_step)

        if update % cfg.get("log_interval", 10) == 0 or update == num_updates:
            print(
                f"update {update}/{num_updates}  step {global_step}  "
                f"return {stats.get('charts/episode_return', float('nan')):.2f}  "
                f"success {stats.get('charts/success_rate', float('nan')):.2f}  "
                f"sps {stats['charts/sps']:.0f}"
            )
        if update % cfg.get("checkpoint_every", 50) == 0:
            save_checkpoint(ckpt_dir / "latest.pt", policy, ppo.optimizer, global_step, cfg)
        if pool is not None and snapshot_every and update % snapshot_every == 0:
            pool.add_snapshot(policy, name=f"snapshot_{global_step}")

    save_checkpoint(ckpt_dir / "final.pt", policy, ppo.optimizer, global_step, cfg)
    envs.close()
    logger.close()
    print(f"saved {ckpt_dir / 'final.pt'}")
    return logger.run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to a YAML config in configs/")
    parser.add_argument("--set", nargs="*", default=[], help="overrides, e.g. ppo.ent_coef=0.0")
    args = parser.parse_args()
    train(load_config(args.config, args.set))


if __name__ == "__main__":
    main()
