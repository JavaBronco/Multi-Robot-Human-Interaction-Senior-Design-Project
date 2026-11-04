"""Evaluate a trained agent against a random baseline on held-out scenarios.

``python -m core.evaluate [--run runs/<dir>] [--episodes N]``

Scenario seeds start at ``EVAL_SEED_START`` (never used in training), and multi-agent apps
face their held-out ``eval_opponents``. Both agents see the same seeds and opponents.
"""

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from core.agent.policy import Policy, obs_to_tensor
from core.interfaces.env_api import EVAL_SEED_START, SCENARIO_SEED_KEY, TRAIN_SEED_RANGE
from core.registry import load_app
from core.run_logger import latest_run, load_checkpoint
from core.selfplay.opponent_pool import random_legal_action
from core.train import build_policy, make_pool

Actor = Callable[[dict[str, np.ndarray], np.random.Generator], int]


def eval_seeds(n: int) -> list[int]:
    """Held-out scenario seeds, guaranteed disjoint from ``TRAIN_SEED_RANGE``."""
    seeds = list(range(EVAL_SEED_START, EVAL_SEED_START + n))
    assert not any(s in TRAIN_SEED_RANGE for s in seeds), "eval seeds overlap training seeds"
    return seeds


def policy_actor(policy: Policy, deterministic: bool = False) -> Actor:
    """Wrap a policy as an actor; samples with the numpy rng so runs are reproducible."""
    policy = policy.cpu().eval()

    def act(obs: dict[str, np.ndarray], rng: np.random.Generator) -> int:
        batch = obs_to_tensor({k: np.asarray(v)[None] for k, v in obs.items()}, torch.device("cpu"))
        with torch.no_grad():
            probs = policy.distribution(batch).probs[0].double().numpy()
        if deterministic:
            return int(probs.argmax())
        return int(rng.choice(len(probs), p=probs / probs.sum()))

    return act


def run_episodes(
    env: gym.Env, actor: Actor, seeds: list[int], seed: int = 0
) -> dict[str, np.ndarray]:
    """Play one episode per scenario seed. Returns per-episode success, length and return."""
    rng = np.random.default_rng(seed)
    success, length, ret = [], [], []
    for i, scenario in enumerate(seeds):
        obs, _ = env.reset(seed=seed if i == 0 else None, options={SCENARIO_SEED_KEY: scenario})
        done, steps, total = False, 0, 0.0
        while not done:
            obs, reward, terminated, truncated, info = env.step(actor(obs, rng))
            done, steps, total = terminated or truncated, steps + 1, total + reward
        success.append(info["is_success"])
        length.append(steps)
        ret.append(total)
    return {
        "success": np.array(success, float),
        "length": np.array(length),
        "return": np.array(ret),
    }


def evaluate(
    run_dir: Path, episodes: int | None = None, deterministic: bool = False, seed: int = 0
) -> dict[str, dict[str, np.ndarray]]:
    """Evaluate ``run_dir``'s final checkpoint and a random-legal-action baseline."""
    ckpt = load_checkpoint(run_dir / "checkpoints" / "final.pt")
    cfg: dict[str, Any] = ckpt["config"]
    app = load_app(cfg["app"])
    seeds = eval_seeds(episodes or cfg.get("eval", {}).get("episodes", 200))

    def make_env() -> gym.Env:
        return app.make_env(cfg.get("env", {}), make_pool(app, cfg, "eval"))

    env = make_env()
    policy = build_policy(cfg, app, env)
    policy.load_state_dict(ckpt["policy"])
    return {
        "trained": run_episodes(env, policy_actor(policy, deterministic), seeds, seed),
        "random": run_episodes(make_env(), random_legal_action, seeds, seed),
    }


def format_table(results: dict[str, dict[str, np.ndarray]]) -> str:
    """Markdown-style table of mean ± std per metric."""
    header = "| agent   | success/win rate | episode length  | return          |"
    lines = [header, "|---------|------------------|-----------------|-----------------|"]
    for name, r in results.items():
        cells = [f"{r[m].mean():.3f} ± {r[m].std():.3f}" for m in ("success", "length", "return")]
        lines.append(f"| {name:<7} | {cells[0]:<16} | {cells[1]:<15} | {cells[2]:<15} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=None, help="run dir (default: latest)")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--deterministic", action="store_true", help="argmax instead of sampling")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run_dir = args.run or latest_run()
    results = evaluate(run_dir, args.episodes, args.deterministic, args.seed)
    n = len(next(iter(results.values()))["success"])
    print(f"run: {run_dir}")
    print(f"{n} held-out episodes, scenario seeds {EVAL_SEED_START}..{EVAL_SEED_START + n - 1}\n")
    print(format_table(results))


if __name__ == "__main__":
    main()
