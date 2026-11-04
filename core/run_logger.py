"""Run directories, TensorBoard (+ optional W&B) logging, and checkpoint save/load."""

import time
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

RUNS_ROOT = Path("runs")


class RunLogger:
    """Creates ``runs/<run_name>_<timestamp>/`` with config.yaml, ``tb/`` and ``checkpoints/``."""

    def __init__(self, cfg: dict[str, Any], root: Path = RUNS_ROOT) -> None:
        self.run_dir = root / f"{cfg['run_name']}_{time.strftime('%Y%m%d-%H%M%S')}"
        (self.run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        with open(self.run_dir / "config.yaml", "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        self.wandb_run = None
        if cfg.get("track_wandb", False):
            import wandb  # optional dependency: pip install -e ".[wandb]"

            self.wandb_run = wandb.init(
                project=cfg.get("wandb_project", "mrhi"),
                name=self.run_dir.name,
                config=cfg,
                dir=str(self.run_dir),
                sync_tensorboard=True,
            )
        self.writer = SummaryWriter(str(self.run_dir / "tb"))

    def log(self, scalars: dict[str, float], step: int) -> None:
        """Write scalars at global step ``step``."""
        for key, value in scalars.items():
            self.writer.add_scalar(key, value, step)

    def close(self) -> None:
        self.writer.close()
        if self.wandb_run is not None:
            self.wandb_run.finish()


def save_checkpoint(
    path: Path,
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    global_step: int,
    cfg: dict[str, Any],
) -> None:
    """Save weights, optimiser state, step and the config needed to rebuild the policy."""
    torch.save(
        {
            "policy": policy.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "global_step": global_step,
            "config": cfg,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    policy: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint dict; restore into ``policy``/``optimizer`` when given."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if policy is not None:
        policy.load_state_dict(ckpt["policy"])
    if optimizer is not None and ckpt["optimizer"] is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt


def latest_run(root: Path = RUNS_ROOT) -> Path:
    """Most recently modified run directory that has a final checkpoint."""
    runs = [p for p in root.glob("*/checkpoints/final.pt")]
    if not runs:
        raise FileNotFoundError(f"no finished runs under {root}/; train one first")
    return max(runs, key=lambda p: p.stat().st_mtime).parent.parent
