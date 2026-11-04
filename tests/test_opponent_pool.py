import numpy as np

from apps.pokemon_stub import APP
from core.interfaces.env_api import ACTION_MASK_KEY
from core.selfplay.opponent_pool import (
    Opponent,
    OpponentPool,
    ScriptedOpponent,
    random_legal_action,
)
from core.train import build_policy


def test_scripted_and_snapshot_opponents_share_interface_and_respect_mask():
    pool = OpponentPool([ScriptedOpponent("random", random_legal_action)], max_snapshots=2)
    env = APP.make_env({}, pool)
    policy = build_policy({}, APP, env)
    for i in range(3):
        pool.add_snapshot(policy, f"snap_{i}")
    assert [o.name for o in pool.opponents] == ["random", "snap_1", "snap_2"]  # oldest evicted

    rng = np.random.default_rng(0)
    obs, _ = env.reset(seed=0)
    for opponent in pool.opponents:
        assert isinstance(opponent, Opponent)
        for _ in range(50):
            assert obs[ACTION_MASK_KEY][opponent.act(obs, rng)] == 1


def test_snapshot_is_frozen_copy():
    pool = OpponentPool([ScriptedOpponent("random", random_legal_action)])
    env = APP.make_env({}, pool)
    policy = build_policy({}, APP, env)
    pool.add_snapshot(policy, "snap")
    for p in policy.parameters():
        p.data.add_(1.0)
    snap = pool.snapshots[0].policy
    assert all(not p.requires_grad for p in snap.parameters())
    assert not any(
        np.allclose(a.detach(), b.detach())
        for a, b in zip(policy.parameters(), snap.parameters(), strict=True)
    )
