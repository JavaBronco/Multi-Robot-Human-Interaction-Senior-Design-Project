import numpy as np
from gymnasium.utils.env_checker import check_env
from pettingzoo.test import parallel_api_test

from apps.pokemon_stub import APP
from apps.pokemon_stub.env import PokemonStubParallelEnv
from apps.pokemon_stub.opponents import train_opponents
from apps.pokemon_stub.rules import N_MOVES, PokemonRuleFilter
from core.interfaces.env_api import check_env_contract
from core.selfplay.opponent_pool import OpponentPool


def make_env():
    return APP.make_env({}, OpponentPool(train_opponents()))


def test_pettingzoo_parallel_api():
    parallel_api_test(PokemonStubParallelEnv(), num_cycles=200)


def test_gymnasium_env_checker_on_wrapped_env():
    check_env(make_env(), skip_render_check=True)


def test_env_contract():
    check_env_contract(make_env(), n_steps=500)


def test_fainted_active_forces_switch_and_fainted_bench_is_masked():
    hp = np.array([0.0, 1.0, 0.0, 1.0, 1.0, 1.0])
    mask = PokemonRuleFilter().mask((hp, 0))
    assert not mask[:N_MOVES].any()  # active fainted: no moves
    assert mask[N_MOVES:].tolist() == [1, 0, 1, 1, 1]  # bench = mons 1..5
