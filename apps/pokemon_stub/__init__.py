"""Pokémon battle app (PLACEHOLDER stub). Registers ``APP`` for core/."""

from apps.pokemon_stub.encoder import TeamMLPEncoder
from apps.pokemon_stub.env import PokemonStubParallelEnv
from apps.pokemon_stub.opponents import eval_opponents, train_opponents
from core.registry import AppSpec
from core.selfplay.opponent_env import OpponentEnv

APP = AppSpec(
    name="pokemon_stub",
    make_env=lambda env_cfg, pool: OpponentEnv(PokemonStubParallelEnv(**env_cfg), pool),
    make_encoder=lambda obs_space, enc_cfg: TeamMLPEncoder(obs_space, **enc_cfg),
    train_opponents=train_opponents,
    eval_opponents=eval_opponents,
)
