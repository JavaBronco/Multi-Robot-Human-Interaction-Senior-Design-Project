# Multi-Robot / Human Interaction — shared RL core

One reinforcement-learning decision-maker (masked PPO) that drives several apps:

- **`apps/robot_grid`**: robot navigation, Phase 1. A seeded procedural 2D grid with energy and health budgets.
- **`apps/pokemon_stub`**: Pokémon battle (Gen 3) **placeholder**. It shows the interfaces only; the battle mechanics are TODOs.

The training, evaluation, logging and self-play code in `core/` is app-agnostic. It is identical for both apps.

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # add ,wandb for Weights & Biases
pytest                           # run the tests
ruff check . && ruff format .    # lint and format
```

## Usage

```bash
python -m core.train --config configs/robot_grid_phase1.yaml        # about 2 min on CPU
python -m core.train --config configs/pokemon_stub.yaml             # about 10 s smoke run
python -m core.train --config configs/robot_grid_phase1.yaml --set seed=2 ppo.ent_coef=0.02

python -m core.evaluate                         # latest run vs random baseline
python -m core.evaluate --run runs/<dir> --episodes 1000

tensorboard --logdir runs
```

Each run writes `runs/<run_name>_<timestamp>/` with these contents:
- `config.yaml`: a snapshot of the config the run used.
- `tb/`: TensorBoard logs.
- `checkpoints/latest.pt` and `checkpoints/final.pt`: saved weights. Each checkpoint embeds the config, so evaluation can rebuild the policy from it.

To also log to Weights & Biases, set `track_wandb: true` in the config.

### Results: robot_grid Phase 1 (8x8, seed 1, 300k steps, ~2 min CPU)

These are from 500 held-out maps (scenario seeds 10⁹ … 10⁹+499). No training run ever generates these maps.

| agent   | success rate  | episode length | return         |
|---------|---------------|----------------|----------------|
| trained | 0.998 ± 0.045 | 6.03 ± 1.36    | 9.71 ± 0.47    |
| random  | 0.056 ± 0.230 | 18.75 ± 2.81   | -0.15 ± 2.35   |

"random" picks uniformly among *legal* actions. Values are mean ± std over episodes.

## Layout

```
core/
  interfaces/   StateEncoder, RuleFilter, env contract + check_env_contract()
  agent/        MaskedCategorical, Policy (encoder -> shared trunk -> action head + value head)
  algo/         RolloutBuffer (stores masks), masked PPO + GAE
  selfplay/     Opponent / ScriptedOpponent / SnapshotOpponent / OpponentPool,
                OpponentEnv (2-player PettingZoo -> single-agent Gymnasium)
  registry.py   AppSpec: what an app hands to core/
  train.py      python -m core.train --config ...
  evaluate.py   python -m core.evaluate
  run_logger.py run dirs, TensorBoard/W&B, checkpoints
  utils/        seeding, device, config loading
apps/
  robot_grid/   env, rules (RuleFilter), encoder (CNN + MLP)
  pokemon_stub/ placeholder PettingZoo env, rules, encoder, scripted opponents
configs/        one YAML per experiment
tests/
```

## The env contract (`core/interfaces/env_api.py`)

Every app env, once wrapped, is a `gymnasium.Env` that meets these rules:

- **Observation:** `observation_space` is a `spaces.Dict`. It contains `"action_mask"` (a `MultiBinary(n)`) alongside whatever keys your encoder reads.
- **Actions:** `action_space` is `Discrete(n)`.
- **Mask:** every mask has at least one legal action. The env fills the mask by calling the app's `RuleFilter`.
- **End of episode:** the final step's `info["is_success"]` is a bool: goal reached, or battle won.
- **Scenario seeds:** `reset()` draws a scenario seed from `TRAIN_SEED_RANGE` unless `options["scenario_seed"]` is given. It reports the seed it used in `info["scenario_seed"]`.
  - Evaluation always passes seeds of `EVAL_SEED_START` (10⁹) or higher, so test maps and battles never appear in training.

Run `check_env_contract(env)` in your tests.

**Why the mask is in the observation, not `info`:** the original spec put it in `info["action_mask"]`. Inside the observation, it travels with its observation through vectorized envs and auto-resets, so the stored mask always matches the observation the action was sampled from. The mask is stored in the rollout buffer and reapplied during the PPO update, so masked actions have zero probability in both places (`tests/test_mask_flow.py`).

## Adding an app

Nothing in `core/` changes. Write four things:

1. **Env.** Write a `gymnasium.Env` that satisfies the contract above.
   - For two-player or self-play apps, write a PettingZoo `ParallelEnv` with two agents and wrap it with `core.selfplay.OpponentEnv`.
2. **Rule filter.** Subclass `core.interfaces.RuleFilter` and implement `mask(state) -> int8 array`. The env calls it to build `obs["action_mask"]`.
3. **Encoder.** Subclass `core.interfaces.StateEncoder`: implement `out_dim` and `forward(obs_dict) -> (B, out_dim)`, ignoring `"action_mask"`.
4. **Config.** Write `configs/<name>.yaml` with `app: apps.<name>` and the env, encoder, policy and ppo sections.

Then expose an `AppSpec` in `apps/<name>/__init__.py`:

```python
from core.registry import AppSpec

APP = AppSpec(
    name="my_app",
    make_env=lambda env_cfg, pool: MyEnv(**env_cfg),  # pool is None unless opponents given
    make_encoder=lambda obs_space, enc_cfg: MyEncoder(obs_space, **enc_cfg),
    train_opponents=None,  # or: lambda: [ScriptedOpponent("npc", fn), ...]
    eval_opponents=None,  # held-out opponents, used only by core.evaluate
)
```

Finally, add a test that runs `check_env_contract`, plus `gymnasium.utils.env_checker.check_env` (and `pettingzoo.test.parallel_api_test` if the app uses PettingZoo).

## Opponents and self-play

`Opponent.act(obs, rng) -> action` is the single interface for both kinds of opponent:

- **`ScriptedOpponent`** wraps a fixed algorithm, such as a Gen 3 NPC trainer script.
- **`SnapshotOpponent`** wraps a frozen copy of a past policy.

`OpponentEnv` samples one opponent from an `OpponentPool` at every reset. When `selfplay.snapshot_every: N` is set, `core.train` adds a snapshot of the current policy to the pool every N updates. The pool keeps the most recent `max_snapshots`.

## Design notes

- **Separate weights per app.** "Shared trunk" means shared architecture and code; each app trains its own weights. Joint multi-app training is not implemented.
- **Truncation.** A time-limit truncation is not treated as failure: the rollout bootstraps from the value of the final observation.
- **Robot grid rules.** The rules are a first draft; all costs are in the config.

  | action | legal when | energy | damage |
  |---|---|---|---|
  | walk | target cell is not big_rock | 1 (sand: 2) | sand: 1 |
  | crawl | target cell is not big_rock | 2 | 0 |
  | climb | target cell is big_rock | 4 | 0 |
  | jump (2 cells) | landing cell is not big_rock | 3 | landing on bedrock: 1 |

  No action may leave the grid. Episodes end at the goal (+10), when energy or health runs out, or at the step limit. Illegal actions only happen if the mask is ignored; they are no-ops that still cost a step.

- **Pokémon stub.** The stub has no real mechanics: moves deal fixed placeholder damage, and the held-out "strongest_move" trainer can't be beaten by more than a coin flip. The `TODO(pokemon team)` comments mark where to plug in the Gen 3 simulator (for example Showdown via poke-env), the real legality rules, the observation and encoder, and the NPC trainer scripts.
