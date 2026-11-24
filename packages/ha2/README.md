# HA2 self-play

CoGrid adaptation of [HA2](https://arxiv.org/abs/2505.04579).
Both players share a learned Manager and Worker and train through self-play.

## Run

From the repository root:

```powershell
uv sync --locked --all-packages
.\.venv\Scripts\ha2.exe evaluate --render --seed 1000 --episodes 3 --fps 5
```

Both players act automatically. Close the window to stop.
Omit `--render` for headless evaluation. Add `--stochastic` to sample Manager tasks;
the Worker already samples actions by default.

The original checkpoint uses the `kitchen` layout and `onion` menu.
Keep `worker.zip`, `manager.zip`, and `config.json` together in `checkpoints/ha2`.

## Train

Use a new output directory:

```powershell
.\.venv\Scripts\ha2.exe train --stage worker --steps 120000 --output checkpoints/new_run
.\.venv\Scripts\ha2.exe train --stage manager --curriculum --steps 160000 --output checkpoints/new_run
.\.venv\Scripts\ha2.exe train --stage manager --resume --steps 60000 --output checkpoints/new_run
.\.venv\Scripts\ha2.exe evaluate --episodes 10 --seed 1000 --output checkpoints/new_run
```

The Worker learns subtasks first and is frozen during Manager training.
`--curriculum` starts from intermediate practice states with sampled menu orders;
the final training stage and evaluation use normal starts and unchanged menu probabilities.
Step budgets count agent transitions, with two per game step. These budgets are starting points for new runs.

For `menu1` (onion, tomato, and broccoli-onion-tomato soups), add
`--menu menu1` to each training command and use a fresh output directory, e.g.
`--output checkpoints/menu1_new_run`. Evaluate that new model with:

```powershell
.\.venv\Scripts\ha2.exe evaluate --output checkpoints/menu1_new_run --render --episodes 3 --fps 5
```

Menu1 training remains experimental: the current policy does not reliably select
recipes from orders. Additional checkpoints and evaluation reports remain local.

## Key settings

- Global state features, including orders; BFS supplies task-distance features.
- PPO Worker and MaskablePPO Manager. Both players contribute training samples.
- Task masks control availability; new models only pick ingredients with a compatible
  pot and only fill pots toward recipes in their menu. This is an explicit domain rule,
  not a learned recipe constraint. Manager reward includes progress shaping.
- Manager selects a task every step; Worker samples a primitive action.
- No FCP. New models distinguish all soup types in inventory (feature version 2);
  the original onion checkpoint retains feature version 1.

## Files

- `world.py`: environment setup, tasks, observations, and practice states.
- `training.py`: shared-policy self-play, rewards, and training interface.
- `run.py`: training, evaluation, checkpoint loading, and viewer.
