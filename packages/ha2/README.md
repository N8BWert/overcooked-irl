# HA2 self-play

A compact CoGrid adaptation of the HA2 manager/worker structure. Both players
use the same learned policy, and transitions from both players train that policy.
There is no scripted teammate, FCP population, imitation dataset, or action fallback.

The first checkpoint targets **kitchen / onion**. It is not a reproduction of the
paper's original environment or an agent validated on every recipe and layout.

## Run the saved agent

From the repository root:

```powershell
uv sync --locked --all-packages
.\.venv\Scripts\ha2.exe evaluate --render --seed 1000 --episodes 3 --fps 5
```

Both players act automatically. The window shows their selected tasks and
deliveries. Close the window to stop. For headless evaluation, omit `--render`.

The default output directory is `checkpoints/ha2`. Keep these files together:

- `worker.zip`: PPO skill policy.
- `manager.zip`: MaskablePPO task policy.
- `config.json`: layout, menu, task order, feature version and worker checksum.

The verified default model bundle and its ten-game evaluation are included in Git.
Other checkpoints, screenshots, and training logs are ignored.
Changing the layout, observation schema, or task order requires a compatible model.
Changing the worker requires retraining or validating the manager.

## Train

Use a new output directory for a new experiment:

```powershell
.\.venv\Scripts\ha2.exe train --stage worker --steps 120000 --output checkpoints/new_run
.\.venv\Scripts\ha2.exe train --stage manager --curriculum --steps 160000 --output checkpoints/new_run
.\.venv\Scripts\ha2.exe train --stage manager --resume --steps 60000 --output checkpoints/new_run
.\.venv\Scripts\ha2.exe evaluate --worker-only --steps 10000 --seed 2000 --output checkpoints/new_run
.\.venv\Scripts\ha2.exe evaluate --episodes 10 --seed 1000 --output checkpoints/new_run
```

These are starting budgets for new runs, not a guarantee of the same trained weights.
The delivered models were trained while the adapter and curriculum were refined.

Step budgets count **agent transitions**: two transitions per joint game step.
Worker training uses sampled intermediate states and balanced feasible tasks.
Manager curriculum also uses intermediate states; the final training command
returns to the map's normal starts. Evaluation always uses normal starts,
except the explicit `--worker-only` skill evaluation.

Manager training freezes the worker. Each primitive step, both managers choose
tasks, then the worker produces both primitive actions. Both manager decisions
enter the PPO rollout. Small MLPs run on CPU; GPU support is not required.

## Design

- **Inputs:** explicit full-state features, not the original partial 875-vector.
  The manager receives inventories, positions, pot contents/timers, orders,
  task distances, and masks (150 numbers on kitchen). The worker receives
  goal-distance changes, adjacent occupancy, facing, and teammate position
  (16 numbers). Path-distance features use BFS; BFS does not select actions.
- **Orders:** all 20 V2 recipe IDs and three order timers have a stable encoding.
  The adapter also enables CoGrid's order observation without changing human play.
- **Tasks:** ingredient/plate retrieval, per-pot filling and soup retrieval,
  delivery, counter placement/retrieval, and waiting. These merge some original
  HA2 subtasks and explicitly identify individual pots.
- **Masks:** task targets must exist and be reachable. Plate retrieval is limited
  by full-pot demand and already-held plates. Waiting is a fallback when no
  productive task is available. These are explicit domain rules, not learned.
- **Worker reward:** goal-distance progress, a small step cost, success reward,
  and incorrect-interaction penalty. Practice states include randomized inventory,
  pot contents, positions, and orientation.
- **Manager reward:** the existing team reward, potential-based progress shaping,
  and a small step cost. The team reward is counted once, not summed twice.
  Reported evaluation return excludes shaping.
- **Inference:** the manager uses argmax by default and the worker samples its
  learned distribution. Sampling helps resolve corridor collisions. Each evaluation
  episode seeds the environment and policy sampling. `--deterministic-worker`
  is a diagnostic option and can deadlock. `--stochastic` also samples the manager.
- **Limits:** full-state features provide more information than the human-play
  environment's default policy observation. Mixed recipes are represented, but
  the delivered checkpoint is trained only on onion orders. No intent predictor
  or human-feedback component is implemented.

## Code map

| File | Responsibility |
| --- | --- |
| `world.py` | CoGrid setup, task targets/masks, features and practice states |
| `training.py` | Two-player vector interface, rewards and PPO progress reporting |
| `run.py` | Training, checkpoint loading, evaluation and Pygame viewer |

The training algorithm comes from Stable-Baselines3 and sb3-contrib.
The hierarchy follows [HA2](https://arxiv.org/abs/2505.04579);
the [official implementation](https://github.com/StephAO/HAHA) is a reference.
This package implements the CoGrid adapter independently rather than importing
the original `overcooked_ai_py` environment.

## Verified checkpoint

On kitchen with the onion menu, seeds 1000 through 1009 each ran 900 steps
from normal map starts. All ten games delivered soup, with 9?12 deliveries
per game (mean 10.7) and mean unshaped team return 138. Both players delivered
in every game. Expired orders still occur; this is a working baseline, not an
optimal policy. See [raw results](../../checkpoints/ha2/evaluation_1000.json).

Validation also covered order visibility for both managers, real delivery reward,
per-player interaction attribution, PPO updates, checkpoint reload, and the viewer.
No permanent test suite was added.
