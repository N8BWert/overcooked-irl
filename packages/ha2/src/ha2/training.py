"""Two-player, shared-parameter PPO without a scripted partner."""

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback

from ha2.world import World


class SelfPlay(VecEnv):
    """Both VecEnv slots belong to the same game and supply PPO samples."""

    def __init__(
        self,
        stage,
        layout="kitchen",
        menu="onion",
        worker=None,
        seed=0,
        curriculum=False,
        feature_version=1,
        recipe_mask=False,
    ):
        self.world = World(
            layout, menu, feature_version=feature_version, recipe_mask=recipe_mask
        )
        self.stage, self.worker = stage, worker
        self.curriculum = curriculum
        self.rng = np.random.default_rng(seed)
        self.goals = np.zeros(2, dtype=int)
        self.ages = np.zeros(2, dtype=int)
        self.returns = np.zeros(2)
        self.task_counts = np.ones(len(self.world.tasks))
        self.render_mode = None
        self.successes = self.attempts = self.deliveries = 0
        self.reset_game()
        example = self.observe()
        actions = 7 if stage == "worker" else len(self.world.tasks)
        super().__init__(
            2,
            spaces.Box(-np.inf, np.inf, shape=example.shape[1:], dtype=np.float32),
            spaces.Discrete(actions),
        )

    def sample_task(self, agent):
        valid = self.world.mask(agent)
        valid[-1] = False
        weights = valid / np.sqrt(self.task_counts)
        if weights.sum() == 0:
            return self.world.wait
        task = int(self.rng.choice(len(weights), p=weights / weights.sum()))
        self.task_counts[task] += 1
        return task

    def reset_game(self):
        if self.stage == "worker":
            self.world.worker_start(self.rng)
            self.goals = np.array([self.sample_task(a) for a in range(2)])
        elif self.curriculum:
            self.world.worker_start(self.rng)
        else:
            self.world.reset(seed=int(self.rng.integers(2**31)))
        self.ages[:] = 0
        self.returns[:] = 0

    def observe(self):
        if self.stage == "worker":
            return np.stack([self.world.worker_obs(a, self.goals[a]) for a in range(2)])
        return np.stack([self.world.manager_obs(a) for a in range(2)])

    def reset(self):
        if hasattr(self, "_seeds") and self._seeds[0] is not None:
            self.rng = np.random.default_rng(self._seeds[0])
            self._reset_seeds()
        self.reset_game()
        return self.observe()

    def step_async(self, actions):
        self.actions = np.asarray(actions, dtype=int)

    def step_wait(self):
        w = self.world
        if self.stage == "manager":
            self.goals = self.actions.copy()
            inputs = np.stack([w.worker_obs(a, self.goals[a]) for a in range(2)])
            primitives = self.worker.predict(inputs, deterministic=False)[0]
            primitives[self.goals == w.wait] = 6
        else:
            primitives = self.actions
        previous = w.state
        goals = [w.targets(a, self.goals[a]) for a in range(2)]
        distances = [w.distance(a, goals[a]) for a in range(2)]
        potential = w.potential()
        _, raw, terms, truncs, _ = w.step(primitives)
        completed = np.array(
            [
                w.completed(a, self.goals[a], previous, goals[a], primitives[a])
                for a in range(2)
            ]
        )
        self.ages += 1
        game_done = any(terms.values()) or any(truncs.values())
        infos = [{}, {}]
        if self.stage == "worker":
            progress = np.array(
                [distances[a] - w.distance(a, goals[a]) for a in range(2)]
            )
            wrong = (primitives == 4) & ~completed
            invalid = np.array([not w.targets(a, self.goals[a]) for a in range(2)])
            done = completed | wrong | invalid | (self.ages >= 64)
            rewards = (
                0.3 * progress - 0.01 + completed.astype(float) - wrong.astype(float)
            )
            self.successes += int(completed.sum())
        else:
            # Potential-based shaping gives no net credit for pickup/drop cycles.
            reward = (
                float(raw[0])
                + 0.99 * (0 if any(terms.values()) else w.potential())
                - potential
                - 0.005
            )
            rewards = np.full(2, reward)
            done = np.full(2, game_done)
            self.deliveries += int(raw[0] > 0)
        self.returns += rewards
        terminal = self.observe()
        if (self.stage == "worker" or self.curriculum) and int(w.state.time) >= 128:
            game_done = True
            if self.stage == "manager":
                done[:] = True
                truncs = {0: True, 1: True}
        if game_done:
            done[:] = True
        if self.stage == "worker":
            self.attempts += int(done.sum())
        for a in range(2):
            if done[a]:
                infos[a] = {
                    "terminal_observation": terminal[a].copy(),
                    "episode": {"r": float(self.returns[a]), "l": int(self.ages[a])},
                }
                if self.stage == "worker":
                    infos[a].update(
                        skill_success=bool(completed[a]), task=int(self.goals[a])
                    )
                if self.stage == "manager" and truncs[a]:
                    infos[a]["TimeLimit.truncated"] = True
                if not game_done and self.stage == "worker":
                    self.goals[a] = self.sample_task(a)
                    self.ages[a] = 0
                    self.returns[a] = 0
        if game_done:
            self.reset_game()
        return self.observe(), rewards.astype(np.float32), done, infos

    def action_masks(self):
        return np.stack([self.world.mask(a) for a in range(2)])

    def close(self):
        self.world.env.close()

    def get_attr(self, name, indices=None):
        return [getattr(self, name) for _ in self._get_indices(indices)]

    def set_attr(self, name, value, indices=None):
        setattr(self, name, value)

    def env_method(self, name, *args, indices=None, **kwargs):
        if name == "action_masks":
            return [self.world.mask(a) for a in self._get_indices(indices)]
        return [
            getattr(self, name)(*args, **kwargs) for _ in self._get_indices(indices)
        ]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._get_indices(indices)]


class Progress(BaseCallback):
    def __init__(self, output, interval=20000):
        super().__init__()
        self.output, self.interval, self.last = output, interval, 0

    def _on_step(self):
        if self.num_timesteps - self.last >= self.interval:
            env = self.training_env
            print(
                f"steps={self.num_timesteps} worker_success={env.successes}/{env.attempts} "
                f"delivery_events={env.deliveries}",
                flush=True,
            )
            self.model.save(self.output)
            self.last = self.num_timesteps
        return True
