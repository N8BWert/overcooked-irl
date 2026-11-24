"""CoGrid task definitions and explicit full-state policy features."""

from collections import deque
from copy import deepcopy
from dataclasses import replace

import numpy as np
from cogrid.envs import _make_v2_env, registry
from cogrid.core.objects.registry import object_to_idx
from cogrid.envs.overcooked.v2_objects import OPEN_POT_RECIPES
from layouts import register_layouts, read_menu_file

INGREDIENTS = ("onion", "tomato", "broccoli", "mushroom")
MOVES = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
DIRECTIONS = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]])
ACTION_DIR = (3, 1, 2, 0)


class World:
    def __init__(
        self,
        layout="kitchen",
        menu="onion",
        horizon=900,
        feature_version=1,
        recipe_mask=False,
    ):
        self.feature_version = feature_version
        self.recipe_mask = recipe_mask
        name = f"OvercookedV2-{layout}-{menu}"
        if name not in registry.ENVIRONMENT_REGISTRY:
            register_layouts(layout, menu)
        config = deepcopy(registry.ENVIRONMENT_REGISTRY[name].keywords["config"])
        config.update(
            features=["local_view", "order_observation"],
            recipes=list(OPEN_POT_RECIPES),
            max_steps=horizon,
        )
        self.env = _make_v2_env(config)
        self.env.reset(seed=0)
        names = (
            *INGREDIENTS,
            "plate",
            "counter",
            "open_pot",
            "open_delivery_zone",
            *(f"{s}_stack" for s in INGREDIENTS),
            "plate_stack",
            *(r.result for r in OPEN_POT_RECIPES),
        )
        self.ids = {s: object_to_idx(s, scope="overcooked") for s in names}
        self.recipes = list(OPEN_POT_RECIPES)
        self.recipe_ids = [self.ids[r.result] for r in self.recipes]
        menu_probs = read_menu_file(menu)
        self.training_recipes = [
            r for r in self.recipes if menu_probs.get(r.result, 0) > 0
        ]
        self.allowed = {ing for r in self.training_recipes for ing in r.ingredients}
        self.menu_ingredients = [
            [self.ids[x] for x in r.ingredients] for r in self.training_recipes
        ]
        self.pots = self.extra("open_pot_positions").copy()
        self.tasks = [f"get_{x}" for x in (*INGREDIENTS, "plate")]
        self.tasks += [f"fill_pot_{i}" for i in range(len(self.pots))]
        self.tasks += [f"get_soup_pot_{i}" for i in range(len(self.pots))]
        self.tasks += ["deliver", "put_on_counter", "get_from_counter", "wait"]
        self.wait = len(self.tasks) - 1
        self.distances = {}
        self.floor = self.state.object_type_map == 0

    @property
    def state(self):
        return self.env._env_state

    def extra(self, key):
        return self.state.extra_state["overcooked." + key]

    def reset(self, seed=None):
        self.env.reset(seed=seed)
        self.distances.clear()
        self.floor = self.state.object_type_map == 0

    def step(self, actions):
        result = self.env.step(dict(enumerate(map(int, actions))))
        floor = self.state.object_type_map == 0
        if not np.array_equal(self.floor, floor):
            self.floor = floor
            self.distances.clear()
        return result

    def locations(self, mask):
        return [tuple(p) for p in np.argwhere(mask)]

    def fits_menu(self, pot, ingredient):
        proposed = [int(x) for x in self.extra("open_pot_contents")[pot] if x >= 0]
        proposed.append(ingredient)
        return any(
            all(proposed.count(x) <= recipe.count(x) for x in set(proposed))
            for recipe in self.menu_ingredients
        )

    def targets(self, agent, task):
        s = self.state
        held = int(s.agent_inv[agent, 0])
        otm, osm = s.object_type_map, s.object_state_map
        counter = otm == self.ids["counter"]
        n = len(self.pots)
        if task < 5:
            item = (*INGREDIENTS, "plate")[task]
            if held != -1 or (item != "plate" and item not in self.allowed):
                return []
            if (
                self.recipe_mask
                and item != "plate"
                and not any(self.fits_menu(i, self.ids[item]) for i in range(n))
            ):
                return []
            return self.locations(
                (otm == self.ids[item + "_stack"])
                | (otm == self.ids[item])
                | (counter & (osm == self.ids[item]))
            )
        if task < 5 + n:
            i = task - 5
            if (
                held in [self.ids[x] for x in INGREDIENTS]
                and np.sum(self.extra("open_pot_contents")[i] >= 0) < 3
                and (not self.recipe_mask or self.fits_menu(i, held))
            ):
                return [tuple(self.pots[i])]
        elif task < 5 + 2 * n:
            i = task - 5 - n
            if (
                held == self.ids["plate"]
                and np.sum(self.extra("open_pot_contents")[i] >= 0) == 3
                and self.extra("open_pot_timer")[i] == 0
            ):
                return [tuple(self.pots[i])]
        elif task == self.wait - 3 and held in self.recipe_ids:
            return self.locations(otm == self.ids["open_delivery_zone"])
        elif task == self.wait - 2 and held != -1:
            return self.locations(counter & (osm == 0))
        elif task == self.wait - 1 and held == -1:
            return self.locations(counter & (osm != 0))
        return []

    def distance_map(self, targets, blocked=None):
        key = (tuple(sorted(targets)), blocked)
        if key not in self.distances:
            d = np.full(self.floor.shape, 100, dtype=np.float32)
            queue = deque()
            for target in targets:
                for move in MOVES:
                    p = tuple(np.asarray(target) + move)
                    if self.inside(p) and self.floor[p] and p != blocked and d[p] != 0:
                        d[p] = 0
                        queue.append(p)
            while queue:
                p = queue.popleft()
                for move in MOVES:
                    q = tuple(np.asarray(p) + move)
                    if (
                        self.inside(q)
                        and self.floor[q]
                        and q != blocked
                        and d[q] > d[p] + 1
                    ):
                        d[q] = d[p] + 1
                        queue.append(q)
            self.distances[key] = d
        return self.distances[key]

    def inside(self, p):
        return 0 <= p[0] < self.floor.shape[0] and 0 <= p[1] < self.floor.shape[1]

    def distance(self, agent, targets, pos=None, direction=None):
        if not targets:
            return 30.0
        pos = self.state.agent_pos[agent] if pos is None else pos
        direction = int(self.state.agent_dir[agent]) if direction is None else direction
        partner = tuple(self.state.agent_pos[1 - agent])
        d = self.distance_map(targets, blocked=partner)[tuple(pos)]
        if d >= 100:
            d = self.distance_map(targets)[tuple(pos)]
        facing = tuple(pos + DIRECTIONS[direction]) in targets
        return float(min(30, d + (d == 0 and not facing)))

    def mask(self, agent):
        valid = [
            bool(g := self.targets(agent, t)) and self.distance(agent, g) < 30
            for t in range(self.wait)
        ]
        # Reserve plates for full pots so both players cannot occupy their hands prematurely.
        full_pots = np.sum(np.sum(self.extra("open_pot_contents") >= 0, axis=1) == 3)
        held_plates = np.sum(self.state.agent_inv == self.ids["plate"])
        valid[4] = valid[4] and full_pots > held_plates
        # Waiting is a fallback while no productive task is available (e.g. soup is cooking).
        return np.array(valid + [not any(valid[: self.wait - 2])])

    def worker_obs(self, agent, task):
        goals = self.targets(agent, task)
        s = self.state
        pos, partner = s.agent_pos[agent], s.agent_pos[1 - agent]
        d = self.distance(agent, goals)
        deltas, blocked = [], []
        for action, move in enumerate(MOVES):
            q = pos + move
            obstacle = (
                not self.inside(q)
                or not self.floor[tuple(q)]
                or np.array_equal(q, partner)
            )
            q = pos if obstacle else q
            deltas.append((d - self.distance(agent, goals, q, ACTION_DIR[action])) / 2)
            blocked.append(float(obstacle))
        facing = tuple(pos + DIRECTIONS[int(s.agent_dir[agent])]) in goals
        return np.asarray(
            deltas
            + blocked
            + [float(facing), d / 30]
            + ((partner - pos) / 15).tolist()
            + np.eye(4)[s.agent_dir[agent]].tolist(),
            dtype=np.float32,
        )

    def inventory(self, held):
        categories = [-1, *(self.ids[x] for x in INGREDIENTS), self.ids["plate"]]
        if self.feature_version >= 2:
            categories += self.recipe_ids
            return np.eye(len(categories), dtype=np.float32)[
                categories.index(int(held))
            ]
        index = categories.index(int(held)) if held in categories else 6
        return np.eye(7, dtype=np.float32)[index]

    def manager_obs(self, agent):
        s = self.state
        order = [agent, 1 - agent]
        parts = [self.inventory(s.agent_inv[a, 0]) for a in order]
        parts += [s.agent_pos[order].ravel() / 15]
        for contents, timer in zip(
            self.extra("open_pot_contents"), self.extra("open_pot_timer")
        ):
            parts.append(
                np.array(
                    [np.sum(contents == self.ids[x]) / 3 for x in INGREDIENTS]
                    + [timer / 20]
                )
            )
        for recipe, timer in zip(self.extra("order_recipe"), self.extra("order_timer")):
            onehot = np.zeros(len(self.recipes))
            if recipe >= 0:
                onehot[int(recipe)] = 1
            parts += [
                onehot,
                np.array([timer / self.env.config["orders"]["time_limit"]]),
            ]
        for a in order:
            parts.append(
                np.array(
                    [
                        self.distance(a, self.targets(a, t)) / 30
                        for t in range(self.wait)
                    ]
                )
            )
        parts.append(self.mask(agent).astype(float))
        return np.concatenate(parts).astype(np.float32)

    def completed(self, agent, task, previous, goals, action):
        if action != 4:
            return False
        facing = tuple(
            previous.agent_pos[agent] + DIRECTIONS[previous.agent_dir[agent]]
        )
        before = int(previous.agent_inv[agent, 0])
        after = int(self.state.agent_inv[agent, 0])
        return facing in goals and before != after

    def potential(self):
        # Progress credit is removed when items leave a pot or inventory.
        contents = self.extra("open_pot_contents")
        held = self.state.agent_inv[:, 0]
        ingredients = [self.ids[x] for x in INGREDIENTS]
        return float(
            np.sum(contents >= 0)
            + 0.5 * np.isin(held, ingredients).sum()
            + 4 * np.isin(held, self.recipe_ids).sum()
        )

    def worker_start(self, rng):
        """Sample intermediate kitchen states for subtask and curriculum practice."""
        self.reset(seed=int(rng.integers(2**31)))
        ex = {k: v.copy() for k, v in self.state.extra_state.items()}
        if self.recipe_mask:
            recipes = [self.recipes.index(r) for r in self.training_recipes]
            slots = len(ex["overcooked.order_recipe"])
            ex["overcooked.order_recipe"][:] = rng.choice(
                recipes, size=slots, replace=len(recipes) < slots
            )
            ex["overcooked.order_timer"][:] = self.env.config["orders"]["time_limit"]
        ingredients = [self.ids[x] for x in INGREDIENTS if x in self.allowed]
        for i in range(len(self.pots)):
            count = int(rng.integers(4))
            recipe = self.training_recipes[
                int(rng.integers(len(self.training_recipes)))
            ]
            items = [self.ids[x] for x in recipe.ingredients]
            ex["overcooked.open_pot_contents"][i] = items[:count] + [-1] * (3 - count)
            ex["overcooked.open_pot_timer"][i] = 0 if count == 3 else 20
        choices = [
            -1,
            *ingredients,
            self.ids["plate"],
            *(self.ids[r.result] for r in self.training_recipes),
        ]
        inv = np.array([[rng.choice(choices)], [rng.choice(choices)]], dtype=np.int32)
        sources = self.locations(self.state.object_type_map == self.ids["plate_stack"])
        positions = np.argwhere(self.floor & (self.distance_map(sources) < 30))
        pos = positions[rng.choice(len(positions), 2, replace=False)]
        self.env._env_state = replace(
            self.state,
            agent_inv=inv,
            agent_pos=pos.astype(np.int32),
            agent_dir=rng.integers(4, size=2, dtype=np.int32),
            extra_state=ex,
        )
