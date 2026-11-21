import numpy as np

from cogrid.backend import xp
from cogrid.backend.array_ops import set_at_2d
from cogrid.core.component_registry import register_feature_type
from cogrid.core.features import Feature
from cogrid.core.objects.registry import get_object_names

from gym_cooking.utils.config import CHOPPING_NUM_STEPS
from gym_cooking.utils.core import (
    ASSEMBLE_COOKING_FOOD, CHARRED_FOOD, CHOPPED_FOOD, CHOPPING_FOOD, COOKED_FOOD, COOKING_FOOD, FRESH_FOOD,
    GRIDSQUARES,
)

from env.mechanics import COOKING_STEPS, FPS
from env.objects import ALL_OBJECTS, GRIDSQUARE_NAMES, SCOPE, make_hla_object

ALL_ENTITIES = FRESH_FOOD + CHOPPING_FOOD + CHOPPED_FOOD + COOKING_FOOD + COOKED_FOOD + CHARRED_FOOD + ['Plate',
                                                                                                        'Fire',
                                                                                                        'FireExtinguisher']
ALL_FOOD_ENTITIES = FRESH_FOOD + CHOPPED_FOOD + \
    COOKED_FOOD + CHARRED_FOOD + ['Plate']


@register_feature_type("hla_state", scope=SCOPE)
class HLAState(Feature):
    per_agent = False
    obs_dim = 0

    @classmethod
    def compute_obs_dim(cls, scope, env_config=None):
        n_agents = env_config["num_agents"]
        n_channels = len(GRIDSQUARES) + len(ALL_ENTITIES) + 2 + n_agents
        return (n_channels * env_config["grid_height"] * env_config["grid_width"]
                + env_config["hla"]["max_num_orders"] * (len(ALL_FOOD_ENTITIES) + 1)
                + n_agents * len(ALL_FOOD_ENTITIES))

    @classmethod
    def build_feature_fn(cls, scope, env_config=None):
        names = get_object_names(scope=SCOPE)
        n = len(names)
        idx = {name: i for i, name in enumerate(names) if name is not None}
        gridsquare_np = np.zeros((n, len(GRIDSQUARES)), dtype=np.float32)
        gridsquare_np[0, GRIDSQUARES.index('Floor')] = 1
        gridsquare_np[idx['free_space'], GRIDSQUARES.index('Floor')] = 1
        gridsquare_np[idx['counter'], GRIDSQUARES.index('Counter')] = 1
        for name in GRIDSQUARE_NAMES.values():
            gridsquare_np[idx[name], GRIDSQUARES.index(name)] = 1
        contents_np = np.zeros((n + 1, len(ALL_ENTITIES)), dtype=np.float32)
        holding_np = np.zeros((n + 1, len(ALL_FOOD_ENTITIES)), dtype=np.float32)
        for name in ALL_OBJECTS:
            for c in make_hla_object(name, None).contents:
                contents_np[idx[name] + 1, ALL_ENTITIES.index(c.full_name)] = 1
            holding_np[idx[name] + 1] = [int(e in name) for e in ALL_FOOD_ENTITIES]
        order_np = np.array([[int(e in goal) for e in ALL_FOOD_ENTITIES] for goal in env_config["hla"]["goals"]],
                            dtype=np.float32)
        time_limit_np = np.asarray(env_config["hla"]["lengths"], dtype=np.float32) * FPS
        is_cooking_np = np.isin(np.arange(n), [idx[c] for c in ASSEMBLE_COOKING_FOOD])
        is_chopping_np = np.isin(np.arange(n), [idx[c] for c in CHOPPING_FOOD])

        def fn(state):
            otm, osm = state.object_type_map, state.object_state_map
            H, W = otm.shape
            gridsquare_map = xp.asarray(gridsquare_np)[otm].transpose(2, 0, 1)

            agent_map = ((xp.arange(H)[None, :, None] == state.agent_pos[:, 0, None, None])
                         & (xp.arange(W)[None, None, :] == state.agent_pos[:, 1, None, None])).astype(xp.float32)
            held = xp.asarray(contents_np)[state.agent_inv[:, 0] + 1]
            obj_map = xp.asarray(contents_np)[osm + 1].transpose(2, 0, 1)
            obj_map = xp.maximum(obj_map, xp.max(agent_map[:, None, :, :] * held[:, :, None, None], axis=0))

            cooking_rest_time = xp.zeros((H, W), dtype=xp.float32)
            for p in range(state.pot_positions.shape[0]):
                r, c = state.pot_positions[p, 0], state.pot_positions[p, 1]
                rest = (COOKING_STEPS - (state.current_time - state.pot_start[p])) / COOKING_STEPS
                cooking_rest_time = set_at_2d(cooking_rest_time, r, c,
                                              xp.where(xp.asarray(is_cooking_np)[osm[r, c]], rest, 0.0))
            chopping_num_steps = xp.zeros((H, W), dtype=xp.float32)
            for b in range(state.board_positions.shape[0]):
                r, c = state.board_positions[b, 0], state.board_positions[b, 1]
                rest = state.board_rest[b] / CHOPPING_NUM_STEPS
                chopping_num_steps = set_at_2d(chopping_num_steps, r, c,
                                               xp.where(xp.asarray(is_chopping_np)[osm[r, c]], rest, 0.0))

            result = xp.concatenate([gridsquare_map, obj_map, cooking_rest_time[None], chopping_num_steps[None],
                                     agent_map]).transpose(0, 2, 1)

            active = state.order_recipe != -1
            r = xp.maximum(state.order_recipe, 0)
            order_onehot = xp.asarray(order_np)[r] * active[:, None]
            t = xp.where(active, state.order_timer / xp.asarray(time_limit_np)[r], 0.0)
            current_orders = xp.concatenate([order_onehot, t[:, None]], axis=1)

            current_holdings = xp.asarray(holding_np)[state.agent_inv[:, 0] + 1]

            return xp.concatenate([result.ravel(), current_orders.ravel(), current_holdings.ravel()]).astype(xp.float32)

        return fn
