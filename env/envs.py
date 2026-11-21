import copy
import functools

import numpy as np

from cogrid.cogrid_env import CoGridEnv
from cogrid.core.grid import layouts
from cogrid.envs import registry
from cogrid.envs.overcooked.agent import OvercookedAgent

from gym_cooking.envs.overcooked_environment import MapSetting, OvercookedEnvironment
from gym_cooking.play_test import MAP_SETTINGS
from gym_cooking.utils.core import Floor, GridSquare, Object, Rep

from env.mechanics import FPS, HLATables, build_extra_state_init, build_interactions, build_tick
from env.objects import SCOPE
from env.rewards import OrderSchedulerReward


def load_level(name):
    env = OvercookedEnvironment(MapSetting(**MAP_SETTINGS[name]))
    env.reset()
    return env


def register_level(env, tables):
    world = env.world
    rows = [[' '] * world.width for _ in range(world.height)]
    state_encoding = np.zeros((world.height, world.width), dtype=np.int32)
    for gs in world.get_object_list():
        if isinstance(gs, GridSquare) and not isinstance(gs, Floor):
            x, y = gs.location
            rows[y][x] = 'C' if gs.rep == Rep.COUNTER else gs.rep
            if isinstance(gs.holding, Object):
                state_encoding[y, x] = tables.idx[gs.holding.full_name]
    for agent in env.sim_agents:
        x, y = agent.location
        rows[y][x] = '+'
    layout_id = f"hla_{env.arglist.level}"
    layouts.register_layout(layout_id, [''.join(r) for r in rows], state_encoding)
    return layout_id


def _make_hla_env(config, **kwargs):
    return CoGridEnv(config=config, agent_class=OvercookedAgent, **kwargs)


_hla_base_config = {
    "name": "hla",
    "num_agents": 2,
    "n_agents": 2,
    "action_set": "cardinal_actions",
    "scope": SCOPE,
    "features": ["hla_state"],
    "rewards": [OrderSchedulerReward(coefficient=1.0)],
}

_tables = HLATables()
ENV_IDS = []

for _name in MAP_SETTINGS:
    _env = load_level(_name)
    _config = copy.deepcopy(_hla_base_config)
    _config["grid"] = {"layout": register_level(_env, _tables)}
    _config["grid_height"] = _env.world.height
    _config["grid_width"] = _env.world.width
    _config["max_steps"] = _env.arglist.max_num_timesteps * FPS
    _config["hla"] = {
        "level": _env.arglist.level,
        "goals": [r.final_task.full_name for r in _env.recipes],
        "lengths": [r.length for r in _env.recipes],
        "bonuses": [r.bonus for r in _env.recipes],
        "max_num_orders": _env.arglist.max_num_orders,
    }
    _config["interactions"] = build_interactions(_tables)
    _config["tick_fn"] = build_tick(_tables, _config["hla"]["goals"], _config["hla"]["lengths"], _config["hla"]["bonuses"])
    _config["extra_state_init_fn"] = build_extra_state_init(_tables, _env.arglist.max_num_orders, _config["num_agents"])

    _env_id = f"HLAKitchen-{_name.title()}-V0"
    ENV_IDS.append(_env_id)
    registry.register(
        _env_id,
        functools.partial(_make_hla_env, config=_config),
    )
