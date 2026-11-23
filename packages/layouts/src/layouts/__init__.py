"""
A way to register our custom layouts with the overcooked cogrid registry.
"""

import os
import copy
import functools
import json

from cogrid.backend import xp
from cogrid.core.grid import layouts
from cogrid.core.pipeline.rewards import Reward, _compute_fwd_positions
from cogrid.envs import registry, _make_v2_env, _v2_base_config, _v2_shaping_rewards
from cogrid.envs.overcooked.config import (
    build_order_extra_state,
    build_order_hud_fn,
    build_order_tick
)
from cogrid.envs.overcooked import v2_objects
from cogrid.envs.overcooked.rewards import ExpiredOrderPenalty

# The directory where the layout files are stored.
LAYOUT_DIR = os.path.join(os.path.dirname(__file__), "../../../../assets/layouts")
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "../../../../assets")


class OrderDeliveryReward(Reward):
    """
    Added coefficient when a delivery consumes an active order (plus a tip
    for leftover time); -coefficient when a soup is delivered that matches no
    order.

    This is the same as OrderDeliveryReward, but recognises V2's open_delivery_zone
    ('X'). 
    """

    def __init__(
        self,
        *,
        order_time_limit: int = 300,
        tip_coefficient: float = 0.0,
        penalize_incorrect: bool = True,
        **kw
    ):
        super().__init__(**kw)
        self.order_time_limit = order_time_limit
        self.tip_coefficient = tip_coefficient
        self.penalize_incorrect = penalize_incorrect

    def compute(
        self,
        prev_state: dict,
        state: dict,
        actions: dict,
        reward_config: dict,
    ):
        type_ids, n_agents = reward_config["type_ids"], reward_config["n_agents"]
        coef = self.get_coefficient(state)

        # Did an agent drop a soup on a delivery zone this step?
        _, _, _, in_bounds, fwd_types = _compute_fwd_positions(prev_state)
        is_interact = actions == reward_config["action_pickup_drop_idx"]
        held = prev_state.agent_inv[:, 0]
        held_col = xp.where(held == -1, xp.int32(0), held  + 1)
        accepts = reward_config["static_tables"]["PLACE_ON_GUARD"][fwd_types, held_col] == 1
        faces_dz = (fwd_types == type_ids["delivery_zone"]) | (fwd_types == type_ids["open_delivery_zone"])
        any_delivery = xp.any(is_interact & (held != -1) & accepts & faces_dz & in_bounds)

        # Did a delivery consume an order slot
        consumed = (prev_state.order_recipe >= 0) & (state.order_recipe == -1)
        any_consumed = xp.any(consumed) & any_delivery

        total = xp.where(any_consumed, coef, xp.float32(0.0))
        if self.tip_coefficient:
            remaining = prev_state.order_timer[xp.argmax(consumed)].astype(xp.float32)
            total = total + xp.where(any_consumed, remaining / self.order_time_limit * self.tip_coefficient, xp.float32(0.0))
        if self.penalize_incorrect:
            total = total - xp.where(any_delivery & ~any_consumed, coef, xp.float32(0.0))
        return xp.full(n_agents, total, dtype=xp.float32)


def read_layout_file(layout_name: str) -> list[str]:
    """
    Read a layout file and return its content.

    Args:
        layout_name: The name of the layout file (without extension).
    Returns:
        The contents of the layout file as a list of strings, where each string is a line in the file.
    """
    contents: list[str] = []
    with open(os.path.join(ASSETS_DIR, "layouts", f"{layout_name}.txt"), "r") as f:
        for line in f.readlines():
            contents.append(line.strip())
    return contents

def read_menu_file(menu_name: str) -> dict[str, float]:
    """
    Read a menu file and return its contents as a dictionary.

    Args:
        menu_name: The name of the menu file (without extension).
    Returns:
        A dictionary mapping recipe names to their spawn probabilities. 
    """
    with open(os.path.join(ASSETS_DIR, "menus", f"{menu_name}.json"), "r") as f:
        return json.load(f)

def register_layouts(
    layout_name: str,
    menu_name: str,
    max_orders_active: int = 3,
    time_limit: int = 300,
    tip_coefficient: float = 0.0,
):
    """
    Register all layouts found in the layout directory with the cogrid registry. 

    Args:
        layout_name: The name of the layout to register.
        menu_name: The name of the menu to register.
        max_orders_active: The maximum number of active orders allowed at any time.
        time_limit: The time limit for each order.
        tip_coefficient: The coefficient for the tip based on remaining time.
    """
    layout_contents = read_layout_file(layout_name)
    menu_probabilities = read_menu_file(menu_name)

    results = [r.result for r in v2_objects.OPEN_POT_RECIPES]
    order_cfg = {
        "spawn_probs": menu_probabilities,
        "max_active": max_orders_active,
        "time_limit": time_limit,
    }

    layouts.register_layout(layout_name, layout_contents)
    cfg = copy.deepcopy(_v2_base_config)
    for k in ("target_recipes", "resample_on_delivery"):
        cfg.pop(k, None)
    cfg["grid"] = {"layout": layout_name}
    cfg["interactions"] = []
    cfg["orders"] = order_cfg
    cfg["rewards"] = [
        OrderDeliveryReward(
            coefficient=20.0,
            common_reward=True,
            order_time_limit=time_limit,
            tip_coefficient=tip_coefficient
        ),
        ExpiredOrderPenalty(coefficient=-20.0)
    ]
    cfg["tick_fn"] = build_order_tick(order_cfg, recipe_results=results)
    cfg["extra_state_init_fn"] = functools.partial(build_order_extra_state, order_cfg)
    cfg["render_hud_fn"] = build_order_hud_fn(order_cfg)
    cfg["max_steps"] = 900

    print(f"Registering layout: {layout_name}")
    registry.register(f"OvercookedV2-{layout_name}-{menu_name}", functools.partial(_make_v2_env, config=cfg))
