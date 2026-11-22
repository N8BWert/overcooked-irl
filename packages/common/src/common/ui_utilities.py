"""
Utility functions for drawing UI elements.
"""

import numpy as np
import pygame
from cogrid.cogrid_env import CoGridEnv
from cogrid.envs.overcooked import v2_objects


INGREDIENT_COLORS = {
    "onion": (255, 200, 50),
    "tomato": (220, 50, 50),
    "broccoli": (34, 139, 34),
    "mushroom": (139, 90, 43),
}

def draw_orders(
    screen: pygame.Surface,
    env: CoGridEnv,
    results: dict,
    time_limit: int = 300,
    top: int = 0,
    card_width: int = 130,
    card_height: int = 56
):
    """
    Draw the current orders on the screen.

    Args:
        screen: The pygame surface to draw on.
        env: The CoGridEnv environment instance.
        results: A list of recipe results corresponding to the order slots.
        time_limit: The maximum time limit for orders.
        top: The y-coordinate of the top of the order display area.
        card_width: The width of each order card.
        card_height: The height of each order card.
    """
    ingredients = {r.result : r.ingredients for r in v2_objects.OPEN_POT_RECIPES}
    ex = env._env_state.extra_state
    slots = np.asarray(ex["overcooked.order_recipe"])
    timers = np.asarray(ex["overcooked.order_timer"])
    pygame.draw.rect(screen, (30, 30, 30), (0, top, screen.get_width(), card_height + 8))
    for i, (rid, t) in enumerate(zip(slots, timers)):
        x = 8 + i * (card_width + 8)
        pygame.draw.rect(screen, (60, 60, 60), (x, top + 4, card_width, card_height), border_radius=6)
        if rid < 0:
            continue
        for j, ing in enumerate(ingredients[results[int(rid)]]):
            pygame.draw.circle(screen, INGREDIENT_COLORS[ing], (x + 24 + j * 34, top + 26), 13)
        frac = max(0.0, min(1.0, float(t) / time_limit))
        color = (255, 80, 80) if frac < 0.2 else (120, 200, 120)
        pygame.draw.rect(screen, color, (x + 8, top + 4 + card_height - 12, int((card_width - 16) * frac), 6))
