"""
Play Overcooked with two human players.
"""

import argparse
import pygame
from cogrid.envs import registry
from cogrid.cogrid_env import CoGridEnv
from cogrid.envs.overcooked import v2_objects
from layouts import register_layouts
from common.ui_utilities import draw_orders
from dataclasses import replace
from human_cooperation.save_gameplay import GameplayLog

# Player 1 action scheme
P1_ACTION_SCHEME = {
    "move": {
        pygame.K_w: 0,
        pygame.K_s: 1,
        pygame.K_a: 2,
        pygame.K_d: 3,
    },
    "act": {
        pygame.K_e: 4,
        pygame.K_q: 5
    }
}
# Player 2 action scheme
P2_ACTION_SCHEME = {
    "move": {
        pygame.K_UP: 0,
        pygame.K_DOWN: 1,
        pygame.K_LEFT: 2,
        pygame.K_RIGHT: 3,
    },
    "act": {
        pygame.K_RETURN: 4,
        pygame.K_RSHIFT: 5
    }
}
# The number of milliseconds between each game tick
TICK_MS: int = 200
# The height of the order display area in pixels
ORDER_DISPLAY_HEIGHT: int = 64
# The scale factor for the game window size
SCALE: int = 2


def parse_args() -> tuple[str, str]:
    """
    Parse command line arguments to get the layout name.

    Returns:
        A tuple containing the layout name and menu name.
    """
    parser = argparse.ArgumentParser(description="Play Overcooked with two human players.")
    parser.add_argument("--layout", type=str, help="The layout to use for the game.", default="kitchen")
    parser.add_argument("--menu", type=str, help="The menu to use for the game.", default="menu1")
    args = parser.parse_args()
    return args.layout, args.menu

def read_action(keys: list[int], pending: set[int], scheme: dict[str, dict[int, int]], noop: int) -> int:
    for k, a in scheme["act"].items():
        if k in pending:
            return a
    for k, a in scheme["move"].items():
        if keys[k]:
            return a
    return noop

def draw_frame(screen: pygame.Surface, env: CoGridEnv, score):
    results = [r.result for r in v2_objects.OPEN_POT_RECIPES]
    frame = env.get_full_render()
    h, w = frame.shape[:2]
    surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
    surf = pygame.transform.scale(surf, (w * SCALE, h * SCALE))
    screen.fill((20, 20, 20))
    screen.blit(surf, (0, ORDER_DISPLAY_HEIGHT))
    draw_orders(screen, env, results)
    text = pygame.font.Font(None, 28).render(str(int(score)), True, (255, 255, 255))
    screen.blit(text, (screen.get_width() - text.get_width() - 8, 8))

def show_seconds(original_render):
    def pot_timer_to_seconds(self, tile_img):
        steps = self.cooking_timer
        if steps == 0:
            self.cooking_timer = "F"
        else:
            self.cooking_timer = f"{steps * TICK_MS / 1000:.0f}s"
        original_render(self, tile_img)
        self.cooking_timer = steps
    return pot_timer_to_seconds

def main():
    layout_name, menu_name = parse_args()
    register_layouts(layout_name, menu_name)
    env = registry.make(f"OvercookedV2-{layout_name}-{menu_name}", render_mode="rgb_array")
    seed = 0
    obs, info = env.reset(seed=seed)
    p1, p2 = env.possible_agents[:2]
    v2_objects.OPEN_POT_RECIPES[:] = [replace(recipe, cook_time=25) for recipe in v2_objects.OPEN_POT_RECIPES]
    v2_objects.OpenPot.render = show_seconds(v2_objects.OpenPot.render)

    pygame.init()
    h, w = env.get_full_render().shape[:2]
    screen = pygame.display.set_mode((w * SCALE, h * SCALE + ORDER_DISPLAY_HEIGHT))
    clock = pygame.time.Clock()
    last_tick, pending = 0, set()
    gameplay_log = GameplayLog(layout_name, menu_name)
    gameplay_log.start_episode(seed)
    score = 0

    running = True
    try:
        while running:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                elif e.type == pygame.KEYDOWN:
                    pending.add(e.key)

            now = pygame.time.get_ticks()
            if now - last_tick >= TICK_MS:
                keys = pygame.key.get_pressed()
                actions = {
                    p1: read_action(keys, pending, P1_ACTION_SCHEME, noop=6),
                    p2: read_action(keys, pending, P2_ACTION_SCHEME, noop=6)
                }
                pending.clear()
                gameplay_log.record(env, actions)   
                obs, rewards, terms, truncs, info = env.step(actions)
                score += sum(rewards.values())
                last_tick = now
                if not env.agents:
                    gameplay_log.finish_episode()
                    obs, info = env.reset()
                    score = 0
                    gameplay_log.start_episode(env.reset.__kwdefaults__["seed"])

                # Render the environment
                draw_frame(screen, env, score)

            pygame.display.flip()
            clock.tick(60)
    finally:
        if gameplay_log._episode:
            gameplay_log.finish_episode()
            gameplay_log.save()
    pygame.quit()
    