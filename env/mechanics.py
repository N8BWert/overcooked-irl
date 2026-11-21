import dataclasses

import numpy as np

from cogrid.backend import xp
from cogrid.backend._dispatch import get_backend
from cogrid.backend.array_ops import set_at, set_at_2d
from cogrid.core.objects.registry import get_object_names
from cogrid.core.pipeline.context import find_facing_instance

from gym_cooking.utils.agent import SimAgent
from gym_cooking.utils.config import (
    CHOPPING_NUM_STEPS, COOKED_BEFORE_FIRE_TIME_SECONDS, COOKING_TIME_SECONDS, FIRE_PUTOUT_TIME_SECONDS,
    FIRE_RECOVER_GAP_TIME_SECONDS, ORDER_EXPIRE_PUNISH,
)
from gym_cooking.utils.core import (
    ASSEMBLE_COOKED_FOOD, ASSEMBLE_COOKING_FOOD, CHOPPING_FOOD, Floor, Object, Rep, RepToClass,
)
from gym_cooking.utils.interact import interact
from gym_cooking.utils.world import World

from env.objects import (
    ASSEMBLE_CHARRED_FIRE_FOOD, GRIDSQUARE_NAMES, GRIDSQUARE_REPS, HELD_FOOD, POT_FOOD, SCOPE, make_hla_object,
)

FPS = 10
NEG = -10 ** 9
COOKING_STEPS = COOKING_TIME_SECONDS * FPS
COOKED_BEFORE_FIRE_STEPS = COOKED_BEFORE_FIRE_TIME_SECONDS * FPS
FIRE_PUTOUT_STEPS = FIRE_PUTOUT_TIME_SECONDS * FPS
FIRE_RECOVER_GAP_STEPS = FIRE_RECOVER_GAP_TIME_SECONDS * FPS
NO_EVENT, CHOP_START, CHOP_STEP, COOK, DELIVER, PUTOUT = range(6)
GS_REPS = [Rep.COUNTER] + GRIDSQUARE_REPS
FACED = {Rep.COUNTER: HELD_FOOD, Rep.CUTBOARD: HELD_FOOD + CHOPPING_FOOD, Rep.POT: POT_FOOD}
_P = f"{SCOPE}."


def _interact_once(held, rep, faced):
    world = World(arglist=None)
    world.width, world.height = 2, 1
    for gs in (Floor((0, 0)), RepToClass[rep]((1, 0))):
        world.insert(gs)
    gs = world.get_gridsquare_at((1, 0))
    agent = SimAgent(name='agent-1', id_color='blue', location=(0, 0))
    if held is not None:
        agent.acquire(make_hla_object(held, (0, 0)))
        world.insert(agent.holding)
    if faced is not None:
        gs.acquire(make_hla_object(faced, (1, 0)))
        world.insert(gs.holding)
    agent.action = (1, 0)
    event = interact(agent=agent, world=world, current_time=1.0).event
    objs = [o for o in world.get_object_list() if isinstance(o, Object) and o.location == (1, 0) and not o.is_held]
    assert len(objs) <= 1
    new_held = None if agent.holding is None else agent.holding.full_name
    new_faced = None if rep == Rep.DELIVERY or not objs else objs[0].full_name
    if event.startswith('Cook_'):
        code = COOK
    elif event.startswith('Deliver_'):
        code = DELIVER
    elif event == 'Putout_Fire':
        code = PUTOUT
    elif new_faced in CHOPPING_FOOD and faced not in CHOPPING_FOOD:
        code = CHOP_START
    elif new_faced in CHOPPING_FOOD and objs[0].contents[0].state._rest_steps < CHOPPING_NUM_STEPS:
        code = CHOP_STEP
    else:
        code = NO_EVENT
    return new_held, new_faced, code


class HLATables:
    def __init__(self):
        names = get_object_names(scope=SCOPE)
        n = len(names)
        idx = {name: i for i, name in enumerate(names) if name is not None}
        self.idx = idx
        self.gs_col = np.full(n, -1, dtype=np.int32)
        for col, rep in enumerate(GS_REPS):
            self.gs_col[idx['counter' if rep == Rep.COUNTER else GRIDSQUARE_NAMES[rep]]] = col
        self.new_held = np.tile((np.arange(n + 1, dtype=np.int32) - 1)[:, None, None], (1, len(GS_REPS), n))
        self.new_faced = np.tile(np.arange(n, dtype=np.int32)[None, None, :], (n + 1, len(GS_REPS), 1))
        self.event = np.zeros((n + 1, len(GS_REPS), n), dtype=np.int32)
        for g, rep in enumerate(GS_REPS):
            for held in [None] + HELD_FOOD:
                for faced in [None] + FACED.get(rep, []):
                    try:
                        new_held, new_faced, code = _interact_once(held, rep, faced)
                    except Exception:
                        continue
                    if (new_held is not None and new_held not in idx) or (new_faced is not None and new_faced not in idx):
                        continue
                    h = 0 if held is None else idx[held] + 1
                    f = 0 if faced is None else idx[faced]
                    self.new_held[h, g, f] = -1 if new_held is None else idx[new_held]
                    self.new_faced[h, g, f] = 0 if new_faced is None else idx[new_faced]
                    self.event[h, g, f] = code
        self.chopped = np.arange(n, dtype=np.int32)
        for c in CHOPPING_FOOD:
            self.chopped[idx[c]] = idx[c.replace('Chopping', 'Chopped')]
        self.cooked = np.arange(n, dtype=np.int32)
        for c in ASSEMBLE_COOKING_FOOD:
            self.cooked[idx[c]] = idx[c.replace('Cooking', 'Cooked')]
        self.charred_fire = np.arange(n, dtype=np.int32)
        for c in ASSEMBLE_COOKED_FOOD:
            self.charred_fire[idx[c]] = idx[c.replace('Cooked', 'Charred') + '-Fire']
        self.charred = np.arange(n, dtype=np.int32)
        for c in ASSEMBLE_CHARRED_FIRE_FOOD:
            self.charred[idx[c]] = idx[c[:-len('-Fire')]]
        self.is_cooking = np.isin(np.arange(n), [idx[c] for c in ASSEMBLE_COOKING_FOOD])
        self.is_cooked = np.isin(np.arange(n), [idx[c] for c in ASSEMBLE_COOKED_FOOD])
        self.has_fire = np.isin(np.arange(n), [idx[c] for c in ASSEMBLE_CHARRED_FIRE_FOOD])


def fire_update_state(rest, latest_time, putout_states, current_time):
    last_putout_time = xp.max(putout_states)
    decrease = xp.maximum(0, xp.minimum(current_time, last_putout_time + FIRE_RECOVER_GAP_STEPS) - xp.maximum(latest_time, last_putout_time))
    increase = xp.maximum(0, current_time - xp.maximum(latest_time, last_putout_time + FIRE_RECOVER_GAP_STEPS))
    rest = rest - decrease
    rest = xp.where(rest <= 0, rest, xp.minimum(FIRE_PUTOUT_STEPS, rest + increase))
    return rest, current_time


def build_interactions(tables):
    def hla_interact(ctx):
        fr, fc = ctx.facing_row, ctx.facing_col
        now = ctx.current_time
        col = xp.asarray(tables.gs_col)[ctx.facing_type]
        g = xp.maximum(col, 0)
        h = ctx.held_item + 1
        faced = ctx.object_state_map[fr, fc]
        ok = ctx.can_interact & (ctx.action == ctx.action_id.pickup_drop) & (col >= 0)
        event = xp.where(ok, xp.asarray(tables.event)[h, g, faced], NO_EVENT)
        new_held = xp.asarray(tables.new_held)[h, g, faced]
        new_faced = xp.asarray(tables.new_faced)[h, g, faced]

        b, _ = find_facing_instance(ctx.board_positions, fr, fc)
        board_rest = ctx.board_rest[b] - 1
        new_faced = xp.where((event == CHOP_STEP) & (board_rest <= 0), xp.asarray(tables.chopped)[faced], new_faced)
        board_rest = xp.where(event == CHOP_START, CHOPPING_NUM_STEPS, xp.where(event == CHOP_STEP, board_rest, ctx.board_rest[b]))

        p, at_pot = find_facing_instance(ctx.pot_positions, fr, fc)
        pot_start = xp.where(event == COOK, now, ctx.pot_start[p])
        fire_rest, fire_latest = fire_update_state(ctx.fire_rest[p], ctx.fire_latest[p], ctx.putout_states[p], now)
        putout = event == PUTOUT
        putout_states = xp.where(putout, set_at(ctx.putout_states, (p, ctx.agent_index), now), ctx.putout_states)
        keep = (ctx.action == ctx.action_id.noop) | ((ctx.action >= ctx.action_id.pickup_drop) & at_pot)
        target = xp.arange(ctx.pot_positions.shape[0]) == p
        putout_states = set_at(putout_states, (slice(None), ctx.agent_index),
                               xp.where(keep & target, putout_states[:, ctx.agent_index], NEG))

        return xp.bool_(True), {
            "agent_inv": xp.where(ok, set_at(ctx.agent_inv, (ctx.agent_index, 0), new_held), ctx.agent_inv),
            "object_state_map": xp.where(ok, set_at_2d(ctx.object_state_map, fr, fc, new_faced), ctx.object_state_map),
            "board_rest": set_at(ctx.board_rest, b, board_rest),
            "pot_start": set_at(ctx.pot_start, p, pot_start),
            "fire_rest": xp.where(putout, set_at(ctx.fire_rest, p, fire_rest), ctx.fire_rest),
            "fire_latest": xp.where(putout, set_at(ctx.fire_latest, p, fire_latest), ctx.fire_latest),
            "putout_states": putout_states,
            "delivered": xp.where(event == DELIVER, set_at(ctx.delivered, ctx.agent_index, ctx.held_item), ctx.delivered),
        }

    return [hla_interact]


def _compact(order_recipe, order_timer):
    order = xp.argsort((order_recipe == -1).astype(xp.int32), stable=True)
    return order_recipe[order], order_timer[order]


def build_tick(tables, goals, lengths, bonuses):
    goals_np = np.asarray([tables.idx[g] for g in goals], dtype=np.int32)
    lengths_np = np.asarray([l * FPS for l in lengths], dtype=np.int32)
    bonuses_np = np.asarray(bonuses, dtype=np.int32)
    n_recipes = len(goals)
    spawn_probs_np = np.full(n_recipes, 1.0 / n_recipes, dtype=np.float32)

    def tick(state, scope_config):
        new_extra = dict(state.extra_state)
        current_time = new_extra[_P + "current_time"]

        # Update Orders
        order_recipe = new_extra[_P + "order_recipe"]
        order_timer = new_extra[_P + "order_timer"]
        delivered = new_extra[_P + "delivered"]
        reward = new_extra[_P + "reward"]
        successful_orders = new_extra[_P + "successful_orders"]
        failed_orders = new_extra[_P + "failed_orders"]
        max_active = order_recipe.shape[0]
        for k in range(max_active):
            r = xp.maximum(order_recipe[k], 0)
            hit = (delivered == xp.asarray(goals_np)[r]) & (order_recipe[k] != -1)
            success = xp.any(hit)
            delivered = xp.where(success, set_at(delivered, xp.argmax(hit), -1), delivered)
            reward = reward + xp.where(success, xp.asarray(bonuses_np)[r], 0)
            successful_orders = successful_orders + success.astype(xp.int32)
            order_recipe = xp.where(success, set_at(order_recipe, k, -1), order_recipe)
        order_recipe, order_timer = _compact(order_recipe, order_timer)

        active = order_recipe != -1
        new_order_timer = xp.where(active, order_timer - 1, order_timer)
        expired = active & (new_order_timer <= 0)
        order_recipe = xp.where(expired, -1, order_recipe)
        new_order_timer = xp.where(expired, 0, new_order_timer)
        failed_orders = failed_orders + xp.sum(expired).astype(xp.int32)
        reward = reward - ORDER_EXPIRE_PUNISH * xp.sum(expired).astype(xp.int32)
        order_recipe, new_order_timer = _compact(order_recipe, new_order_timer)

        cdf = xp.cumsum(spawn_probs_np)

        if get_backend() == "jax":
            import jax

            key, subkey = jax.random.split(state.rng_key)
            rolls = jax.random.uniform(subkey, shape=(max_active,))
        else:
            rolls = xp.array(
                np.random.default_rng().uniform(size=(max_active,)),
                dtype=xp.float32,
            )
            key = state.rng_key

        hit = rolls[:, None] < cdf[None, :]
        any_hit = xp.any(hit, axis=1)
        sampled_recipe = xp.argmax(hit.astype(xp.int32), axis=1)

        empty = order_recipe == -1
        spawn_mask = empty & any_hit

        order_recipe = xp.where(spawn_mask, sampled_recipe, order_recipe)
        new_order_timer = xp.where(spawn_mask, xp.asarray(lengths_np)[sampled_recipe], new_order_timer)

        new_extra[_P + "order_recipe"] = order_recipe.astype(xp.int32)
        new_extra[_P + "order_timer"] = new_order_timer.astype(xp.int32)
        new_extra[_P + "reward"] = reward.astype(xp.int32)
        new_extra[_P + "successful_orders"] = successful_orders.astype(xp.int32)
        new_extra[_P + "failed_orders"] = failed_orders.astype(xp.int32)
        new_extra[_P + "delivered"] = xp.full_like(delivered, -1)

        # update cooking or cooked objects
        osm = state.object_state_map
        pot_positions = new_extra[_P + "pot_positions"]
        pot_start = new_extra[_P + "pot_start"]
        fire_rest = new_extra[_P + "fire_rest"]
        fire_latest = new_extra[_P + "fire_latest"]
        putout_states = new_extra[_P + "putout_states"]
        for p in range(pot_positions.shape[0]):
            r, c = pot_positions[p, 0], pot_positions[p, 1]
            item = osm[r, c]
            cooking_done = xp.asarray(tables.is_cooking)[item] & (current_time - pot_start[p] >= COOKING_STEPS)
            cooked_done = xp.asarray(tables.is_cooked)[item] & (current_time - pot_start[p] >= COOKED_BEFORE_FIRE_STEPS)
            on_fire = xp.asarray(tables.has_fire)[item]
            rest, latest = fire_update_state(fire_rest[p], fire_latest[p], putout_states[p], current_time)
            fire_finished = on_fire & (rest <= 0)
            item = xp.where(cooking_done, xp.asarray(tables.cooked)[item],
                            xp.where(cooked_done, xp.asarray(tables.charred_fire)[item],
                                     xp.where(fire_finished, xp.asarray(tables.charred)[item], item)))
            osm = set_at_2d(osm, r, c, item)
            pot_start = xp.where(cooking_done, set_at(pot_start, p, current_time), pot_start)
            fire_rest = xp.where(cooked_done, set_at(fire_rest, p, FIRE_PUTOUT_STEPS),
                                 xp.where(on_fire, set_at(fire_rest, p, rest), fire_rest))
            fire_latest = xp.where(cooked_done, set_at(fire_latest, p, NEG),
                                   xp.where(on_fire, set_at(fire_latest, p, latest), fire_latest))
            putout_states = xp.where(cooked_done, set_at(putout_states, (p, slice(None)), NEG), putout_states)
        new_extra[_P + "pot_start"] = pot_start
        new_extra[_P + "fire_rest"] = fire_rest
        new_extra[_P + "fire_latest"] = fire_latest
        new_extra[_P + "putout_states"] = putout_states
        new_extra[_P + "current_time"] = current_time + 1

        return dataclasses.replace(state, object_state_map=osm, extra_state=new_extra, rng_key=key)

    return tick


def build_extra_state_init(tables, max_num_orders, num_agents):
    def init(parsed_state, scope=SCOPE):
        otm = np.asarray(parsed_state["object_type_map"])
        pots = np.argwhere(otm == tables.idx['Pot']).astype(np.int32)
        boards = np.argwhere(otm == tables.idx['Cutboard']).astype(np.int32)
        return {
            _P + "current_time": np.int32(0),
            _P + "pot_positions": pots,
            _P + "pot_start": np.zeros(len(pots), dtype=np.int32),
            _P + "fire_rest": np.zeros(len(pots), dtype=np.int32),
            _P + "fire_latest": np.full(len(pots), NEG, dtype=np.int32),
            _P + "putout_states": np.full((len(pots), num_agents), NEG, dtype=np.int32),
            _P + "board_positions": boards,
            _P + "board_rest": np.zeros(len(boards), dtype=np.int32),
            _P + "delivered": np.full(num_agents, -1, dtype=np.int32),
            _P + "order_recipe": np.full(max_num_orders, -1, dtype=np.int32),
            _P + "order_timer": np.zeros(max_num_orders, dtype=np.int32),
            _P + "reward": np.int32(0),
            _P + "successful_orders": np.int32(0),
            _P + "failed_orders": np.int32(0),
        }

    return init
