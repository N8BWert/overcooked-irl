from cogrid.core import objects as grid_object
from cogrid.core.objects import register_object_type

import gym_cooking.utils.core as core
from gym_cooking.utils.core import (
    ASSEMBLE_CHARRED_FOOD,
    ASSEMBLE_CHARRED_PLATE_FOOD,
    ASSEMBLE_CHOPPED_FOOD,
    ASSEMBLE_CHOPPED_PLATE_FOOD,
    ASSEMBLE_COOKED_FOOD,
    ASSEMBLE_COOKED_PLATE_FOOD,
    ASSEMBLE_COOKING_FOOD,
    CHOPPING_FOOD,
    FRESH_FOOD,
    FoodSequence,
    Object,
    Rep,
    RepToClass,
)

SCOPE = "hla"

HELD_FOOD = (
    FRESH_FOOD + ASSEMBLE_CHOPPED_FOOD + ASSEMBLE_CHOPPED_PLATE_FOOD + ASSEMBLE_COOKED_PLATE_FOOD
    + ASSEMBLE_CHARRED_PLATE_FOOD + ['FireExtinguisher', 'Plate']
)
ASSEMBLE_CHARRED_FIRE_FOOD = [f + '-Fire' for f in ASSEMBLE_CHARRED_FOOD]
POT_FOOD = ASSEMBLE_COOKING_FOOD + ASSEMBLE_COOKED_FOOD + ASSEMBLE_CHARRED_FOOD + ASSEMBLE_CHARRED_FIRE_FOOD
ALL_OBJECTS = HELD_FOOD + CHOPPING_FOOD + POT_FOOD

GRIDSQUARE_REPS = [
    Rep.CUTBOARD,
    Rep.DELIVERY,
    Rep.BIN,
    Rep.POT,
    Rep.TOMATO_TILE,
    Rep.LETTUCE_TILE,
    Rep.ONION_TILE,
    Rep.PLATE_TILE,
]
GRIDSQUARE_NAMES = {rep: RepToClass[rep]((0, 0)).name for rep in GRIDSQUARE_REPS}


def make_hla_object(full_name, location):
    contents = []
    for name in full_name.split('-'):
        if name in ('Plate', 'Fire', 'FireExtinguisher'):
            contents.append(getattr(core, name)())
            continue
        for state_index, state in enumerate(FoodSequence.FRESH_CHOPPING_CHOPPED_COOKING_COOKED_CHARRED):
            if name.startswith(state.__name__):
                contents.append(getattr(core, name[len(state.__name__):])(state_index=state_index))
                break
    return Object(location=location, contents=contents)


_CHAR_POOL = iter(chr(c) for c in range(0x100, 0x200))


def _init(self, *args, **kwargs):
    grid_object.GridObj.__init__(self, state=0)


def make_hla_type(object_id, char):
    cls = type(object_id.replace("-", ""), (grid_object.GridObj,), {"char": char})
    cls.__init__ = _init
    register_object_type(object_id, scope=SCOPE)(cls)
    return cls


for _rep in GRIDSQUARE_REPS:
    make_hla_type(GRIDSQUARE_NAMES[_rep], _rep)

for _name in ALL_OBJECTS:
    assert make_hla_object(_name, None).full_name == _name
    make_hla_type(_name, next(_CHAR_POOL))
