from cogrid.core.component_registry import register_scope_modules

from env.objects import SCOPE

register_scope_modules(SCOPE, ["env.objects", "env.features"])

from env.envs import ENV_IDS
