from cogrid.backend import xp
from cogrid.core.pipeline.rewards import Reward


class OrderSchedulerReward(Reward):
    def compute(self, prev_state, state, actions, reward_config):
        newly_rewarded = state.reward - prev_state.reward

        n_agents = reward_config["n_agents"]
        coefficient = self.get_coefficient(state)

        return xp.full(
            n_agents,
            newly_rewarded.astype(xp.float32) * coefficient,
            dtype=xp.float32,
        )
