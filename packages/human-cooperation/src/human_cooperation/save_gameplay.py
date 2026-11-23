import json
from datetime import datetime
from pathlib import Path

class GameplayLog:
    def __init__(self, layout, menu):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.path = Path("trajectories") / f"gameplay-{timestamp}.json"
        self.layout = layout
        self.menu = menu
        self.episodes = []
        self._episode = None

    def start_episode(self, seed):
        self._episode = {
            "seed": seed,
            "layout": self.layout,
            "menu": self.menu,
            "p1": [],
            "p2": [],
        }
        self.step_index = 0

    def record(self, env, actions):
        for player in env.possible_agents[:2]:
            position = env._env_state.agent_pos[player]
            self._episode[f"p{player+1}"].append({
                "position": [int(position[0]), int(position[1])],
                "action": int(actions[player]),
            })
        self.step_index += 1

    def finish_episode(self):
        self.episodes.append(self._episode)

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({
            "episodes": self.episodes,
        }, indent=2))
        return self.path