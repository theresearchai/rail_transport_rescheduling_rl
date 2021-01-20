import gym
import numpy as np
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.rail_env import RailEnv

from envs.flatland.observations import Observation, register_obs
np.random.seed(1)


@register_obs("random_action")
class RandomActionObservation(Observation):

    def __init__(self, config) -> None:
        super().__init__(config)
        self._config = config
        self._builder = RandomActionForRailEnv()

    def builder(self) -> ObservationBuilder:
        return self._builder

    def observation_space(self) -> gym.Space:
        return gym.spaces.Tuple([
            gym.spaces.Box(low=0, high=np.Inf, shape=(1,)),  # shortest path action
        ])


class RandomActionForRailEnv(ObservationBuilder):
    def __init__(self):
        super().__init__()

    def reset(self):
        pass

    def get(self, handle: int = 0):
        _num_outputs = self.env.action_space[0]
        action = np.random.randint(0, _num_outputs)
        return action