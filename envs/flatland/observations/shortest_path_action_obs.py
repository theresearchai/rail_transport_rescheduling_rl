import gym
import numpy as np
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.rail_env import RailEnv

from envs.flatland.observations import Observation, register_obs

def get_shortest_path_action(env,handle):
    distance_map = env.distance_map.get()

    agent = env.agents[handle]

    if agent.status == RailAgentStatus.READY_TO_DEPART:
        agent_virtual_position = agent.initial_position
    elif agent.status == RailAgentStatus.ACTIVE:
        agent_virtual_position = agent.position
    elif agent.status == RailAgentStatus.DONE:
        agent_virtual_position = agent.target
    else:
        return None

    if agent.position:
        possible_transitions = env.rail.get_transitions(
            *agent.position, agent.direction)
    else:
        possible_transitions = env.rail.get_transitions(
            *agent.initial_position, agent.direction)

    num_transitions = np.count_nonzero(possible_transitions)                    
    
    min_distances = []
    for direction in [(agent.direction + i) % 4 for i in range(-1, 2)]:
        if possible_transitions[direction]:
            new_position = get_new_position(
                agent_virtual_position, direction)
            min_distances.append(
                distance_map[handle, new_position[0],
                            new_position[1], direction])
        else:
            min_distances.append(np.inf)

    if num_transitions == 1:
        observation = [0, 1, 0]

    elif num_transitions == 2:
        idx = np.argpartition(np.array(min_distances), 2)
        observation = [0, 0, 0]
        observation[idx[0]] = 1
    return np.argmax(observation) + 1

@register_obs("shortest_path_action")
class ShortestPathObservation(Observation):

    def __init__(self, config) -> None:
        super().__init__(config)
        self._config = config
        self._builder = ShortestPathActionForRailEnv()

    def builder(self) -> ObservationBuilder:
        return self._builder

    def observation_space(self) -> gym.Space:
        return gym.spaces.Tuple([
            gym.spaces.Box(low=0, high=np.Inf, shape=(1,)),  # shortest path action
        ])


class ShortestPathActionForRailEnv(ObservationBuilder):
    def __init__(self):
        super().__init__()

    def reset(self):
        pass

    def get(self, handle: int = 0):
        action = get_shortest_path_action(self.env,handle)
        return action