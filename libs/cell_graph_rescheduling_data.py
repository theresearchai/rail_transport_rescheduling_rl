from libs.cell_graph import CellGraph
from libs.cell_graph_locker import CellGraphLocker
from libs.cell_graph_agent import AgentWayStep, CellGraphAgent

from flatland.envs.rail_env import RailEnv, RailAgentStatus, RailEnvActions
from flatland.envs.agent_utils import EnvAgent

def get_rescheduling_data(env: RailEnv, step_idx, controllers, graph: CellGraph, locker: CellGraphLocker):
    cached_ways = []
    cached_way_vertexes = []

    def save_previous_ways():
        for i, c in enumerate(controllers):
            agent = env.agents[i]
            way = list(reversed(c.get_cached_way()))

            if agent.status in [RailAgentStatus.ACTIVE, RailAgentStatus.READY_TO_DEPART] and len(way):
                point = agent.position if agent.position is not None else agent.initial_position
                vertex_idx = graph._vertex_idx_from_point(point)
                c._update_selected_way_new_position(vertex_idx)
                way = list(reversed(c.get_cached_way()))
                cached_ways.append(way)
                cached_way_vertexes.append(set([(w.vertex_idx, (w.arrival_time, w.departure_time)) for w in way]))
            else:
                cached_ways.append([])
                cached_way_vertexes.append(set([]))

    save_previous_ways()

    vertex_agent_order = []

    def update_controllers_position():
        for agent_id, c in enumerate(controllers):
            agent = env.agents[agent_id]
            if agent.position is None:
                continue

            vertex_idx = graph._vertex_idx_from_point(agent.position)
            c._update_selected_way_new_position(vertex_idx)

    update_controllers_position()

    def get_agents_order_for_cells():
        for vertex_idx, lock_data in enumerate(locker.data):
            order = []
            for duration, agent_id in lock_data:
                if agent_done(env, agent_id) or (vertex_idx, duration) not in cached_way_vertexes[agent_id]:
                    # if agent_done(env, agent_id):
                    continue
                order.append(agent_id)
            vertex_agent_order.append(order)

    get_agents_order_for_cells()

    # locker.reset()

    cached_ways = [list(reversed(agent.get_cached_way())) for agent in controllers]

    agent_way_position = [0 for i in range(len(controllers))]
    agent_position_duration = []

    def init_first_rescheduling_step():
        # initial first cell duration for all agents
        for i, agent in enumerate(env.agents):
            if (not len(cached_ways[i])) or agent_done(env, i):
                agent_position_duration.append(None)
                continue

            ticks_per_step = int(round(1 / agent.speed_data['speed']))
            start_time = step_idx
            end_time = start_time + ticks_per_step

            if agent.status == RailAgentStatus.READY_TO_DEPART:
                end_time += 1
            if agent.speed_data['position_fraction'] != 0.0:
                end_time -= int(
                    round(min(agent.speed_data['position_fraction'], 1 - agent.speed_data['speed']) * ticks_per_step))
            if agent.malfunction_data['malfunction'] > 0:
                end_time += agent.malfunction_data['malfunction']

            agent_position_duration.append((start_time, end_time))

    init_first_rescheduling_step()

    return cached_ways, vertex_agent_order, agent_way_position, agent_position_duration


def agent_done(env, agent_id):
    return env.agents[agent_id].status in [RailAgentStatus.DONE, RailAgentStatus.DONE_REMOVED]