import libs.cell_graph_agent
from libs.cell_graph import CellGraph
from libs.cell_graph_locker import CellGraphLocker
from libs.cell_graph_agent import AgentWayStep, CellGraphAgent

from flatland.envs.rail_env import RailEnv, RailAgentStatus, RailEnvActions
from flatland.envs.agent_utils import EnvAgent

from typing import List

def reschedule(cached_ways, vertex_agent_order, agent_way_position, agent_position_duration,
               env: RailEnv, step_idx, controllers, graph: CellGraph, locker: CellGraphLocker):
    locker.reset()

    new_way = [[] for i in range(len(controllers))]

    def rescheduling_main():
        # recalculate new duration for each agent on each cell of the cached way
        position_updated = True
        full_recalc_needed = False
        while position_updated:
            position_updated = False

            for i in range(len(controllers)):
                agent = env.agents[i]
                if (agent_way_position[i] >= len(cached_ways[i])) or agent_done(env, i):
                    continue

                vertex_idx = cached_ways[i][agent_way_position[i]].vertex_idx
                duration = agent_position_duration[i]

                if agent_way_position[i] == len(cached_ways[i])-1:
                    if vertex_idx == controllers[i].target_vertex:  # target vertex
                        new_way[i].append(AgentWayStep(vertex_idx=vertex_idx,
                                                       direction=None,
                                                       arrival_time=duration[0],
                                                       departure_time=duration[1],
                                                       wait_steps = 0,
                                                       action = None,
                                                       prev_way_id = -1))
                        locker.lock(vertex_idx, i, (duration[0], duration[1]))

                        assert len(vertex_agent_order[vertex_idx]) and vertex_agent_order[vertex_idx][0] == i
                        vertex_agent_order[vertex_idx].pop(0)
                        agent_position_duration[i] = None
                        agent_way_position[i] += 1
                        position_updated = True
                else:
                    next_vertex_idx = cached_ways[i][agent_way_position[i] + 1].vertex_idx
                    ticks_per_step = int(round(1 / env.agents[i].speed_data['speed']))

                    # if vertex_agent_order[next_vertex_idx][0] == i and vertex_agent_order[vertex_idx][0] == i: # possible move to next vertex
                    if vertex_agent_order[next_vertex_idx][0] == i:  # possible move to next vertex
                        new_duration = (duration[0], max(duration[1], locker.last_time_step(next_vertex_idx, i)))
                        # if agent_way_position[i]==0 and agent.speed_data['position_fraction'] > 0:
                        #     if new_duration != duration:
                        #         continue
                        duration = new_duration

                        #if not possible to reschedule right, do it with mistakes
                        if locker.is_locked(vertex_idx, i , duration):
                            d0 = duration[0]
                            d1 = duration[1]
                            ind = locker.equal_or_greater_index_end(vertex_idx, duration[0])
                            if ind>=0 and ind<len(locker.data[vertex_idx]):
                                d0 = max(d0, locker.data[vertex_idx][ind][0][1]+1)
                            ind = locker.equal_or_greater_index(vertex_idx, duration[0])
                            if ind >= 0 and ind < len(locker.data[vertex_idx]):
                                d1 = min(d1, locker.data[vertex_idx][ind][0][0])

                            if d1<=d0:
                                d1 = d0+1

                            print(f"Rescheduling mistake for train {i}. {duration[0]},{duration[1]} -> {d0}, {d1}")
                            duration = (d0, d1)
                            full_recalc_needed = True


                        new_way[i].append(AgentWayStep(vertex_idx=vertex_idx,
                                                       direction=None,
                                                       arrival_time=duration[0],
                                                       departure_time=duration[1],
                                                       wait_steps=0,
                                                       action=None,
                                                       prev_way_id=-1))

                        locker.lock(vertex_idx, i, (duration[0], duration[1]))

                        assert len(vertex_agent_order[vertex_idx]) and vertex_agent_order[vertex_idx][0] == i
                        vertex_agent_order[vertex_idx].pop(0)

                        # #bad situation... what can we do - the best we can
                        # assert len(vertex_agent_order[vertex_idx])
                        # index = vertex_agent_order[vertex_idx].index(i)
                        # if index>0:
                        #     print("Malfunction swapped order, ignore it for now")
                        #     full_recalc_needed = True
                        # vertex_agent_order[vertex_idx].pop(index)

                        position_updated = True

                        agent_way_position[i] += 1
                        if agent_way_position[i] == len(cached_ways[i]) - 1: # next vertex is target
                            duration = (duration[1], duration[1] + 1)
                        else:
                            duration = (duration[1], duration[1] + ticks_per_step)
                        agent_position_duration[i] = duration
        return full_recalc_needed
    full_recalc_needed = rescheduling_main()

    INF_STEP = 10000

    def check_last_rescheduled_step():
        # stop in last possible cell, if no way to target
        for i, agent in enumerate(env.agents):
            controller = controllers[i]
            if agent.status not in [RailAgentStatus.ACTIVE, RailAgentStatus.READY_TO_DEPART]:
                continue

            if agent_position_duration[i] is None or len(cached_ways[i]) == 0:
                continue

            pos = agent_way_position[i]

            if pos == len(cached_ways[i]):
                #last vertex in new way is not target - stay here to end of simulation
                if cached_ways[i][-1].vertex_idx != controller.target_vertex:
                    locker.unlock(new_way[i][-1].vertex_idx, i, (new_way[i][-1].arrival_time, new_way[i][-1].departure_time + 1))

                    new_duration = (new_way[i][-1].arrival_time, INF_STEP)
                    locker.lock(new_way[i][-1].vertex_idx, i, new_duration )
                    new_way[i][-1].wait_steps = INF_STEP
                    new_way[i][-1].departure_time = new_duration[1]
            else:
                vertex_idx = cached_ways[i][pos].vertex_idx
                new_duration = (agent_position_duration[i][0], INF_STEP)
                locker.lock(vertex_idx, i, new_duration)
                new_way[i].append(AgentWayStep(vertex_idx=vertex_idx,
                                               direction=None,
                                               arrival_time=new_duration[0],
                                               departure_time=new_duration[1],
                                               wait_steps=INF_STEP,
                                               action=None,
                                               prev_way_id=-1))

    check_last_rescheduled_step()

    # print('rescheduling end')

    return new_way, full_recalc_needed

def recover_agent_way(controller: CellGraphAgent, agent : EnvAgent, graph: CellGraph,  new_way: List[AgentWayStep]):
    N = len(new_way)

    new_way = sorted(new_way, key=lambda step: -step.arrival_time)
    #assert len(new_way)<=len(controller.selected_way)
    #assert len(new_way) == len(controller.selected_way)

    shift = len(controller.selected_way) - len(new_way)
    for i in range(N):
        assert new_way[i].vertex_idx==controller.selected_way[i+shift].vertex_idx

    for i in range(N):
        new_way[i].direction = controller.selected_way[i+shift].direction
        new_way[i].action = controller.selected_way[i+shift].action

    ticks_per_element = int(round(1. / agent.speed_data['speed']))

    changed = False
    for i in range(N):
        changed = changed or new_way[i].arrival_time != controller.selected_way[i].arrival_time or \
                  new_way[i].departure_time != controller.selected_way[i].departure_time
        new_way[i].wait_steps = max(new_way[i].departure_time - new_way[i].arrival_time - ticks_per_element, 0)
        if i == N - 1:
            new_way[i].wait_steps -= agent.malfunction_data['malfunction']
            new_way[i].wait_steps += int(round(min(agent.speed_data['position_fraction'], 1-agent.speed_data['speed']) * ticks_per_element))

    controller.selected_way = new_way
    # controller.selected_way = controller.selected_way[:shift] + new_way
    return changed



def agent_done(env, agent_id):
    return env.agents[agent_id].status in [RailAgentStatus.DONE, RailAgentStatus.DONE_REMOVED]




