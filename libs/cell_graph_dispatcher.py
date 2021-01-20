import traceback
from copy import deepcopy
from typing import Dict

from flatland.envs.rail_env import RailEnv, RailAgentStatus, RailEnvActions
from libs import cell_graph_rescheduling, cell_graph_partial_rescheduling, cell_graph_rescheduling_data
from libs.cell_graph import CellGraph
from libs.cell_graph_agent import CellGraphAgent
from libs.cell_graph_locker import CellGraphLocker


class CellGraphDispatcher:
    def __init__(self, env: RailEnv, sort_function=None):
        self.env = env

        self.graph = CellGraph(env)
        self.locker = CellGraphLocker(self.graph)

        max_steps = env._max_episode_steps
        self.controllers = [CellGraphAgent(agent, self.graph, self.locker, i, max_steps) for i, agent in
                            enumerate(env.agents)]

        self.action_dict = {}

        if sort_function is None:
            sort_function = lambda idx: self.controllers[idx].dist_to_target[
                                            self.graph._vertex_idx_from_point(env.agents[idx].initial_position),
                                            env.agents[idx].initial_direction] \
                                        - 10000 * env.agents[idx].speed_data['speed']
        else:
            sort_function = sort_function(self)

        self.agents_order = sorted(range(len(env.agents)), key=sort_function)
        self.agent_locked_by_malfunction = []
        for agent in env.agents:
            self.agent_locked_by_malfunction.append(agent.malfunction_data['malfunction'] > 0)
        self.crashed = False
        self.blocked_agents = set()

    def step(self, step) -> Dict[int, RailEnvActions]:
        try:
            has_new_malfunctions = False
            for i, agent in enumerate(self.env.agents):
                is_locked = agent.malfunction_data['malfunction']

                if agent.status == RailAgentStatus.ACTIVE:
                    if (not self.agent_locked_by_malfunction[i]) and is_locked:
                        has_new_malfunctions = True

                self.agent_locked_by_malfunction[i] = is_locked

            updated = set()

            full_recalc_needed = False
            # old_locker = None
            try:
                if has_new_malfunctions:
                    # print('new malfunction at step', step)
                    # old_locker = deepcopy(self.locker)
                    cached_ways, vertex_agent_order, agent_way_position, agent_position_duration = \
                        cell_graph_rescheduling_data.get_rescheduling_data(self.env, step, self.controllers, self.graph,
                                                                           self.locker)

                    vertex_agent_order2 = deepcopy(vertex_agent_order)
                    agent_way_position2 = deepcopy(agent_way_position)
                    agent_position_duration2 = deepcopy(agent_position_duration)

                    new_way, full_recalc_needed = cell_graph_rescheduling.reschedule(cached_ways, vertex_agent_order,
                                                                                     agent_way_position,
                                                                                     agent_position_duration,
                                                                                     self.env, step, self.controllers,
                                                                                     self.graph, self.locker)
                    for i in self.agents_order:
                        if len(new_way[i]):
                            changed = cell_graph_rescheduling.recover_agent_way(self.controllers[i], self.env.agents[i],
                                                                                self.graph, new_way[i])
                            if changed:
                                updated.add(i)
            # resheduling failed, try to make a partial rescheduling
            except Exception as e:
                print("-----------------Rescheduling Exception----------------")
                print("Step: ", step)
                # traceback.print_exc()
                print("-----------------Rescheduling Exception----------------")

                updated.clear()
                full_recalc_needed = False
                # if old_locker is not None:
                #     self.locker.data = old_locker.data

                self.partial_resheduling(cached_ways, vertex_agent_order2, agent_way_position2,
                                         agent_position_duration2, step)
                self.limit_max_visited()

            for i in self.agents_order:
                try:
                    agent = self.env.agents[i]

                    # if agent.speed_data['position_fraction'] >= 1.0:
                    #     print('agent', i, 'blocked by some another agent, fraction:', agent.speed_data['position_fraction'])

                    force_new_path = full_recalc_needed or self.crashed or i in updated
                    # force_new_path = full_recalc_needed or i in updated
                    # if (force_new_path and i in self.blocked_agents):
                    #     # self.action_dict.update({i: RailEnvActions.DO_NOTHING})
                    #     force_new_path = False
                    #     # continue
                    if i in self.blocked_agents:
                        force_new_path = True

                    if agent.speed_data['position_fraction'] > 0.0 and not force_new_path:
                        self.action_dict.update({i: RailEnvActions.DO_NOTHING})
                        continue

                    # action = self.controllers[i].act(agent, step, force_new_path=has_new_malfunctions)
                    action = self.controllers[i].act(agent, step, force_new_path=force_new_path)
                    self.action_dict.update({i: action})
                # act crashed tor one agent
                except Exception as e:
                    print("-----------------Agent step Exception----------------", i)
                    print("Step: ", step)
                    # traceback.print_exc()
                    print("-----------------Agent step Exception----------------")

                    self.action_dict.update({i: RailEnvActions.DO_NOTHING})
                    self.limit_max_visited()
                    # pass

            self.blocked_agents.clear()
            self.crashed = False

        # global step exception handling, no idea what to do here
        except Exception as e:
            # except ArithmeticError:
            self.crashed = True
            print("-----------------Step Exception----------------")
            print("Step: ", step)
            traceback.print_exc()
            print("-----------------Step Exception----------------")

            # hit_problem = False
            # for j in self.agents_order:
            #     if j == i:
            #         hit_problem = True
            #     if hit_problem:
            #         self.action_dict.update({j: RailEnvActions.STOP_MOVING })
            self.action_dict = {i: RailEnvActions.STOP_MOVING for i in self.agents_order}
            self.limit_max_visited()

            # raise e

        return self.action_dict

    def partial_resheduling(self, cached_ways, vertex_agent_order2, agent_way_position2, agent_position_duration2,
                            step):
        print('partial_resheduling')
        try:
            new_way, blocked_agents = cell_graph_partial_rescheduling.partial_reschedule(cached_ways,
                                                                                         vertex_agent_order2,
                                                                                         agent_way_position2,
                                                                                         agent_position_duration2,
                                                                                         self.env, step,
                                                                                         self.controllers, self.graph,
                                                                                         self.locker)

            for i in self.agents_order:
                if len(new_way[i]):
                    cell_graph_rescheduling.recover_agent_way(self.controllers[i], self.env.agents[i], self.graph,
                                                              new_way[i])

            self.blocked_agents.update(blocked_agents)

            print('blocked agents', self.blocked_agents)

        except Exception as e:
            self.crashed = True
            print("-----------------Partial rescheduing Exception----------------")
            traceback.print_exc()
            print("-----------------Partial rescheduing Exception----------------")
            self.limit_max_visited()

    def limit_max_visited(self):
        for c in self.controllers:
            c.set_max_visited(100)
