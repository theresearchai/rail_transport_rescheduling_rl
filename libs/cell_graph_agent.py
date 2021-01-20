import numpy as np
from collections import deque
from flatland.envs.rail_env import RailEnvActions, EnvAgent, RailAgentStatus

from libs.cell_graph import CellGraph



import heapq
from copy import deepcopy

class AgentCurrentPosition:
    def __init__(self, vertex_id, direction, arrival_time, wait_steps, prev_way_id, action, dist_to_target):
        self.vertex_id = vertex_id
        self.direction = direction
        self.arrival_time = arrival_time
        self.wait_steps = wait_steps
        self.prev_way_id = prev_way_id
        self.action = action
        self.dist_to_target = dist_to_target
        self._d = arrival_time + wait_steps + dist_to_target

    def calc_d(self):
        self._d = self.arrival_time + self.wait_steps + self.dist_to_target

    def __lt__(self, other):
        return self._d < other._d if self._d != other._d else self.dist_to_target < other.dist_to_target

class AgentWayStep:
    def __init__(self, vertex_idx, direction, arrival_time, departure_time, wait_steps, action, prev_way_id):
        self.vertex_idx = vertex_idx
        self.direction = direction
        self.arrival_time = arrival_time
        self.departure_time = departure_time
        self.wait_steps = wait_steps
        self.action = action
        self.prev_way_id = prev_way_id


class CellGraphAgent:
    def __init__(self, agent, graph : CellGraph, locker, agent_id, max_steps):
        self.graph = graph
        self.locker = locker
        self.agent_id = agent_id
        self.max_steps = max_steps

        self.selected_way = []
        self.locked_vertexes = []

        self._calc_distances_to_target(agent.target, int(round(1/agent.speed_data['speed'])))

        self.target_vertex = graph._vertex_idx_from_point(agent.target)

        self.visited = {}
        self.max_visited = None

    def _calc_distances_to_target(self, target_point, ticks):
        dist_to_target = np.zeros((len(self.graph.vertexes), 4), dtype=np.int)
        dist_to_target.fill(-1)

        target_v_idx = self.graph._vertex_idx_from_point(target_point)

        dist_to_target[target_v_idx] = 0

        q = deque()
        for d in range(4):
            q.append((target_v_idx, d, 0))

        while len(q):
            u, d, dist = q.popleft()

            for e in self.graph.vertexes[u].in_edges[d]:
                v = e.start_v
                v_d = e.start_direction

                if dist_to_target[v, v_d] == -1:
                    dist_to_target[v, v_d] = dist + ticks
                    q.append((v, v_d, dist + ticks))

        self.dist_to_target = dist_to_target

    def _update_selected_way_new_position(self, vertex_idx):
        while len(self.selected_way) and self.selected_way[-1].vertex_idx != vertex_idx:
            duration = (self.selected_way[-1].arrival_time, self.selected_way[-1].departure_time)
            if self.locker.is_locked(self.selected_way[-1].vertex_idx, self.agent_id, duration):
                self.locker.unlock(self.selected_way[-1].vertex_idx, self.agent_id, duration)

            self.selected_way.pop()
            self.locked_vertexes.pop()

    def _update_selected_way(self, step_idx, vertex_idx, agent):
        #vertex_idx, direction, arrival_time, departure_time, wait_steps, action, prev_way_id

        self._update_selected_way_new_position(vertex_idx)

        if len(self.selected_way)==0:
            return False

        if step_idx<self.selected_way[-1].arrival_time or step_idx>=self.selected_way[-1].departure_time:
            # assert True
            # self.unlock_path()
            # self.selected_way = []
            # self.locked_vertexes = []
            return False

        ticks = int(round(1/agent.speed_data['speed']))
        if self.selected_way[-1].departure_time < step_idx+ticks*(1-agent.speed_data['position_fraction']):
            # self.unlock_path()
            # self.selected_way = []
            # self.locked_vertexes = []
            return False

        self.selected_way[-1].wait_steps = self.selected_way[-1].departure_time - step_idx - ticks*(1-agent.speed_data['position_fraction'])
        # if agent.old_position is None:
        #     self.selected_way[-1].wait_steps -= 1

        # self.selected_way[-1].arrival_time = step_idx

        # locked = [self.locker.is_locked(w.vertex_idx, self.agent_id, (w.arrival_time, w.departure_time + 1)) for w in self.selected_way]
        # locked = np.any(locked)
        #
        # if locked:
        #     self.unlock_path()
        #     self.selected_way = []
        #     self.locked_vertexes = []

        return True

    def clean_visited(self):
        list(v.clear() for v in self.visited.values())

    def check_and_update_visited(self, end_time: int, vertex_id: int, direction: int) -> bool:
        vis_list = self.visited.setdefault(direction*1000000 + vertex_id, [])

        if self.max_visited is not None and len(vis_list)>=self.max_visited:
            return True

        if end_time in vis_list:
            return True

        vis_list.append(end_time)
        return False

    def act(self, agent : EnvAgent, step_idx, force_new_path = False, repeat=False):
        if agent.status == RailAgentStatus.DONE or agent.status == RailAgentStatus.DONE_REMOVED:
            return RailEnvActions.DO_NOTHING

        not_started = agent.status == RailAgentStatus.READY_TO_DEPART
        ticks_per_step = int(round(1 / agent.speed_data['speed']))

        point = agent.position if agent.position is not None else agent.initial_position
        direction = agent.direction if agent.position is not None else agent.initial_direction

        start_vertex_idx = self.graph._vertex_idx_from_point(point)
        # min_time = self.dist_to_target[start_vertex_idx, direction]
        # cut_time = min_time*2+step_idx if agent.status == RailAgentStatus.READY_TO_DEPART else min_time*5+step_idx

        start_action = None
        first_fixed_action = None

        if agent.speed_data['position_fraction'] != 0.0:
            start_action = RailEnvActions.DO_NOTHING
            first_fixed_action = agent.speed_data['transition_action_on_cellexit']
        elif agent.status == RailAgentStatus.READY_TO_DEPART:
            start_action = RailEnvActions.MOVE_FORWARD

        path_is_ok = self._update_selected_way(step_idx, start_vertex_idx, agent)
        if not repeat and not path_is_ok:
            force_new_path = True

        way = []
        if force_new_path:
            # way = deepcopy(self.selected_way)
            self.unlock_path()
            way = self.selected_way
            self.selected_way = []

        if len(self.selected_way):

            # self.lock_selected_way()

            #assert len(self.selected_way) >= 2
            if self.selected_way[-1].wait_steps:
                return  RailEnvActions.STOP_MOVING

            return self.selected_way[-2].action if len(self.selected_way) > 1 else RailEnvActions.STOP_MOVING


        pq = [] # priority queue with current agent positions
        # heapq.heappush(pq, AgentCurrentPosition(start_vertex_idx, direction, step_idx, 0, -1, None,
        #                                         dist_to_target=self.dist_to_target[start_vertex_idx, direction]))
        next_q_el = AgentCurrentPosition(start_vertex_idx, direction, step_idx, 0, -1, None,
                                         dist_to_target=self.dist_to_target[start_vertex_idx, direction])

        self.clean_visited()

        way_history = []
        best_way = -1

        while next_q_el is not None or len(pq):
        # while len(pq):

            if self.max_visited is not None and len(pq) >= 500:
                break

            # curr_pos = heapq.heappop(pq)
            if next_q_el is None:
                curr_pos = heapq.heappop(pq)
            else:
                curr_pos, next_q_el = next_q_el, None
            # curr_pos = pq[0]
            if curr_pos.arrival_time >= 10000:
                continue

            end_time = curr_pos.arrival_time + curr_pos.wait_steps + ticks_per_step

            # if self.dist_to_target[curr_pos.vertex_id, curr_pos.direction] + curr_pos.arrival_time > cut_time:
            #     continue

            # if not_started and \
            #     curr_pos.arrival_time + self.dist_to_target[curr_pos.vertex_id, curr_pos.direction] - step_idx > 3*min_time:
            #         continue

            # if end_time + curr_pos.dist_to_target > self.max_steps + 4:
            #     continue

            def update_end_time(end_time):
                if curr_pos.prev_way_id == -1: # agent at begin of a way, need to check intermediate position, malfunctions etc.
                    if agent.speed_data['position_fraction'] != 0.0:
                        end_time -= int(round(min(agent.speed_data['position_fraction'], 1 - agent.speed_data['speed'])*ticks_per_step))
                    elif agent.status == RailAgentStatus.READY_TO_DEPART:
                        # end_time += int(round(1/agent.speed_data['speed']))
                        end_time += 1
                    if agent.malfunction_data['malfunction'] > 0:
                        end_time += agent.malfunction_data['malfunction']

                if curr_pos.dist_to_target == 0: # current positon is target - reduce end time
                    end_time = curr_pos.arrival_time + 1
                    # end_time = curr_pos.arrival_time
                return end_time
            end_time = update_end_time(end_time)

            if self.locker.is_locked(curr_pos.vertex_id, self.agent_id, (curr_pos.arrival_time, end_time)):
                # continue
                #rollback to previous step with new waiting time
                if curr_pos.prev_way_id != -1:
                    def rollback_to_prev():
                        next_free_time = self.locker.next_free_time(curr_pos.vertex_id, self.agent_id, (curr_pos.arrival_time, end_time-curr_pos.wait_steps))
                        if next_free_time == curr_pos.arrival_time:
                            next_free_time += 1

                        prev_step = way_history[curr_pos.prev_way_id]

                        curr_pos.vertex_id = prev_step.vertex_idx
                        curr_pos.direction = prev_step.direction
                        curr_pos.wait_steps = prev_step.wait_steps + next_free_time - curr_pos.arrival_time
                        curr_pos.arrival_time = prev_step.arrival_time
                        curr_pos.prev_way_id = prev_step.prev_way_id
                        curr_pos.action = prev_step.action
                        curr_pos.dist_to_target = self.dist_to_target[prev_step.vertex_idx, prev_step.direction]
                        curr_pos.calc_d()
                        heapq.heappush(pq, curr_pos)
                        # next_q_el = curr_pos

                        # next_pos = AgentCurrentPosition(vertex_id=prev_step.vertex_idx,
                        #                                 direction=prev_step.direction,
                        #                                 arrival_time=prev_step.arrival_time,
                        #                                 wait_steps=prev_step.wait_steps + next_free_time - curr_pos.arrival_time,
                        #                                 prev_way_id=prev_step.prev_way_id,
                        #                                 action=prev_step.action,
                        #                                 dist_to_target=self.dist_to_target[prev_step.vertex_idx,prev_step.direction])
                        # # heapq.heapreplace(pq, next_pos)
                        # heapq.heappush(pq, next_pos)
                    rollback_to_prev()
                else:
                    # heapq.heappop(pq)
                    continue

                continue

            if self.check_and_update_visited(end_time, curr_pos.vertex_id, curr_pos.direction):
                # heapq.heappop(pq)
                continue


            # save information about current agent step
            def save_current_step():
                step_info = AgentWayStep(vertex_idx=curr_pos.vertex_id,
                                         direction=curr_pos.direction,
                                         arrival_time=curr_pos.arrival_time,
                                         departure_time=end_time,
                                         wait_steps=curr_pos.wait_steps,
                                         action=curr_pos.action,
                                         prev_way_id=curr_pos.prev_way_id)
                way_history.append(step_info)
            save_current_step()
            curr_way_id = len(way_history)-1

            if curr_pos.dist_to_target == 0: # current position is target - stop search
                best_way = curr_way_id
                break

            # check all out edges
            def check_out_edges():
                out_edges = self.graph.vertexes[curr_pos.vertex_id].out[curr_pos.direction]
                if len(out_edges) == 1:
                    curr_pos.vertex_id=out_edges[0].end_v
                    curr_pos.direction=out_edges[0].end_direction
                    curr_pos.arrival_time=end_time
                    curr_pos.wait_steps = 0
                    curr_pos.prev_way_id = curr_way_id
                    curr_pos.action = out_edges[0].action_type
                    curr_pos.dist_to_target=self.dist_to_target[out_edges[0].end_v, out_edges[0].end_direction]
                    curr_pos.calc_d()
                    heapq.heappush(pq, curr_pos)
                    return None
                    # return curr_pos
                else:
                    list(heapq.heappush(pq, AgentCurrentPosition(vertex_id=edge.end_v,
                                                            direction=edge.end_direction,
                                                            arrival_time=end_time,
                                                            wait_steps=0,
                                                            prev_way_id=curr_way_id,
                                                            action=edge.action_type,
                                                            dist_to_target=self.dist_to_target[edge.end_v, edge.end_direction]))
                         for edge in out_edges
                         if self.dist_to_target[edge.end_v, edge.end_direction] != -1 and
                         not(curr_pos.prev_way_id == -1 and first_fixed_action != None and first_fixed_action != edge.action_type))
                return None

                # if len(out_edges)==0:
                #     return None
                # r = list([AgentCurrentPosition(vertex_id=edge.end_v,
                #                               direction=edge.end_direction,
                #                               arrival_time=end_time,
                #                               wait_steps=0,
                #                               prev_way_id=curr_way_id,
                #                               action=edge.action_type,
                #                               dist_to_target=self.dist_to_target[edge.end_v, edge.end_direction])
                #          for edge in out_edges
                #          if self.dist_to_target[edge.end_v, edge.end_direction] != -1 and
                #          not (
                #                 curr_pos.prev_way_id == -1 and first_fixed_action != None and
                #                 first_fixed_action != edge.action_type
                #          )
                #          ])
                # min_index = np.argmin(r)
                # list(heapq.heappush(pq, el) for index, el in enumerate(r) if index != min_index)
                # curr_pos.vertex_id = out_edges[min_index].end_v
                # curr_pos.direction = out_edges[min_index].end_direction
                # curr_pos.arrival_time = end_time
                # curr_pos.wait_steps = 0
                # curr_pos.prev_way_id = curr_way_id
                # curr_pos.action = out_edges[min_index].action_type
                # curr_pos.dist_to_target = self.dist_to_target[out_edges[min_index].end_v, out_edges[min_index].end_direction]
                # return curr_pos


                # r = list([AgentCurrentPosition(vertex_id=edge.end_v,
                #                                direction=edge.end_direction,
                #                                arrival_time=end_time,
                #                                wait_steps=0,
                #                                prev_way_id=curr_way_id,
                #                                action=edge.action_type,
                #                                dist_to_target=self.dist_to_target[edge.end_v, edge.end_direction])
                #           for edge in out_edges
                #           if self.dist_to_target[edge.end_v, edge.end_direction] != -1 and
                #           not (
                #                  curr_pos.prev_way_id == -1 and first_fixed_action != None and
                #                  first_fixed_action != edge.action_type and out_edges_cnt > 1
                #           )
                #           ])
                # return r

            # check_out_edges()
            next_q_el = check_out_edges()

            # newl = check_out_edges()
            # if len(newl) == 1:
            #     next_q_el = newl[0]
            # else:
            #     list(heapq.heappush(pq, el) for el in newl)

            # newl = check_out_edges()
            # if len(newl)==1:
            #     heapq.heapreplace(pq, newl[0])
            # else:
            #     heapq.heappop(pq)
            #     list(heapq.heappush(pq, el) for el in newl)

        if best_way != -1:
            # get best way
            way = []

            def apply_best_way(best_way):
                while best_way != -1:
                    way.append(way_history[best_way])
                    best_way = way_history[best_way].prev_way_id

                self.selected_way = way

                # lock best way
                self.lock_selected_way()
            apply_best_way(best_way)

            # if self.agent_id==7:
            #     print(step_idx, agent.position, agent.speed_data['position_fraction'], self.locker.data[vertex_idx])

            # return first actions
            if start_action is not None:
                return start_action
            else:
                assert len(way) >= 2
                if way[-1].wait_steps:
                    return  RailEnvActions.STOP_MOVING
                return way[-2].action
        else:
            # no way to target

            if force_new_path and len(way):
                self.selected_way = way
                self.lock_selected_way()
                return self.act(agent, step_idx, force_new_path=False, repeat=True)

            if agent.status == RailAgentStatus.ACTIVE:
                pass
                # print('no way to target')
                # assert False
                # self.locker.lock()

            # TODO check what to do when no way to target
            # self.lock_current_position(agent, step_idx)
            return RailEnvActions.STOP_MOVING


        #
        # start_time = step_idx
        # end_time = step_idx + ticks_per_step
        #
        # if agent.speed_data['position_fraction'] != 0.0:
        #     end_time -= int(round(agent.speed_data['position_fraction']*ticks_per_step))
        #     start_action = RailEnvActions.DO_NOTHING
        #
        # if agent.status == RailAgentStatus.READY_TO_DEPART:
        #     end_time += 1
        #     start_action = RailEnvActions.MOVE_FORWARD
        #
        # if agent.malfunction_data['malfunction'] > 0:
        #     end_time += agent.malfunction_data['malfunction']
        #     start_action = RailEnvActions.DO_NOTHING
        #
        # return RailEnvActions.DO_NOTHING

    def lock_current_position(self, agent : EnvAgent, step_idx):
        if agent.status not in (RailAgentStatus.DONE, RailAgentStatus.DONE_REMOVED, RailAgentStatus.READY_TO_DEPART):
            ticks_per_step = int(round(1 / agent.speed_data['speed']))
            start_time = step_idx
            end_time = step_idx + ticks_per_step

            if agent.speed_data['position_fraction'] != 0.0:
                end_time -= int(round(min(agent.speed_data['position_fraction'], 1 - agent.speed_data['speed']) * ticks_per_step))

            end_time += agent.malfunction_data['malfunction']

            vertex_idx = self.graph._vertex_idx_from_point(agent.position)
            self.locker.lock(vertex_idx, self.agent_id, (start_time, end_time))
            self.locked_vertexes.append(vertex_idx)

    def unlock_path(self):
        self.locker.unlock_agent_with_list(self.agent_id, self.locked_vertexes)
        self.locked_vertexes = []

    def lock_selected_way(self):
        # self.unlock_path()
        for w in self.selected_way:
            self.locker.lock(w.vertex_idx, self.agent_id, (w.arrival_time, w.departure_time))
        self.locked_vertexes = [w.vertex_idx for w in self.selected_way]

    def get_cached_way(self):
        return self.selected_way

    def clear_path(self):
        self.selected_way = []
        self.locked_vertexes = []

    def set_max_visited(self, value):
        self.max_visited = value
