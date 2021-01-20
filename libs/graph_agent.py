import numpy as np
from typing import List
from copy import deepcopy

from flatland.envs.rail_env import RailEnvActions, EnvAgent, RailAgentStatus

from libs.graph import Vertice, GraphPathsLocker

class GraphAgent:
    def __init__(self, vs : List[Vertice], es, rev_es, distances,
                 start_position, start_direction,
                 target_position,
                 locker : GraphPathsLocker,
                 env=None, agent_id=None):
        self.vs = vs
        self.es = es
        self.rev_es = rev_es
        self.distances = distances
        self.target = target_position
        self.path = []
        self.path_position = -1
        self.vertice = -1
        self.next_vertice = -1
        self.locker = locker
        self.env = env
        self.agent_id = agent_id
        if self.env:
            self.env.dev_pred_dict[self.agent_id] = []

        self.current_position = (-1, -1)
        self.current_action = -1

        self.start_position = start_position
        self.start_direction = start_direction
        self.finished = False

        self._simplify_graph_by_distances()
        # self._start_movement()


    def _start_movement(self):
        self.vertice = self._choose_new_vertice(self._find_verticies(self.start_position, self.start_direction))
        if self.vertice!=-1:
            self._start_vertice_trip(self.vertice)
            self.locker.lock_vertice(self.vs[self.vertice], self.agent_id)
            self.path_position = -1

    def act(self, agent : EnvAgent, recursive_call=False):
        if self.vertice == -1:
            self._start_movement()
            if self.vertice == -1:
                return RailEnvActions.DO_NOTHING

        if agent.status == RailAgentStatus.DONE or agent.status == RailAgentStatus.DONE_REMOVED:
            if self.vertice != -1:
                self.locker.unlock_vertice(self.vs[self.vertice], self.agent_id)
                self.locker.unlock_position(agent.target[0], agent.target[1])
                self.vertice = -1
                self.finished = True
            return RailEnvActions.STOP_MOVING

        if agent.position is None and agent.status==RailAgentStatus.READY_TO_DEPART and self.vertice!=-1:
            return RailEnvActions.MOVE_FORWARD

        if self.current_position == agent.position:
            if self.current_action!=-1:
                return self.current_action
        else:
            self.locker.unlock_position(self.current_position[0], self.current_position[1])
            self.current_position = agent.position
            self.path_position += 1

        if self.path_position==len(self.path):
            self._start_vertice_trip(self.next_vertice)
        if self.path_position==len(self.path)-1:
            self.next_vertice = self._choose_new_vertice(self.es[self.vertice])
            if self.next_vertice==-1:
                v = list(filter(lambda v: v!=self.vertice, self._find_verticies(agent.position, agent.direction)))
                v = self._choose_new_vertice(v)
                if v!=-1 and not recursive_call:
                    self._start_vertice_trip(v)
                    return self.act(agent, recursive_call=True)

                self.current_action = -1
                return RailEnvActions.STOP_MOVING
            self.locker.lock_vertice(self.vs[self.next_vertice], self.agent_id)

        assert self.current_position==self.path[self.path_position][0]

        new_direction = self.path[self.path_position][1]
        self.current_action = self._action_from_directions(agent.direction, new_direction)
        return self.current_action


    def _action_from_directions(self, in_direction, new_direction):
        if in_direction==new_direction:
            return RailEnvActions.MOVE_FORWARD
        if (in_direction+1)%4 == new_direction:
            return RailEnvActions.MOVE_RIGHT
        elif (in_direction-1)%4 == new_direction:
            return RailEnvActions.MOVE_LEFT
        else:
            return RailEnvActions.MOVE_FORWARD


    def _find_verticies(self, start_position, start_direction):
        res = []
        for idx, v in enumerate(self.vs):
            if v.start==start_position and v.start_direction==start_direction:
                if not self.locker.is_locked(v, self.agent_id):
                    res.append(idx)
        return res

    def _choose_new_vertice(self, possible):
        p = list(filter(lambda id: not self.locker.is_locked(self.vs[id], self.agent_id) and np.isfinite(self.distances[id]), possible))
        if len(p)==0:
            return -1
        idx = np.argmin([self.distances[idx] for idx in p])
        return p[idx]

    def _start_vertice_trip(self, idx):
        assert idx>=0

        self.vertice = idx
        self.path = self.vs[idx].list[:-1]
        if self.target == self.vs[idx].list[-1][0]:
            self.path = self.vs[idx].list
        self.path_position = 0
        self.current_action = -1

        if self.env is not None:
            self.env.dev_pred_dict[self.agent_id] = [el[0] for el in self.path]

    def _merge_verticies(self, first_v, second_v):
        res = deepcopy(first_v)
        res.end = second_v.end
        res.end_direction = second_v.end_direction
        res.list = res.list[:-1] + second_v.list
        return res

    def _simplify_graph_by_distances(self):
        vs = deepcopy(self.vs)
        es = deepcopy(self.es)

        es = {k : list(filter(lambda e: np.isfinite(e), edges)) for k, edges in es.items()}

        for i in range(len(vs)):
            while vs[i].end != self.target and len(es[i])==1 and len(self.rev_es[es[i][0]])==1:
                nid = es[i][0]
                vs[i] = self._merge_verticies(vs[i], vs[nid])
                es[i] = es[nid]

        self.vs = vs
        self.es = es

    def min_distance(self):
        vs = self._find_verticies(self.start_position, self.start_direction)
        return np.min([self.distances[v] for v in vs])

    def is_finished(self):
        return self.finished


class AgentsList:
    def __init__(self, controllers: List[GraphAgent], agents: List[EnvAgent], simultaneous = 15):
        self.controllers = controllers
        self.agents = agents
        self.cnt = simultaneous

        self.full_list = list(range(len(self.agents)))
        self.activel = []
        self._sort_by_time()
        print(self.full_list)


    def active(self):
        self._update_active_list()
        return self.activel

    def not_started(self):
        return len(self.full_list)

    def _sort_by_time(self):
        self.full_list = sorted(self.full_list, key=lambda idx: self.controllers[idx].min_distance()*self.agents[idx].speed_data['speed'])


    def _update_active_list(self):
        self.activel = list(filter(lambda idx: not self.controllers[idx].is_finished(), self.activel))
        n = self.cnt - len(self.activel)
        self.activel.extend(self.full_list[:n])
        self.full_list = self.full_list[n:]




