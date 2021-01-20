import numpy as np
from collections import deque
import time

from flatland.envs.rail_env import RailEnv


class Vertice:
    def __init__(self, start=None, start_direction=None, first_move_direction=None, end=None, end_direction=None, list=None):
        self.start = start
        self.start_direction = start_direction
        self.first_move_direction = first_move_direction
        self.end = end
        self.end_direction = end_direction
        self.list = [] if list is None else list

    def __repr__(self):
        return str(self.start) + "|" + str(self.start_direction) + "|" + str(self.first_move_direction) + "->" +\
               str(self.end) + "|" + str(self.end_direction) + " || " + str(self.list)


class BuildGraphFromEnvironment:
    def __init__(self, env : RailEnv):
        self.env = env
        self.vs = []
        self.es = {}
        self.rev_es = {}
        self._build_graph()

    def _transition_to_coordinate(self, transition, coords):
        if transition==0:
            return (coords[0] - 1, coords[1])
        elif transition==1:
            return (coords[0] - 1, coords[1])

    def _next_point(self, point, direction):
        if direction==0:
            return (point[0]-1, point[1])
        elif direction==1:
            return (point[0], point[1]+1)
        elif direction==2:
            return (point[0]+1, point[1])
        else:
            return (point[0], point[1]-1)

    def _possible_directions(self, point, in_direction):
        return np.flatnonzero(self.env.rail.get_transitions(point[0], point[1], in_direction))


    def _make_vertice(self, point, in_direction, start_out_direction, stop_points):
        v = Vertice(start=point, start_direction=in_direction, first_move_direction=start_out_direction)

        ds = [start_out_direction]
        while len(ds) == 1:
            v.list.append((point, ds[0]))
            point = self._next_point(point, ds[0])

            oldds = ds
            ds = self._possible_directions(point, ds[0])

            if point in stop_points:
                break

        v.end = point
        v.list.append((point, -1))

        v.end_direction = oldds[0]

        return v

    def _max_transitions(self, y, x):
        res = 0
        for d in range(4):
            next_directions = self._possible_directions((y, x), d)
            res = max(res, len(next_directions))
        return res


    def _build_graph(self):
        print("Start graph building")
        timev = time.time()

        vs = []
        start_position_2_v = {}
        stop_points = {}
        # vs2index = {}
        # point_in_edges = {}
        es = {}
        q = deque()

        # self.env.
        #
        # for i in range(self.env.height):
        #     for j in range(self.env.width):
        #         if self.env.rail.grid[i][j]>0:
        #             vs2index[(i,j)] = len(vs)
        #             vs.append((i, j))

        for agent in self.env.agents:
            points = [(agent.target, i, -1) for i in range(4)] + [(agent.initial_position, agent.direction, -1)]
            for point, d, from_v in points:
                q.append((point, d, from_v))
                stop_points[point] = True

        # for y in range(self.env.height):
        #     for x in range(self.env.width):
        #         max_directions = self._max_transitions(y, x)
        #         if max_directions>1:
        #             stop_points[(y, x)] = True

        visited = {}
        while len(q):
            point, d, source = q.popleft()
            if source !=-1 and source not in es:
                es[source] = []

            d_outs = self._possible_directions(point, d)
            for d_out in d_outs:
                if (point, d_out) in visited:
                    if source != -1:
                        es[source].append(visited[(point, d_out)])
                    continue

                v = self._make_vertice(point, d, d_out, stop_points)
                vidx = len(vs)
                visited[(point, d_out)] = vidx

                vs.append(v)
                if source !=-1:
                    es[source].append(vidx)

                q.append((v.end, v.end_direction, vidx))
                stop_points[v.end] = True

        self.vs = vs
        self.es = es

        self.rev_es = self._reversed_es()

        print("Vs: ", len(vs))
        [print(v) for v in vs]
        print("Es: ")
        [print(k, "->", e) for k, e in es.items()]
        print("total edges:", np.sum([len(e) for _, e in es.items()]))
        print("Time:", time.time() - timev)

    def _reversed_es(self):
        rev_es = {i: [] for i in range(len(self.vs))}
        for f, t in self.es.items():
            for tel in t:
                rev_es[tel].append(f)
        return rev_es

    def calc_distances(self, target_point):
        import heapq
        q = [(len(v.list)-1, i) for i, v in enumerate(self.vs) if v.end==target_point]
        heapq.heapify(q)

        distances = {i: np.Inf for i in range(len(self.vs))}
        while q:
            d, i = heapq.heappop(q)
            if np.isfinite(distances[i]):
                continue

            distances[i] = d
            for el in self.rev_es[i]:
                heapq.heappush(q, (d + len(self.vs[el].list)-1, el))

        return distances


class GraphPathsLocker:
    def __init__(self, height, width):
        self.locks = np.zeros((height, width), dtype=np.int16)

    def _change_vertice_status(self, v: Vertice, lock_id=0):
        for el in v.list[:-1]:
            self.locks[el[0][0], el[0][1]] = lock_id+1

    def lock_vertice(self, v: Vertice, lock_id):
        self._change_vertice_status(v, lock_id)

    def unlock_vertice(self, v: Vertice, lock_id):
        for el in v.list[:-1]:
            if self.locks[el[0][0], el[0][1]]==lock_id+1:
                self.locks[el[0][0], el[0][1]] = 0


    def unlock_position(self, row, col):
        self.locks[row][col] = 0

    def is_locked(self, v: Vertice, lock_id):
        return np.any([self.locks[el[0][0], el[0][1]]>0 and self.locks[el[0][0], el[0][1]]!=lock_id+1 for el in v.list])
