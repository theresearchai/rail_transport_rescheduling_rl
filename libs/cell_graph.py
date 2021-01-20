import numpy as np
from collections import deque

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env import RailEnvActions

class Vertex:
    def __init__(self, y, x, idx):
        self.point = (y, x)
        self.idx = idx
        self.out = [[], [], [], []]
        self.in_edges = [[], [], [], []]


class Edge:
    def __init__(self, start_v, end_v, start_dir, end_dir, action_type):
        self.start_v = start_v
        self.end_v = end_v
        self.start_direction = start_dir
        self.end_direction = end_dir
        self.action_type = action_type


class CellGraph:
    def __init__(self, env : RailEnv):
        self.env = env
        self._build_graph()

    def _build_graph(self):
        width = self.env.width
        height = self.env.height

        self.vertex_idx = np.zeros((height, width), dtype=np.int)
        self.vertex_idx.fill(-1)

        self.vertexes = []

        for y in range(height):
            for x in range(width):
                if self._is_rail(y, x):
                    idx = len(self.vertexes)
                    self.vertexes.append(Vertex(y, x, idx))
                    self.vertex_idx[y, x] = idx

        # print('vertexes:', len(self.vertexes))
        edges_cnt = 0

        for v_idx, v in enumerate(self.vertexes):
            start_point = v.point
            for direction in range(4):
                directions = self._possible_directions(start_point, direction)
                # assert len(directions) <= 2

                for end_direction in directions:
                    next_point = self._next_point(start_point, end_direction)
                    end_v = self._vertex_idx_from_point(next_point)
                    action_type = self._action_from_directions(direction, end_direction)

                    e = Edge(v_idx, end_v, direction, end_direction, action_type)
                    v.out[direction].append(e)
                    self.vertexes[end_v].in_edges[end_direction].append(e)
                    edges_cnt += 1

        # print('edges_cnt', edges_cnt)




    def _is_rail(self, y, x):
        return self.env.rail.grid[y, x] != 0

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

    def _vertex_idx_from_point(self, point):
        assert (point[0] >= 0) and (point[0] < self.vertex_idx.shape[0])
        assert (point[1] >= 0) and (point[1] < self.vertex_idx.shape[1])

        return self.vertex_idx[point[0], point[1]]

    def position_from_vertexid(self, vertexid: int):
        return self.vertexes[vertexid].point

    def _action_from_directions(self, in_direction, new_direction):
        if in_direction==new_direction:
            return RailEnvActions.MOVE_FORWARD
        if (in_direction+1)%4 == new_direction:
            return RailEnvActions.MOVE_RIGHT
        elif (in_direction-1)%4 == new_direction:
            return RailEnvActions.MOVE_LEFT
        else:
            return RailEnvActions.MOVE_FORWARD

