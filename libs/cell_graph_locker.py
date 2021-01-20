import numpy as np

class CellGraphLocker:
    def __init__(self, graph):
        self.graph = graph
        self.data = []
        self.reset()


    def reset(self):
        vertexes = len(self.graph.vertexes)
        self.data = [[] for i in range(vertexes)]


    def lock(self, vertex_idx, agent_idx, duration):
        # assert not self.is_locked(vertex_idx, agent_idx, duration)

        if len(self.data[vertex_idx])==0:
            self.data[vertex_idx].append((duration, agent_idx))
            return

        # index = self.equal_or_greater_index(vertex_idx, duration[0])
        index = self.equal_or_greater_index_end(vertex_idx, duration[1])
        if index < len(self.data[vertex_idx]):
            curr_lock_info = self.data[vertex_idx][index]

            if (curr_lock_info[1] == agent_idx) and self._has_intersection(curr_lock_info[0], duration):
                assert (curr_lock_info[0][0] <= duration[0]) and (duration[1] <= curr_lock_info[0][1])
                return

            assert curr_lock_info[0][0] >= duration[1]

            self.data[vertex_idx].insert(index, (duration, agent_idx))

            # if (curr_lock_info[1]==agent_idx) and (curr_lock_info[0][1] == duration[1]) and (curr_lock_info[0][0] <= duration[0]):
            #     self.data[vertex_idx][index] = (duration, agent_idx)
            # else:
            #     self.data[vertex_idx].insert(index, (duration, agent_idx) )
        else:
            self.data[vertex_idx].append((duration, agent_idx))


    def is_locked(self, vertex_idx, agent_idx, duration):
        if len(self.data[vertex_idx])==0:
            return False

        new_lock = (duration, agent_idx)
        left_lock = None
        right_lock = None

        index = self.equal_or_greater_index(vertex_idx, duration[0])
        if index < len(self.data[vertex_idx]):
            if self.data[vertex_idx][index][0][0] == duration:
                return True

            right_lock = self.data[vertex_idx][index]

            if index>0:
                left_lock = self.data[vertex_idx][index - 1]
        else:
            left_lock = self.data[vertex_idx][index - 1]

        return self._has_conflict(left_lock, new_lock) or self._has_conflict(new_lock, right_lock)




        # index = self.equal_or_greater_index(vertex_idx, duration[0])
        # if index < len(self.data[vertex_idx]):
        #     lock_duration, lock_agent_idx = self.data[vertex_idx][index]
        #     if (lock_duration[0] < duration[1]) and (agent_idx != lock_agent_idx):
        #         return True
        #
        # if index > 0:
        #     lock_duration, lock_agent_idx = self.data[vertex_idx][index - 1]
        #     if (lock_duration[1] > duration[0]) and (agent_idx != lock_agent_idx):
        #         return True
        #
        # return False

    def _has_conflict(self, left, right):
        if (left is None) or (right is None):
            return False

        d1 = left[0]
        d2 = right[0]

        if left[1] > right[1]:
            d1 = (d1[0], d1[1] + 1)

        return self._has_intersection(d1, d2)


    def next_free_time(self, vertex_idx, agent_idx, duration):
        index = self.equal_or_greater_index_end(vertex_idx, duration[1])

        if index < len(self.data[vertex_idx]):
            # lock_duration = self.data[vertex_idx][index][0]
            # if self._has_intersection(duration, lock_duration):
            #     return lock_duration[1]
            if self._has_conflict((duration, agent_idx), self.data[vertex_idx][index]):
                return self.data[vertex_idx][index][0][1]

        if index > 0:
            # lock_duration = self.data[vertex_idx][index - 1][0]
            # if self._has_intersection(duration, lock_duration):
            #     return lock_duration[1]
            if self._has_conflict(self.data[vertex_idx][index-1], (duration, agent_idx)):
                return self.data[vertex_idx][index-1][0][1]

        # print('already free')
        return duration[0]

        # index = self.equal_or_greater_index(vertex_idx, duration[0])
        # if index < len(self.data[vertex_idx]):
        #     lock_duration, lock_agent_idx = self.data[vertex_idx][index]
        #     if (lock_duration[0] < duration[1]) and (agent_idx != lock_agent_idx):
        #         return True
        #
        # if index > 0:
        #     lock_duration, lock_agent_idx = self.data[vertex_idx][index - 1]
        #     if (lock_duration[1] > duration[0]) and (agent_idx != lock_agent_idx):
        #         return True
        #
        # return False


    def unlock(self, vertex_idx, agent_idx, duration):
        assert len(self.data[vertex_idx])

        index = self.equal_or_greater_index(vertex_idx, duration[0])
        assert (index >= 0) and (index < len(self.data[vertex_idx]))

        lock_duration, lock_agent_idx = self.data[vertex_idx][index]
        assert (lock_duration == duration) and (lock_agent_idx == agent_idx)

        self.data[vertex_idx].pop(index)


    def equal_or_greater_index(self, vertex_idx, start_time):
        # d = self.data[vertex_idx]
        #
        # if not len(d):
        #     return 0
        #
        # l = 0
        # r = len(d) - 1
        #
        # while l <= r:
        #     c = (l + r) // 2
        #
        #     lock_duration_start = d[c][0][0]
        #     if lock_duration_start == start_time:
        #         return c
        #     elif lock_duration_start < start_time:
        #         l = c + 1
        #     else:
        #         r = c - 1
        #
        # return max(l, r)


        #

        for i, (lock_duration, lock_agent_idx) in enumerate(self.data[vertex_idx]):
            if lock_duration[0] >= start_time:
                return i

        return len(self.data[vertex_idx])

    def equal_or_greater_index_end(self, vertex_idx, end_time):
        for i, (lock_duration, lock_agent_idx) in enumerate(self.data[vertex_idx]):
            if lock_duration[1] >= end_time:
                return i

        return len(self.data[vertex_idx])

    def _has_intersection(self, a, b):
        return not ((a[1] <= b[0]) or (b[1] <= a[0]))

    def unlock_agent(self, agent_id):
        for i in range(len(self.data)):
            for j in reversed(range(len(self.data[i]))):
                if self.data[i][j][1] == agent_id:
                    self.data[i].pop(j)

    def unlock_agent_with_list(self, agent_id, vertex_list):
        for i in vertex_list:
            for j in reversed(range(len(self.data[i]))):
                if self.data[i][j][1] == agent_id:
                    self.data[i].pop(j)

    def last_time_step(self, vertex_idx, agent_idx):
        if not len(self.data[vertex_idx]):
            return 0

        res = self.data[vertex_idx][-1][0][1]
        if self.data[vertex_idx][-1][1] != agent_idx:
            res += 1

        return res
