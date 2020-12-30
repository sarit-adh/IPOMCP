import numpy as np


class Node:

    def __init__(self, history, parent):

        # self.belief_dict = belief_dict
        self.history = history
        # self.horizon = horizon
        self.parent = parent
        # self.action = action
        # self.observation = observation
        self.children = set()
        self.value_sum = 0
        self.times_visited = 0
        # self.immediate_reward = immediate_reward
        self.particle_set = []

    def is_leaf(self):
        return len(self.children) == 0

    def is_root(self):
        return self.parent is None

    def get_mean_value(self):
        return self.value_sum / self.times_visited if self.times_visited != 0 else 0

    def ucb_score(self, scale=10, max_value=1e100):
        # TODO : modify for POMDP
        if self.times_visited == 0:
            return max_value
        u = np.sqrt(2 * np.log(self.parent.times_visited) / self.times_visited)
        return self.get_mean_value() + scale * u

    def propagate(self, child_value):
        # TODO : modify for POMDP
        new_value = self.immediate_reward + child_value

        self.value_sum += new_value
        self.times_visited += 1

        if not self.is_root():
            self.parent.propagate(new_value)

    def safe_delete(self):
        del self.parent
        for child in self.children:
            child.safe_delete()

    def __str__(self):
        return str(self.history) + " " + str(self.value_sum) + " " + str(self.times_visited) + " " + str(
            self.particle_set) + " [" + str(self.parent) + "]"
