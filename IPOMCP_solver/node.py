import numpy as np
from Environment.objects import *
from collections import OrderedDict


class Node(object):

    def __init__(self, parent, history: str, name: str) -> None:
        self.parent = parent
        self.history = history
        self.children = OrderedDict()
        self.value_sum = 0
        self.times_visited = 1
        self.rewards = []
        self.name = name

    def __hash__(self):
        return hash(self.history)

    def __eq__(self, other) -> bool:
        if isinstance(other, Node):
            return self.history == other.history
        return False

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def is_root(self) -> bool:
        return self.parent is None

    def get_mean_value(self, max_value=1e4) -> float:
        return self.value_sum / self.times_visited if self.times_visited != 0 else max_value

    def add_child(self, child):
        if child not in self.children:
            self.children[child.name] = child


class ActionNode(Node):

    def __str__(self) -> str:
        return f'Action node with {self.times_visited} visits and {self.value_sum} value'

    def update_value(self, reward):
        self.rewards.append(reward)
        self.times_visited += 1
        self.value_sum += (reward - self.value_sum) / self.times_visited


class ObservationNode(Node):

    def __init__(self, parent, history: str, name: str, is_empty: bool = True) -> None:
        super().__init__(parent, history, name)
        self.particle_set = []
        self.is_empty = is_empty

    def add_particle(self, particle: State) -> None:
        self.particle_set.append(particle)

    def sample_from_particle_set(self):
        return np.random.choice(self.particle_set)

    def update_value(self):
        self.times_visited += 1

    def ucb_score(self, scale=10):
        pouct = np.array(
            [child.get_mean_value() + scale * np.sqrt(np.log(self.times_visited) / child.times_visited) for child in
             self.children.values()])
        pouct[np.isnan(pouct)] = 1e4
        children_list = list(self.children.values())
        return children_list[np.argmax(pouct).item()]

    def compute_q_values(self) -> dict:
        q_values = [child.get_mean_value() for child in self.children.values()]
        actions = list(self.children.keys())
        return dict(zip(actions, q_values))

    def get_q_max(self):
        q_values = self.compute_q_values()
        max_value = max(q_values.values())
        max_keys = [k for k, v in q_values.items() if v == max_value]
        return self.children[max_keys[0]], max_value
