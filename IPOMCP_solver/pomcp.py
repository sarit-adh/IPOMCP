from typing import Any

from Agent.agenttype import *
from IPOMCP_solver.node import *
from collections import OrderedDict


class POMCP:

    def __init__(self, environment: AgentType, gamma=1, epsilon=0.01, horizon=3):
        self.environment = environment
        self.gamma = gamma
        self.epsilon = epsilon
        self.horizon = horizon
        self.tree = OrderedDict()

    def search(self, root_node: ObservationNode, time_out=10000):
        for i in range(time_out):
            if root_node.parent is None:
                # Sample from the environment
                sampled_state = self.environment.sample_states()
            else:
                # Sample from the existing particle set
                sampled_state = root_node.sample_from_particle_set()
            self.simulate(sampled_state, root_node, 0)
            if root_node.history not in self.tree:
                self.tree[root_node.history] = root_node
        return root_node.get_q_max()

    def simulate(self, s: State, h: ObservationNode, depth: int) -> float:
        if self.gamma ** depth < self.epsilon or depth == self.horizon:
            return 0
        if not h.children:
            for action in self.environment.list_all_actions():
                new_node = ActionNode(h, h.history + "--" + action.name, action.name)
                h.add_child(new_node)
                self.tree[new_node.history] = new_node
            # Following BAMCP algorithm
            action = self.environment.rollout_policy()
            action_node = self.tree[h.history + "--" + str(action.name)]
            new_state, observation, reward = self.environment.frame.pomdp.step(s, action)
            new_history_node = ObservationNode(action_node, action_node.history + "--" + str(observation.name), str(observation.name), True)
            action_node.add_child(new_history_node)
            R = reward + self.gamma * self.rollout(new_state, new_history_node, depth + 1)
            self.tree[new_history_node.history] = new_history_node
            h.update_value()
            action_node.update_value(R)
            return R
        action = h.ucb_score()
        action_node = self.tree[h.history + "--" + action.name]
        new_state, observation, reward = self.environment.frame.pomdp.step(s, action)
        new_history_node = ObservationNode(action_node, action_node.history + "--" + str(observation.name),
                                           str(observation.name), True)
        action_node.add_child(new_history_node)
        r = reward + self.gamma * self.simulate(new_state, new_history_node, depth + 1)
        self.tree[new_history_node.history] = new_history_node
        h.add_particle(new_state)
        h.update_value()
        action_node.update_value(r)
        return r

    def rollout(self, s, h, depth) -> float:
        if self.gamma ** depth < self.epsilon or depth == self.horizon:
            return 0
        action = self.environment.rollout_policy()
        new_state, observation, reward = self.environment.frame.pomdp.step(s, action)
        action_node = ActionNode(h, h.history + action.name, action.name)
        new_history_node = ObservationNode(action_node, action_node.history + "--" + str(observation.name), str(observation.name), True)
        r = reward + self.gamma * self.rollout(new_state, new_history_node, depth + 1)
        return r











