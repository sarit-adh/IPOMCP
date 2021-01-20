from typing import Any
from Agent.agent import *
from IPOMCP_solver.pomcp import POMCP
from IPOMCP_solver.node import *
from collections import OrderedDict


class ToMZeroLaborMarketAgent(Agent):

    def __init__(self, planning_horizon: int, agent_type: AgentType, planner) -> None:
        self.planning_horizon = planning_horizon
        self.observations = []
        self.actions = []
        self.action_nodes = OrderedDict()
        self.current_node = None
        super().__init__(agent_type, planner)

    @property
    def compute_optimal_policy(self) -> tuple[Any, Any]:
        if isinstance(self.planner, POMCP):
            if not bool(self.action_nodes):
                root_node = ObservationNode(None, '', '')
            else:
                # next(reversed(od.values()))
                last_action_node = self.action_nodes[next(reversed(self.action_nodes))]
                try:
                    root_node = last_action_node.children[self.observations[len(self.observations)-1].name]
                except KeyError:
                    root_node = ObservationNode(None, '', '')
            br_node = self.planner.search(root_node)
            self.action_nodes[hash(br_node)] = br_node
            action = [a for a in self.agent_type.frame.pomdp.actions if a.name == br_node.name][0]
            return action, br_node.value_sum

    @property
    def execute_action(self) -> (Observation, float):
        action, _ = self.compute_optimal_policy
        self.actions.append(action)
        self.planning_horizon -= 1
        new_state, observation, reward = \
            self.agent_type.frame.pomdp.step(self.agent_type.frame.pomdp.current_state, action)
        self.observations.append(observation)
        self.agent_type.frame.pomdp.update_current_state(new_state)
        self.update_history(action, observation)
        self.update_reward(reward, self.planning_horizon)
        return observation, reward

    def update_history(self, action: Action, observation: Observation) -> None:
        self.agent_type.beliefs.update_belief(action, observation)
