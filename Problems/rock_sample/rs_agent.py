from Agent.agent import *
from Problems.rock_sample.rs_objects import *
from IPOMCP_solver.pomcp import POMCP
from IPOMCP_solver.node import *


class RockSampleBelief(Belief):

    def update_belief(self, action, observation) -> None:
        if not isinstance(action, CheckAction):
            new_belief = self.current_belief
        else:
            new_belief = self.current_belief
        self.current_belief = new_belief

    def get_current_belief(self):
        return self.current_belief

    def mpe(self) -> State:
        pass

    def plot_belief(self):
        pass


class RockSampleType(AgentType):

    def update_belief(self, action: Action, observation: Observation, **kwargs) -> None:
        self.beliefs.update_belief(action, observation)

    def rollout_policy(self):
        action = np.random.choice(self.frame.pomdp.actions)
        return action


class RockSampleAgent(Agent):

    def __init__(self, planning_horizon: int, agent_type: AgentType, planner) -> None:
        self.planning_horizon = planning_horizon
        self.observations = [None]
        self.actions = []
        self.current_node = None
        super().__init__(agent_type, planner)

    @property
    def compute_optimal_policy(self) -> Action:
        if isinstance(self.planner, POMCP):
            if self.current_node is None:
                root_node = ObservationNode(None, '', '')
            else:
                root_node = self.current_node.children[self.observations[len(self.observations)-1].name]
            br_node, br_value, = self.planner.search(root_node)
            self.current_node = br_node
            return Action(br_node.name)
        if self.planning_horizon == 0:
            return np.random.choice(np.array([Action('open-left'), Action('open-right')]))
        else:
            p = self.agent_type.beliefs.get_current_belief()
            if np.dot(p.T, np.array([-100, 10])) > -1 or np.dot(p.T, np.array([10, -100])) > -1:
                return np.array([Action('open-left'), Action('open-right')])[np.argmin(p)]
            return Action('listen')

    @property
    def execute_action(self) -> (Observation, float):
        action = self.compute_optimal_policy
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
