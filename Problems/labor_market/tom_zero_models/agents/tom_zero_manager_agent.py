from Agent.agent import *
import matplotlib.pyplot as plt
from scipy.stats import gamma
from Problems.labor_market.labor_market_environment import *
from IPOMCP_solver.pomcp import POMCP
from IPOMCP_solver.node import *


def left_truncated_gamma_distribution(base_measure, lower_limit, states):
    lc = base_measure.cdf(lower_limit)
    new_pdf = base_measure.pdf(states) / (1-lc)
    return new_pdf / new_pdf.sum()


class ToMZeroManagerLaborMarketBelief(Belief):

    def __init__(self, a, states):
        self.base_measure = gamma(a=a)
        self.states = states
        super().__init__(self.base_measure.pdf(states) / self.base_measure.pdf(states).sum())

    def update_belief(self, action, observation) -> None:
        if observation.value == 'accept' or isinstance(action, QuitAction) or isinstance(action, AcceptAction):
            new_belief = self.base_measure
        else:
            new_belief = left_truncated_gamma_distribution(self.base_measure, action, self.states)
        self.current_belief = new_belief

    def get_current_belief(self):
        return self.current_belief

    def mpe(self) -> State:
        return State(np.argmax(self.current_belief).item(), False)

    def plot_belief(self):
        plt.hist(self.current_belief)


class ToMZeroManagerLaborMarketType(AgentType):

    def sample_states(self) -> State:
        state = self.frame.pomdp.sample_state()
        return state

    def update_belief(self, action: Action, observation: Observation, **kwargs) -> None:
        self.beliefs.update_belief(action, observation)

    def rollout_policy(self):
        action = np.random.choice(self.frame.pomdp.actions)
        return action


class ToMZeroManagerLaborMarketAgent(Agent):

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
            action = [a for a in self.agent_type.frame.pomdp.actions if a.name == br_node.name][0]
            return action

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