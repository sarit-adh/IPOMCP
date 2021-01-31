from Agent.agent import *
import matplotlib.pyplot as plt
from scipy.stats import gamma
from Problems.labor_market.labor_market_problem import *
from IPOMCP_solver.pomcp import POMCP
from IPOMCP_solver.node import *
from Problems.labor_market.tom_zero_models.agents.tom_zero_agent import ToMZeroLaborMarketAgent

def left_truncated_gamma_distribution(base_measure, lower_limit, states):
    lc = base_measure.cdf(lower_limit)
    new_pdf = base_measure.pdf(states) / (1-lc)
    return new_pdf / new_pdf.sum()


class ToMZeroManagerLaborMarketBelief(Belief):

    def __init__(self, a, states, distance):
        self.base_measure = gamma(a=a)
        self.states = states
        self.distance = distance
        super().__init__(self.base_measure.pdf(states) / self.base_measure.pdf(states).sum())

    def update_belief(self, action, observation) -> None:
        if observation.value == 'accept' or isinstance(action, QuitAction) or isinstance(action, AcceptAction):
            new_belief = self.base_measure
        else:
            new_belief = left_truncated_gamma_distribution(self.base_measure, action.value / self.distance, self.states)
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


class ToMZeroManager:

    def __init__(self, manager_agent: ToMZeroLaborMarketAgent, manager_type: ToMZeroManagerLaborMarketType,
                 planning_horizon=5):
        self.manager_agent = manager_agent
        self.manager_type = manager_type
        self.planning_horizon = planning_horizon

    def best_response(self):
        tom_zero_manager_pomcp = POMCP(self.manager_type, horizon=self.planning_horizon,
                                       exploration_bonus=self.manager_agent.exploration_bonus)
        self.manager_agent.planner = tom_zero_manager_pomcp
        action, q_value = self.manager_agent.compute_optimal_policy
        return action, q_value
