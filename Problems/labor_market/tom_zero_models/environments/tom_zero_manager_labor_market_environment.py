from Problems.labor_market.labor_market_environment import *
from scipy.stats import norm, gamma


class TomZeroWorkerWorkerModel:

    def __init__(self, labor_costs: float, distance: float):
        self.labor_costs = labor_costs
        self.distance = distance

    def action(self, state, offer):
        if offer.value / self.distance >= state.value:
            return True, 'accept'
        return False, 'reject'


class ToMZeroManagerLaborMarketEnvironment(LaborMarketEnvironment):

    def __init__(self, states, actions, budget: float,  fee: float, distance: float, model):
        self.budget = budget
        super().__init__(states, actions, fee, distance, model)

    def __repr__(self) -> str:
        return f'ToM manager environment with budget {self.budget}, fee {self.fee} and distance {self.distance}'

    def transition_function(self, state, actions, **kwargs):
        if isinstance(actions, AcceptAction) or state.is_terminal:
            return State(-1, True)
        if isinstance(actions, QuitAction):
            return State(0, True)
        if isinstance(actions, OfferAction) and actions.value / self.distance > state.value:
            return State(-1, True)
        feasible_states = self.states[self.states >= state.value]
        new_state = np.random.choice(feasible_states)
        return State(new_state, False)

    def observation_function(self, state, actions, next_state, **kwargs):
        if not next_state.is_terminal and isinstance(actions, OfferAction):
            observation, name = self.model.action(state, actions)
            return Observation(observation, name)
        else:
            return Observation(True, 'quit')

    def reward_function(self, state, actions, **kwargs):
        if state.is_terminal:
            return 0.0
        if isinstance(actions, AcceptAction):
            return 0.0
        if isinstance(actions, OfferAction):
            if actions.value / self.distance > state.value:
                return (self.budget - actions.value).item()
            else:
                return -self.fee
        if isinstance(actions, QuitAction):
            return 0.0


