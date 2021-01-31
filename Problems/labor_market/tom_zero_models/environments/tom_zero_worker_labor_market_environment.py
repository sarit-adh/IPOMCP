from Problems.labor_market.labor_market_problem import *
from scipy.stats import norm, gamma


class TomZeroWorkerManagerModel:

    def __init__(self, budget: float):
        self.budget = budget

    @staticmethod
    def action(state, offer):
        if offer.value < state.value:
            return True, 'accept'
        return False, 'reject'


class ToMZeroWorkerLaborMarketEnvironment(LaborMarketProblem):

    def __init__(self, states, actions, labor_cost: float,  fee: float, distance: float, model):
        self.labor_cost = labor_cost
        self.labor_costs = self._compute_labor_costs(labor_cost, distance)
        super().__init__(states, actions, fee, distance, model)

    def __repr__(self) -> str:
        return f'ToM worker environment with labor costs {self.labor_costs}, fee {self.fee} and distance {self.distance}'

    @staticmethod
    def _compute_labor_costs(labor_cost, distance) -> float:
        return labor_cost * distance

    def transition_function(self, state, actions, **kwargs):
        if isinstance(actions, AcceptAction):
            return State(state.value, 'I-accept', True)
        if state.is_terminal:
            return State(state.value, 'terminal', True)
        if isinstance(actions, QuitAction):
            return State(state.value, 'quit', True)
        if isinstance(actions, OfferAction) and actions.value <= state.value:
            return State(state.value, 'accept', True)
        feasible_states = self.states[self.states <= state.value]
        new_state = np.random.choice(feasible_states)
        return State(new_state, 'reject', False)

    def observation_function(self, state, actions, next_state, **kwargs):
        if not next_state.is_terminal and isinstance(actions, OfferAction):
            observation, name = self.model.action(state, actions)
            return Observation(observation, name)
        else:
            return Observation(True, 'terminal')

    def reward_function(self, state, actions, **kwargs):
        if state.is_terminal:
            return 0.0
        if isinstance(actions, AcceptAction):
            return 0.0
        if isinstance(actions, QuitAction):
            return 0.0
        if isinstance(actions, OfferAction):
            if actions.value <= state.value:
                return actions.value - self.labor_costs
            else:
                return -self.fee


def test_bid():
    states = norm(35, 5).rvs(1000)
    lc = gamma(5).rvs(1)
    distance = 7
    fee = 1.5
    model = TomZeroWorkerManagerModel(np.random.choice(states))
    worker = ToMZeroWorkerLaborMarketEnvironment(states, states, lc, fee, distance, model)
    worker.step(worker.initial_state, OfferAction(np.random.choice(worker.actions)))


if __name__ == '__main__':
    test_bid()
