from Environment.problem import IpomdpProbelm
from Problems.labor_market.labor_market_objects import *
import numpy as np


class LaborMarketProblem(IpomdpProbelm):

    def __init__(self, states, actions, fee: float, distance: float, model) -> None:
        self.fee = fee
        self.distance = distance
        self.model = model
        full_actions = self._list_all_actions(actions)
        super().__init__(states, full_actions, None)

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return f'General labor market problem with fee {self.fee} and distance {self.distance}'

    @staticmethod
    def _set_states(states) -> object:
        s = [State(val) for val in states]
        return s

    @staticmethod
    def _set_actions(actions) -> object:
        a = [OfferAction(val) for val in actions]
        return a

    def _set_initial_state(self, state=None) -> State:
        """

        :param state: optional
        :return: State object
        """
        if state is None:
            initial_state = np.random.choice(self.states)
            initial_state = State(initial_state, 'reject', False)
            return initial_state
        else:
            return state

    def sample_state(self) -> State:
        state = self._set_initial_state()
        return state

    def transition_function(self, state, actions, **kwargs):
        if isinstance(actions, AcceptAction) or state.is_terminal:
            return State(-1, True)
        if isinstance(actions, QuitAction):
            return State(0, True)
        new_state = np.random.choice(self.states)
        return new_state

    def observation_function(self, state, actions, next_state, **kwargs):
        if not next_state.is_terminal and isinstance(actions, OfferAction):
            observation, name = self.model(state, actions)
            return Observation(observation, name)
        else:
            return Observation(-1, None)

    def reward_function(self, state, actions, **kwargs):
        pass

    def step(self, state, actions, **kwargs) -> object:
        """
        :rtype: object
        """
        next_state = self.transition_function(state, actions)
        observation = self.observation_function(state, actions, next_state)
        reward = self.reward_function(state, actions)
        return next_state, observation, reward

    def pomdp_step(self, actions):
        state = self.current_state
        next_state = self.transition_function(state, actions)
        observation = self.observation_function(state, actions, next_state)
        reward = self.reward_function(state, actions)
        return observation, reward

    def _list_all_actions(self, actions):
        quit_action = QuitAction()
        offers = [OfferAction(val) for val in actions]
        offers.append(quit_action)
        return offers

