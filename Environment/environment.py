from abc import ABC, abstractmethod
from Environment.objects import *


class IPOMDPEnvironment(ABC):
    """This environment class represents the POMDP component of the IPOMDP -
    .. math::
    <S,A_i,\Omega_i, T_i, O_i, R_i>
    """

    def __init__(self, states, actions, observations) -> None:
        """
    :param states: dictionary of Interactive State objects
    :param actions: dictionary of Action objects
    :param observations: dictionary of Observations objects
    """
        self.states = states
        self.actions = actions
        self.observations = observations
        self.initial_state = self._set_initial_state()
        self.update_current_state(self.initial_state)

    def get_state_name(self, index) -> str:
        return self.states[index].name()

    def update_current_state(self, state) -> None:
        self.current_state = state

    def _get_current_state(self) -> State:
        return self.current_state

    @abstractmethod
    def _set_initial_state(self, **kwargs) -> State:
        pass

    @abstractmethod
    def transition_function(self, state, actions, **kwargs) -> State:
        pass

    @abstractmethod
    def observation_function(self, state, actions, next_state, **kwargs) -> Observation:
        pass

    @abstractmethod
    def reward_function(self, state, actions, **kwargs) -> float:
        pass

    @abstractmethod
    def step(self, state, actions, **kwargs):
        next_state = self.transition_function(state, actions)
        observations = self.observation_function(state, actions, next_state)
        rewards = self.reward_function(state, actions)
        self.update_current_state(next_state)
        return next_state, observations, rewards
