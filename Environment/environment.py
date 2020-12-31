import random
from abc import ABC, abstractmethod


class MDPEnvironment(ABC):
    """
	The environment class represents the MDP - tracking the states and rewards of the problem
	"""

    def __init__(self, states, actions):
        """

        :param states: dictionary of State objects
        :param actions: dictionary of Action objects
        """
        self.initial_state = self._set_initial_state()
        self.states = states
        self.actions = actions
        self.current_state = self._set_current_state(self.initial_state)

    def get_state_name(self, index):
        return self.states[index].name()

    def get_action_name(self, index):
        return self.actions[index].name()

    @abstractmethod
    def _set_initial_state(self):
        initial_state = random.choice(self.states)
        return initial_state

    @abstractmethod
    def _set_current_state(self, state):
        self.current_state = state

    @abstractmethod
    def transition_function(self, state, action):
        pass

    @abstractmethod
    def reward_function(self, state, action, next_state):
        pass

    @abstractmethod
    def step(self, state, action):
        next_state = self.transition_function(state, action)
        reward = self.reward_function(state, action, next_state)
        self._set_current_state(next_state)
        return next_state, reward


def _set_current_state(state):
    return state


class IPOMDPEnvironment(ABC):
    """
	The environment class represents the IPOMDP
	"""

    def __init__(self, states, actions, observations):
        """

    :param states: dictionary of Interactive State objects
    :param actions: dictionary of Action objects
    :param observations: dictionary of Observations objects
    """
        self.states = states
        self.actions = actions
        self.observations = observations
        self.initial_state = self._set_initial_state()
        self.current_state = _set_current_state(self.initial_state)

    def get_state_name(self, index):
        return self.states[index].name()

    def get_action_name(self, index):
        return self.actions[index].name()

    @abstractmethod
    def _set_initial_state(self):
        initial_state = random.choice(self.states)
        return initial_state

    @abstractmethod
    def transition_function(self, state, actions, **kwargs):
        pass

    @abstractmethod
    def observation_function(self, state, actions, next_state, **kwargs):
        pass

    @abstractmethod
    def reward_function(self, state, actions, **kwargs):
        pass

    @abstractmethod
    def step(self, state, actions):
        next_state = self.transition_function(state, actions)
        observations = self.observation_function(state, actions, next_state)
        rewards = self.reward_function(state, actions, next_state)
        _set_current_state(next_state)
        return next_state, observations, rewards
