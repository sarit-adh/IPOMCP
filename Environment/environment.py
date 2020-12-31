import abc


class Environment(abc.ABC):

    @abc.abstractmethod
    def step(self):
        pass

    @abc.abstractmethod
    def sample_action(self):
        pass

    @abc.abstractmethod
    def sample_state(self):
        pass

    @abc.abstractmethod
    def transition_function(self, action, state):
        pass

    @abc.abstractmethod
    def observation_function(self, state, action, next_state):
        pass

    @abc.abstractmethod
    def reward_function(self, action, next_state):
        pass