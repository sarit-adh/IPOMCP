from abc import ABC, abstractmethod


class TransitionFunction(ABC):

    @abstractmethod
    def state_probability(self, state, action, next_state, **kwargs):
        """

        :param next_state: State object, representing the next state
        :param state: State object, representing the current state
        :param action: Action object, representing the current action
        :param kwargs: Additional parameters required
        :return: probability - float [0,1] the probability of :math:`P(s'|s,a)`
        """
        pass

    @abstractmethod
    def sample(self, state, action, **kwargs):
        """

        :param state: State object, representing the current state
        :param action: Action object, representing the current action
        :param kwargs: Additional parameters required
        :return: State object, representing the next state
        """
        pass

    @abstractmethod
    def get_transition_probability(self, state, action, **kwargs):
        """

        :param state: State object, representing the current state
        :param action: Action object, representing the current action
        :param kwargs: Additional parameters required
        :return: array: the conditional distribution :math: `P(s'|s,a)`
        """
        pass


class ObservationFunction(ABC):

    @abstractmethod
    def observation_probability(self, next_state, action, **kwargs):
        """

        :param next_state: State object, representing the next state
        :param action: Action object, representing the current action
        :param kwargs: Additional parameters required
        :return: probability - float [0,1] the probability of :math:`P(o|s',a)`
        """
        pass

    @abstractmethod
    def sample(self, next_state, action, **kwargs):
        """

        :param next_state: State object, representing the current state
        :param action: Action object, representing the current action
        :param kwargs: Additional parameters required
        :return: Observation object, representing a plausible observation
        """
        pass

    @abstractmethod
    def get_observation_probability(self, next_state, action, **kwargs):
        """

        :param next_state: State object, representing the current state
        :param action: Action object, representing the current action
        :param kwargs: Additional parameters required
        :return: array: the conditional distribution :math: `P(o|s,a)`
        """
        pass


class RewardFunction(ABC):

    @abstractmethod
    def reward_probability(self, state, action, next_state, **kwargs):
        """

        :param state: State object, representing the current state
        :param action: Action object, representing the current action
        :param next_state: State object, representing the next state
        :param kwargs: Additional parameters required
        :return: probability - float [0,1] the probability of :math:`P(r|s,a,s')`
        """
        pass

    @abstractmethod
    def sample(self, state, action, next_state, **kwargs):
        """

        :param state: State object, representing the current state
        :param action: Action object, representing the current action
        :param next_state: State object, representing the current state
        :param kwargs: Additional parameters required
        :return: float - representing a plausible reward
        """
        pass

    @abstractmethod
    def get_reward_probability(self, state, action, next_state, **kwargs):
        """

        :param state: State object, representing the current state
        :param action: Action object, representing the current action
        :param next_state: State object, representing the current state
        :param kwargs: Additional parameters required
        :return: array: the conditional distribution :math: `P(r|s,a,s')`
        """
        pass
# TODO(Nitay/Sarit) - add the abstract class for the interactive state
# The POMCP/IPOMCP algorithm requires a black-box model of the environment


class BlackBoxModel(ABC):

    @abstractmethod
    def sample_state(self, state, action, **kwargs):
        """
        :param state: State object, representing the current state
        :param action: Action object, representing the current action
        :param kwargs: Additional parameters required
        :return: State object, representing the next state
        """
        pass

    @abstractmethod
    def sample_observation(self, state, action, new_state, **kwargs):
        """

        :param state: State object, representing the current state
        :param action: Action object, representing the current action
        :param new_state: State object, representing the next state
        :param kwargs: Additional parameters required
        :return: Observation state
        """
        pass

    @abstractmethod
    def sample_reward(self, state, action, new_state, **kwargs):
        """

        :param state: State object, representing the current state
        :param action: Action object, representing the current action
        :param new_state: State object, representing the next state
        :param kwargs: Additional parameters required
        :return: float, reward
        """
        pass

    @abstractmethod
    def step(self, state, action,  **kwargs):
        """

        :param state: State object, representing the current state
        :param action: Action object, representing the current action
        :param kwargs: Additional parameters required
        :return: State object, Observation object, Reward float
        """
        pass
