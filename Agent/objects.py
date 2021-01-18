from abc import ABC, abstractmethod
from Environment.objects import *


class Belief(ABC):

    def __init__(self, initial_belief) -> None:
        self.initial_belief = initial_belief
        self.current_belief = self.initial_belief

    @abstractmethod
    def update_belief(self, action, observation) -> None:
        """
        This function updates the belief distribution based on the last action played and the last observation received
        """
        pass

    @abstractmethod
    def get_current_belief(self):
        """
        This function returns the current belief distribution
        """
        pass

    @abstractmethod
    def mpe(self) -> State:
        """
        This function returns the Maximum a posteriori estimation of the state
        """
        pass

    @abstractmethod
    def plot_belief(self):
        """
        This function returns a plot of the current belief distribution
        """
        pass

    def reset_belief(self):
        self.current_belief = self.initial_belief
