from abc import ABC

from Environment.problem import *


class Environment(ABC):
    """
    This class represents the physical arena where agents meet and interact with each other
    """
    def __init__(self, number_of_trails, input_file):
        self.input_file = input_file
        self.number_of_trails = number_of_trails

    @abstractmethod
    def simulate_environment(self, agent_types_list: list, starting_agent: str):
        """

        :param starting_agent:
        :param agent_types_list: list containing all the agents' types
        :return:
        """
        pass


