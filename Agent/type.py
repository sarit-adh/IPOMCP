from Agent.frame import *
from Agent.functions import *


class Type(ABC):
    """
    This class implements the type (definition 1) - i.e., pompd + beliefs + optimality criteria
    """
    def __init__(self, frame: Frame, beliefs: BeliefFunction) -> None:
        """

        :type beliefs: object
        """
        self.frame = frame
        self.beliefs = beliefs

    @abstractmethod
    def update_belief(self, action: Action, observation: Observation, **kwargs) -> None:
        self.beliefs.update_belief(action, observation)
