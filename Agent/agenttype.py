from Agent.frame import *
from Agent.objects import *
import numpy as np


class AgentType(ABC):
    """
    This class implements the type (definition 1) - i.e., pompd + beliefs + optimality criteria
    """
    def __init__(self, frame: Frame, beliefs: Belief) -> None:
        """

        :type beliefs: object
        """
        self.frame = frame
        self.beliefs = beliefs

    def list_all_actions(self):
        return self.frame.pomdp.actions

    def list_all_states(self):
        return self.frame.pomdp.states

    def sample_states(self) -> State:
        state = np.random.choice(self.frame.pomdp.states)
        return state

    @abstractmethod
    def update_belief(self, action: Action, observation: Observation, **kwargs) -> None:
        self.beliefs.update_belief(action, observation)

    @abstractmethod
    def rollout_policy(self) -> Action:
        """
        This method computes the rollout policy for a given AgentType using a given planner
        """
        pass
