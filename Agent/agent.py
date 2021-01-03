from Agent.type import *


class Agent(ABC):

    def __init__(self, agent_type: Type, planner) -> None:
        self.agent_type = agent_type
        self.planner = planner
        self.history = None

    def update_reward(self, reward: float, t: int) -> None:
        self.agent_type.frame.oc.update_cumulative_reward(reward, t)

    def get_current_reward(self) -> float:
        return self.agent_type.frame.oc.get_current_reward()

    @abstractmethod
    def compute_optimal_policy(self):
        """
        This method computes the optimal policy for a given Type using a given planner
        """
        pass

    @abstractmethod
    def execute_action(self) -> (Observation, float):
        """
        This method selects an action from |OPT_i| and gets observation and reward
        """
        pass

    @abstractmethod
    def update_history(self, action: Action, observation: Observation) -> None:
        """
        This method updates the agent's history
        """
        pass

