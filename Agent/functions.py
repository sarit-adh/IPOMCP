from Agent.objects import *


class BeliefFunction:

    def __init__(self, belief: Belief) -> None:
        self.belief = belief

    def update_belief(self, action, observation) -> None:
        self.belief.update_belief(action, observation)

    def get_current_belief(self):
        return self.belief.get_current_belief()

    def plot_belief(self):
        return self.belief.plot_belief()

    def reset_belief(self):
        self.belief.current_belief = self.belief.initial_belief


class OptimalityCriterion:

    def __init__(self, discount_factor=0.95) -> None:
        self.gamma = discount_factor
        self._total_reward = 0.0

    def get_current_reward(self) -> float:
        return self._total_reward

    def update_cumulative_reward(self, reward: float, t: float) -> None:
        self._total_reward += self.gamma ** t * reward


