

class OptimalityCriterion:

    def __init__(self, discount_factor=0.95) -> None:
        self.gamma = discount_factor
        self._total_reward = 0.0

    def get_current_reward(self) -> float:
        return self._total_reward

    def update_cumulative_reward(self, reward: float, t: float) -> None:
        self._total_reward += self.gamma ** t * reward

    # def optimality criteria

