from Agent.agent import *
import numpy as np
import matplotlib.pyplot as plt


class TigerBelief(Belief):
    def update_belief(self, action, observation) -> None:
        if action.name.startswith('open'):
            new_belief = np.array([0.5, 0.5])
        else:
            if observation.name.endswith('left'):
                new_belief = self.current_belief * np.array([0.85, 0.15])
            else:
                new_belief = self.current_belief * np.array([0.15, 0.85])
            new_belief = new_belief / new_belief.sum()
        self.current_belief = new_belief

    def get_current_belief(self):
        return self.current_belief

    def mpe(self) -> State:
        return [State('tiger-left'), State('tiger-right')][np.argmax(self.current_belief).item()]

    def plot_belief(self):
        plt.hist(self.current_belief)


class TigerType(Type):

    def update_belief(self, action: Action, observation: Observation, **kwargs) -> None:
        self.beliefs.update_belief(action, observation)


class TigerAgent(Agent):

    def __init__(self, planning_horizon: int, agent_type: Type, planner) -> None:
        self.planning_horizon = planning_horizon
        super().__init__(agent_type, planner)

    @property
    def compute_optimal_policy(self) -> Action:
        p = self.agent_type.beliefs.get_current_belief()
        if self.planning_horizon <= 0:
            return np.random.choice(np.array([Action('open-left'), Action('open-right')]), p)
        else:
            if (p * [-100, 10]).sum() > -1 or (p * [10, -100]).sum() > -1:
                return np.array([Action('open-left'), Action('open-right')])[np.argmax(p)]
            return Action('listen')

    def execute_action(self) -> (Observation, float):
        action = self.compute_optimal_policy
        self.planning_horizon -= 1
        _, observation, reward = self.agent_type.frame.pomdp.step(self.agent_type.frame.pomdp.current_state, action)
        self.update_history(action, observation)
        self.update_reward(reward, self.planning_horizon)
        return observation, reward

    def update_history(self, action: Action, observation: Observation) -> None:
        self.agent_type.beliefs.update_belief(action, observation)
