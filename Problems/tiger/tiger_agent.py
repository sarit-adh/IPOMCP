from Agent.agent import *
import numpy as np
import matplotlib.pyplot as plt


class TigerBelief(Belief):
    def update_belief(self, action, observation) -> None:
        if action.name.startswith('open'):
            new_belief = np.array([0.5, 0.5])
        else:
            if observation.name.endswith('left'):
                p_l = self.current_belief[0] * np.array([0.85])
                p_r = self.current_belief[1] * np.array([0.15])
                new_belief = np.array([p_l / (p_l+p_r), p_r / (p_l + p_r)])
            else:
                p_l = self.current_belief[0] * np.array([0.15])
                p_r = self.current_belief[1] * np.array([0.85])
                new_belief = np.array([p_l / (p_l + p_r), p_r / (p_l + p_r)])
        self.current_belief = new_belief

    def get_current_belief(self):
        return self.current_belief

    def mpe(self) -> State:
        return [State('tiger-left'), State('tiger-right')][np.argmax(self.current_belief).item()]

    def plot_belief(self):
        plt.hist(self.current_belief)


class TigerType(AgentType):

    def update_belief(self, action: Action, observation: Observation, **kwargs) -> None:
        self.beliefs.update_belief(action, observation)

    def rollout_policy(self):
        action = np.random.choice(self.frame.pomdp.actions)
        return action


class TigerAgent(Agent):

    def __init__(self, planning_horizon: int, agent_type: AgentType, planner) -> None:
        self.planning_horizon = planning_horizon
        super().__init__(agent_type, planner)

    @property
    def compute_optimal_policy(self) -> Action:
        if self.planning_horizon == 0:
            return np.random.choice(np.array([Action('open-left'), Action('open-right')]))
        else:
            p = self.agent_type.beliefs.get_current_belief()
            if np.dot(p.T, np.array([-100, 10])) > -1 or np.dot(p.T, np.array([10, -100])) > -1:
                return np.array([Action('open-left'), Action('open-right')])[np.argmin(p)]
            return Action('listen')

    def execute_action(self) -> (Observation, float):
        action = self.compute_optimal_policy
        self.planning_horizon -= 1
        new_state, observation, reward = self.agent_type.frame.pomdp.step(self.agent_type.frame.pomdp.current_state, action)
        self.agent_type.frame.pomdp.update_current_state(new_state)
        self.update_history(action, observation)
        self.update_reward(reward, self.planning_horizon)
        return observation, reward

    def update_history(self, action: Action, observation: Observation) -> None:
        self.agent_type.beliefs.update_belief(action, observation)
