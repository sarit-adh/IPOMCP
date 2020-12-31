from Environment.environment import IPOMDPEnvironment
from Problems.tiger.objects import *
import random


class TigerEnvironment(IPOMDPEnvironment):

    def __init__(self, states, actions, observations):
        super(TigerEnvironment, self).__init__(states, actions, observations)

    def _set_initial_state(self, state=None):
        if state is None:
            initial_state = random.choice(self.states)
            return initial_state
        return state

    def transition_function(self, state, actions, **kwargs):
        if actions.name.startswith("open"):
            return State(random.choice(self.states).name)
        if actions.name.startswith("listen"):
            return State(state.name)

    def observation_function(self, state, actions, next_state, noise=0.15):
        if actions.name.startswith("open"):
            return Observation(random.choice(self.states).name)
        else:
            obs = next_state.name if random.random() <= (1 - noise) else next_state.other().name
            return Observation(obs)

    def reward_function(self, state, actions, **kwargs):
        if actions.name.startswith('open'):
            if state.name.split("-")[1] == actions.name.split("-")[1]:
                return -100
            else:
                return 10
        return -1

    def step(self, state, actions) -> object:
        next_state = self.transition_function(state, actions)
        observation = self.observation_function(state, actions, next_state, 0.15)
        reward = self.reward_function(state, actions)
        return next_state, observation, reward


def unittest():
    states = [State("tiger-left"), State("tiger-right")]
    observations = []
    actions = [Action("open-left"), Action("open-right"), Action("listen")]
    tiger_problem = TigerEnvironment(states, actions, observations)
    s, o, r = tiger_problem.step(State("tiger-left"), Action("listen"))
    print(f'action {Action("listen")} yields observation {o} ,reward {r} and new state {s}')

    s, o, r = tiger_problem.step(State("tiger-left"), Action("open-left"))
    print(f'action {Action("open-left")} yields observation {o} ,reward {r} and new state {s}')

    s, o, r = tiger_problem.step(State("tiger-left"), Action("open-right"))
    print(f'action {Action("open-right")} yields observation {o} ,reward {r} and new state {s}')

    
if __name__ == '__main__':
    unittest()
