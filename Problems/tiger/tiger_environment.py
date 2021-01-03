from Environment.environment import IPOMDPEnvironment
from Problems.tiger.tiger_objects import *
import random


class TigerEnvironment(IPOMDPEnvironment):

    def __init__(self, states, observations, noise=0.15):
        super().__init__(states, observations)
        self.noise = noise

    def __str__(self):
        return f'Tiger problem with noise {self.noise} and true tiger location in {self.initial_state.name}'

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

    def observation_function(self, state, actions, next_state, **kwargs):
        if actions.name.startswith("open"):
            return Observation(random.choice(self.states).name)
        else:
            obs = next_state.name if random.random() <= (1-self.noise) else next_state.other().name
            return Observation(obs)

    def reward_function(self, state, actions, **kwargs):
        if actions.name.startswith('open'):
            if state.name.split("-")[1] == actions.name.split("-")[1]:
                return -100
            else:
                return 10
        return -1

    def step(self, state, actions, **kwargs) -> object:
        """

        :rtype: object
        """
        next_state = self.transition_function(state, actions)
        observation = self.observation_function(state, actions, next_state)
        reward = self.reward_function(state, actions)
        return next_state, observation, reward
