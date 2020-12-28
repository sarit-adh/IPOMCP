import itertools
from temp_storage import *

class InteractiveState:

    def __init__(self, state, model):
        self.state = state
        self.model = model

    def __str__(self):
        return str(self.state) + " " + str(self.model)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.state == other.state and self.model == other.model

    def __hash__(self):
        return hash(str(self.state) + str(self.model))
        # return hash(self.state) + hash(self.model)
        # return hash(str(self.state)) + hash(self.model)


class Model:

    def __init__(self, belief, history, frame, solver=None, best_action=None):
        self.belief = belief
        self.frame = frame
        self.history = history
        self.solver = solver
        self.best_action = best_action

    def __str__(self):
        return str(self.history) + str(self.frame)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.history == other.history and self.frame == other.frame #should it be history

    def __hash__(self):
        return hash(str(self))



class Frame:

    def __init__(self, id, actions, observations, transition_table, observation_table, reward_table,
                 optimality_criterion, parent_id=None, level=0):
        self.id = id
        self.actions = actions
        self.observations = observations
        self.transition_table = transition_table
        self.observation_table = observation_table
        self.reward_table = reward_table
        self.parent_id = parent_id
        self.optimality_criterion = optimality_criterion  # defined as tuple ("finite / infinite" , discount_factor, horizon)
        self.level = level

    def get_observation_likelihood(self, state, observation, action, action_other):
        return self.observation_table[state][action][action_other][observation]

    def get_transition_likelihood(self, state, action, action_other, next_state):
        return self.transition_table[state][action][action_other][next_state]

    def get_reward(self, state, action, action_other):
        return self.reward_table[state][action][action_other]

    def __str__(self):
        return str(self.id)

    def __repr__(self):
        return self.__str__()
