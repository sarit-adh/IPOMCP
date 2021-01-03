from Environment.objects import *


class State(State):

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, State):
            return self.name == other.name
        return False

    def other(self):
        if self.name.endswith("left"):
            return State("tiger-right")
        else:
            return State("tiger-left")

    def __str__(self):
        return f'{self.name}'


class Observation(Observation):

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, Observation):
            return self.name == other.name
        return False

    def __str__(self):
        return f'{self.name}'


class Action(Action):

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, Action):
            return self.name == other.name
        return False

    def __str__(self):
        return f'{self.name}'