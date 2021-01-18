from abc import ABC
"""
Since States, Actions and Observations equality may be defined differently we enable
override of the hash/equal methods
"""


class State:

    def __init__(self, name, is_terminal=False):
        self.name = name
        self.is_terminal = is_terminal

    def __hash__(self):
        raise NotImplementedError

    def __eq__(self, other):
        raise NotImplementedError


class Action:

    def __init__(self, name):
        self.name = name

    def __hash__(self):
        raise NotImplementedError

    def __eq__(self, other):
        raise NotImplementedError


class Observation:

    def __init__(self, name):
        self.name = name

    def __hash__(self):
        raise NotImplementedError

    def __eq__(self, other):
        raise NotImplementedError


class InteractiveState:

    def __init__(self, state, model) -> None:
        self.state = state
        self.model = model

    def __str__(self) -> str:
        return str(self.state) + " " + str(self.model)

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other) -> bool:
        return self.state == other.state and self.model == other.model

    def __hash__(self) -> int:
        return hash(str(self.state) + str(self.model))

