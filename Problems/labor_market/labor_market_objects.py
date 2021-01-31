from Environment.objects import *


class State(State):

    def __init__(self, value: float, name: str, is_terminal=False):
        """

        :param value: float, representing a feasible state value
        """
        self.value = value
        super().__init__(name, is_terminal)

    def __hash__(self):
        return hash((self.value, self.name, self.is_terminal))

    def __eq__(self, other):
        if isinstance(other, State):
            return self.value == other.value and \
                   self.name == other.name and \
                   self.is_terminal == other.is_terminal
        return False

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return f'State with value = {self.value}, name = {self.name} and is terminal {self.is_terminal}'


class Action(Action):

    def __init__(self, name, value=0.0) -> None:
        self.name = name
        self.value = value

    def __hash__(self):
        return hash((self.name, self.value))

    def __eq__(self, other):
        if isinstance(other, Action):
            return self.name == other.name and self.value == other.value
        elif type(other) == str:
            return self.name == other

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f'Action {self.name} with value {self.value}'


class OfferAction(Action):

    def __init__(self, value):
        super().__init__(str(value), value)


class AcceptAction(Action):
    def __init__(self):
        super().__init__("accept", -1)


class QuitAction(Action):
    def __init__(self):
        super().__init__('quit', 0)


class Observation(Observation):

    def __init__(self, value, name='offer'):
        self.value = value
        self.name = name

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        if isinstance(other, Observation):
            return self.value == other.value

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f'{self.name} offer with value {self.value}'
