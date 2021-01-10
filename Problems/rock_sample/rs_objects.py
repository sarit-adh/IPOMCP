from Environment.objects import *
import numpy as np


def euclidean_dist(x: np.array,y: np.array) -> float:
    dist = np.linalg.norm(x - y)
    return dist


class Rock:
    GOOD = 'good'
    BAD = 'bad'

    def __init__(self, x: int, y: int, initial_type: str) -> None:
        self.x = x
        self.y = y
        self.type = initial_type

    def invert(self) -> str:
        if self.type == 'good':
            return 'bad'
        else:
            return 'good'

    def measure(self, signal_quality) -> str:
        p = np.random.rand(1)
        if p <= signal_quality:
            return self.type
        return self.invert()


class State(State):

    def __init__(self, position, rock_types, terminal=False):
        """
        :param position: (x,y) coordinates representing the location of the rover
        :param rock_types: list of size k, indicating the quality of each rock
        :param terminal: bool, is the state terminal
        """
        self.position = position
        self.rock_types = rock_types
        self.is_terminal = terminal

    def __hash__(self):
        return hash((self.position, self.rock_types, self.is_terminal))

    def __eq__(self, other):
        if isinstance(other, State):
            return self.position == other.position \
                   and self.rock_types == other.rock_types \
                   and self.is_terminal == other.is_terminal
        return False

    def __str__(self)-> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return f'Rover location = {self.position} with {len(self.rock_types)} rocks and terminal = {self.is_terminal}'


class Action(Action):

    def __init__(self, name):
        self.name = name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, Action):
            return self.name == other.name
        elif type(other) == str:
            return self.name == other

    def __str__(self):
        return self.name

    def __repr__(self):
        return f'Action {self.name}'


class MoveAction(Action):
    EAST = (1, 0)  # x is horizontal; x+ is right. y is vertical; y+ is up.
    WEST = (-1, 0)
    NORTH = (0, -1)
    SOUTH = (0, 1)

    def __init__(self, motion, name):
        if motion not in (MoveAction.EAST, MoveAction.WEST,
                          MoveAction.NORTH, MoveAction.SOUTH):
            raise ValueError(f'Invalid action {motion}')
        self.motion = motion
        super().__init__("move-%s" % str(name))


MoveEast = MoveAction(MoveAction.EAST, "EAST")
MoveWest = MoveAction(MoveAction.WEST, "WEST")
MoveNorth = MoveAction(MoveAction.NORTH, "NORTH")
MoveSouth = MoveAction(MoveAction.SOUTH, "SOUTH")


class SampleAction(Action):
    def __init__(self):
        super().__init__("sample")


class CheckAction(Action):
    def __init__(self, rock_id):
        self.rock_id = rock_id
        super().__init__("check-%d" % self.rock_id)


ALL_ACTIONS = [MoveEast, MoveWest, MoveNorth, MoveSouth, SampleAction, CheckAction]


class Observation(Observation):

    def __init__(self, quality):
        self.quality = quality

    def __hash__(self):
        return hash(self.quality)

    def __eq__(self, other):
        if isinstance(other, Observation):
            return self.quality == other.quality
        elif type(other) == str:
            return self.quality == other

    def __str__(self):
        return str(self.quality)

    def __repr__(self):
        return "Observation(%s)" % str(self.quality)