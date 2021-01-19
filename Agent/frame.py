from Environment.problem import *
from Agent.functions import *


class Frame(ABC):
    """
        This class implements the frame (definition 2) of an agent - i.e., pompd + optimality criteria
    """

    def __init__(self, pomdp: IpomdpProbelm, optimality_criteria: OptimalityCriterion) -> None:
        self.pomdp = pomdp
        self.oc = optimality_criteria

