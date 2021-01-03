from Agent.objects import *
from Agent.functions import *
from Agent.frame import *
from Agent.type import *
from tiger_environment import *
from tiger_objects import *
from tiger_agent import *


def test_env():
    states = [State("tiger-left"), State("tiger-right")]
    observations = []
    tiger_problem = TigerEnvironment(states, observations)
    s, o, r = tiger_problem.step(State("tiger-left"), Action("listen"))
    print(f'action {Action("listen")} yields observation {o} ,reward {r} and new state {s}')

    s, o, r = tiger_problem.step(State("tiger-left"), Action("open-left"))
    print(f'action {Action("open-left")} yields observation {o} ,reward {r} and new state {s}')

    s, o, r = tiger_problem.step(State("tiger-left"), Action("open-right"))
    print(f'action {Action("open-right")} yields observation {o} ,reward {r} and new state {s}')


def test_agent():
    states = [State("tiger-left"), State("tiger-right")]
    observations = []
    tiger_problem = TigerEnvironment(states, observations)
    oc = OptimalityCriterion(0.95)
    beliefs = TigerBelief(np.array([0.5, 0.5]))
    agent_beliefs = BeliefFunction(beliefs)
    tiger_frame = Frame(tiger_problem, oc)
    tiger_type = TigerType(tiger_frame, agent_beliefs)
    agent = TigerAgent(5, tiger_type, None)
    o, r = agent.execute_action()
    print(o.name)
    o, r = agent.execute_action()
    print(o.name)
    o, r = agent.execute_action()
    print(o.name)


if __name__ == '__main__':
    test_agent()
