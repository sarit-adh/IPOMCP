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
    print(tiger_problem)
    oc = OptimalityCriterion(0.95)
    beliefs = TigerBelief(np.array([0.5, 0.5]))
    agent_beliefs = BeliefFunction(beliefs)
    tiger_frame = Frame(tiger_problem, oc)
    tiger_type = TigerType(tiger_frame, agent_beliefs)
    agent = TigerAgent(20, tiger_type, None)
    while agent.planning_horizon > 0:
        o, r = agent.execute_action()
        print(o.name, r)
        print(agent.agent_type.beliefs.get_current_belief())
        print(agent.planning_horizon)
    print(agent.agent_type.frame.oc.get_current_reward())


if __name__ == '__main__':
    test_agent()
