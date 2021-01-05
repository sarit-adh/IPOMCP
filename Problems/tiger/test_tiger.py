from tiger_environment import *
from tiger_agent import *
from IPOMCP_solver.pomcp import POMCP
from IPOMCP_solver.node import *

states = [State("tiger-left"), State("tiger-right")]
actions = [Action("open-left"), Action("open-right"), Action('listen')]
observations = []
tiger_problem = TigerEnvironment(states, actions, observations)
print(tiger_problem)
oc = OptimalityCriterion(0.95)
beliefs = TigerBelief(np.array([0.5, 0.5]))
agent_beliefs = BeliefFunction(beliefs)
tiger_frame = Frame(tiger_problem, oc)
tiger_type = TigerType(tiger_frame, agent_beliefs)
agent = TigerAgent(20, tiger_type, None)


def test_agent():
    while agent.planning_horizon > 0:
        o, r = agent.execute_action
        print(o.name, r)
        print(agent.agent_type.beliefs.get_current_belief())
        print(agent.planning_horizon)
    print(agent.agent_type.frame.oc.get_current_reward())


def test_pomcp():
    root_node = ObservationNode(None, '', '')
    tiger_pomcp = POMCP(tiger_type, horizon=3)
    br_node, br_value = tiger_pomcp.search(root_node)
    obs, reward = tiger_problem.pomdp_step(br_node)


def test_agent_planner():
    tiger_pomcp = POMCP(tiger_type, horizon=3)
    agent.planner = tiger_pomcp
    obs, reward = agent.execute_action
    print(obs.name, reward)


if __name__ == '__main__':
    test_agent_planner()
