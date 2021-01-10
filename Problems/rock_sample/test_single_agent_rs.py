from rs_environment import *
from rs_agent import *
from IPOMCP_solver.pomcp import POMCP


rock_sample_problem = RockSampleEnvironment(8, 5)
oc = OptimalityCriterion(0.95)
agent_beliefs = RockSampleBelief(None)
rs_frame = Frame(rock_sample_problem, oc)
rs_type = RockSampleType(rs_frame, agent_beliefs)
agent = RockSampleAgent(5, rs_type, None)


def test_agent_planner():
    rs_pomcp = POMCP(rs_type, horizon=5)
    agent.planner = rs_pomcp
    while agent.planning_horizon >= 0:
        obs, reward = agent.execute_action
        print(f'Action {agent.actions[len(agent.actions)-1].name}, observation: {obs.name} and reward {reward}')
    print(f'Total reward accumulated: {agent.agent_type.frame.oc.get_current_reward()}')


if __name__ == '__main__':
    test_agent_planner()
