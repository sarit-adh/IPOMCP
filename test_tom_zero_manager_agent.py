from Problems.labor_market.tom_zero_models.environments.tom_zero_manager_labor_market_environment import *
from Problems.labor_market.tom_zero_models.agents.tom_zero_manager_agent import *
from IPOMCP_solver.pomcp import POMCP
from scipy.stats import norm, gamma

n = 30
states = gamma(5).rvs(n)
distance = 7
actions = states * distance
budget = norm(45, 2).rvs(1)
fee = 1.5
oc = OptimalityCriterion(0.95)
manager_beliefs = ToMZeroManagerLaborMarketBelief(5, states, distance)
worker_model = TomZeroWorkerWorkerModel(np.random.choice(states), distance)
manager_model = ToMZeroManagerLaborMarketEnvironment(states, actions, budget, fee, distance, worker_model)
manager_frame = Frame(manager_model, oc)
manager_type = ToMZeroManagerLaborMarketType(manager_frame, manager_beliefs)
manager = ToMZeroManagerLaborMarketAgent(5, manager_type, None)


def test_manager_planner():
    tom_zero_manager_pomcp = POMCP(manager_type, horizon=5)
    manager.planner = tom_zero_manager_pomcp
    while manager.planning_horizon >= 0:
        obs, reward = manager.execute_action
        print(f'Action {manager.actions[len(manager.actions)-1].name}, observation: {obs.name} and reward {reward}')
        if obs.value:
            break
    print(f'Total reward accumulated: {manager.agent_type.frame.oc.get_current_reward()}')


if __name__ == '__main__':
    test_manager_planner()
