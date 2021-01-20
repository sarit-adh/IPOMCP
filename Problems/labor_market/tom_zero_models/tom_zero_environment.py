from scipy.stats import uniform
from Problems.labor_market.labor_market_emvironment import Environment
from Problems.labor_market.tom_zero_models.environments.tom_zero_manager_labor_market_environment import *
from Problems.labor_market.tom_zero_models.agents.tom_zero_manager_agent import *
from Problems.labor_market.tom_zero_models.environments.tom_zero_worker_labor_market_environment import *
from Problems.labor_market.tom_zero_models.agents.tom_zero_worker_agent import *
from Problems.labor_market.labor_market_objects import *
import numpy as np
import json


class LaborMarketResults:

    def __init__(self, budget, labor_costs, fee, distance):
        self.fee = fee
        self.distance = distance
        self.budget = budget
        self.labor_costs = labor_costs
        self.bids = []
        self.asks = []
        self.manager_reward = 0.0
        self.worker_reward = 0.0
        self.salary = 0.0
        self.n_trials = 0.0
        self.last_offer = None
        self.first_offer = None
        self.is_cleared = False
        self.is_closed = False
        self.is_feasible = self._is_feasible_deal()

    def _is_feasible_deal(self):
        return self.budget - self.labor_costs > 0

    def visualize_market(self):
        pass

    def summarize_market(self):
        print(f'Market is cleared: {self.is_cleared}')
        print(f'The market is feasible: {self.is_feasible}')
        print(f'The market is over after: {self.n_trials} iterations')
        print(f'Salary agreed: {self.salary}')
        print(f'Managers cumulative reward is: {self.manager_reward}')
        print(f'Workers cumulative reward is: {self.worker_reward}')
        print(f'Managers budget is: {self.budget}')
        print(f'Workers labor costs are: {self.labor_costs}')
        print(f'Managers asks are: {self.asks}')
        print(f'Workers bids are: {self.bids}')
        print("--------- NEW MARKET ------------")


class TomZeroEnvironment(Environment):

    def __init__(self, number_of_trails, input_file) -> None:
        super().__init__(number_of_trails, input_file)
        with open(self.input_file) as json_file:
            data = json.load(json_file)
            self.manager_planning_horizon = data['manager'][0]['planning_horizon']
            self.worker_planning_horizon = data['worker'][0]['planning_horizon']
            self.n_samples = data['market'][0]['n_samples']
            self.oc = OptimalityCriterion(0.95)
            self.labor_cost_location = data['worker'][0]['a']
            self.budget_location = data['manager'][0]['loc']
            self.budget_scale = data['manager'][0]['scale']
            self.average_budget = data['market'][0]['avg_distance']
            self.min_fee = data['market'][0]['min_fee']
            self.max_fee = data['market'][0]['max_fee']

    def create_market(self) -> dict:
        fee = uniform(self.min_fee, self.max_fee).rvs(1).item()
        distance = gamma(a=self.average_budget).rvs(1).item()
        manager_states = norm(loc=self.budget_location, scale=self.budget_scale).rvs(
            self.n_samples)
        worker_states = gamma(a=self.labor_cost_location).rvs(self.n_samples)
        budget = np.random.choice(manager_states, 1).item()
        labor_cost = np.random.choice(worker_states, 1).item()
        managers_model = self.create_manager_problem(worker_states, distance, budget, fee)
        workers_model = self.create_worker_problem(manager_states, labor_cost, distance, fee)
        return {'manager': managers_model, 'worker': workers_model}

    def create_manager_problem(self, states, distance, budget, fee):
        manager_beliefs = ToMZeroManagerLaborMarketBelief(self.labor_cost_location, states, distance)
        worker_model = TomZeroWorkerWorkerModel(np.random.choice(states), distance)
        manager_model = ToMZeroManagerLaborMarketEnvironment(states, states * distance,
                                                             budget, fee, distance, worker_model)
        manager_frame = Frame(manager_model, self.oc)
        manager_type = ToMZeroManagerLaborMarketType(manager_frame, manager_beliefs)
        manager_agent = ToMZeroLaborMarketAgent(self.manager_planning_horizon, manager_type, None)
        return ToMZeroManager(manager_agent, manager_type)

    def create_worker_problem(self, states, labor_cost, distance, fee):
        worker_beliefs = ToMZeroWorkerLaborMarketBelief(self.budget_location, self.budget_scale, states)
        manager_model = TomZeroWorkerManagerModel(np.random.choice(states))
        worker_model = ToMZeroWorkerLaborMarketEnvironment(states, states,
                                                           labor_cost,
                                                           fee, distance, manager_model)
        worker_frame = Frame(worker_model, self.oc)
        worker_type = ToMZeroWorkerLaborMarketType(worker_frame, worker_beliefs)
        worker_agent = ToMZeroLaborMarketAgent(self.worker_planning_horizon, worker_type, None)
        return ToMZeroWorker(worker_agent, worker_type)

    @staticmethod
    def manager_first_negotiation(budget, ask,
                                  labor_costs, bid_qv,
                                  lm_object: LaborMarketResults, starting_agent_name='manager'):
        if ask.name == 'quit':
            salary = 0.0
            manager_reward = 0.0
            worker_reward = 0.0
            lm_object.last_offer = starting_agent_name
            lm_object.salary = salary
            lm_object.manager_reward += manager_reward
            lm_object.worker_reward += worker_reward
            lm_object.is_cleared = False
        elif ask.value - labor_costs >= bid_qv:
            salary = ask.value
            lm_object.salary = salary
            lm_object.manager_reward += budget - salary
            lm_object.worker_reward += salary - labor_costs
            lm_object.is_cleared = True
        else:
            lm_object.manager_reward += -lm_object.fee
        return lm_object

    @staticmethod
    def worker_first_negotiation(budget, ask_qv,
                                 labor_costs, bid,
                                 lm_object: LaborMarketResults, starting_agent_name='worker'):
        if bid.name == 'quit':
            salary = 0.0
            manager_reward = 0.0
            worker_reward = 0.0
            lm_object.last_offer = starting_agent_name
            lm_object.salary = salary
            lm_object.manager_reward += manager_reward
            lm_object.worker_reward += worker_reward
            lm_object.is_cleared = False
            lm_object.is_closed = True
        elif budget - bid.value >= ask_qv:
            salary = bid.value
            lm_object.last_offer = starting_agent_name
            lm_object.salary = salary
            lm_object.manager_reward += budget - salary
            lm_object.worker_reward += salary - labor_costs
            lm_object.is_cleared = True
            lm_object.is_closed = True
        else:
            lm_object.worker_reward += -lm_object.fee
        return lm_object

    def simulate_environment(self, agent_types_list: dict, starting_agent):
        manager_ipomdp = agent_types_list['manager']
        worker_ipomdp = agent_types_list['worker']
        lm_results = LaborMarketResults(manager_ipomdp.manager_type.frame.pomdp.budget,
                                        worker_ipomdp.worker_type.frame.pomdp.labor_costs,
                                        manager_ipomdp.manager_type.frame.pomdp.fee,
                                        manager_ipomdp.manager_type.frame.pomdp.distance)
        lm_results.manager_reward = 0.0
        lm_results.worker_reward = 0.0
        i = 0
        while not lm_results.is_closed:
            ask, ask_qv = manager_ipomdp.best_response()
            bid, bid_qv = worker_ipomdp.best_response()
            if starting_agent == 'manager':
                lm_results.first_offer = 'manager'
                lm_results.asks.append(ask.value)
                lm_results.n_trials += 1
                lm_results = self.manager_first_negotiation(manager_ipomdp.manager_type.frame.pomdp.budget,
                                                            ask, worker_ipomdp.worker_type.frame.pomdp.labor_costs,
                                                            bid_qv, lm_results)
                if lm_results.is_closed:
                    break
                lm_results.bids.append(bid.value)
                lm_results.n_trials += 1
                lm_results = self.worker_first_negotiation(manager_ipomdp.manager_type.frame.pomdp.budget,
                                                           ask_qv, worker_ipomdp.worker_type.frame.pomdp.labor_costs,
                                                           bid, lm_results)
                if lm_results.is_closed:
                    break
            else:
                lm_results.first_offer = 'worker'
                lm_results.bids.append(bid.value)
                lm_results.n_trials += 1
                lm_results = self.worker_first_negotiation(manager_ipomdp.manager_type.frame.pomdp.budget,
                                                           ask_qv, worker_ipomdp.worker_type.frame.pomdp.labor_costs,
                                                           bid, lm_results)
                if lm_results.is_closed:
                    break
                lm_results.asks.append(ask.value)
                lm_results.n_trials += 1
                lm_results = self.manager_first_negotiation(manager_ipomdp.manager_type.frame.pomdp.budget,
                                                            ask, worker_ipomdp.worker_type.frame.pomdp.labor_costs,
                                                            bid_qv, lm_results)
                if lm_results.is_closed:
                    break
            reject_observation = Observation(False, 'reject')
            manager_ipomdp.manager_agent.observations.append(reject_observation)
            worker_ipomdp.worker_agent.observations.append(reject_observation)
            i += 1
        return lm_results


if __name__ == '__main__':
    m = TomZeroEnvironment(20, 'market_input.json')
    while m.number_of_trails >= 0:
        agents_list = m.create_market()
        market_results = m.simulate_environment(agents_list, 'manager')
        market_results.summarize_market()
        del agents_list
        m.number_of_trails -= 1
