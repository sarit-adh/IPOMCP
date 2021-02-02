
import logging
import os

from scipy.stats import uniform

from Problems.labor_market.labor_market_data import *
from Problems.labor_market.labor_market_emvironment import Environment
from Problems.labor_market.tom_zero_models.agents.tom_zero_manager_agent import *
from Problems.labor_market.tom_zero_models.agents.tom_zero_worker_agent import *
from Problems.labor_market.tom_zero_models.environments.tom_zero_manager_labor_market_environment import *
from Problems.labor_market.tom_zero_models.environments.tom_zero_worker_labor_market_environment import *

logging.basicConfig(filename='tom_zero_market.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

class TomZeroEnvironment(Environment):

    def __init__(self, number_of_trails, input_file) -> None:
        super().__init__(number_of_trails, input_file)
        with open(self.input_file) as json_file:
            data = json.load(json_file)
            self.manager_planning_horizon = data['manager'][0]['planning_horizon']
            self.worker_planning_horizon = data['worker'][0]['planning_horizon']
            self.n_samples = data['market'][0]['n_samples']
            self.n_actions = data['market'][0]['n_actions']
            self.oc = OptimalityCriterion(0.95)
            self.labor_cost_location = data['worker'][0]['a']
            self.budget_location = data['manager'][0]['loc']
            self.budget_scale = data['manager'][0]['scale']
            self.average_budget = data['market'][0]['avg_distance']
            self.min_fee = data['market'][0]['min_fee']
            self.max_fee = data['market'][0]['max_fee']
        manager_parent_distribution = norm(loc=self.budget_location, scale=self.budget_scale)
        worker_parent_distribution = gamma(a=self.labor_cost_location)
        self.manager_states = manager_parent_distribution.rvs(
            self.n_samples)
        self.worker_states = worker_parent_distribution.rvs(self.n_samples)
        self.manager_actions = self._create_ask_vector(worker_parent_distribution)
        self.worker_actions = self._create_bid_vector(manager_parent_distribution)

    def create_market(self) -> dict:
        fee = uniform(self.min_fee, self.max_fee).rvs(1).item()
        distance = gamma(a=self.average_budget).rvs(1).item()
        budget = np.random.choice(self.manager_states, 1).item()
        labor_cost = np.random.choice(self.worker_states, 1).item()
        managers_model = self.create_manager_problem(self.worker_states, self.manager_actions, distance, budget, fee)
        workers_model = self.create_worker_problem(self.manager_states, self.worker_actions, labor_cost, distance, fee)
        return {'manager': managers_model, 'worker': workers_model}

    def _create_ask_vector(self, parent_distribution):
        actions = parent_distribution.ppf(np.arange(1 / self.n_actions, 1 - 1 / self.n_actions, 1 / self.n_actions))
        return actions

    def _create_bid_vector(self, parent_distribution):
        actions = parent_distribution.ppf(np.arange(1 / self.n_actions, 1 - 1 / self.n_actions, 1 / self.n_actions))
        return actions

    def create_manager_problem(self, states, actions, distance, budget, fee):
        manager_beliefs = ToMZeroManagerLaborMarketBelief(self.labor_cost_location, states, distance)
        worker_model = TomZeroWorkerWorkerModel(np.random.choice(states), distance)
        manager_model = ToMZeroManagerLaborMarketEnvironment(states, actions * distance,
                                                             budget, fee, distance, worker_model)
        manager_frame = Frame(manager_model, self.oc)
        manager_type = ToMZeroManagerLaborMarketType(manager_frame, manager_beliefs)
        manager_agent = ToMZeroLaborMarketAgent(self.manager_planning_horizon, manager_type, None)
        return ToMZeroManager(manager_agent, manager_type, self.manager_planning_horizon)

    def create_worker_problem(self, states, actions, labor_cost, distance, fee):
        worker_beliefs = ToMZeroWorkerLaborMarketBelief(self.budget_location, self.budget_scale, states)
        manager_model = TomZeroWorkerManagerModel(np.random.choice(states))
        worker_model = ToMZeroWorkerLaborMarketEnvironment(states, actions,
                                                           labor_cost,
                                                           fee, distance, manager_model)
        worker_frame = Frame(worker_model, self.oc)
        worker_type = ToMZeroWorkerLaborMarketType(worker_frame, worker_beliefs)
        worker_agent = ToMZeroLaborMarketAgent(self.worker_planning_horizon, worker_type, None)
        return ToMZeroWorker(worker_agent, worker_type, self.worker_planning_horizon)

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
            lm_object.is_closed = True
        elif ask.value - labor_costs >= bid_qv:
            salary = ask.value
            lm_object.salary = salary
            lm_object.manager_reward += budget - salary
            lm_object.worker_reward += salary - labor_costs
            lm_object.is_cleared = True
            lm_object.is_closed = True
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
                lm_results = self.manager_first_negotiation(manager_ipomdp.manager_type.frame.pomdp.budget, ask,
                                                            worker_ipomdp.worker_type.frame.pomdp.labor_costs,
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
    m = TomZeroEnvironment(15, os.path.join(__location__, 'market_input.json'))
    while m.number_of_trails >= 0:
        agents_list = m.create_market()
        market_results = m.simulate_environment(agents_list, 'manager')
        js = market_results.summarize_market()
        print(json.dumps(js, indent=4, sort_keys=True))
        print(m.number_of_trails)
        del agents_list
        m.number_of_trails -= 1
