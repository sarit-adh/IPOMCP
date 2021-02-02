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
        results = {'is_market_closed': self.is_cleared, 'is_market_feasible': self.is_feasible,
                   'n_iters': self.n_trials, 'fee': self.fee, 'salary': self.salary, 'budget': self.budget,
                   'labor_costs': self.labor_costs, 'managers_cumulative_reward': self.manager_reward,
                   'workers_cumulative_reward': self.worker_reward, 'workers_bids': self.bids,
                   'managers_bids': self.asks}
        return json.dumps(results)
